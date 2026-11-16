"""Train a signed pathway auxiliary ranker on top of the best frozen flow model."""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
HYPO_CODE = Path(__import__('hypo_release.paths', fromlist=['resolve']).resolve('source', ''))
if str(HYPO_CODE) not in sys.path:
    sys.path.insert(0, str(HYPO_CODE))
from experiments.perturbation.evaluate_prototype_mass_flow import load_evaluation_module
PUBLIC_TEST = {'H31', 'H132', 'H228', 'H237'}

def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow-checkpoint', required=True)
    parser.add_argument('--pair-records', required=True)
    parser.add_argument('--pathway-base', required=True)
    parser.add_argument('--panel-csv', required=True)
    parser.add_argument('--drug-pathway-sign', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--activity-strength', type=float, default=0.75)
    parser.add_argument('--n-ode-steps', type=int, default=20)
    parser.add_argument('--weak-purity', type=float, default=0.4)
    parser.add_argument('--weak-delta', type=float, default=0.05)
    parser.add_argument('--feature-mode', choices=('all', 'no_activity', 'relation_only', 'target_only'), default='all')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

def norm_drug(name: str) -> str:
    return ''.join((ch.lower() for ch in name if ch.isalnum()))

@dataclass
class Example:
    sample_id: str
    target_celltype: str
    feature: torch.Tensor
    label: torch.Tensor
    truth: dict[str, str]
    confidence: float
    curated: bool

class ResidualRanker(nn.Module):

    def __init__(self, feature_dim: int, action_count: int):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(feature_dim, 128), nn.SiLU(), nn.LayerNorm(128))
        self.block = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, 128))
        self.output = nn.Linear(128, action_count)
        self.baseline_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, feature):
        hidden = self.input(feature)
        hidden = F.silu(hidden + self.block(hidden))
        return self.output(hidden) + self.baseline_scale * feature[:, :self.output.out_features]

def truth_for_record(record, drug_truth):
    truth = {}
    for drug in map(str, record.get('drug_names', [])):
        mapped = drug_truth.get(norm_drug(drug))
        if mapped is None:
            raise ValueError(f"unmapped drug {drug} in {record['sample_id']}")
        (pathway, direction) = mapped
        if pathway in truth and truth[pathway] != direction:
            return None
        truth[pathway] = direction
    return truth

def make_feature(module, record, target_index, pathway_names, mapping, strength):
    device = mapping.device
    ids = torch.as_tensor(record['drug_ids']).long().view(1, -1).to(device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    base_encoding = module.condition_encoder(ids, mask)
    base_activity = base_encoding.activities['selected']
    source_mass = torch.as_tensor(record['initial_mass_logit']).float().view(1, -1).to(device)
    source_presence = torch.as_tensor(record['initial_presence']).float().view(1, -1).to(device)
    (base_mass, base_endpoint) = module.adapter.rollout_endpoint(source_mass, source_presence, base_encoding.condition_tokens, base_encoding.pathway_context)
    base_prop = module.adapter.velocity_model.presence_gated_proportion(base_mass, torch.sigmoid(base_endpoint.presence_logits)) @ mapping
    baseline = base_prop[0, target_index]
    gains = torch.empty(len(pathway_names), 2, device=device)
    for (direction_index, direction_value) in enumerate((1.0, -1.0)):
        activities = base_activity.repeat(len(pathway_names), 1)
        index = torch.arange(len(pathway_names), device=device)
        activities[index, index] += direction_value * strength
        encoding = module.condition_encoder.encode_activities({'selected': activities})
        (mass, endpoint) = module.adapter.rollout_endpoint(source_mass.repeat(len(pathway_names), 1), source_presence.repeat(len(pathway_names), 1), encoding.condition_tokens, encoding.pathway_context)
        prop = module.adapter.velocity_model.presence_gated_proportion(mass, torch.sigmoid(endpoint.presence_logits)) @ mapping
        gains[:, direction_index] = prop[:, target_index] - baseline
    signed_gain = gains.reshape(-1)
    signed_gain = (signed_gain - signed_gain.mean()) / signed_gain.std().clamp_min(1e-06)
    initial_prop = torch.as_tensor(record['initial_celltype_proportion']).float().to(device)
    target_onehot = F.one_hot(torch.tensor(target_index, device=device), num_classes=mapping.shape[1]).float()
    feature = torch.cat([signed_gain, base_activity[0], initial_prop, target_onehot])
    return feature.cpu()

def make_label(truth, pathway_names):
    label = torch.zeros(len(pathway_names), 2)
    for (pathway, direction) in truth.items():
        label[pathway_names.index(pathway), 0 if direction == '+' else 1] = 1
    return label.reshape(-1)

def rank_actions(logits, pathway_names, k):
    actions = logits.view(len(pathway_names), 2)
    (score, direction) = actions.max(dim=1)
    order = score.argsort(descending=True)[:k]
    return [(pathway_names[i], '+' if int(direction[i]) == 0 else '-') for i in order.tolist()]

def score_predictions(examples, logits, pathway_names):
    rows = []
    for (example, sample_logits) in zip(examples, logits):
        selected = rank_actions(sample_logits, pathway_names, len(example.truth))
        identity = sum((pathway in example.truth for (pathway, _) in selected))
        signed = sum((example.truth.get(pathway) == direction for (pathway, direction) in selected))
        rows.append({'sample_id': example.sample_id, 'target_celltype': example.target_celltype, 'denominator': len(example.truth), 'identity_hits': identity, 'signed_hits': signed, 'identity_score': f'{identity}/{len(example.truth)}', 'signed_score': f'{signed}/{len(example.truth)}', 'truth': '; '.join((f'{p}{d}' for (p, d) in sorted(example.truth.items()))), 'selected': '; '.join((f'{p}{d}' for (p, d) in selected))})
    return rows

def train_one(examples, validation, seed, device, epochs=500):
    random.seed(seed)
    torch.manual_seed(seed)
    feature = torch.stack([e.feature for e in examples]).to(device)
    label = torch.stack([e.label for e in examples]).to(device)
    weight = torch.tensor([e.confidence for e in examples], device=device)
    (mean, std) = (feature.mean(0), feature.std(0).clamp_min(1e-05))
    x = (feature - mean) / std
    model = ResidualRanker(x.shape[1], label.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    (best_state, best_value) = (None, float('inf'))
    val_x = None
    if validation:
        val_x = (torch.stack([e.feature for e in validation]).to(device) - mean) / std
        val_y = torch.stack([e.label for e in validation]).to(device)
    for _ in range(epochs):
        model.train()
        logits = model(x)
        bce = F.binary_cross_entropy_with_logits(logits, label, reduction='none').mean(1)
        target_dist = label / label.sum(1, keepdim=True).clamp_min(1)
        listwise = -(target_dist * F.log_softmax(logits, dim=1)).sum(1)
        action = logits.view(len(examples), -1, 2)
        labels = label.view(len(examples), -1, 2)
        positive_direction = labels.argmax(2)
        positive_mask = labels.sum(2) > 0
        chosen = action.gather(2, positive_direction.unsqueeze(-1)).squeeze(-1)
        opposite = action.gather(2, (1 - positive_direction).unsqueeze(-1)).squeeze(-1)
        margin = (F.relu(0.5 - chosen + opposite) * positive_mask).sum(1) / positive_mask.sum(1).clamp_min(1)
        loss = ((bce + 0.5 * listwise + 0.5 * margin) * weight).sum() / weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            if validation:
                val_logits = model(val_x)
                value = float(F.binary_cross_entropy_with_logits(val_logits, val_y))
            else:
                value = float(loss)
            if value < best_value:
                best_value = value
                best_state = {key: tensor.detach().cpu().clone() for (key, tensor) in model.state_dict().items()}
    model.load_state_dict(best_state)
    return (model, mean, std)

def main():
    args = arguments()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True)
    device = torch.device(args.device)
    with Path(args.panel_csv).open(newline='', encoding='utf-8') as handle:
        panel_rows = list(csv.DictReader(handle))
    panel_celltype = {row['sample_id']: row['expected_celltype'] for row in panel_rows}
    panel_ids = set(panel_celltype)
    with Path(args.drug_pathway_sign).open(newline='', encoding='utf-8') as handle:
        drug_truth = {norm_drug(row['drug']): (row['pathway'], '+' if int(row['sign']) > 0 else '-') for row in csv.DictReader(handle)}
    payload = torch.load(args.pair_records, map_location='cpu', weights_only=True)
    checkpoint = torch.load(args.flow_checkpoint, map_location='cpu', weights_only=True)
    records = {str(row['sample_id']): row for row in payload['records']}
    module = load_evaluation_module(dict(checkpoint['config']), checkpoint, payload, args.pathway_base, device)
    module.adapter.endpoint_rollout_steps = args.n_ode_steps
    module.eval()
    pathway_names = [str(name) for name in checkpoint['pathway_names']]
    celltypes = [str(name) for name in checkpoint['celltype_classes']]
    mapping = module.adapter.prototype_to_celltype.to(device).float()
    examples = []
    with torch.inference_mode():
        for (sample_id, record) in records.items():
            truth = truth_for_record(record, drug_truth)
            if not truth:
                continue
            if sample_id in panel_ids:
                target = panel_celltype[sample_id]
                curated = True
                confidence = 1.0
            else:
                terminal = torch.as_tensor(record['terminal_celltype_proportion']).float()
                initial = torch.as_tensor(record['initial_celltype_proportion']).float()
                target_index = int(terminal.argmax())
                if float(terminal[target_index]) < args.weak_purity:
                    continue
                if float(terminal[target_index] - initial[target_index]) < args.weak_delta:
                    continue
                target = celltypes[target_index]
                curated = False
                confidence = 0.3
            target_index = celltypes.index(target)
            feature = make_feature(module, record, target_index, pathway_names, mapping, args.activity_strength)
            if args.feature_mode == 'no_activity':
                feature[38:57] = 0
            elif args.feature_mode == 'relation_only':
                feature[:57] = 0
            elif args.feature_mode == 'target_only':
                feature[:67] = 0
            examples.append(Example(sample_id, target, feature, make_label(truth, pathway_names), truth, confidence, curated))
    public_train = [e for e in examples if e.curated and e.sample_id not in PUBLIC_TEST]
    public_test = [e for e in examples if e.sample_id in PUBLIC_TEST]
    weak = [e for e in examples if not e.curated]
    if {e.sample_id for e in public_test} != PUBLIC_TEST:
        raise ValueError('fixed public test samples missing')
    seeds = [43, 44, 45, 46, 47]
    variants = {'strong_only': public_train, 'weak_augmented': public_train + weak}
    result = {'feature_mode': args.feature_mode, 'public_train_ids': [e.sample_id for e in public_train], 'public_test_ids': [e.sample_id for e in public_test], 'weak_example_count': len(weak), 'variants': {}}
    baseline_logits = torch.stack([e.feature[:len(pathway_names) * 2] for e in public_test])
    baseline_rows = score_predictions(public_test, baseline_logits, pathway_names)
    result['baseline'] = baseline_rows
    for (variant, train_examples) in variants.items():
        seed_logits = []
        for seed in seeds:
            weak_validation = []
            actual_train = train_examples
            if variant == 'weak_augmented':
                weak_validation = [e for (i, e) in enumerate(weak) if i % 7 == seed % 7]
                validation_ids = {e.sample_id for e in weak_validation}
                actual_train = public_train + [e for e in weak if e.sample_id not in validation_ids]
            (model, mean, std) = train_one(actual_train, weak_validation, seed, device)
            test_x = (torch.stack([e.feature for e in public_test]).to(device) - mean) / std
            with torch.inference_mode():
                seed_logits.append(model(test_x).cpu())
        ensemble_logits = torch.stack(seed_logits).mean(0)
        rows = score_predictions(public_test, ensemble_logits, pathway_names)
        result['variants'][variant] = {'train_count': len(train_examples), 'ensemble_seeds': seeds, 'rows': rows, 'signed_hits': sum((row['signed_hits'] for row in rows)), 'denominator': sum((row['denominator'] for row in rows)), 'identity_hits': sum((row['identity_hits'] for row in rows))}
        with (output / f'{variant}_test_scores.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'public_train_ids': result['public_train_ids'], 'public_test_ids': result['public_test_ids'], 'weak_example_count': len(weak), 'baseline_signed': f"{sum((r['signed_hits'] for r in baseline_rows))}/{sum((r['denominator'] for r in baseline_rows))}", 'variants': {name: f"{value['signed_hits']}/{value['denominator']}" for (name, value) in result['variants'].items()}}, indent=2), flush=True)
if __name__ == '__main__':
    main()
