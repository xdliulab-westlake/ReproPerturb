"""Rank all 17 v741 pathways in observed two-drug training backgrounds.

Target identities and directions are used only after ranking to calculate metrics.
They never enter the intervention score.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import torch
HYPO_CODE = Path(__import__('hypo_release.paths', fromlist=['resolve']).resolve('source', ''))
if str(HYPO_CODE) not in sys.path:
    sys.path.insert(0, str(HYPO_CODE))
from experiments.perturbation.evaluate_prototype_mass_flow import load_evaluation_module, select_split_records
TARGETS = {'nodal_tgfb': -1, 'bmp': 1, 'wnt': 1, 'fgf': 1, 'jak_stat': 1, 'pdgf': 1}

def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow-checkpoint', required=True)
    parser.add_argument('--pair-records', required=True)
    parser.add_argument('--pathway-base', required=True)
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--context-drug-count', type=int, default=2)
    parser.add_argument('--tail-fraction', type=float, default=0.1)
    parser.add_argument('--activity-strength', type=float, default=0.75)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--n-ode-steps', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

def main():
    args = arguments()
    if not 0 < args.tail_fraction <= 1 or args.activity_strength <= 0:
        raise ValueError('invalid tail fraction or activity strength')
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True)
    device = torch.device(args.device)
    payload = torch.load(args.pair_records, map_location='cpu', weights_only=True)
    checkpoint = torch.load(args.flow_checkpoint, map_location='cpu', weights_only=True)
    module = load_evaluation_module(dict(checkpoint['config']), checkpoint, payload, args.pathway_base, device)
    module.adapter.endpoint_rollout_steps = args.n_ode_steps
    train = select_split_records(list(payload['records']), args.split_manifest, split='train')
    controls = {}
    for record in train:
        drug_ids = tuple(sorted(map(int, record.get('original_drug_ids', record['drug_ids']))))
        if len(drug_ids) == args.context_drug_count and len(set(drug_ids)) == len(drug_ids):
            controls.setdefault((str(record['control_sample_id']), drug_ids), record)
    contexts = list(controls.items())
    if not contexts:
        raise ValueError('no matching training contexts')
    pathway_names = [str(name) for name in checkpoint['pathway_names']]
    if len(pathway_names) != 17 or set(TARGETS) - set(pathway_names):
        raise ValueError(f'unexpected v741 pathways: {pathway_names}')
    target_index = [str(name) for name in checkpoint['celltype_classes']].index('Hypoblast-like cell')
    mapping = module.adapter.prototype_to_celltype.to(device).float()
    source_mass = torch.stack([torch.as_tensor(record['initial_mass_logit']).float() for (_, record) in contexts]).to(device)
    source_presence = torch.stack([torch.as_tensor(record['initial_presence']).float() for (_, record) in contexts]).to(device)
    max_drugs = max((len(key[1]) for (key, _) in contexts))
    drug_ids = torch.zeros(len(contexts), max_drugs, dtype=torch.long, device=device)
    drug_mask = torch.zeros(len(contexts), max_drugs, dtype=torch.bool, device=device)
    for (index, (key, _)) in enumerate(contexts):
        ids = key[1]
        drug_ids[index, :len(ids)] = torch.tensor(ids, device=device)
        drug_mask[index, :len(ids)] = True

    def predict(activity, mass, presence):
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(activity), args.batch_size):
                end = min(start + args.batch_size, len(activity))
                encoding = module.condition_encoder.encode_activities({'selected': activity[start:end]})
                (terminal_mass, endpoint) = module.adapter.rollout_endpoint(mass[start:end], presence[start:end], encoding.condition_tokens, encoding.pathway_context)
                proportion = module.adapter.velocity_model.presence_gated_proportion(terminal_mass, torch.sigmoid(endpoint.presence_logits)) @ mapping
                predictions.append(proportion[:, target_index].cpu())
        return torch.cat(predictions)
    with torch.inference_mode():
        base_activity = module.condition_encoder(drug_ids, drug_mask).activities['selected']
    baseline = predict(base_activity, source_mass, source_presence)
    (n, p) = (len(contexts), len(pathway_names))
    expanded_mass = source_mass.repeat_interleave(p, dim=0)
    expanded_presence = source_presence.repeat_interleave(p, dim=0)
    directional = {}
    for direction in (1, -1):
        intervention = base_activity.repeat_interleave(p, dim=0)
        pathway_index = torch.arange(p, device=device).repeat(n)
        intervention[torch.arange(n * p, device=device), pathway_index] += direction * args.activity_strength
        directional[direction] = predict(intervention, expanded_mass, expanded_presence).reshape(n, p)
    rows = []
    tail_count = max(1, math.ceil(n * args.tail_fraction))
    for (pathway_index, pathway) in enumerate(pathway_names):
        candidates = []
        for direction in (1, -1):
            gains = (directional[direction][:, pathway_index] - baseline).tolist()
            upper_tail = sorted(gains, reverse=True)[:tail_count]
            candidates.append((sum(upper_tail) / tail_count, direction, gains))
        (tail_mean, direction, gains) = max(candidates, key=lambda item: item[0])
        expected = TARGETS.get(pathway)
        rows.append({'pathway': pathway, 'predicted_direction': '+' if direction > 0 else '-', 'upper_tail_mean_gain': tail_mean, 'mean_gain': sum(gains) / n, 'max_gain': max(gains), 'context_count': n, 'is_ground_truth_target': pathway in TARGETS, 'expected_direction': '' if expected is None else '+' if expected > 0 else '-', 'direction_correct': '' if expected is None else direction == expected})
    rows.sort(key=lambda row: (-row['upper_tail_mean_gain'], row['pathway']))
    for (rank, row) in enumerate(rows, 1):
        row['rank'] = rank
        row['selected_top6'] = rank <= 6
    selected_rows = rows[:6]
    selected = [row['pathway'] for row in selected_rows]
    hit_rows = [row for row in selected_rows if row['pathway'] in TARGETS]
    signed_hit_rows = [row for row in hit_rows if row['direction_correct']]
    summary = {'method': 'upper-tail direct pathway marginal over observed train drug backgrounds', 'target_identities_used_in_score': False, 'target_directions_used_in_score': False, 'candidate_count': p, 'context_count': n, 'context_drug_count': args.context_drug_count, 'tail_fraction': args.tail_fraction, 'activity_strength': args.activity_strength, 'ground_truth_targets': {name: '+' if direction > 0 else '-' for (name, direction) in TARGETS.items()}, 'selected_pathways': selected, 'selected_signed': [f"{row['pathway']}{row['predicted_direction']}" for row in selected_rows], 'hits_at_6': len(hit_rows), 'signed_hits_at_6': len(signed_hit_rows), 'hit_pathways': [row['pathway'] for row in hit_rows], 'signed_hit_pathways': [row['pathway'] for row in signed_hit_rows], 'missed_pathways': [name for name in TARGETS if name not in selected]}
    with (output / 'pathway_ranking.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
