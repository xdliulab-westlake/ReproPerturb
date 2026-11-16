"""Old-report-compatible 11-protocol validation using frozen-Flow context."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
import torch
HYPO_CODE = Path(__import__('hypo_release.paths', fromlist=['resolve']).resolve('source', ''))
if str(HYPO_CODE) not in sys.path:
    sys.path.insert(0, str(HYPO_CODE))
from experiments.perturbation.evaluate_prototype_mass_flow import load_evaluation_module

def load_aux():
    path = Path(__import__('hypo_release.paths', fromlist=['resolve']).resolve('support', 'pathway_celltype_aux_experiment.py'))
    spec = importlib.util.spec_from_file_location('v741_context_aux', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow-checkpoint', required=True)
    p.add_argument('--pair-records', required=True)
    p.add_argument('--pathway-base', required=True)
    p.add_argument('--panel-csv', required=True)
    p.add_argument('--sign-csv', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--activity-strength', type=float, default=0.75)
    p.add_argument('--n-ode-steps', type=int, default=20)
    p.add_argument('--weak-purity', type=float, default=0.4)
    p.add_argument('--weak-delta', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def net_truth(drugs, lookup):
    score = defaultdict(int)
    for drug in drugs:
        (pathway, sign) = lookup[str(drug)]
        score[pathway] += sign
    return {p: '+' if value > 0 else '-' for (p, value) in score.items() if value}

def any_protocol(examples, logits, pathways, aux):
    predictions = {}
    for (item, logit) in zip(examples, logits):
        predictions[item.sample_id] = dict(aux.rank_actions(logit, pathways, len(item.truth)))
    groups = defaultdict(list)
    for item in examples:
        groups[item.target_celltype].append(item)
    rows = []
    for (celltype, items) in sorted(groups.items()):
        exact = []
        best = None
        for prediction_item in items:
            prediction = predictions[prediction_item.sample_id]
            for truth_item in items:
                truth = truth_item.truth
                signed = len(set(prediction.items()) & set(truth.items()))
                identity = len(set(prediction) & set(truth))
                candidate = (signed, identity, prediction_item.sample_id, truth_item.sample_id)
                if best is None or candidate > best:
                    best = candidate
                if prediction == truth:
                    exact.append((prediction_item.sample_id, truth_item.sample_id))
        (signed, identity, pred_id, truth_id) = best
        rows.append({'celltype': celltype, 'passed': bool(exact), 'exact_matches': str(exact), 'best_signed_hits': signed, 'best_identity_hits': identity, 'prediction_context': pred_id, 'matched_protocol': truth_id})
    return rows

def main():
    a = args()
    out = Path(a.output_dir)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True)
    aux = load_aux()
    device = torch.device(a.device)
    panel = read_csv(a.panel_csv)
    targets = {row['sample_id']: row['expected_celltype'] for row in panel}
    sign_rows = read_csv(a.sign_csv)
    lookup = {row['drug']: (row['pathway'], int(row['sign'])) for row in sign_rows}
    payload = torch.load(a.pair_records, map_location='cpu', weights_only=True)
    checkpoint = torch.load(a.flow_checkpoint, map_location='cpu', weights_only=True)
    records = {str(row['sample_id']): row for row in payload['records']}
    module = load_evaluation_module(dict(checkpoint['config']), checkpoint, payload, a.pathway_base, device)
    module.adapter.endpoint_rollout_steps = a.n_ode_steps
    module.eval()
    pathways = list(map(str, checkpoint['pathway_names']))
    celltypes = list(map(str, checkpoint['celltype_classes']))
    mapping = module.adapter.prototype_to_celltype.to(device).float()
    (train, test) = ([], [])
    with torch.inference_mode():
        for (sample_id, record) in records.items():
            truth = net_truth(record['drug_names'], lookup)
            if not truth:
                continue
            if sample_id in targets:
                target = targets[sample_id]
                (destination, confidence, curated) = (test, 1.0, True)
            else:
                terminal = torch.as_tensor(record['terminal_celltype_proportion']).float()
                initial = torch.as_tensor(record['initial_celltype_proportion']).float()
                target_index = int(terminal.argmax())
                if float(terminal[target_index]) < a.weak_purity:
                    continue
                if float(terminal[target_index] - initial[target_index]) < a.weak_delta:
                    continue
                target = celltypes[target_index]
                (destination, confidence, curated) = (train, 0.3, False)
            feature = aux.make_feature(module, record, celltypes.index(target), pathways, mapping, a.activity_strength)
            p = len(pathways)
            feature[2 * p:3 * p] = 0
            destination.append(aux.Example(sample_id, target, feature, aux.make_label(truth, pathways), truth, confidence, curated))
    if set(targets) != {item.sample_id for item in test}:
        raise ValueError('the fixed 11-protocol panel is incomplete')
    seed_logits = []
    for seed in [43, 44, 45, 46, 47]:
        (model, mean, std) = aux.train_one(train, [], seed, device, epochs=a.epochs)
        x = (torch.stack([item.feature for item in test]).to(device) - mean) / std
        model.eval()
        with torch.inference_mode():
            seed_logits.append(model(x).cpu())
    logits = torch.stack(seed_logits).mean(0)
    sample_rows = aux.score_predictions(test, logits, pathways)
    any_rows = any_protocol(test, logits, pathways, aux)
    write_csv(out / 'sample_scores.csv', sample_rows)
    write_csv(out / 'any_protocol.csv', any_rows)
    full_rows = []
    for (item, logit) in zip(test, logits):
        action = logit.view(len(pathways), 2)
        (score, direction) = action.max(1)
        order = score.argsort(descending=True)
        for (rank, index) in enumerate(order.tolist(), 1):
            pathway = pathways[index]
            full_rows.append({'sample_id': item.sample_id, 'target_celltype': item.target_celltype, 'pathway': pathway, 'rank': rank, 'score': float(score[index]), 'predicted_direction': '+' if int(direction[index]) == 0 else '-', 'truth_direction': item.truth.get(pathway, ''), 'selected': rank <= len(item.truth)})
    write_csv(out / 'full_rankings.csv', full_rows)
    denominator = sum((row['denominator'] for row in sample_rows))
    summary = {'protocol': 'old-data weak-relation training; 11 annotations held out', 'ontology': 'v741 17 pathways', 'flow_checkpoint': str(Path(a.flow_checkpoint).resolve()), 'feature_mode': 'no explicit pathway activity; retain Flow contextual marginal gains + initial state + target cell type', 'test_labels_used_in_training': False, 'test_terminal_endpoints_used_as_features': False, 'test_drug_context_used_by_frozen_flow': True, 'weak_train_count': len(train), 'test_count': len(test), 'ensemble_seeds': [43, 44, 45, 46, 47], 'identity_hits': sum((row['identity_hits'] for row in sample_rows)), 'signed_hits': sum((row['signed_hits'] for row in sample_rows)), 'denominator': denominator, 'any_protocol_celltypes_passed': sum((row['passed'] for row in any_rows)), 'celltype_total': len(any_rows)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
