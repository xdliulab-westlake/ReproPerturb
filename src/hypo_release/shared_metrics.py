import numpy as np
def metric(items, logits, mask, pathways):
    rows = []
    predictions = []
    for (item, signed_scores, chosen) in zip(items, logits, mask):
        pred = {pathways[i]: '+' if signed_scores[i, 0] >= signed_scores[i, 1] else '-' for i in np.flatnonzero(chosen)}
        predictions.append(pred)
        rows.append({'sample': item.sample_id, 'celltype': item.target_celltype, 'truth_k': len(item.truth), 'predicted_k': len(pred), 'identity': len(set(pred) & set(item.truth)), 'signed': len(set(pred.items()) & set(item.truth.items())), 'exact': pred == item.truth, 'prediction': pred})
    signed = sum((row['signed'] for row in rows))
    truth_total = sum((row['truth_k'] for row in rows))
    predicted_total = sum((row['predicted_k'] for row in rows))
    celltypes = sorted({e.target_celltype for e in items})
    macro_exact = np.mean([np.mean([r['exact'] for r in rows if r['celltype'] == celltype]) for celltype in celltypes])
    passed = sum((any((predictions[i] == e.truth for (i, q) in enumerate(items) if q.target_celltype == celltype for e in items if e.target_celltype == celltype)) for celltype in celltypes))
    f1 = 2 * signed / max(truth_total + predicted_total, 1)
    return {'identity': sum((r['identity'] for r in rows)), 'signed': signed, 'truth_total': truth_total, 'predicted_total': predicted_total, 'signed_f1': f1, 'exact_protocols': sum((r['exact'] for r in rows)), 'any_protocol_celltypes': passed, 'macro_exact': float(macro_exact), 'selection_utility': float(f1 + 0.15 * macro_exact), 'rows': rows}

def compact(result):
    return {key: value for (key, value) in result.items() if key != 'rows'}
