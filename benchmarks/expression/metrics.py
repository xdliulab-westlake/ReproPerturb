import numpy as np
from scipy.stats import rankdata

def mean(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if valid.size else None


def correlation(a, b, axis):
    a = a - a.mean(axis, keepdims=True)
    b = b - b.mean(axis, keepdims=True)
    denominator = np.sqrt((a*a).sum(axis) * (b*b).sum(axis))
    return np.divide((a*b).sum(axis), denominator,
                     out=np.full_like(denominator, np.nan), where=denominator > 1e-12)


def r2(prediction, truth):
    denominator = ((truth - truth.mean(1, keepdims=True))**2).sum(1)
    return 1 - np.divide(((prediction-truth)**2).sum(1), denominator,
                         out=np.full_like(denominator, np.nan), where=denominator > 1e-12)


def expression_metrics(prediction, truth, control):
    prediction, truth, control = [np.asarray(x, dtype=np.float64) for x in (prediction, truth, control)]
    if prediction.ndim != 2 or prediction.shape != truth.shape or truth.shape != control.shape or min(truth.shape) == 0:
        raise ValueError('Expression matrices must have identical nonempty [conditions, genes] shapes')
    if not all(np.isfinite(x).all() for x in (prediction, truth, control)):
        raise ValueError('Nonfinite expression cannot be omitted')
    p, t = prediction-control, truth-control
    values = {
        'PCC_sample_raw': correlation(prediction, truth, 1),
        'PCC_gene_raw': correlation(prediction, truth, 0),
        'R2_raw': r2(prediction, truth),
        'PCC_sample_delta': correlation(p, t, 1),
        'PCC_gene_delta': correlation(p, t, 0),
        'R2_delta': r2(p, t),
        'Spearman_delta': correlation(rankdata(p, axis=1, method='average'), rankdata(t, axis=1, method='average'), 1),
        'MSE_raw': ((prediction-truth)**2).mean(1),
    }
    jaccard, overlap, mae = [], [], []
    for i in range(len(p)):
        pi = np.argsort(-np.abs(p[i]), kind='stable')
        ti = np.argsort(-np.abs(t[i]), kind='stable')
        valid = np.ptp(np.abs(p[i])) > 1e-12 and np.ptp(np.abs(t[i])) > 1e-12
        n100, n20 = min(100, len(pi)), min(20, len(pi))
        a, b = set(pi[:n100]), set(ti[:n100])
        jaccard.append(len(a & b) / len(a | b) if valid else np.nan)
        overlap.append(len(set(pi[:n20]) & set(ti[:n20])) / n20 if valid else np.nan)
        mae.append(np.abs(prediction[i]-truth[i])[ti[:n100]].mean())
    values.update(DEG_Jaccard100=np.array(jaccard), DEG_overlap20=np.array(overlap), MAE_top100=np.array(mae))
    summary = {key: mean(value) for key, value in values.items()}
    summary.update(conditions=len(truth), genes=truth.shape[1])
    for key, value in values.items():
        unit = 'genes' if key.startswith('PCC_gene') else 'conditions'
        summary[f'{key}_defined_{unit}'] = int(np.isfinite(value).sum())
    if not np.isclose(summary['MSE_raw'], ((p-t)**2).mean(), rtol=1e-12, atol=1e-12):
        raise ValueError('Raw/delta MSE identity failed')
    rows = [{key: mean([value[i]]) for key, value in values.items() if not key.startswith('PCC_gene')}
            for i in range(len(truth))]
    return {'summary': summary, 'per_condition': rows}
