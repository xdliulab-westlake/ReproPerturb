"""Shared relation math; no per-seed hyperparameter selection."""
import numpy as np
import torch
from scipy.linalg import solve
def kernel(a, b, c):
    dr = ((a[:, None, :34] - b[None, :, :34]) ** 2).mean(2)
    ds = ((a[:, None, 51:61] - b[None, :, 51:61]) ** 2).mean(2)
    same = a[:, None, 61:].argmax(2) == b[None, :, 61:].argmax(2)
    return np.exp(-(dr + c['state'] * ds) / c['length']) * (c['sharing'] + same)

def fitpredict(x, y, z, c):
    mu = x.mean(0)
    sd = np.maximum(x.std(0), 1e-05)
    a = (x - mu) / sd
    b = (z - mu) / sd
    a[:, 61:] = x[:, 61:]
    b[:, 61:] = z[:, 61:]
    weight = solve(kernel(a, a, c) + c['alpha'] * np.eye(len(x)), y, assume_a='pos')
    return (kernel(b, a, c) @ weight, dict(train=a, mean=mu, std=sd, weight=weight))

def combine(kernel, raw, ia, da):
    k = kernel.reshape(-1, 17, 2)
    r = raw[:, :34].reshape(-1, 17, 2)
    ki = k.max(2)
    ri = r.max(2)
    ki = (ki - ki.mean(1, keepdims=True)) / ki.std(1, keepdims=True).clip(1e-05)
    ri = (ri - ri.mean(1, keepdims=True)) / ri.std(1, keepdims=True).clip(1e-05)
    ident = ia * ki + (1 - ia) * ri
    kd = k[:, :, 0] - k[:, :, 1]
    rd = r[:, :, 0] - r[:, :, 1]
    kd = kd / kd.std(1, keepdims=True).clip(1e-05)
    rd = rd / rd.std(1, keepdims=True).clip(1e-05)
    direction = da * kd + (1 - da) * rd
    return np.stack([ident + direction / 2, ident - direction / 2], 2).reshape(-1, 34)

def evaluate(items, logits, paths, threshold=None):
    rows = []
    preds = []
    for (e, l) in zip(items, logits):
        (s, d) = l.reshape(17, 2).max(1)
        order = s.argsort(descending=True)
        ids = order[:len(e.truth)] if threshold is None else order[((s - s.mean()) / s.std().clamp_min(1e-05))[order] >= threshold]
        p = {paths[i]: '+' if d[i] == 0 else '-' for i in ids.tolist()}
        preds.append(p)
        rows.append(dict(sample=e.sample_id, celltype=e.target_celltype, prediction=p, truth_k=len(e.truth), predicted_k=len(p), signed=len(set(p.items()) & set(e.truth.items())), identity=len(set(p) & set(e.truth)), exact=p == e.truth))
    tp = sum((r['signed'] for r in rows))
    n = sum((r['predicted_k'] for r in rows))
    den = sum((r['truth_k'] for r in rows))
    passed = sum((any((preds[i] == e.truth for (i, q) in enumerate(items) if q.target_celltype == c for e in items if e.target_celltype == c)) for c in set((e.target_celltype for e in items))))
    exact = sum((r['exact'] for r in rows))
    f1 = 2 * tp / max(n + den, 1)
    return dict(signed=tp, identity=sum((r['identity'] for r in rows)), denominator=den, predicted_total=n, signed_f1=f1, exact=exact, celltypes_passed=passed, utility=f1 + 0.2 * exact / len(items), rows=rows)
