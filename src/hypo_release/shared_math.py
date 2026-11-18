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
