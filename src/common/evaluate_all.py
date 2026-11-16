"""Unified expression evaluation metrics used by perturbation benchmarks."""

from __future__ import annotations

import torch


def _corr(left: torch.Tensor, right: torch.Tensor, *, spearman: bool = False) -> torch.Tensor:
    if spearman:
        left = left.argsort(dim=-1).argsort(dim=-1).float()
        right = right.argsort(dim=-1).argsort(dim=-1).float()
    left = left.float() - left.float().mean(dim=-1, keepdim=True)
    right = right.float() - right.float().mean(dim=-1, keepdim=True)
    denominator = left.square().sum(dim=-1).sqrt() * right.square().sum(dim=-1).sqrt()
    numerator = (left * right).sum(dim=-1)
    return torch.where(denominator > 0, numerator / denominator, torch.nan)


def _r2(truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    residual = (truth - prediction).square().sum()
    centered = (truth - truth.mean()).square().sum()
    return torch.where(centered > 0, 1.0 - residual / centered, torch.nan)


def _energy_distance(truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    cross = torch.cdist(truth, prediction, p=2).mean()
    within_truth = torch.cdist(truth, truth, p=2).mean()
    within_prediction = torch.cdist(prediction, prediction, p=2).mean()
    return 2.0 * cross - within_truth - within_prediction


def evaluate_all(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    control: torch.Tensor,
    *,
    top_deg: int = 20,
    top_mae: int = 100,
) -> dict[str, float]:
    """Compute the 11 benchmark metrics from matched-control expression.

    ``truth`` and ``prediction`` are treated-sample expression matrices with
    shape ``[n_samples, n_genes]``; ``control`` contains the matched controls.
    Delta metrics use treated minus control. ``calibrated R²`` fits one affine
    calibration on all delta entries before calculating R². ``E-distance`` is
    the sample-distribution energy distance.
    """
    if truth.shape != prediction.shape or truth.shape != control.shape:
        raise ValueError("truth, prediction, and control must share [samples, genes] shape")
    if top_deg < 1 or top_deg > truth.shape[1] or top_mae < 1 or top_mae > truth.shape[1]:
        raise ValueError("top_deg and top_mae must be within the gene dimension")

    truth_delta = truth - control
    prediction_delta = prediction - control
    gene_corr = torch.stack([
        _corr(truth_delta[:, gene].unsqueeze(0), prediction_delta[:, gene].unsqueeze(0))
        for gene in range(truth.shape[1])
    ]).flatten()
    sample_corr = _corr(truth_delta, prediction_delta)
    raw_sample_corr = _corr(truth, prediction)
    spearman = _corr(truth_delta, prediction_delta, spearman=True)
    truth_centered = truth_delta - truth_delta.mean()
    prediction_centered = prediction_delta - prediction_delta.mean()
    slope = (truth_centered * prediction_centered).sum() / prediction_centered.square().sum().clamp_min(1e-12)
    intercept = truth_delta.mean() - slope * prediction_delta.mean()
    calibrated = slope * prediction_delta + intercept
    truth_top = torch.topk(truth_delta.abs(), top_deg, dim=-1).indices
    prediction_top = torch.topk(prediction_delta.abs(), top_deg, dim=-1).indices
    overlap = torch.stack([
        torch.isin(prediction_top[row], truth_top[row]).float().mean()
        for row in range(truth.shape[0])
    ])
    top_errors = torch.topk(truth_delta.abs(), top_mae, dim=-1).indices
    top_error_values = torch.gather((prediction_delta - truth_delta).abs(), 1, top_errors)

    def finite_mean(values: torch.Tensor) -> float:
        values = values[torch.isfinite(values)]
        return round(float(values.mean().item()), 6) if values.numel() else float("nan")

    def scalar(value: torch.Tensor) -> float:
        return round(float(value.item()), 6)

    return {
        "R²_delta": finite_mean(torch.stack([_r2(truth_delta[i], prediction_delta[i]) for i in range(truth.shape[0])])),
        "PCC_gene": finite_mean(gene_corr),
        "PCC_sample": finite_mean(sample_corr),
        "PCC_raw_sample": finite_mean(raw_sample_corr),
        "MSE": scalar((prediction_delta - truth_delta).square().mean()),
        "DEG@20": finite_mean(overlap),
        "MAE_top100": scalar(top_error_values.mean()),
        "per-gene R²": finite_mean(torch.stack([_r2(truth_delta[:, i], prediction_delta[:, i]) for i in range(truth.shape[1])])),
        "calibrated R²": scalar(_r2(truth_delta, calibrated)),
        "E-distance": scalar(_energy_distance(truth_delta, prediction_delta)),
        "Spearman": finite_mean(spearman),
    }
