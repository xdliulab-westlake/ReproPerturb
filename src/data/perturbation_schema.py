"""Validation for sign-only perturbation metadata."""

from __future__ import annotations

import csv

import torch


def validate_sign_matrix(sign_matrix: torch.Tensor) -> torch.Tensor:
    signs = torch.as_tensor(sign_matrix, dtype=torch.float32)
    if signs.ndim != 2:
        raise ValueError("sign matrix must be two-dimensional")
    if not torch.all((signs == -1) | (signs == 0) | (signs == 1)):
        raise ValueError("sign matrix may contain only -1, 0, and +1")
    return signs


def read_sign_csv(path, drugs, pathways) -> torch.Tensor:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    from stage2.model.pathway_condition import canonicalize_sign_table

    return validate_sign_matrix(canonicalize_sign_table(rows, drugs, pathways))
