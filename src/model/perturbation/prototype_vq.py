"""Independent cell-prototype SoftVQ in the frozen Stage-1 latent space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PrototypeVQOutput:
    assignments: Tensor
    quantized: Tensor
    reconstruction_loss: Tensor
    balance_loss: Tensor

    @property
    def loss(self) -> Tensor:
        return self.reconstruction_loss + self.balance_loss


class PrototypeVQ(nn.Module):
    """One global K-slot prototype vocabulary for 128-D cell latents."""

    def __init__(
        self,
        latent_dim: int = 128,
        num_prototypes: int = 128,
        temperature: float = 0.07,
        balance_weight: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.balance_weight = float(balance_weight)
        self.codebook = nn.Parameter(torch.empty(num_prototypes, latent_dim))
        nn.init.normal_(self.codebook, std=latent_dim ** -0.5)

    def assignment_probabilities(self, latent: Tensor) -> Tensor:
        distance = torch.cdist(latent, self.codebook, p=2).square()
        return torch.softmax(-distance / self.temperature, dim=-1)

    def forward(self, latent: Tensor) -> PrototypeVQOutput:
        assignments = self.assignment_probabilities(latent)
        quantized = assignments @ self.codebook
        reconstruction_loss = F.mse_loss(quantized, latent)
        average_assignment = assignments.mean(dim=0)
        # KL(pi_bar || Uniform(K)): non-negative and zero for uniform usage.
        balance_loss = self.balance_weight * torch.sum(
            average_assignment * torch.log(
                (average_assignment * self.num_prototypes).clamp_min(1e-8)
            )
        )
        return PrototypeVQOutput(
            assignments=assignments,
            quantized=quantized,
            reconstruction_loss=reconstruction_loss,
            balance_loss=balance_loss,
        )


def select_high_frequency_product_prototypes(
    code_ids: Tensor,
    codebook: Tensor,
    num_prototypes: int,
    fit_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select frequent complete combinations from Stage-1 grouped codebooks.

    Stage-1 uses ``G`` independent codebooks of width ``D``.  A population
    prototype is a complete tuple of one code from every group, represented
    by concatenating the corresponding vectors into ``[G * D]``.  Marginal
    group codes are deliberately not treated as biological prototypes.
    """
    if fit_mask is None:
        fit_mask = torch.ones(code_ids.shape[0], dtype=torch.bool, device=code_ids.device)
    fit_mask = fit_mask.to(device=code_ids.device, dtype=torch.bool)
    unique_codes, counts = torch.unique(code_ids[fit_mask], dim=0, return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    n_selected = min(int(num_prototypes), int(unique_codes.shape[0]))
    selected_codes = unique_codes[order[:n_selected]].long()
    selected_counts = counts[order[:n_selected]].long()
    groups = torch.arange(codebook.shape[0], device=codebook.device).unsqueeze(0)
    vectors = codebook[groups, selected_codes.to(codebook.device)]
    return vectors.reshape(n_selected, -1).float(), selected_codes, selected_counts


def select_celltype_balanced_product_prototypes(
    code_ids: Tensor,
    codebook: Tensor,
    labels: Tensor,
    sample_ids: list[str],
    num_prototypes: int,
    num_celltypes: int,
    min_per_celltype: int = 6,
    min_sample_support: int = 2,
    score_weights: dict[str, float] | None = None,
    fit_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor | list[str]]]:
    """Reserve class-specific product codes, then fill by global frequency.

    Class candidates balance within-class coverage, class specificity, and
    cross-sample support.  The legacy global-frequency selector remains a
    separate function so existing prototype vocabularies stay reproducible.
    """
    if code_ids.ndim != 2:
        raise ValueError("code_ids must be [cells, groups]")
    if len(labels) != len(code_ids) or len(sample_ids) != len(code_ids):
        raise ValueError("code_ids, labels, and sample_ids must have matching rows")
    requested = int(num_prototypes)
    quota = int(min_per_celltype)
    required = quota * int(num_celltypes)
    if requested < required:
        raise ValueError(
            f"num_prototypes={requested} cannot reserve {quota} slots for "
            f"each of {num_celltypes} cell types"
        )
    if quota < 0 or min_sample_support < 1:
        raise ValueError("selection quota must be nonnegative and sample support positive")
    weights = {
        "coverage": 0.5,
        "specificity": 0.5,
        "sample_support": 0.25,
        **(score_weights or {}),
    }
    if set(weights) != {"coverage", "specificity", "sample_support"}:
        raise ValueError("score_weights must contain only coverage, specificity, sample_support")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("selection score weights must be nonnegative")

    if fit_mask is None:
        fit_mask = torch.ones(len(code_ids), dtype=torch.bool, device=code_ids.device)
    fit_mask = fit_mask.to(device=code_ids.device, dtype=torch.bool)
    fit_codes = code_ids[fit_mask].long()
    fit_labels = labels.to(code_ids.device)[fit_mask].long()
    if fit_labels.numel() == 0:
        raise ValueError("cannot select prototypes without fitting cells")
    label_counts = torch.bincount(fit_labels, minlength=num_celltypes)
    if (label_counts == 0).any():
        missing = torch.where(label_counts == 0)[0].tolist()
        raise ValueError(f"cannot reserve prototypes without cells for labels: {missing}")

    unique_codes, inverse, global_counts = torch.unique(
        fit_codes, dim=0, return_inverse=True, return_counts=True,
    )
    class_counts = torch.zeros(
        len(unique_codes), num_celltypes, dtype=torch.long, device=code_ids.device,
    )
    class_counts.index_put_((inverse, fit_labels), torch.ones_like(inverse), accumulate=True)
    sample_to_id = {sample: index for index, sample in enumerate(dict.fromkeys(sample_ids))}
    sample_index = torch.tensor(
        [sample_to_id[sample] for sample in sample_ids],
        dtype=torch.long,
        device=code_ids.device,
    )[fit_mask]
    code_sample_pairs = torch.unique(torch.stack([inverse, sample_index], dim=1), dim=0)
    global_sample_support = torch.bincount(
        code_sample_pairs[:, 0], minlength=len(unique_codes),
    )
    code_class_sample = torch.unique(
        torch.stack([inverse, fit_labels, sample_index], dim=1), dim=0,
    )
    class_sample_support = torch.zeros(
        len(unique_codes), num_celltypes, dtype=torch.long, device=code_ids.device,
    )
    class_sample_support.index_put_(
        (code_class_sample[:, 0], code_class_sample[:, 1]),
        torch.ones(len(code_class_sample), dtype=torch.long, device=code_ids.device),
        accumulate=True,
    )

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    sources: list[str] = []
    target_celltypes: list[int] = []
    selected_scores: list[float] = []
    eps = torch.finfo(torch.float32).eps
    coverage = class_counts.float() / label_counts.clamp_min(1).float().unsqueeze(0)
    specificity = class_counts.float() / global_counts.clamp_min(1).float().unsqueeze(1)
    normalized_support = torch.log1p(class_sample_support.float())
    normalized_support = normalized_support / normalized_support.amax(dim=0, keepdim=True).clamp_min(1.0)
    scores = (
        coverage.clamp_min(eps).pow(float(weights["coverage"]))
        * specificity.clamp_min(eps).pow(float(weights["specificity"]))
        * normalized_support.clamp_min(eps).pow(float(weights["sample_support"]))
    )

    for celltype in range(num_celltypes):
        supported_candidates = torch.where(
            (class_counts[:, celltype] > 0)
            & (class_sample_support[:, celltype] >= min_sample_support)
        )[0]
        all_candidates = torch.where(class_counts[:, celltype] > 0)[0]
        rank_key = lambda index: (
            -float(scores[index, celltype]),
            -int(class_counts[index, celltype]),
            tuple(int(value) for value in unique_codes[index].tolist()),
        )
        candidate_phases = (
            (sorted(supported_candidates.tolist(), key=rank_key), "celltype_quota"),
            (sorted(all_candidates.tolist(), key=rank_key), "celltype_quota_fallback"),
        )
        chosen = 0
        for candidate_order, source in candidate_phases:
            for index in candidate_order:
                if index in selected_set:
                    continue
                selected_indices.append(index)
                selected_set.add(index)
                sources.append(source)
                target_celltypes.append(celltype)
                selected_scores.append(float(scores[index, celltype]))
                chosen += 1
                if chosen == quota:
                    break
            if chosen == quota:
                break
        if chosen < quota:
            # A very homogeneous cell type (e.g. fibroblasts under a sharp
            # quantizer) can occupy fewer distinct tuples than the quota.
            # Keep every tuple it has; the global-frequency pass below fills
            # the freed slots, so the vocabulary still reaches num_prototypes.
            # metadata["target_celltype"] records how many slots each type got.
            # Previously this raised, so no earlier successful vocabulary changes.
            print(
                f"[prototype selection] cell type {celltype} has only {chosen} "
                f"unique codes; reserved {chosen} of {quota} quota slots, "
                f"{quota - chosen} go to global frequency",
                flush=True,
            )

    global_order = sorted(
        range(len(unique_codes)),
        key=lambda index: (
            -int(global_counts[index]),
            tuple(int(value) for value in unique_codes[index].tolist()),
        ),
    )
    for index in global_order:
        if len(selected_indices) == requested:
            break
        if index in selected_set:
            continue
        selected_indices.append(index)
        selected_set.add(index)
        sources.append("global_frequency")
        target_celltypes.append(-1)
        selected_scores.append(float("nan"))

    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=code_ids.device)
    selected_codes = unique_codes[index_tensor]
    selected_counts = global_counts[index_tensor]
    groups = torch.arange(codebook.shape[0], device=codebook.device).unsqueeze(0)
    vectors = codebook[groups, selected_codes.to(codebook.device)]
    metadata: dict[str, Tensor | list[str]] = {
        "source": sources,
        "target_celltype": torch.tensor(target_celltypes, dtype=torch.long),
        "score": torch.tensor(selected_scores, dtype=torch.float32),
        "sample_support": torch.tensor([
            int(class_sample_support[index, target]) if target >= 0
            else int(global_sample_support[index])
            for index, target in zip(selected_indices, target_celltypes)
        ], dtype=torch.long),
        "class_sample_support": class_sample_support[index_tensor].cpu(),
        "class_counts": class_counts[index_tensor].cpu(),
    }
    return (
        vectors.reshape(len(selected_codes), -1).float(),
        selected_codes,
        selected_counts,
        metadata,
    )


@torch.no_grad()
def product_code_assignment_probabilities(
    z_e: Tensor,
    codebook: Tensor,
    selected_codes: Tensor,
    temperature: float = 0.07,
    l2_norm: bool = True,
) -> Tensor:
    """Return assignments to selected complete Stage-1 product-code tuples.

    The probability of a tuple is the product of its four grouped SoftVQ
    probabilities, renormalized over the selected vocabulary.  This retains
    the original Stage-1 activation semantics instead of re-assigning cells
    to label-mean centroids in the post-quantized space.
    """
    if z_e.ndim == 2:
        z_e = z_e.reshape(z_e.shape[0], codebook.shape[0], codebook.shape[2])
    vectors = torch.nn.functional.normalize(codebook.float(), dim=-1) if l2_norm else codebook.float()
    states = torch.nn.functional.normalize(z_e.float(), dim=-1) if l2_norm else z_e.float()
    logits = torch.einsum("ngd,gmd->ngm", states, vectors) / float(temperature)
    probabilities = logits.softmax(dim=-1)
    selected = selected_codes.to(probabilities.device)
    index = selected.unsqueeze(0).expand(probabilities.shape[0], -1, -1)
    grouped = probabilities.unsqueeze(1).expand(-1, selected.shape[0], -1, -1)
    tuple_prob = grouped.gather(-1, index.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12).log().sum(-1)
    return tuple_prob.softmax(dim=-1)


def build_celltype_prototype_codebook(
    latents: Tensor,
    labels: Tensor,
    num_celltypes: int,
    fit_mask: Tensor | None = None,
) -> Tensor:
    """Return one frozen prototype mean for each labeled cell type.

    ``fit_mask`` makes it possible to exclude validation/test cells when the
    prototype vocabulary is precomputed from frozen Stage-1 ``z_q`` states.
    """
    if fit_mask is None:
        fit_mask = torch.ones(len(latents), dtype=torch.bool, device=latents.device)
    fit_mask = fit_mask.to(device=latents.device, dtype=torch.bool)
    fit_labels = labels[fit_mask].long()
    fit_latents = latents[fit_mask]
    counts = torch.bincount(fit_labels, minlength=num_celltypes)
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"cannot fit prototypes without cells for labels: {missing}")
    sums = latents.new_zeros(num_celltypes, latents.shape[1])
    sums.index_add_(0, fit_labels, fit_latents)
    return sums / counts.to(latents.dtype).unsqueeze(-1)


def build_prototype_celltype_map(
    assignments: Tensor, labels: Tensor, num_celltypes: int,
) -> Tensor:
    """Estimate R[k,c] = P(celltype=c | prototype=k) from soft assignments."""
    target = F.one_hot(labels.long(), num_classes=num_celltypes).to(assignments.dtype)
    counts = assignments.transpose(0, 1) @ target
    empty = counts.sum(dim=-1, keepdim=True) == 0
    counts = torch.where(empty, torch.ones_like(counts), counts)
    return counts / counts.sum(dim=-1, keepdim=True)


def build_prototype_celltype_map_from_sample_counts(
    sample_counts: dict[str, Tensor], sample_ids: list[str],
) -> Tensor:
    """Estimate ``R`` from per-sample prototype×celltype counts.

    This helper lets Stage-2 fit the mapping after the biological sample split,
    preventing validation/test labels from entering the prototype annotation.
    """
    values = [torch.as_tensor(sample_counts[sample]).float() for sample in sample_ids]
    return _map_from_counts(torch.stack(values).sum(dim=0))


def _map_from_counts(counts: Tensor) -> Tensor:
    counts = torch.as_tensor(counts).float()
    denominator = counts.sum(dim=-1, keepdim=True)
    # A prototype absent from the training split has no label evidence.  Keep
    # it usable with a neutral (uniform) mapping instead of propagating NaNs.
    uniform = torch.full_like(counts, 1.0 / counts.shape[1])
    return torch.where(denominator > 0, counts / denominator.clamp_min(1e-8), uniform)
