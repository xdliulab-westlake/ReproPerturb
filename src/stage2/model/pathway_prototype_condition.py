"""Pathway-to-prototype conditions for AdaLN-injected prototype flow."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from stage2.model.condition_types import ConditionEncoding


class PathwayPrototypeCondition(nn.Module):
    """Build fixed pathway tokens and prototype-specific drug contexts."""

    def __init__(
        self,
        pathway_gene_embedding: Tensor,
        num_drugs: int,
        num_prototypes: int,
        mode: str,
        sign_matrix: Tensor | None = None,
        unknown_effect_rank: int = 4,
    ):
        super().__init__()
        if pathway_gene_embedding.ndim != 2 or pathway_gene_embedding.shape[1] != 768:
            raise ValueError("pathway_gene_embedding must have shape [P,768]")
        if num_drugs < 1 or num_prototypes < 1:
            raise ValueError("num_drugs and num_prototypes must be positive")
        if mode not in {"direct", "known_prior", "known_unknown_prior"}:
            raise ValueError(f"unsupported pathway-prototype mode: {mode!r}")
        if mode == "direct" and sign_matrix is not None:
            raise ValueError("direct mode must not receive a sign_matrix")
        if mode != "direct" and sign_matrix is None:
            raise ValueError(f"{mode} requires a sign_matrix")
        if sign_matrix is not None and tuple(sign_matrix.shape) != (
            num_drugs, pathway_gene_embedding.shape[0],
        ):
            raise ValueError("sign_matrix must have shape [D,P]")
        if unknown_effect_rank < 1:
            raise ValueError("unknown_effect_rank must be positive")

        self.register_buffer("pathway_gene_embedding", pathway_gene_embedding.float())
        self.num_pathways = int(pathway_gene_embedding.shape[0])
        self.num_prototypes = int(num_prototypes)
        self.mode = mode
        self.unknown_effect_rank = int(unknown_effect_rank)
        self.pathway_projection = nn.Linear(768, 256)
        if mode == "direct":
            self.direct_drug_pathway = nn.Parameter(torch.empty(num_drugs, self.num_pathways))
            self.direct_pathway_prototype = nn.Parameter(
                torch.empty(self.num_pathways, self.num_prototypes)
            )
            nn.init.normal_(self.direct_drug_pathway, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.direct_pathway_prototype)
        else:
            self.register_buffer("sign_matrix", sign_matrix.float())
            known_init = torch.log(torch.expm1(torch.tensor(1.0))).item()
            self.raw_known_magnitude = nn.Parameter(
                torch.full((num_drugs, self.num_pathways), known_init)
            )
            self.known_pathway_prototype = nn.Parameter(
                torch.empty(self.num_pathways, self.num_prototypes)
            )
            nn.init.xavier_uniform_(self.known_pathway_prototype)
            if mode == "known_unknown_prior":
                self.unknown_drug_factors = nn.Parameter(
                    torch.randn(num_drugs, self.unknown_effect_rank) * 0.02
                )
                self.unknown_pathway_factors = nn.Parameter(
                    torch.zeros(self.num_pathways, self.unknown_effect_rank)
                )
                self.unknown_pathway_gate = nn.Parameter(torch.ones(self.num_pathways))
                self.unknown_pathway_prototype = nn.Parameter(
                    torch.empty(self.num_pathways, self.num_prototypes)
                )
                nn.init.xavier_uniform_(self.unknown_pathway_prototype)

    @property
    def condition_token_dim(self) -> int:
        return 256

    @property
    def activity_keys(self) -> tuple[str, ...]:
        if self.mode == "direct":
            return ("direct",)
        if self.mode == "known_unknown_prior":
            return ("known", "unknown")
        return ("known",)

    def pathway_tokens(self) -> Tensor:
        return self.pathway_projection(self.pathway_gene_embedding)

    def pathway_activities(self, drug_ids: Tensor, drug_mask: Tensor) -> dict[str, Tensor]:
        if self.mode == "direct":
            individual = self.direct_drug_pathway[drug_ids]
            individual = individual * drug_mask.unsqueeze(-1).to(individual)
            return {"direct": torch.tanh(individual.sum(dim=1))}

        known_individual = self.known_effect_matrix()[drug_ids]
        known_individual = known_individual * drug_mask.unsqueeze(-1).to(known_individual)
        activities = {"known": torch.tanh(known_individual.sum(dim=1))}
        if self.mode == "known_unknown_prior":
            unknown_individual = self.unknown_effect_matrix()[drug_ids]
            unknown_individual = unknown_individual * drug_mask.unsqueeze(-1).to(unknown_individual)
            activities["unknown"] = torch.tanh(unknown_individual.sum(dim=1))
        return activities

    def prototype_pathway_strength(self, activities: dict[str, Tensor]) -> Tensor:
        if self.mode == "direct":
            activity = activities["direct"]
            return activity.unsqueeze(1) * self.direct_pathway_prototype.T.unsqueeze(0)
        strength = (
            activities["known"].unsqueeze(1)
            * self.known_pathway_prototype.T.unsqueeze(0)
        )
        if self.mode == "known_unknown_prior":
            strength = strength + (
                activities["unknown"].unsqueeze(1)
                * self.unknown_pathway_prototype.T.unsqueeze(0)
            )
        return strength

    def encode_activities(self, activities: dict[str, Tensor]) -> ConditionEncoding:
        if set(activities) != set(self.activity_keys):
            raise ValueError(
                f"{self.mode} pathway activities must contain {self.activity_keys}"
            )
        batch_sizes = set()
        for name, activity in activities.items():
            if activity.ndim != 2 or activity.shape[1] != self.num_pathways:
                raise ValueError(f"{name} pathway activity must have shape [B,P]")
            batch_sizes.add(int(activity.shape[0]))
        if len(batch_sizes) != 1:
            raise ValueError("all pathway activity channels must share a batch size")
        batch_size = batch_sizes.pop()
        pathway_tokens = self.pathway_tokens()
        strength = self.prototype_pathway_strength(activities)
        return ConditionEncoding(
            condition_tokens=pathway_tokens.unsqueeze(0).expand(batch_size, -1, -1),
            pathway_context=strength @ pathway_tokens,
            activities=activities,
            diagnostics={"prototype_pathway_strength": strength},
        )

    def known_effect_matrix(self) -> Tensor:
        return self.sign_matrix * F.softplus(self.raw_known_magnitude)

    def unknown_effect_matrix(self) -> Tensor:
        if self.mode != "known_unknown_prior":
            return self.pathway_gene_embedding.new_zeros(self.sign_matrix.shape)
        score = self.unknown_drug_factors @ self.unknown_pathway_factors.T
        score = score / math.sqrt(self.unknown_effect_rank)
        unknown = torch.tanh(score * self.unknown_pathway_gate.unsqueeze(0))
        return torch.where(self.sign_matrix == 0, unknown, torch.zeros_like(unknown))

    def singleton_effect_matrices(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return known, unknown, and selected singleton Drug→Pathway effects."""
        if self.mode == "direct":
            direct = self.direct_drug_pathway
            return direct, torch.zeros_like(direct), direct
        known = self.known_effect_matrix()
        unknown = self.unknown_effect_matrix()
        selected = torch.where(self.sign_matrix == 0, unknown, known)
        return known, unknown, selected

    def unknown_regularization_terms(self) -> dict[str, Tensor]:
        zero = self.pathway_gene_embedding.new_zeros(())
        if self.mode != "known_unknown_prior":
            return {"unknown_effect": zero, "unknown_gate": zero}
        unknown_mask = self.sign_matrix == 0
        if not bool(unknown_mask.any()):
            return {"unknown_effect": zero, "unknown_gate": zero}
        unknown_effect = self.unknown_effect_matrix()
        return {
            "unknown_effect": unknown_effect.abs().masked_select(unknown_mask).mean(),
            "unknown_gate": self.unknown_pathway_gate.abs().mean(),
        }

    def forward(self, drug_ids: Tensor, drug_mask: Tensor) -> ConditionEncoding:
        if drug_ids.ndim != 2 or drug_mask.shape != drug_ids.shape:
            raise ValueError("drug_ids and drug_mask must both have shape [B,L]")
        activities = self.pathway_activities(drug_ids, drug_mask)
        return self.encode_activities(activities)
