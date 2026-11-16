"""Shared pathway-to-cell-type bottleneck for pathway AdaLN conditioning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.perturbation.condition_types import ConditionEncoding
from model.perturbation.pathway_condition import SignedInteractionCondition


class PathwayCelltypeCondition(nn.Module):
    """Wrap the canonical pathway encoder with one auditable ``W[P,C]``."""

    def __init__(
        self,
        base_encoder: SignedInteractionCondition,
        prototype_to_celltype: Tensor,
        expression_dim: int,
        *,
        dynamic_token_route: str,
        route_dropout_probability: float,
        adaln_context_dim: int = 256,
        shuffle_contrastive: bool = False,
        shuffle_margin_fraction: float = 0.05,
        combination_rank: int | None = None,
        combination_scale: float = 0.5,
    ):
        super().__init__()
        if dynamic_token_route not in {"canonical", "static"}:
            raise ValueError("dynamic_token_route must be 'canonical' or 'static'")
        if not 0.0 <= route_dropout_probability <= 1.0:
            raise ValueError("route_dropout_probability must be in [0,1]")
        if prototype_to_celltype.ndim != 2:
            raise ValueError("prototype_to_celltype must have shape [K,C]")
        if expression_dim < 1 or adaln_context_dim < 1:
            raise ValueError("expression and AdaLN context dimensions must be positive")
        if combination_rank is not None and combination_rank < 1:
            raise ValueError("combination_rank must be positive when enabled")
        if shuffle_margin_fraction < 0:
            raise ValueError("shuffle_margin_fraction must be nonnegative")

        self.base_encoder = base_encoder
        self.register_buffer("prototype_to_celltype", prototype_to_celltype.float())
        self.dynamic_token_route = dynamic_token_route
        self.route_dropout_probability = float(route_dropout_probability)
        self.shuffle_contrastive = bool(shuffle_contrastive)
        self.shuffle_margin_fraction = float(shuffle_margin_fraction)
        self.combination_rank = combination_rank
        self.combination_scale = float(combination_scale)

        num_pathways = int(base_encoder.pathway_base.shape[0])
        num_celltypes = int(prototype_to_celltype.shape[1])
        self.pathway_celltype_weight = nn.Parameter(
            torch.zeros(num_pathways, num_celltypes)
        )
        self.pathway_projection = nn.Linear(
            base_encoder.pathway_base.shape[1], adaln_context_dim
        )
        self.expression_projection = nn.Linear(
            adaln_context_dim, expression_dim, bias=False
        )
        self.register_buffer(
            "pathway_shuffle_permutation",
            torch.roll(torch.arange(num_pathways), shifts=1)
            if self.shuffle_contrastive
            else None,
        )
        if combination_rank is not None:
            self.combination_left = nn.Parameter(
                torch.empty(num_pathways, combination_rank)
            )
            self.combination_right = nn.Parameter(
                torch.empty(num_pathways, combination_rank)
            )
            self.combination_out = nn.Parameter(
                torch.zeros(combination_rank, num_pathways)
            )
            nn.init.xavier_uniform_(self.combination_left)
            nn.init.xavier_uniform_(self.combination_right)

    @property
    def condition_token_dim(self) -> int:
        return self.base_encoder.condition_token_dim

    @property
    def activity_keys(self) -> tuple[str, ...]:
        return self.base_encoder.activity_keys

    @property
    def pathway_base(self) -> Tensor:
        return self.base_encoder.pathway_base

    @property
    def sign_matrix(self) -> Tensor:
        return self.base_encoder.sign_matrix

    def singleton_effect_matrices(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.base_encoder.singleton_effect_matrices()

    def unknown_regularization_terms(self) -> dict[str, Tensor]:
        return self.base_encoder.unknown_regularization_terms()

    def _combination_correction(
        self, activity: Tensor, drug_count: Tensor | None
    ) -> Tensor:
        if self.combination_rank is None:
            return torch.zeros_like(activity)
        if drug_count is None:
            drug_count = torch.ones(activity.shape[0], device=activity.device)
        interaction = (
            (activity @ self.combination_left)
            * (activity @ self.combination_right)
        ) @ self.combination_out
        multi_drug = (drug_count > 1).to(activity).unsqueeze(-1)
        return multi_drug * self.combination_scale * torch.tanh(interaction)

    def _token_activities(self, selected: Tensor) -> Tensor:
        if self.dynamic_token_route == "static":
            return torch.zeros_like(selected)
        if self.training and self.route_dropout_probability:
            drop = torch.rand(
                selected.shape[0], 1, device=selected.device
            ) < self.route_dropout_probability
            return torch.where(drop, torch.zeros_like(selected), selected)
        return selected

    def encode_activities(
        self,
        activities: dict[str, Tensor],
        *,
        drug_count: Tensor | None = None,
        pathway_permutation: Tensor | None = None,
    ) -> ConditionEncoding:
        if set(activities) != {"selected"}:
            raise ValueError("pathway-celltype activities must contain only 'selected'")
        selected = activities["selected"]
        if selected.ndim != 2 or selected.shape[1] != self.pathway_base.shape[0]:
            raise ValueError("selected pathway activity must have shape [B,P]")

        correction = self._combination_correction(selected, drug_count)
        bottleneck_activity = selected + correction
        weight = self.pathway_celltype_weight
        if pathway_permutation is not None:
            if tuple(pathway_permutation.shape) != (weight.shape[0],):
                raise ValueError("pathway_permutation must have shape [P]")
            weight = weight[pathway_permutation]
        contribution = bottleneck_activity.unsqueeze(-1) * weight.unsqueeze(0)
        score = contribution.sum(dim=1)
        pathway_semantics = self.pathway_projection(self.pathway_base)
        celltype_context = torch.einsum("bpc,pd->bcd", contribution, pathway_semantics)
        prototype_context = torch.einsum(
            "kc,bcd->bkd", self.prototype_to_celltype, celltype_context
        )
        token_selected = self._token_activities(selected)
        condition_tokens = self.base_encoder.tokens_from_effect(token_selected)
        return ConditionEncoding(
            condition_tokens=condition_tokens,
            pathway_context=prototype_context,
            activities={"selected": selected},
            diagnostics={
                "pathway_celltype_contribution": contribution,
                "celltype_effect_score": score,
                "celltype_pathway_context": celltype_context,
                "combination_correction": correction,
                "bottleneck_activity": bottleneck_activity,
            },
        )

    def _basic_auxiliary_loss_terms(
        self, encoding: ConditionEncoding, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        required = (
            "initial_celltype_proportion",
            "terminal_celltype_proportion",
            "initial_celltype_expression",
            "terminal_celltype_expression",
            "initial_celltype_expression_mask",
            "terminal_celltype_expression_mask",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"pathway-celltype auxiliary targets are missing: {missing}")

        initial = batch["initial_celltype_proportion"].to(
            encoding.pathway_context
        )
        terminal = batch["terminal_celltype_proportion"].to(
            encoding.pathway_context
        )
        initial_clr = torch.log(initial.clamp_min(1e-8))
        terminal_clr = torch.log(terminal.clamp_min(1e-8))
        target_proportion = (
            terminal_clr - terminal_clr.mean(dim=-1, keepdim=True)
            - initial_clr + initial_clr.mean(dim=-1, keepdim=True)
        )
        score = encoding.diagnostics["celltype_effect_score"]
        predicted_proportion = score - score.mean(dim=-1, keepdim=True)
        proportion_loss = F.smooth_l1_loss(predicted_proportion, target_proportion)

        initial_expression = batch["initial_celltype_expression"].to(
            encoding.pathway_context
        )
        terminal_expression = batch["terminal_celltype_expression"].to(
            encoding.pathway_context
        )
        target_expression = terminal_expression - initial_expression
        predicted_expression = self.expression_projection(
            encoding.diagnostics["celltype_pathway_context"]
        )
        valid = (
            batch["initial_celltype_expression_mask"].bool()
            & batch["terminal_celltype_expression_mask"].bool()
        ).to(predicted_expression.device)
        expression_loss = (
            F.mse_loss(predicted_expression[valid], target_expression[valid])
            if bool(valid.any())
            else predicted_expression.sum() * 0.0
        )
        return {
            "aux_proportion": proportion_loss,
            "aux_expression": expression_loss,
        }

    def auxiliary_loss_terms(
        self, encoding: ConditionEncoding, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        terms = self._basic_auxiliary_loss_terms(encoding, batch)
        contrastive = terms["aux_proportion"].new_zeros(())
        if self.shuffle_contrastive:
            activity = encoding.diagnostics["bottleneck_activity"]
            weight = self.pathway_celltype_weight[self.pathway_shuffle_permutation]
            contribution = activity.unsqueeze(-1) * weight.unsqueeze(0)
            pathway_semantics = self.pathway_projection(self.pathway_base)
            celltype_context = torch.einsum(
                "bpc,pd->bcd", contribution, pathway_semantics
            )
            shuffled = ConditionEncoding(
                condition_tokens=encoding.condition_tokens,
                pathway_context=torch.einsum(
                    "kc,bcd->bkd", self.prototype_to_celltype, celltype_context
                ),
                activities=encoding.activities,
                diagnostics={
                    "celltype_effect_score": contribution.sum(dim=1),
                    "celltype_pathway_context": celltype_context,
                },
            )
            shuffled_terms = self._basic_auxiliary_loss_terms(shuffled, batch)
            correct_loss = terms["aux_proportion"] + 0.1 * terms["aux_expression"]
            shuffled_loss = (
                shuffled_terms["aux_proportion"]
                + 0.1 * shuffled_terms["aux_expression"]
            )
            margin = self.shuffle_margin_fraction * correct_loss.detach()
            contrastive = F.relu(correct_loss - shuffled_loss + margin)
        terms["pathway_shuffle_contrastive"] = contrastive
        return terms

    def forward(self, drug_ids: Tensor, drug_mask: Tensor) -> ConditionEncoding:
        base = self.base_encoder(drug_ids, drug_mask)
        encoding = self.encode_activities(
            base.activities,
            drug_count=drug_mask.sum(dim=1),
        )
        encoding.diagnostics.update(base.diagnostics)
        return encoding
