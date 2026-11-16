"""Sign-constrained, dose-free pathway condition encoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from stage2.model.condition_types import ConditionEncoding


def canonicalize_sign_table(rows, drugs, pathways) -> Tensor:
    drug_to_i = {str(name): i for i, name in enumerate(drugs)}
    pathway_to_i = {str(name): i for i, name in enumerate(pathways)}
    result = torch.zeros(len(drugs), len(pathways), dtype=torch.float32)
    for row in rows:
        drug = str(row["drug"])
        pathway = str(row["pathway"])
        if drug not in drug_to_i or pathway not in pathway_to_i:
            raise ValueError("drug/pathway row is not present in the declared vocabulary")
        sign = float(row["sign"])
        if sign not in (-1.0, 1.0):
            raise ValueError(f"sign must be -1 or +1, got {sign}")
        result[drug_to_i[drug], pathway_to_i[pathway]] = sign
    return result


def pool_pathway_gene_embeddings(pathways, gene_embeddings):
    pooled = {}
    for pathway, genes in pathways.items():
        vectors = [gene_embeddings[g] for g in genes if g in gene_embeddings]
        if not vectors:
            raise ValueError(f"no embedding available for pathway {pathway}")
        pooled[pathway] = torch.stack(vectors).mean(dim=0)
    return pooled


class SignedInteractionCondition(nn.Module):
    """Permutation-invariant combination encoder with hard sign masking."""

    def __init__(
        self,
        pathway_base: Tensor,
        sign_matrix: Tensor,
        pathway_specific_magnitude: bool = False,
        interaction_hidden_dim: int = 32,
        interaction_scale: float = 0.5,
        use_drug_embedding: bool = False,
        constrain_sign: bool = True,
        learn_unknown_pathway_effects: bool = False,
        known_effect_initialization: float = 1.0,
        unknown_effect_mode: str = "none",
        unknown_effect_rank: int = 4,
        pair_interaction_rank: int | None = None,
        triple_interaction_rank: int | None = None,
        higher_order_scale: float = 1.25,
        token_sizes: Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("pathway_base", pathway_base.float())
        self.register_buffer("sign_matrix", sign_matrix.float())
        self.register_buffer(
            "token_sizes",
            torch.ones(sign_matrix.shape[0]) if token_sizes is None else token_sizes.float(),
        )
        self.pathway_specific_magnitude = bool(pathway_specific_magnitude)
        self.interaction_scale = float(interaction_scale)
        self.use_drug_embedding = bool(use_drug_embedding)
        self.constrain_sign = bool(constrain_sign)
        self.learn_unknown_pathway_effects = bool(learn_unknown_pathway_effects)
        self.known_effect_initialization = float(known_effect_initialization)
        self.unknown_effect_mode = str(unknown_effect_mode)
        self.unknown_effect_rank = int(unknown_effect_rank)
        self.pair_interaction_rank = pair_interaction_rank
        self.triple_interaction_rank = triple_interaction_rank
        self.higher_order_scale = float(higher_order_scale)
        if self.known_effect_initialization <= 0:
            raise ValueError("known_effect_initialization must be positive")
        if self.unknown_effect_mode not in {"none", "lowrank"}:
            raise ValueError(
                "unknown_effect_mode must be 'none' or 'lowrank', got "
                f"{self.unknown_effect_mode!r}"
            )
        if self.unknown_effect_mode == "lowrank" and self.unknown_effect_rank < 1:
            raise ValueError("unknown_effect_rank must be positive for lowrank unknown effects")
        if pair_interaction_rank is not None and pair_interaction_rank < 1:
            raise ValueError("pair_interaction_rank must be positive when enabled")
        if triple_interaction_rank is not None and triple_interaction_rank < 1:
            raise ValueError("triple_interaction_rank must be positive when enabled")
        if self.higher_order_scale <= 0:
            raise ValueError("higher_order_scale must be positive")
        # Initialize softplus(raw_magnitude) to the configured value. With pathway-specific
        # magnitudes this starts exactly at A[d, p] while allowing the model to
        # learn different strengths for different drug/pathway edges.
        magnitude_shape = (
            (sign_matrix.shape[0], sign_matrix.shape[1])
            if self.pathway_specific_magnitude else (sign_matrix.shape[0],)
        )
        if self.constrain_sign:
            initial_magnitude = torch.full(
                magnitude_shape,
                torch.log(torch.expm1(torch.tensor(self.known_effect_initialization))).item(),
            )
        elif self.pathway_specific_magnitude:
            # For an unbiased direction check, initialize every permitted
            # drug-module edge at zero.  The sign table is used only as a
            # sparsity mask in this mode, never as an initial polarity.
            initial_magnitude = torch.zeros_like(self.sign_matrix)
        else:
            initial_magnitude = torch.ones(magnitude_shape)
        self.raw_magnitude = nn.Parameter(initial_magnitude)
        if self.learn_unknown_pathway_effects:
            self.unknown_effect = nn.Parameter(torch.zeros_like(self.sign_matrix))
        if self.unknown_effect_mode == "lowrank":
            # Zero pathway factors make the full unlisted-edge matrix exactly
            # neutral at initialization, preserving the hard-sign baseline.
            self.unknown_drug_factors = nn.Parameter(
                torch.randn(sign_matrix.shape[0], self.unknown_effect_rank) * 0.02
            )
            self.unknown_pathway_factors = nn.Parameter(
                torch.zeros(sign_matrix.shape[1], self.unknown_effect_rank)
            )
            self.unknown_pathway_gate = nn.Parameter(torch.ones(sign_matrix.shape[1]))
        self.effect_direction = nn.Parameter(torch.randn_like(pathway_base) * 0.02)
        self.layer_norm = nn.LayerNorm(pathway_base.shape[1])
        if self.use_drug_embedding:
            self.drug_embedding = nn.Embedding(sign_matrix.shape[0], interaction_hidden_dim)
        interaction_input_dim = pathway_base.shape[0] + (
            interaction_hidden_dim if self.use_drug_embedding else 0
        )
        self.interaction = nn.Sequential(
            nn.Linear(interaction_input_dim, interaction_hidden_dim),
            nn.SiLU(),
            nn.Linear(interaction_hidden_dim, pathway_base.shape[0]),
        )
        # Start from the signed additive prior; the interaction residual is
        # learned only when combination data provides evidence for it.
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)
        num_pathways = int(pathway_base.shape[0])
        if pair_interaction_rank is not None:
            rank = int(pair_interaction_rank)
            self.pair_left = nn.Parameter(torch.empty(num_pathways, rank))
            self.pair_right = nn.Parameter(torch.empty(num_pathways, rank))
            self.pair_out = nn.Parameter(torch.zeros(rank, num_pathways))
            nn.init.xavier_uniform_(self.pair_left)
            nn.init.xavier_uniform_(self.pair_right)
        if triple_interaction_rank is not None:
            rank = int(triple_interaction_rank)
            self.triple_left = nn.Parameter(torch.empty(num_pathways, rank))
            self.triple_middle = nn.Parameter(torch.empty(num_pathways, rank))
            self.triple_right = nn.Parameter(torch.empty(num_pathways, rank))
            self.triple_out = nn.Parameter(torch.zeros(rank, num_pathways))
            nn.init.xavier_uniform_(self.triple_left)
            nn.init.xavier_uniform_(self.triple_middle)
            nn.init.xavier_uniform_(self.triple_right)

    def higher_order_residual(self, base_effect: Tensor, active: Tensor) -> Tensor:
        """Low-rank pair/triple correction, gated by the actual drug count.

        The output projections start at zero, so training begins at the signed
        additive prior. A scale above tanh(1) permits a learned combination to
        reverse a singleton effect without changing the pharmacological sign
        table itself.
        """
        residual = torch.zeros_like(base_effect)
        if self.pair_interaction_rank is not None:
            pair = (
                (base_effect @ self.pair_left)
                * (base_effect @ self.pair_right)
            ) @ self.pair_out
            residual = residual + (active > 1).to(pair) * pair
        if self.triple_interaction_rank is not None:
            triple = (
                (base_effect @ self.triple_left)
                * (base_effect @ self.triple_middle)
                * (base_effect @ self.triple_right)
            ) @ self.triple_out
            residual = residual + (active > 2).to(triple) * triple
        return torch.tanh(residual) * self.higher_order_scale

    @property
    def condition_token_dim(self) -> int:
        return int(self.pathway_base.shape[-1])

    @property
    def activity_keys(self) -> tuple[str, ...]:
        return ("selected",)

    def tokens_from_effect(self, effect: Tensor) -> Tensor:
        direction = F.normalize(self.effect_direction, dim=-1)
        tokens = self.pathway_base.unsqueeze(0) + effect.unsqueeze(-1) * direction.unsqueeze(0)
        return self.layer_norm(tokens)

    def encode_activities(self, activities: dict[str, Tensor]) -> ConditionEncoding:
        if set(activities) != {"selected"}:
            raise ValueError("legacy pathway activities must contain only 'selected'")
        selected = activities["selected"]
        if selected.ndim != 2 or selected.shape[1] != self.pathway_base.shape[0]:
            raise ValueError("selected pathway activity must have shape [B,P]")
        return ConditionEncoding(
            condition_tokens=self.tokens_from_effect(selected),
            pathway_context=None,
            activities={"selected": selected},
            diagnostics={},
        )

    def unknown_effect_matrix(self) -> Tensor:
        """Return the unlisted-edge effect matrix before drug-set pooling."""
        unknown_mask = self.sign_matrix == 0
        if self.unknown_effect_mode == "lowrank":
            score = self.unknown_drug_factors @ self.unknown_pathway_factors.T
            score = score / math.sqrt(self.unknown_effect_rank)
            effect = torch.tanh(score * self.unknown_pathway_gate.unsqueeze(0))
            return torch.where(unknown_mask, effect, torch.zeros_like(effect))
        if self.learn_unknown_pathway_effects and self.constrain_sign:
            return torch.where(unknown_mask, self.unknown_effect, torch.zeros_like(self.unknown_effect))
        return self.pathway_base.new_zeros(self.sign_matrix.shape)

    def unknown_regularization_terms(self) -> dict[str, Tensor]:
        """Return sparse unknown-edge penalties without affecting legacy modes."""
        zero = self.pathway_base.new_zeros(())
        if self.unknown_effect_mode != "lowrank":
            return {"unknown_effect": zero, "unknown_gate": zero}
        unknown_mask = self.sign_matrix == 0
        unknown_effect = self.unknown_effect_matrix()
        effect_l1 = unknown_effect.abs().masked_select(unknown_mask).mean()
        gate_l1 = self.unknown_pathway_gate.abs().mean()
        return {"unknown_effect": effect_l1, "unknown_gate": gate_l1}

    def singleton_effect_matrices(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return known, unknown, and selected per-drug pathway edge effects."""
        signs = self.sign_matrix
        magnitude = F.softplus(self.raw_magnitude) if self.constrain_sign else self.raw_magnitude
        if magnitude.ndim == 1:
            magnitude = magnitude.unsqueeze(-1)
        known = signs * magnitude if self.constrain_sign else signs.abs() * magnitude
        unknown = self.unknown_effect_matrix()
        selected = torch.where(signs == 0, unknown, known)
        return known, unknown, selected

    def forward(self, drug_ids: Tensor, drug_mask: Tensor):
        if drug_ids.numel() == 0:
            batch_size = drug_ids.shape[0]
            zero_effect = self.pathway_base.new_zeros(batch_size, self.pathway_base.shape[0])
            individual = self.pathway_base.new_zeros(
                batch_size, 0, self.pathway_base.shape[0]
            )
            return ConditionEncoding(
                condition_tokens=self.tokens_from_effect(zero_effect),
                pathway_context=None,
                activities={"selected": zero_effect},
                diagnostics={"individual": individual},
            )
        signs = self.sign_matrix[drug_ids]
        _, _, edge_effect = self.singleton_effect_matrices()
        individual = edge_effect[drug_ids] * drug_mask.unsqueeze(-1).to(edge_effect)
        # a_p(S) = tanh(sum_{d in S} A[d, p] * rho_d).  Saturation provides
        # the intended dose-free sub-additivity prior for drug combinations.
        base_effect = torch.tanh(individual.sum(dim=1))
        # A masked mean over drug embeddings gives a permutation-invariant set
        # summary.  The residual is gated off for singleton drugs and hard-
        # masked on pathways with no supplied sign prior.
        # BPE combo tokens count as their number of constituent drugs, so pair
        # and triple gates retain their original biological meaning after
        # compression into a single input token.
        active = (
            self.token_sizes[drug_ids] * drug_mask.to(self.pathway_base)
        ).sum(dim=1, keepdim=True)
        pooled_effect = individual.sum(dim=1) / active.clamp_min(1.0)
        if self.use_drug_embedding:
            embedding = self.drug_embedding(drug_ids)
            embedding = embedding * drug_mask.unsqueeze(-1).to(embedding)
            pooled_embedding = embedding.sum(dim=1) / active.clamp_min(1.0)
            interaction_input = torch.cat((pooled_embedding, pooled_effect), dim=-1)
        else:
            interaction_input = pooled_effect
        residual = torch.tanh(self.interaction(interaction_input)) * self.interaction_scale
        known_pathways = (
            torch.ones_like(base_effect, dtype=torch.bool)
            if self.learn_unknown_pathway_effects
            else (self.sign_matrix[drug_ids] * drug_mask.unsqueeze(-1)).abs().sum(dim=1) > 0
        )
        multi_drug_gate = (active > 1).to(residual)
        combo_effect = base_effect + (
            multi_drug_gate * known_pathways.to(residual) * residual
        )
        combo_effect = combo_effect + (
            known_pathways.to(base_effect)
            * self.higher_order_residual(base_effect, active)
        )
        return ConditionEncoding(
            condition_tokens=self.tokens_from_effect(combo_effect),
            pathway_context=None,
            activities={"selected": combo_effect},
            diagnostics={
                "individual": individual,
                "base_effect": base_effect,
                "higher_order_residual": self.higher_order_residual(
                    base_effect, active
                ),
            },
        )
