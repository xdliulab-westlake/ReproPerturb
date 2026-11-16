"""Cross-attention flow over prototype CLR mass logits and presence states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stage2.model.time_embedder import TimeEmbedder


@dataclass
class PrototypeMassFlowOutput:
    mass_velocity: Tensor
    presence_logits: Tensor
    state_delta: Tensor
    hidden: Tensor


class PrototypeFlowBlock(nn.Module):
    """Self-attention, pathway cross-attention, AdaLN time conditioning, FFN."""

    def __init__(
        self,
        model_dim: int,
        ff_dim: int,
        nhead: int,
        time_dim: int,
        *,
        use_pathway_adaln: bool = False,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(model_dim, nhead, batch_first=True)
        self.cross_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(model_dim, nhead, batch_first=True)
        self.adaln_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.adaln = nn.Linear(time_dim, 2 * model_dim)
        self.ff_norm = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, model_dim)
        )
        self.pathway_injection = (
            PrototypeAdaLNZeroInjection(model_dim, ff_dim)
            if use_pathway_adaln
            else None
        )

    def forward(
        self,
        tokens: Tensor,
        condition: Tensor,
        time_embedding: Tensor,
        pathway_context: Tensor | None = None,
        *,
        disable_pathway_adaln: bool = False,
    ) -> Tensor:
        x = self.self_norm(tokens)
        tokens = tokens + self.self_attention(x, x, x, need_weights=False)[0]
        x = self.cross_norm(tokens)
        tokens = tokens + self.cross_attention(x, condition, condition, need_weights=False)[0]
        scale, shift = self.adaln(time_embedding).chunk(2, dim=-1)
        tokens = tokens + self.adaln_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        if self.pathway_injection is not None and not disable_pathway_adaln:
            if pathway_context is None:
                raise ValueError("inside-block pathway AdaLN requires pathway_context")
            tokens = self.pathway_injection(tokens, pathway_context)
        return tokens + self.ff(self.ff_norm(tokens))


class PrototypeAdaLNZeroInjection(nn.Module):
    """DiT-style prototype context injection with an identity initialization."""

    def __init__(self, model_dim: int, ff_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, 3 * model_dim))
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, model_dim)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        shift, scale, gate = self.modulation(context).chunk(3, dim=-1)
        modulated = self.norm(tokens) * (1 + scale) + shift
        return tokens + gate * self.ff(modulated)


class PrototypeMassFlow(nn.Module):
    """Keep K global prototype slots while transporting their CLR mass logits."""

    def __init__(
        self,
        prototype_codebook: Tensor,
        condition_dim: int = 768,
        model_dim: int = 256,
        time_dim: int = 1024,
        num_layers: int = 3,
        ff_dim: int = 4096,
        nhead: int = 8,
        use_pathway_prototype_adaln: bool = False,
        pathway_prototype_injection_after_block: int = 0,
        pathway_adaln_mode: str | None = None,
        pathway_adaln_blocks: tuple[int, ...] = (),
    ):
        super().__init__()
        self.register_buffer("prototype_codebook", prototype_codebook.float())
        self.prototype_dim = int(prototype_codebook.shape[-1])
        self.model_dim = int(model_dim)
        self.prototype_input = nn.Linear(self.prototype_dim, model_dim)
        self.mass_embed = nn.Sequential(nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.presence_embed = nn.Sequential(nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.condition_proj = nn.Linear(condition_dim, model_dim)
        self.time_embedder = TimeEmbedder(time_dim, 10000)
        self.time_proj = nn.Sequential(nn.Linear(time_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        if pathway_adaln_mode is not None and use_pathway_prototype_adaln:
            raise ValueError("legacy and configured pathway AdaLN cannot both be enabled")
        configured_mode = pathway_adaln_mode or "none"
        if configured_mode not in {"none", "after_blocks", "inside_blocks"}:
            raise ValueError(f"unsupported pathway_adaln_mode: {configured_mode!r}")
        configured_blocks = tuple(int(index) for index in pathway_adaln_blocks)
        if configured_mode == "none" and configured_blocks:
            raise ValueError("pathway_adaln_blocks requires an enabled pathway AdaLN mode")
        if configured_mode != "none" and not configured_blocks:
            raise ValueError("enabled pathway AdaLN requires pathway_adaln_blocks")
        if len(set(configured_blocks)) != len(configured_blocks):
            raise ValueError("pathway_adaln_blocks cannot contain duplicates")
        if any(index < 0 or index >= num_layers for index in configured_blocks):
            raise ValueError("pathway_adaln_blocks contains an out-of-range block")
        self.pathway_adaln_mode = configured_mode
        self.pathway_adaln_blocks = configured_blocks
        self.blocks = nn.ModuleList([
            PrototypeFlowBlock(
                model_dim,
                ff_dim,
                nhead,
                model_dim,
                use_pathway_adaln=(
                    configured_mode == "inside_blocks" and index in configured_blocks
                ),
            )
            for index in range(num_layers)
        ])
        self.pathway_injections = nn.ModuleDict()
        if configured_mode == "after_blocks":
            self.pathway_injections.update({
                str(index): PrototypeAdaLNZeroInjection(model_dim, ff_dim)
                for index in configured_blocks
            })
        self.use_pathway_prototype_adaln = bool(use_pathway_prototype_adaln)
        self.pathway_prototype_injection_after_block = int(pathway_prototype_injection_after_block)
        if self.use_pathway_prototype_adaln:
            if not 0 <= self.pathway_prototype_injection_after_block < num_layers - 1:
                raise ValueError("pathway prototype injection must be between Flow blocks")
            self.pathway_injection = PrototypeAdaLNZeroInjection(model_dim, ff_dim)
        else:
            self.pathway_injection = None
        self.mass_head = nn.Linear(model_dim, 1)
        self.presence_head = nn.Linear(model_dim, 1)
        self.state_head = nn.Linear(model_dim, self.prototype_dim)

    def token_state(self, mass_logit: Tensor, source_presence: Tensor, time: Tensor) -> Tensor:
        bsz = mass_logit.shape[0]
        prototype = self.prototype_input(self.prototype_codebook).unsqueeze(0).expand(bsz, -1, -1)
        time_token = self.time_proj(self.time_embedder(time)).unsqueeze(1)
        return (
            prototype
            + self.mass_embed(mass_logit.unsqueeze(-1))
            + self.presence_embed(source_presence.to(mass_logit).unsqueeze(-1))
            + time_token
        )

    def forward(
        self,
        mass_logit: Tensor,
        source_presence: Tensor,
        condition_tokens: Tensor,
        time: Tensor,
        pathway_context: Tensor | None = None,
        *,
        disable_pathway_adaln: bool = False,
    ) -> PrototypeMassFlowOutput:
        tokens = self.token_state(mass_logit, source_presence, time)
        condition = self.condition_proj(condition_tokens)
        time_embedding = self.time_proj(self.time_embedder(time))
        pathway_adaln_enabled = (
            self.pathway_injection is not None or self.pathway_adaln_mode != "none"
        )
        if pathway_adaln_enabled:
            expected = (len(tokens), tokens.shape[1], self.model_dim)
            if pathway_context is None or tuple(pathway_context.shape) != expected:
                raise ValueError(
                    f"pathway_context must have shape {expected} when AdaLN injection is enabled"
                )
        for index, block in enumerate(self.blocks):
            if self.pathway_adaln_mode == "inside_blocks":
                tokens = block(
                    tokens,
                    condition,
                    time_embedding,
                    pathway_context,
                    disable_pathway_adaln=disable_pathway_adaln,
                )
            else:
                tokens = block(tokens, condition, time_embedding)
            if (
                not disable_pathway_adaln
                and self.pathway_injection is not None
                and index == self.pathway_prototype_injection_after_block
            ):
                tokens = self.pathway_injection(tokens, pathway_context)
            key = str(index)
            if not disable_pathway_adaln and key in self.pathway_injections:
                tokens = self.pathway_injections[key](tokens, pathway_context)
        return PrototypeMassFlowOutput(
            mass_velocity=self.mass_head(tokens).squeeze(-1),
            presence_logits=self.presence_head(tokens).squeeze(-1),
            state_delta=self.state_head(tokens),
            hidden=tokens,
        )

    @staticmethod
    def presence_gated_proportion(mass_logit: Tensor, presence: Tensor) -> Tensor:
        """p_k = m_k exp(l_k) / sum_j m_j exp(l_j)."""
        shifted = mass_logit - mass_logit.amax(dim=-1, keepdim=True)
        weight = presence * torch.exp(shifted)
        return weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class PrototypeMassFlowAdapter(nn.Module):
    """Conditional flow matching over CLR mass logits with terminal support loss."""

    def __init__(
        self,
        velocity_model: PrototypeMassFlow,
        prototype_to_celltype: Tensor | None = None,
        stage1_boundary: nn.Module | None = None,
        presence_weight: float = 1.0,
        proportion_weight: float = 1.0,
        expression_weight: float = 0.0,
        random_flow_weight: float = 0.25,
        source_flow_weight: float = 1.0,
        endpoint_rollout_steps: int = 20,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.register_buffer(
            "prototype_to_celltype",
            prototype_to_celltype.float() if prototype_to_celltype is not None else None,
        )
        self.stage1_boundary = stage1_boundary
        if self.stage1_boundary is not None:
            for parameter in self.stage1_boundary.parameters():
                parameter.requires_grad_(False)
            self.stage1_boundary.eval()
        self.presence_weight = float(presence_weight)
        self.proportion_weight = float(proportion_weight)
        self.expression_weight = float(expression_weight)
        self.random_flow_weight = float(random_flow_weight)
        self.source_flow_weight = float(source_flow_weight)
        self.endpoint_rollout_steps = int(endpoint_rollout_steps)

    def rollout_endpoint(
        self,
        initial_mass_logit: Tensor,
        initial_presence: Tensor,
        condition_tokens: Tensor,
        pathway_context: Tensor | None = None,
    ) -> tuple[Tensor, PrototypeMassFlowOutput]:
        mass = initial_mass_logit
        for step in range(self.endpoint_rollout_steps):
            time = torch.full(
                (len(mass),), (step + 0.5) / self.endpoint_rollout_steps,
                device=mass.device,
            )
            mass = mass + self.velocity_model(
                mass, initial_presence, condition_tokens, time, pathway_context,
            ).mass_velocity / self.endpoint_rollout_steps
        terminal = self.velocity_model(
            mass, initial_presence, condition_tokens,
            torch.ones(len(mass), device=mass.device), pathway_context,
        )
        return mass, terminal

    def loss(
        self,
        initial_mass_logit: Tensor,
        terminal_mass_logit: Tensor,
        initial_presence: Tensor,
        terminal_presence: Tensor,
        terminal_proportion: Tensor,
        terminal_celltype_proportion: Tensor,
        condition_tokens: Tensor,
        pathway_context: Tensor | None = None,
        time: Tensor | None = None,
        terminal_expression: Tensor | None = None,
    ) -> Tensor:
        return self.loss_terms(
            initial_mass_logit, terminal_mass_logit, initial_presence,
            terminal_presence, terminal_proportion, terminal_celltype_proportion,
            condition_tokens, pathway_context, time, terminal_expression,
        )["total"]

    def loss_terms(
        self,
        initial_mass_logit: Tensor,
        terminal_mass_logit: Tensor,
        initial_presence: Tensor,
        terminal_presence: Tensor,
        terminal_proportion: Tensor,
        terminal_celltype_proportion: Tensor,
        condition_tokens: Tensor,
        pathway_context: Tensor | None = None,
        time: Tensor | None = None,
        terminal_expression: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return independently logged Stage-2 supervision terms."""
        batch_size = initial_mass_logit.shape[0]
        if time is None:
            time = torch.rand(batch_size, device=initial_mass_logit.device)
        t = time.unsqueeze(-1).to(initial_mass_logit)
        mass_t = (1 - t) * initial_mass_logit + t * terminal_mass_logit
        output = self.velocity_model(
            mass_t, initial_presence, condition_tokens, time, pathway_context,
        )
        random_flow_loss = torch.nn.functional.mse_loss(
            output.mass_velocity, terminal_mass_logit - initial_mass_logit
        )
        # Compute auxiliary terminal supervision from the true inference
        # source state, never from l_t which contains interpolated target
        # information during flow matching.
        source_time = torch.zeros(batch_size, device=initial_mass_logit.device)
        source_output = self.velocity_model(
            initial_mass_logit, initial_presence, condition_tokens, source_time, pathway_context,
        )
        source_flow_loss = torch.nn.functional.mse_loss(
            source_output.mass_velocity, terminal_mass_logit - initial_mass_logit,
        )
        flow_loss = (
            self.random_flow_weight * random_flow_loss
            + self.source_flow_weight * source_flow_loss
        )
        predicted_terminal_mass, terminal_output = self.rollout_endpoint(
            initial_mass_logit, initial_presence, condition_tokens, pathway_context,
        )
        presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            terminal_output.presence_logits, terminal_presence.to(terminal_output.presence_logits)
        )
        # This differentiable rollout starts from l_0, matching inference and
        # preventing terminal information in l_t from making the endpoint
        # proportion loss artificially easy.
        predicted = self.velocity_model.presence_gated_proportion(
            predicted_terminal_mass,
            torch.sigmoid(terminal_output.presence_logits),
        )
        # Supervise the biological cell-type composition after the frozen
        # prototype-to-celltype mapping.  Prototype proportions remain
        # available for diagnostics, while this loss targets the requested
        # cell-type output directly.
        predicted_celltype = (
            predicted if self.prototype_to_celltype is None
            else predicted @ self.prototype_to_celltype.to(predicted)
        )
        target = terminal_celltype_proportion.to(predicted_celltype)
        midpoint = 0.5 * (predicted_celltype + target)
        js_loss = 0.5 * (
            torch.sum(predicted_celltype * torch.log(predicted_celltype.clamp_min(1e-8) / midpoint.clamp_min(1e-8)), dim=-1).mean()
            + torch.sum(target * torch.log(target.clamp_min(1e-8) / midpoint.clamp_min(1e-8)), dim=-1).mean()
        )
        expression_loss = initial_mass_logit.new_tensor(0.0)
        expression_prediction = None
        if terminal_expression is not None and self.expression_weight:
            expression_prediction = self.decode_population_expression(
                predicted, terminal_output.state_delta,
            )
            target_expression = terminal_expression.to(expression_prediction)
            expression_loss = torch.nn.functional.mse_loss(
                expression_prediction, target_expression,
            )
        return {
            "flow": flow_loss,
            "flow_random": random_flow_loss,
            "flow_start": source_flow_loss,
            "presence": presence_loss,
            "proportion": js_loss,
            "expression": expression_loss,
            "total": (
                flow_loss
                + self.presence_weight * presence_loss
                + self.proportion_weight * js_loss
                + self.expression_weight * expression_loss
            ),
        }

    def decode_population_expression(
        self, prototype_proportion: Tensor, state_delta: Tensor,
    ) -> Tensor:
        """Decode the predicted prototype states into a pseudo-bulk profile."""
        codebook = self.velocity_model.prototype_codebook
        states = codebook.unsqueeze(0) + state_delta
        batch_size, num_prototypes, width = states.shape
        decoded = self.stage1_boundary.decode_product_code(
            states.reshape(batch_size * num_prototypes, width)
        ).reshape(batch_size, num_prototypes, -1)
        return torch.sum(prototype_proportion.unsqueeze(-1) * decoded, dim=1)

    def decode_celltype_expression(
        self,
        prototype_proportion: Tensor,
        state_delta: Tensor,
        prototype_to_celltype: Tensor | None = None,
    ) -> Tensor:
        """Decode one pseudo-bulk expression profile for every cell type.

        Prototype states are weighted by ``p_hat[k] * R[k, c]`` and
        renormalized independently for each cell type.  The returned tensor
        has shape ``[B, C, expression_dim]``.
        """
        mapping = prototype_to_celltype
        if mapping is None:
            mapping = self.prototype_to_celltype
        codebook = self.velocity_model.prototype_codebook
        states = codebook.unsqueeze(0) + state_delta
        batch_size, num_prototypes, width = states.shape
        decoded = self.stage1_boundary.decode_product_code(
            states.reshape(batch_size * num_prototypes, width)
        ).reshape(batch_size, num_prototypes, -1)
        mapping = mapping.to(prototype_proportion)
        weights = prototype_proportion.unsqueeze(-1) * mapping.unsqueeze(0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bkc,bkg->bcg", weights, decoded)
