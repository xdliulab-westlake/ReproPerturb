"""Grouped hard vector quantizer with straight-through gradients.

Drop-in replacement for SoftVQQuantizer when every cell must map to exactly
one codeword per group (a true discrete state), instead of a probability-
weighted mixture of codewords.

Math per group g (L2 mode, the default and the one used for Hypoblast):
    z_g   = normalize(z_e segment)
    E_g   = normalize(codebook_g)
    k*    = argmax_k  z_g . E_g[k]          (cosine nearest neighbour)
    e     = E_g[k*]
    loss  = ||e - sg(z_g)||^2  +  beta * ||z_g - sg(e)||^2
    z_q   = z_g + sg(e - z_g)               (straight-through)
so the decoder only ever sees unit-norm codewords; a concatenated 4-group
latent always has norm 2.0.

Design notes
------------
* The codebook is one Parameter shaped [G, M, D] named `embedding`, exactly
  like SoftVQ, so the Stage-2 prototype export (`quantizer.embedding`) and the
  diagnostics keep working unchanged.
* Dead-code restart keeps all M codes alive: usage is tracked as an EMA of
  all-reduced assignment counts, and codes whose usage falls below
  `dead_code_threshold / M` are re-seeded from the current batch's encoder
  outputs.  The restart runs at the START of forward, before the codebook
  enters the autograd graph, so modifying parameter data can never corrupt
  saved tensors.  The codebook is then broadcast from rank 0, so DDP ranks
  stay identical.
* Usage tracking and restarts only happen on gradient-enabled training
  forwards.  model/train.py also calls `encode()` under no_grad in train
  mode after every optimizer step; that call must not double-count usage.
* An optional entropy regularizer (the SoftVQ/UniTok formulation) pushes the
  batch to spread over codes; set entropy_loss_ratio=0 to disable.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stage1.components.base import BaseQuantizer
from stage1.components.quantizers.soft_vq_quantizer import compute_entropy_loss
from stage1.registry import register_quantizer


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


@register_quantizer("hard_vq")
class HardVQQuantizer(BaseQuantizer):
    latent_format = "sequence"

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        num_codebooks: int = 1,
        beta: float = 0.25,
        l2_norm: bool = True,
        entropy_loss_ratio: float = 0.0,
        entropy_temperature: float = 0.01,
        dead_code_threshold: float = 0.03,
        dead_code_interval: int = 200,
        usage_decay: float = 0.99,
    ):
        super().__init__()
        for name, value in (
            ("n_embeddings", n_embeddings),
            ("embedding_dim", embedding_dim),
            ("num_codebooks", num_codebooks),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"beta must be finite and >= 0, got {beta}")
        if not math.isfinite(entropy_loss_ratio) or entropy_loss_ratio < 0:
            raise ValueError("entropy_loss_ratio must be finite and >= 0")
        if not 0.0 <= dead_code_threshold < 1.0:
            raise ValueError("dead_code_threshold must be in [0, 1)")
        if not 0.0 < usage_decay < 1.0:
            raise ValueError("usage_decay must be in (0, 1)")

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.num_codebooks = num_codebooks
        self.beta = float(beta)
        self.l2_norm = bool(l2_norm)
        self.entropy_loss_ratio = float(entropy_loss_ratio)
        self.entropy_temperature = float(entropy_temperature)
        self.dead_code_threshold = float(dead_code_threshold)
        self.dead_code_interval = int(dead_code_interval)
        self.usage_decay = float(usage_decay)

        embedding = torch.randn(num_codebooks, n_embeddings, embedding_dim)
        if self.l2_norm:
            embedding = F.normalize(embedding, dim=-1)
        else:
            embedding = embedding / math.sqrt(embedding_dim)
        self.embedding = nn.Parameter(embedding)

        self.register_buffer(
            "usage", torch.full((num_codebooks, n_embeddings), 1.0 / n_embeddings)
        )
        self.register_buffer("tracked_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("restarted_codes", torch.zeros((), dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            f"groups={self.num_codebooks}, codes={self.n_embeddings}, "
            f"dim={self.embedding_dim}, beta={self.beta}, l2_norm={self.l2_norm}, "
            f"entropy_ratio={self.entropy_loss_ratio}"
        )

    # ------------------------------------------------------------------ utils

    def _codebook(self) -> Tensor:
        return F.normalize(self.embedding, dim=-1) if self.l2_norm else self.embedding

    def _segments(self, z_e: Tensor) -> tuple[Tensor, int]:
        batch, seq, dim = z_e.shape
        if dim != self.embedding_dim:
            raise ValueError(f"hard_vq expected D={self.embedding_dim}, got D={dim}")
        if seq % self.num_codebooks != 0:
            raise ValueError(
                f"sequence length {seq} is not divisible by num_codebooks {self.num_codebooks}"
            )
        segment = seq // self.num_codebooks
        z = z_e.view(batch, self.num_codebooks, segment, dim)
        return (F.normalize(z, dim=-1) if self.l2_norm else z), segment

    def _scores(self, z: Tensor, codebook: Tensor) -> Tensor:
        """z [B,G,S,D], codebook [G,M,D] -> similarity [B,G,S,M] (higher = closer)."""
        if self.l2_norm:
            return torch.einsum("bgsd,gmd->bgsm", z, codebook)
        distance = (
            z.pow(2).sum(-1, keepdim=True)
            - 2 * torch.einsum("bgsd,gmd->bgsm", z, codebook)
            + codebook.pow(2).sum(-1)[None, :, None, :]
        )
        return -distance

    @torch.no_grad()
    def _maybe_restart_dead_codes(self, z: Tensor) -> None:
        if self.dead_code_threshold <= 0 or self.dead_code_interval <= 0:
            return
        step = int(self.tracked_steps.item())
        if step == 0 or step % self.dead_code_interval != 0:
            return
        dead = self.usage < (self.dead_code_threshold / self.n_embeddings)
        pool = z.detach().permute(1, 0, 2, 3).reshape(self.num_codebooks, -1, self.embedding_dim)
        replaced = 0
        for group in range(self.num_codebooks):
            dead_idx = torch.nonzero(dead[group], as_tuple=False).flatten()
            if dead_idx.numel() == 0:
                continue
            pick = torch.randint(0, pool.shape[1], (dead_idx.numel(),), device=pool.device)
            seeds = pool[group, pick]
            seeds = seeds + 0.01 * seeds.std() * torch.randn_like(seeds)
            if self.l2_norm:
                seeds = F.normalize(seeds, dim=-1)
            self.embedding.data[group, dead_idx] = seeds.to(self.embedding.dtype)
            self.usage[group, dead_idx] = 1.0 / self.n_embeddings
            replaced += int(dead_idx.numel())
        # Every rank reaches this point on the same step, so the collective is
        # always matched; rank 0's random seeds win.
        if _dist_ready():
            dist.broadcast(self.embedding.data, src=0)
            dist.broadcast(self.usage, src=0)
            count = torch.tensor([replaced], device=self.usage.device, dtype=torch.long)
            dist.broadcast(count, src=0)
            replaced = int(count.item())
        self.restarted_codes += replaced

    @torch.no_grad()
    def _track_usage(self, indices: Tensor) -> None:
        counts = torch.zeros(
            self.num_codebooks, self.n_embeddings, device=indices.device, dtype=torch.float32
        )
        flat = indices.permute(1, 0, 2).reshape(self.num_codebooks, -1)
        counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
        if _dist_ready():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        fraction = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.usage.mul_(self.usage_decay).add_(fraction, alpha=1.0 - self.usage_decay)
        self.tracked_steps += 1

    # ---------------------------------------------------------------- forward

    def forward(self, z_e: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        batch, seq, dim = z_e.shape
        z, segment = self._segments(z_e)
        tracking = self.training and torch.is_grad_enabled()
        if tracking:
            self._maybe_restart_dead_codes(z)

        codebook = self._codebook()
        scores = self._scores(z, codebook)
        indices = scores.argmax(dim=-1)  # [B, G, S]

        group_index = torch.arange(self.num_codebooks, device=z.device).view(1, -1, 1)
        selected = codebook[group_index, indices]  # [B, G, S, D]

        codebook_loss = F.mse_loss(selected, z.detach())
        commitment_loss = F.mse_loss(z, selected.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q = z + (selected - z).detach()
        z_q = z_q.reshape(batch, seq, dim)

        entropy = z_e.new_tensor(0.0)
        if self.entropy_loss_ratio > 0:
            logits = torch.einsum("bgsd,gmd->bgsm", z, codebook.detach())
            entropy = self.entropy_loss_ratio * compute_entropy_loss(
                logits.reshape(-1, self.n_embeddings), temperature=self.entropy_temperature
            )

        if tracking:
            self._track_usage(indices)

        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(
                z.reshape(-1, dim), selected.reshape(-1, dim), dim=-1
            ).mean()
            active = (self.usage > (self.dead_code_threshold / self.n_embeddings)).float().mean()

        losses = {
            "vq": vq_loss,
            "codebook": codebook_loss,
            "commitment": commitment_loss,
            "entropy": entropy,
            "embedding": vq_loss + entropy,
            "codebook_usage": active,
            "zq_z_cos": zq_z_cos,
        }
        return z_q, losses, indices.contiguous()

    def lookup(self, indices: Tensor) -> Tensor:
        """(B, G, S) -> (B, G*S, D), identical to the vectors forward emits."""
        batch, groups, segment = indices.shape
        if groups != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} groups, got {groups}")
        codebook = self._codebook()
        group_index = torch.arange(groups, device=indices.device).view(1, -1, 1)
        return codebook[group_index, indices].reshape(batch, groups * segment, self.embedding_dim)

    def broadcast_codebooks(self, src: int = 0) -> None:
        if _dist_ready():
            dist.broadcast(self.embedding.data, src=src)
