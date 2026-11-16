# model/components/quantizers/multi_codebook_vq_quantizer.py
"""
Multi-Codebook VQ Quantizer for scRNA — UniTok style (ByteDance, 2025).

Borrows from /Users/lzz/Documents/GitHub/UniTok/models/quant.py:
  - NormalizedEmbedding + cosine-similarity matching (more stable than L2)
  - Entropy loss: per_sample_entropy - codebook_entropy (prevents collapse)
  - VectorQuantizerM: split D-dim cell token into G groups, each with its own
    sub-codebook of size K_sub and width D/G.

This quantizer uses the single-cell token contract directly:
    z_e     : (B, L, D)
    z_q     : (B, L, D)
    indices : (B, G, L)

Config keys (under "quantizer"):
    name                    : "unitok"

Fixed UniTok-style settings for this project:
    vocab_size     : 256 total codes
    vocab_width    : 128 quantizer width
    num_codebooks  : 8 groups, so each group has 32 codes
    beta           : 0.25
    entropy_temp   : 0.01
"""
import torch
import torch.distributed as tdist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.components.base import BaseQuantizer
from model.registry import register_quantizer


# ── borrowed from UniTok/models/quant.py ─────────────────────────────────────

def _get_entropy_loss(latent_embed: Tensor, codebook_embed: Tensor, inv_tau: float) -> Tensor:
    """
    per_sample_entropy − codebook_entropy.
    Minimising this pushes assignments to be sharp (low per-sample entropy)
    while keeping the overall code usage spread out (high codebook entropy).
    """
    # pairwise L2² distances (N, K)
    d = (
        latent_embed.square().sum(1, keepdim=True)
        + codebook_embed.square().sum(1)
        - 2 * latent_embed @ codebook_embed.T
    )
    logits = -d.float() * inv_tau
    prob      = logits.softmax(dim=-1)
    log_prob  = logits.log_softmax(dim=-1)
    per_sample_entropy = (-prob * log_prob).sum(dim=-1).mean()
    avg_prob  = prob.mean(dim=0)
    codebook_entropy   = -(avg_prob * (avg_prob + 1e-7).log()).sum()
    return per_sample_entropy - codebook_entropy


class _NormalizedEmbedding(nn.Embedding):
    """Codebook whose weights are always L2-normalised at lookup time."""

    def forward(self, idx: Tensor) -> Tensor:
        return F.embedding(
            idx, F.normalize(self.weight, dim=1),
            self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )

    def get_norm_weight(self) -> Tensor:
        return F.normalize(self.weight, dim=1)


class _SingleGroupVQ(nn.Module):
    """One VQ group — mirrors UniTok's VectorQuantizer."""

    def __init__(self, k: int, d: int, beta: float, use_entropy_loss: bool, entropy_temp: float):
        super().__init__()
        self.beta             = beta
        self.vocab_size       = k
        self.use_entropy_loss = use_entropy_loss
        self.inv_tau          = 1.0 / entropy_temp
        self.codebook         = _NormalizedEmbedding(k, d)
        self.record_times     = 0
        self.register_buffer("vocab_usage", torch.zeros(k))

    def forward(self, z: Tensor):
        """
        z : (N, d)  already L2-normalised
        returns z_q (N, d), vq_loss scalar, entropy_loss scalar, vocab_usage %
        """
        E = self.codebook.get_norm_weight()          # (K, d)
        idx = torch.argmax(z.detach() @ E.T, dim=1)  # (N,)  cosine NN

        entropy_loss = (
            _get_entropy_loss(z, E, self.inv_tau) if self.use_entropy_loss
            else z.new_tensor(0.0)
        )

        z_q = self.codebook(idx)                     # (N, d)
        vq_loss = (
            F.mse_loss(z_q.detach(), z) * self.beta
            + F.mse_loss(z_q, z.detach())
        )

        # straight-through
        z_q_st = (z_q.detach() - z.detach()) + z     # gradient flows through z

        # EMA vocab usage (same as UniTok)
        prob = idx.bincount(minlength=self.vocab_size).float()
        if self.training and tdist.is_initialized():
            tdist.all_reduce(prob, op=tdist.ReduceOp.SUM)
        prob = prob / (prob.sum() + 1e-7)
        usage_pct = (prob > 0.01 / self.vocab_size).float().mean() * 100
        if self.record_times == 0:
            self.vocab_usage.copy_(prob)
        elif self.record_times < 100:
            self.vocab_usage.mul_(0.9).add_(prob, alpha=0.1)
        else:
            self.vocab_usage.mul_(0.99).add_(prob, alpha=0.01)
        self.record_times += 1

        return z_q_st, vq_loss, entropy_loss, usage_pct, idx


# ── main quantizer ────────────────────────────────────────────────────────────

@register_quantizer("unitok")
class MultiCodebookVQQuantizer(BaseQuantizer):
    latent_format = "sequence"

    def __init__(
        self,
        vocab_size: int = 256,
        vocab_width: int = 128,
        num_codebooks: int = 8,
        beta: float = 0.25,
        use_entropy_loss: bool = True,
        entropy_temp: float = 0.01,
    ):
        super().__init__()
        if vocab_size % num_codebooks != 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be divisible by num_codebooks ({num_codebooks})"
            )
        if vocab_width % num_codebooks != 0:
            raise ValueError(
                f"vocab_width ({vocab_width}) must be divisible by num_codebooks ({num_codebooks})"
            )
        self.vocab_size = vocab_size
        self.vocab_width = vocab_width
        self.num_codebooks = num_codebooks
        self.n_groups = num_codebooks
        self.n_embeddings_per_group = vocab_size // num_codebooks
        self.embedding_dim = vocab_width
        self.sub_dim = vocab_width // num_codebooks

        self.groups = nn.ModuleList([
            _SingleGroupVQ(self.n_embeddings_per_group, self.sub_dim,
                           beta, use_entropy_loss, entropy_temp)
            for _ in range(num_codebooks)
        ])

    @property
    def n_embeddings(self) -> int:
        """K_sub (per group) — used for perplexity / utilisation logging."""
        return self.n_embeddings_per_group

    def initialize_from_celltype_means(
        self,
        celltype_vectors: Tensor,
        seed: int = 42,
        noise_scale: float = 0.01,
    ) -> None:
        """Init each sub-codebook from the corresponding slice of celltype means."""
        n_ct = celltype_vectors.shape[0]
        assert celltype_vectors.shape[1] == self.embedding_dim
        assert n_ct <= self.n_embeddings_per_group

        alphas = torch.tensor(
            [0.25, 0.5, 0.75],
            dtype=celltype_vectors.dtype,
            device=celltype_vectors.device,
        )
        chunks = celltype_vectors.chunk(self.n_groups, dim=1)  # G × (n_ct, sub_dim)

        with torch.no_grad():
            for grp, chunk in zip(self.groups, chunks):
                w = grp.codebook.weight.data.clone()
                # L2-normalise the init vectors to match NormalizedEmbedding
                w[:n_ct] = F.normalize(chunk.float(), dim=1)
                n_extra = self.n_embeddings_per_group - n_ct
                for i in range(n_extra):
                    left  = (i // len(alphas)) % n_ct
                    right = (left + 1) % n_ct
                    a     = alphas[i % len(alphas)]
                    w[n_ct + i] = F.normalize(
                        ((1 - a) * chunk[left] + a * chunk[right]).unsqueeze(0).float(), dim=1
                    ).squeeze(0)
                grp.codebook.weight.copy_(w)

    def broadcast_codebooks(self, src: int = 0) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for grp in self.groups:
            torch.distributed.broadcast(grp.codebook.weight.data, src=src)

    # ------------------------------------------------------------------

    def forward(self, z_e: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """
        z_e : (B, L, D)
        returns:
          z_q      (B, L, D)
          losses   {"embedding", "vq", "entropy"}
          indices  (B, G, L)
        """
        B, L, D = z_e.shape
        z_flat = z_e.reshape(-1, D)                  # (B*L, D)

        # UniTok splits first; each VectorQuantizer normalizes its own chunk.
        chunks = [F.normalize(chunk, dim=-1) for chunk in z_flat.chunk(self.n_groups, dim=1)]

        zq_parts, idx_parts = [], []
        total_vq = z_e.new_tensor(0.0)
        total_entropy = z_e.new_tensor(0.0)

        for chunk, grp in zip(chunks, self.groups):
            z_q_chunk, vq_loss, ent_loss, usage, idx = grp(chunk)
            zq_parts.append(z_q_chunk)
            idx_parts.append(idx)
            total_vq      += vq_loss
            total_entropy += ent_loss

        total_vq      = total_vq      / self.n_groups
        total_entropy = total_entropy / self.n_groups

        z_q = torch.cat(zq_parts, dim=1).view(B, L, D)

        losses = {
            "vq":        total_vq,
            "entropy":   total_entropy,
            "embedding": total_vq + total_entropy,  # used by train.py
            "commitment": total_vq,                 # kept for logging compat
            "codebook":   z_e.new_tensor(0.0),
        }

        indices = torch.stack(idx_parts, dim=1).view(B, self.n_groups, L)

        return z_q, losses, indices

    def lookup(self, indices: Tensor) -> Tensor:
        """
        indices : (B, G, L)
        returns  : z_q (B, L, D)
        """
        B, G, L = indices.shape
        assert G == self.n_groups
        chunks = []
        for g, grp in enumerate(self.groups):
            idx_g = indices[:, g].reshape(-1)             # (B*L,)
            zq_g = grp.codebook(idx_g)                   # (B*L, sub_dim)
            chunks.append(zq_g)
        return torch.cat(chunks, dim=1).view(B, L, self.embedding_dim)
