import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stage1.components.base import BaseQuantizer
from stage1.registry import register_quantizer


def compute_entropy_loss(
    affinity: Tensor,
    loss_type: str = "softmax",
    temperature: float = 0.01,
) -> Tensor:
    """Match continuous_tokenizer/modelling/quantizers/softvq.py."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")
    if affinity.shape[-1] <= 0:
        raise ValueError("affinity must contain at least one code")
    flat_affinity = affinity.reshape(-1, affinity.shape[-1]) / temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError(f"Entropy loss {loss_type} not supported")
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    return sample_entropy - avg_entropy


@register_quantizer("soft_vq")
class SoftVQQuantizer(BaseQuantizer):
    """
    Official SoftVQ-style quantizer adapted to the iPStem quantizer API.

    The internal math follows continuous_tokenizer's SoftVectorQuantizer:
      - grouped codebook tensor: (num_codebooks, n_embeddings, embedding_dim)
      - optional L2 normalization of z and codebook
      - dot-product logits with detached codebook for assignment
      - softmax(logits / tau)
      - z_q as the probability-weighted codebook average
      - entropy loss, with zero VQ and commitment losses

    iPStem keeps its existing external contract:
      forward(z_e) -> (z_q, losses_dict, indices)
      lookup(indices) -> z_q
    """
    latent_format = "sequence"

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        entropy_loss_ratio: float = 0.01,
        tau: float = 0.07,
        num_codebooks: int = 1,
        l2_norm: bool = False,
        show_usage: bool = False,
        beta: float | None = None,
        init_temperature: float | None = None,
        min_temperature: float | None = None,
        anneal_steps: int | None = None,
    ):
        super().__init__()
        del beta, init_temperature, min_temperature, anneal_steps

        for name, value in (
            ("n_embeddings", n_embeddings),
            ("embedding_dim", embedding_dim),
            ("num_codebooks", num_codebooks),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not math.isfinite(entropy_loss_ratio) or entropy_loss_ratio < 0.0:
            raise ValueError(
                "entropy_loss_ratio must be finite and >= 0, "
                f"got {entropy_loss_ratio}"
            )
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"tau must be finite and > 0, got {tau}")

        self.num_codebooks = num_codebooks
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.tau = tau

        self.embedding = nn.Parameter(
            torch.empty(num_codebooks, n_embeddings, embedding_dim)
        )
        self.embedding.data.uniform_(-1.0 / self.n_embeddings, 1.0 / self.n_embeddings)

        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)

        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(num_codebooks, 65536))

    def initialize_from_celltype_means(
        self,
        celltype_vectors: Tensor,
        seed: int = 42,
        noise_scale: float = 0.01,
    ) -> None:
        """Initialize each codebook group from cell-type means."""
        if not math.isfinite(noise_scale) or noise_scale < 0.0:
            raise ValueError(
                f"noise_scale must be finite and >= 0, got {noise_scale}"
            )
        if celltype_vectors.dim() == 2:
            celltype_vectors = celltype_vectors.unsqueeze(1).expand(
                -1, self.num_codebooks, -1
            )
        if celltype_vectors.dim() != 3:
            raise ValueError(
                "soft_vq celltype_vectors must be (N,D) or (N,G,D), "
                f"got {tuple(celltype_vectors.shape)}"
            )
        n_ct, groups, dim = celltype_vectors.shape
        if n_ct == 0:
            raise ValueError("celltype_vectors must contain at least one cell type")
        if groups != self.num_codebooks:
            raise ValueError(
                f"celltype_vectors groups {groups} != num_codebooks {self.num_codebooks}"
            )
        if dim != self.embedding_dim:
            raise ValueError(
                f"celltype_vectors dim {dim} != embedding_dim {self.embedding_dim}"
            )
        if n_ct > self.n_embeddings:
            raise ValueError(
                f"More cell types ({n_ct}) than codebook slots ({self.n_embeddings})"
            )
        if not torch.isfinite(celltype_vectors).all():
            raise ValueError("celltype_vectors contains non-finite values")

        vectors = celltype_vectors.permute(1, 0, 2).contiguous()
        with torch.no_grad():
            new_weight = self.embedding.data.clone()
            new_weight[:, :n_ct] = vectors
            n_extra = self.n_embeddings - n_ct
            if n_extra > 0:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed))
                left_idx = torch.randint(
                    n_ct,
                    (n_extra,),
                    generator=generator,
                )
                if n_ct == 1:
                    right_idx = left_idx
                else:
                    right_offset = torch.randint(
                        1,
                        n_ct,
                        (n_extra,),
                        generator=generator,
                    )
                    right_idx = (left_idx + right_offset) % n_ct
                alphas = (
                    0.05
                    + 0.90
                    * torch.rand(
                        n_extra,
                        generator=generator,
                    )
                ).to(
                    dtype=vectors.dtype,
                    device=vectors.device,
                )
                left_idx = left_idx.to(device=vectors.device)
                right_idx = right_idx.to(device=vectors.device)
                extra_vectors = (
                    (1.0 - alphas[None, :, None]) * vectors[:, left_idx, :]
                    + alphas[None, :, None] * vectors[:, right_idx, :]
                )
                if noise_scale > 0.0:
                    noise = torch.randn(
                        self.num_codebooks,
                        n_extra,
                        self.embedding_dim,
                        generator=generator,
                        dtype=vectors.dtype,
                    ).to(device=vectors.device)
                    group_scale = (
                        vectors.square()
                        .mean(dim=(1, 2))
                        .sqrt()
                        .clamp_min(1e-6)
                    )
                    extra_vectors = extra_vectors + (
                        float(noise_scale)
                        * group_scale[:, None, None]
                        * noise
                    )
                new_weight[:, n_ct:] = extra_vectors
            if self.l2_norm:
                new_weight = F.normalize(new_weight, p=2, dim=-1)
            self.embedding.copy_(new_weight)

    def forward(self, z_e: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        batch_size, seq_length, dim = z_e.shape

        segment_length = seq_length // self.num_codebooks
        z_segments = z_e.view(
            batch_size, self.num_codebooks, segment_length, self.embedding_dim
        )

        embedding = (
            F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        )
        if self.l2_norm:
            z_segments = F.normalize(z_segments, p=2, dim=-1)

        z_flat = (
            z_segments.permute(1, 0, 2, 3)
            .contiguous()
            .view(self.num_codebooks, -1, self.embedding_dim)
        )

        logits = torch.einsum("nbe,nke->nbk", z_flat, embedding.detach())
        probs = F.softmax(logits / self.tau, dim=-1)
        z_q = torch.einsum("nbk,nke->nbe", probs, embedding)
        z_q = (
            z_q.view(self.num_codebooks, batch_size, segment_length, self.embedding_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(batch_size, seq_length, self.embedding_dim)
        )

        indices_by_group = torch.argmax(probs, dim=-1)
        indices = indices_by_group.view(
            self.num_codebooks, batch_size, segment_length
        ).permute(1, 0, 2).contiguous()

        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(
                z_segments.reshape(-1, self.embedding_dim),
                z_q.reshape(-1, self.embedding_dim),
                dim=-1,
            ).mean()

        if self.show_usage and self.training:
            for group_idx in range(self.num_codebooks):
                group_indices = indices_by_group[group_idx]
                cur_len = group_indices.numel()
                history_len = self.codebook_used.shape[1]
                if cur_len >= history_len:
                    self.codebook_used[group_idx].copy_(
                        group_indices[-history_len:]
                    )
                else:
                    self.codebook_used[group_idx, :-cur_len].copy_(
                        self.codebook_used[group_idx, cur_len:].clone()
                    )
                    self.codebook_used[group_idx, -cur_len:].copy_(group_indices)
        codebook_usage = (
            z_e.new_tensor(
                [
                    torch.unique(self.codebook_used[group_idx]).numel()
                    / self.n_embeddings
                    for group_idx in range(self.num_codebooks)
                ]
            ).mean().to(device=z_e.device)
            if self.show_usage
            else z_e.new_tensor(0.0)
        )
        avg_probs = torch.mean(torch.mean(probs, dim=-1))
        max_probs = torch.mean(torch.max(probs, dim=-1)[0])

        zero = z_e.new_tensor(0.0)
        # Keep the official pooled entropy objective. Per-codebook balance is
        # reported separately by the epoch-level diagnostics in model/train.py.
        entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(
            logits.reshape(-1, self.n_embeddings)
        )
        losses = {
            "vq": zero,
            "commitment": zero,
            "codebook": zero,
            "entropy": entropy_loss,
            "embedding": entropy_loss,
            "codebook_usage": codebook_usage,
            "avg_probs": avg_probs,
            "max_probs": max_probs,
            "zq_z_cos": zq_z_cos,
        }

        return z_q, losses, indices

    @torch.no_grad()
    def assignment_probabilities(self, z_e: Tensor) -> Tensor:
        """Return the continuous SoftVQ assignments for inference exports.

        The regular training API intentionally remains
        ``forward(z_e) -> (z_q, losses, indices)``.  This helper exposes the
        same temperature-scaled probabilities without changing that contract.
        The result has shape ``[B, G, segment_length, K]``.
        """
        batch_size, seq_length, dim = z_e.shape
        segment_length = seq_length // self.num_codebooks
        z_segments = z_e.view(batch_size, self.num_codebooks, segment_length, dim)
        embedding = (
            F.normalize(self.embedding, p=2, dim=-1)
            if self.l2_norm
            else self.embedding
        )
        if self.l2_norm:
            z_segments = F.normalize(z_segments, p=2, dim=-1)
        z_flat = (
            z_segments.permute(1, 0, 2, 3)
            .contiguous()
            .view(self.num_codebooks, -1, dim)
        )
        logits = torch.einsum("nbe,nke->nbk", z_flat, embedding.detach())
        probs = F.softmax(logits / self.tau, dim=-1)
        return probs.view(self.num_codebooks, batch_size, segment_length, self.n_embeddings).permute(1, 0, 2, 3).contiguous()

    def lookup(self, indices: Tensor) -> Tensor:
        """Hard lookup for iPStem decode_from_indices: (B, G, S) -> (B, L, D)."""
        batch_size, groups, segment_length = indices.shape

        codebook = (
            F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        )
        pieces = []
        for group_idx in range(self.num_codebooks):
            group_indices = indices[:, group_idx].reshape(-1)
            group_vectors = codebook[group_idx].index_select(0, group_indices)
            pieces.append(group_vectors.view(batch_size, segment_length, self.embedding_dim))
        return torch.stack(pieces, dim=1).view(
            batch_size, self.num_codebooks * segment_length, self.embedding_dim
        )

    def broadcast_codebooks(self, src: int = 0) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.embedding.data, src=src)
