import torch
import torch.nn as nn
from torch import Tensor
from stage1.components.base import BaseQuantizer
from stage1.evaluation.losses import embedding_loss as compute_embedding_loss
from stage1.registry import register_quantizer


@register_quantizer("vq")
class VQQuantizer(BaseQuantizer):
    """
    Vector Quantizer for scRNA token sequences with straight-through gradients.

    Contract:
        z_e     : (B, L, D)
        z_q     : (B, L, D)
        indices : (B, 1, L)
    """
    latent_format = "sequence"

    def __init__(self, n_embeddings: int, embedding_dim: int, beta: float):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)

    def initialize_from_celltype_means(
        self,
        celltype_vectors: Tensor,
        seed: int = 42,
        noise_scale: float = 0.01,
    ) -> None:
        """Initialize codebook from per-cell-type mean vectors.

        The first len(celltype_vectors) slots are set directly to the provided
        vectors (which should already be mean-centered and in quantization space).
        Remaining slots are filled by deterministic interpolations between
        pairs of celltype_vectors, keeping the extra codes inside the observed
        cell-type manifold.

        Args:
            celltype_vectors: (n_celltypes, embedding_dim) float32 tensor,
                already projected to 64-d quantization space and mean-centered.
            seed: Reserved for backward-compatible call sites.
            noise_scale: Reserved for backward-compatible call sites.
        """
        n_ct = celltype_vectors.shape[0]
        assert celltype_vectors.shape[1] == self.embedding_dim, (
            f"celltype_vectors dim {celltype_vectors.shape[1]} != "
            f"embedding_dim {self.embedding_dim}"
        )
        assert n_ct <= self.n_embeddings, (
            f"More cell types ({n_ct}) than codebook slots ({self.n_embeddings})"
        )

        with torch.no_grad():
            new_weight = self.embedding.weight.data.clone()
            # first n_ct slots: cell-type means
            new_weight[:n_ct] = celltype_vectors

            # remaining slots: deterministic interpolations between cell types
            n_extra = self.n_embeddings - n_ct
            if n_extra > 0:
                alphas = torch.tensor(
                    [0.25, 0.5, 0.75],
                    dtype=celltype_vectors.dtype,
                    device=celltype_vectors.device,
                )
                for extra_idx in range(n_extra):
                    left_idx = (extra_idx // len(alphas)) % n_ct
                    right_idx = (left_idx + 1) % n_ct
                    alpha = alphas[extra_idx % len(alphas)]
                    new_weight[n_ct + extra_idx] = (
                        (1.0 - alpha) * celltype_vectors[left_idx]
                        + alpha * celltype_vectors[right_idx]
                    )

            self.embedding.weight.copy_(new_weight)

    def forward(self, z_e: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        B, L, D = z_e.shape
        z_flat = z_e.reshape(-1, self.embedding_dim)        # (B*L, D)

        # distances to all embeddings: (N, K)
        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        min_indices = torch.argmin(d, dim=1)                # (N,)
        z_q_flat = self.embedding(min_indices)              # (N, D)
        z_q = z_q_flat.view(B, L, D)                        # (B, L, D)

        losses = compute_embedding_loss(z_e, z_q, self.beta)

        z_q_out = z_e + (z_q - z_e).detach()

        indices = min_indices.view(B, L).unsqueeze(1)       # (B, 1, L)
        return z_q_out, losses, indices

    def lookup(self, indices: Tensor) -> Tensor:
        """indices (B, 1, L) -> z_q (B, L, D)."""
        B, G, L = indices.shape
        z_q = self.embedding(indices.view(-1))
        return z_q.view(B, L, self.embedding_dim)
