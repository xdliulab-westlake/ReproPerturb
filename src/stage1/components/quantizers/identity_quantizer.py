"""Pass-through 'quantizer' that turns the Stage-1 VQ-VAE into a plain AE.

This exists so that `quantizer.name = "identity"` in a Stage-1 config skips
vector quantization entirely while every other part of the pipeline
(`VQVAE`, `train_vqvae_scrna`, the codebook-usage diagnostics, checkpointing
and the final test evaluation) keeps working unchanged.

Contract notes
--------------
* ``forward`` returns ``z_e`` itself, so the decoder sees the continuous
  pre-quantization latent.  Every loss term is an exact zero tensor that is
  still attached to the right device/dtype, which keeps
  ``_quantizer_training_loss`` and the TensorBoard writers happy no matter
  which ``quantizer.loss`` weights a config declares.
* ``indices`` is an all-zero ``(B, 1, L)`` tensor.  With ``n_embeddings = 1``
  and ``num_codebooks = 1`` the usage statistics degrade to the honest
  "one code, always used" answer: perplexity 1.0, utilization 1.0.
* ``optional l2_norm`` mirrors SoftVQ's ``l2_norm=True``.  Turning it on
  isolates *discretization* as the only difference against the SoftVQ
  baseline; leaving it off gives a fully unconstrained continuous latent.
* ``lookup`` cannot exist for a pass-through latent: an index carries no
  information, so reconstructing ``z_q`` from it is impossible.  It raises
  instead of silently returning a wrong tensor.  Only the (unused for scRNA)
  PixelCNN-prior path calls it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stage1.components.base import BaseQuantizer
from stage1.registry import register_quantizer


@register_quantizer("identity")
class IdentityQuantizer(BaseQuantizer):
    """No-op quantizer: ``z_q = z_e`` with all-zero quantizer losses."""

    latent_format = "sequence"

    def __init__(
        self,
        embedding_dim: int,
        l2_norm: bool = False,
        n_embeddings: int = 1,
        num_codebooks: int = 1,
        **unused,
    ):
        super().__init__()
        # Swallow SoftVQ-only knobs so a config can be flipped to "identity"
        # by changing just the name, without silently pretending they matter.
        self._ignored_config_keys = sorted(unused)

        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim!r}"
            )
        if n_embeddings != 1 or num_codebooks != 1:
            raise ValueError(
                "identity quantizer has no codebook; set quantizer.n_embeddings=1 "
                "and quantizer.num_codebooks=1 (or omit them). Got "
                f"n_embeddings={n_embeddings}, num_codebooks={num_codebooks}"
            )

        self.embedding_dim = embedding_dim
        self.vocab_width = embedding_dim
        self.l2_norm = bool(l2_norm)
        # Reported to the usage diagnostics; there is exactly one trivial code.
        self.n_embeddings = 1
        self.num_codebooks = 1
        # Keeps `model.quantizer.parameters()` a valid (empty) optimizer group.
        self._placeholder = nn.ParameterList()

    def extra_repr(self) -> str:
        ignored = (
            f", ignored_config={self._ignored_config_keys}"
            if self._ignored_config_keys
            else ""
        )
        return f"embedding_dim={self.embedding_dim}, l2_norm={self.l2_norm}{ignored}"

    def forward(self, z_e: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        if z_e.dim() != 3:
            raise ValueError(
                f"identity quantizer expects (B, L, D), got {tuple(z_e.shape)}"
            )
        batch_size, seq_length, dim = z_e.shape
        if dim != self.embedding_dim:
            raise ValueError(
                f"identity quantizer expected D={self.embedding_dim}, got D={dim}"
            )

        z_q = F.normalize(z_e, p=2, dim=-1) if self.l2_norm else z_e

        indices = torch.zeros(
            batch_size,
            self.num_codebooks,
            seq_length,
            dtype=torch.long,
            device=z_e.device,
        )

        zero = z_e.new_tensor(0.0)
        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(
                z_e.reshape(-1, dim), z_q.reshape(-1, dim), dim=-1
            ).mean()
        losses = {
            "vq": zero,
            "commitment": zero,
            "codebook": zero,
            "entropy": zero,
            "embedding": zero,
            "codebook_usage": z_e.new_tensor(1.0),
            "avg_probs": z_e.new_tensor(1.0),
            "max_probs": z_e.new_tensor(1.0),
            "zq_z_cos": zq_z_cos,
        }
        return z_q, losses, indices

    @torch.no_grad()
    def assignment_probabilities(self, z_e: Tensor) -> Tensor:
        """Degenerate all-ones assignment, shaped like SoftVQ's export."""
        batch_size, seq_length, _ = z_e.shape
        return z_e.new_ones(batch_size, self.num_codebooks, seq_length, 1)

    def lookup(self, indices: Tensor) -> Tensor:
        raise NotImplementedError(
            "identity quantizer keeps a continuous latent, so discrete indices "
            "carry no information and cannot be decoded back to z_q. Use "
            "decode_from_z_e instead of decode_from_indices with quantizer.name="
            "'identity'."
        )

    def broadcast_codebooks(self, src: int = 0) -> None:
        """No codebook to synchronize across ranks."""
        return None
