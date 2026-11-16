from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        scRNA path: x (B, n_features) -> z_e (B, D)
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Encoder output channels/features before pre-quantization projection."""


class BaseDecoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, z_q: Tensor) -> Tensor:
        """z_q -> x_hat, matching the dataset-specific input contract."""


class BaseQuantizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, z_e: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """
        Returns:
          z_q      — quantized latent, same shape as z_e
          losses   — {"embedding": scalar, "commitment": scalar, "codebook": scalar}
          indices  — quantizer-specific discrete code indices.
                      scRNA path uses (B, G, L), where G=1 for single-codebook VQ.
        """

    @abstractmethod
    def lookup(self, indices: Tensor) -> Tensor:
        """
        scRNA path: indices (B, G, L) -> z_q (B, L, D)
        """


class BasePrior(nn.Module, ABC):
    @abstractmethod
    def forward(self, indices: Tensor, labels: Tensor | None = None) -> Tensor:
        """
        indices (B, H, W), labels (B,) -> logits (B, K, H, W)
        """

    @abstractmethod
    def sample(
        self,
        n: int,
        device: torch.device,
        labels: Tensor | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Autoregressive sampling -> indices (n, H, W)."""
