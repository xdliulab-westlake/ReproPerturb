import torch.nn as nn
from torch import Tensor
from stage1.components.base import BaseDecoder
from stage1.components.encoders.conv_encoder import ResidualStack
from stage1.registry import register_decoder


@register_decoder("conv")
class ConvDecoder(BaseDecoder):
    """ConvTranspose decoder adapted from MishaLaskin decoder.py. 7x7 -> 28x28."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_residual_layers: int,
        res_hidden_dim: int = 32,
    ):
        super().__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_dim, hidden_dim, res_hidden_dim, num_residual_layers),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z_q: Tensor) -> Tensor:
        return self.inverse_conv_stack(z_q)
