import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from stage1.components.base import BaseEncoder
from stage1.registry import register_encoder


class ResidualLayer(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.res_block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int, n_res_layers: int):
        super().__init__()
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)


@register_encoder("conv")
class ConvEncoder(BaseEncoder):
    """Conv encoder adapted from MishaLaskin encoder.py. 28x28 -> 7x7."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_residual_layers: int,
        res_hidden_dim: int = 32,
    ):
        super().__init__()
        self._output_dim = hidden_dim
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_dim, hidden_dim, res_hidden_dim, num_residual_layers),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_stack(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim
