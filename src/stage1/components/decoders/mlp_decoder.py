import torch.nn as nn
from torch import Tensor
from stage1.components.base import BaseDecoder
from stage1.registry import register_decoder


@register_decoder("mlp")
class MlpDecoder(BaseDecoder):
    """
    MLP decoder for flat VQ-VAE latents.
    z_q (B, embedding_dim) -> x_hat (B, out_dim)
    """

    def __init__(self, embedding_dim: int, hidden_dims: list[int], out_dim: int):
        super().__init__()
        layers = []
        in_dim = embedding_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_q: Tensor) -> Tensor:
        return self.net(z_q)
