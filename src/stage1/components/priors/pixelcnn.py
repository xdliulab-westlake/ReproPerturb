import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from stage1.components.base import BasePrior
from stage1.registry import register_prior


class GatedActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type: str, dim: int, kernel: int, residual: bool, n_classes: int):
        super().__init__()
        assert kernel % 2 == 1
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(n_classes, 2 * dim)

        self.vert_stack = nn.Conv2d(dim, dim * 2, (kernel // 2 + 1, kernel), 1, (kernel // 2, kernel // 2))
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.horiz_stack = nn.Conv2d(dim, dim * 2, (1, kernel // 2 + 1), 1, (0, kernel // 2))
        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        self.gate = GatedActivation()

    def make_causal(self) -> None:
        self.vert_stack.weight.data[:, :, -1].zero_()
        self.horiz_stack.weight.data[:, :, :, -1].zero_()

    def forward(self, x_v: Tensor, x_h: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        if self.mask_type == "A":
            self.make_causal()

        h_emb = self.class_cond_embedding(h)                 # (B, 2*dim)
        h_vert = self.vert_stack(x_v)[:, :, : x_v.size(-2), :]
        out_v = self.gate(h_vert + h_emb[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)[:, :, :, : x_h.size(-1)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz + h_emb[:, :, None, None])
        out_h = self.horiz_resid(out) + x_h if self.residual else self.horiz_resid(out)
        return out_v, out_h


@register_prior("pixelcnn")
class GatedPixelCNNPrior(BasePrior):
    """
    Class-conditional GatedPixelCNN prior.
    Adapted from MishaLaskin pixelcnn/models.py.
    """

    def __init__(self, n_embeddings: int, hidden_dim: int, n_layers: int, n_classes: int = 10):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding = nn.Embedding(n_embeddings, hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = "A" if i == 0 else "B"
            kernel = 7 if i == 0 else 3
            residual = i > 0
            self.layers.append(GatedMaskedConv2d(mask_type, hidden_dim, kernel, residual, n_classes))
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, n_embeddings, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try:
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                except ValueError:
                    pass

    def forward(self, indices: Tensor, labels: Tensor | None = None) -> Tensor:
        """indices (B, H, W), labels (B,) -> logits (B, K, H, W)"""
        B, H, W = indices.shape
        x = self.embedding(indices.view(-1)).view(B, H, W, -1).permute(0, 3, 1, 2)
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=indices.device)
        x_v, x_h = x, x
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h, labels)
        return self.output_conv(x_h)                         # (B, K, H, W)

    def sample(
        self,
        n: int,
        device: torch.device,
        labels: Tensor | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Autoregressive sampling -> indices (n, H, W)."""
        H, W = shape if shape is not None else (7, 7)
        if labels is None:
            labels = torch.zeros(n, dtype=torch.long, device=device)
        x = torch.zeros(n, H, W, dtype=torch.long, device=device)
        for i in range(H):
            for j in range(W):
                logits = self.forward(x, labels)             # (n, K, H, W)
                probs = F.softmax(logits[:, :, i, j], dim=-1)
                x[:, i, j] = probs.multinomial(1).squeeze(1)
        return x
