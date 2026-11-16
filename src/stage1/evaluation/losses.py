import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(x_hat: Tensor, x: Tensor, x_train_var: float) -> Tensor:
    """MSE(x_hat, x) / x_train_var. Matches MishaLaskin main.py."""
    return torch.mean((x_hat - x) ** 2) / x_train_var


def embedding_loss(z_e: Tensor, z_q: Tensor, beta: float) -> dict[str, Tensor]:
    """
    VQ loss terms matching MishaLaskin quantizer.py exactly.

    commitment_loss = mean((z_q.detach() - z_e)^2)   — gradient to encoder
    codebook_loss   = beta * mean((z_q - z_e.detach())^2) — gradient to codebook
    embedding_loss  = commitment_loss + codebook_loss
    """
    commitment = torch.mean((z_q.detach() - z_e) ** 2)
    codebook = beta * torch.mean((z_q - z_e.detach()) ** 2)
    return {
        "commitment": commitment,
        "codebook": codebook,
        "embedding": commitment + codebook,
    }


def prior_nll_loss(logits: Tensor, indices: Tensor) -> Tensor:
    """
    Cross-entropy loss for autoregressive prior.
    logits: (B, K, H, W)
    indices: (B, H, W) — long tensor
    """
    return F.cross_entropy(logits, indices)
