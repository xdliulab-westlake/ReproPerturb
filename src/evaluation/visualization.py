import torch
import os
from torch import Tensor
from torchvision.utils import save_image


def save_reconstruction_grid(
    x: Tensor,
    x_hat: Tensor,
    path: str,
    nrow: int = 8,
) -> None:
    """Save side-by-side real vs reconstructed grid. x, x_hat: (B, C, H, W) in [0,1]."""
    comparison = torch.cat([x[:nrow], x_hat[:nrow].clamp(0, 1)])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(comparison, path, nrow=nrow)


def save_generation_grid(
    x_hat: Tensor,
    path: str,
    nrow: int = 10,
) -> None:
    """Save generated samples grid. x_hat: (B, C, H, W) in [0,1]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(x_hat.clamp(0, 1), path, nrow=nrow)


def save_codebook_histogram(
    indices: Tensor,
    n_embeddings: int,
    path: str,
) -> None:
    """Save per-code usage bar chart as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    counts = torch.bincount(indices.view(-1), minlength=n_embeddings).float()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(range(n_embeddings), counts.numpy())
    ax.set_xlabel("Code index")
    ax.set_ylabel("Usage count")
    ax.set_title("Codebook usage histogram")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
