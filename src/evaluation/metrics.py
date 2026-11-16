import torch
from torch import Tensor


def compute_perplexity(encodings: Tensor) -> Tensor:
    """
    encodings: (N, K) one-hot or soft assignment matrix.
    Returns exp(-sum(p * log(p))), matching MishaLaskin quantizer.py.
    """
    e_mean = encodings.mean(dim=0)  # (K,)
    return torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))


def codebook_utilization(indices: Tensor, n_embeddings: int) -> float:
    """Fraction of codebook entries used at least once."""
    return active_codes(indices, n_embeddings) / n_embeddings


def active_codes(indices: Tensor, n_embeddings: int) -> int:
    """Number of distinct codes used in indices."""
    return int(indices.unique().numel())


def compute_ssim(x_hat: Tensor, x: Tensor) -> float:
    """
    Structural similarity (mean over batch). Requires torchvision >= 0.14.
    x_hat, x: (B, C, H, W) in [0, 1].
    """
    try:
        from torchvision.transforms.functional import to_pil_image
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        return ssim(x_hat.clamp(0, 1), x.clamp(0, 1)).item()
    except ImportError:
        return float("nan")
