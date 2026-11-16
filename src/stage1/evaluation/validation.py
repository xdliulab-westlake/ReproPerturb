from __future__ import annotations
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from stage1.model import VQVAE
from stage1.components.base import BasePrior
from stage1.evaluation.losses import reconstruction_loss, prior_nll_loss
from stage1.evaluation.metrics import compute_perplexity, codebook_utilization, active_codes


def validate_vqvae(
    model: VQVAE,
    loader: DataLoader,
    x_train_var: float,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation pass. Returns averaged metrics dict."""
    model.eval()
    recon_losses, perplexities, all_indices = [], [], []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat, losses = model(x)
            recon = reconstruction_loss(x_hat, x, x_train_var)
            indices = model.encode(x)

            # one-hot encodings for perplexity
            B, H, W = indices.shape
            n_e = model.quantizer.n_embeddings
            min_encodings = torch.zeros(B * H * W, n_e, device=device)
            min_encodings.scatter_(1, indices.view(-1, 1), 1)

            recon_losses.append(recon.item())
            perplexities.append(compute_perplexity(min_encodings).item())
            all_indices.append(indices.cpu())

    all_indices_flat = torch.cat([i.view(-1) for i in all_indices])
    n_e = model.quantizer.n_embeddings
    return {
        "val_recon_loss": float(np.mean(recon_losses)),
        "val_perplexity": float(np.mean(perplexities)),
        "val_codebook_utilization": codebook_utilization(all_indices_flat, n_e),
        "val_active_codes": active_codes(all_indices_flat, n_e),
    }


def validate_prior(
    prior: BasePrior,
    model: VQVAE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run one prior validation pass."""
    prior.eval()
    model.eval()
    nll_losses = []

    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            indices = model.encode(x)
            logits = prior(indices, labels)
            nll_losses.append(prior_nll_loss(logits, indices).item())

    return {"val_prior_nll": float(np.mean(nll_losses))}
