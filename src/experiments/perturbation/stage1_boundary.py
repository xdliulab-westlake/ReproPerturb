"""Frozen Stage-1 post-quantizer boundary reused by Stage-3 outputs."""

from __future__ import annotations

import importlib.util
import importlib
from pathlib import Path

import torch
from torch import Tensor, nn


class FrozenStage1Boundary(nn.Module):
    """Map pooled `[B,G,D]` SoftVQ states into fixed Stage-1 outputs."""

    def __init__(self, post_quant_proj: nn.Module, decoder: nn.Module):
        super().__init__()
        self.post_quant_proj = post_quant_proj
        self.decoder = decoder
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # This module is an inference-only boundary; keep BatchNorm/Dropout in
        # evaluation mode even when the surrounding Stage-3 model trains.
        super().train(False)
        return self

    def latent(self, pooled_tokens: Tensor) -> Tensor:
        projected = self.post_quant_proj(pooled_tokens)
        return projected[:, 0, :]

    def forward(self, pooled_tokens: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.latent(pooled_tokens)
        return latent, self.decoder(latent)

    def decode_latent(self, latent: Tensor) -> Tensor:
        """Decode a post-quantization latent without changing frozen weights."""
        return self.decoder(latent)

    def decode_product_code(self, product_code: Tensor) -> Tensor:
        """Decode flattened grouped code vectors shaped ``[B, G*Dq]``."""
        groups = getattr(self.post_quant_proj, "groups", None)
        grouped = product_code.reshape(product_code.shape[0], int(groups), -1)
        return self.decode_latent(self.latent(grouped))


def _load_state(path: str | Path) -> dict[str, Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def load_frozen_stage1_boundary(
    config_path: str | Path, checkpoint_path: str | Path, device: torch.device,
) -> FrozenStage1Boundary:
    config_path = Path(config_path)
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("stage1_boundary_config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load Stage-1 config {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(str(config_path))
    config = dict(module.CONFIG)
    state = _load_state(checkpoint_path)
    decoder_weights = [
        value for key, value in state.items()
        if key.startswith("decoder.net.") and getattr(value, "ndim", 0) == 2
    ]
    if not decoder_weights:
        raise KeyError("cannot infer Stage-1 decoder output dimension")
    config["decoder"] = {
        **config["decoder"],
        "out_dim": int(max(value.shape[0] for value in decoder_weights)),
    }
    config["encoder"] = {**config["encoder"], "use_precomputed": True}
    from model.registry import build_vqvae

    stage1 = build_vqvae(config).to(device)
    # Older Stage-1 checkpoints do not contain the optional cell-prototype
    # codebook introduced later.  The frozen boundary only needs the encoder
    # projection and decoder weights, so tolerate unrelated missing keys while
    # preserving all available learned parameters.
    missing, unexpected = stage1.load_state_dict(state, strict=False)
    required_prefixes = ("decoder.", "pre_quantization_linear.")
    missing_required = [key for key in missing if key.startswith(required_prefixes)]
    if missing_required:
        raise RuntimeError(f"Stage-1 checkpoint missing boundary weights: {missing_required}")
    return FrozenStage1Boundary(stage1.post_quant_proj, stage1.decoder).to(device)
