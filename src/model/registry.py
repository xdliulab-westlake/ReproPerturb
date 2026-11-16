from __future__ import annotations
from typing import Callable
from model.components.base import BaseEncoder, BaseDecoder, BaseQuantizer, BasePrior

ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {}
DECODER_REGISTRY: dict[str, type[BaseDecoder]] = {}
QUANTIZER_REGISTRY: dict[str, type[BaseQuantizer]] = {}
PRIOR_REGISTRY: dict[str, type[BasePrior]] = {}


def register_encoder(name: str) -> Callable:
    def decorator(cls: type[BaseEncoder]) -> type[BaseEncoder]:
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def register_decoder(name: str) -> Callable:
    def decorator(cls: type[BaseDecoder]) -> type[BaseDecoder]:
        DECODER_REGISTRY[name] = cls
        return cls
    return decorator


def register_quantizer(name: str) -> Callable:
    def decorator(cls: type[BaseQuantizer]) -> type[BaseQuantizer]:
        QUANTIZER_REGISTRY[name] = cls
        return cls
    return decorator


def register_prior(name: str) -> Callable:
    def decorator(cls: type[BasePrior]) -> type[BasePrior]:
        PRIOR_REGISTRY[name] = cls
        return cls
    return decorator


def _strip_name(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "name"}


def _quantizer_constructor_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in {"latent", "loss"}}


def build_vqvae(cfg: dict):
    import model.components.encoders.conv_encoder       # noqa: F401
    import model.components.encoders.scfoundation_encoder  # noqa: F401
    import model.components.decoders.conv_decoder       # noqa: F401
    import model.components.decoders.mlp_decoder        # noqa: F401
    import model.components.quantizers.vq_quantizer     # noqa: F401
    import model.components.quantizers.soft_vq_quantizer  # noqa: F401
    import model.components.quantizers.multi_codebook_vq_quantizer  # noqa: F401
    import model.components.quantizers.identity_quantizer  # noqa: F401
    import model.components.quantizers.hard_vq_quantizer  # noqa: F401

    from model.model import VQVAE

    encoder = ENCODER_REGISTRY[cfg["encoder"]["name"]](**_strip_name(cfg["encoder"]))
    decoder = DECODER_REGISTRY[cfg["decoder"]["name"]](**_strip_name(cfg["decoder"]))
    quantizer_cfg = cfg["quantizer"]
    latent_cfg = quantizer_cfg["latent"]
    loss_weights = quantizer_cfg["loss"]
    ctor_cfg = _quantizer_constructor_config(quantizer_cfg)
    quantizer = QUANTIZER_REGISTRY[ctor_cfg["name"]](**_strip_name(ctor_cfg))
    return VQVAE(
        encoder,
        decoder,
        quantizer,
        cfg["pre_quantization"],
        latent_cfg=latent_cfg,
        loss_weights=loss_weights,
        classifier_cfg=cfg.get("celltype_classifier"),
        contrastive_cfg=cfg.get("contrastive"),
    )


def build_prior(cfg: dict) -> BasePrior:
    import model.components.priors.pixelcnn  # noqa: F401
    return PRIOR_REGISTRY[cfg["prior"]["name"]](**_strip_name(cfg["prior"]))
