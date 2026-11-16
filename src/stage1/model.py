import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from stage1.components.base import BaseEncoder, BaseDecoder, BaseQuantizer


class GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, eps=1e-6)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)) * self.w1(x))


class PlainAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        if in_dim > out_dim:
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        else:
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer("zero_k_bias", torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        qkv = F.linear(
            input=x,
            weight=self.qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        )
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        ).unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)
        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = F.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        return self.proj(x)


class AttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, int(out_dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ReshapeProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if in_dim == out_dim:
            self.mode = "identity"
            self.groups = 1
        elif in_dim > out_dim:
            if in_dim % out_dim != 0:
                raise ValueError(f"cannot reshape feature dim {in_dim} into width {out_dim}")
            self.mode = "split"
            self.groups = in_dim // out_dim
        else:
            if out_dim % in_dim != 0:
                raise ValueError(f"cannot merge width {in_dim} into feature dim {out_dim}")
            self.mode = "merge"
            self.groups = out_dim // in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        if D != self.in_dim:
            raise ValueError(f"reshape projection expected D={self.in_dim}, got D={D}")
        if self.mode == "identity":
            return x
        if self.mode == "split":
            if L != 1:
                raise ValueError(f"reshape split expects one token, got L={L}")
            return x.reshape(B, self.groups, self.out_dim)
        if L != self.groups:
            raise ValueError(f"reshape merge expects L={self.groups}, got L={L}")
        return x.reshape(B, 1, self.out_dim)


class ContrastiveProjectionHead(nn.Module):
    """Small projection head used only by the supervised contrastive objective."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class VQVAE(nn.Module):
    """
    VQ-VAE assembly for scRNA embeddings.

    Current scRNA path: pre_quant_cfg["flat"] = True.
      pre_quantization_linear: Linear(in_features, embedding_dim)
      encoder output: (B, D)
      quantizer input: (B, 1, D)
      single-codebook indices: (B, 1, 1)
      UniTok multi-codebook indices: (B, G, 1)
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        quantizer: BaseQuantizer,
        pre_quant_cfg: dict,
        latent_cfg: dict | None = None,
        loss_weights: dict[str, float] | None = None,
        classifier_cfg: dict | None = None,
        contrastive_cfg: dict | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self._flat = pre_quant_cfg.get("flat", False)
        self._quantizer_latent_cfg = latent_cfg or {
            "format": getattr(quantizer, "latent_format", "sequence"),
            "projection": None,
        }
        self._quantizer_latent_format = self._quantizer_latent_cfg["format"]
        self._quantizer_loss_weights = loss_weights or (
            {"vq": 1.0, "entropy": 1.0}
            if self._quantizer_latent_format == "sequence"
            else {"embedding": 1.0}
        )

        if self._flat:
            self.pre_quantization_linear = nn.Linear(
                pre_quant_cfg["in_features"],
                pre_quant_cfg["embedding_dim"],
            )
            if self._quantizer_latent_format == "sequence":
                quantizer_width = getattr(
                    quantizer,
                    "vocab_width",
                    getattr(quantizer, "embedding_dim", pre_quant_cfg["embedding_dim"]),
                )
                if self._quantizer_latent_cfg.get("projection") == "attn":
                    num_heads = pre_quant_cfg["embedding_dim"] // quantizer_width
                    self.quant_proj = AttnProjection(
                        pre_quant_cfg["embedding_dim"], quantizer_width, num_heads
                    )
                    self.post_quant_proj = AttnProjection(
                        quantizer_width, pre_quant_cfg["embedding_dim"], num_heads
                    )
                elif self._quantizer_latent_cfg.get("projection") == "reshape":
                    self.quant_proj = ReshapeProjection(
                        pre_quant_cfg["embedding_dim"], quantizer_width
                    )
                    self.post_quant_proj = ReshapeProjection(
                        quantizer_width, pre_quant_cfg["embedding_dim"]
                    )
                elif self._quantizer_latent_cfg.get("projection") is None:
                    if pre_quant_cfg["embedding_dim"] != quantizer_width:
                        raise ValueError(
                            "projection=None requires pre_quantization.embedding_dim "
                            f"({pre_quant_cfg['embedding_dim']}) to match quantizer width "
                            f"({quantizer_width})"
                        )
                    self.quant_proj = nn.Identity()
                    self.post_quant_proj = nn.Identity()
                else:
                    raise ValueError(
                        f"unknown quantizer projection: {self._quantizer_latent_cfg.get('projection')}"
                    )
            else:
                raise ValueError(
                    f"flat scRNA VQVAE expects sequence quantizer, got {self._quantizer_latent_format}"
                )
        else:
            raise ValueError("VQVAE is scRNA sequence-only; set pre_quantization.flat=True")

        classifier_cfg = classifier_cfg or {}
        num_classes = int(classifier_cfg.get("num_classes", 0))
        if num_classes < 0:
            raise ValueError("celltype_classifier.num_classes must be non-negative")
        self.celltype_classifier = (
            nn.Linear(pre_quant_cfg["embedding_dim"], num_classes)
            if num_classes > 0
            else None
        )

        contrastive_cfg = contrastive_cfg or {}
        contrastive_weight = float(contrastive_cfg.get("loss_weight", 0.0))
        if contrastive_weight < 0.0:
            raise ValueError("contrastive.loss_weight must be non-negative")
        self.contrastive_source = str(contrastive_cfg.get("source", "z_e"))
        if self.contrastive_source not in {"z_e", "z_q"}:
            raise ValueError("contrastive.source must be 'z_e' or 'z_q'")
        self.contrastive_projector = None
        if contrastive_weight > 0.0:
            hidden_dim = int(contrastive_cfg.get("hidden_dim", pre_quant_cfg["embedding_dim"]))
            projection_dim = int(contrastive_cfg.get("projection_dim", 128))
            if hidden_dim <= 0 or projection_dim <= 0:
                raise ValueError("contrastive hidden_dim/projection_dim must be positive")
            self.contrastive_projector = ContrastiveProjectionHead(
                pre_quant_cfg["embedding_dim"], hidden_dim, projection_dim
            )

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        h = self.encoder(x)
        if self._flat:
            z_e = self.pre_quantization_linear(h)          # (B, D)
            z_tokens = self.quant_proj(z_e.unsqueeze(1))   # (B, 1, Dq)
            z_q_seq, vq_losses, _ = self.quantizer(z_tokens)
            z_q = self._single_token_latent(self.post_quant_proj(z_q_seq))
        else:
            raise ValueError("VQVAE is scRNA sequence-only; set pre_quantization.flat=True")
        x_hat = self.decoder(z_q)
        losses = {
            "recon": None,          # caller computes recon_loss (needs x_train_var)
            "embedding": vq_losses["embedding"],
            "commitment": vq_losses["commitment"],
            "codebook": vq_losses["codebook"],
            "vq": vq_losses.get("vq", vq_losses["embedding"]),
            "entropy": vq_losses.get("entropy", vq_losses["embedding"].new_tensor(0.0)),
            "total": None,          # caller sums recon + embedding
        }
        if self.celltype_classifier is not None:
            losses["celltype_logits"] = self.celltype_classifier(z_q)
        if self.contrastive_projector is not None:
            contrastive_input = z_e if self.contrastive_source == "z_e" else z_q
            losses["contrastive_embedding"] = self.contrastive_projector(contrastive_input)
        return x_hat, losses

    def encode(self, x: Tensor) -> Tensor:
        """x -> discrete indices. Flat scRNA path returns (B, G, 1)."""
        h = self.encoder(x)
        if self._flat:
            z_e = self.pre_quantization_linear(h)
            _, _, indices = self.quantizer(self.quant_proj(z_e.unsqueeze(1)))
        else:
            raise ValueError("VQVAE is scRNA sequence-only; set pre_quantization.flat=True")
        return indices

    def decode_from_z_e(self, z_e: Tensor) -> Tensor:
        """z_e (continuous latent) → quantize → z_q → decoder → x_hat.

        Flat scRNA path: z_e (B, D) → quant_proj → quantizer → post_quant_proj → squeeze → decoder.
        """
        if self._flat:
            z_tokens = self.quant_proj(z_e.unsqueeze(1))      # (B, 1, Dq)
            z_q_seq, _, _ = self.quantizer(z_tokens)
            z_q = self._single_token_latent(self.post_quant_proj(z_q_seq))
            return self.decoder(z_q)
        raise ValueError("decode_from_z_e only supports flat scRNA path")

    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """indices -> x_hat. Flat scRNA path: indices (B,G,1) -> x_hat (B, out_dim)."""
        z_q = self.quantizer.lookup(indices)
        if self._flat:
            z_q = self._single_token_latent(self.post_quant_proj(z_q))
        return self.decoder(z_q)

    @staticmethod
    def _single_token_latent(z: Tensor) -> Tensor:
        if z.shape[1] != 1:
            raise ValueError(f"MLP decoder path expects exactly one token, got L={z.shape[1]}")
        return z[:, 0, :]
