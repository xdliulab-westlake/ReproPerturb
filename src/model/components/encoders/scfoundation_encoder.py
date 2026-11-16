# model/components/encoders/scfoundation_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from model.components.base import BaseEncoder
from model.registry import register_encoder
from scfoundation.load import load_model_frommmf, gatherData

_SCF_TOKEN_DIM = 768    # dimension of each scFoundation gene token
_SCF_POOL_TOKENS = 4    # last, second-last, max-pool, mean-pool -> 3072


class FlashSelfAttentionEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with explicit Flash/SDPA self-attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    def _attention_mask(self, padding_mask: Tensor | None, dtype: torch.dtype) -> Tensor | None:
        if padding_mask is None:
            return None
        padding_mask = padding_mask.to(dtype=torch.bool)
        if not torch.any(padding_mask):
            return None
        mask = torch.zeros(padding_mask.shape, device=padding_mask.device, dtype=dtype)
        mask = mask.masked_fill(padding_mask, float("-inf"))
        return mask[:, None, None, :]

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None) -> Tensor:
        dropout_p = self.dropout.p if self.training else 0.0
        if q.is_cuda:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if src_mask is not None:
            raise NotImplementedError("FlashSelfAttentionEncoderLayer does not support src_mask")
        if is_causal:
            raise NotImplementedError("FlashSelfAttentionEncoderLayer does not support causal masks")
        h = self.norm1(src)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)
        attn_mask = self._attention_mask(src_key_padding_mask, q.dtype)
        attn = self._sdpa(q, k, v, attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(src.shape)
        src = src + self.dropout(self.out_proj(attn))

        h = self.norm2(src)
        ff = self.linear2(self.dropout(self.activation(self.linear1(h))))
        return src + self.dropout(ff)


@register_encoder("scfoundation")
class ScFoundationEncoder(BaseEncoder):
    """
    Frozen scFoundation backbone + trainable head.

    Input:  x (B, n_genes+2) raw expression  OR  precomputed scFoundation output
            (B, 3072) pooled embedding or (B, S, 768) gene-token sequence
    Output: z_e (B, mlp_dims[-1])

    Head variants (controlled by config):
      use_attention=False (default):
        Linear stack:  3072 → mlp_dims

      use_attention=True:
        Apply one Flash/SDPA self-attention encoder layer on the gene-token
        sequence before scFoundation-style 4-way pooling, then
        Linear(4*768 → mlp_dims[0]) + MLP tail.
        n_heads must divide 768 evenly (e.g. 8, 12, 16).

    Config keys (under "encoder"):
        use_attention : bool  (default False)
        n_heads       : int   (default 8, only used when use_attention=True)
    """

    def __init__(
        self,
        ckpt_path: str,
        gene_index_path: str,
        mlp_dims: list[int],
        use_precomputed: bool = False,
        use_attention: bool = False,
        n_heads: int = 8,
    ):
        super().__init__()
        self.gene_index_path = gene_index_path
        self.use_precomputed = use_precomputed
        self.use_attention = use_attention
        self._gatherData = gatherData

        if not use_precomputed:
            sc_model, sc_config = load_model_frommmf(ckpt_path, key="cell")
            self.sc_model = sc_model
            self.pad_token_id = sc_config["pad_token_id"]
            for p in self.sc_model.parameters():
                p.requires_grad = False
            self.sc_model.eval()
        else:
            self.sc_model = None
            self.pad_token_id = None

        self._build_head(mlp_dims, n_heads)
        self._output_dim = mlp_dims[-1]

    # ------------------------------------------------------------------
    # head construction
    # ------------------------------------------------------------------

    def _build_head(self, mlp_dims: list[int], n_heads: int) -> None:
        if self.use_attention:
            self.attn = FlashSelfAttentionEncoderLayer(
                d_model=_SCF_TOKEN_DIM,
                nhead=n_heads,
                dim_feedforward=_SCF_TOKEN_DIM * 2,
            )
            # flatten pooled 4×768 → first MLP width
            self.attn_proj = nn.Linear(_SCF_POOL_TOKENS * _SCF_TOKEN_DIM, mlp_dims[0])
            mlp_in_dim = mlp_dims[0]
            tail_dims = mlp_dims[1:]
        else:
            mlp_in_dim = _SCF_POOL_TOKENS * _SCF_TOKEN_DIM   # 3072
            tail_dims = mlp_dims

        layers = []
        in_dim = mlp_in_dim
        for h in tail_dims[:-1]:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, tail_dims[-1]))
        self.mlp_head = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # properties / train-mode guard
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if self.sc_model is not None:
            self.sc_model.eval()
        return self

    # ------------------------------------------------------------------
    # forward helpers
    # ------------------------------------------------------------------

    def _infer_padding_mask(self, geneemb: Tensor) -> Tensor:
        """Infer padding from zero-filled token rows saved by run_token_embedding.py."""
        return geneemb.abs().sum(dim=-1).eq(0)

    def _pool_gene_tokens(
        self,
        geneemb: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Pool scFoundation gene-token outputs into the legacy 3072 cell embedding."""
        if geneemb.dim() != 3 or geneemb.shape[-1] != _SCF_TOKEN_DIM:
            raise ValueError(
                "scFoundation gene-token input must have shape (B, S, 768), "
                f"got {tuple(geneemb.shape)}"
            )

        if padding_mask is None:
            padding_mask = torch.zeros(
                geneemb.shape[:2], dtype=torch.bool, device=geneemb.device
            )
        else:
            padding_mask = padding_mask.to(device=geneemb.device, dtype=torch.bool)
            if padding_mask.shape != geneemb.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape (B, S), "
                    f"got {tuple(padding_mask.shape)} for tokens {tuple(geneemb.shape)}"
                )

        valid = ~padding_mask
        lengths = valid.sum(dim=1)
        if torch.any(lengths < 3):
            raise ValueError(
                "scFoundation gene-token input must contain at least one gene token "
                "plus T/S tokens"
            )

        batch_idx = torch.arange(geneemb.shape[0], device=geneemb.device)
        s_idx = lengths - 1
        t_idx = lengths - 2
        geneemb1 = geneemb[batch_idx, s_idx, :]
        geneemb2 = geneemb[batch_idx, t_idx, :]

        token_idx = torch.arange(geneemb.shape[1], device=geneemb.device).unsqueeze(0)
        gene_mask = valid & (token_idx < t_idx.unsqueeze(1))
        masked_geneemb = geneemb.masked_fill(~gene_mask.unsqueeze(-1), float("-inf"))
        geneemb3, _ = torch.max(masked_geneemb, dim=1)
        geneemb4 = (
            geneemb.masked_fill(~gene_mask.unsqueeze(-1), 0.0).sum(dim=1)
            / gene_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        )
        return torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)  # (B, 3072)

    def _apply_gene_token_attention(
        self,
        geneemb: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if self.use_attention:
            geneemb = self.attn(geneemb, src_key_padding_mask=padding_mask)
        return geneemb

    def _embed(self, x: Tensor) -> Tensor:
        """
        Raw expression (B, n_genes+2) → (B, 3072) via scFoundation pool_type='all'.
        With use_attention=True, the trainable attention head runs before pooling.
        """
        B = x.shape[0]
        n_total = x.shape[1]
        value_labels = x > 0
        x_gathered, x_padding = self._gatherData(x, value_labels, self.pad_token_id)
        data_gene_ids = torch.arange(n_total, device=x.device).repeat(B, 1)
        position_gene_ids, _ = self._gatherData(data_gene_ids, value_labels, self.pad_token_id)

        tok = self.sc_model.token_emb(
            x_gathered.unsqueeze(2).float(), output_weight=0
        )
        tok = tok + self.sc_model.pos_emb(position_gene_ids)
        geneemb = self.sc_model.encoder(tok, x_padding)        # (B, S, 768)
        geneemb = self._apply_gene_token_attention(geneemb, x_padding)
        return self._pool_gene_tokens(geneemb, x_padding)

    def _apply_head(self, cell_emb: Tensor) -> Tensor:
        """(B, 3072) → (B, output_dim) through attention head or plain MLP."""
        if self.use_attention:
            h = self.attn_proj(cell_emb)                  # (B, mlp_dims[0])
        else:
            h = cell_emb
        return self.mlp_head(h)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_precomputed:
            if x.dim() == 3:
                padding_mask = self._infer_padding_mask(x)
                geneemb = self._apply_gene_token_attention(x, padding_mask)
                cell_emb = self._pool_gene_tokens(geneemb, padding_mask)
                return self._apply_head(cell_emb)
            return self._apply_head(x)          # x is already (B, 3072)
        else:
            cell_emb = self._embed(x)       # (B, 3072)
            return self._apply_head(cell_emb)
