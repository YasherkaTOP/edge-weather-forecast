"""StudentModel2 — inference-ready модель с предвычисленным z."""
from __future__ import annotations

import torch
import torch.nn as nn

from .tcn import EncoderDecoderTCN2


class StudentModel(nn.Module):
    """
    Студент: использует предвычисленный GNN-эмбеддинг z станции.

    Параметры конструктора напрямую пробрасываются в EncoderDecoderTCN2.
    """

    def __init__(
        self,
        d_hist: int,
        d_fut_cov: int,
        gnn_emb_dim: int = 32,
        enc_channels: int = 128,
        dec_channels: int = 128,
        n_enc_layers: int = 9,
        n_dec_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,

        enc_block: str | type = "gated",
        dec_block_first: str | type = "gated",
        dec_block_rest: str | type = "dw_sep",
        dec_first_k: int = 4,
        block_kwargs_enc: dict | None = None,
        block_kwargs_dec_first: dict | None = None,
        block_kwargs_dec_rest: dict | None = None,

        use_attn_pool: bool = True,
        attn_pool_hidden: int = 64,
        use_unet_skip: bool = True,

        use_bias_head: bool = False,
        bias_hidden: int = 64,
        bias_scale: float = 1.0,
        bias_clip_deg: float = 3.0,

        use_spectral: bool = True,
        spectral_hidden: int = 128,
    ):
        super().__init__()
        self.tcn = EncoderDecoderTCN2(
            d_hist=d_hist,
            d_fut_cov=d_fut_cov,
            d_gnn=gnn_emb_dim,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            enc_block=enc_block,
            dec_block_first=dec_block_first,
            dec_block_rest=dec_block_rest,
            dec_first_k=dec_first_k,
            block_kwargs_enc=block_kwargs_enc,
            block_kwargs_dec_first=block_kwargs_dec_first,
            block_kwargs_dec_rest=block_kwargs_dec_rest,
            use_attn_pool=use_attn_pool,
            attn_pool_hidden=attn_pool_hidden,
            use_unet_skip=use_unet_skip,
            use_bias_head=use_bias_head,
            bias_hidden=bias_hidden,
            bias_scale=bias_scale,
            bias_clip_deg=bias_clip_deg,
            use_spectral=use_spectral,
            spectral_hidden=spectral_hidden,
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        x_fut_cov: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.tcn(x_hist, x_fut_cov, z)
