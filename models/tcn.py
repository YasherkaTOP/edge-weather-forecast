"""
EncoderDecoderTCN2             – базовый энкодер-декодер с GNN-эмбеддингом станции
EncoderDecoderTCN_static2      – расширение: добавляет статические признаки в декодер
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    GatedResidualTCNBlock,
    ResidualDWSeparableGatedTCNBlock,
    AttnPool1D,
    BiasHead,
    StationCondMultiBandSpectralBlock
)


_BLOCKS = {
    "gated":    GatedResidualTCNBlock,
    "dw_sep":   ResidualDWSeparableGatedTCNBlock,
}


def _resolve_block(spec: str | type) -> type:
    if isinstance(spec, str):
        key = spec.lower()
        if key not in _BLOCKS:
            raise ValueError(f"Неизвестный block='{spec}'. Доступно: {list(_BLOCKS)}")
        return _BLOCKS[key]
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec
    raise TypeError("Block spec должен быть строкой или подклассом nn.Module.")


def _make_stack_uniform(ch, n, Block, kwargs, kernel_size, dropout):
    return nn.Sequential(*[
        Block(ch, ch, kernel_size, 2 ** i, dropout, **kwargs)
        for i in range(n)
    ])


def _make_stack_mixed(ch, n, BlockA, kwargsA, BlockB, kwargsB, first_k, kernel_size, dropout):
    blocks = []
    for i in range(n):
        B = BlockA if i < first_k else BlockB
        kw = kwargsA if i < first_k else kwargsB
        blocks.append(B(ch, ch, kernel_size, 2 ** i, dropout, **kw))
    return nn.Sequential(*blocks)


class EncoderDecoderTCN2(nn.Module):
    """
    Encoder-Decoder TCN с GNN-эмбеддингом станции.

    Вход:
        x_hist:     [B, L, d_hist]
        x_fut_cov:  [B, H, d_fut_cov]
        z_station:  [B, d_gnn]
    Выход: [B, H]
    """

    def __init__(
        self,
        d_hist: int,
        d_fut_cov: int,
        d_gnn: int,
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

        use_spectral: bool = False,
        spectral_hidden: int = 64,
        T_in_hist: int = 672,
    ):
        super().__init__()

        EncBlock = _resolve_block(enc_block)
        DecFirst = _resolve_block(dec_block_first)
        DecRest = _resolve_block(dec_block_rest)

        bk_enc = block_kwargs_enc or {}
        bk_df = block_kwargs_dec_first or {}
        bk_dr = block_kwargs_dec_rest or {}

        self.use_attn_pool = use_attn_pool
        self.use_unet_skip = use_unet_skip
        self.use_bias_head = use_bias_head
        self.bias_scale = bias_scale
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.use_spectral = use_spectral
        self.spectral_hidden = spectral_hidden
        self.T_in_hist = T_in_hist

        # ---- Encoder ----
        self.enc_in = nn.Conv1d(d_hist + d_gnn, enc_channels, 1)
        self.encoder = _make_stack_uniform(enc_channels, n_enc_layers, EncBlock, bk_enc, kernel_size, dropout)

        self.spectral = (
            StationCondMultiBandSpectralBlock(
                channels=enc_channels,
                z_dim=d_gnn,
                T_in=T_in_hist,
                dropout=dropout,
                hidden=spectral_hidden,
            )
            if use_spectral else None
        )

        self.pool = AttnPool1D(enc_channels, attn_pool_hidden, dropout) if use_attn_pool else None

        if use_unet_skip:
            self.enc2dec = nn.Conv1d(enc_channels, dec_channels, 1)
            skip_ch = dec_channels
        else:
            self.enc2dec = None
            skip_ch = 0

        # ---- Decoder ----
        dec_in_dim = d_fut_cov + d_gnn + enc_channels + skip_ch
        self.dec_in_dim = dec_in_dim
        self.dec_in = nn.Conv1d(dec_in_dim, dec_channels, 1)

        use_mixed = dec_first_k > 0 and (DecRest is not DecFirst or bk_df != bk_dr)
        if use_mixed:
            self.decoder = _make_stack_mixed(
                dec_channels, n_dec_layers,
                DecFirst, bk_df, DecRest, bk_dr,
                dec_first_k, kernel_size, dropout,
            )
        else:
            self.decoder = _make_stack_uniform(dec_channels, n_dec_layers, DecFirst, bk_df, kernel_size, dropout)

        self.head = nn.Conv1d(dec_channels, 1, 1)

        # ---- Bias head ----
        self.bias_head = BiasHead(d_gnn, d_fut_cov, bias_hidden, dropout, bias_clip_deg) if use_bias_head else None

    def forward(
        self,
        x_hist: torch.Tensor,
        x_fut_cov: torch.Tensor,
        z_station: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = x_hist.shape
        H = x_fut_cov.shape[1]

        # Encoder
        z_hist = z_station.unsqueeze(1).expand(B, L, -1)
        h = self.enc_in(torch.cat([x_hist, z_hist], -1).transpose(1, 2))
        h = self.encoder(h)
        if self.spectral is not None:
            if L != self.T_in_hist:
                raise ValueError(f"Spectral expects L==T_in_hist ({self.T_in_hist}), got L={L}.")
            h = self.spectral(h, z_station)

        enc_context = self.pool(h) if self.pool is not None else h[:, :, -1]

        # U-Net skip
        if self.use_unet_skip:
            h_skip_H = F.interpolate(self.enc2dec(h), size=H, mode="linear", align_corners=False)
        else:
            h_skip_H = None

        # Decoder
        z_fut = z_station.unsqueeze(1).expand(B, H, -1)
        ctx_seq = enc_context.unsqueeze(1).expand(B, H, -1)

        parts = [x_fut_cov, z_fut, ctx_seq]
        if h_skip_H is not None:
            parts.append(h_skip_H.transpose(1, 2))

        d = self.dec_in(torch.cat(parts, -1).transpose(1, 2))
        d = self.decoder(d)
        yhat = self.head(d).squeeze(1)

        if self.bias_head is not None:
            yhat = yhat + self.bias_scale * self.bias_head(z_station, x_fut_cov)

        return yhat


class EncoderDecoderTCN_static2(EncoderDecoderTCN2):
    """
    Расширение EncoderDecoderTCN2 со статическими признаками станции в декодере.
    """

    def __init__(
        self,
        d_hist: int,
        d_fut_cov: int,
        d_static: int,
        d_gnn: int,
        enc_channels: int = 128,
        dec_channels: int = 128,
        n_enc_layers: int = 9,
        n_dec_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        p_drop_static: float = 0.3,

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

        use_bias_head: bool = True,
        bias_hidden: int = 64,
        bias_scale: float = 1.0,
        bias_clip_deg: float = 3.0,

        use_spectral: bool = False,
        spectral_hidden: int = 64,
        T_in_hist: int = 672,
    ):
        super().__init__(
            d_hist=d_hist,
            d_fut_cov=d_fut_cov,
            d_gnn=d_gnn,
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
            T_in_hist=T_in_hist,
        )

        self.p_drop_static = p_drop_static
        self.force_zero_static = False

        skip_ch = dec_channels if use_unet_skip else 0

        self.dec_in_dim = d_fut_cov + d_static + d_gnn + enc_channels + skip_ch
        self.dec_in = nn.Conv1d(self.dec_in_dim, dec_channels, 1)


    def forward(
        self,
        x_hist: torch.Tensor,
        x_fut_cov: torch.Tensor,
        x_raw_static: torch.Tensor,
        z_station: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = x_hist.shape
        H = x_fut_cov.shape[1]

        # Encoder
        z_hist = z_station.unsqueeze(1).expand(B, L, -1)
        h = self.enc_in(torch.cat([x_hist, z_hist], -1).transpose(1, 2))
        h = self.encoder(h)

        if self.spectral is not None:
            if L != self.T_in_hist:
                raise ValueError(f"Spectral expects L==T_in_hist ({self.T_in_hist}), got L={L}.")
            h = self.spectral(h, z_station)

        enc_context = self.pool(h) if self.pool is not None else h[:, :, -1]

        # U-Net skip
        if self.use_unet_skip:
            h_skip_H = F.interpolate(self.enc2dec(h), size=H, mode="linear", align_corners=False)
        else:
            h_skip_H = None

        # Static dropout
        if self.force_zero_static:
            x_raw_static = torch.zeros_like(x_raw_static)
        elif self.training and self.p_drop_static > 0:
            keep = (torch.rand(B, 1, device=x_raw_static.device) > self.p_drop_static).float()
            x_raw_static = x_raw_static * keep

        # Decoder
        stat_seq = x_raw_static.unsqueeze(1).expand(B, H, -1)
        z_fut    = z_station.unsqueeze(1).expand(B, H, -1)
        ctx_seq  = enc_context.unsqueeze(1).expand(B, H, -1)

        parts = [x_fut_cov, stat_seq, z_fut, ctx_seq]
        if h_skip_H is not None:
            parts.append(h_skip_H.transpose(1, 2))

        d = self.dec_in(torch.cat(parts, -1).transpose(1, 2))
        d = self.decoder(d)
        yhat = self.head(d).squeeze(1)

        if self.bias_head is not None:
            yhat = yhat + self.bias_scale * self.bias_head(z_station, x_fut_cov)

        return yhat
