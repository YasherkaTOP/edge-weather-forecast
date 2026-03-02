"""TeacherModel2 — training-time модель с встроенным GNN."""
from __future__ import annotations

import torch
import torch.nn as nn

from .gnn import StationGNN
from .tcn import EncoderDecoderTCN_static2


class TeacherModel(nn.Module):
    """
    Учитель: строит z через GNN из графа станций, затем прогоняет
    статический EncoderDecoder TCN.
    """

    def __init__(
        self,
        node_in_dim: int,
        d_hist: int,
        d_fut_cov: int,
        d_static: int,
        gnn_emb_dim: int = 32,
        gnn_hidden: int = 64,
        enc_channels: int = 128,
        dec_channels: int = 128,
        n_enc_layers: int = 9,
        n_dec_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        p_drop_static: float = 0.3,

        gnn_kwargs: dict | None = None,

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
        gnn_kwargs = gnn_kwargs or {}
        self.gnn = StationGNN(
            node_in_dim=node_in_dim,
            gnn_hidden=gnn_hidden,
            emb_dim=gnn_emb_dim,
            dropout=dropout,
            **gnn_kwargs,
        )
        self.tcn = EncoderDecoderTCN_static2(
            d_hist=d_hist,
            d_fut_cov=d_fut_cov,
            d_static=d_static,
            d_gnn=gnn_emb_dim,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            p_drop_static=p_drop_static,
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
        x_raw_static: torch.Tensor,
        station_id: torch.Tensor,
        node_features_all: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)

        z_all = self.gnn(node_features_all, edge_index, edge_weight=edge_weight)
        z = z_all[station_id]
        return self.tcn(x_hist, x_fut_cov, x_raw_static, z)
