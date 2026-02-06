import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class StationGNN(nn.Module):
    ''' Graph Neural Network for station embeddings '''
    def __init__(self, node_in_dim: int, gnn_hidden: int = 64, emb_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(node_in_dim, gnn_hidden)
        self.conv2 = GCNConv(gnn_hidden, emb_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.drop(h)
        z = self.conv2(h, edge_index, edge_weight=edge_weight)
        return z  # [N, emb_dim]
    
class ResidualTCNBlock(nn.Module):
    ''' Residual Temporal Convolutional Network Block '''
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)

        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        self.padding = padding

    def _chomp(self, x):
        ''' Remove padding from the end of the sequence '''
        if self.padding == 0:
            return x
        return x[:, :, :-self.padding]

    def forward(self, x):
        h = self.conv1(x)
        h = self._chomp(h)
        h = self.act(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self._chomp(h)
        h = self.act(h)
        h = self.drop(h)

        return h + self.skip(x)  # [B, C, T]
    

class SEGate(nn.Module):
    ''' Squeeze-and-Excitation Gate for 1D inputs '''
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # [B, C, 1]
            nn.Conv1d(ch, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x) # [B, C, T]


class ResidualDWSeparableGatedTCNBlock(nn.Module):
    ''' Residual Dilated Depthwise Separable Gated TCN Block '''
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1, se_reduction: int = 4):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding

        self.dw1 = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding, dilation=dilation, groups=in_ch)
        self.pw1 = nn.Conv1d(in_ch, out_ch, kernel_size=1)

        self.dw2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation, groups=out_ch)
        self.pw2 = nn.Conv1d(out_ch, out_ch, kernel_size=1)

        self.gate = SEGate(out_ch, reduction=se_reduction)

        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def _chomp(self, x):
        ''' Remove padding from the end of the sequence '''
        if self.padding == 0:
            return x
        return x[:, :, :-self.padding]

    def forward(self, x):
        h = self.dw1(x)
        h = self._chomp(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.drop(h)

        h = self.dw2(h)
        h = self._chomp(h)
        h = self.pw2(h)
        h = self.act(h)

        h = self.gate(h)
        h = self.drop(h)
        return h + self.skip(x) # [B, C, T]
    
class GatedResidualTCNBlock(nn.Module):
    '''f Causal dilated gated conv block (WaveNet style) with residual skip'''
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv = nn.Conv1d(in_ch, 2 * out_ch, kernel_size, padding=padding, dilation=dilation)
        self.drop = nn.Dropout(dropout)

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def _chomp(self, x):
        ''' Remove padding from the end of the sequence '''
        if self.padding == 0:
            return x
        return x[:, :, :-self.padding]

    def forward(self, x):
        h = self.conv(x)
        h = self._chomp(h)

        # split into filter and gate
        out_ch = h.shape[1] // 2
        h_f = h[:, :out_ch, :]
        h_g = h[:, out_ch:, :]

        h = torch.tanh(h_f) * torch.sigmoid(h_g)
        h = self.drop(h)

        return self.skip(x) + h  # [B, C_out, T]
    
class BiasHead(nn.Module):
    ''' Bias head predicting bias per station given station embedding and future covariates '''
    def __init__(self, z_dim: int, fut_dim: int, hidden: int = 64, dropout: float = 0.1, clip_deg: float = 3.0):
        super().__init__()
        self.clip_deg = clip_deg
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + fut_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # ZERO INIT
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, z_station: torch.Tensor, x_fut_cov: torch.Tensor) -> torch.Tensor:
        fut_summary = x_fut_cov.mean(dim=1)
        inp = torch.cat([z_station, fut_summary], dim=-1)
        b = self.mlp(inp)

        # CLIP
        b = self.clip_deg * torch.tanh(b / self.clip_deg)
        return b # [B, 1]
    
class EncoderDecoderTCN(nn.Module):
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
        tcn_block="gated",
        block_kwargs: dict | None = None,
        use_bias_head: bool = True,
        bias_hidden: int = 64,
        bias_dropout: float = 0.1,
        bias_scale: float = 1.0,
    ):
        '''Основной модуль Энкодер-Декодер'''
        super().__init__()

        BLOCKS = {
            "gated": GatedResidualTCNBlock,
            "dw_sep": ResidualDWSeparableGatedTCNBlock,
            "residual": ResidualTCNBlock,
        }
        block_kwargs = block_kwargs or {}

        if isinstance(tcn_block, str):
            key = tcn_block.lower()
            if key not in BLOCKS:
                raise ValueError(f"Unknown tcn_block='{tcn_block}'. Available: {list(BLOCKS.keys())}")
            Block = BLOCKS[key]
        elif isinstance(tcn_block, type) and issubclass(tcn_block, nn.Module):
            Block = tcn_block
        else:
            raise TypeError("tcn_block must be a string key or an nn.Module class")

        def make_stack(ch: int, n_layers: int) -> nn.Sequential:
            blocks = []
            for i in range(n_layers):
                dilation = 2 ** i
                blocks.append(Block(ch, ch, kernel_size, dilation, dropout, **block_kwargs))
            return nn.Sequential(*blocks)

        self.use_bias_head = use_bias_head
        self.bias_scale = bias_scale
        self.dec_channels = dec_channels
        self.enc_channels = enc_channels

        # --- Encoder ---
        self.enc_in = nn.Conv1d(d_hist + d_gnn, enc_channels, kernel_size=1)
        self.encoder = make_stack(enc_channels, n_enc_layers)

        # --- Decoder ---
        self.dec_in_dim = d_fut_cov + d_gnn + enc_channels
        self.dec_in = nn.Conv1d(self.dec_in_dim, dec_channels, kernel_size=1)
        self.decoder = make_stack(dec_channels, n_dec_layers)

        self.head = nn.Conv1d(dec_channels, 1, kernel_size=1)

        if self.use_bias_head:
            self.bias_head = BiasHead(z_dim=d_gnn, fut_dim=d_fut_cov, hidden=bias_hidden, dropout=bias_dropout)
        else:
            self.bias_head = None

    def forward(self, x_hist, x_fut_cov, z_station):
        B, L, _ = x_hist.shape
        H = x_fut_cov.shape[1]

        # --- Encoder ---
        z_hist = z_station.unsqueeze(1).expand(B, L, z_station.shape[-1])
        x_enc = torch.cat([x_hist, z_hist], dim=-1).transpose(1, 2)
        h = self.enc_in(x_enc)
        h = self.encoder(h)
        enc_context = h[:, :, -1]  # [B, C]

        # --- Decoder ---
        z_fut = z_station.unsqueeze(1).expand(B, H, z_station.shape[-1])
        enc_ctx_seq = enc_context.unsqueeze(1).expand(B, H, enc_context.shape[-1])

        x_dec = torch.cat([x_fut_cov, z_fut, enc_ctx_seq], dim=-1).transpose(1, 2)
        d = self.dec_in(x_dec)
        d = self.decoder(d)
        yhat = self.head(d).squeeze(1)  # [B, H]

        # --- Bias correction (optional) ---
        if self.bias_head is not None:
            b = self.bias_head(z_station, x_fut_cov)  # [B, 1]
            yhat = yhat + self.bias_scale * b

        return yhat
    
class EncoderDecoderTCN_static(EncoderDecoderTCN):
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
        tcn_block="gated",
        block_kwargs: dict | None = None,
        use_bias_head: bool = True,
        bias_hidden: int = 64,
        bias_dropout: float = 0.1,
        bias_scale: float = 1.0,
    ):
        '''Модуль энкодер-декодер с "сливом" статиков в декодер'''
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
            tcn_block=tcn_block,
            block_kwargs=block_kwargs,
            use_bias_head=use_bias_head,
            bias_hidden=bias_hidden,
            bias_dropout=bias_dropout,
            bias_scale=bias_scale,
        )

        self.p_drop_static = p_drop_static
        self.force_zero_static = False

        self.dec_in_dim = d_fut_cov + d_static + d_gnn + enc_channels
        self.dec_in = nn.Conv1d(self.dec_in_dim, dec_channels, kernel_size=1)

    def forward(self, x_hist, x_fut_cov, x_raw_static, z_station):
        B, L, _ = x_hist.shape
        H = x_fut_cov.shape[1]

        # --- Encoder ---
        z_hist = z_station.unsqueeze(1).expand(B, L, z_station.shape[-1])
        x_enc = torch.cat([x_hist, z_hist], dim=-1).transpose(1, 2)
        h = self.enc_in(x_enc)
        h = self.encoder(h)
        enc_context = h[:, :, -1]  # [B, C]

        # --- Static dropout ---
        if self.force_zero_static:
            x_raw_static = torch.zeros_like(x_raw_static)
        elif self.training and self.p_drop_static > 0:
            keep = (torch.rand(B, 1, device=x_raw_static.device) > self.p_drop_static).float()
            x_raw_static = x_raw_static * keep

        # --- Decoder ---
        stat_seq = x_raw_static.unsqueeze(1).expand(B, H, x_raw_static.shape[-1])
        z_fut = z_station.unsqueeze(1).expand(B, H, z_station.shape[-1])
        enc_ctx_seq = enc_context.unsqueeze(1).expand(B, H, enc_context.shape[-1])

        x_dec = torch.cat([x_fut_cov, stat_seq, z_fut, enc_ctx_seq], dim=-1).transpose(1, 2)
        d = self.dec_in(x_dec)
        d = self.decoder(d)
        yhat = self.head(d).squeeze(1)

        # --- Bias correction (optional) ---
        if self.bias_head is not None:
            b = self.bias_head(z_station, x_fut_cov)
            yhat = yhat + self.bias_scale * b

        return yhat

class EMA:
    '''Класс для коррекции весов модели методом скользящего среднего'''
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name])
        self.backup = {}

class StudentModel(nn.Module):
    '''Финальная модель'''
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
        tcn_block='gated',
        use_bias_head: bool = False,
        bias_hidden: int = 64,
        bias_dropout: float = 0.1,
        bias_scale: float = 1

    ):
        super().__init__()
        self.tcn = EncoderDecoderTCN(
            d_hist=d_hist,
            d_fut_cov=d_fut_cov,
            d_gnn=gnn_emb_dim,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            tcn_block=tcn_block,
            use_bias_head=use_bias_head,
            bias_hidden=bias_hidden,
            bias_dropout=bias_dropout,
            bias_scale=bias_scale
        )

    def forward(self, x_hist, x_fut_cov, z):
        yhat = self.tcn(x_hist, x_fut_cov, z)
        return yhat
    
class TeacherModel(nn.Module):
    '''Модель-учитель для GNN'''
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
        tcn_block='gated',
        use_bias_head: bool = False,
        bias_hidden: int = 64,
        bias_dropout: float = 0.1,
        bias_scale: float = 1

    ):
        super().__init__()
        self.gnn = StationGNN(node_in_dim=node_in_dim, gnn_hidden=gnn_hidden, emb_dim=gnn_emb_dim, dropout=dropout)
        self.tcn = EncoderDecoderTCN_static(
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
            tcn_block=tcn_block,
            use_bias_head=use_bias_head,
            bias_hidden=bias_hidden,
            bias_dropout=bias_dropout,
            bias_scale=bias_scale
        )

    def forward(self, x_hist, x_fut_cov, x_raw_static, station_id,
                node_features_all, edge_index, edge_weight):
        edge_weight = None if edge_weight.numel() == 0 else edge_weight
        z_all = self.gnn(node_features_all, edge_index, edge_weight=edge_weight)
        z = z_all[station_id]
        yhat = self.tcn(x_hist, x_fut_cov, x_raw_static, z)
        return yhat
    