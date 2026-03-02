import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from utils.norm import make_gn


class SEGate(nn.Module):
    """Squeeze-and-Excitation Gate для 1D сигналов [B, C, T]."""

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(ch, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)

class AttnPool1D(nn.Module):
    """
    Обучаемый attention-пулинг по временной оси.
    Вход:  [B, C, T]  →  Выход: [B, C]
    """

    def __init__(self, channels: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.proj(h), dim=-1)   # [B, 1, T]
        return (h * attn).sum(dim=-1)                # [B, C]


# ---------------------------------------------------------------------------
# TCN-блоки
# ---------------------------------------------------------------------------

class GatedResidualTCNBlock(nn.Module):
    """WaveNet-style gated dilated conv с GroupNorm и residual-связью."""

    def __init__(
        self, in_ch: int, out_ch: int,
        kernel_size: int = 3, dilation: int = 1,
        dropout: float = 0.1, gn_groups: int = 8,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv = nn.Conv1d(in_ch, 2 * out_ch, kernel_size, padding=padding, dilation=dilation)
        self.gn_f = make_gn(out_ch, gn_groups)
        self.gn_g = make_gn(out_ch, gn_groups)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._chomp(self.conv(x))
        out_ch = h.shape[1] // 2
        h = torch.tanh(self.gn_f(h[:, :out_ch])) * torch.sigmoid(self.gn_g(h[:, out_ch:]))
        return self.skip(x) + self.drop(h)


class ResidualDWSeparableGatedTCNBlock(nn.Module):
    """Depthwise-Separable причинный TCN с SEGate, GroupNorm и residual-связью."""

    def __init__(
        self, in_ch: int, out_ch: int,
        kernel_size: int = 3, dilation: int = 1,
        dropout: float = 0.1, se_reduction: int = 4, gn_groups: int = 8,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding

        self.dw1 = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding, dilation=dilation, groups=in_ch)
        self.pw1 = nn.Conv1d(in_ch, out_ch, 1)
        self.gn1 = make_gn(out_ch, gn_groups)

        self.dw2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation, groups=out_ch)
        self.pw2 = nn.Conv1d(out_ch, out_ch, 1)
        self.gn2 = make_gn(out_ch, gn_groups)

        self.gate = SEGate(out_ch, reduction=se_reduction)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.gn1(self.pw1(self._chomp(self.dw1(x))))))
        h = self.act(self.gn2(self.pw2(self._chomp(self.dw2(h)))))
        return self.gate(h) + self.skip(x)
    
# ---------------------------------------------------------------------------
# Спектральный блок с условием на станцию
# ---------------------------------------------------------------------------

class StationCondMultiBandSpectralBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        z_dim: int,
        T_in: int = 672,
        day_width_init: float = 2.0,
        week_width_init: float = 1.0,
        low_k_init: float = 2.0,
        dropout: float = 0.1,
        gn_groups: int = 8,
        hidden: int = 128,
    ):
        super().__init__()
        self.C = channels
        self.Z = z_dim
        self.T_in = T_in
        self.F = T_in // 2 + 1
        self.n_bands = 4

        self.k_day = int(round(T_in / 24))   # суточный цикл
        self.k_week = int(round(T_in / 168))  # недельный цикл

        self.log_sigma_day = nn.Parameter(torch.tensor(math.log(math.expm1(day_width_init))))
        self.log_sigma_week = nn.Parameter(torch.tensor(math.log(math.expm1(week_width_init))))

        self.log_low_k = nn.Parameter(torch.tensor(math.log(math.expm1(low_k_init))))

        self.log_mag_base = nn.Parameter(torch.zeros(self.n_bands, channels))
        self.phase_base = nn.Parameter(torch.zeros(self.n_bands, channels))

        self.delta_net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_bands * 2),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(z_dim + self.n_bands, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_bands),
            nn.Sigmoid()
        )

        self.mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.gn = make_gn(channels, gn_groups)
        self.drop = nn.Dropout1d(dropout)
        self.act = nn.SiLU()

        bins = torch.arange(self.F, dtype=torch.float32)
        self.register_buffer("bins", bins, persistent=False)

    def _build_soft_masks(self) -> torch.Tensor:
        """Возвращает мягкие маски [n_bands, F] в [0,1]."""
        bins = self.bins

        sigma_day = F.softplus(self.log_sigma_day)   # ширина суточной полосы
        sigma_week = F.softplus(self.log_sigma_week)  # ширина недельной полосы
        low_k  = F.softplus(self.log_low_k)       # граница тренд-полосы

        day = torch.exp(-0.5 * ((bins - self.k_day)  / sigma_day)  ** 2)
        week = torch.exp(-0.5 * ((bins - self.k_week) / sigma_week) ** 2)

        low = torch.sigmoid(2.0 * (low_k - bins))

        other = (1.0 - low - week - day).clamp(min=0.0)

        return torch.stack([low, week, day, other], dim=0)

    def forward(self, x: torch.Tensor, z_station: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C == self.C
        if T != self.T_in:
            raise ValueError(f"Expected T={self.T_in}, got T={T}.")

        X = torch.fft.rfft(x, dim=-1)

        masks = self._build_soft_masks().to(X.real.dtype)

        mag2 = X.real ** 2 + X.imag ** 2
        eps = 1e-8
        bp = []
        for b in range(self.n_bands):
            m = masks[b][None, None, :]
            num = (mag2 * m).sum(dim=(1, 2))
            den = m.sum() * C + eps
            bp.append((num / den).unsqueeze(-1))
        bandpower = torch.log1p(torch.cat(bp, dim=-1))

        d = self.delta_net(z_station).view(B, self.n_bands, 2)
        d_logmag = d[:, :, 0]
        d_phase = d[:, :, 1]

        gates = self.gate_net(torch.cat([z_station, bandpower], dim=-1))

        Y = X
        for b in range(self.n_bands):
            m = masks[b][None, None, :]

            logmag = torch.clamp(
                self.log_mag_base[b][None, :] + d_logmag[:, b:b+1], -1.5, 1.5
            )
            phase = self.phase_base[b][None, :] + d_phase[:, b:b+1]

            mag = torch.exp(logmag).to(X.real.dtype)
            Hr = mag * torch.cos(phase.to(X.real.dtype))
            Hi = mag * torch.sin(phase.to(X.real.dtype))
            H = torch.complex(Hr, Hi).to(X.dtype)

            gate = gates[:, b:b+1].to(X.real.dtype)
            H_eff = (1.0 + gate * (H - 1.0)).to(X.dtype)

            Y = Y * (1.0 - m) + (Y * H_eff[:, :, None]) * m

        y = torch.fft.irfft(Y, n=T, dim=-1)
        y = self.mix(y)
        y = self.gn(y)
        y = self.act(y)
        y = self.drop(y)

        return x + y

# ---------------------------------------------------------------------------
# BiasHead
# ---------------------------------------------------------------------------

class BiasHead(nn.Module):
    """
    Предсказывает поправку к прогнозу на каждый шаг горизонта.
    Вход:  z_station [B, z_dim], x_fut_cov [B, H, fut_dim]
    Выход: [B, H]
    """

    def __init__(self, z_dim: int, fut_dim: int, hidden: int = 64, dropout: float = 0.1, clip_deg: float = 3.0):
        super().__init__()
        self.clip_deg = clip_deg
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + fut_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_station: torch.Tensor, x_fut_cov: torch.Tensor) -> torch.Tensor:
        B, H, _ = x_fut_cov.shape
        z_seq = z_station.unsqueeze(1).expand(B, H, z_station.shape[-1])
        inp = torch.cat([z_seq, x_fut_cov], dim=-1)
        b = self.mlp(inp).squeeze(-1)
        return self.clip_deg * torch.tanh(b / self.clip_deg)
