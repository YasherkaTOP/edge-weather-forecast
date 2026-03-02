import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class HorizonLossConfig:
    """Конфиг для взвешенного лосса по горизонту прогноза."""
    w_0_6h: float = 2.0           # вес горизонта 0–6 ч
    w_6_24h: float = 1.5          # вес горизонта 6–24 ч
    w_1_3d: float = 1.2           # вес горизонта 1–3 дня
    w_3_7d: float = 1.0           # вес горизонта 3–7 дней

    use_big_err_penalty: bool = False  # штраф за большие ошибки
    big_err_thresh: float = 3.0        # порог большой ошибки (°C)
    big_err_weight: float = 1.0        # вес штрафа

    lam_diff: float = 0.3              # вес дифференциального лосса (0 → отключён)

    use_extreme_weighting: bool = False  # взвешивание экстремальных температур
    cold_thresh: float = -10.0
    hot_thresh: float = 25.0
    cold_weight: float = 1.5
    hot_weight: float = 1.5
    extreme_weight: float = 0.3

    use_std_penalty: bool = False  # штраф за несоответствие std на хвосте
    tail_std_weight: float = 0.4
    tail_start: int = 24
    tail_end: int = 168


class HorizonWeightedLoss(nn.Module):
    """
    Взвешенный MAE по горизонтам с опциональными штрафами.

    Входы:
        yhat: [B, H]  — предсказания
        y:    [B, H]  — истинные значения

    Возвращает:
        (total_loss, dict_of_components)
    """

    def __init__(self, cfg: HorizonLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self, yhat: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        H = y.shape[1]
        err = yhat - y
        ae = err.abs()

        def seg_mae(a: int, b: int) -> torch.Tensor:
            a, b = max(0, a), min(H, b)
            if a >= b:
                return ae.new_tensor(0.0)
            return ae[:, a:b].mean()

        losses: Dict[str, torch.Tensor] = {
            "mae_0_6h":  seg_mae(0, 6),
            "mae_6_24h": seg_mae(6, 24),
            "mae_1_3d":  seg_mae(24, 72),
            "mae_3_7d":  seg_mae(72, 168),
        }

        total = (
            self.cfg.w_0_6h  * losses["mae_0_6h"]
            + self.cfg.w_6_24h * losses["mae_6_24h"]
            + self.cfg.w_1_3d  * losses["mae_1_3d"]
            + self.cfg.w_3_7d  * losses["mae_3_7d"]
        )

        # --- std penalty на хвосте ---
        if self.cfg.use_std_penalty:
            ts, te = self.cfg.tail_start, self.cfg.tail_end
            std_p = yhat[:, ts:te].std(dim=1)
            std_t = y[:, ts:te].std(dim=1)
            std_err = ((std_p - std_t).abs() / std_t.clamp_min(1e-6)).mean()
            total = total + self.cfg.tail_std_weight * std_err
            losses["std_penalty"] = std_err

        # --- штраф за большие ошибки ---
        if self.cfg.use_big_err_penalty:
            big_pen = (F.relu(ae - self.cfg.big_err_thresh) ** 2).mean()
        else:
            big_pen = ae.new_tensor(0.0)
        losses["big_err_pen"] = big_pen
        total = total + self.cfg.big_err_weight * big_pen

        # --- дифференциальный лосс ---
        if self.cfg.lam_diff > 0 and H >= 2:
            loss_diff = ((yhat[:, 1:] - yhat[:, :-1]) - (y[:, 1:] - y[:, :-1])).abs().mean()
        else:
            loss_diff = ae.new_tensor(0.0)
        losses["loss_diff"] = loss_diff
        total = total + self.cfg.lam_diff * loss_diff

        # --- экстремальное взвешивание ---
        if self.cfg.use_extreme_weighting:
            w = torch.ones_like(y)
            w = torch.where(y <= self.cfg.cold_thresh, w * self.cfg.cold_weight, w)
            w = torch.where(y >= self.cfg.hot_thresh,  w * self.cfg.hot_weight,  w)
            w_norm = w / w.mean().clamp_min(1e-6)
            loss_extreme = (w_norm * ae).mean()
        else:
            loss_extreme = ae.new_tensor(0.0)
        losses["loss_extreme"] = loss_extreme
        total = total + self.cfg.extreme_weight * loss_extreme

        losses["loss"] = total
        losses["mae_all"] = ae.mean()
        return total, losses
