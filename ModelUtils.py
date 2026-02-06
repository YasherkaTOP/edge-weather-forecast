import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class HorizonLossConfig:
    '''General config for horizon-weighted loss'''
    w_0_6h: float = 2.0 # вес для горизонта 0-6ч
    w_6_24h: float = 1.5 # вес для горизонта 6-24ч
    w_1_3d: float = 1.2 # вес для горизонта 1-3д
    w_3_7d: float = 1.0 # вес для горизонта 3-7д
    use_big_err_penalty: bool = False # использовать ли штраф за большие ошибки
    big_err_thresh: float = 3.0 # порог большой ошибки (°C)
    big_err_weight: float = 1.0 # вес штрафа за большие ошибки
    lam_diff: float = 0.3 # вес для дифференциальной ошибки, если <=0 - отключено
    use_extreme_weighting: bool = False # использовать ли экстремальное взвешивание
    cold_thresh: float = -10.0 # порог для холодной температуры
    hot_thresh: float = 25.0 # порог для жаркой температуры
    cold_weight: float = 1.5 # коэффициент добавки на холод
    hot_weight: float = 1.5 # коэффициент добавки на жару
    extreme_weight: float = 0.3 # вес для экстремального взвешивания
    use_std_penalty: bool = False # использовать ли штраф за несоответствие в stddev в хвосте
    tail_std_weight: float = 0.4 # вес для штрафа за stddev в хвосте
    tail_start: int = 24 # начало хвоста для stddev penalty (часы)
    tail_end: int = 168 # конец хвоста для stddev penalty (часы)


class HorizonWeightedLoss(nn.Module):
    '''Loss for temperature forecasting: weighted MAE over different forecast horizons, with optional penalties'''
    def __init__(self, cfg: HorizonLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # yhat, y: [B, H] predicted and true temps over forecast horizon
        err = yhat - y
        ae = err.abs()

        # segment losses
        H = y.shape[1]
        def seg(a, b):
            a = max(0, a); b = min(H, b)
            if a >= b:
                return None
            return ae[:, a:b].mean()

        losses = {}
        l0 = seg(0, 6);    losses["mae_0_6h"] = l0 if l0 is not None else ae.new_tensor(0.0)
        l1 = seg(6, 24);   losses["mae_6_24h"] = l1 if l1 is not None else ae.new_tensor(0.0)
        l2 = seg(24, 72);  losses["mae_1_3d"] = l2 if l2 is not None else ae.new_tensor(0.0)
        l3 = seg(72, 168); losses["mae_3_7d"] = l3 if l3 is not None else ae.new_tensor(0.0)

        total = (
            self.cfg.w_0_6h * losses["mae_0_6h"] +
            self.cfg.w_6_24h * losses["mae_6_24h"] +
            self.cfg.w_1_3d * losses["mae_1_3d"] +
            self.cfg.w_3_7d * losses["mae_3_7d"]
        )
        
        # penalty for mismatch in stddev in the tail
        if self.cfg.use_std_penalty:
            tail_start = self.cfg.tail_start
            tail_end = self.cfg.tail_end
            yt = y[:, tail_start:tail_end]
            yp = yhat[:, tail_start:tail_end]

            std_p = yp.std(dim=1)
            std_t = yt.std(dim=1)

            err = (std_p / std_t).abs().mean()

            total = total + self.cfg.tail_std_weight * err 

        # big error penalty
        if self.cfg.use_big_err_penalty:
            big = F.relu(ae - self.cfg.big_err_thresh)
            big_pen = (big ** 2).mean()
        else:
            big_pen = ae.new_tensor(0.0)
        losses["big_err_pen"] = big_pen
        total = total + self.cfg.big_err_weight * big_pen

        # diff loss: MAE on first differences
        if self.cfg.lam_diff > 0 and H >= 2:
            dy_hat = yhat[:, 1:] - yhat[:, :-1]
            dy = y[:, 1:] - y[:, :-1]
            loss_diff = (dy_hat - dy).abs().mean()
        else:
            loss_diff = ae.new_tensor(0.0)
        losses["loss_diff"] = loss_diff
        total = total + self.cfg.lam_diff * loss_diff

        # extreme weighting: upweight errors when true temp is very low/high
        if self.cfg.use_extreme_weighting:
            w = torch.ones_like(y)
            w = torch.where(y <= self.cfg.cold_thresh, w * self.cfg.cold_weight, w)
            w = torch.where(y >= self.cfg.hot_thresh,  w * self.cfg.hot_weight,  w)
            w_norm = w / (w.mean().clamp_min(1e-6))
            loss_extreme = (w_norm * ae).mean()
        else:
            loss_extreme = ae.new_tensor(0.0)

        losses["loss_extreme"] = loss_extreme
        total = total + self.cfg.extreme_weight * loss_extreme

        losses["loss"] = total
        losses["mae_all"] = ae.mean()
        return total, losses