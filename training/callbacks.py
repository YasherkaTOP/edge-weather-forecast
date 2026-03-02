"""
Training callbacks for PyTorch Lightning.

Classes
-------
StaticDropoutAnnealCallback   – linearly anneals TCN p_keep_static over epochs
FreezeGNNCallback             – freezes GNN at a fixed epoch
FreezeGNNOnDeltaCallback      – freezes GNN when a monitored metric is stable
LossWarmupAnnealCallback      – linearly anneals loss hyper-params over epochs
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import lightning.pytorch as pl

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StaticDropoutAnnealCallback
# ---------------------------------------------------------------------------

class StaticDropoutAnnealCallback(pl.Callback):
    """
    Линейный дроп ``tcn.p_drop_static`` с *p_start* до *p_end*.

    p_drop_static — это вероятность ОБНУЛИТЬ статик-вектор (drop probability):
      - p=0.0  → статик никогда не обнуляется (нет дропа)
      - p=1.0  → статик всегда обнуляется (максимальный дроп)

    Parameters
    ----------
    p_start:
        Drop-probability в начале отжига.
    p_end:
        Целевая drop-probability (достигается на эпохе ``start_epoch + anneal_duration``).
    start_epoch:
        Первая эпоха, с которой начинается отжиг.
    anneal_duration:
        Количество эпох для интерполяции.
    """

    def __init__(
        self,
        p_start: float = 0.3,
        p_end: float = 1.0,
        start_epoch: int = 0,
        anneal_duration: int = 4,
        log_prog_bar: bool = True,
    ):
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.start_epoch = int(start_epoch)
        self.anneal_duration = int(max(1, anneal_duration))
        self.log_prog_bar = log_prog_bar

    def _current_p(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return self.p_start
        t = (epoch - self.start_epoch) / self.anneal_duration
        t = max(0.0, min(1.0, t))
        return self.p_start + t * (self.p_end - self.p_start)

    def _get_tcn(self, pl_module: pl.LightningModule) -> Optional[object]:
        tcn = getattr(getattr(pl_module, "model", None), "tcn", None)
        if tcn is None:
            log.warning("StaticDropoutAnnealCallback: не найден model.tcn — пропуск.")
        return tcn

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        p = self._current_p(trainer.current_epoch)
        tcn = self._get_tcn(pl_module)
        if tcn is not None and hasattr(tcn, "p_drop_static"):
            tcn.p_drop_static = float(p)
        pl_module.log(
            "p_drop_static", float(p),
            prog_bar=self.log_prog_bar, logger=True,
            on_step=False, on_epoch=True,
        )

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        tcn = self._get_tcn(pl_module)
        if tcn is not None and hasattr(tcn, "force_zero_static"):
            tcn.force_zero_static = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        tcn = self._get_tcn(pl_module)
        if tcn is not None and hasattr(tcn, "force_zero_static"):
            tcn.force_zero_static = False


# ---------------------------------------------------------------------------
# FreezeGNNCallback
# ---------------------------------------------------------------------------

class FreezeGNNCallback(pl.Callback):
    """
    Замораживает веса GNN начиная с указанной эпохи.

    Parameters
    ----------
    freeze_epoch:
        Эпоха (0-indexed), начиная с которой замораживаются веса GNN.
    """

    def __init__(self, freeze_epoch: int = 6, log_prog_bar: bool = True):
        self.freeze_epoch = int(freeze_epoch)
        self.log_prog_bar = log_prog_bar
        self._done: bool = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Размораживаем GNN при старте на случай resume с уже замороженными весами
        self._done = False
        gnn = getattr(getattr(pl_module, "model", None), "gnn", None)
        if gnn is not None:
            for p in gnn.parameters():
                p.requires_grad = True

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log(
            "gnn_frozen", float(self._done),
            prog_bar=self.log_prog_bar, logger=True,
            on_step=False, on_epoch=True,
        )
        if self._done:
            return
        if trainer.current_epoch >= self.freeze_epoch:
            gnn = getattr(getattr(pl_module, "model", None), "gnn", None)
            if gnn is None:
                log.warning("FreezeGNNCallback: не найден model.gnn — пропуск.")
                return
            for p in gnn.parameters():
                p.requires_grad = False
            self._done = True
            log.info("FreezeGNNCallback: GNN заморожен на эпохе %d.", trainer.current_epoch)


# ---------------------------------------------------------------------------
# FreezeGNNOnDeltaCallback
# ---------------------------------------------------------------------------

class FreezeGNNOnDeltaCallback(pl.Callback):
    """
    Замораживает GNN, когда мониторируемая метрика стабилизируется.

    Parameters
    ----------
    monitor:
        Имя метрики для отслеживания.
    threshold:
        Метрика должна быть ниже этого значения для накопления patience.
    patience:
        Число подряд идущих эпох ниже threshold для заморозки.
    min_epoch:
        Минимальная эпоха, раньше которой заморозка невозможна.
    require_p_keep_static:
        Заморозка блокируется, пока ``model.tcn.p_keep_static`` не достигнет
        этого значения.
    """

    def __init__(
        self,
        monitor: str = "gnn_z_delta_l1_mean",
        threshold: float = 5e-4,
        patience: int = 2,
        min_epoch: int = 3,
        require_p_drop_static: float = 0.999,
        log_prog_bar: bool = True,
    ):
        self.monitor = monitor
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.min_epoch = int(min_epoch)
        self.require_p_drop_static = float(require_p_drop_static)
        self.log_prog_bar = log_prog_bar
        self._count: int = 0
        self._done: bool = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._count = 0
        self._done = False

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log(
            "gnn_frozen", float(self._done),
            prog_bar=self.log_prog_bar, logger=True,
            on_step=False, on_epoch=True,
        )
        if self._done or trainer.current_epoch < self.min_epoch:
            return

        tcn = getattr(getattr(pl_module, "model", None), "tcn", None)
        p_keep = getattr(tcn, "p_keep_static", None)
        if p_keep is None or p_keep < self.require_p_keep_static:
            self._count = 0
            return

        m = trainer.callback_metrics.get(self.monitor)
        if m is None:
            log.warning(
                "FreezeGNNOnDeltaCallback: метрика '%s' не найдена — пропуск.", self.monitor
            )
            return

        m_val = float(m.detach().cpu().item()) if isinstance(m, torch.Tensor) else float(m)
        if not math.isfinite(m_val):
            log.warning(
                "FreezeGNNOnDeltaCallback: метрика '%s' = %s (не конечное) — сброс счётчика.",
                self.monitor, m_val,
            )
            self._count = 0
            return

        if m_val < self.threshold:
            self._count += 1
        else:
            self._count = 0

        if self._count >= self.patience:
            gnn = getattr(getattr(pl_module, "model", None), "gnn", None)
            if gnn is None:
                log.warning("FreezeGNNOnDeltaCallback: не найден model.gnn — пропуск.")
                return
            for p in gnn.parameters():
                p.requires_grad = False
            self._done = True
            log.info(
                "FreezeGNNOnDeltaCallback: GNN заморожен на эпохе %d (метрика=%.2e).",
                trainer.current_epoch, m_val,
            )


# ---------------------------------------------------------------------------
# LossWarmupAnnealCallback
# ---------------------------------------------------------------------------

_DEFAULT_LOSS_START: dict[str, float] = {
    "lam_diff": 0.0,
    "big_err_weight": 0.1,
    "extreme_weight": 0.0,
}

_DEFAULT_LOSS_END: dict[str, float] = {
    "lam_diff": 0.2,
    "big_err_weight": 0.3,
    "extreme_weight": 0.2,
}


class LossWarmupAnnealCallback(pl.Callback):
    """
    Линейно изменяет гиперпараметры функции потерь в течение обучения.

    Parameters
    ----------
    epoch_start / epoch_end:
        Диапазон эпох для интерполяции.
    start / end:
        Словари с начальными и конечными значениями параметров cfg лосса.
    """

    def __init__(
        self,
        epoch_start: int = 0,
        epoch_end: int = 4,
        start: dict[str, float] | None = None,
        end: dict[str, float] | None = None,
        log_prog_bar: bool = True,
    ):
        self.epoch_start = int(epoch_start)
        self.epoch_end = int(epoch_end)
        self.log_prog_bar = log_prog_bar
        self.start: dict[str, float] = dict(start) if start is not None else _DEFAULT_LOSS_START.copy()
        self.end: dict[str, float] = dict(end) if end is not None else _DEFAULT_LOSS_END.copy()

        # Предупреждение о ключах в start, которых нет в end
        orphan = set(self.start) - set(self.end)
        if orphan:
            log.warning(
                "LossWarmupAnnealCallback: ключи %s есть в start, но не в end — они будут проигнорированы.",
                orphan,
            )

    @staticmethod
    def _interp(a: float, b: float, t: float) -> float:
        return float(a + (b - a) * t)

    def _compute_t(self, epoch: int) -> float:
        if self.epoch_end <= self.epoch_start:
            return 1.0
        if epoch <= self.epoch_start:
            return 0.0
        if epoch >= self.epoch_end:
            return 1.0
        return (epoch - self.epoch_start) / (self.epoch_end - self.epoch_start)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        cfg = pl_module.loss_fn.cfg
        t = self._compute_t(trainer.current_epoch)

        for k in self.end:
            if not hasattr(cfg, k):
                log.warning("LossWarmupAnnealCallback: cfg не имеет атрибута '%s' — пропуск.", k)
                continue
            a = self.start.get(k, float(getattr(cfg, k)))
            b = self.end[k]
            setattr(cfg, k, self._interp(a, b, t))

        pl_module.log_dict(
            {k: float(getattr(cfg, k)) for k in self.end if hasattr(cfg, k)},
            prog_bar=self.log_prog_bar, logger=True,
            on_step=False, on_epoch=True,
        )
