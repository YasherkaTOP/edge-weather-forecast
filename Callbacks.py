import lightning.pytorch as pl
import torch

class StaticDropoutAnnealCallback(pl.Callback):
    '''Anneals the static dropout probability p_drop_static in the TCN model over epochs'''
    def __init__(self, p_start=0.3, p_end=1.0, start_epoch=0, anneal_epochs=4, log_prog_bar=True):
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.start_epoch = int(start_epoch)
        self.anneal_epochs = int(max(1, anneal_epochs))
        self.log_prog_bar = log_prog_bar

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch < self.start_epoch:
            p = self.p_start
        else:
            t = (epoch - self.start_epoch) / self.anneal_epochs
            t = max(0.0, min(1.0, t))
            p = self.p_start + t * (self.p_end - self.p_start)

        pl_module.model.tcn.p_drop_static = float(p)

        pl_module.log(
            "p_drop_static",
            float(p),
            prog_bar=self.log_prog_bar,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        
    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            pl_module.model.tcn.force_zero_static = True

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.model.tcn.force_zero_static = False

class FreezeGNNCallback(pl.Callback):
    """
    Freeze GNN parameters at a given epoch (inclusive).
    """
    def __init__(self, freeze_epoch: int = 6, log_prog_bar: bool = True):
        self.freeze_epoch = int(freeze_epoch)
        self.log_prog_bar = log_prog_bar
        self._done = False

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.log(
                "gnn_frozen",
                self._done,
                prog_bar=self.log_prog_bar,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
        if self._done:
            return
        epoch = trainer.current_epoch
        if epoch >= self.freeze_epoch:
            gnn = pl_module.model.gnn
            for p in gnn.parameters():
                p.requires_grad = False
            self._done = True

class FreezeGNNOnDeltaCallback(pl.Callback):
    '''gnn freezing based on monitored metric (e.g., delta loss)'''
    def __init__(
        self,
        monitor: str = "gnn_z_delta_l1_mean",
        threshold: float = 5e-4,
        patience: int = 2,
        min_epoch: int = 3,
        require_p_drop_static: float = 0.999,
        log_prog_bar: bool = True
    ):
        self.monitor = monitor
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.min_epoch = int(min_epoch)
        self.require_p_drop_static = float(require_p_drop_static)
        self.log_prog_bar = log_prog_bar

        self._count = 0
        self._done = False

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log(
            "gnn_frozen",
            self._done,
            prog_bar=self.log_prog_bar,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        if self._done:
            return

        epoch = trainer.current_epoch
        if epoch < self.min_epoch:
            return

        p_drop = getattr(pl_module.model.tcn, "p_drop_static", None)

        if p_drop is None or p_drop < self.require_p_drop_static:
            self._count = 0
            return

        m = trainer.callback_metrics.get(self.monitor, None)
        if m is None:
            return

        if isinstance(m, torch.Tensor):
            m_val = float(m.detach().cpu().item())
        else:
            m_val = float(m)

        if m_val < self.threshold:
            self._count += 1
        else:
            self._count = 0

        if self._count >= self.patience:
            gnn = pl_module.model.gnn
            for p in gnn.parameters():
                p.requires_grad = False

            self._done = True

class LossWarmupAnnealCallback(pl.Callback):
    '''Anneals certain loss parameters (e.g., lam_diff, big_err_weight) from start to end values over epochs'''
    def __init__(
        self,
        epoch_start: int = 0,
        epoch_end: int = 4,
        start: dict | None = None,
        end: dict | None = None,
        log_prog_bar: bool = True,
    ):
        self.epoch_start = int(epoch_start)
        self.epoch_end = int(epoch_end)
        self.log_prog_bar = log_prog_bar

        if start is None:
            start = {
                "lam_diff": 0.0,
                "big_err_weight": 0.1,
                "extreme_weight": 0.0,
            }
        if end is None:
            end = {
                "lam_diff": 0.2,
                "big_err_weight": 0.3,
                "extreme_weight": 0.2,
            }

        self.start = dict(start)
        self.end = dict(end)

    def _interp(self, a: float, b: float, t: float) -> float:
        return float(a + (b - a) * t)

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        cfg = pl_module.loss_fn.cfg

        if self.epoch_end <= self.epoch_start:
            t = 1.0
        elif epoch <= self.epoch_start:
            t = 0.0
        elif epoch >= self.epoch_end:
            t = 1.0
        else:
            t = (epoch - self.epoch_start) / (self.epoch_end - self.epoch_start)

        for k in self.end.keys():
            if not hasattr(cfg, k):
                continue
            a = self.start.get(k, getattr(cfg, k))
            b = self.end[k]
            v = self._interp(float(a), float(b), t)
            setattr(cfg, k, v)

        self.log_dict(
            {
                "lam_diff": cfg.lam_diff,
                "big_err_weight": cfg.big_err_weight,
                "extreme_weight": cfg.extreme_weight,
            },
            prog_bar=self.log_prog_bar,
            logger=True,
            on_step=False,
            on_epoch=True,
        )