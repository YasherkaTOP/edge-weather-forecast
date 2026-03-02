"""Exponential Moving Average для весов модели."""
import torch


class EMA:
    """
    Накапливает скользящее среднее весов модели.

    Использование:
        ema = EMA(model, decay=0.999)
        # после каждого шага оптимизатора:
        ema.update(model)
        # для валидации:
        ema.apply_shadow(model)
        ... evaluate ...
        ema.restore(model)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Обновляет теневые веса после шага оптимизатора."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Подставляет EMA-веса в модель (сохраняя оригиналы в backup)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        """Возвращает оригинальные веса из backup."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name])
        self.backup = {}
