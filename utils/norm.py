import torch.nn as nn


def make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Создаёт GroupNorm, автоматически подбирая число групп-делителей."""
    g = min(max_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)
