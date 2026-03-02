"""
Утилиты для оценки качества прогнозов.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Инференс по DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_on_loader(
    lit_model: torch.nn.Module,
    loader,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Прогоняет модель через весь загрузчик данных.

    Returns
    -------
    yhat : [N, H]
    y    : [N, H]
    sid  : [N]     station_id
    t0h  : [N]     t0 в часах от Unix-эпохи
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lit_model.eval().to(device)

    all_yhat, all_y, all_sid, all_t0h = [], [], [], []

    for batch in loader:
        batch_t = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        yhat = lit_model(batch_t)
        all_yhat.append(yhat.detach().cpu())
        all_y.append(batch_t["y"].detach().cpu())
        all_sid.append(batch_t["station_id"].detach().cpu())
        all_t0h.append(batch_t["t0"].detach().cpu())

    return (
        torch.cat(all_yhat).numpy(),
        torch.cat(all_y).numpy(),
        torch.cat(all_sid).numpy(),
        torch.cat(all_t0h).numpy().astype(np.int64),
    )


def t0h_to_datetime(t0h: np.ndarray) -> pd.DatetimeIndex:
    """Конвертирует int64 (часы от Unix-эпохи) в pd.DatetimeIndex UTC."""
    return pd.to_datetime(t0h, unit="h", utc=True)


# ---------------------------------------------------------------------------
# MAE по диапазонам
# ---------------------------------------------------------------------------

RANGES = [
    ("0-6h",  0,  6),
    ("6-24h", 6,  24),
    ("1-3d",  24, 72),
    ("3-7d",  72, 168),
]


def mae_ranges(yhat: np.ndarray, y: np.ndarray, a: int, b: int) -> Optional[float]:
    H = y.shape[1]
    a, b = max(0, min(a, H)), max(0, min(b, H))
    if a >= b:
        return None
    return float(np.abs(yhat[:, a:b] - y[:, a:b]).mean())


def mae_by_ranges_per_station(
    yhat: np.ndarray, y: np.ndarray, sid: np.ndarray
) -> dict[int, dict[str, float]]:
    """MAE по горизонтам для каждой станции."""
    out = {}
    for st in np.unique(sid):
        m = sid == st
        e = np.abs(yhat[m] - y[m])
        stats = {"MAE_all": float(e.mean())}
        for name, a, b in RANGES:
            v = mae_ranges(yhat[m], y[m], a, b)
            if v is not None:
                stats[f"MAE_{name}"] = v
        out[int(st)] = stats
    return out


# ---------------------------------------------------------------------------
# Сезонные метрики
# ---------------------------------------------------------------------------

def _month_to_season(m: int) -> str:
    if m in (12, 1, 2):  return "DJF"
    if m in (3, 4, 5):   return "MAM"
    if m in (6, 7, 8):   return "JJA"
    return "SON"


def seasonal_mae_bias(
    yhat: np.ndarray, y: np.ndarray,
    sid: np.ndarray, t0_dt: pd.DatetimeIndex,
) -> pd.DataFrame:
    """MAE и Bias по сезонам и станциям."""
    seasons = pd.Series(t0_dt.month).map(_month_to_season).values
    rows = []
    for st in np.unique(sid):
        m_st = sid == st
        for season in ["DJF", "MAM", "JJA", "SON"]:
            m = m_st & (seasons == season)
            if m.sum() == 0:
                continue
            err = yhat[m] - y[m]
            rows.append({
                "station_id": int(st),
                "season": season,
                "n_windows": int(m.sum()),
                "MAE_all": float(np.abs(err).mean()),
                "Bias_all": float(err.mean()),
            })
    return pd.DataFrame(rows).sort_values(["station_id", "season"])


def seasonal_horizon_metrics(
    yhat: np.ndarray, y: np.ndarray,
    sid: np.ndarray, t0_dt: pd.DatetimeIndex,
) -> pd.DataFrame:
    """MAE и Bias по сезонам, станциям и горизонтам."""
    seasons = pd.Series(t0_dt.month).map(_month_to_season).values
    H = y.shape[1]
    rows = []
    for st in np.unique(sid):
        m_st = sid == st
        for season in ["DJF", "MAM", "JJA", "SON"]:
            m = m_st & (seasons == season)
            if m.sum() == 0:
                continue
            err = yhat[m] - y[m]
            row = {
                "station_id": int(st),
                "season": season,
                "n_windows": int(m.sum()),
                "MAE_all": float(np.abs(err).mean()),
                "Bias_all": float(err.mean()),
            }
            for name, a, b in RANGES:
                a2, b2 = max(0, a), min(H, b)
                if a2 >= b2:
                    continue
                e_seg = err[:, a2:b2]
                row[f"MAE_{name}"]  = float(np.abs(e_seg).mean())
                row[f"Bias_{name}"] = float(e_seg.mean())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["station_id", "season"])


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_monthly_forecast_one_shot(
    yhat: np.ndarray,
    y: np.ndarray,
    sid: np.ndarray,
    t0h: np.ndarray,
    station_id: int,
    H: int = 168,
    months=range(1, 13),
    year: Optional[int] = None,
) -> None:
    """
    Рисует прогноз vs истину для каждого месяца указанной станции.
    """
    mask = sid == station_id
    if mask.sum() == 0:
        print(f"Нет данных для станции {station_id}")
        return

    yhat_s = yhat[mask]
    y_s    = y[mask]
    t0_dt  = t0h_to_datetime(np.asarray(t0h)[mask])
    t0_series = pd.Series(t0_dt)

    if year is None:
        year = int(t0_series.dt.year.max())

    for m in months:
        idxs = np.where(
            (t0_series.dt.year == year) & (t0_series.dt.month == m)
        )[0]
        if len(idxs) == 0:
            continue

        desired = pd.Timestamp(year=year, month=m, day=1, hour=0, tz="UTC")
        diffs_h = np.abs(
            (t0_series.iloc[idxs] - desired).values
            .astype("timedelta64[h]").astype(int)
        )
        pick = int(idxs[np.argmin(diffs_h)])

        plt.figure(figsize=(10, 4))
        plt.plot(y_s[pick, :H],    label="truth")
        plt.plot(yhat_s[pick, :H], label="forecast")
        plt.title(
            f"Station {station_id} | {year}-{m:02d}-01 | "
            f"t0={t0_series.iloc[pick]}"
        )
        plt.xlabel("horizon hour")
        plt.ylabel("temperature (°C)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
