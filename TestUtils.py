import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def predict_on_loader(lit_model, loader, device=None):
    '''Метод для инференса модели по загрузчику'''
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lit_model.eval().to(device)

    all_yhat, all_y, all_sid, all_t0h = [], [], [], []

    for batch in loader:
        batch_t = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_t[k] = v.to(device)
            else:
                batch_t[k] = v

        yhat = lit_model(batch_t)
        y = batch_t["y"]
        sid = batch_t["station_id"]
        t0h = batch_t["t0"]

        all_yhat.append(yhat.detach().cpu())
        all_y.append(y.detach().cpu())
        all_sid.append(sid.detach().cpu())
        all_t0h.append(t0h.detach().cpu())

    yhat = torch.cat(all_yhat, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    sid = torch.cat(all_sid, dim=0).numpy()
    t0h = torch.cat(all_t0h, dim=0).numpy().astype(np.int64)

    return yhat, y, sid, t0h


def t0h_to_datetime(t0h: np.ndarray) -> pd.DatetimeIndex:
    # t0h: int64 hours since Unix epoch
    return pd.to_datetime(t0h, unit="h", utc=True)

def mae_ranges(yhat, y, a, b):
    H = y.shape[1]
    a = max(0, min(a, H))
    b = max(0, min(b, H))
    if a >= b:
        return None
    return float(np.abs(yhat[:, a:b] - y[:, a:b]).mean())

def mae_by_ranges_per_station(yhat, y, sid):
    H = y.shape[1]
    ranges = [
        ("0-6h", 0, 6),
        ("6-24h", 6, 24),
        ("1-3d", 24, 72),
        ("3-7d", 72, 168),
    ]

    out = {}
    for st in np.unique(sid):
        m = (sid == st)
        yh, yt = yhat[m], y[m]
        e = np.abs(yh - yt)
        stats = {"MAE_all": float(e.mean())}
        for name, a, b in ranges:
            v = mae_ranges(yh, yt, a, b)
            if v is not None:
                stats[f"MAE_{name}"] = v
        out[int(st)] = stats
    return out

def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM" 
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def seasonal_mae_bias(yhat, y, sid, t0_dt):
    seasons = pd.Series(t0_dt.month).map(month_to_season).values

    rows = []
    for st in np.unique(sid):
        m_st = (sid == st)
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

RANGES = [
    ("0-6h", 0, 6),
    ("6-24h", 6, 24),
    ("1-3d", 24, 72),
    ("3-7d", 72, 168),
]

def seasonal_horizon_metrics(yhat, y, sid, t0_dt):
    seasons = pd.Series(t0_dt.month).map(month_to_season).values
    H = y.shape[1]

    rows = []
    for st in np.unique(sid):
        m_st = (sid == st)
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
                row[f"MAE_{name}"] = float(np.abs(e_seg).mean())
                row[f"Bias_{name}"] = float(e_seg.mean())

            rows.append(row)

    return pd.DataFrame(rows).sort_values(["station_id", "season"])

def plot_monthly_forecast_one_shot(
    yhat, y, sid, t0h,
    station_id: int,
    H: int = 168,
    months=range(1, 13),
):
    mask = (sid == station_id)
    if mask.sum() == 0:
        print("No samples for station", station_id)
        return

    yhat_s = yhat[mask]
    y_s = y[mask]
    t0_dt = t0h_to_datetime(np.array(t0h)[mask])

    year = int(pd.Series(t0_dt.year).max())

    t0_series = pd.Series(t0_dt)

    for m in months:
        desired = pd.Timestamp(year=year, month=m, day=1, hour=0, tz="UTC")
        idxs = np.where((t0_series.dt.year == year) & (t0_series.dt.month == m))[0]
        if len(idxs) == 0:
            continue

        diffs_h = np.abs((t0_series.iloc[idxs] - desired).values.astype("timedelta64[h]").astype(int))
        pick_local = int(np.argmin(diffs_h))
        pick = int(idxs[pick_local])

        pred = yhat_s[pick, :H]
        tru = y_s[pick, :H]

        plt.figure(figsize=(10, 4))
        plt.plot(tru, label="truth")
        plt.plot(pred, label="forecast")
        plt.title(f"Station {station_id} | {year}-{m:02d}-01 старт | t0={t0_series.iloc[pick]}")
        plt.xlabel("horizon hour")
        plt.ylabel("temperature (°C)")
        plt.grid(True)
        plt.legend()
        plt.show()
        