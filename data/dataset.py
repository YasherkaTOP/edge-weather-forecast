from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .scaler import StandardScalerNP


@dataclass
class WindowSpec:
    """Спецификация одного обучающего окна."""
    station_id: int
    end_pos: int


class MultiStationWindowDataset(Dataset):
    """
    Датасет скользящих окон для нескольких метеостанций.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        L: int,
        H: int,
        hist_num_cols: List[str],
        hist_time_cols: List[str],
        fut_cov_cols: List[str],
        target_col: str,
        static_by_station: Dict[int, np.ndarray],
        hist_num_scaler: Optional[StandardScalerNP],
        mode: str,
        test_start: pd.Timestamp,
        val_days: int = 90,
        time_col: str = "time",
        station_col: str = "station_id",
        allowed_station_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        assert mode in ("train", "val", "test"), f"mode должен быть train/val/test, получено: {mode}"
        self.L = L
        self.H = H
        self.hist_num_cols = hist_num_cols
        self.hist_time_cols = hist_time_cols
        self.fut_cov_cols = fut_cov_cols
        self.target_col = target_col
        self.static_by_station = static_by_station
        self.hist_num_scaler = hist_num_scaler
        self.mode = mode

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.sort_values([station_col, time_col]).reset_index(drop=True)

        test_start_ts = pd.to_datetime(test_start, utc=True)
        val_start_ts = test_start_ts - pd.Timedelta(days=val_days)

        self.test_start = np.datetime64(test_start_ts.to_datetime64())
        self.val_start = np.datetime64(val_start_ts.to_datetime64())

        # Построение рядов по станциям
        self.series: Dict[int, dict] = {}
        for sid, g in df.groupby(station_col, sort=True):
            sid = int(sid)
            if allowed_station_ids is not None and sid not in allowed_station_ids:
                continue

            times = g[time_col].values.astype("datetime64[ns]")
            X_num = g[hist_num_cols].to_numpy(np.float32)
            X_time = g[hist_time_cols].to_numpy(np.float32)
            X_cov = g[fut_cov_cols].to_numpy(np.float32)
            y = g[target_col].to_numpy(np.float32)

            if self.hist_num_scaler is not None:
                X_num = self.hist_num_scaler.transform(X_num)

            X_hist = np.concatenate([X_num, X_time], axis=1).astype(np.float32)

            self.series[sid] = {
                "time": times,
                "X_hist": X_hist,
                "X_cov": X_cov,
                "y": y,
            }

        # Построение окон
        self.windows: List[WindowSpec] = []
        for sid, d in self.series.items():
            times = d["time"]
            T = len(times)

            # Маска часовых разрывов (True = разрыв перед этим индексом)
            gap_mask: Optional[np.ndarray] = None
            if T > 1:
                dt_hours = (times[1:] - times[:-1]) / np.timedelta64(1, "h")
                gap_mask = (dt_hours != 1)

            for end_pos in range(L - 1, T - H - 1):
                t0 = times[end_pos]

                # Фильтрация по режиму
                if mode == "train" and not (t0 < self.val_start):
                    continue
                if mode == "val" and not (self.val_start <= t0 < self.test_start):
                    continue
                if mode == "test" and not (t0 >= self.test_start):
                    continue

                # Проверка целостности временного ряда
                if gap_mask is not None:
                    start = end_pos - (L - 1)
                    if gap_mask[start : end_pos + H].any():
                        continue

                self.windows.append(WindowSpec(sid, end_pos))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        sid = w.station_id
        end_pos = w.end_pos
        d = self.series[sid]

        start = end_pos - (self.L - 1)
        x_hist = d["X_hist"][start : end_pos + 1]
        x_fut_cov = d["X_cov"][end_pos + 1 : end_pos + 1 + self.H]
        y_fut = d["y"][end_pos + 1 : end_pos + 1 + self.H]
        raw_static = self.static_by_station[sid].astype(np.float32)

        t0_h = d["time"][end_pos].astype("datetime64[h]").astype(np.int64)

        return {
            "x_hist":      torch.from_numpy(x_hist),
            "x_fut_cov":   torch.from_numpy(x_fut_cov),
            "y":           torch.from_numpy(y_fut),
            "station_id":  torch.tensor(sid, dtype=torch.long),
            "x_raw_static": torch.from_numpy(raw_static),
            "t0":          torch.tensor(t0_h, dtype=torch.long),
        }


def make_loader(
    ds: MultiStationWindowDataset,
    batch_size: int = 56,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Создаёт DataLoader из датасета."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def fit_scalers(
    df: pd.DataFrame,
    *,
    test_start: pd.Timestamp,
    val_days: int,
    hist_num_cols: List[str],
    time_col: str = "time",
) -> StandardScalerNP:
    """Обучает скейлер только на тренировочных данных (до val_start)."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    test_start = pd.to_datetime(test_start, utc=True)
    val_start = test_start - pd.Timedelta(days=val_days)

    df_train = df[df[time_col] < val_start]
    return StandardScalerNP().fit(df_train[hist_num_cols].to_numpy(np.float32))
