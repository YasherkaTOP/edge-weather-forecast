from torch.utils.data import DataLoader, Dataset
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import math

class StandardScalerNP:
    ''' StandardScaler для numpy массивов'''
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.maximum(self.std_, self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def state_dict(self) -> dict:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        return {
            "mean": self.mean_,
            "std": self.std_,
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "StandardScalerNP":
        obj = cls(eps=state.get("eps", 1e-6))
        obj.mean_ = state["mean"]
        obj.std_ = state["std"]
        return obj
    

@dataclass
class WindowSpec:
    ''' Спецификация окна для одной станции'''
    station_id: int
    end_pos: int

class MultiStationWindowDataset(Dataset):
    ''' Датасет с окнами для нескольких станций'''
    def __init__(
        self,
        df: pd.DataFrame, # весь датасет с данными по всем станциям
        *,
        L: int, # длина истории
        H: int, # длина горизонта прогноза
        hist_num_cols: List[str],   # числовые признаки (будут масштабироваться)
        hist_time_cols: List[str],  # time feats (НЕ масштабируем)
        fut_cov_cols: List[str],    # обычно те же time feats
        target_col: str,            # temp
        static_by_station: Dict[int, np.ndarray],  # raw static из node_features_all
        hist_num_scaler: Optional[StandardScalerNP], # скейлер для числовых признаков истории
        mode: str,                  # train/val/test
        test_start: pd.Timestamp, # начало тестового периода
        val_days: int = 90, # длина валидационного периода в днях
        time_col: str = "time", # имя колонки с временной меткой
        station_col: str = "station_id", # имя колонки с id станции
        allowed_station_ids: Optional[List[int]] = None, # если указаны, то только эти станции используются
    ):
        super().__init__()
        assert mode in ("train", "val", "test")
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

        self.test_start = np.datetime64(pd.to_datetime(test_start, utc=True).to_datetime64())
        self.val_start = self.test_start - np.timedelta64(val_days, "D")

        # построение рядов по станциям
        self.series = {}
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
                "X_cov": X_cov.astype(np.float32),
                "y": y.astype(np.float32),
            }

        # построение окон
        self.windows: List[WindowSpec] = []
        for sid, d in self.series.items():
            times = d["time"]
            T = len(times)

            bad_mask = None
            if T > 1:
                dt = (times[1:] - times[:-1]) / np.timedelta64(1, "h")
                bad_mask = (dt != 1)

            for end_pos in range(L - 1, T - H - 1):
                t0 = times[end_pos]

                if mode == "train":
                    if not (t0 < self.val_start):
                        continue
                elif mode == "val":
                    if not (self.val_start <= t0 < self.test_start):
                        continue
                else:  # test
                    if not (t0 >= self.test_start):
                        continue

                if bad_mask is not None:
                    start = end_pos - (L - 1)
                    if bad_mask[start:end_pos].any():
                        continue
                    if bad_mask[end_pos:end_pos + H].any():
                        continue

                self.windows.append(WindowSpec(sid, end_pos))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        sid = w.station_id
        end_pos = w.end_pos

        d = self.series[sid]
        X_hist = d["X_hist"]
        X_cov = d["X_cov"]
        y = d["y"]

        start = end_pos - (self.L - 1)
        x_hist = X_hist[start:end_pos + 1]
        x_fut_cov = X_cov[end_pos + 1:end_pos + 1 + self.H]
        y_fut = y[end_pos + 1:end_pos + 1 + self.H]

        raw_static = self.static_by_station[sid].astype(np.float32)

        t0 = d["time"][end_pos]
        t0_h = t0.astype("datetime64[h]").astype(np.int64)
        return {
            "x_hist": torch.from_numpy(x_hist),
            "x_fut_cov": torch.from_numpy(x_fut_cov),
            "y": torch.from_numpy(y_fut),
            "station_id": torch.tensor(sid, dtype=torch.long),
            "x_raw_static": torch.from_numpy(raw_static),
            "t0": torch.tensor(t0_h, dtype=torch.long)
        }
    

def make_loader(ds, batch_size=56, shuffle=True, num_workers=0):
    ''' Создание DataLoader-а из датасета '''
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
    time_col="time",
):
    ''' Подгонка скейлеров на тренировочных данных '''
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    test_start = pd.to_datetime(test_start, utc=True)
    val_start = test_start - pd.Timedelta(days=val_days)

    # отбор только тренировочных данных
    df_train = df[df[time_col] < val_start]

    hist_scaler = StandardScalerNP().fit(df_train[hist_num_cols].to_numpy(np.float32))

    return hist_scaler

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float):
    '''Вычисляет расстояние в километрах между двумя точками на поверхности Земли, заданными их широтой и долготой'''
    R = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2 - lat1))
    dl = math.radians(float(lon2 - lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def build_node_features_all(geo_points, alt_scale_m=1000.0):
    '''Построение признаков для всех узлов графа на основе их географических координат и высоты над уровнем моря'''
    lats = np.array([p["lat"] for p in geo_points], dtype=np.float32)
    lons = np.array([p["lng"] for p in geo_points], dtype=np.float32)
    alts = np.array([p["alt"] for p in geo_points], dtype=np.float32)

    lat_r = np.deg2rad(lats)
    lon_r = np.deg2rad(lons)

    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)

    alt_norm = alts / float(alt_scale_m)

    X = np.stack([x, y, z, alt_norm], axis=1).astype(np.float32)
    return torch.tensor(X, dtype=torch.float32)

def build_dist_matrix_km(geo_points):
    '''Построение матрицы попарных расстояний между геоточками в километрах'''
    N = len(geo_points)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            D[i, j] = haversine_km(
                geo_points[i]["lat"], geo_points[i]["lng"],
                geo_points[j]["lat"], geo_points[j]["lng"],
            )
    return D

def build_knn_edge_index(geo_points, k=4, undirected=True):
    ''''Построение индекса рёбер для k-ближайших соседей на основе географических расстояний между точками. Если undirected=True, то добавляем рёбра в обоих направлениях'''
    dist = build_dist_matrix_km(geo_points)
    N = dist.shape[0]

    edges = []
    for i in range(N):
        nn = np.argsort(dist[i])  # includes i at start
        nn = [int(j) for j in nn if j != i][:k]
        for j in nn:
            edges.append((i, j))
            if undirected:
                edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, dist

def build_edge_weight(edge_index, dist_km, tau_km=200.0):
    '''Построение весов рёбер на основе расстояний между точками. Вес пропорционален exp(-dist/tau)'''
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    d = dist_km[src, dst]
    w = np.exp(-d / float(tau_km)).astype(np.float32)
    return torch.tensor(w, dtype=torch.float32)

def summarize_graph(geo_points, edge_index, dist_km, k_show=4):
    '''Вывод информации о графе: степени вершин и ближайшие соседи по расстоянию'''
    N = len(geo_points)
    deg_out = np.zeros(N, dtype=np.int32)
    src = edge_index[0].cpu().numpy()
    for s in src:
        deg_out[int(s)] += 1

    print("Out-degrees (should be ~k or ~2k depending on undirected):")
    for i, p in enumerate(geo_points):
        print(f"  {i:02d} {p['name']}: {deg_out[i]}")

    print("\nNearest neighbors by distance:")
    for i, p in enumerate(geo_points):
        nn = np.argsort(dist_km[i])
        nn = [int(j) for j in nn if j != i][:k_show]
        nn_str = ", ".join([f"{geo_points[j]['name']}({dist_km[i,j]:.0f}km)" for j in nn])
        print(f"  {i:02d} {p['name']}: {nn_str}")

def solar_datetime_and_geometry(
    df: pd.DataFrame,
    time_col="time",
    lat_col="latitude",
    lon_col="longitude",
    prefix="solar_",
):
    '''Вычисляет солнечное время и геометрию солнца'''
    out = df.copy()

    t = pd.to_datetime(out[time_col])
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC")
    else:
        t = t.dt.tz_convert("UTC")

    lat = out[lat_col].to_numpy(dtype=np.float64)
    lon = out[lon_col].to_numpy(dtype=np.float64)

    N = t.dt.dayofyear.to_numpy(dtype=np.float64)

    utc_minutes = (
        t.dt.hour.to_numpy(dtype=float) * 60
        + t.dt.minute.to_numpy(dtype=float)
        + t.dt.second.to_numpy(dtype=float) / 60
        + t.dt.microsecond.to_numpy(dtype=float) / 60_000_000
    )


    B = np.deg2rad((360 / 365) * (N - 81))
    EoT = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B) 

    solar_offset_minutes = 4.0 * lon + EoT

    solar_datetime = (
        t + pd.to_timedelta(solar_offset_minutes, unit="m")
    ).dt.tz_localize(None)

    out[prefix + "datetime"] = solar_datetime

    solar_minutes = (utc_minutes + solar_offset_minutes) % 1440
    solar_hours = solar_minutes / 60

    delta = np.deg2rad(23.45) * np.sin(B)

    H = np.deg2rad(15 * (solar_hours - 12))
    phi = np.deg2rad(lat)

    sin_h = (
        np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.cos(H)
    )
    sin_h = np.clip(sin_h, -1.0, 1.0)

    out[prefix + "cos_zenith"] = sin_h 

    return out

def process_data(df):
    '''Основная функция для подготовки данных для одного региона'''
    data = df.copy()
    data['time'] = pd.to_datetime(data['time'], utc=True)
    data = solar_datetime_and_geometry(data)
    solar_dt = data["solar_datetime"]

    solar_doy = solar_dt.dt.dayofyear

    doy = solar_doy.to_numpy()
    year_len = 365.2422
    ang = 2*np.pi * (doy / year_len)
    data["solar_doy_sin"] = np.sin(ang)
    data["solar_doy_cos"] = np.cos(ang)

    data.drop(columns=['latitude', 'longitude', 'solar_datetime', 'altitude'], inplace=True)

    base_features = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure']
    lags = [1, 24]
    windows = [6, 24]
    stats = ['std', 'mean']

    for feature in base_features:
        for lag in lags:
            col_name = f'{feature}_diff_{lag}'
            data[col_name] = data[feature] - data[feature].shift(lag)
    
    for feature in base_features:
        for window in windows:
            for stat in stats:
                col_name = f'{feature}_rolling_{stat}_{window}'
                data[col_name] = data[feature].rolling(window=window, min_periods=window).agg(stat)

    a = 17.27
    b = 237.7
    
    T = data['temperature_2m']
    RH = data['relative_humidity_2m']
    
    gamma = np.log(RH / 100) + (a * T) / (b + T)
    dewpoint = (b * gamma) / (a - gamma)
    
    data['dewpoint'] = dewpoint
    
    data = data.iloc[24:].reset_index(drop=True)
    return data