"""
Утилиты для инференса.

Использование:
    lat, lon, alt = 52.6031, 39.5708, 143

    # Один раз генерируем и кешируем эмбеддинг станции
    z = generate_z_for_station(lat, lon, alt, "embedder_model.pt")

    # Загружаем модель и скейлер
    model, scaler = load_model_and_scaler("edge_weather_model.pt")

    # Строим прогноз по DataFrame с последними 672 часами
    pred = generate_prediction(df, lat, lon, alt, z, model, scaler)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data.scaler import StandardScalerNP
from data.graph import build_node_features_all, build_knn_edge_index, build_edge_weight
from data.features import process_data, solar_datetime_and_geometry
from models.gnn import StationGNN
from models.student import StudentModel


COLS_TO_SCALE = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure", "dewpoint"
]


def generate_z_for_station(
    lat: float, lon: float, alt: float, path: str
) -> torch.Tensor:
    """
    Генерирует GNN-эмбеддинг для новой станции.
    """
    artifacts = torch.load(path, map_location="cpu", weights_only=False)
    gnn_state = artifacts["gnn"]

    geo_points = list(artifacts["train_geo_points"])
    geo_points.append({"name": "New", "lat": lat, "lng": lon, "alt": alt})

    node_features = build_node_features_all(geo_points)
    edge_index, dist_km = build_knn_edge_index(geo_points, k=4, undirected=True)
    edge_weight = build_edge_weight(edge_index, dist_km, tau_km=250.0)

    gnn = StationGNN(
        node_in_dim=node_features.shape[1],
        gnn_hidden=64,
        emb_dim=32,
        dropout=0.1,
    )
    gnn.load_state_dict(gnn_state, strict=True)
    gnn.eval()

    with torch.no_grad():
        output = gnn(node_features, edge_index, edge_weight)

    return output[-1]  # эмбеддинг последнего (нового) узла


def load_model_and_scaler(path: str) -> tuple[StudentModel, StandardScalerNP]:
    """Загружает StudentModel2 и скейлер из файла артефактов."""
    artifacts = torch.load(path, map_location="cpu", weights_only=False)
    scaler = StandardScalerNP.from_state_dict(artifacts["scaler"])
    model = StudentModel(
        d_hist=25,
        d_fut_cov=3,
        gnn_emb_dim=32,
        enc_channels=96,
        dec_channels=64,
        n_enc_layers=7,
        n_dec_layers=7,
        kernel_size=3,
        dropout=0.05,
    )
    model.load_state_dict(artifacts["model"])
    model.eval()
    return model, scaler


def make_future_cov(
    *,
    t_last_utc: pd.Timestamp,
    H: int,
    latitude: float,
    longitude: float,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Строит матрицу ковариат будущего [H, 3]:
    (solar_cos_zenith, solar_doy_sin, solar_doy_cos).
    """
    future_time = pd.date_range(
        start=t_last_utc + pd.Timedelta(hours=1),
        periods=H,
        freq="h",
        tz="UTC",
    )
    df_fut = pd.DataFrame({
        "time": future_time,
        "latitude": float(latitude),
        "longitude": float(longitude),
    })
    df_fut = solar_datetime_and_geometry(df_fut)

    solar_doy = df_fut["solar_datetime"].dt.dayofyear.to_numpy(dtype=np.float64)
    ang = 2 * np.pi * (solar_doy / 365.2422)
    df_fut["solar_doy_sin"] = np.sin(ang)
    df_fut["solar_doy_cos"] = np.cos(ang)

    X_fut_cov = df_fut[["solar_cos_zenith", "solar_doy_sin", "solar_doy_cos"]].to_numpy(np.float32)
    return X_fut_cov, future_time


def generate_prediction(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    alt: float,
    z: torch.Tensor,
    model: StudentModel,
    scaler: StandardScalerNP,
    L: int = 672,
    H: int = 168,
) -> pd.DataFrame:
    """
    Генерирует прогноз температуры на H часов вперёд.

    Parameters
    ----------
    df : pd.DataFrame
        Должен содержать колонки: time, temperature_2m,
        relative_humidity_2m, surface_pressure.
        Время — в UTC, минимум L строк.
    """
    data = df.copy()
    data["latitude"] = lat
    data["longitude"] = lon
    data["altitude"] = alt
    data = process_data(data)

    data = data.iloc[-L:].reset_index(drop=True)
    t_last_utc = pd.to_datetime(df["time"].iat[-1], utc=True)

    data_scaled = data.copy()
    data_scaled[COLS_TO_SCALE] = scaler.transform(data[COLS_TO_SCALE].to_numpy(np.float32))

    feature_cols = COLS_TO_SCALE + ["solar_cos_zenith", "solar_doy_sin", "solar_doy_cos"]
    X_hist = data_scaled[feature_cols].to_numpy(np.float32)

    x_hist   = torch.from_numpy(X_hist).unsqueeze(0)   # [1, L, D]
    z_t      = z.unsqueeze(0)                            # [1, emb_dim]

    X_fut_cov, future_time = make_future_cov(
        t_last_utc=t_last_utc, H=H, latitude=lat, longitude=lon,
    )
    x_fut_cov = torch.from_numpy(X_fut_cov).unsqueeze(0)  # [1, H, 3]

    with torch.no_grad():
        y_hat = model(x_hist, x_fut_cov, z_t)

    return pd.DataFrame({
        "time": future_time,
        "temperature_2m": y_hat.squeeze(0).numpy(),
    })
