import numpy as np
import pandas as pd


def solar_datetime_and_geometry(
    df: pd.DataFrame,
    time_col: str = "time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    prefix: str = "solar_",
) -> pd.DataFrame:
    """
    Вычисляет солнечное время и косинус зенитного угла (sin высоты солнца).

    Добавляет колонки:
      - {prefix}datetime       — солнечное время (naive, без tz)
      - {prefix}cos_zenith     — sin высоты солнца (= cos зенитного угла), [-1, 1]
    """
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

    # Уравнение времени (EoT, минуты)
    B = np.deg2rad((360 / 365) * (N - 81))
    EoT = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    solar_offset_minutes = 4.0 * lon + EoT

    solar_datetime = (
        t + pd.to_timedelta(solar_offset_minutes, unit="m")
    ).dt.tz_localize(None)
    out[prefix + "datetime"] = solar_datetime

    solar_minutes = (utc_minutes + solar_offset_minutes) % 1440
    solar_hours = solar_minutes / 60

    # Склонение солнца
    delta = np.deg2rad(23.45) * np.sin(B)

    # Часовой угол
    H_angle = np.deg2rad(15 * (solar_hours - 12))
    phi = np.deg2rad(lat)

    sin_elev = (
        np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.cos(H_angle)
    )
    out[prefix + "cos_zenith"] = np.clip(sin_elev, -1.0, 1.0)

    return out


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полная подготовка данных для одной станции:
      - солнечная геометрия
      - циклические признаки дня года
      - лаговые разности (1ч, 24ч)
      - скользящие статистики (6ч, 24ч)
      - точка росы

    Ожидаемые входные колонки:
      time, latitude, longitude, altitude,
      temperature_2m, relative_humidity_2m, surface_pressure
    """
    data = df.copy()
    data["time"] = pd.to_datetime(data["time"], utc=True)
    data = solar_datetime_and_geometry(data)

    solar_doy = data["solar_datetime"].dt.dayofyear.to_numpy()
    ang = 2 * np.pi * (solar_doy / 365.2422)
    data["solar_doy_sin"] = np.sin(ang)
    data["solar_doy_cos"] = np.cos(ang)

    data.drop(columns=["latitude", "longitude", "solar_datetime", "altitude"], inplace=True)

    base_features = ["temperature_2m", "relative_humidity_2m", "surface_pressure"]

    # Лаговые разности
    for feature in base_features:
        for lag in [1, 24]:
            data[f"{feature}_diff_{lag}"] = data[feature] - data[feature].shift(lag)

    # Скользящие статистики
    for feature in base_features:
        for window in [6, 24]:
            for stat in ["std", "mean"]:
                data[f"{feature}_rolling_{stat}_{window}"] = (
                    data[feature].rolling(window=window, min_periods=window).agg(stat)
                )

    # Точка росы (формула Августа-Роша-Магнуса)
    T = data["temperature_2m"]
    RH = data["relative_humidity_2m"]
    a, b = 17.27, 237.7
    gamma = np.log(RH / 100.0) + (a * T) / (b + T)
    data["dewpoint"] = (b * gamma) / (a - gamma)

    # Отбрасываем первые 24 строки (NaN из лагов/скользящих)
    data = data.iloc[24:].reset_index(drop=True)
    return data
