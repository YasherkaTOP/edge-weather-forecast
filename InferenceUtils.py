import numpy as np
from DataUtils import StandardScalerNP, build_node_features_all, build_knn_edge_index, build_edge_weight, process_data, solar_datetime_and_geometry
from ModelModules import StudentModel, StationGNN
import torch
import pandas as pd

COLS_TO_SCALE =  ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'temperature_2m_diff_1', 'temperature_2m_diff_24', 
                 'relative_humidity_2m_diff_1', 'relative_humidity_2m_diff_24', 'surface_pressure_diff_1', 'surface_pressure_diff_24', 
                 'temperature_2m_rolling_std_6', 'temperature_2m_rolling_mean_6', 'temperature_2m_rolling_std_24', 'temperature_2m_rolling_mean_24', 
                 'relative_humidity_2m_rolling_std_6', 'relative_humidity_2m_rolling_mean_6', 'relative_humidity_2m_rolling_std_24', 'relative_humidity_2m_rolling_mean_24', 
                 'surface_pressure_rolling_std_6', 'surface_pressure_rolling_mean_6', 'surface_pressure_rolling_std_24', 'surface_pressure_rolling_mean_24', 'dewpoint']

def generate_z_for_station(lat: float, lon: float, alt: float, path: str):
    '''Генерирует эмбеддинг новой станции'''
    artifacts = torch.load(path, map_location='cpu', weights_only=False)
    gnn_state = artifacts['gnn']
    geo_points = artifacts['train_geo_points']
    geo_points = np.append(geo_points, {'name': 'New', 'lat': lat, 'lng': lon, 'alt': alt})
    node_features = build_node_features_all(geo_points)
    edge_index, dist_km = build_knn_edge_index(geo_points, k=4, undirected=True)
    edge_weight = build_edge_weight(edge_index, dist_km, tau_km=250.0)

    gnn = StationGNN(node_in_dim=node_features.shape[1], gnn_hidden=64, emb_dim=32, dropout=0.1)
    gnn.load_state_dict(gnn_state, strict=True)
    output = gnn(node_features, edge_index, edge_weight)
    return output[-1]

def load_model_and_scaler(path: str):
    '''Загружает модель и скалер'''
    artifacts = torch.load(path, map_location='cpu', weights_only=False)
    scaler = StandardScalerNP.from_state_dict(artifacts['scaler'])
    model = StudentModel(
        d_hist=25,
        d_fut_cov=3,
        gnn_emb_dim=32,
        enc_channels=96,
        dec_channels=64,
        n_enc_layers=7,
        n_dec_layers=7,
        kernel_size=3,
        dropout=0.05)
    model.load_state_dict(artifacts['model'])
    model.eval()
    return model, scaler

def make_future_cov_from_time_latlon(
    *,
    t_last_utc: pd.Timestamp,
    H: int,
    latitude: float,
    longitude: float,
) -> np.ndarray:
    """
    Возвращает X_fut_cov: [H, D_cov] float32
    """
    future_time = pd.date_range(
        start=t_last_utc + pd.Timedelta(hours=1),
        periods=H,
        freq="H",
        tz="UTC",
    )

    df_fut = pd.DataFrame({
        "time": future_time,
        "latitude": float(latitude),
        "longitude": float(longitude),
    })

    df_fut = solar_datetime_and_geometry(df_fut)

    solar_dt = df_fut["solar_datetime"]
    solar_doy = solar_dt.dt.dayofyear.to_numpy(dtype=np.float64)
    year_len = 365.2422
    ang = 2 * np.pi * (solar_doy / year_len)
    df_fut["solar_doy_sin"] = np.sin(ang)
    df_fut["solar_doy_cos"] = np.cos(ang)

    X_fut_cov = df_fut[['solar_cos_zenith', 'solar_doy_sin', 'solar_doy_cos']].to_numpy(np.float32)
    return X_fut_cov, future_time

def generate_prediction(df: pd.DataFrame, lat: float, lon: float, alt: float, z: torch.Tensor, model: StudentModel, scaler: StandardScalerNP, L: int=672, H: int=168):
    '''Генерация предсказания для конкретной тчоки по 28-дневной истории'''
    data = df.copy()
    data = data.loc[-L:]
    t_last_utc = data['time'].iat[-1]
    data['latitude'] = lat
    data['longitude'] = lon
    data['altitude'] = alt
    data = process_data(data)
    data = data.drop(columns=['time'])
    data.loc[:, COLS_TO_SCALE] = scaler.transform(data.loc[:, COLS_TO_SCALE])
    data = data[COLS_TO_SCALE+['solar_cos_zenith', 'solar_doy_sin', 'solar_doy_cos']]
    X_hist = data.to_numpy(dtype=np.float32)
    x_hist = torch.from_numpy(X_hist).unsqueeze(0)
    z_t = z.unsqueeze(0)
    X_fut_cov, future_time = make_future_cov_from_time_latlon(
        t_last_utc=t_last_utc,
        H=H,
        latitude=lat,
        longitude=lon,
    )
    print(X_fut_cov)
    x_fut_cov = torch.from_numpy(X_fut_cov).unsqueeze(0)  
     

    with torch.no_grad():
        y_hat = model(x_hist, x_fut_cov, z_t)

    y_hat_np = y_hat.detach().numpy()[0]
    
    return pd.DataFrame(
        {
            "time": future_time,
            "temperature_2m": y_hat_np
        }
    )

# --- КАК ВЫГЛЯДИТ ИНФЕРЕНС ---

# 1) Заполняем метаданные
# lat=52.6031 
# lon=39.5708
# alt=143

# 2) Грузим датафрейм. 
# ВНИМАНИЕ!: Датафрейм должен содержать такие колонки за последние полные 672 часа на момент предсказания.
# - time (Время в !!!UTC!!!. Пример: 2018-01-01 02:00:00+00:00) 
# - temperature_2m (температура воздуха в °C. Пример: -0.25)
# - relative_humidity_2m (относительная влажность воздуха в %. Пример: 97.85404)
# - surface_pressure (давление над поверхностью земли в hPa. Пример: 986.15814)
# df = pd.read_csv('data/Lipetsk.csv')
# df['time'] = pd.to_datetime(df['time'])

# 3) Генерируем эмбеддинг станции (Можно сгенерировать 1 раз и сохранить для переиспользования).
# ВНИМАНИЕ: GNN и метаданные упакованы в единый файл!!! Для замены файла на свой смотреть как выглядит исходный!!!
# z = generate_z_for_station(lat, lon, alt, 'embedder_model.pt')

# 4) Загружаем модель и скалер
# ВНИМАНИЕ: Модель и скалер упакованы в единый файл!!! Для замены файла на свой смотреть как выглядит исходный!!!
# model, scaler = load_model_and_scaler('edge_weather_model.pt')

# 5) Генерируем предсказания
# pred = generate_prediction(df, lat, lon, alt, z, model, scaler)

# 6) Можем визуализировать для наглядности
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 4))
# plt.plot(pred["time"], pred["temperature_2m"])
# plt.xlabel("Time")
# plt.ylabel("Temperature (2m)")
# plt.title("Temperature over time")
# plt.grid(True)
# plt.tight_layout()
# plt.show()