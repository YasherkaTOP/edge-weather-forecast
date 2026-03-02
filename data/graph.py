import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние по формуле Хаверсина между двумя точками (в км)."""
    R = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2 - lat1))
    dl   = math.radians(float(lon2 - lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_node_features_all(
    geo_points: List[dict], alt_scale_m: float = 1000.0
) -> torch.Tensor:
    """
    Строит матрицу признаков узлов [N, 4]:
    (cos_lat·cos_lon, cos_lat·sin_lon, sin_lat, alt_norm).
    """
    lats = np.array([p["lat"] for p in geo_points], dtype=np.float32)
    lons = np.array([p["lng"] for p in geo_points], dtype=np.float32)
    alts = np.array([p["alt"] for p in geo_points], dtype=np.float32)

    lat_r = np.deg2rad(lats)
    lon_r = np.deg2rad(lons)

    X = np.stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
        alts / float(alt_scale_m),
    ], axis=1).astype(np.float32)
    return torch.tensor(X, dtype=torch.float32)


def build_dist_matrix_km(geo_points: List[dict]) -> np.ndarray:
    """Матрица попарных расстояний [N, N] в км."""
    N = len(geo_points)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = haversine_km(
                geo_points[i]["lat"], geo_points[i]["lng"],
                geo_points[j]["lat"], geo_points[j]["lng"],
            )
            D[i, j] = D[j, i] = d
    return D


def build_knn_edge_index(
    geo_points: List[dict], k: int = 4, undirected: bool = True
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Строит edge_index для k-ближайших соседей.

    Returns
    -------
    edge_index : [2, E]
    dist_km    : [N, N] матрица расстояний
    """
    dist = build_dist_matrix_km(geo_points)
    N = dist.shape[0]

    edge_set: set[tuple[int, int]] = set()
    for i in range(N):
        nn_sorted = np.argsort(dist[i])
        neighbors = [int(j) for j in nn_sorted if j != i][:k]
        for j in neighbors:
            edge_set.add((i, j))
            if undirected:
                edge_set.add((j, i))

    edges = sorted(edge_set)  # детерминированный порядок
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, dist


def build_edge_weight(
    edge_index: torch.Tensor, dist_km: np.ndarray, tau_km: float = 200.0
) -> torch.Tensor:
    """Веса рёбер: exp(-dist / tau)."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    w = np.exp(-dist_km[src, dst] / float(tau_km)).astype(np.float32)
    return torch.tensor(w, dtype=torch.float32)


def summarize_graph(
    geo_points: List[dict], edge_index: torch.Tensor, dist_km: np.ndarray, k_show: int = 4
) -> None:
    """Выводит степени вершин и ближайших соседей для диагностики графа."""
    N = len(geo_points)
    deg_out = np.zeros(N, dtype=np.int32)
    for s in edge_index[0].cpu().numpy():
        deg_out[int(s)] += 1

    print("Out-degrees:")
    for i, p in enumerate(geo_points):
        print(f"  {i:02d} {p['name']}: {deg_out[i]}")

    print("\nNearest neighbors by distance:")
    for i, p in enumerate(geo_points):
        nn = np.argsort(dist_km[i])
        nn = [int(j) for j in nn if j != i][:k_show]
        nn_str = ", ".join(f"{geo_points[j]['name']}({dist_km[i, j]:.0f}km)" for j in nn)
        print(f"  {i:02d} {p['name']}: {nn_str}")

def plot_station_graph(
    geo_points,
    edge_index,
    edge_weight=None,
    title="Station graph (lat/lon)",
    min_lw=0.5,
    max_lw=4.0,
):
    '''Визуализация графа на карте с помощью matplotlib. Узлы отображаются в виде точек, рёбра - в виде линий. Если edge_weight задан, то толщина линий пропорциональна весу ребра'''
    lats = np.array([p["lat"] for p in geo_points], dtype=float)
    lons = np.array([p["lng"] for p in geo_points], dtype=float)
    names = [p["name"] for p in geo_points]

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    if edge_weight is not None:
        w = edge_weight.detach().cpu().numpy().astype(float)

        # нормализация весов в диапазон [min_lw, max_lw]
        if np.allclose(w.max(), w.min()):
            lw = np.full_like(w, (min_lw + max_lw) / 2)
        else:
            w_norm = (w - w.min()) / (w.max() - w.min())
            lw = min_lw + w_norm * (max_lw - min_lw)
    else:
        lw = np.ones(len(src))

    plt.figure(figsize=(10, 8))

    # рёбра
    for i, (s, d) in enumerate(zip(src, dst)):
        plt.plot(
            [lons[s], lons[d]],
            [lats[s], lats[d]],
            linewidth=lw[i],
            alpha=0.7,
            color="tab:blue",
        )

    # узлы
    plt.scatter(lons, lats, s=80, color="black", zorder=3)

    # подписи
    for i, (x, y) in enumerate(zip(lons, lats)):
        plt.text(
            x, y,
            f" {i}:{names[i]}",
            fontsize=9,
            ha="left",
            va="center",
            zorder=4,
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.show()
