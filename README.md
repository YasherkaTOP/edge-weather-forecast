# edge-weather-prediction (python 3.11)

### Edge-first neural weather forecasting for local stations
-------------------------------------------------------------
# Overview 

This repository contains a lightweight neural model for forecasting
air temperature at individual weather stations with hourly resolution
up to 168 hours (7 days) ahead.
The system is designed for **offline inference**, **CPU-only execution**,
and **predictable runtime**, without relying on global numerical weather
prediction (NWP) models, reanalysis data, or external APIs.

-------------------------------------------------------------
# Key properties
- ~2 MB model size
- ~600k trainable parameters
- CPU-only inference
- Fully offline
- Hourly resolution, 168h forecast horizon
- Generalization to unseen stations
- Open-source and reproducible
-------------------------------------------------------------
# Inputs/Outputs
### Inputs
The model relies exclusevily on station-level data:
- Air temperature
- Surface pressure
- Relative humidity
- UTC time
  
Context horizon: 672 hours

No gridded fields, reanalysis data, or NWP inputs are used.

### Outputs
- Absolute air temperature (°C)
  
Forecast horizon: 168 hours

The model predicts the full horizon in a single forward pass.

-------------------------------------------------------------
# Evaluation
Model performance is evaluated using Mean Absolute Error (MAE),
reported separately for different forecast horizons.

**Horizon	MAE (°C)** on seen/unseen stations
| Horizon  | MAE (°C) seen | MAE (°C) unseen |
|----------|-------|--------|
| 0-6h     | 0.8   | 0.85   |
| 6-24h    | 1.75  | 1.8    |
| 1-3d     | 2.95  | 3.0    |
| 3-7d     | 3.85  | 3.85   |

In addition to MAE, amplitude preservation
(std(pred) / std(true)) is monitored to detect over-smoothing,
which is common in long-horizon forecasting.
**AP = ~0.79**

-------------------------------------------------------------
# Limitations
- The model does not explicitly represent atmospheric dynamics.
- Performance depends on the quality and continuity of station data.
- Extreme events driven by large-scale circulation may be underestimated.
  
-------------------------------------------------------------
# License
Open-source (MIT)

-------------------------------------------------------------
# Repository structure

FullResearch.ipynb - General research, training/testing models

DataUtils.py - Scaler, custom dataset, some functions for data preparing

InferenceUtils.py - Inference functions and **Inference GUIDE**

ModelModules.py - Different model modules/layers and EMA

Callbacks - Custom callbacks for training

ModelUtils.py - Custom loss

TestUtils.py - Some functions for model testing

embedder_model.pt - Contains GNN-embedder and metadata

edge_weather_model.pt - Contains Scaler and Model

#### Most experimental details and ablation studies are documented in the notebooks.
