# edge-weather-forecast
The test and training were conducted in approximately the same climate zone.
### Edge-first neural weather forecasting for local stations
Habr articles: 
- https://habr.com/ru/articles/995120/
- not yet

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
- ~600 KB model size
- ~130k trainable parameters
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
| 0-6h     | 0.83  | 0.88   |
| 6-24h    | 1.72  | 1.95   |
| 1-3d     | 2.70  | 3.0    |
| 3-7d     | 3.35  | 3.40   |

In addition to MAE, amplitude preservation
(std(pred) / std(true)) is monitored to detect over-smoothing,
which is common in long-horizon forecasting.
**AP = ~0.69**

-------------------------------------------------------------
# Limitations
- The model does not explicitly represent atmospheric dynamics.
- Performance depends on the quality and continuity of station data.
- Extreme events driven by large-scale circulation may be underestimated.
  
-------------------------------------------------------------
# License
Open-source (MIT)

-------------------------------------------------------------
#### Most experimental details and ablation studies are documented in the notebooks.
