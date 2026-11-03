# Intelligent Carbon-Aware Scheduling of Data Center Workloads Using Machine Learning and Agentic AI

Build an intelligent system that predicts future electricity carbon intensity and schedules data-center workloads during lower-carbon periods. The objective is to reduce estimated CO₂ emissions while meeting workload deadlines.

We use environmental data from HPE SustainCluster, including historical carbon intensity, weather, and electricity prices. Machine learning models predict carbon intensity 24 hours ahead, and a scheduling simulation uses these predictions to identify lower-carbon execution times.

#### Algorithms:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. LSTM
5. GRU
6. Temporal Fusion Transformer

#### Validation & Evaluation:

We compare the models on the same chronological training and testing periods using:

1. Mean Absolute Error (MAE)
2. Root Mean Squared Error (RMSE)
3. R-squared (R²)
4. Actual vs. predicted carbon-intensity plots
5. Per-region prediction errors
6. Training and prediction time (planned)

For carbon-aware scheduling, we evaluate:

1. Estimated CO₂ emissions reduction (%)
2. Deadline success rate (100% by design: only start times that finish before the deadline are allowed)
3. Average workload delay (planned)
4. Comparison with immediate execution and other feasible scheduling strategies

## Abstract

Data centers use a lot of electricity, and the carbon cost of that electricity changes by hour and by region.
Flexible workloads such as ML training jobs can be moved to cleaner hours or cleaner regions.
This project forecasts the hourly carbon intensity (gCO₂eq/kWh) of 16 electricity grids 24 hours ahead,
using hourly carbon, weather and electricity-price data from 2021–2024 (HPE SustainCluster).
We compare naive baselines, traditional models (Linear Regression, Random Forest, XGBoost) and neural network
models (LSTM, GRU and a simplified Temporal Fusion Transformer) on a chronological split (train 2021–2023, test 2024).
A scheduler then uses the forecasts to pick the lowest-carbon start time and region that still meets the job's
deadline. The best models (TFT and GRU) cut the test error by about 18% compared with the persistence baseline,
and in simulation the forecasts capture about 87% of the achievable time-shifting saving.

## Team

- Amruta Kirtikumar Khot
- Anmol Ratan Tirkey
- Cazal Tirkey

## Repository structure

```
├── config.json                    # all settings: paths, seed, horizon, split dates, features, model settings
├── code/
│   ├── main.py                    # runs the whole pipeline end to end
│   ├── settings.py                # reads config.json for the other modules
│   ├── data_loading.py            # download and load the raw data
│   ├── eda.py                     # exploratory analysis and figures
│   ├── preprocessing.py           # cleaning, UTC alignment, merging, missing values
│   ├── features.py                # feature engineering, chronological split, encoding/scaling
│   ├── models/
│   │   ├── baselines.py           # naive persistence baselines
│   │   ├── linear_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── dl_common.py           # shared code for the neural networks (input windows, training loop)
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   └── tft_model.py           # simplified Temporal Fusion Transformer
│   ├── evaluation.py              # metrics, comparison table, plots
│   ├── scheduling.py              # CO₂-saving simulation
│   └── 01_preprocessing_and_ml_models.ipynb   # notebook version of the classical-model part, with commentary
├── data/                          # datasets and trained models (see data/README.md)
├── resources/                     # figures, result tables, proposal, references
├── requirements.txt
└── README.md
```
