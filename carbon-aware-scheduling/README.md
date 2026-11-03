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

## How to run

```bash
# 1. clone and install
git clone https://github.com/amrutakhot16/Intelligent-Carbon-Aware-Scheduling-of-Data-Center-Workloads-Using-Machine-Learning-and-Agentic-AI.git
cd Intelligent-Carbon-Aware-Scheduling-of-Data-Center-Workloads-Using-Machine-Learning-and-Agentic-AI
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. run the pipeline (downloads ~400 MB of data on the first run)
python code/main.py                # all models, fast mode
python code/main.py --full         # larger hyper-parameter searches and full-length neural network training
python code/main.py --skip-rf      # skip Random Forest
python code/main.py --skip-dl      # skip LSTM, GRU and TFT (PyTorch not needed)
python code/main.py --skip-tft     # skip only the TFT (slowest neural network)
python code/main.py --skip-eda     # skip EDA plots
```

Outputs:
- `resources/figures/*.png`: EDA, feature importance, learning curves and forecast plots
- `resources/model_results.csv`: model comparison (train/test MAE, RMSE, R², MAPE)
- `resources/per_region_mae.csv`: test MAE per region and model
- `resources/scheduling_time_shift_savings.csv`, `resources/scheduling_region_time_savings.csv`: CO₂ savings
- `data/models/`: trained models, `data/processed/model_table.parquet`: modelling table

To change the seed, forecast horizon, train/test dates, features, `fast_mode`, the neural network
settings or the scheduling job length, edit **`config.json`** in the repository root. No Python file needs to change.

To use the notebook instead, run `jupyter notebook code/01_preprocessing_and_ml_models.ipynb`.
On Google Colab, set `PROJECT_ROOT` to the repository folder before importing the modules.

## Method

- **Data:** HPE SustainCluster, hourly carbon intensity, weather and electricity prices for 16 grid regions, 2021–2024.
- **Cleaning:** all timestamps converted to UTC, duplicates removed, short gaps interpolated, price spikes clipped
  using training-year percentiles only.
- **Features:** carbon intensity from 24 hours to 1 week earlier, rolling statistics, lagged grid mix and price,
  weather, and local-time calendar features (hour, day of week, month). Every carbon and price feature is at least
  24 hours old, so nothing from the future is used.
- **Split:** chronological, train 2021–2023 and test 2024. Hyper-parameters are tuned with time-series
  cross-validation; XGBoost and the neural networks use October–December 2023 for early stopping.
- **Models:** two naive baselines, Linear Regression, Random Forest, XGBoost, LSTM, GRU and a simplified TFT.
- **Metrics:** MAE, RMSE, R² and MAPE on train and test, plus CO₂ saved in a scheduling simulation.

## Results (test year 2024)

| Model | Train MAE | Test MAE (gCO₂/kWh) | Test RMSE | Test R² | vs persistence |
|---|---|---|---|---|---|
| Naive persistence (24 h) | 28.0 | 29.0 | 48.3 | 0.945 | – |
| Naive same hour last week | 39.0 | 40.1 | 64.2 | 0.903 | −38.5% |
| Linear Regression | 27.0 | 28.2 | 42.0 | 0.959 | +2.8% |
| Random Forest | 14.5 | 24.8 | 39.0 | 0.964 | +14.3% |
| XGBoost | 17.5 | 24.3 | 38.2 | 0.966 | +16.0% |
| LSTM | ? | ? | ? | ? | ? |
| GRU | ? | ? | ? | ? | ? |
| TFT | ? | ? | ? | ? | ? |

See `resources/model_results.csv` for the full table.

### Scheduling simulation

Every day of 2024 a 4-hour job arrives at 00:00 UTC and must finish within 24 hours. Each model picks a start
hour using only its forecast, and is scored on the actual carbon intensity of that window.

| Strategy | Mean CO₂ saving vs running immediately | Share of the best possible saving |
|---|---|---|
| Oracle (perfect knowledge) | 16.9% | 100% |
| GRU | 14.7% | 87% |
| XGBoost | 14.7% | 87% |
| LSTM | 14.6% | 87% |
| TFT | 14.6% | 87% |
| Random Forest* | 14.5% | 86% |
| Linear Regression | 14.3% | 85% |
| Naive persistence | 13.3% | 79% |

\*Random Forest comes from a separate run.

All the trained models give almost the same saving, so small differences in forecast accuracy do not lead to
noticeably different scheduling decisions. The savings are largest in grids with strong daily swings
(Chile 52%, California 47%, Ontario 38%) and close to zero in grids that stay flat during the day (Australia, Korea, Japan).
Allowing the job to move to any region as well cuts CO₂ by about 81% compared with running it immediately in New York.

## Limitations

- Weather at the target hour is the observed value, used as a stand-in for a weather forecast, so results are
  slightly optimistic.
- One model is trained for all 16 regions; errors are largest on wind- and solar-heavy grids (Germany, Texas, Victoria, GB).
- Tuning uses a subsample of the training data and small search budgets in fast mode.
