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
