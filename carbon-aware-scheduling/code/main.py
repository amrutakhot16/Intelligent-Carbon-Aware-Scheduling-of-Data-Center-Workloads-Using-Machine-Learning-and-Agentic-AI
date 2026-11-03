"""
Run the whole pipeline end to end.

    python code/main.py                 # fast mode (small searches / fewer epochs, minutes on a laptop)
    python code/main.py --full          # bigger searches, full-length deep-learning training
    python code/main.py --skip-eda      # skip the EDA plots
    python code/main.py --skip-rf       # skip Random Forest (slowest classic model)
    python code/main.py --skip-dl       # skip LSTM, GRU and TFT (no PyTorch needed)
    python code/main.py --skip-tft      # train LSTM and GRU only (TFT is the slowest deep model)

Outputs: data/processed/model_table.parquet, data/models/*, resources/figures/*.png, resources/*.csv
"""
import argparse
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")          # save figures to files, never open windows

import joblib
import pandas as pd

import settings
from settings import CAT_FEATURES, HORIZON, MODEL_DIR, NUM_FEATURES, PROC_DIR, TARGET
from data_loading import download_data, load_all
from eda import plot_feature_correlation, run_eda
from evaluation import comparison_table, evaluate, per_region_mae, plot_forecast_examples, plot_model_comparison
from features import build_model_table, chronological_split, get_xy
from models import linear_regression, random_forest, xgboost_model
from models.baselines import predict_baselines
from preprocessing import preprocess
from scheduling import build_cubes, example_recommendation, spatio_temporal_savings, time_shift_savings

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160, "display.max_columns", 40)

ALL_MODELS = ["Linear Regression", "Random Forest", "XGBoost", "LSTM", "GRU", "TFT"]


def section(title: str) -> None:
    """
    Print a header line so the console output is easy to follow.

    :param title: section title
    :return: None
    """
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def save_artifacts(model_df, models: dict, region_tz: dict) -> None:
    """
    Save the modelling table, the trained classical models and the feature settings
    they were trained with, so they can be reloaded later without retraining.

    :param model_df: final modelling table
    :param models: {model name: fitted model}; XGBoost is stored as (preprocessor, model)
    :param region_tz: {region: timezone name}
    :return: None
    """
    model_df.to_parquet(PROC_DIR / "model_table.parquet", index=False)
    if "Linear Regression" in models:
        joblib.dump(models["Linear Regression"], MODEL_DIR / "linear_regression.joblib")
    if "Random Forest" in models:
        joblib.dump(models["Random Forest"], MODEL_DIR / "random_forest.joblib", compress=3)
    if "XGBoost" in models:
        prep, booster = models["XGBoost"]
        joblib.dump(prep, MODEL_DIR / "xgb_preprocessor.joblib")
        booster.get_booster().save_model(MODEL_DIR / "xgboost.json")
    with open(MODEL_DIR / "feature_config.json", "w") as f:
        json.dump({"NUM_FEATURES": NUM_FEATURES, "CAT_FEATURES": CAT_FEATURES,
                   "HORIZON": HORIZON, "REGION_TZ": region_tz}, f, indent=2)
    print("Saved:", sorted(p.name for p in MODEL_DIR.iterdir()), "+", sorted(p.name for p in PROC_DIR.iterdir()))


def main(args) -> None:
    """
    Run every step of the pipeline in order.

    :param args: parsed command-line arguments (see the bottom of this file)
    :return: None
    """
    t_start = time.time()
    fast = settings.FAST_MODE and not args.full
    settings.make_dirs()

    section("1. Load raw data")
    download_data()
    raw = load_all()

    if not args.skip_eda:
        section("2. Exploratory data analysis")
        run_eda(raw)

    section("3-4. Clean, merge, impute")
    df = preprocess(raw)

    section("5. Feature engineering")
    model_df = build_model_table(df, raw["region_tz"])
    if not args.skip_eda:
        plot_feature_correlation(model_df, NUM_FEATURES, TARGET)

    section("6. Chronological train/test split")
    train, test = chronological_split(model_df)
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)
    sub_step = 8 if fast else 3                      # every k-th row keeps time order (tuning subsample)
    Xs, ys = X_train.iloc[::sub_step], y_train.iloc[::sub_step]

    results, preds, fitted = [], {}, {}

    section("7. Naive baselines")
    for name, (p_tr, p_te) in predict_baselines(train, test).items():
        results.append(evaluate(name, y_train, p_tr, y_test, p_te)); preds[name] = p_te

    section("8. Linear Regression")
    lr, p_tr, p_te = linear_regression.train_linear_regression(X_train, y_train, X_test)
    results.append(evaluate("Linear Regression", y_train, p_tr, y_test, p_te)); preds["Linear Regression"] = p_te
    fitted["Linear Regression"] = lr
    linear_regression.plot_coefficients(lr)

    if not args.skip_rf:
        section("9. Random Forest")
        best_rf = random_forest.tune_random_forest(Xs, ys, fast_mode=fast)
        rf, p_tr, p_te = random_forest.train_random_forest(X_train, y_train, X_test, best_rf)
        results.append(evaluate("Random Forest", y_train, p_tr, y_test, p_te)); preds["Random Forest"] = p_te
        fitted["Random Forest"] = rf
        random_forest.plot_importance(rf)

    section("10. XGBoost")
    prep, xgb_final, p_tr, p_te, info = xgboost_model.train_xgboost(train, X_train, y_train, X_test, Xs, ys, fast_mode=fast)
    results.append(evaluate("XGBoost", y_train, p_tr, y_test, p_te)); preds["XGBoost"] = p_te
    fitted["XGBoost"] = (prep, xgb_final)
    xgboost_model.plot_curve_and_importance(xgb_final, info)

    dl_fitted, seq = {}, None
    if not args.skip_dl:
        section("10b. Deep learning: LSTM, GRU, TFT (sequence models)")
        from models import dl_common, gru_model, lstm_model, tft_model     # PyTorch is only needed for this step
        seq = dl_common.build_sequence_data(train, test)
        steps = [("LSTM", lstm_model.train_lstm), ("GRU", gru_model.train_gru)]
        if not args.skip_tft:
            steps.append(("TFT", tft_model.train_tft))
        for name, train_fn in steps:
            net, p_tr, p_te, dl_info = train_fn(seq, fast_mode=fast)
            results.append(evaluate(name, y_train, p_tr, y_test, p_te)); preds[name] = p_te
            dl_fitted[name] = (net, dl_info)
        dl_common.plot_training_curves({n: i for n, (_, i) in dl_fitted.items()})
        if "TFT" in dl_fitted:
            print("TFT: features the variable-selection network weights most:\n",
                  tft_model.plot_variable_importance(dl_fitted["TFT"][0], seq).iloc[::-1].head(8).round(3).to_string())

    section("11. Model comparison")
    res = comparison_table(results, save_name="model_results.csv" if args.skip_dl else "checkin2_results.csv")
    plot_model_comparison(res)
    strategies = ["Naive: persistence 24h"] + [m for m in ALL_MODELS if m in preds]
    per_region_mae(test, preds, strategies)
    plot_forecast_examples(test, preds)

    section("12. Scheduling simulation (CO₂ saved)")
    cubes = build_cubes(test, preds, strategies)
    time_shift_savings(cubes, strategies)
    spatio_temporal_savings(cubes, strategies, raw["regions"])
    example_recommendation(cubes, raw["regions"])

    section("13. Save artefacts")
    save_artifacts(model_df, fitted, raw["region_tz"])
    if dl_fitted:
        dl_common.save_dl_artifacts(dl_fitted, seq)
        print("Saved deep-learning weights:", sorted(p.name for p in MODEL_DIR.glob("*.pt")), "+ dl_config.json")
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Carbon-intensity forecasting pipeline")
    p.add_argument("--full", action="store_true", help="larger searches and full-length deep-learning training (slower)")
    p.add_argument("--skip-eda", action="store_true", help="skip EDA plots")
    p.add_argument("--skip-rf", action="store_true", help="skip the Random Forest model")
    p.add_argument("--skip-dl", action="store_true", help="skip the deep-learning models (LSTM, GRU, TFT)")
    p.add_argument("--skip-tft", action="store_true", help="skip only the TFT (slowest deep-learning model)")
    main(p.parse_args())
