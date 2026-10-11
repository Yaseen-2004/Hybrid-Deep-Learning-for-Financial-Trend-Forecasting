from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.data_utils import FEATURES, download_prices, save_prices, validate_ticker
from src.evaluation import build_metric_row, select_direction_threshold
from src.forecasting import make_targets, make_windows, reconstruct_close
from src.model_factory import build_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("FTF_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
MODEL_DIR = Path(os.getenv("FTF_MODELS_DIR", str(ROOT / "models"))).expanduser().resolve()
RESULTS_DIR = Path(os.getenv("FTF_RESULTS_DIR", str(ROOT / "results"))).expanduser().resolve()
MODEL_NAMES = ["cnn", "rnn", "lstm", "cnn_lstm"]
LOOKBACK = 30
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_callbacks() -> list[tf.keras.callbacks.Callback]:
    # Create fresh callbacks for every fit call so callback state isn't reused.
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]


def scaled_windows(
    df: pd.DataFrame,
    target_indices: np.ndarray,
    lookback: int,
    feature_scaler: StandardScaler,
) -> np.ndarray:
    feature_rows = feature_scaler.transform(df[FEATURES].to_numpy(dtype=np.float64)).astype(np.float32)
    return make_windows(feature_rows, target_indices, lookback)


def add_split_info(row: dict, split: str, fold: int | None = None) -> dict:
    row["split"] = split
    if fold is not None:
        row["fold"] = int(fold)
    return row


def add_baseline_rows(
    rows: list[dict],
    *,
    split: str,
    fold: int | None,
    close: np.ndarray,
    target_indices: np.ndarray,
    actual_direction: np.ndarray,
    majority_direction: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_close = close[target_indices - 1]
    # Persistence forecast: tomorrow's close equals the latest observed close.
    price_row = build_metric_row(
        "price_persistence_baseline",
        actual_close=close[target_indices],
        predicted_close=prev_close,
        actual_direction=actual_direction,
        predicted_direction=None,
    )
    rows.append(add_split_info(price_row, split, fold))

    # Momentum baseline: repeat the previous session's up/down direction.
    prev_direction = (close[target_indices - 1] > close[target_indices - 2]).astype(int)
    momentum_row = build_metric_row(
        "direction_momentum_baseline",
        actual_close=close[target_indices],
        predicted_close=None,
        actual_direction=actual_direction,
        predicted_direction=prev_direction,
    )
    rows.append(add_split_info(momentum_row, split, fold))

    majority_prediction = np.full(len(target_indices), int(majority_direction), dtype=int)
    majority_row = build_metric_row(
        "direction_majority_baseline",
        actual_close=close[target_indices],
        predicted_close=None,
        actual_direction=actual_direction,
        predicted_direction=majority_prediction,
    )
    rows.append(add_split_info(majority_row, split, fold))
    return prev_close, prev_direction, majority_prediction


def fit_one_model(
    name: str,
    X_train: np.ndarray,
    y_return_train_scaled: np.ndarray,
    y_direction_train: np.ndarray,
    X_val: np.ndarray,
    y_return_val_scaled: np.ndarray,
    y_direction_val: np.ndarray,
    *,
    lookback: int,
    n_features: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    seed_everything(seed)
    model = build_model(name, lookback, n_features)
    model.fit(
        X_train,
        {"log_return": y_return_train_scaled, "direction": y_direction_train},
        validation_data=(
            X_val,
            {"log_return": y_return_val_scaled, "direction": y_direction_val},
        ),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=make_callbacks(),
        verbose=0,
    )
    return model


def evaluate_walk_forward(
    *,
    df: pd.DataFrame,
    y_log_return: np.ndarray,
    y_direction: np.ndarray,
    target_indices: np.ndarray,
    development_end: int,
    lookback: int,
    folds: int,
    epochs: int,
    batch_size: int,
) -> pd.DataFrame:
    """Expanding-window evaluation on development data, leaving final test untouched.

    Each fold fits its own scalers using only that fold's model-fit targets/features,
    uses a tail section of the historical training window for early stopping, and
    evaluates on the later fold block.
    """
    if folds == 0:
        print("Walk-forward evaluation disabled (--walk-forward-folds 0).")
        return pd.DataFrame()

    development_indices = np.arange(development_end, dtype=int)
    splitter = TimeSeriesSplit(n_splits=folds)
    close = df["Close"].to_numpy(dtype=np.float64)
    feature_values = df[FEATURES].to_numpy(dtype=np.float64)
    rows: list[dict] = []

    for fold_number, (outer_train, evaluation_indices) in enumerate(
        splitter.split(development_indices), start=1
    ):
        # outer_train and evaluation_indices are sample indices into the full series.
        validation_start = int(len(outer_train) * 0.85)
        fit_indices = outer_train[:validation_start]
        validation_indices = outer_train[validation_start:]
        if len(fit_indices) < 50 or len(validation_indices) < 10 or len(evaluation_indices) < 10:
            warnings.warn(
                f"Skipping walk-forward fold {fold_number}: too few samples for fit/validation/evaluation.",
                RuntimeWarning,
            )
            continue

        # A sample targeted at row i only uses features through row i-1. The scaler
        # therefore fits rows strictly before the last training target row.
        scaler_fit_end = int(target_indices[fit_indices[-1]])
        feature_scaler = StandardScaler().fit(feature_values[:scaler_fit_end])
        target_scaler = StandardScaler().fit(y_log_return[fit_indices].reshape(-1, 1))
        X_fold = scaled_windows(df, target_indices, lookback, feature_scaler)
        y_return_scaled = target_scaler.transform(y_log_return.reshape(-1, 1)).astype(np.float32).reshape(-1)

        majority_direction = int(np.mean(y_direction[outer_train]) >= 0.5)
        eval_target_rows = target_indices[evaluation_indices]
        eval_actual_direction = y_direction[evaluation_indices].astype(int)
        add_baseline_rows(
            rows,
            split="walk_forward",
            fold=fold_number,
            close=close,
            target_indices=eval_target_rows,
            actual_direction=eval_actual_direction,
            majority_direction=majority_direction,
        )
        actual_close = close[eval_target_rows]
        previous_close = close[eval_target_rows - 1]

        print(
            f"Walk-forward fold {fold_number}/{folds}: "
            f"fit={len(fit_indices)}, early-stop validation={len(validation_indices)}, "
            f"evaluation={len(evaluation_indices)}"
        )
        for name in MODEL_NAMES:
            model = fit_one_model(
                name,
                X_fold[fit_indices],
                y_return_scaled[fit_indices],
                y_direction[fit_indices],
                X_fold[validation_indices],
                y_return_scaled[validation_indices],
                y_direction[validation_indices],
                lookback=lookback,
                n_features=len(FEATURES),
                epochs=epochs,
                batch_size=batch_size,
                seed=SEED + fold_number,
            )
            # Select the direction threshold from this fold's validation window only.
            validation_output = model.predict(X_fold[validation_indices], verbose=0)
            validation_probability = np.asarray(validation_output["direction"]).reshape(-1)
            direction_threshold = select_direction_threshold(
                y_direction[validation_indices], validation_probability
            )

            output = model.predict(X_fold[evaluation_indices], verbose=0)
            predicted_return_scaled = np.asarray(output["log_return"]).reshape(-1, 1)
            predicted_return = target_scaler.inverse_transform(predicted_return_scaled).reshape(-1)
            predicted_close = reconstruct_close(previous_close, predicted_return)
            predicted_probability = np.asarray(output["direction"]).reshape(-1)
            predicted_direction = (predicted_probability >= direction_threshold).astype(int)
            row = build_metric_row(
                name,
                actual_close=actual_close,
                predicted_close=predicted_close,
                actual_direction=eval_actual_direction,
                predicted_direction=predicted_direction,
            )
            row["direction_threshold"] = float(direction_threshold)
            rows.append(add_split_info(row, "walk_forward", fold_number))
            del model

    return pd.DataFrame(rows)


def summarize_walk_forward(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "price_MAE_USD", "price_RMSE_USD", "direction_accuracy",
        "direction_balanced_accuracy", "direction_precision", "direction_recall",
        "direction_f1", "direction_threshold",
    ]
    if fold_metrics.empty:
        empty_columns = ["model"] + [
            f"{metric}_{stat}" for metric in metric_columns for stat in ("mean", "std")
        ] + ["folds_evaluated"]
        return pd.DataFrame(columns=empty_columns)
    summary = fold_metrics.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    fold_counts = fold_metrics.groupby("model")["fold"].nunique().rename("folds_evaluated")
    return summary.merge(fold_counts, on="model", how="left")


def plot_results(predictions: pd.DataFrame, ticker: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(predictions["Date"], predictions["actual_close"], label="Actual", linewidth=2.2, color="black")
    if "persistence_pred_close" in predictions:
        ax.plot(
            predictions["Date"], predictions["persistence_pred_close"],
            label="Persistence baseline", alpha=0.8, linestyle="--",
        )
    for name in MODEL_NAMES:
        column = f"{name}_pred_close"
        if column in predictions:
            ax.plot(predictions["Date"], predictions[column], label=name.upper(), alpha=0.8)
    ax.set_title(f"{ticker}: held-out next-session adjusted-close forecasts")
    ax.set_xlabel("Target session")
    ax.set_ylabel("Adjusted close (USD)")
    ax.legend(ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sha256_frame(df: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare CNN, RNN, LSTM, and CNN-LSTM financial forecasters."
    )
    parser.add_argument("--ticker", default="AAPL", help="US ticker symbol, e.g. AAPL, MSFT, NVDA")
    parser.add_argument("--period", default="5y", help="yfinance period, e.g. 2y, 5y, 10y, max")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument(
        "--walk-forward-folds", type=int, default=2,
        help="Number of expanding-window evaluation folds on development data; use 0 to skip (default: 2).",
    )
    parser.add_argument("--run-id", default=None, help="Optional unique run ID. Existing run IDs are not overwritten.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lookback < 5:
        parser.error("epochs and batch-size must be positive; lookback must be at least 5")
    if args.walk_forward_folds < 0 or args.walk_forward_folds > 5:
        parser.error("walk-forward-folds must be between 0 and 5")

    ticker = validate_ticker(args.ticker)
    run_id = args.run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        parser.error("run-id may contain only letters, numbers, underscores, and hyphens")

    model_run_dir = MODEL_DIR / ticker / run_id
    results_run_dir = RESULTS_DIR / ticker / run_id
    data_run_dir = DATA_DIR / ticker / run_id
    if model_run_dir.exists() or results_run_dir.exists() or data_run_dir.exists():
        parser.error(f"Run ID '{run_id}' already exists for {ticker}; choose a new --run-id")
    for folder in (model_run_dir, results_run_dir, data_run_dir):
        folder.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED)
    print(f"Ticker: {ticker} | period: {args.period} | run: {run_id}")
    print(f"Downloading daily market data for {ticker}...")
    df = download_prices(ticker, args.period)
    save_prices(df, data_run_dir / "historical_prices.csv")
    data_fingerprint = sha256_frame(df)

    y_log_return, y_direction, target_indices = make_targets(df, args.lookback)
    n_samples = len(target_indices)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)
    if train_end < 50 or (val_end - train_end) < 15 or (n_samples - val_end) < 20:
        raise ValueError(
            f"Not enough usable samples for 70/15/15 split: {n_samples} samples. "
            "Use a longer --period (the default 5y is recommended)."
        )
    if args.walk_forward_folds and val_end < 100 * (args.walk_forward_folds + 1):
        warnings.warn(
            "Not enough development samples for the requested walk-forward folds; skipping walk-forward evaluation. "
            "Use a longer period or set --walk-forward-folds 0.",
            RuntimeWarning,
        )
        effective_folds = 0
    else:
        effective_folds = args.walk_forward_folds

    print("Running expanding-window walk-forward evaluation...")
    walk_forward = evaluate_walk_forward(
        df=df,
        y_log_return=y_log_return,
        y_direction=y_direction,
        target_indices=target_indices,
        development_end=val_end,
        lookback=args.lookback,
        folds=effective_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    if walk_forward.empty:
        walk_forward = pd.DataFrame(columns=[
            "model", "n_samples", "price_MAE_USD", "price_RMSE_USD",
            "direction_accuracy", "direction_balanced_accuracy", "direction_precision",
            "direction_recall", "direction_f1", "direction_threshold", "split", "fold",
        ])
    walk_forward_path = results_run_dir / "walk_forward_metrics.csv"
    walk_forward.to_csv(walk_forward_path, index=False)
    walk_forward_summary = summarize_walk_forward(walk_forward)
    walk_forward_summary.to_csv(results_run_dir / "walk_forward_summary.csv", index=False)

    # The final 15% is held out from all walk-forward development folds.
    last_fit_target_row = int(target_indices[train_end - 1])
    feature_scaler = StandardScaler().fit(
        df[FEATURES].iloc[:last_fit_target_row].to_numpy(dtype=np.float64)
    )
    target_scaler = StandardScaler().fit(y_log_return[:train_end].reshape(-1, 1))
    X = scaled_windows(df, target_indices, args.lookback, feature_scaler)
    y_return_scaled = target_scaler.transform(y_log_return.reshape(-1, 1)).astype(np.float32).reshape(-1)

    X_train, y_train_ret, y_train_dir = X[:train_end], y_return_scaled[:train_end], y_direction[:train_end]
    X_val, y_val_ret, y_val_dir = X[train_end:val_end], y_return_scaled[train_end:val_end], y_direction[train_end:val_end]
    X_test, y_test_ret, y_test_dir = X[val_end:], y_return_scaled[val_end:], y_direction[val_end:]
    test_target_indices = target_indices[val_end:]
    close = df["Close"].to_numpy(dtype=np.float64)
    actual_close = close[test_target_indices]
    previous_close = close[test_target_indices - 1]
    actual_direction = y_test_dir.astype(int)
    majority_direction = int(np.mean(y_train_dir) >= 0.5)

    joblib.dump(feature_scaler, model_run_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, model_run_dir / "target_scaler.joblib")

    predictions = pd.DataFrame({
        "Date": [df.index[i].strftime("%Y-%m-%d") for i in test_target_indices],
        "actual_close": actual_close,
        "actual_log_return": y_log_return[val_end:],
        "actual_direction_up": actual_direction,
        "persistence_pred_close": previous_close,
    })
    test_metrics: list[dict] = []
    direction_thresholds: dict[str, float] = {}
    add_baseline_rows(
        test_metrics,
        split="heldout_test",
        fold=None,
        close=close,
        target_indices=test_target_indices,
        actual_direction=actual_direction,
        majority_direction=majority_direction,
    )
    previous_direction = (close[test_target_indices - 1] > close[test_target_indices - 2]).astype(int)
    predictions["momentum_pred_direction_up"] = previous_direction
    predictions["majority_pred_direction_up"] = majority_direction

    for name in MODEL_NAMES:
        print(f"Training final {name.upper()} model...")
        model = fit_one_model(
            name,
            X_train, y_train_ret, y_train_dir,
            X_val, y_val_ret, y_val_dir,
            lookback=args.lookback,
            n_features=len(FEATURES),
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=SEED,
        )
        model_path = model_run_dir / f"{name}.keras"
        model.save(model_path)

        # The validation period is used for early stopping and threshold selection;
        # the final held-out test set remains untouched until final evaluation.
        validation_output = model.predict(X_val, verbose=0)
        validation_probability = np.asarray(validation_output["direction"]).reshape(-1)
        direction_threshold = select_direction_threshold(y_val_dir, validation_probability)
        direction_thresholds[name] = float(direction_threshold)

        output = model.predict(X_test, verbose=0)
        predicted_return_scaled = np.asarray(output["log_return"]).reshape(-1, 1)
        predicted_return = target_scaler.inverse_transform(predicted_return_scaled).reshape(-1)
        predicted_close = reconstruct_close(previous_close, predicted_return)
        up_probability = np.asarray(output["direction"]).reshape(-1)
        predicted_direction = (up_probability >= direction_threshold).astype(int)

        predictions[f"{name}_pred_log_return"] = predicted_return
        predictions[f"{name}_pred_close"] = predicted_close
        predictions[f"{name}_up_probability"] = up_probability
        predictions[f"{name}_direction_threshold"] = direction_threshold
        predictions[f"{name}_pred_direction_up"] = predicted_direction
        row = build_metric_row(
            name,
            actual_close=actual_close,
            predicted_close=predicted_close,
            actual_direction=actual_direction,
            predicted_direction=predicted_direction,
        )
        row["direction_threshold"] = float(direction_threshold)
        test_metrics.append(add_split_info(row, "heldout_test"))
        print(f"Saved model: {model_path}")
        del model

    metrics = pd.DataFrame(test_metrics)
    metrics.to_csv(results_run_dir / "metrics.csv", index=False)
    predictions.to_csv(results_run_dir / "test_predictions.csv", index=False)
    plot_results(predictions, ticker, results_run_dir / "test_predictions.png")

    config = {
        "project": "Hybrid Deep Learning for Financial Trend Forecasting",
        "ticker": ticker,
        "run_id": run_id,
        "period": args.period,
        "lookback": args.lookback,
        "features": FEATURES,
        "models": MODEL_NAMES,
        "target": "next-session log return; close is reconstructed from prior close * exp(predicted log return)",
        "direction_target": "1 if next adjusted close > previous observed adjusted close, else 0",
        "direction_probability_note": "Neural network sigmoid output; not calibrated unless separately evaluated/calibrated.",
        "direction_thresholds": direction_thresholds,
        "direction_threshold_selection": "Per-model threshold maximizes balanced accuracy on the chronological validation segment; ties prefer a validation predicted-positive rate near validation prevalence. Never selected on final test data.",
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "walk_forward_folds": int(effective_folds),
        "seed": SEED,
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "data_sha256": data_fingerprint,
        "versions": {
            "python": os.sys.version.split()[0],
            "tensorflow": tf.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "yfinance": yf.__version__,
        },
        "artifact_files": {
            "models": [f"{name}.keras" for name in MODEL_NAMES],
            "feature_scaler": "feature_scaler.joblib",
            "target_scaler": "target_scaler.joblib",
            "prices": f"data/{ticker}/{run_id}/historical_prices.csv",
            "metrics": f"results/{ticker}/{run_id}/metrics.csv",
            "test_predictions": f"results/{ticker}/{run_id}/test_predictions.csv",
            "walk_forward_metrics": f"results/{ticker}/{run_id}/walk_forward_metrics.csv",
            "walk_forward_summary": f"results/{ticker}/{run_id}/walk_forward_summary.csv",
        },
    }
    config_path = model_run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    latest_path = MODEL_DIR / ticker / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_latest = latest_path.with_suffix(".tmp")
    temp_latest.write_text(json.dumps({"run_id": run_id, "config": str(config_path.relative_to(ROOT))}, indent=2), encoding="utf-8")
    temp_latest.replace(latest_path)

    display_columns = [
        "model", "price_MAE_USD", "price_RMSE_USD", "direction_accuracy",
        "direction_balanced_accuracy", "direction_precision", "direction_recall", "direction_f1",
        "direction_threshold",
    ]
    print("\nHeld-out test results (these observations were not used for training or walk-forward development):")
    print(metrics[display_columns].round(4).to_string(index=False))
    if not walk_forward_summary.empty:
        print("\nWalk-forward summary (mean over development folds):")
        columns = ["model", "folds_evaluated", "price_MAE_USD_mean", "direction_accuracy_mean", "direction_f1_mean"]
        print(walk_forward_summary[[c for c in columns if c in walk_forward_summary.columns]].round(4).to_string(index=False))
    print(f"\nRun artifacts: {model_run_dir}")
    print(f"Evaluation results: {results_run_dir}")
    print("Launch dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
