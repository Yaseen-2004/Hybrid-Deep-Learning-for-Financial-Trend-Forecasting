from __future__ import annotations

import numpy as np
import pandas as pd


def make_targets(df: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return next-session log returns, direction labels, and target row indices.

    For a target row i, the input window ends at row i-1; row i is the unseen target.
    """
    if lookback < 1:
        raise ValueError("lookback must be positive")
    close = df["Close"].to_numpy(dtype=np.float64)
    if len(close) <= lookback:
        raise ValueError("Not enough observations to build sequences")
    target_indices = np.arange(lookback, len(close), dtype=int)
    log_returns = np.log(close[target_indices] / close[target_indices - 1])
    if not np.isfinite(log_returns).all():
        raise ValueError("Target returns contain non-finite values")
    direction = (log_returns > 0).astype(np.float32)
    return log_returns.astype(np.float32), direction, target_indices


def make_windows(scaled_feature_rows: np.ndarray, target_indices: np.ndarray, lookback: int) -> np.ndarray:
    """Create windows using rows strictly before each target row."""
    return np.stack(
        [scaled_feature_rows[i - lookback:i] for i in target_indices], axis=0
    ).astype(np.float32)


def reconstruct_close(previous_close: np.ndarray, predicted_log_return: np.ndarray) -> np.ndarray:
    """Convert log-return forecasts into price forecasts; reject non-finite results."""
    previous_close = np.asarray(previous_close, dtype=np.float64).reshape(-1)
    predicted_log_return = np.asarray(predicted_log_return, dtype=np.float64).reshape(-1)
    if previous_close.shape != predicted_log_return.shape:
        raise ValueError("previous_close and predicted_log_return must have matching shapes")
    if not np.isfinite(previous_close).all() or (previous_close <= 0).any():
        raise ValueError("previous_close must contain finite positive values")
    if not np.isfinite(predicted_log_return).all():
        raise ValueError("Predicted log returns contain non-finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        predicted_close = previous_close * np.exp(predicted_log_return)
    if not np.isfinite(predicted_close).all():
        raise ValueError("The model produced an invalid or extreme price forecast")
    return predicted_close
