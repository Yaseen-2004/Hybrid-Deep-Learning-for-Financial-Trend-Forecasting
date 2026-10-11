from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

PRICE_METRICS = ["price_MAE_USD", "price_RMSE_USD"]
DIRECTION_METRICS = [
    "direction_accuracy", "direction_balanced_accuracy",
    "direction_precision", "direction_recall", "direction_f1",
]


def build_metric_row(
    model: str,
    *,
    actual_close: np.ndarray,
    predicted_close: np.ndarray | None,
    actual_direction: np.ndarray,
    predicted_direction: np.ndarray | None,
) -> dict[str, float | str]:
    """Compute task-specific metrics. Missing task outputs are represented by NaN."""
    actual_close = np.asarray(actual_close, dtype=np.float64).reshape(-1)
    actual_direction = np.asarray(actual_direction, dtype=int).reshape(-1)
    row: dict[str, float | str] = {"model": model, "n_samples": int(len(actual_direction))}

    if predicted_close is None:
        row["price_MAE_USD"] = float("nan")
        row["price_RMSE_USD"] = float("nan")
    else:
        predicted_close = np.asarray(predicted_close, dtype=np.float64).reshape(-1)
        row["price_MAE_USD"] = float(mean_absolute_error(actual_close, predicted_close))
        row["price_RMSE_USD"] = float(np.sqrt(mean_squared_error(actual_close, predicted_close)))

    if predicted_direction is None:
        for metric in DIRECTION_METRICS:
            row[metric] = float("nan")
    else:
        predicted_direction = np.asarray(predicted_direction, dtype=int).reshape(-1)
        row["direction_accuracy"] = float(accuracy_score(actual_direction, predicted_direction))
        row["direction_balanced_accuracy"] = float(
            balanced_accuracy_score(actual_direction, predicted_direction)
        )
        row["direction_precision"] = float(
            precision_score(actual_direction, predicted_direction, zero_division=0)
        )
        row["direction_recall"] = float(
            recall_score(actual_direction, predicted_direction, zero_division=0)
        )
        row["direction_f1"] = float(
            f1_score(actual_direction, predicted_direction, zero_division=0)
        )
    return row


def select_direction_threshold(
    y_true: np.ndarray, probabilities: np.ndarray
) -> float:
    """Select a classification threshold using validation data only.

    The primary criterion is balanced accuracy. Exact ties are resolved by
    choosing a threshold whose validation positive-prediction rate is closest
    to the validation class prevalence, then by closeness to 0.5. This reduces
    arbitrary all-one-class decisions when several thresholds score identically.

    Callers must pass a chronological validation segment, never the final test
    set. If validation contains only one class, use the conventional 0.5 cutoff.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if y_true.size == 0 or probabilities.size != y_true.size:
        raise ValueError("y_true and probabilities must be non-empty and have equal length")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    classes = np.unique(y_true)
    if not np.isin(classes, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if len(classes) < 2:
        return 0.5

    unique = np.unique(probabilities)
    if unique.size == 1:
        return 0.5
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], midpoints)))
    prevalence = float(np.mean(y_true == 1))
    scored: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        score = float(balanced_accuracy_score(y_true, predicted))
        predicted_rate_gap = abs(float(np.mean(predicted == 1)) - prevalence)
        scored.append((score, -predicted_rate_gap, -abs(float(threshold) - 0.5), float(threshold)))
    # Lexicographic maximum: best balanced accuracy, best prevalence match,
    # then most conventional threshold. Threshold is the final deterministic tie-break.
    return max(scored)[3]
