import numpy as np
import pytest

from src.evaluation import build_metric_row, select_direction_threshold


def test_price_only_baseline_leaves_direction_metrics_undefined():
    row = build_metric_row(
        "persistence",
        actual_close=np.asarray([101.0, 99.0]),
        predicted_close=np.asarray([100.0, 100.0]),
        actual_direction=np.asarray([1, 0]),
        predicted_direction=None,
    )
    assert row["price_MAE_USD"] == 1.0
    assert np.isnan(row["direction_accuracy"])


def test_direction_metrics_are_calculated():
    row = build_metric_row(
        "direction",
        actual_close=np.asarray([101.0, 99.0]),
        predicted_close=None,
        actual_direction=np.asarray([1, 0]),
        predicted_direction=np.asarray([1, 1]),
    )
    assert np.isnan(row["price_MAE_USD"])
    assert row["direction_accuracy"] == 0.5
    assert row["direction_precision"] == 0.5


def test_direction_threshold_is_selected_from_probabilities():
    y_true = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.60, 0.62, 0.80, 0.91])
    threshold = select_direction_threshold(y_true, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    assert 0.62 < threshold <= 0.80
    assert np.mean(predictions == y_true) == 1.0

def test_direction_threshold_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        select_direction_threshold(np.asarray([0, 1]), np.asarray([0.2, np.nan]))
