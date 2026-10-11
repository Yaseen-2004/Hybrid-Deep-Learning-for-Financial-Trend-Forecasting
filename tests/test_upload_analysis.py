import numpy as np
import pandas as pd
import pytest

from src.upload_analysis import (
    clean_historical_data,
    detect_upload_type,
    historical_summary,
    prediction_metrics,
)


def test_detect_historical_price_csv():
    df = pd.DataFrame({"Date": ["2025-01-01", "2025-01-02"], "Adj Close": [10, 11]})
    assert detect_upload_type(df) == "history"


def test_clean_historical_data_sorts_deduplicates_and_calculates_indicators():
    df = pd.DataFrame({
        "Date": ["2025-01-02", "2025-01-01", "2025-01-02", "2025-01-03"],
        "Adj Close": [12, 10, 11, 12],
        "Volume": [200, 100, 150, 250],
    })
    clean, source = clean_historical_data(df)
    assert source == "Adj Close"
    assert len(clean) == 3
    assert clean.index.is_monotonic_increasing
    assert clean.loc[pd.Timestamp("2025-01-02"), "Price"] == 11
    assert "SMA 20" in clean.columns and "Drawdown" in clean.columns
    summary = historical_summary(clean)
    assert summary["observations"] == 3
    assert summary["total_return_pct"] == pytest.approx(20.0)


def test_detect_and_score_prediction_export():
    df = pd.DataFrame({
        "Date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "actual_close": [10.0, 11.0, 10.0, 12.0],
        "actual_direction_up": [1, 1, 0, 1],
        "cnn_pred_close": [10.0, 10.0, 11.0, 12.0],
        "cnn_pred_direction_up": [1, 0, 1, 1],
        "persistence_pred_close": [9.0, 10.0, 11.0, 10.0],
    })
    assert detect_upload_type(df) == "predictions"
    scored = prediction_metrics(df)
    cnn = scored.loc[scored["model"] == "CNN"].iloc[0]
    assert cnn["n_price"] == 4
    assert cnn["price_MAE_USD"] == pytest.approx(0.5)
    assert cnn["direction_accuracy"] == pytest.approx(0.5)


def test_detect_metrics_file():
    df = pd.DataFrame({"model": ["cnn"], "price_MAE_USD": [2.4], "direction_f1": [0.5]})
    assert detect_upload_type(df) == "metrics"


def test_history_requires_date_and_price():
    df = pd.DataFrame({"Date": ["2025-01-01", "2025-01-02"], "Volume": [100, 200]})
    with pytest.raises(ValueError):
        clean_historical_data(df)


def test_probability_metrics_detect_and_score_up_scores():
    from src.upload_analysis import probability_column_map, probability_score_metrics

    df = pd.DataFrame({
        "actual_direction_up": [0, 1, 0, 1],
        "cnn_up_probability": [0.1, 0.8, 0.3, 0.7],
    })
    assert probability_column_map(df) == {"CNN": "cnn_up_probability"}
    metrics = probability_score_metrics(df)
    assert metrics.iloc[0]["model"] == "CNN"
    assert metrics.iloc[0]["roc_auc"] == pytest.approx(1.0)
    assert metrics.iloc[0]["brier_score"] < 0.1


def test_read_csv_upload_with_semicolon_delimiter():
    from src.upload_analysis import read_uploaded_table

    result = read_uploaded_table("sample.csv", b"Date;Close\n2025-01-01;10\n2025-01-02;11\n")
    assert list(result.columns) == ["Date", "Close"]
    assert result["Close"].tolist() == [10, 11]


def test_read_excel_upload_selects_requested_sheet():
    from io import BytesIO
    from src.upload_analysis import excel_sheet_names, read_uploaded_table

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"Date": ["2025-01-01"], "Close": [10]}).to_excel(writer, sheet_name="Prices", index=False)
        pd.DataFrame({"model": ["CNN"], "direction_f1": [0.5]}).to_excel(writer, sheet_name="Metrics", index=False)
    payload = buffer.getvalue()
    assert excel_sheet_names(payload) == ["Prices", "Metrics"]
    metrics = read_uploaded_table("sample.xlsx", payload, sheet_name="Metrics")
    assert metrics.loc[0, "model"] == "CNN"
