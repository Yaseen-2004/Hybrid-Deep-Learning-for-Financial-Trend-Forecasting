"""Helpers for safe, exploratory analysis of uploaded CSV files.

Supported CSV types:
- historical price data (Date + Close/Adj Close/Price; optional OHLCV)
- held-out prediction exports (actual_close and *_pred_close columns)
- model metrics / walk-forward summary tables
- generic tabular data (numeric column exploration only)
"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def normalized_header(value: Any) -> str:
    """Lowercase a column name and remove punctuation/spacing."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())




def excel_sheet_names(payload: bytes) -> list[str]:
    """List worksheet names from an XLSX upload without persisting it to disk."""
    try:
        with pd.ExcelFile(BytesIO(payload), engine="openpyxl") as book:
            return list(book.sheet_names)
    except Exception as exc:
        raise ValueError(f"Could not read this Excel workbook: {exc}") from exc


def read_uploaded_table(filename: str, payload: bytes, sheet_name: str | None = None) -> pd.DataFrame:
    """Read CSV or XLSX bytes into a DataFrame with encoding/delimiter fallbacks."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(BytesIO(payload), sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(payload), sep=None, engine="python", encoding="cp1252")
    if suffix == ".xlsx":
        sheets = excel_sheet_names(payload)
        if not sheets:
            raise ValueError("The Excel workbook contains no worksheets.")
        if sheet_name is None:
            if len(sheets) > 1:
                raise ValueError("This workbook has multiple worksheets; select one worksheet to analyze.")
            sheet_name = sheets[0]
        if sheet_name not in sheets:
            raise ValueError(f"Worksheet {sheet_name!r} was not found in the workbook.")
        try:
            return pd.read_excel(BytesIO(payload), sheet_name=sheet_name, engine="openpyxl")
        except Exception as exc:
            raise ValueError(f"Could not read worksheet {sheet_name!r}: {exc}") from exc
    raise ValueError("Unsupported file format. Upload a CSV or .xlsx file.")


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {normalized_header(col): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def detect_upload_type(df: pd.DataFrame) -> str:
    """Classify a DataFrame as history, prediction results, metrics, or generic."""
    if df.empty or len(df.columns) == 0:
        return "empty"
    normalized = {normalized_header(col) for col in df.columns}
    has_actual_close = "actualclose" in normalized
    has_predictions = any(col.endswith("predclose") for col in normalized)
    if has_actual_close and has_predictions:
        return "predictions"
    metric_markers = {
        "pricemaeusd", "pricermseusd", "directionaccuracy",
        "directionbalancedaccuracy", "directionf1", "pricemaeusdmean",
    }
    if "model" in normalized and normalized.intersection(metric_markers):
        return "metrics"
    date_col = _find_column(df, ("date", "datetime", "timestamp", "tradingdate", "time", "unnamed0", "index"))
    price_col = _find_column(df, ("adjclose", "adjustedclose", "close", "price", "last", "settle", "adjustedprice"))
    if date_col is not None and price_col is not None:
        return "history"
    # Also accept files whose first column contains parseable dates but has an unusual header.
    if len(df.columns) >= 2:
        first_col = str(df.columns[0])
        parsed = pd.to_datetime(df[first_col], errors="coerce", utc=True)
        if parsed.notna().mean() >= 0.8 and price_col is not None:
            return "history"
    return "generic"


def _numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[$€£₹]", "", regex=True)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def clean_historical_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Normalize uploaded OHLCV history and return (clean_frame, price_source_label).

    Result has a DateTimeIndex and a required Price column. Open/High/Low/Close/Volume
    are included if present. The adjusted close column is preferred when available.
    """
    if raw is None or raw.empty:
        raise ValueError("The uploaded CSV is empty.")
    date_col = _find_column(raw, ("date", "datetime", "timestamp", "tradingdate", "time", "unnamed0", "index"))
    if date_col is None and len(raw.columns) >= 2:
        parsed_first = pd.to_datetime(raw.iloc[:, 0], errors="coerce", utc=True)
        if parsed_first.notna().mean() >= 0.8:
            date_col = str(raw.columns[0])
    if date_col is None:
        raise ValueError("Couldn't identify a date column. Include a column named Date, Datetime, or Timestamp.")

    adjusted_col = _find_column(raw, ("adjclose", "adjustedclose"))
    close_col = _find_column(raw, ("close",))
    other_price_col = _find_column(raw, ("price", "last", "settle", "adjustedprice"))
    price_col = adjusted_col or close_col or other_price_col
    if price_col is None:
        raise ValueError("Couldn't identify a price column. Include Adj Close, Close, or Price.")

    # Parse to UTC first so mixed timezone inputs can be normalized, then use session dates.
    dates = pd.to_datetime(raw[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    result = pd.DataFrame(index=pd.DatetimeIndex(dates, name="Date"))
    canonical_sources = {
        "Open": _find_column(raw, ("open",)),
        "High": _find_column(raw, ("high",)),
        "Low": _find_column(raw, ("low",)),
        "Close": close_col,
        "Volume": _find_column(raw, ("volume", "vol")),
    }
    for canonical, source in canonical_sources.items():
        if source is not None:
            result[canonical] = _numeric_series(raw[source]).to_numpy()
    result["Price"] = _numeric_series(raw[price_col]).to_numpy()
    result = result.reset_index()
    result = result.dropna(subset=["Date", "Price"])
    result = result[np.isfinite(result["Price"]) & (result["Price"] > 0)]
    result = result.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    if len(result) < 3:
        raise ValueError("At least three valid dated prices are required for analysis.")
    if "Volume" in result.columns:
        result["Volume"] = result["Volume"].clip(lower=0)
    result["Daily Return"] = result["Price"].pct_change(fill_method=None)
    result["SMA 20"] = result["Price"].rolling(20, min_periods=1).mean()
    result["SMA 50"] = result["Price"].rolling(50, min_periods=1).mean()
    result["Rolling Volatility (20d, annualized)"] = (
        result["Daily Return"].rolling(20, min_periods=5).std() * np.sqrt(252) * 100
    )
    result["Running Peak"] = result["Price"].cummax()
    result["Drawdown"] = (result["Price"] / result["Running Peak"] - 1.0) * 100
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    label = str(price_col)
    return result, label


def historical_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Calculate descriptive metrics for a cleaned historical price frame."""
    price = frame["Price"].dropna().astype(float)
    returns = frame["Daily Return"].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(price) < 2:
        raise ValueError("Not enough valid prices to calculate summary metrics.")
    volatility = float(returns.std(ddof=1) * np.sqrt(252) * 100) if len(returns) > 1 else float("nan")
    return {
        "observations": int(len(price)),
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
        "first_price": float(price.iloc[0]),
        "last_price": float(price.iloc[-1]),
        "total_return_pct": float((price.iloc[-1] / price.iloc[0] - 1.0) * 100),
        "annualized_volatility_pct": volatility,
        "max_drawdown_pct": float(frame["Drawdown"].min()),
        "best_day_pct": float(returns.max() * 100) if len(returns) else float("nan"),
        "worst_day_pct": float(returns.min() * 100) if len(returns) else float("nan"),
        "positive_day_share_pct": float((returns > 0).mean() * 100) if len(returns) else float("nan"),
        "annualized_sharpe_approx": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) > 0 else float("nan"),
    }


def _resolve_column(df: pd.DataFrame, target_normalized: str) -> str | None:
    for col in df.columns:
        if normalized_header(col) == target_normalized:
            return str(col)
    return None


def prediction_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive price and direction metrics from an exported test_predictions.csv."""
    actual_close_col = _resolve_column(raw, "actualclose")
    if actual_close_col is None:
        raise ValueError("Prediction analysis requires an actual_close column.")
    actual_close = _numeric_series(raw[actual_close_col])
    actual_dir_col = _resolve_column(raw, "actualdirectionup")
    actual_direction = _numeric_series(raw[actual_dir_col]) if actual_dir_col else None
    price_cols = [str(c) for c in raw.columns if normalized_header(c).endswith("predclose")]
    direction_cols = [str(c) for c in raw.columns if normalized_header(c).endswith("preddirectionup")]
    rows: list[dict[str, Any]] = []
    predictors: dict[str, dict[str, str]] = {}
    for col in price_cols:
        stem = normalized_header(col)[:-len("predclose")].rstrip("_")
        if stem.endswith("pred"):
            stem = stem[:-4]
        predictors.setdefault(stem, {})["price"] = col
    for col in direction_cols:
        stem = normalized_header(col)[:-len("preddirectionup")].rstrip("_")
        predictors.setdefault(stem, {})["direction"] = col
    labels = {
        "cnn": "CNN", "rnn": "Vanilla RNN", "lstm": "LSTM", "cnnlstm": "CNN–LSTM",
        "persistence": "Price persistence baseline", "momentum": "Direction momentum baseline",
        "majority": "Direction majority baseline",
    }
    for stem, cols in predictors.items():
        label = labels.get(stem, stem.upper() if stem else "Model")
        row: dict[str, Any] = {"model": label, "n_price": 0, "price_MAE_USD": np.nan, "price_RMSE_USD": np.nan,
                              "n_direction": 0, "direction_accuracy": np.nan, "direction_balanced_accuracy": np.nan,
                              "direction_precision": np.nan, "direction_recall": np.nan, "direction_f1": np.nan}
        if "price" in cols:
            predicted = _numeric_series(raw[cols["price"]])
            mask = actual_close.notna() & predicted.notna() & np.isfinite(actual_close) & np.isfinite(predicted)
            if mask.any():
                errors = predicted[mask].to_numpy(dtype=float) - actual_close[mask].to_numpy(dtype=float)
                row["n_price"] = int(mask.sum())
                row["price_MAE_USD"] = float(mean_absolute_error(actual_close[mask], predicted[mask]))
                row["price_RMSE_USD"] = float(np.sqrt(mean_squared_error(actual_close[mask], predicted[mask])))
        if "direction" in cols and actual_direction is not None:
            pred = _numeric_series(raw[cols["direction"]])
            mask = actual_direction.notna() & pred.notna() & actual_direction.isin([0, 1]) & pred.isin([0, 1])
            if mask.any():
                y_true = actual_direction[mask].astype(int).to_numpy()
                y_pred = pred[mask].astype(int).to_numpy()
                row["n_direction"] = int(mask.sum())
                row["direction_accuracy"] = float(accuracy_score(y_true, y_pred))
                row["direction_balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
                row["direction_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
                row["direction_recall"] = float(recall_score(y_true, y_pred, zero_division=0))
                row["direction_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        rows.append(row)
    if not rows:
        raise ValueError("No *_pred_close or *_pred_direction_up columns were found.")
    result = pd.DataFrame(rows)
    sort_col = "price_MAE_USD"
    return result.sort_values(sort_col, na_position="last").reset_index(drop=True)


def prediction_column_map(raw: pd.DataFrame) -> dict[str, str]:
    """Map display labels to prediction close columns, preserving source names."""
    labels = {
        "cnnpredclose": "CNN", "rnnpredclose": "Vanilla RNN", "lstmpredclose": "LSTM",
        "cnnlstmpredclose": "CNN–LSTM", "persistencepredclose": "Price persistence baseline",
    }
    out: dict[str, str] = {}
    for col in raw.columns:
        normalized = normalized_header(col)
        if normalized.endswith("predclose"):
            stem = normalized[:-len("predclose")]
            display = labels.get(stem + "predclose", labels.get(normalized, str(col)))
            out[display] = str(col)
    return out


def probability_column_map(raw: pd.DataFrame) -> dict[str, str]:
    """Map human-readable model labels to columns containing upward scores."""
    suffixes = ("upprobability", "probabilityup", "upscore", "directionscore")
    labels = {"cnn": "CNN", "rnn": "Vanilla RNN", "lstm": "LSTM", "cnnlstm": "CNN–LSTM"}
    output: dict[str, str] = {}
    for col in raw.columns:
        normalized = normalized_header(col)
        suffix = next((item for item in suffixes if normalized.endswith(item)), None)
        if suffix is None:
            continue
        stem = normalized[:-len(suffix)].rstrip("_")
        output[labels.get(stem, stem.upper() if stem else str(col))] = str(col)
    return output


def probability_score_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Score uploaded upward-probability columns against actual binary direction.

    Brier score measures probability accuracy (lower is better); ROC-AUC measures
    ranking ability. These are descriptive for the uploaded sample only. A score
    is not assumed to be calibrated merely because it lies between zero and one.
    """
    actual_col = _resolve_column(raw, "actualdirectionup")
    score_cols = probability_column_map(raw)
    if actual_col is None:
        raise ValueError("Probability diagnostics require an actual_direction_up column.")
    if not score_cols:
        raise ValueError("No upward probability/score columns were found.")
    actual = _numeric_series(raw[actual_col])
    rows: list[dict[str, Any]] = []
    for label, col in score_cols.items():
        score = _numeric_series(raw[col])
        mask = actual.isin([0, 1]) & score.between(0, 1) & np.isfinite(score)
        y_true = actual[mask].astype(int).to_numpy()
        y_score = score[mask].astype(float).to_numpy()
        if len(y_true) == 0:
            continue
        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan")
        rows.append({
            "model": label,
            "n_samples": int(len(y_true)),
            "brier_score": float(brier_score_loss(y_true, y_score)),
            "roc_auc": auc,
            "mean_up_score_pct": float(np.mean(y_score) * 100),
            "actual_up_rate_pct": float(np.mean(y_true) * 100),
        })
    if not rows:
        raise ValueError("No rows with valid actual directions and probability scores were found.")
    return pd.DataFrame(rows).sort_values("brier_score", na_position="last").reset_index(drop=True)


def metrics_column(df: pd.DataFrame, base: str) -> str | None:
    """Find a metrics table column by normalized base (case/underscore insensitive)."""
    return _resolve_column(df, normalized_header(base))
