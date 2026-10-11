from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_utils import FEATURES, download_prices
from src.forecasting import make_windows, reconstruct_close
from src.upload_analysis import (
    clean_historical_data,
    detect_upload_type,
    historical_summary,
    metrics_column,
    normalized_header,
    prediction_column_map,
    prediction_metrics,
    probability_column_map,
    probability_score_metrics,
    excel_sheet_names,
    read_uploaded_table,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("FTF_MODELS_DIR", str(ROOT / "models"))).expanduser().resolve()
RESULTS_DIR = Path(os.getenv("FTF_RESULTS_DIR", str(ROOT / "results"))).expanduser().resolve()
MODEL_LABELS = {
    "cnn": "CNN",
    "rnn": "Vanilla RNN",
    "lstm": "LSTM",
    "cnn_lstm": "CNN–LSTM",
}

st.set_page_config(page_title="Financial Trend Forecasting", page_icon="📈", layout="wide")
st.title("Hybrid Deep Learning for Financial Trend Forecasting")
st.caption("Research and educational dashboard · Forecasts are uncertain and are not investment advice")


def discover_runs() -> list[tuple[Path, dict]]:
    discovered: list[tuple[Path, dict]] = []
    for path in MODEL_DIR.glob("*/*/config.json"):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            if config.get("run_id") and config.get("ticker") and config.get("models"):
                discovered.append((path, config))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(discovered, key=lambda item: item[1].get("trained_at_utc", ""), reverse=True)


@st.cache_data(ttl=900, show_spinner=False)
def get_market_data(symbol: str, requested_period: str) -> pd.DataFrame:
    return download_prices(symbol, requested_period)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    # Lazy import lets the CSV analysis workspace open even if TensorFlow is not installed yet.
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def get_next_nyse_session(last_observation_date: str) -> str | None:
    start_date = pd.Timestamp(last_observation_date).normalize() + pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=14)
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start_date.date(), end_date=end_date.date())
    return None if schedule.empty else schedule.index[0].date().isoformat()


def _price_label(stem: str) -> str:
    return {
        "cnn": "CNN", "rnn": "Vanilla RNN", "lstm": "LSTM", "cnn_lstm": "CNN–LSTM",
        "persistence": "Price persistence baseline",
    }.get(stem, stem.replace("_", " ").title())


def render_historical_upload(raw: pd.DataFrame) -> None:
    try:
        history, price_source = clean_historical_data(raw)
        summary = historical_summary(history)
    except Exception as exc:
        st.error(f"Couldn't analyze this as historical price data: {exc}")
        st.info("Expected columns include Date and Close (or Adj Close / Price). Open, High, Low and Volume are optional.")
        return

    st.success(f"Detected historical price data · {len(history):,} clean observations · price column: {price_source}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Observations", f"{summary['observations']:,}")
    m2.metric("Total return", f"{summary['total_return_pct']:+.2f}%")
    m3.metric("Annualized volatility", f"{summary['annualized_volatility_pct']:.2f}%")
    m4.metric("Maximum drawdown", f"{summary['max_drawdown_pct']:.2f}%")
    m5.metric("Positive sessions", f"{summary['positive_day_share_pct']:.1f}%")
    st.caption(f"Period: {summary['first_date']} to {summary['last_date']} · First/last price: ${summary['first_price']:,.2f} / ${summary['last_price']:,.2f}")

    show_rows = st.slider("Rows to display", min_value=10, max_value=max(10, min(500, len(history))), value=min(100, max(10, len(history))), key="upload_history_rows")
    with st.expander("Cleaned data preview", expanded=False):
        st.dataframe(history.tail(show_rows).reset_index(), use_container_width=True, hide_index=True)
        st.download_button("Download cleaned data with indicators", history.to_csv(index=True).encode("utf-8"), file_name="cleaned_price_analysis.csv", mime="text/csv", key="download_cleaned_history")

    price_tab, perf_tab, returns_tab, risk_tab, distribution_tab, relation_tab = st.tabs([
        "Price & indicators", "Relative performance", "Returns & volatility", "Drawdown & risk", "Distribution", "Relationships"
    ])
    with price_tab:
        min_points = min(60, len(history))
        max_points = st.slider("Chart history", min_value=min_points, max_value=len(history), value=min(250, len(history)), key="upload_chart_history")
        chart_data = history.tail(max_points)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.76, 0.24])
        is_ohlc = all(col in chart_data.columns for col in ("Open", "High", "Low", "Close")) and price_source.lower().replace("_", " ") == "close"
        if is_ohlc:
            fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"], low=chart_data["Low"], close=chart_data["Close"], name="OHLC", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Price"], mode="lines", name=f"Price ({price_source})", line=dict(color="#2563eb", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA 20"], name="20-session SMA", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA 50"], name="50-session SMA", line=dict(color="#7c3aed", width=1.5)), row=1, col=1)
        if "Volume" in chart_data.columns:
            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data["Volume"], name="Volume", marker_color="#94a3b8", opacity=0.75), row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_layout(height=650, title="Price action, trailing averages and volume", hovermode="x unified", xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Moving averages are trailing descriptive indicators and should not be interpreted as future signals.")

    with perf_tab:
        base_price = float(history["Price"].iloc[0])
        indexed = history["Price"] / base_price * 100.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=indexed, mode="lines", name="Price index (start = 100)", line=dict(color="#2563eb", width=2.2)))
        fig.add_hline(y=100, line_dash="dash", line_color="#64748b")
        fig.update_layout(title="Price-only relative performance", height=420, xaxis_title="Date", yaxis_title="Index value", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        window = st.selectbox("Rolling risk window (sessions)", [20, 60, 120], index=1, key="upload_rolling_window")
        returns = history["Daily Return"]
        rolling_mean = returns.rolling(window, min_periods=max(5, window // 2)).mean()
        rolling_std = returns.rolling(window, min_periods=max(5, window // 2)).std(ddof=1)
        rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=(f"{window}-session rolling Sharpe (approx.)", f"{window}-session mean return (annualized, %)"))
        fig.add_trace(go.Scatter(x=history.index, y=rolling_sharpe, mode="lines", name="Rolling Sharpe", line=dict(color="#7c3aed", width=1.8)), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#64748b", row=1, col=1)
        fig.add_trace(go.Scatter(x=history.index, y=rolling_mean * 252 * 100, mode="lines", name="Annualized mean return", line=dict(color="#0d9488", width=1.8)), row=2, col=1)
        fig.update_layout(height=560, hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Price index excludes dividends, transaction costs and financing. Rolling Sharpe is a descriptive statistic, not a forecast.")

    with returns_tab:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Daily return (%)", "20-session annualized rolling volatility (%)"))
        daily_returns_pct = history["Daily Return"] * 100
        colors = np.where(daily_returns_pct.fillna(0) >= 0, "#16a34a", "#dc2626")
        fig.add_trace(go.Bar(x=history.index, y=daily_returns_pct, name="Daily return", marker_color=colors), row=1, col=1)
        fig.add_trace(go.Scatter(x=history.index, y=history["Rolling Volatility (20d, annualized)"], name="Rolling volatility", line=dict(color="#7c3aed", width=2)), row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_layout(height=600, hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with risk_tab:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history["Drawdown"], mode="lines", name="Drawdown", fill="tozeroy", line=dict(color="#dc2626", width=1.7), fillcolor="rgba(220,38,38,0.15)"))
        fig.update_layout(title="Drawdown from the running peak", height=360, xaxis_title="Date", yaxis_title="Drawdown (%)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("Best day", f"{summary['best_day_pct']:+.2f}%")
        r2.metric("Worst day", f"{summary['worst_day_pct']:+.2f}%")
        r3.metric("Approx. annualized Sharpe", f"{summary['annualized_sharpe_approx']:.2f}" if np.isfinite(summary['annualized_sharpe_approx']) else "N/A")
        st.caption("Approximate Sharpe assumes a zero risk-free rate and 252 sessions/year. Historical drawdown does not bound future losses.")

    with distribution_tab:
        rets = (history["Daily Return"].dropna() * 100).astype(float)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Daily return distribution", "Daily return box plot"))
        fig.add_trace(go.Histogram(x=rets, nbinsx=50, name="Daily returns", marker_color="#2563eb", opacity=0.8), row=1, col=1)
        fig.add_trace(go.Box(x=rets, name="Daily returns", marker_color="#7c3aed", boxpoints="outliers", orientation="h"), row=1, col=2)
        fig.update_xaxes(title_text="Daily return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Daily return (%)", row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with relation_tab:
        candidate_columns = [c for c in ["Price", "Daily Return", "SMA 20", "SMA 50", "Rolling Volatility (20d, annualized)", "Volume", "Open", "High", "Low", "Close"] if c in history.columns]
        defaults = [c for c in ["Price", "Daily Return", "Volume", "Rolling Volatility (20d, annualized)"] if c in candidate_columns]
        selected_cols = st.multiselect("Columns for correlation analysis", candidate_columns, default=defaults[:min(5, len(defaults))], key="history_corr_cols")
        if len(selected_cols) >= 2:
            corr = history[selected_cols].corr(numeric_only=True)
            fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu", text=np.round(corr.to_numpy(), 2), texttemplate="%{text}"))
            fig.update_layout(title="Pearson correlation between selected series", height=max(420, 55 * len(selected_cols)))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Correlation describes linear co-movement in the uploaded sample; it does not establish causality.")
        else:
            st.info("Select at least two numeric columns to see a correlation heatmap.")

def render_prediction_upload(raw: pd.DataFrame) -> None:
    try:
        performance = prediction_metrics(raw)
    except Exception as exc:
        st.error(f"Couldn't analyze this as model prediction results: {exc}")
        return
    st.success("Detected model prediction results (for example, test_predictions.csv).")
    st.dataframe(performance, use_container_width=True, hide_index=True)
    price_cols = prediction_column_map(raw)
    prob_cols = probability_column_map(raw)
    actual_col = next((c for c in raw.columns if normalized_header(c) == "actualclose"), None)
    date_col = next((c for c in raw.columns if normalized_header(c) in {"date", "datetime", "timestamp"}), None)

    if price_cols and actual_col:
        metrics_mae = performance.dropna(subset=["price_MAE_USD"])
        default_label = str(metrics_mae.iloc[0]["model"]) if not metrics_mae.empty else next(iter(price_cols))
        if default_label not in price_cols:
            default_label = next(iter(price_cols))
        selected_series_label = st.selectbox("Prediction series", list(price_cols), index=list(price_cols).index(default_label), key="uploaded_prediction_series")
        selected_column = price_cols[selected_series_label]
        actual = pd.to_numeric(raw[actual_col], errors="coerce")
        predicted = pd.to_numeric(raw[selected_column], errors="coerce")
        plot_data = pd.DataFrame({"Actual close": actual, "Predicted close": predicted})
        if date_col:
            plot_data.index = pd.to_datetime(raw[date_col], errors="coerce")
            plot_data.index.name = "Date"
        else:
            plot_data.index = pd.RangeIndex(1, len(plot_data) + 1, name="Observation")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        if not plot_data.empty:
            residuals = plot_data["Predicted close"] - plot_data["Actual close"]
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(np.square(residuals))))
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Compared observations", f"{len(plot_data):,}")
            p2.metric("MAE", f"${mae:,.3f}")
            p3.metric("RMSE", f"${rmse:,.3f}")
            p4.metric("Mean signed error", f"${residuals.mean():+,.3f}")
            plot_tab, scatter_tab, error_tab, residual_tab = st.tabs(["Actual vs predicted", "Prediction scatter", "Errors over time", "Error distribution"])
            with plot_tab:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["Actual close"], mode="lines", name="Actual close", line=dict(color="#111827", width=2.2)))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["Predicted close"], mode="lines", name=f"{selected_series_label} prediction", line=dict(color="#2563eb", width=1.8, dash="dash")))
                fig.update_layout(title=f"Actual vs predicted closing price · {selected_series_label}", height=460, xaxis_title="Date / observation", yaxis_title="Price", hovermode="x unified", xaxis_rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True)
            with scatter_tab:
                lo = float(min(plot_data["Actual close"].min(), plot_data["Predicted close"].min()))
                hi = float(max(plot_data["Actual close"].max(), plot_data["Predicted close"].max()))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_data["Actual close"], y=plot_data["Predicted close"], mode="markers", name="Observations", marker=dict(size=7, opacity=0.65, color="#2563eb")))
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction", line=dict(dash="dash", color="#dc2626")))
                fig.update_layout(title="Predicted vs actual values", height=440, xaxis_title="Actual close", yaxis_title="Predicted close", hovermode="closest")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points closer to the dashed diagonal have smaller absolute price errors.")
            with error_tab:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Pointwise residual", "Rolling 20-observation MAE"))
                err_colors = np.where(residuals >= 0, "#16a34a", "#dc2626")
                fig.add_trace(go.Bar(x=plot_data.index, y=residuals, name="Predicted − actual", marker_color=err_colors), row=1, col=1)
                rolling_abs_error = residuals.abs().rolling(20, min_periods=5).mean()
                fig.add_trace(go.Scatter(x=plot_data.index, y=rolling_abs_error, mode="lines", name="Rolling MAE", line=dict(color="#7c3aed", width=2)), row=2, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="#64748b", row=1, col=1)
                fig.update_layout(title="Forecast error diagnostics", height=560, xaxis_title="Date / observation", yaxis_title="Error (price units)", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            with residual_tab:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Residual histogram", "Residual box plot"))
                fig.add_trace(go.Histogram(x=residuals, nbinsx=40, name="Residuals", marker_color="#2563eb"), row=1, col=1)
                fig.add_trace(go.Box(x=residuals, name="Residuals", marker_color="#7c3aed", boxpoints="outliers", orientation="h"), row=1, col=2)
                fig.update_layout(title="Forecast error distribution", height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download analyzed prediction metrics", performance.to_csv(index=False).encode("utf-8"), file_name="uploaded_prediction_metrics.csv", mime="text/csv", key="download_upload_prediction_metrics")
        else:
            st.warning("No overlapping valid actual and predicted closing prices were found.")

    actual_dir_col = next((c for c in raw.columns if normalized_header(c) == "actualdirectionup"), None)
    if actual_dir_col and prob_cols:
        st.subheader("Direction score diagnostics")
        try:
            score_table = probability_score_metrics(raw)
            st.dataframe(score_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download probability diagnostics",
                score_table.to_csv(index=False).encode("utf-8"),
                file_name="uploaded_probability_diagnostics.csv",
                mime="text/csv",
                key="download_upload_probability_diagnostics",
            )
        except ValueError as exc:
            st.info(str(exc))
            score_table = pd.DataFrame()
        selected_prob_label = st.selectbox("Direction score model", list(prob_cols), key="uploaded_direction_score_model")
        score_col = prob_cols[selected_prob_label]
        actual = pd.to_numeric(raw[actual_dir_col], errors="coerce")
        score = pd.to_numeric(raw[score_col], errors="coerce")
        mask = actual.isin([0, 1]) & score.between(0, 1) & np.isfinite(score)
        y_true = actual[mask].astype(int).to_numpy()
        y_score = score[mask].astype(float).to_numpy()
        threshold_col = next((c for c in raw.columns if normalized_header(c) == normalized_header(score_col.replace("up_probability", "direction_threshold"))), None)
        default_threshold = 0.5
        if threshold_col and pd.to_numeric(raw[threshold_col], errors="coerce").notna().any():
            default_threshold = float(np.clip(pd.to_numeric(raw[threshold_col], errors="coerce").dropna().iloc[0], 0, 1))
        else:
            candidate_threshold_col = next((c for c in raw.columns if normalized_header(c) == normalized_header(selected_prob_label.replace("–", "_").replace(" ", "_") + "_direction_threshold")), None)
            if candidate_threshold_col and pd.to_numeric(raw[candidate_threshold_col], errors="coerce").notna().any():
                default_threshold = float(np.clip(pd.to_numeric(raw[candidate_threshold_col], errors="coerce").dropna().iloc[0], 0, 1))
        threshold = st.slider("Exploratory direction threshold", 0.0, 1.0, float(default_threshold), 0.01, key="uploaded_direction_threshold")
        st.caption("This slider is for diagnostics only. Do not select a threshold using a held-out test set and report the resulting score as unbiased evaluation.")
        if len(y_true):
            predicted_direction = (y_score >= threshold).astype(int)
            from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
            a, b, c, d = st.columns(4)
            a.metric("Threshold", f"{threshold:.2f}")
            a_rate = float(np.mean(predicted_direction == 1) * 100)
            b.metric("Predicted UP share", f"{a_rate:.1f}%")
            c.metric("Balanced accuracy", f"{balanced_accuracy_score(y_true, predicted_direction):.3f}")
            d.metric("F1 score", f"{f1_score(y_true, predicted_direction, zero_division=0):.3f}")
            cm = confusion_matrix(y_true, predicted_direction, labels=[0, 1])
            available_tabs = st.tabs(["Confusion matrix", "Score distribution", "Calibration diagnostic"])
            with available_tabs[0]:
                fig = go.Figure(go.Heatmap(z=cm, x=["Predicted DOWN", "Predicted UP"], y=["Actual DOWN", "Actual UP"], text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
                fig.update_layout(title="Direction confusion matrix", height=370, xaxis_title="Prediction", yaxis_title="Actual class")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Precision: {precision_score(y_true, predicted_direction, zero_division=0):.3f} · Recall: {recall_score(y_true, predicted_direction, zero_division=0):.3f}")
            with available_tabs[1]:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=y_score[y_true == 0], nbinsx=25, name="Actual DOWN", opacity=0.65, marker_color="#dc2626"))
                fig.add_trace(go.Histogram(x=y_score[y_true == 1], nbinsx=25, name="Actual UP", opacity=0.65, marker_color="#16a34a"))
                fig.update_layout(title="Upward score by actual class", barmode="overlay", height=380, xaxis_title="Model upward score", yaxis_title="Count", xaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)
            with available_tabs[2]:
                score_frame = pd.DataFrame({"score": y_score, "actual": y_true})
                score_frame["bin"] = pd.cut(score_frame["score"], bins=np.linspace(0, 1, 11), include_lowest=True)
                calibration = score_frame.groupby("bin", observed=False).agg(mean_score=("score", "mean"), observed_up_rate=("actual", "mean"), count=("actual", "size")).dropna()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=calibration["mean_score"], y=calibration["observed_up_rate"], mode="lines+markers+text", text=calibration["count"].astype(int).astype(str), textposition="top center", name="Observed bins"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal calibration", line=dict(dash="dash", color="#64748b")))
                fig.update_layout(title="Reliability diagram (labels show samples per bin)", height=420, xaxis_title="Mean model score", yaxis_title="Observed UP rate", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Calibration is diagnostic only; bins with few samples are noisy. Sigmoid scores are not automatically probabilities.")
    elif actual_dir_col:
        st.info("No upward probability columns were detected, so score calibration charts are unavailable. Hard-label direction metrics are shown in the table above when present.")
    st.caption("Uploaded-file metrics describe only the uploaded sample. They are not evidence of future performance or an investment recommendation.")

def render_metrics_upload(raw: pd.DataFrame) -> None:
    st.success("Detected a model metrics or walk-forward summary CSV.")
    st.dataframe(raw, use_container_width=True, hide_index=True)
    model_col = next((c for c in raw.columns if normalized_header(c) == "model"), None)
    if model_col is None:
        st.info("No model column found for charting.")
        return
    labels = raw[model_col].astype(str)
    price_mae = next((c for c in raw.columns if normalized_header(c) in {"pricemaeusd", "pricemaeusdmean"}), None)
    price_rmse = next((c for c in raw.columns if normalized_header(c) in {"pricermseusd", "pricermseusdmean"}), None)
    direction_balanced = next((c for c in raw.columns if normalized_header(c) in {"directionbalancedaccuracy", "directionbalancedaccuracymean"}), None)
    direction_f1 = next((c for c in raw.columns if normalized_header(c) in {"directionf1", "directionf1mean"}), None)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        if price_mae:
            fig.add_trace(go.Bar(x=labels, y=pd.to_numeric(raw[price_mae], errors="coerce"), name="MAE", marker_color="#2563eb"))
        if price_rmse:
            fig.add_trace(go.Bar(x=labels, y=pd.to_numeric(raw[price_rmse], errors="coerce"), name="RMSE", marker_color="#f59e0b"))
        if price_mae or price_rmse:
            fig.update_layout(title="Price error comparison · lower is better", barmode="group", height=420, xaxis_title="Model / baseline", yaxis_title="Error (USD)", legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price MAE/RMSE columns were recognized in this file.")
    with right:
        fig = go.Figure()
        if direction_balanced:
            fig.add_trace(go.Bar(x=labels, y=pd.to_numeric(raw[direction_balanced], errors="coerce"), name="Balanced accuracy", marker_color="#0d9488"))
        if direction_f1:
            fig.add_trace(go.Bar(x=labels, y=pd.to_numeric(raw[direction_f1], errors="coerce"), name="F1", marker_color="#7c3aed"))
        if direction_balanced or direction_f1:
            fig.update_layout(title="Direction classification · higher is better", barmode="group", height=420, xaxis_title="Model / baseline", yaxis_title="Score", yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No direction score columns were recognized in this file.")

    fold_col = next((c for c in raw.columns if normalized_header(c) == "fold"), None)
    if fold_col:
        st.subheader("Fold-by-fold stability")
        metric_options = [c for c in raw.columns if normalized_header(c) in {"pricemaeusd", "pricermseusd", "directionbalancedaccuracy", "directionf1"}]
        if metric_options:
            chosen_metric = st.selectbox("Metric across folds", metric_options, key="uploaded_fold_metric")
            chart_frame = raw[[model_col, fold_col, chosen_metric]].copy()
            chart_frame[chosen_metric] = pd.to_numeric(chart_frame[chosen_metric], errors="coerce")
            fig = go.Figure()
            for name, group in chart_frame.groupby(model_col):
                group = group.sort_values(fold_col)
                fig.add_trace(go.Scatter(x=group[fold_col].astype(str), y=group[chosen_metric], mode="lines+markers", name=str(name)))
            fig.update_layout(title=f"{chosen_metric} by fold", height=420, xaxis_title="Fold", yaxis_title=chosen_metric, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)


def render_generic_upload(raw: pd.DataFrame) -> None:
    st.info("This file was not recognized as stock history, model predictions, or an evaluation metrics export. You can still explore numeric relationships.")
    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    left, right, third = st.columns(3)
    left.metric("Rows", f"{len(raw):,}")
    right.metric("Columns", f"{len(raw.columns):,}")
    third.metric("Numeric columns", f"{len(numeric_cols):,}")
    with st.expander("Data preview and missing values", expanded=False):
        st.dataframe(raw.head(100), use_container_width=True, hide_index=True)
        missing = raw.isna().sum().rename("missing_count").to_frame()
        missing["missing_pct"] = (raw.isna().mean() * 100).round(2)
        st.dataframe(missing.sort_values("missing_pct", ascending=False), use_container_width=True)
        st.download_button("Download uploaded data as CSV", raw.to_csv(index=False).encode("utf-8"), file_name="uploaded_data.csv", mime="text/csv", key="download_generic_data")
    if not numeric_cols:
        st.warning("No numeric columns were detected. Check delimiters and decimal formatting in the file.")
        return
    line_tab, scatter_tab, distribution_tab, corr_tab = st.tabs(["Time series / row trend", "Scatter relationship", "Distribution", "Correlation matrix"])
    date_col = next((c for c in raw.columns if normalized_header(c) in {"date", "datetime", "timestamp", "time"}), None)
    plot_frame = raw.copy()
    if date_col:
        plot_frame[date_col] = pd.to_datetime(plot_frame[date_col], errors="coerce")
        plot_frame = plot_frame.sort_values(date_col)
        x_values = plot_frame[date_col]
        x_title = date_col
    else:
        x_values = np.arange(1, len(plot_frame) + 1)
        x_title = "Row number"
    with line_tab:
        selected = st.multiselect("Numeric columns to trend", numeric_cols, default=numeric_cols[: min(4, len(numeric_cols))], key="generic_numeric_plot")
        if selected:
            fig = go.Figure()
            for col in selected:
                fig.add_trace(go.Scatter(x=x_values, y=pd.to_numeric(plot_frame[col], errors="coerce"), mode="lines", name=str(col)))
            fig.update_layout(title="Selected numeric trends", height=450, xaxis_title=x_title, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
    with scatter_tab:
        if len(numeric_cols) >= 2:
            x_col, y_col = st.columns(2)
            scatter_x = x_col.selectbox("X variable", numeric_cols, index=0, key="generic_scatter_x")
            default_y = 1 if len(numeric_cols) > 1 else 0
            scatter_y = y_col.selectbox("Y variable", numeric_cols, index=default_y, key="generic_scatter_y")
            x_num = pd.to_numeric(plot_frame[scatter_x], errors="coerce")
            y_num = pd.to_numeric(plot_frame[scatter_y], errors="coerce")
            valid = x_num.notna() & y_num.notna() & np.isfinite(x_num) & np.isfinite(y_num)
            corr_value = x_num[valid].corr(y_num[valid]) if valid.sum() >= 2 else np.nan
            st.metric("Pearson correlation", f"{corr_value:.3f}" if np.isfinite(corr_value) else "N/A", help="Correlation is descriptive, not causal.")
            fig = go.Figure(go.Scatter(x=x_num[valid], y=y_num[valid], mode="markers", marker=dict(color="#2563eb", opacity=0.65), name="Rows"))
            fig.update_layout(title=f"{scatter_y} versus {scatter_x}", height=430, xaxis_title=scatter_x, yaxis_title=scatter_y, hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numeric columns are needed for a scatter plot.")
    with distribution_tab:
        hist_col = st.selectbox("Inspect the distribution of", numeric_cols, key="generic_hist_column")
        hist_values = pd.to_numeric(plot_frame[hist_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box plot"))
        fig.add_trace(go.Histogram(x=hist_values, nbinsx=40, marker_color="#2563eb", name="Values"), row=1, col=1)
        fig.add_trace(go.Box(x=hist_values, orientation="h", marker_color="#7c3aed", boxpoints="outliers", name="Values"), row=1, col=2)
        fig.update_layout(title=f"Distribution · {hist_col}", height=390, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with corr_tab:
        corr_cols = st.multiselect("Columns in correlation matrix", numeric_cols, default=numeric_cols[: min(12, len(numeric_cols))], key="generic_corr_cols")
        if len(corr_cols) >= 2:
            corr = plot_frame[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
            fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu", text=np.round(corr.to_numpy(), 2), texttemplate="%{text}"))
            fig.update_layout(title="Numeric feature correlations", height=max(430, 40 * len(corr_cols)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Choose at least two columns. Correlation is only calculated over rows with paired numeric observations.")

def render_upload_workspace() -> None:
    st.header("Upload & analyze data")
    st.write("Upload historical prices, saved model predictions, model metrics or any numeric dataset. Analysis is performed in memory for this session; uploads are not automatically saved on the server.")
    st.caption("Supported formats: CSV and Excel (.xlsx). Historical files work best with Date plus Close / Adj Close / Price; OHLCV columns enable candlestick and volume analysis.")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], help="Maximum upload size is 25 MB. Excel workbooks can contain multiple sheets.", key="analysis_csv_uploader")
    if uploaded_file is None:
        st.markdown("**Examples you can analyze**")
        st.write("Historical OHLCV prices · this project's `test_predictions.csv` · `metrics.csv` · `walk_forward_metrics.csv` · `walk_forward_summary.csv` · general numeric data")
        sample_history = pd.DataFrame({"Date": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"], "Open": [100, 102, 101, 104], "High": [103, 103, 105, 106], "Low": [99, 100, 100, 103], "Close": [102, 101, 104, 105], "Volume": [1200000, 980000, 1310000, 1150000]})
        st.download_button("Download sample OHLCV template", sample_history.to_csv(index=False).encode("utf-8"), file_name="sample_ohlcv_template.csv", mime="text/csv", key="download_sample_ohlcv")
        st.info("Choose a file above to begin. CSV analysis does not require trained neural network models.")
        return
    payload = uploaded_file.getvalue()
    if len(payload) > 25 * 1024 * 1024:
        st.error("This file exceeds the 25 MB analysis limit. Please upload a smaller subset.")
        return
    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        if suffix == ".xlsx":
            sheet_names = excel_sheet_names(payload)
            if not sheet_names:
                st.error("The Excel workbook contains no worksheets.")
                return
            selected_sheet = st.selectbox("Worksheet to analyze", sheet_names, key="upload_excel_sheet") if len(sheet_names) > 1 else sheet_names[0]
            raw = read_uploaded_table(uploaded_file.name, payload, sheet_name=selected_sheet)
        elif suffix == ".csv":
            raw = read_uploaded_table(uploaded_file.name, payload)
        else:
            st.error("Unsupported file format. Upload a CSV or .xlsx file.")
            return
        if raw.empty:
            st.error("The uploaded file has no data rows.")
            return
        if len(raw) > 250_000:
            st.error("This upload has more than 250,000 rows. Upload a smaller subset for interactive analysis.")
            return
        if len(raw.columns) > 200:
            st.error("This upload has more than 200 columns. Reduce the number of columns and try again.")
            return
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.info("For Excel files, save as .xlsx. For CSV, check that the first row contains column names and values use consistent delimiters.")
        return
    kind = detect_upload_type(raw)
    st.caption(f"File: {uploaded_file.name} · Type: {suffix[1:].upper()} · {len(raw):,} rows · {len(raw.columns):,} columns · Analysis type: {kind}")
    with st.expander("Raw file preview", expanded=False):
        st.dataframe(raw.head(100), use_container_width=True, hide_index=True)
    if kind == "history":
        render_historical_upload(raw)
    elif kind == "predictions":
        render_prediction_upload(raw)
    elif kind == "metrics":
        render_metrics_upload(raw)
    else:
        render_generic_upload(raw)


workspace = st.sidebar.radio("Workspace", ["Forecast dashboard", "Upload & analyze data"], index=0)
if workspace == "Upload & analyze data":
    render_upload_workspace()
    st.stop()

runs = discover_runs()
if not runs:
    st.warning("No new-format trained runs were found. You can switch Workspace in the sidebar to upload a CSV for analysis, or train a model run first.")
    st.code("python train.py --ticker AAPL --period 5y --epochs 20\nstreamlit run app.py", language="powershell")
    st.info("This version saves artifacts in models/TICKER/RUN_ID/. Old flat model files must be retrained.")
    st.stop()

ticker_list = sorted({config["ticker"] for _, config in runs})
selected_ticker = st.sidebar.selectbox("Stock ticker", ticker_list)
ticker_runs = [(path, config) for path, config in runs if config["ticker"] == selected_ticker]
run_paths = [path for path, _ in ticker_runs]
run_configs = {path: config for path, config in ticker_runs}


def format_run(path: Path) -> str:
    run_config = run_configs[path]
    trained = run_config.get("trained_at_utc", "unknown time").replace("T", " ")[:19]
    return f"{run_config['run_id']} · trained {trained} · data through {run_config.get('last_date', 'unknown')}"


selected_config_path = st.sidebar.selectbox("Training run", run_paths, format_func=format_run, help="Different runs keep separate models and results.")
config = run_configs[selected_config_path]
run_id = config["run_id"]
run_model_dir = selected_config_path.parent
run_results_dir = RESULTS_DIR / config["ticker"] / run_id
model_choices = [name for name in MODEL_LABELS if name in config["models"]]
model_choice = st.sidebar.selectbox("Forecast model", model_choices, format_func=lambda name: MODEL_LABELS[name])
period = st.sidebar.selectbox("Recent chart history", ["6mo", "1y", "2y", "5y"], index=1)

try:
    with st.spinner(f"Loading recent market data for {selected_ticker}..."):
        prices = get_market_data(selected_ticker, period)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.stop()

model_path = run_model_dir / f"{model_choice}.keras"
feature_scaler_path = run_model_dir / "feature_scaler.joblib"
target_scaler_path = run_model_dir / "target_scaler.joblib"
missing_artifacts = [p.name for p in (model_path, feature_scaler_path, target_scaler_path) if not p.exists()]
if missing_artifacts:
    st.error(f"This run is missing required files: {', '.join(missing_artifacts)}. Retrain this run.")
    st.stop()

try:
    feature_scaler = load_scaler(str(feature_scaler_path))
    target_scaler = load_scaler(str(target_scaler_path))
    model = load_model(str(model_path))
    if config.get("features") != FEATURES:
        raise ValueError("Feature columns/order differ from the selected model's metadata.")
    feature_values = prices[FEATURES].to_numpy(dtype=np.float64)
    scaled_features = feature_scaler.transform(feature_values).astype(np.float32)
    lookback = int(config["lookback"])
    if len(scaled_features) < lookback:
        raise ValueError(f"Need at least {lookback} recent observations to forecast.")
    sequence = make_windows(scaled_features, np.asarray([len(scaled_features)]), lookback)
    prediction = model.predict(sequence, verbose=0)
    predicted_return_scaled = np.asarray(prediction["log_return"]).reshape(-1, 1)
    predicted_log_return = float(target_scaler.inverse_transform(predicted_return_scaled)[0, 0])
    last_close = float(prices["Close"].iloc[-1])
    predicted_close = float(reconstruct_close(np.asarray([last_close]), np.asarray([predicted_log_return]))[0])
    up_probability = float(np.asarray(prediction["direction"]).reshape(-1)[0])
    direction_thresholds = config.get("direction_thresholds", {})
    direction_threshold = float(direction_thresholds.get(model_choice, 0.5))
    if not 0.0 <= up_probability <= 1.0:
        raise ValueError("Model returned a direction output outside [0, 1].")
except Exception as exc:
    st.error(f"Could not create forecast: {exc}")
    st.stop()

last_date_timestamp = pd.Timestamp(prices.index[-1])
last_date_iso = last_date_timestamp.strftime("%Y-%m-%d")
try:
    next_session_iso = get_next_nyse_session(last_date_iso)
except Exception:
    next_session_iso = None
change = predicted_close - last_close
change_pct = change / last_close * 100 if last_close else 0.0
st.subheader(f"{selected_ticker} · Forecast based on latest available close ({last_date_timestamp:%b %d, %Y})")
if next_session_iso:
    st.caption(f"Forecast target session: {pd.Timestamp(next_session_iso):%A, %B %d, %Y} (NYSE calendar)")
else:
    st.caption("Forecast target: next available NYSE session after the latest data date; exact date unavailable.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest adjusted close", f"${last_close:,.2f}")
col2.metric("Predicted next close", f"${predicted_close:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
col3.metric("Model upward score", f"{up_probability:.1%}")
col4.metric("Predicted direction", "UP" if up_probability >= direction_threshold else "DOWN")
st.caption(f"Predicted next-session log return: {predicted_log_return:+.3%}. Direction threshold selected on validation: {direction_threshold:.3f}. The upward score is not calibrated as a real-world probability. Forecasts are uncertain and are not trading recommendations.")
if (predicted_log_return > 0) != (up_probability >= direction_threshold):
    st.warning("The return head and direction-classification head imply different directions. This is possible because they are separately learned outputs.")

st.subheader("Market graph analysis")
chart_tab, return_tab, drawdown_tab = st.tabs(["Price, moving averages & volume", "Returns & volatility", "Drawdown"])
with chart_tab:
    enriched_prices = prices.copy()
    enriched_prices["SMA 20"] = enriched_prices["Close"].rolling(20, min_periods=1).mean()
    enriched_prices["SMA 50"] = enriched_prices["Close"].rolling(50, min_periods=1).mean()
    chart_prices = enriched_prices.tail(180)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.76, 0.24])
    fig.add_trace(go.Scatter(x=chart_prices.index, y=chart_prices["Close"], mode="lines", name="Adjusted close", line=dict(color="#111827", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_prices.index, y=chart_prices["SMA 20"], mode="lines", name="20-session SMA", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_prices.index, y=chart_prices["SMA 50"], mode="lines", name="50-session SMA", line=dict(color="#7c3aed", width=1.5)), row=1, col=1)
    if next_session_iso:
        fig.add_trace(go.Scatter(x=[last_date_timestamp, pd.Timestamp(next_session_iso)], y=[last_close, predicted_close], mode="lines+markers", name=f"{MODEL_LABELS[model_choice]} next-session forecast", line=dict(color="#0d9488", width=2, dash="dash"), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Bar(x=chart_prices.index, y=chart_prices["Volume"], name="Volume", marker_color="#94a3b8", opacity=0.7), row=2, col=1)
    fig.update_yaxes(title_text="Adjusted price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(height=600, hovermode="x unified", xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
with return_tab:
    ret = prices["Close"].pct_change(fill_method=None) * 100
    vol20 = ret.rolling(20, min_periods=5).std() * np.sqrt(252)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Daily return (%)", "20-session annualized volatility (%)"))
    ret_colors = np.where(ret.fillna(0) >= 0, "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(x=prices.index, y=ret, marker_color=ret_colors, name="Daily return"), row=1, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=vol20 * 100, line=dict(color="#7c3aed", width=2), name="Annualized volatility"), row=2, col=1)
    fig.update_layout(height=560, hovermode="x unified", showlegend=False)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
with drawdown_tab:
    close = prices["Close"]
    drawdown = (close / close.cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", name="Drawdown", fill="tozeroy", line=dict(color="#dc2626", width=1.5), fillcolor="rgba(220,38,38,0.15)"))
    fig.update_layout(title="Drawdown from previous peak", height=380, xaxis_title="Date", yaxis_title="Drawdown (%)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Maximum drawdown in selected history", f"{drawdown.min():.2f}%")

st.subheader("Held-out test-set comparison")
metrics_path = run_results_dir / "metrics.csv"
predictions_path = run_results_dir / "test_predictions.csv"
walk_forward_summary_path = run_results_dir / "walk_forward_summary.csv"
walk_forward_path = run_results_dir / "walk_forward_metrics.csv"

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    model_col = next((c for c in metrics.columns if normalized_header(c) == "model"), "model")
    mae_col = next((c for c in metrics.columns if normalized_header(c) == "pricemaeusd"), None)
    rmse_col = next((c for c in metrics.columns if normalized_header(c) == "pricermseusd"), None)
    balanced_col = next((c for c in metrics.columns if normalized_header(c) == "directionbalancedaccuracy"), None)
    f1_col = next((c for c in metrics.columns if normalized_header(c) == "directionf1"), None)
    left, right = st.columns(2)
    with left:
        error_fig = go.Figure()
        if mae_col:
            subset = metrics.dropna(subset=[mae_col]).sort_values(mae_col)
            error_fig.add_trace(go.Bar(x=subset[model_col], y=subset[mae_col], name="MAE", marker_color="#2563eb"))
        if rmse_col:
            subset = metrics.dropna(subset=[rmse_col]).sort_values(rmse_col)
            error_fig.add_trace(go.Bar(x=subset[model_col], y=subset[rmse_col], name="RMSE", marker_color="#f59e0b"))
        error_fig.update_layout(title="Price errors · lower is better", barmode="group", height=430, yaxis_title="Error (USD)", xaxis_title="Model / baseline", legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(error_fig, use_container_width=True)
    with right:
        dir_fig = go.Figure()
        if balanced_col:
            dir_fig.add_trace(go.Bar(x=metrics[model_col], y=metrics[balanced_col], name="Balanced accuracy", marker_color="#0d9488"))
        if f1_col:
            dir_fig.add_trace(go.Bar(x=metrics[model_col], y=metrics[f1_col], name="F1 score", marker_color="#7c3aed"))
        dir_fig.update_layout(title="Direction classification · higher is better", barmode="group", height=430, yaxis=dict(range=[0, 1], title="Score"), xaxis_title="Model / baseline", legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(dir_fig, use_container_width=True)
else:
    st.warning("Held-out metrics are missing for this run.")

if predictions_path.exists():
    test_predictions = pd.read_csv(predictions_path)
    actual_col = next((c for c in test_predictions.columns if normalized_header(c) == "actualclose"), None)
    date_col = next((c for c in test_predictions.columns if normalized_header(c) in {"date", "datetime", "timestamp"}), None)
    pred_cols = {name: col for name, col in prediction_column_map(test_predictions).items() if name != "Price persistence baseline"}
    if actual_col and pred_cols:
        st.subheader("Held-out prediction diagnostics")
        selected_test_model_label = st.selectbox("Model to inspect on the test period", list(pred_cols), key="heldout_graph_model")
        selected_pred_col = pred_cols[selected_test_model_label]
        actual_values = pd.to_numeric(test_predictions[actual_col], errors="coerce")
        predicted_values = pd.to_numeric(test_predictions[selected_pred_col], errors="coerce")
        x_vals = pd.to_datetime(test_predictions[date_col], errors="coerce") if date_col else np.arange(len(test_predictions))
        plot_frame = pd.DataFrame({"x": x_vals, "Actual": actual_values, "Predicted": predicted_values}).replace([np.inf, -np.inf], np.nan).dropna()
        residuals = plot_frame["Predicted"] - plot_frame["Actual"]
        ptab, etab, htab = st.tabs(["Actual vs predicted", "Errors over time", "Error distribution"])
        with ptab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_frame["x"], y=plot_frame["Actual"], mode="lines", name="Actual close", line=dict(color="#111827", width=2)))
            fig.add_trace(go.Scatter(x=plot_frame["x"], y=plot_frame["Predicted"], mode="lines", name=selected_test_model_label, line=dict(color="#2563eb", dash="dash", width=1.8)))
            fig.update_layout(title=f"Held-out actual vs forecast · {selected_test_model_label}", height=450, xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)
        with etab:
            fig = go.Figure(go.Bar(x=plot_frame["x"], y=residuals, marker_color=np.where(residuals >= 0, "#16a34a", "#dc2626"), name="Predicted − actual"))
            fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
            fig.update_layout(title="Forecast residuals over time", height=380, xaxis_title="Date", yaxis_title="Error (USD)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        with htab:
            fig = go.Figure(go.Histogram(x=residuals, nbinsx=40, marker_color="#2563eb", name="Residuals"))
            fig.update_layout(title="Distribution of forecast errors", height=350, xaxis_title="Predicted − actual (USD)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    with st.expander("View/download held-out test predictions"):
        st.dataframe(test_predictions, use_container_width=True, hide_index=True)
        st.download_button("Download test predictions CSV", test_predictions.to_csv(index=False).encode("utf-8"), file_name=f"{selected_ticker}_{run_id}_test_predictions.csv", mime="text/csv")

if walk_forward_summary_path.exists():
    walk_summary = pd.read_csv(walk_forward_summary_path)
    if not walk_summary.empty:
        with st.expander("Walk-forward evaluation across expanding training windows", expanded=True):
            st.caption("These folds use development data only; the final held-out test period remains separate.")
            st.dataframe(walk_summary, use_container_width=True, hide_index=True)
            summary_model_col = next((c for c in walk_summary.columns if normalized_header(c) == "model"), "model")
            wf_mae_col = next((c for c in walk_summary.columns if normalized_header(c) == "pricemaeusdmean"), None)
            wf_rmse_col = next((c for c in walk_summary.columns if normalized_header(c) == "pricermseusdmean"), None)
            if wf_mae_col or wf_rmse_col:
                fig = go.Figure()
                if wf_mae_col:
                    items = walk_summary.dropna(subset=[wf_mae_col]).sort_values(wf_mae_col)
                    fig.add_trace(go.Bar(x=items[summary_model_col], y=items[wf_mae_col], name="Mean MAE", marker_color="#2563eb"))
                if wf_rmse_col:
                    items = walk_summary.dropna(subset=[wf_rmse_col]).sort_values(wf_rmse_col)
                    fig.add_trace(go.Bar(x=items[summary_model_col], y=items[wf_rmse_col], name="Mean RMSE", marker_color="#f59e0b"))
                fig.update_layout(title="Walk-forward average price error · lower is better", barmode="group", height=420, yaxis_title="Error (USD)", xaxis_title="Model / baseline", legend=dict(orientation="h", y=1.02, x=0))
                st.plotly_chart(fig, use_container_width=True)
            if walk_forward_path.exists():
                fold_data = pd.read_csv(walk_forward_path)
                fold_col = next((c for c in fold_data.columns if normalized_header(c) == "fold"), None)
                metric_options = [c for c in fold_data.columns if normalized_header(c) in {"pricemaeusd", "pricermseusd", "directionbalancedaccuracy", "directionf1"}]
                if fold_col and metric_options:
                    selected_wf_metric = st.selectbox("Walk-forward metric by fold", metric_options, key="dashboard_wf_metric")
                    fig = go.Figure()
                    for name, group in fold_data.groupby("model"):
                        group = group.sort_values(fold_col)
                        values = pd.to_numeric(group[selected_wf_metric], errors="coerce")
                        if values.notna().any():
                            fig.add_trace(go.Scatter(x=group[fold_col].astype(str), y=values, mode="lines+markers", name=str(name)))
                    fig.update_layout(title=f"{selected_wf_metric} by fold", height=420, xaxis_title="Fold", yaxis_title=selected_wf_metric, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Walk-forward evaluation was skipped for this run.")
else:
    st.info("No walk-forward summary CSV was found for this run.")

with st.expander("Run metadata and evaluation safeguards"):
    st.write(f"Ticker: {config['ticker']} · Run ID: {run_id}")
    st.write(f"Training data period: {config['first_date']} to {config['last_date']}")
    st.write(f"Latest inference data date: {last_date_iso}")
    st.write(f"Lookback: {lookback} trading sessions · Features: {len(config['features'])}")
    st.write(f"Chronological sample split: {config['train_samples']} train, {config['validation_samples']} validation, {config['test_samples']} final test.")
    st.write(f"Data fingerprint (SHA-256): `{config['data_sha256']}`")
    st.write(f"Direction threshold for {MODEL_LABELS[model_choice]}: {direction_threshold:.3f}")
    st.caption(config.get("direction_threshold_selection", "Legacy run: uses default 0.50 direction threshold."))
    st.write("The return scaler and feature scaler were fitted on training-only data. Baselines and neural networks are evaluated on the same final test period. Past performance does not guarantee future results.")
