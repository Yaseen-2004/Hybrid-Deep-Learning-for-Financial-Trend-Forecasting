# Hybrid Deep Learning for Financial Trend Forecasting

A TensorFlow project comparing **CNN, vanilla RNN, LSTM, and CNN–LSTM** architectures for US-stock next-session forecasting, with a Streamlit dashboard, CSV/XLSX upload analysis, enhanced diagnostics, and deployment files for Docker and Streamlit Community Cloud. It predicts both next-session log return (reconstructed into an adjusted closing-price estimate) and upward/downward direction, compares against transparent baselines, performs expanding-window walk-forward evaluation, and serves the latest run in a Streamlit dashboard.

> Educational/research software only. Forecasts are uncertain; this project is not investment advice, a trading recommendation, or evidence of profitability.

## What's improved in Phase 1

- Models, scalers, configs, data and results are separated by ticker and unique run ID. A run won't overwrite another ticker/run's outputs.
- The regression target is next-session log return: `log(C[t+1] / C[t])`. Predicted close is reconstructed as `C[t] * exp(predicted_log_return)`.
- Up/down classification shares the same target session: `1` if the next adjusted close is higher than the previous observed close, else `0`.
- Training is chronological. Input feature windows stop at the day before their target, scalers are fitted only on each training fold's historical rows/targets, and `shuffle=False` is used.
- Two expanding-window walk-forward folds run on development data by default. The final chronological 15% test segment is held out from walk-forward evaluation and final model fitting.
- Metrics compare models with a price persistence baseline, prior-day direction momentum baseline, and training-majority direction baseline. Metrics not meaningful for a baseline are recorded as blank/NaN instead of fabricated.
- Direction metrics include accuracy, balanced accuracy, precision, recall, and F1. Regression metrics include price MAE and RMSE in USD.
- Each model's binary direction threshold is selected using its chronological validation segment to maximize balanced accuracy, with deterministic tie-breaking. The final held-out test segment is never used to choose the threshold. Threshold selection may still overfit validation data; it does not guarantee improved out-of-sample results.
- Test predictions and run metadata store each selected direction threshold. The dashboard uses that threshold instead of blindly applying 0.50. Sigmoid scores remain uncalibrated scores, not real-world probabilities.
- Config stores package versions, dates, run ID, selected thresholds, and a SHA-256 data fingerprint.
- Dashboard detects each run separately and uses the NYSE calendar to label the next session, rather than adding one calendar day.
- Unit tests cover target alignment, no target-row leakage in feature windows, price reconstruction, and metrics.

## Upload & graph analysis

The sidebar includes **Upload & analyze data**. It accepts CSV and Excel (`.xlsx`) files up to 25 MB, with up to 250,000 rows and 200 columns per worksheet. Uploaded data is analyzed in memory for the active session and is not written into the project directories.

Supported and auto-detected formats include:

- **Historical prices:** Date + Adj Close / Close / Price; OHLCV unlocks candlesticks and volume. Charts include moving averages, normalized price performance, rolling Sharpe, returns, rolling volatility, drawdown, distributions and correlations.
- **Prediction exports:** Actual-vs-predicted lines and scatter, MAE/RMSE, residuals over time, rolling MAE, residual distribution, direction confusion matrix, score distribution, Brier score, ROC-AUC and a reliability diagram when upward-score columns are included.
- **Metrics / walk-forward CSVs:** Price error comparisons, directional metrics and fold-by-fold stability charts.
- **Other numeric data:** Missing-data inspection, time/row series, scatter relationships, histograms, box plots and correlation heatmaps.

The threshold control in the upload workspace is explicitly exploratory. Do not pick a threshold using a final held-out test set and then report those same scores as unbiased results.

## Deployment

Deployment files included in this repository:

- `Dockerfile` and `compose.yaml` for a non-root Docker deployment with a health check and read-only artifact mounts.
- `.streamlit/config.toml` for a 25 MB file upload limit and production-friendly server settings.
- `.github/workflows/ci.yml` for syntax compilation and unit tests on pushes/pull requests.
- `DEPLOYMENT.md` for Docker/VPS and Streamlit Community Cloud steps.
- `SECURITY.md` for handling secrets, uploads and trusted model artifacts.
- `requirements.txt` for Streamlit Community Cloud analysis-first deployment without TensorFlow.
- `requirements-models.txt` for the full dashboard runtime with TensorFlow inference, suitable for Docker/VPS deployments with approved model artifacts.
- `requirements-training.txt` for local model training plus tests.
- `requirements-test.txt` for CI tests without TensorFlow.

The source package does not include your locally trained model artifacts. The upload/analysis workspace can operate without model files; the forecasting workspace needs compatible model run files and result CSVs. See `DEPLOYMENT.md` before publishing, particularly for artifact management and access controls.

## 1. Requirements

- Windows 10/11, 64-bit Python 3.11 recommended
- VS Code
- Internet access to retrieve historical price data

Use the project's virtual environment in every terminal:

### Command Prompt (CMD)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

If `.venv` already exists and is active, you only need:

```bat
python -m pip install -r requirements-training.txt
```

Verify the important packages:

```bat
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import plotly, pandas_market_calendars; print('dashboard dependencies OK')"
```

If you're using PowerShell, activate via `\.venv\Scripts\Activate.ps1` instead of the CMD command.

## 2. Train all models

Recommended baseline run:

```bat
python train.py --ticker AAPL --period 5y --epochs 20
```

Optional symbols:

```bat
python train.py --ticker MSFT --period 5y --epochs 20
python train.py --ticker NVDA --period 5y --epochs 20
```

For a quicker smoke test, use fewer epochs. Walk-forward evaluation trains additional models, so it adds compute time:

```bat
python train.py --ticker AAPL --period 5y --epochs 5 --walk-forward-folds 0
```

Arguments:

- `--ticker`: symbol like `AAPL`, `MSFT`, `NVDA`
- `--period`: yfinance period such as `2y`, `5y`, `10y`, `max`
- `--epochs`: maximum epochs per fit; early stopping may finish sooner
- `--batch-size`: batch size (default 32)
- `--lookback`: historical sessions per input sequence (default 30)
- `--walk-forward-folds`: 0–5 expanding-window folds (default 2; 0 disables these extra fits)
- `--run-id`: optional unique identifier; existing run IDs are not overwritten

### What training does

1. Download auto-adjusted daily OHLCV market data.
2. Create trailing-only return/volatility features.
3. Create a sequence ending at session `t` to predict the return/direction for session `t+1` (the target day's feature row is not inside the input window).
4. Run expanding-window evaluation on development data, fitting fresh feature and target scalers inside every fold; select direction thresholds using only each fold's validation section.
5. Train final versions of CNN, RNN, LSTM and CNN–LSTM using a chronological 70% training / 15% validation / 15% held-out test split.
6. Select each final model's direction threshold on the 15% validation segment, then evaluate it once on the held-out test segment.
7. Save test predictions, selected thresholds, baselines, metrics, plots and run metadata.

## 3. Launch the dashboard

After training completes:

```bat
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Select a ticker, model run and architecture. Use the sidebar workspace selector to switch to the CSV/Excel upload analysis workspace.

The dashboard does not train models on page load. It loads saved model artifacts and retrieves recent market data for inference. To see results for another stock, train that ticker first.

## 4. Output structure

Each training invocation gets its own run directory:

```text
financial-trend-forecasting/
├── app.py
├── train.py
├── requirements.txt
├── src/
│   ├── data_utils.py
│   ├── evaluation.py
│   ├── forecasting.py
│   └── model_factory.py
├── tests/
├── data/
│   └── AAPL/<RUN_ID>/historical_prices.csv
├── models/
│   └── AAPL/<RUN_ID>/
│       ├── cnn.keras
│       ├── rnn.keras
│       ├── lstm.keras
│       ├── cnn_lstm.keras
│       ├── feature_scaler.joblib
│       ├── target_scaler.joblib
│       └── config.json
└── results/
    └── AAPL/<RUN_ID>/
        ├── metrics.csv
        ├── test_predictions.csv
        ├── test_predictions.png
        ├── walk_forward_metrics.csv
        └── walk_forward_summary.csv
```

The actual run ID is generated automatically. Each run is isolated, so a new MSFT/AAPL run will not overwrite another run's metrics or models.

## 5. Metrics and baselines

- **Price persistence baseline:** predicts that the next close equals the previous observed close.
- **Direction momentum baseline:** repeats the previous session's up/down direction.
- **Direction majority baseline:** always predicts the majority direction in the relevant training window.
- **CNN / RNN / LSTM / CNN–LSTM:** produce a log-return estimate and a separate sigmoid direction score. A model's direction threshold is selected on validation data and then fixed for its held-out test evaluation.

Regression metrics: price MAE and RMSE in USD. Direction metrics: accuracy, balanced accuracy, precision, recall and F1. The selected threshold is recorded as `direction_threshold` in metrics and test-prediction outputs. Compare the neural networks against relevant baselines; the hybrid is not guaranteed to perform best. Threshold tuning may improve the class trade-off but cannot create predictive signal where the model has none.

Walk-forward folds expand the available training history. Within each fold, the tail of the historical training window is reserved for early stopping and the next contiguous block is evaluated. Those folds use only the first 85% development segment. The final last 15% remains the held-out test set and is evaluated after model selection. Avoid repeatedly tuning on that held-out test period.

## 6. Run tests

```bat
python -m pytest -q
```

The CI workflow also starts the Streamlit entry point with empty artifact directories using Streamlit AppTest, so the dashboard shell is checked without needing the large TensorFlow package in CI.

## 7. Important limitations

- yfinance/Yahoo Finance data availability and terms may not suit a commercial service. Verify data-provider rights and reliability before production use.
- Sigmoid outputs are not automatically calibrated probabilities.
- Model results depend on ticker, time period, market regime and data revisions.
- This repository is not a trading engine. Before any real-world financial use, add point-in-time data controls, stronger walk-forward studies across symbols/regimes, uncertainty analysis, operational monitoring, security review, and a realistic backtest with transaction costs/slippage.
- Do not interpret forecast output as guaranteed direction, investment advice, or evidence of a profitable strategy.

## Troubleshooting

**`ModuleNotFoundError` for a package:** check that `(.venv)` is visible, then run `python -m pip install -r requirements-training.txt`.

**No trained run found in dashboard:** train a new run using this version first. Older artifacts from the starter version used flat filenames and predicted price directly; they are not compatible with the new log-return model and must be retrained.

**Not enough usable samples:** use `--period 5y` or another longer period. Short periods may be insufficient for walk-forward evaluation.

**Training takes too long:** use `--epochs 5 --walk-forward-folds 0` for a quick smoke test, then run the full experiment afterward.

**No data returned:** verify internet access and the ticker symbol. The dashboard and training CLI need working market-data access.

## Windows one-click helpers

To make local setup easier, the project includes:

- `install_dependencies.bat` — creates `.venv` with Python 3.11 if needed and installs `requirements-training.txt`.
- `run_tests.bat` — activates `.venv` and runs the test suite.
- `train_quick.bat` — quick AAPL smoke-training run with walk-forward folds disabled.
- `train_full.bat` — full example run with two walk-forward folds.
- `run_dashboard.bat` — starts Streamlit.

You can double-click these scripts or run their names from CMD in the project folder. The `.venv`, market data, and trained artifacts are intentionally not included in the ZIP.

## Python/NumPy compatibility

Dependency compatibility note: the dashboard requirements select NumPy 2.3.5 on Python 3.14 (needed by pandas 3.0.x) and NumPy 2.1.3 on Python versions below 3.14 (compatible with the TensorFlow 2.19 training setup). Keep Streamlit Cloud on the lightweight root requirements; use Python 3.11 for local training.
