# Deployment Guide

This repository includes a Docker deployment, a Compose service definition, Streamlit configuration, and CI tests. These files prepare the application for deployment; they do not publish it to a public server automatically.

## Choose a deployment mode

### A. Docker / a Linux VPS (recommended for the full forecasting app)

Requirements: Docker Engine or Docker Desktop with the Compose plugin.

1. Train the required model run on your development machine:

   ```bat
   python train.py --ticker AAPL --period 5y --epochs 20
   ```

2. Confirm the selected run has four `.keras` files, both scalers, and `config.json` under `models/<TICKER>/<RUN_ID>/`. Confirm the corresponding evaluation CSVs exist under `results/<TICKER>/<RUN_ID>/`.
3. Copy the project and the approved `models/` and `results/` run artifacts to the target host. Keep model/result folders outside the container image; Compose mounts them read-only.
4. Build and start the application from the project root:

   ```bash
   docker compose up --build -d
   docker compose ps
   docker compose logs -f financial-trend-forecasting
   ```

5. Open `http://localhost:8501` locally, or `http://<server-host>:8501` on a secured network. For a public server, put a TLS reverse proxy and authentication/access controls in front of the app; do not expose an unauthenticated upload application directly to the public internet.
6. Stop it with `docker compose down`.

The Docker image runs as a non-root user, includes a health check, has a 25 MB upload limit, and runs `pip check` during image build. Model artifacts are deliberately omitted from the image build context. The Compose file mounts local `models`, `results`, and `data` directories read-only. Keep a backup of trained artifacts and pin them to a documented run ID.

If the mounted `models/` folder contains no valid run, the upload-and-analysis workspace still works, but the forecast dashboard cannot produce a neural-network forecast until a compatible run is supplied.

### B. Streamlit Community Cloud

Streamlit Community Cloud builds from a GitHub repository, requires a dependency file in the repository or beside the app, and supports selecting the Python version in app settings. See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), and [file organization guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

1. Push the project source to a GitHub repository. Do not commit `.venv/`, secrets, or the whole local training dataset.
2. Keep `.streamlit/config.toml`, `app.py`, `src/`, and the appropriate `requirements.txt` at the repository root.
3. Open `share.streamlit.io`, create an app, and select the repository, branch, and `app.py` entry point. Select Python 3.11 in Advanced settings to match the local setup, then deploy.
4. If the deployed repo does not include trained model artifacts, use **Upload & analyze data** for CSV/Excel uploads, predictions, metrics and generic tabular visualizations. To enable forecast inference, the deployed runtime must also have an approved compatible model run: `models/<TICKER>/<RUN_ID>/` plus the corresponding `results/<TICKER>/<RUN_ID>/` files. The app cannot access files on your laptop.
5. The normal `requirements.txt` includes TensorFlow for forecasting inference/training compatibility and can be a larger cloud build. If deploying an **analysis-only** variant without trained models, use `requirements-dashboard.txt` as the deployment branch's `requirements.txt`; it omits TensorFlow and matplotlib. Do not use this lightweight profile if you intend to load `.keras` models.

The application currently uses local directories for artifact discovery. Streamlit Community Cloud files are part of the deployed repository/runtime, not automatically synchronized with your Windows computer. For larger/private artifacts, use persistent or object storage and implement controlled artifact retrieval; do not place credentials in source code.

### C. Local Docker smoke test

From CMD or PowerShell in the project root:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f financial-trend-forecasting
```

Check the container health status. To stop the app:

```bash
docker compose down
```

## Configuration

- `.streamlit/config.toml`: 25 MB uploader maximum, XSRF protection and app theme.
- `FTF_MODELS_DIR`: optional absolute path to model run folders (default: `<project>/models`).
- `FTF_RESULTS_DIR`: optional absolute path to result folders (default: `<project>/results`).
- `FTF_DATA_DIR`: optional absolute path to training-data folders (default: `<project>/data`).
- No API keys are required by this project currently. If future providers require credentials, store them in the hosting platform's secret manager, never in Git.

## Release checklist

- [ ] `python -m pytest -q` passes.
- [ ] `python -m compileall -q app.py train.py src tests` passes.
- [ ] A new unique training run completes; the run ID, metrics, thresholds and data hash are saved.
- [ ] Walk-forward and final-test metrics are present and baselines are included.
- [ ] The forecast dashboard has a compatible model run, or the deployment is explicitly analysis-only.
- [ ] CSV and XLSX upload works; large, empty, malformed, or unsupported files produce helpful errors.
- [ ] The selected model and matching run ID are visible; no model files are overwritten by inference.
- [ ] Health checks pass and logs are reviewed.
- [ ] Public deployments have appropriate authentication, network controls, and a data-provider license review.
- [ ] Any live or investment use has separate legal, security, data-quality, model-risk, backtesting and monitoring review. This demo is not a trading service.

## Important limitations

- A green Docker health check means the Streamlit process responds; it does not certify model quality or profitability.
- The default market data pipeline uses `yfinance`. Confirm data provider terms before commercial use.
- A sigmoid output is not automatically a calibrated probability.
- Uploaded files are analyzed in memory for the current session and are not persisted by this application.
- A production finance system would require robust point-in-time data, authentication/authorization, rate limiting, audit logging, artifact signing/version controls, drift monitoring, backup and disaster-recovery procedures, and rigorous independent model validation.
