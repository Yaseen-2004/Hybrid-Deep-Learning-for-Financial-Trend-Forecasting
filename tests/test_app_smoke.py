from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


def test_dashboard_starts_without_trained_artifacts(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    models = tmp_path / "models"
    results = tmp_path / "results"
    models.mkdir()
    results.mkdir()
    monkeypatch.setenv("FTF_MODELS_DIR", str(models))
    monkeypatch.setenv("FTF_RESULTS_DIR", str(results))

    app = AppTest.from_file(str(project_root / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert any("Hybrid Deep Learning for Financial Trend Forecasting" in item.value for item in app.title)
