from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "return_1d", "return_5d", "volatility_10d",
]
_REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_ticker(ticker: str) -> str:
    """Normalize a ticker and reject characters that could become filesystem paths."""
    normalized = ticker.strip().upper()
    if not normalized or not re.fullmatch(r"[A-Z0-9.^=\-]+", normalized):
        raise ValueError("Invalid ticker. Use a symbol such as AAPL, MSFT, or BRK-B.")
    return normalized


def download_prices(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    """Download daily auto-adjusted OHLCV and create trailing-only features."""
    ticker = validate_ticker(ticker)
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception as exc:
        raise RuntimeError(f"Market-data download failed for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise ValueError(
            f"No market data returned for {ticker}. Check the symbol, period, and internet connection."
        )

    # yfinance can return MultiIndex columns even for a single ticker. Select that
    # ticker's field columns instead of silently mixing ticker levels.
    if isinstance(df.columns, pd.MultiIndex):
        selected = None
        for level in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(level):
                candidate = df.xs(ticker, axis=1, level=level, drop_level=True)
                if all(col in candidate.columns for col in _REQUIRED_PRICE_COLUMNS):
                    selected = candidate
                    break
        if selected is not None:
            df = selected
        else:
            df.columns = df.columns.get_level_values(0)

    missing = [c for c in _REQUIRED_PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Market data for {ticker} is missing columns: {missing}")

    df = df[_REQUIRED_PRICE_COLUMNS].copy()
    for column in _REQUIRED_PRICE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.index = pd.to_datetime(df.index)
    # Daily stock data should be in chronological order and have one row per session.
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "Date"
    df = df.dropna(subset=_REQUIRED_PRICE_COLUMNS)
    df = df[df["Close"] > 0]

    # If yfinance includes today's still-forming daily candle, remove it. Training
    # and inference should use completed NYSE sessions only. Apply a short buffer
    # after the published close to avoid racing a delayed end-of-day data update.
    if not df.empty:
        now_utc = pd.Timestamp.now(tz="UTC")
        final_session_date = pd.Timestamp(df.index[-1]).date()
        if final_session_date == now_utc.date():
            schedule = mcal.get_calendar("NYSE").schedule(
                start_date=final_session_date, end_date=final_session_date
            )
            if not schedule.empty:
                market_close = pd.Timestamp(schedule["market_close"].iloc[0])
                if market_close.tzinfo is None:
                    market_close = market_close.tz_localize("UTC")
                if now_utc < market_close + pd.Timedelta(minutes=15):
                    df = df.iloc[:-1]

    df["return_1d"] = df["Close"].pct_change(fill_method=None)
    df["return_5d"] = df["Close"].pct_change(5, fill_method=None)
    df["volatility_10d"] = df["return_1d"].rolling(10, min_periods=10).std()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} usable rows were downloaded for {ticker}. Use a longer period, such as 2y or 5y."
        )
    return df


def save_prices(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, index_label="Date")
