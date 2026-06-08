"""
market_cycle/indicators.py
==========================
Download OHLCV via yfinance and compute all Market Cycle indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _compute_from_df(df: pd.DataFrame) -> dict:
    """Compute all indicators from a pre-downloaded OHLCV DataFrame."""
    if df is None or df.empty:
        return {}

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Require at least Close
    if "Close" not in df.columns:
        print(f"No Close column. Available: {df.columns.tolist()}")
        return {}

    close = df["Close"].dropna()
    n = len(close)

    if n < 20:
        return {}

    price = float(close.iloc[-1])

    # Moving averages
    ma50 = float(close.rolling(50).mean().iloc[-1]) if n >= 50 else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None

    # RSI
    rsi_series = _rsi(close)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.dropna().empty else None
    if rsi is not None and np.isnan(rsi):
        rsi = None

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd = float(macd_line.iloc[-1])
    macd_signal = float(macd_signal_line.iloc[-1])
    macd_diff = macd - macd_signal

    # ATR
    atr = None
    if "High" in df.columns and "Low" in df.columns:
        high = df["High"].reindex(close.index)
        low = df["Low"].reindex(close.index)
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        _atr = tr.rolling(14).mean().iloc[-1]
        atr = float(_atr) if not np.isnan(_atr) else None

    # Volume
    volume_avg20 = None
    volume_vs_avg20 = 1.0
    if "Volume" in df.columns:
        vol = df["Volume"].reindex(close.index).replace(0, np.nan)
        if not vol.dropna().empty:
            _avg20 = float(vol.rolling(20).mean().iloc[-1])
            _last_vol = float(vol.iloc[-1])
            if not np.isnan(_avg20) and _avg20 > 0 and not np.isnan(_last_vol):
                volume_avg20 = _avg20
                volume_vs_avg20 = _last_vol / _avg20

    # Price vs MAs
    price_vs_ma50 = ((price / ma50) - 1) * 100 if ma50 and ma50 > 0 else None
    price_vs_ma200 = ((price / ma200) - 1) * 100 if ma200 and ma200 > 0 else None

    # Momentum (percentage change over N periods)
    def _mom(n_periods: int):
        if n < n_periods + 1:
            return None
        ref = float(close.iloc[-n_periods])
        if ref <= 0:
            return None
        return float((price / ref - 1) * 100)

    momentum_30 = _mom(30)
    momentum_60 = _mom(60)
    momentum_90 = _mom(90)

    # Drawdown from 90-day (or available) high
    lookback = min(90, n)
    high_n = float(close.iloc[-lookback:].max())
    drawdown_90 = float((price / high_n - 1) * 100) if high_n > 0 else 0.0

    return {
        "price": price,
        "ma50": ma50,
        "ma200": ma200,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_diff": macd_diff,
        "atr": atr,
        "volume_avg20": volume_avg20,
        "price_vs_ma50": price_vs_ma50,
        "price_vs_ma200": price_vs_ma200,
        "volume_vs_avg20": volume_vs_avg20,
        "momentum_30": momentum_30,
        "momentum_60": momentum_60,
        "momentum_90": momentum_90,
        "drawdown_90": drawdown_90,
    }


def compute_indicators(ticker: str, period: str = "1y") -> dict:
    """Download OHLCV for ticker/period and return computed indicator dict."""
    import traceback
    import yfinance as _yf_mod
    print(f"yfinance version: {_yf_mod.__version__}")

    if not ticker or not ticker.strip():
        return {}

    try:
        ticker_clean = ticker.strip().upper()

        try:
            df = yf.download(
                ticker_clean,
                period=period,
                auto_adjust=True,
                progress=False,
                show_errors=False,
                multi_level_index=False,
            )
            if df is None or df.empty:
                print(f"Empty df for {ticker_clean} (multi_level_index=False), period={period} — retrying without kwarg")
                df = yf.download(
                    ticker_clean,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                )
        except TypeError:
            print(f"multi_level_index kwarg not supported — falling back for {ticker_clean}")
            df = yf.download(
                ticker_clean,
                period=period,
                auto_adjust=True,
                progress=False,
            )

        print(f"Downloaded {ticker_clean}: shape={df.shape}, columns={df.columns.tolist()}")

        if df is None or df.empty:
            print(f"Empty df for {ticker_clean}, period={period}")
            return {}

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            print(f"After MultiIndex flatten: columns={df.columns.tolist()}")

        # Some yfinance versions return ticker as second level
        if "Close" not in df.columns:
            print(f"No Close column. Available: {df.columns.tolist()}")
            close_cols = [c for c in df.columns if str(c).lower() == "close"]
            if close_cols:
                df = df.rename(columns={close_cols[0]: "Close"})
                print(f"Renamed '{close_cols[0]}' -> 'Close'")
            else:
                print(f"Cannot find Close column for {ticker_clean} — returning empty")
                return {}

        return _compute_from_df(df)

    except Exception as exc:
        print(f"compute_indicators ERROR for {ticker}: {exc}")
        print(traceback.format_exc())
        return {}


def download_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download raw OHLCV without processing (used for history rolling)."""
    if not ticker or not ticker.strip():
        return pd.DataFrame()
    try:
        df = yf.download(
            ticker.strip().upper(),
            period=period,
            auto_adjust=True,
            progress=False,
            show_errors=False,
            multi_level_index=False,
        )
    except TypeError:
        try:
            df = yf.download(
                ticker.strip().upper(),
                period=period,
                auto_adjust=True,
                progress=False,
            )
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df
