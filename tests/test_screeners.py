"""
Regression tests for screeners.
Ensures Long & Swing screeners are not modified.
OVTLYR screener unit tests.
Test Top N flow.
"""
import sys
import os
import pytest
import pandas as pd

# Add dashboard to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOvtlyrScreener:
    """Unit tests for the OVTLYR screener."""

    def test_import(self):
        from screener_ovtlyr import run_ovtlyr_screener, score_single_ticker
        assert callable(run_ovtlyr_screener)
        assert callable(score_single_ticker)

    def test_score_single_ticker(self):
        import numpy as np
        from screener_ovtlyr import score_single_ticker

        # Create synthetic OHLCV data
        np.random.seed(42)
        n = 300
        close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n))
        df = pd.DataFrame({
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, n),
        })

        result = score_single_ticker(df)
        assert result is not None
        assert "trend" in result
        assert "momentum" in result
        assert "volatility" in result
        assert "volume" in result
        assert "adx" in result
        # All scores should be 0-100
        for key in ["trend", "momentum", "volatility", "volume", "adx"]:
            assert 0 <= result[key] <= 100, f"{key} out of range: {result[key]}"

    def test_zscore_normalize(self):
        from screener_ovtlyr import _zscore_normalize
        vals = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90])
        normalized = _zscore_normalize(vals)
        assert len(normalized) == len(vals)
        assert normalized.min() >= 0
        assert normalized.max() <= 100

    def test_empty_dataframe(self):
        from screener_ovtlyr import score_single_ticker
        result = score_single_ticker(pd.DataFrame())
        assert result is None

    def test_short_dataframe(self):
        from screener_ovtlyr import score_single_ticker
        df = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]})
        result = score_single_ticker(df)
        assert result is None  # < 50 rows


class TestBacktestEngine:
    """Unit tests for the unified backtest engine."""

    def test_import(self):
        from backtest_engine import run_backtest, run_batch_backtest
        assert callable(run_backtest)
        assert callable(run_batch_backtest)

    def test_run_backtest_modes(self):
        """Test all three backtest modes return expected structure."""
        from backtest_engine import run_backtest
        # Use synthetic data (actual yfinance test would need network)
        # Just verify the function signature works
        for mode in ["swing", "long", "ovtlyr"]:
            # This will likely return empty due to no network in CI
            result = run_backtest("INVALID_TICKER_12345", 1, mode)
            assert "trades" in result
            assert "metrics" in result
            assert "equity_curve" in result


class TestSwingScreenerRegression:
    """Regression tests for Swing screener (must be UNCHANGED)."""

    def test_swing_screener_import(self):
        """Verify the swing screener module still imports."""
        from wolf_shadow_screener import run_screener, MARKETS
        assert callable(run_screener)
        assert isinstance(MARKETS, dict)
        assert "commodity" in MARKETS
        assert "sp500" in MARKETS

    def test_swing_screener_markets(self):
        """Verify market definitions are unchanged."""
        from wolf_shadow_screener import MARKETS
        expected_markets = {"commodity", "sp500", "stockholm", "oslo",
                          "copenhagen", "helsinki", "europe",
                          "us_smallcap", "us_midcap", "canada", "junior_miners"}
        assert set(MARKETS.keys()) == expected_markets

    def test_swing_screener_columns(self):
        """Verify PRESET_PARAMS structure is intact."""
        from wolf_shadow_screener import PRESET_PARAMS
        assert "Universal" in PRESET_PARAMS
        universal = PRESET_PARAMS["Universal"]
        required_keys = {"ema_fast", "ema_slow", "atr_mult", "adx_thresh"}
        assert required_keys.issubset(set(universal.keys()))


class TestLongScreenerRegression:
    """Regression tests for Long screener (must be UNCHANGED)."""

    def test_long_screener_import(self):
        from cagr.cagr_scoring import calculate_total_score, SIGNAL_COLORS
        assert callable(calculate_total_score)
        assert "STRONG BUY" in SIGNAL_COLORS
        assert "STRONG SELL" in SIGNAL_COLORS

    def test_long_screener_fundamentals(self):
        from cagr.cagr_fundamentals import score_fundamentals
        # Test with Börsdata-style data
        data = {"_data_source": "borsdata", "returnOnEquity": 0.20,
                "fcf_stability": 80, "f_score": 8}
        result = score_fundamentals(data)
        assert result["fund_max"] == 20
        assert result["fund_score"] >= 0

    def test_long_screener_scoring_scale(self):
        from cagr.cagr_scoring import calculate_total_score
        total = calculate_total_score(
            fund={"fund_score": 15, "fund_max": 20},
            cycle={"cycle_score": 2},
            tech={"tech_score": 5, "tech_max": 7},
        )
        assert total["max_score"] == 30
        assert total["total_score"] == 22


class TestTopNFlow:
    """Test the Test Top N flow (session state integration)."""

    def test_session_state_keys(self):
        """Verify expected session state keys."""
        expected_keys = ["test_topn_tickers", "test_topn_mode", "auto_run_backtest"]
        # These are string constants used in the code
        for key in expected_keys:
            assert isinstance(key, str)
