#!/usr/bin/env python3
"""
WOLF x SHADOW BACKTESTER v2.2
==============================
Full backtesting engine mirroring the Pine Script strategy.
Core/Trim position management, 4-layer regime scoring,
partial exits, walk-forward analysis.

Usage:
    python wolf_shadow_backtest.py                          # Default: SPY 3 years
    python wolf_shadow_backtest.py --ticker XOM             # Single stock
    python wolf_shadow_backtest.py --ticker XOM DVN COP     # Multiple stocks
    python wolf_shadow_backtest.py --ticker EQNR.OL         # Oslo stock
    python wolf_shadow_backtest.py --ticker HEXA-B.ST       # Stockholm stock
    python wolf_shadow_backtest.py --years 5                # 5 years of data
    python wolf_shadow_backtest.py --walk-forward           # Walk-forward analysis
"""

import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION (mirrors Pine Script inputs)
# =============================================================================
CONFIG = {
    # EMA
    "ema_pulse": 10,
    "ema_fast": 20,
    "ema_slow": 50,
    "ema_macro": 200,
    # Ichimoku
    "ichi_conv": 9,
    "ichi_base": 26,
    "ichi_span_b": 52,
    "ichi_displacement": 26,
    # RSI
    "rsi_len": 14,
    "rsi_hot": 70,
    # Extension
    "ext_pct": 2.7,
    # Order blocks
    "ob_lookback": 5,
    # Exit
    "atr_len": 14,
    "atr_mult": 2.0,
    "tp1_rr": 2.0,
    "tp1_pct": 0.30,
    "tp2_rr": 3.0,
    "tp2_pct": 0.30,
    # Position
    "core_pct": 0.50,
    "risk_pct": 0.015,
    # Circuit breakers
    "daily_breaker": -0.05,
    "weekly_breaker": -0.10,
    # Regime thresholds
    "entry_min_score": 40,
    "add_min_score": 50,
    # Cooldown
    "cooldown_bars": 4,
    # ADX filter
    "adx_threshold": 19,    # default Universal preset
    # Commission & slippage
    "commission": 0.0005,
    "slippage": 0.001,
}


# =============================================================================
# PRESET PARAMETERS (v3 — 21 presets: US Sector ETFs, Nordic, Individual, Universal)
# =============================================================================
PRESET_PARAMS = {
    # US Sector ETFs
    "XLE": {"ema_pulse":9,"ema_fast":38,"ema_slow":80,"tenkan":7,"kijun":35,"spanb":42,"atr_mult":2.1,"adx_thresh":5,"tp1_rr":1.75,"tp1_pct":0.10,"tp2_rr":5.5,"tp2_pct":0.10,"core_pct":0.40,"min_regime":63},
    "XLB": {"ema_pulse":11,"ema_fast":12,"ema_slow":31,"tenkan":15,"kijun":29,"spanb":57,"atr_mult":1.5,"adx_thresh":21,"tp1_rr":2.25,"tp1_pct":0.10,"tp2_rr":6.0,"tp2_pct":0.15,"core_pct":0.55,"min_regime":41},
    "XLF": {"ema_pulse":18,"ema_fast":36,"ema_slow":91,"tenkan":5,"kijun":34,"spanb":61,"atr_mult":3.2,"adx_thresh":28,"tp1_rr":2.5,"tp1_pct":0.25,"tp2_rr":3.5,"tp2_pct":0.15,"core_pct":0.55,"min_regime":51},
    "XLK": {"ema_pulse":7,"ema_fast":39,"ema_slow":108,"tenkan":6,"kijun":29,"spanb":73,"atr_mult":2.3,"adx_thresh":3,"tp1_rr":2.5,"tp1_pct":0.20,"tp2_rr":5.25,"tp2_pct":0.05,"core_pct":0.70,"min_regime":36},
    "XLV": {"ema_pulse":15,"ema_fast":34,"ema_slow":105,"tenkan":10,"kijun":29,"spanb":64,"atr_mult":2.0,"adx_thresh":2,"tp1_rr":4.0,"tp1_pct":0.05,"tp2_rr":4.5,"tp2_pct":0.20,"core_pct":0.40,"min_regime":44},
    "XLI": {"ema_pulse":14,"ema_fast":18,"ema_slow":103,"tenkan":5,"kijun":34,"spanb":65,"atr_mult":1.8,"adx_thresh":6,"tp1_rr":3.75,"tp1_pct":0.10,"tp2_rr":4.0,"tp2_pct":0.10,"core_pct":0.60,"min_regime":61},
    "XLY": {"ema_pulse":10,"ema_fast":18,"ema_slow":75,"tenkan":10,"kijun":29,"spanb":38,"atr_mult":2.0,"adx_thresh":2,"tp1_rr":3.0,"tp1_pct":0.20,"tp2_rr":3.75,"tp2_pct":0.25,"core_pct":0.55,"min_regime":60},
    "XLP": {"ema_pulse":18,"ema_fast":19,"ema_slow":56,"tenkan":7,"kijun":32,"spanb":47,"atr_mult":2.3,"adx_thresh":0,"tp1_rr":2.25,"tp1_pct":0.05,"tp2_rr":5.75,"tp2_pct":0.25,"core_pct":0.50,"min_regime":64},
    "XLRE": {"ema_pulse":13,"ema_fast":25,"ema_slow":33,"tenkan":5,"kijun":29,"spanb":57,"atr_mult":2.5,"adx_thresh":1,"tp1_rr":3.5,"tp1_pct":0.05,"tp2_rr":5.25,"tp2_pct":0.20,"core_pct":0.40,"min_regime":64},
    "XLU": {"ema_pulse":5,"ema_fast":45,"ema_slow":117,"tenkan":12,"kijun":29,"spanb":61,"atr_mult":1.1,"adx_thresh":13,"tp1_rr":3.5,"tp1_pct":0.15,"tp2_rr":4.5,"tp2_pct":0.20,"core_pct":0.45,"min_regime":31},
    "XLC": {"ema_pulse":17,"ema_fast":44,"ema_slow":100,"tenkan":7,"kijun":30,"spanb":45,"atr_mult":3.1,"adx_thresh":14,"tp1_rr":3.25,"tp1_pct":0.25,"tp2_rr":3.5,"tp2_pct":0.05,"core_pct":0.60,"min_regime":28},
    # Nordic Exchanges
    "OMX Stockholm": {"ema_pulse":13,"ema_fast":22,"ema_slow":55,"tenkan":8,"kijun":29,"spanb":56,"atr_mult":2.0,"adx_thresh":18,"tp1_rr":3.15,"tp1_pct":0.20,"tp2_rr":4.8,"tp2_pct":0.22,"core_pct":0.46,"min_regime":61},
    "OMX Copenhagen": {"ema_pulse":10,"ema_fast":27,"ema_slow":58,"tenkan":10,"kijun":26,"spanb":72,"atr_mult":2.1,"adx_thresh":12,"tp1_rr":2.2,"tp1_pct":0.12,"tp2_rr":5.15,"tp2_pct":0.12,"core_pct":0.48,"min_regime":47},
    "Oslo OSEBX": {"ema_pulse":13,"ema_fast":24,"ema_slow":68,"tenkan":9,"kijun":30,"spanb":66,"atr_mult":2.3,"adx_thresh":20,"tp1_rr":3.55,"tp1_pct":0.16,"tp2_rr":4.8,"tp2_pct":0.19,"core_pct":0.49,"min_regime":51},
    "OMX Helsinki": {"ema_pulse":13,"ema_fast":26,"ema_slow":84,"tenkan":10,"kijun":33,"spanb":61,"atr_mult":2.1,"adx_thresh":11,"tp1_rr":3.05,"tp1_pct":0.21,"tp2_rr":5.55,"tp2_pct":0.15,"core_pct":0.54,"min_regime":48},
    # Individual stocks
    "OXY": {"ema_pulse":6,"ema_fast":23,"ema_slow":40,"tenkan":7,"kijun":24,"spanb":65,"atr_mult":2.8,"adx_thresh":27,"tp1_rr":3.0,"tp1_pct":0.20,"tp2_rr":5.5,"tp2_pct":0.25,"core_pct":0.70,"min_regime":53},
    "GOLD": {"ema_pulse":5,"ema_fast":18,"ema_slow":71,"tenkan":15,"kijun":36,"spanb":67,"atr_mult":1.5,"adx_thresh":16,"tp1_rr":1.75,"tp1_pct":0.20,"tp2_rr":5.75,"tp2_pct":0.05,"core_pct":0.50,"min_regime":49},
    "NEM": {"ema_pulse":11,"ema_fast":13,"ema_slow":99,"tenkan":9,"kijun":33,"spanb":41,"atr_mult":3.1,"adx_thresh":17,"tp1_rr":3.0,"tp1_pct":0.05,"tp2_rr":4.25,"tp2_pct":0.25,"core_pct":0.60,"min_regime":44},
    "XOM": {"ema_pulse":13,"ema_fast":23,"ema_slow":39,"tenkan":8,"kijun":39,"spanb":48,"atr_mult":2.7,"adx_thresh":27,"tp1_rr":3.5,"tp1_pct":0.10,"tp2_rr":5.5,"tp2_pct":0.10,"core_pct":0.70,"min_regime":60},
    "GLD": {"ema_pulse":7,"ema_fast":28,"ema_slow":37,"tenkan":13,"kijun":22,"spanb":59,"atr_mult":2.6,"adx_thresh":7,"tp1_rr":1.75,"tp1_pct":0.10,"tp2_rr":5.0,"tp2_pct":0.20,"core_pct":0.60,"min_regime":50},
    # Universal fallback
    "Universal": {"ema_pulse":8,"ema_fast":21,"ema_slow":57,"tenkan":10,"kijun":31,"spanb":56,"atr_mult":2.5,"adx_thresh":19,"tp1_rr":2.6,"tp1_pct":0.13,"tp2_rr":5.2,"tp2_pct":0.17,"core_pct":0.62,"min_regime":51},
}


def get_preset_for_ticker(ticker):
    """Auto-detect which preset to use based on ticker symbol."""
    # Direct match first
    if ticker in PRESET_PARAMS:
        return PRESET_PARAMS[ticker]
    # Nordic exchange suffix detection
    if ticker.endswith(".ST"):
        return PRESET_PARAMS["OMX Stockholm"]
    if ticker.endswith(".OL"):
        return PRESET_PARAMS["Oslo OSEBX"]
    if ticker.endswith(".CO"):
        return PRESET_PARAMS["OMX Copenhagen"]
    if ticker.endswith(".HE"):
        return PRESET_PARAMS["OMX Helsinki"]
    return PRESET_PARAMS["Universal"]


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================
def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.ewm(com=p-1, min_periods=p).mean()
    al = l.ewm(com=p-1, min_periods=p).mean()
    return 100 - (100 / (1 + ag / al))

def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.ewm(span=period, adjust=False).mean()

def donchian(h, l, p):
    return (h.rolling(p).max() + l.rolling(p).min()) / 2

def add_indicators(df, cfg=CONFIG):
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    # EMAs
    d["ema10"] = calc_ema(c, cfg["ema_pulse"])
    d["ema20"] = calc_ema(c, cfg["ema_fast"])
    d["ema50"] = calc_ema(c, cfg["ema_slow"])
    d["ema200"] = calc_ema(c, cfg["ema_macro"])

    # RSI, ATR & ADX
    d["rsi"] = calc_rsi(c, cfg["rsi_len"])
    d["atr"] = calc_atr(h, l, c, cfg["atr_len"])
    d["adx"] = calc_adx(h, l, c, 14)
    d["vol_ma"] = v.rolling(20).mean()

    # Ichimoku
    d["tenkan"] = donchian(h, l, cfg["ichi_conv"])
    d["kijun"] = donchian(h, l, cfg["ichi_base"])
    d["senkou_a"] = ((d["tenkan"] + d["kijun"]) / 2).shift(cfg["ichi_displacement"])
    d["senkou_b"] = donchian(h, l, cfg["ichi_span_b"]).shift(cfg["ichi_displacement"])

    # EMA conditions
    d["ema_trend"] = (d["ema10"] > d["ema20"]) & (d["ema20"] > d["ema50"]) & (c > d["ema50"])
    d["ema_stack"] = d["ema_trend"] & (d["ema50"] > d["ema200"])

    # EMA events
    d["ema_cross_up"] = (d["ema10"] > d["ema20"]) & (d["ema10"].shift(1) <= d["ema20"].shift(1)) & (c > d["ema50"])
    d["ema_reclaim"] = (c > d["ema10"]) & (c.shift(1) <= d["ema10"].shift(1))

    # Momentum
    d["mom_up"] = (d["rsi"] > d["rsi"].shift(1)) & (d["rsi"].shift(1) > d["rsi"].shift(2))
    d["mom_down"] = (d["rsi"] < d["rsi"].shift(1)) & (d["rsi"].shift(1) < d["rsi"].shift(2))

    # Extension
    ext = cfg["ext_pct"] / 100.0
    d["overextended"] = (c > d["ema10"] * (1 + ext)) | (d["rsi"] > cfg["rsi_hot"])
    d["weakness"] = (c < d["ema10"]) & (c.shift(1) < d["ema10"].shift(1))

    # Order blocks
    ob = cfg["ob_lookback"]
    d["bull_ob"] = (l <= l.rolling(ob).min().shift(1)) & (c > d["Open"])
    d["bear_ob"] = (h >= h.rolling(ob).max().shift(1)) & (c < d["Open"])

    # Ichimoku scores
    kumo_top = pd.concat([d["senkou_a"], d["senkou_b"]], axis=1).max(axis=1)
    d["kumo_top"] = kumo_top
    d["ichi_above"] = c > kumo_top
    d["ichi_tk_bull"] = d["tenkan"] > d["kijun"]
    d["ichi_chikou"] = c > c.shift(cfg["ichi_displacement"])
    d["ichi_twist"] = d["senkou_a"] > d["senkou_b"]

    d["ichi_score"] = (d["ichi_above"].astype(int) * 5 +
                       d["ichi_tk_bull"].astype(int) * 5 +
                       d["ichi_chikou"].astype(int) * 3 +
                       d["ichi_twist"].astype(int) * 2)

    # Stock score
    d["stock_score"] = ((d["ema10"] > d["ema20"]).astype(int) * 8 +
                        (d["ema20"] > d["ema50"]).astype(int) * 8 +
                        (c > d["ema50"]).astype(int) * 8 +
                        (c > d["ema200"]).astype(int) * 8 +
                        (d["rsi"] > 50).astype(int) * 8 +
                        d["mom_up"].astype(int) * 10)

    return d


# =============================================================================
# REGIME SCORING (SPY + SECTOR)
# =============================================================================
def fetch_regime_data(spy_ticker="SPY", sector_ticker="XLE", years=3):
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 100)

    spy_df = yf.download(spy_ticker, start=start, end=end, progress=False)
    sec_df = yf.download(sector_ticker, start=start, end=end, progress=False)

    # Flatten MultiIndex columns if needed
    for d in [spy_df, sec_df]:
        if d is not None and isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    return spy_df, sec_df

def calc_regime_scores(spy_df, sec_df, stock_df):
    """Add market_score, sector_score, and total regime_score to stock_df."""
    d = stock_df.copy()

    # Market score (aligned to stock dates)
    if spy_df is not None and len(spy_df) > 200:
        spy = spy_df.reindex(d.index, method="ffill")
        spy_c = spy["Close"]
        spy_e50 = calc_ema(spy_c, 50)
        spy_e200 = calc_ema(spy_c, 200)
        spy_rsi = calc_rsi(spy_c, 14)
        spy_atr = calc_atr(spy["High"], spy["Low"], spy_c, 14)
        spy_atr_pct = spy_atr / spy_c * 100

        d["market_score"] = ((spy_c > spy_e50).astype(int) * 10 +
                             (spy_c > spy_e200).astype(int) * 10 +
                             (spy_rsi > 50).astype(int) * 5 +
                             ((spy_atr_pct > 0.3) & (spy_atr_pct < 4.0)).astype(int) * 5)
    else:
        d["market_score"] = 15  # neutral default

    # Sector score
    if sec_df is not None and len(sec_df) > 200:
        sec = sec_df.reindex(d.index, method="ffill")
        sec_c = sec["Close"]
        sec_e50 = calc_ema(sec_c, 50)
        sec_e200 = calc_ema(sec_c, 200)
        sec_rsi = calc_rsi(sec_c, 14)

        d["sector_score"] = ((sec_c > sec_e50).astype(int) * 10 +
                             (sec_c > sec_e200).astype(int) * 10 +
                             (sec_rsi > 50).astype(int) * 10)
    else:
        d["sector_score"] = 15

    d["regime_score"] = d["market_score"] + d["sector_score"] + d["stock_score"] + d["ichi_score"]
    return d


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================
class Trade:
    def __init__(self, entry_price, qty, entry_bar, trade_type="CORE"):
        self.entry_price = entry_price
        self.qty = qty
        self.remaining_qty = qty
        self.entry_bar = entry_bar
        self.trade_type = trade_type
        self.tp1_hit = False
        self.tp2_hit = False
        self.exit_price = None
        self.exit_bar = None
        self.exit_reason = ""
        self.pnl = 0.0

class Backtester:
    def __init__(self, df, cfg=CONFIG, initial_capital=100000):
        self.df = df
        self.cfg = cfg
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = []  # list of Trade objects
        self.closed_trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.cooldown_until = -1
        self.daily_start_equity = initial_capital
        self.weekly_start_equity = initial_capital

    def position_value(self, price):
        return sum(t.remaining_qty * price for t in self.position)

    def total_equity(self, price):
        return self.capital + self.position_value(price)

    def apply_cost(self, value):
        return value * (self.cfg["commission"] + self.cfg["slippage"])

    def run(self):
        df = self.df
        cfg = self.cfg
        prev_equity = self.initial_capital

        for i in range(200, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            price = row["Close"]
            bar = i

            equity = self.total_equity(price)
            self.equity_curve.append({"date": df.index[i], "equity": equity})

            # Daily return
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_ret)
            prev_equity = equity

            # Circuit breakers (simplified: check drawdown from peak)
            if equity < self.daily_start_equity * (1 + cfg["daily_breaker"]):
                self._close_all(price, bar, "DAILY BREAKER")
                self.cooldown_until = bar + cfg["cooldown_bars"]
                continue

            # Reset daily equity at day boundary (approximate)
            if i > 0 and df.index[i].date() != df.index[i-1].date():
                self.daily_start_equity = equity
            if i > 0 and df.index[i].isocalendar()[1] != df.index[i-1].isocalendar()[1]:
                self.weekly_start_equity = equity

            # --- MANAGE OPEN POSITIONS ---
            for trade in self.position[:]:
                bars_held = bar - trade.entry_bar
                sl_dist = row["atr"] * cfg["atr_mult"]

                if trade.trade_type == "TRIM":
                    # TP1: R:R 2:1
                    tp1_price = trade.entry_price + sl_dist * cfg["tp1_rr"]
                    if not trade.tp1_hit and price >= tp1_price:
                        close_qty = trade.remaining_qty * cfg["tp1_pct"]
                        self._partial_close(trade, close_qty, price, bar, "TP1 2R")
                        trade.tp1_hit = True

                    # TP2: R:R 3:1
                    tp2_price = trade.entry_price + sl_dist * cfg["tp2_rr"]
                    if trade.tp1_hit and not trade.tp2_hit and price >= tp2_price:
                        close_qty = trade.remaining_qty * (cfg["tp2_pct"] / (1 - cfg["tp1_pct"]))
                        close_qty = min(close_qty, trade.remaining_qty)
                        self._partial_close(trade, close_qty, price, bar, "TP2 3R")
                        trade.tp2_hit = True

                    # Cycle exit: overextended + momentum down
                    if (row["overextended"] and row["mom_down"] and
                        (row["bear_ob"] or row["weakness"]) and
                        row["regime_score"] < 85):
                        self._close_trade(trade, price, bar, "CYCLE EXIT")
                        continue

                # SL for CORE (wider)
                if trade.trade_type == "CORE":
                    core_sl = trade.entry_price - sl_dist * 1.5
                    if price <= core_sl:
                        self._close_trade(trade, price, bar, "CORE SL")
                        continue

                # SL for TRIM (breakeven after TP1)
                if trade.trade_type == "TRIM":
                    sl_price = trade.entry_price if trade.tp1_hit else trade.entry_price - sl_dist
                    if price <= sl_price:
                        self._close_trade(trade, price, bar, "TRIM SL")
                        continue

                # Confirmed trend exit (2 bars of broken trend)
                if bars_held > 2 and not row["ema_trend"] and not prev["ema_trend"]:
                    self._close_trade(trade, price, bar, "TREND EXIT")
                    continue

            # Clean up fully closed trades
            self.position = [t for t in self.position if t.remaining_qty > 0.01]

            # --- ENTRY ---
            if bar <= self.cooldown_until:
                continue

            has_position = len(self.position) > 0
            regime = row["regime_score"]

            # Entry conditions
            entry_signal = (row["ema_trend"] and
                           regime >= cfg["entry_min_score"] and
                           row["adx"] >= cfg["adx_threshold"] and
                           (row["bull_ob"] or row["ema_cross_up"] or
                            row["ema_reclaim"] or row["mom_up"]))

            if not has_position and entry_signal:
                # Size based on regime
                core_mult = 1.0 if regime >= 70 else 0.7 if regime >= 50 else 0.4
                trim_mult = core_mult

                core_value = equity * cfg["core_pct"] * core_mult
                trim_value = equity * (1 - cfg["core_pct"]) * trim_mult

                cost_c = self.apply_cost(core_value)
                cost_t = self.apply_cost(trim_value)

                core_qty = (core_value - cost_c) / price
                trim_qty = (trim_value - cost_t) / price

                self.capital -= (core_value + trim_value)
                self.position.append(Trade(price, core_qty, bar, "CORE"))
                self.position.append(Trade(price, trim_qty, bar, "TRIM"))

            # Add to TRIM
            elif has_position and row["ema_trend"] and row["ema_reclaim"] and regime >= cfg["add_min_score"]:
                add_mult = 2.0 if regime >= 90 else 1.25 if regime >= 70 else 0.75 if regime >= 50 else 0.3
                add_value = equity * 0.10 * add_mult
                cost = self.apply_cost(add_value)
                add_qty = (add_value - cost) / price

                if add_value < self.capital:
                    self.capital -= add_value
                    self.position.append(Trade(price, add_qty, bar, "TRIM"))

        # Close any remaining positions at end
        if self.position:
            final_price = df["Close"].iloc[-1]
            self._close_all(final_price, len(df)-1, "END OF TEST")

        return self._build_results()

    def _partial_close(self, trade, qty, price, bar, reason):
        qty = min(qty, trade.remaining_qty)
        if qty <= 0:
            return
        pnl = (price - trade.entry_price) * qty
        cost = self.apply_cost(price * qty)
        net_pnl = pnl - cost
        self.capital += price * qty - cost
        trade.remaining_qty -= qty

        self.closed_trades.append({
            "type": trade.trade_type,
            "entry_price": trade.entry_price,
            "exit_price": price,
            "qty": qty,
            "pnl": net_pnl,
            "pnl_pct": (price / trade.entry_price - 1) * 100,
            "bars_held": bar - trade.entry_bar,
            "reason": reason,
            "entry_date": self.df.index[trade.entry_bar],
            "exit_date": self.df.index[bar],
        })

    def _close_trade(self, trade, price, bar, reason):
        self._partial_close(trade, trade.remaining_qty, price, bar, reason)

    def _close_all(self, price, bar, reason):
        for trade in self.position[:]:
            self._close_trade(trade, price, bar, reason)
        self.position = []
        self.cooldown_until = bar + self.cfg["cooldown_bars"]

    def _build_results(self):
        eq = pd.DataFrame(self.equity_curve)
        if eq.empty:
            return None

        eq.set_index("date", inplace=True)
        trades_df = pd.DataFrame(self.closed_trades) if self.closed_trades else pd.DataFrame()

        # Metrics
        total_return = (eq["equity"].iloc[-1] / self.initial_capital - 1) * 100
        returns = pd.Series(self.daily_returns)
        n_trades = len(self.closed_trades)

        # Drawdown
        rolling_max = eq["equity"].cummax()
        drawdown = (eq["equity"] - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        # Win/loss
        if n_trades > 0:
            winners = [t for t in self.closed_trades if t["pnl"] > 0]
            losers = [t for t in self.closed_trades if t["pnl"] <= 0]
            winrate = len(winners) / n_trades * 100
            avg_win = np.mean([t["pnl"] for t in winners]) if winners else 0
            avg_loss = abs(np.mean([t["pnl"] for t in losers])) if losers else 1
            profit_factor = sum(t["pnl"] for t in winners) / abs(sum(t["pnl"] for t in losers)) if losers and sum(t["pnl"] for t in losers) != 0 else 999
            avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
            avg_bars = np.mean([t["bars_held"] for t in self.closed_trades])
        else:
            winrate = profit_factor = avg_rr = avg_bars = 0

        # Sharpe, Sortino, Calmar
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            downside = returns[returns < 0].std()
            sortino = (returns.mean() / downside) * np.sqrt(252) if downside > 0 else 0
        else:
            sharpe = sortino = 0

        years = len(returns) / 252 if len(returns) > 0 else 1
        cagr = ((eq["equity"].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        metrics = {
            "Total Return %": round(total_return, 2),
            "CAGR %": round(cagr, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Profit Factor": round(profit_factor, 2),
            "Winrate %": round(winrate, 1),
            "Avg R:R": round(avg_rr, 2),
            "Max Drawdown %": round(max_dd, 2),
            "Calmar Ratio": round(calmar, 2),
            "Total Trades": n_trades,
            "Avg Bars Held": round(avg_bars, 1) if n_trades > 0 else 0,
            "Final Equity": round(eq["equity"].iloc[-1], 2),
        }

        return {
            "metrics": metrics,
            "equity": eq,
            "drawdown": drawdown,
            "trades": trades_df,
            "returns": returns,
        }


# =============================================================================
# ACCEPT CRITERIA VALIDATION
# =============================================================================
ACCEPT_CRITERIA = {
    "Sharpe Ratio": (">=", 1.5),
    "Sortino Ratio": (">=", 2.0),
    "Profit Factor": (">=", 1.5),
    "Winrate %": (">=", 45),
    "Max Drawdown %": (">=", -15),
    "Avg R:R": (">=", 2.0),
    "Calmar Ratio": (">=", 1.0),
}

def validate_criteria(metrics):
    results = []
    for metric, (op, threshold) in ACCEPT_CRITERIA.items():
        value = metrics.get(metric, 0)
        if op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        results.append({
            "Metric": metric,
            "Value": value,
            "Threshold": f"{op} {threshold}",
            "Status": "PASS" if passed else "FAIL",
        })
    return pd.DataFrame(results)


# =============================================================================
# CHARTS
# =============================================================================
PALETTE = ['#20808D', '#A84B2F', '#1B474D', '#BCE2E7', '#944454', '#FFC553']

def plot_results(results, ticker, output_dir):
    eq = results["equity"]
    dd = results["drawdown"]
    metrics = results["metrics"]
    returns = results["returns"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"WOLF x SHADOW Backtest: {ticker}", fontsize=16, fontweight="bold", y=0.98)

    # --- Equity Curve ---
    ax1 = axes[0]
    ax1.plot(eq.index, eq["equity"], color=PALETTE[0], linewidth=1.5, label="Equity")
    ax1.axhline(y=100000, color="#757575", linestyle="--", alpha=0.5, label="Initial Capital")
    ax1.fill_between(eq.index, 100000, eq["equity"],
                     where=eq["equity"] >= 100000, alpha=0.15, color=PALETTE[0])
    ax1.fill_between(eq.index, 100000, eq["equity"],
                     where=eq["equity"] < 100000, alpha=0.15, color=PALETTE[1])
    ax1.set_ylabel("Equity ($)", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Total Return: {metrics['Total Return %']}% | Sharpe: {metrics['Sharpe Ratio']} | PF: {metrics['Profit Factor']}", fontsize=11)

    # --- Drawdown ---
    ax2 = axes[1]
    ax2.fill_between(dd.index, dd.values * 100, 0, color=PALETTE[1], alpha=0.6)
    ax2.set_ylabel("Drawdown %", fontweight="bold")
    ax2.set_title(f"Max Drawdown: {metrics['Max Drawdown %']}%", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # --- Monthly Returns Heatmap ---
    ax3 = axes[2]
    if len(returns) > 20:
        dates = eq.index[1:]  # returns is 1 shorter if aligned
        if len(dates) > len(returns):
            dates = dates[:len(returns)]
        elif len(returns) > len(dates):
            returns = returns[:len(dates)]

        monthly = pd.DataFrame({"date": dates, "return": returns.values})
        monthly["date"] = pd.to_datetime(monthly["date"])
        monthly["year"] = monthly["date"].dt.year
        monthly["month"] = monthly["date"].dt.month
        monthly_ret = monthly.groupby(["year", "month"])["return"].sum().unstack()

        if not monthly_ret.empty:
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            existing_months = [m for m in range(1, 13) if m in monthly_ret.columns]
            display_labels = [month_labels[m-1] for m in existing_months]

            im = ax3.imshow(monthly_ret[existing_months].values * 100,
                           cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
            ax3.set_yticks(range(len(monthly_ret.index)))
            ax3.set_yticklabels(monthly_ret.index)
            ax3.set_xticks(range(len(existing_months)))
            ax3.set_xticklabels(display_labels, fontsize=9)
            ax3.set_title("Monthly Returns %", fontsize=11)

            # Add text annotations
            for yi in range(len(monthly_ret.index)):
                for xi in range(len(existing_months)):
                    val = monthly_ret[existing_months].values[yi, xi]
                    if not np.isnan(val):
                        ax3.text(xi, yi, f"{val*100:.1f}", ha="center", va="center",
                                fontsize=8, color="black" if abs(val) < 0.05 else "white")
        else:
            ax3.text(0.5, 0.5, "Insufficient data for heatmap", transform=ax3.transAxes,
                    ha="center", va="center")
    else:
        ax3.text(0.5, 0.5, "Insufficient data for heatmap", transform=ax3.transAxes,
                ha="center", va="center")

    plt.tight_layout()
    chart_path = output_dir / f"backtest_{ticker.replace('.', '_')}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return chart_path


# =============================================================================
# WALK-FORWARD ANALYSIS
# =============================================================================
def walk_forward(df, spy_df, sec_df, is_months=12, oos_months=3):
    """Rolling walk-forward: optimize on IS, validate on OOS."""
    results = []
    total_days = len(df)
    is_days = is_months * 21  # approx trading days
    oos_days = oos_months * 21

    window = is_days + oos_days
    if total_days < window:
        print(f"  Not enough data for walk-forward ({total_days} < {window} days)")
        return None

    step = 0
    start = 200  # need lookback for indicators

    while start + window <= total_days:
        is_end = start + is_days
        oos_end = min(is_end + oos_days, total_days)

        is_slice = df.iloc[start:is_end]
        oos_slice = df.iloc[is_end:oos_end]

        if len(is_slice) < 100 or len(oos_slice) < 20:
            break

        # Run IS
        bt_is = Backtester(is_slice)
        res_is = bt_is.run()

        # Run OOS
        bt_oos = Backtester(oos_slice)
        res_oos = bt_oos.run()

        if res_is and res_oos:
            results.append({
                "Step": step + 1,
                "IS Start": is_slice.index[0].strftime("%Y-%m-%d"),
                "IS End": is_slice.index[-1].strftime("%Y-%m-%d"),
                "OOS Start": oos_slice.index[0].strftime("%Y-%m-%d"),
                "OOS End": oos_slice.index[-1].strftime("%Y-%m-%d"),
                "IS Return %": res_is["metrics"]["Total Return %"],
                "OOS Return %": res_oos["metrics"]["Total Return %"],
                "IS Sharpe": res_is["metrics"]["Sharpe Ratio"],
                "OOS Sharpe": res_oos["metrics"]["Sharpe Ratio"],
                "IS PF": res_is["metrics"]["Profit Factor"],
                "OOS PF": res_oos["metrics"]["Profit Factor"],
                "IS MaxDD %": res_is["metrics"]["Max Drawdown %"],
                "OOS MaxDD %": res_oos["metrics"]["Max Drawdown %"],
                "IS Trades": res_is["metrics"]["Total Trades"],
                "OOS Trades": res_oos["metrics"]["Total Trades"],
            })

        start += oos_days
        step += 1

    if not results:
        return None

    wf_df = pd.DataFrame(results)

    # OOS consistency check
    oos_positive = len(wf_df[wf_df["OOS Return %"] > 0])
    oos_total = len(wf_df)
    consistency = (oos_positive / oos_total * 100) if oos_total > 0 else 0

    return wf_df, consistency


# =============================================================================
# MAIN
# =============================================================================
def run_backtest(tickers, sector_etf="XLE", years=3, do_walk_forward=False):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  WOLF x SHADOW BACKTESTER v1.0")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Fetch SPY + sector
    print("\nFetching regime data (SPY + sector)...")
    spy_df, sec_df = fetch_regime_data("SPY", sector_etf, years)

    for ticker in tickers:
        print(f"\n{'='*70}")
        print(f"  BACKTESTING: {ticker} ({years} years)")
        print(f"{'='*70}")

        # Fetch stock data
        end = datetime.now()
        start = end - timedelta(days=years * 365 + 100)
        df = yf.download(ticker, start=start, end=end, progress=False)

        # Flatten MultiIndex columns if needed (yfinance batch format)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df is None or len(df) < 250:
            print(f"  Insufficient data for {ticker} ({len(df) if df is not None else 0} bars)")
            continue

        # Add indicators
        df = add_indicators(df)
        df = calc_regime_scores(spy_df, sec_df, df)
        df = df.dropna()

        print(f"  Data: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

        # Run backtest
        bt = Backtester(df)
        results = bt.run()

        if results is None:
            print("  No results (no trades generated)")
            continue

        # Print metrics
        print(f"\n  PERFORMANCE METRICS:")
        print(f"  {'-'*40}")
        for k, v in results["metrics"].items():
            print(f"  {k:25s}: {v}")

        # Validate accept criteria
        criteria = validate_criteria(results["metrics"])
        print(f"\n  ACCEPT CRITERIA:")
        print(f"  {'-'*40}")
        for _, row in criteria.iterrows():
            icon = "PASS" if row["Status"] == "PASS" else "FAIL"
            print(f"  {icon} {row['Metric']:25s}: {row['Value']:>8} (need {row['Threshold']})")

        passed = len(criteria[criteria["Status"] == "PASS"])
        total = len(criteria)
        print(f"\n  Result: {passed}/{total} criteria passed")

        # Generate charts
        chart_path = plot_results(results, ticker, output_dir)
        print(f"  Chart saved: {chart_path}")

        # Save trades
        if not results["trades"].empty:
            trades_path = output_dir / f"trades_{ticker.replace('.', '_')}.csv"
            results["trades"].to_csv(trades_path, index=False)
            print(f"  Trades saved: {trades_path}")

        # Walk-forward
        if do_walk_forward:
            print(f"\n  WALK-FORWARD ANALYSIS (12m IS / 3m OOS):")
            print(f"  {'-'*40}")
            wf_result = walk_forward(df, spy_df, sec_df)
            if wf_result:
                wf_df, consistency = wf_result
                print(wf_df.to_string(index=False))
                print(f"\n  OOS Consistency: {consistency:.0f}%")
                print(f"  {'PASS' if consistency >= 60 else 'FAIL'}: Need >= 60% positive OOS periods")

                wf_path = output_dir / f"walkforward_{ticker.replace('.', '_')}.csv"
                wf_df.to_csv(wf_path, index=False)
                print(f"  Walk-forward saved: {wf_path}")
            else:
                print("  Not enough data for walk-forward analysis")

    print(f"\n{'='*70}")
    print(f"  All results saved to: {output_dir}/")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="WOLF x SHADOW Backtester v2.2")
    parser.add_argument("--ticker", nargs="+", default=["SPY"],
                        help="Ticker(s) to backtest (default: SPY)")
    parser.add_argument("--sector", type=str, default="XLE",
                        help="Sector ETF for regime scoring (default: XLE)")
    parser.add_argument("--years", type=int, default=3,
                        help="Years of data (default: 3)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward analysis")
    parser.add_argument("--preset", type=str, default=None,
                        help="Preset name to use: any key from PRESET_PARAMS (XLE, XLB, ..., 'OMX Stockholm', 'Oslo OSEBX', OXY, Universal, etc.) or 'auto' for auto-detect (default: auto)")
    args = parser.parse_args()

    # Apply preset params to CONFIG
    if args.preset and args.preset.lower() != "auto":
        preset_key = args.preset
        if preset_key not in PRESET_PARAMS:
            print(f"  Unknown preset '{preset_key}'. Available: {list(PRESET_PARAMS.keys())}")
        else:
            p = PRESET_PARAMS[preset_key]
            CONFIG["ema_pulse"]      = p["ema_pulse"]
            CONFIG["ema_fast"]       = p["ema_fast"]
            CONFIG["ema_slow"]       = p["ema_slow"]
            CONFIG["atr_mult"]       = p["atr_mult"]
            CONFIG["adx_threshold"]  = p["adx_thresh"]
            CONFIG["tp1_rr"]         = p["tp1_rr"]
            CONFIG["tp1_pct"]        = p["tp1_pct"]
            CONFIG["tp2_rr"]         = p["tp2_rr"]
            CONFIG["tp2_pct"]        = p["tp2_pct"]
            CONFIG["core_pct"]       = p["core_pct"]
            CONFIG["entry_min_score"]= p["min_regime"]
            print(f"  Using preset: {preset_key} (ADX threshold: {p['adx_thresh']})")
    elif len(args.ticker) == 1:
        # Auto-detect preset from ticker
        ticker = args.ticker[0]
        p = get_preset_for_ticker(ticker)
        CONFIG["ema_pulse"]      = p["ema_pulse"]
        CONFIG["ema_fast"]       = p["ema_fast"]
        CONFIG["ema_slow"]       = p["ema_slow"]
        CONFIG["atr_mult"]       = p["atr_mult"]
        CONFIG["adx_threshold"]  = p["adx_thresh"]
        CONFIG["tp1_rr"]         = p["tp1_rr"]
        CONFIG["tp1_pct"]        = p["tp1_pct"]
        CONFIG["tp2_rr"]         = p["tp2_rr"]
        CONFIG["tp2_pct"]        = p["tp2_pct"]
        CONFIG["core_pct"]       = p["core_pct"]
        CONFIG["entry_min_score"]= p["min_regime"]
        print(f"  Auto-detected preset for {ticker} (ADX threshold: {p['adx_thresh']})")

    run_backtest(args.ticker, args.sector, args.years, args.walk_forward)


if __name__ == "__main__":
    main()
