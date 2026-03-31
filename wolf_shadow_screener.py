#!/usr/bin/env python3
"""
WOLF x SHADOW SCREENER v2.2
============================
Scans Commodity stocks, S&P 500, Stockholm (OMXSTO), and Oslo (OBX)
using the same 4-layer regime scoring as the Pine Script strategy.

Usage:
    python wolf_shadow_screener.py                  # Full scan, all markets
    python wolf_shadow_screener.py --market commodity  # Only commodity
    python wolf_shadow_screener.py --market sp500      # Only S&P 500
    python wolf_shadow_screener.py --market stockholm   # Only Stockholm
    python wolf_shadow_screener.py --market oslo        # Only Oslo
    python wolf_shadow_screener.py --min-score 70      # Only show score >= 70
"""

import argparse
import sys
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*auto_adjust.*")

# =============================================================================
# TICKER UNIVERSES
# =============================================================================

COMMODITY_TICKERS = {
    # Oil & Gas
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "EOG": "EOG Resources", "DVN": "Devon Energy", "FANG": "Diamondback",
    "OXY": "Occidental", "MPC": "Marathon Petroleum", "VLO": "Valero",
    "PSX": "Phillips 66", "HES": "Hess Corp", "CTRA": "Coterra Energy",
    "APA": "APA Corp", "EQT": "EQT Corp",
    # Gold miners
    "NEM": "Newmont", "GOLD": "Barrick Gold", "FNV": "Franco-Nevada",
    "WPM": "Wheaton Precious", "RGLD": "Royal Gold", "AGI": "Alamos Gold",
    "KGC": "Kinross Gold", "EQX": "Equinox Gold", "CDE": "Coeur Mining",
    "HL": "Hecla Mining", "PAAS": "Pan American Silver",
    "GDX": "VanEck Gold Miners ETF", "GDXJ": "VanEck Junior Gold ETF",
    # Silver miners
    "SIL": "Global X Silver Miners", "MAG": "MAG Silver",
    "AG": "First Majestic Silver",
    # Sector ETFs
    "XLE": "Energy Select ETF", "XLB": "Materials Select ETF",
    "GLD": "SPDR Gold Trust", "SLV": "iShares Silver Trust",
}

# S&P 500 top 100 by weight (covers ~75% of index)
SP500_TICKERS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet A", "META": "Meta", "BRK-B": "Berkshire", "LLY": "Eli Lilly",
    "AVGO": "Broadcom", "JPM": "JPMorgan", "TSLA": "Tesla", "UNH": "UnitedHealth",
    "XOM": "ExxonMobil", "V": "Visa", "MA": "Mastercard", "PG": "Procter & Gamble",
    "JNJ": "Johnson & Johnson", "HD": "Home Depot", "COST": "Costco",
    "ABBV": "AbbVie", "MRK": "Merck", "CVX": "Chevron", "CRM": "Salesforce",
    "KO": "Coca-Cola", "BAC": "BofA", "WMT": "Walmart", "PEP": "PepsiCo",
    "NFLX": "Netflix", "AMD": "AMD", "TMO": "Thermo Fisher",
    "LIN": "Linde", "ADBE": "Adobe", "ACN": "Accenture", "ORCL": "Oracle",
    "DIS": "Disney", "WFC": "Wells Fargo", "PM": "Philip Morris",
    "ABT": "Abbott", "CSCO": "Cisco", "IBM": "IBM", "NOW": "ServiceNow",
    "GE": "GE Aerospace", "CAT": "Caterpillar", "INTU": "Intuit",
    "QCOM": "Qualcomm", "TXN": "Texas Instruments", "ISRG": "Intuitive Surgical",
    "AMAT": "Applied Materials", "AMGN": "Amgen", "MS": "Morgan Stanley",
    "GS": "Goldman Sachs", "BLK": "BlackRock", "RTX": "RTX Corp",
    "DE": "Deere", "LOW": "Lowes", "BKNG": "Booking", "SYK": "Stryker",
    "UNP": "Union Pacific", "MDT": "Medtronic", "PFE": "Pfizer",
    "SPGI": "S&P Global", "ADP": "ADP", "MMC": "Marsh McLennan",
    "CB": "Chubb", "SCHW": "Charles Schwab", "DHR": "Danaher",
    "SO": "Southern Company", "CL": "Colgate", "BMY": "Bristol-Myers",
    "MO": "Altria", "GILD": "Gilead", "ZTS": "Zoetis", "CME": "CME Group",
}

# Stockholm OMXS30 + commodity-relevant Swedish stocks
STOCKHOLM_TICKERS = {
    "ABB.ST": "ABB", "ALFA.ST": "Alfa Laval", "ASSA-B.ST": "ASSA ABLOY",
    "AZN.ST": "AstraZeneca", "ATCO-A.ST": "Atlas Copco A",
    "ATCO-B.ST": "Atlas Copco B", "BOL.ST": "Boliden",
    "ELUX-B.ST": "Electrolux", "ERIC-B.ST": "Ericsson",
    "ESSITY-B.ST": "Essity", "EVO.ST": "Evolution",
    "GETI-B.ST": "Getinge", "HEXA-B.ST": "Hexagon",
    "HM-B.ST": "H&M", "INVE-B.ST": "Investor",
    "KINV-B.ST": "Kinnevik", "LUND-B.ST": "Lundin Mining",
    "SAND.ST": "Sandvik", "SCA-B.ST": "SCA",
    "SEB-A.ST": "SEB", "SHB-A.ST": "Handelsbanken",
    "SKA-B.ST": "Skanska", "SKF-B.ST": "SKF",
    "SSAB-A.ST": "SSAB A", "SSAB-B.ST": "SSAB B",
    "SWED-A.ST": "Swedbank", "SWMA.ST": "Swedish Match",
    "TEL2-B.ST": "Tele2", "TELIA.ST": "Telia",
    "VOLV-B.ST": "Volvo",
}

# Oslo OBX - commodity-heavy
OSLO_TICKERS = {
    "EQNR.OL": "Equinor", "DNB.OL": "DNB Bank", "MOWI.OL": "Mowi",
    "TEL.OL": "Telenor", "ORK.OL": "Orkla", "YAR.OL": "Yara",
    "NHY.OL": "Norsk Hydro", "AKRBP.OL": "Aker BP",
    "SALM.OL": "SalMar", "GOGL.OL": "Golden Ocean",
    "FRO.OL": "Frontline", "SUBC.OL": "Subsea 7",
    "VAR.OL": "Var Energi", "HAFNI.OL": "Hafnia",
    "TGS.OL": "TGS NOPEC", "HAUTO.OL": "Hoegh Autoliners",
    "AKER.OL": "Aker ASA", "AKSO.OL": "Aker Solutions",
    "NOD.OL": "Nordic Semi", "NEL.OL": "Nel ASA",
}

MARKETS = {
    "commodity": COMMODITY_TICKERS,
    "sp500": SP500_TICKERS,
    "stockholm": STOCKHOLM_TICKERS,
    "oslo": OSLO_TICKERS,
}

# =============================================================================
# PRESET PARAMETERS (v2.2 — ADX filter + per-instrument tuning)
# =============================================================================
PRESET_PARAMS = {
    "OXY":         {"ema_pulse":6,  "ema_fast":23, "ema_slow":40,  "tenkan":7,  "kijun":24, "spanb":65, "atr_mult":2.8, "adx_thresh":27, "tp1_rr":3.0, "tp1_pct":0.20, "tp2_rr":5.5, "tp2_pct":0.25, "core_pct":0.70, "min_regime":53},
    "XOM":         {"ema_pulse":13, "ema_fast":23, "ema_slow":39,  "tenkan":8,  "kijun":39, "spanb":48, "atr_mult":2.7, "adx_thresh":27, "tp1_rr":3.5, "tp1_pct":0.10, "tp2_rr":5.5, "tp2_pct":0.10, "core_pct":0.70, "min_regime":60},
    "GOLD":        {"ema_pulse":5,  "ema_fast":18, "ema_slow":71,  "tenkan":15, "kijun":36, "spanb":67, "atr_mult":1.5, "adx_thresh":16, "tp1_rr":1.75,"tp1_pct":0.20, "tp2_rr":5.75,"tp2_pct":0.05, "core_pct":0.50, "min_regime":49},
    "NEM":         {"ema_pulse":11, "ema_fast":13, "ema_slow":99,  "tenkan":9,  "kijun":33, "spanb":41, "atr_mult":3.1, "adx_thresh":17, "tp1_rr":3.0, "tp1_pct":0.05, "tp2_rr":4.25,"tp2_pct":0.25, "core_pct":0.60, "min_regime":44},
    "GLD":         {"ema_pulse":7,  "ema_fast":28, "ema_slow":37,  "tenkan":13, "kijun":22, "spanb":59, "atr_mult":2.6, "adx_thresh":7,  "tp1_rr":1.75,"tp1_pct":0.10, "tp2_rr":5.0, "tp2_pct":0.20, "core_pct":0.60, "min_regime":50},
    "Oil Sector":  {"ema_pulse":10, "ema_fast":23, "ema_slow":40,  "tenkan":8,  "kijun":32, "spanb":56, "atr_mult":2.8, "adx_thresh":27, "tp1_rr":3.2, "tp1_pct":0.15, "tp2_rr":5.5, "tp2_pct":0.18, "core_pct":0.70, "min_regime":56},
    "Gold Miners": {"ema_pulse":8,  "ema_fast":16, "ema_slow":85,  "tenkan":12, "kijun":34, "spanb":54, "atr_mult":2.3, "adx_thresh":16, "tp1_rr":2.4, "tp1_pct":0.12, "tp2_rr":5.0, "tp2_pct":0.15, "core_pct":0.55, "min_regime":46},
    "Universal":   {"ema_pulse":8,  "ema_fast":21, "ema_slow":57,  "tenkan":10, "kijun":31, "spanb":56, "atr_mult":2.5, "adx_thresh":19, "tp1_rr":2.6, "tp1_pct":0.13, "tp2_rr":5.2, "tp2_pct":0.17, "core_pct":0.62, "min_regime":51},
}


def get_preset_for_ticker(ticker):
    """Auto-detect which preset to use based on ticker symbol."""
    oil_tickers = {"XOM","CVX","COP","EOG","DVN","FANG","OXY","MPC","VLO","PSX","CTRA","APA","EQT","XLE","EQNR.OL","AKRBP.OL","VAR.OL"}
    gold_tickers = {"NEM","GOLD","FNV","WPM","RGLD","AGI","KGC","EQX","CDE","HL","PAAS","GDX","GDXJ","SIL","MAG","AG","GLD","SLV","BOL.ST","LUND-B.ST","NHY.OL"}
    if ticker in PRESET_PARAMS:
        return PRESET_PARAMS[ticker]
    if ticker in oil_tickers:
        return PRESET_PARAMS["Oil Sector"]
    if ticker in gold_tickers:
        return PRESET_PARAMS["Gold Miners"]
    return PRESET_PARAMS["Universal"]


# Sector ETF mapping (for regime scoring)
SECTOR_MAP = {
    "Energy": "XLE", "Materials": "XLB", "Financials": "XLF",
    "Technology": "XLK", "Healthcare": "XLV", "Industrials": "XLI",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Real Estate": "XLRE", "Utilities": "XLU",
    "Communication Services": "XLC",
}


# =============================================================================
# TECHNICAL CALCULATIONS
# =============================================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


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


def calc_ichimoku(high, low, close, conv=9, base=26, span_b=52, displacement=26):
    def donchian(h, l, period):
        return (h.rolling(period).max() + l.rolling(period).min()) / 2

    tenkan = donchian(high, low, conv)
    kijun = donchian(high, low, base)
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = donchian(high, low, span_b).shift(displacement)
    chikou = close.shift(-displacement)

    return tenkan, kijun, senkou_a, senkou_b, chikou


# =============================================================================
# SCORING ENGINE (mirrors Pine Script exactly)
# =============================================================================

def score_stock(df):
    """Score a single stock. Returns dict with all scores or None if insufficient data."""
    if df is None or len(df) < 200:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # EMAs
    ema10 = calc_ema(close, 10)
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    ema200 = calc_ema(close, 200)

    # RSI
    rsi = calc_rsi(close, 14)

    # ATR
    atr = calc_atr(high, low, close, 14)

    # ADX
    adx = calc_adx(high, low, close, 14)

    # Ichimoku
    tenkan, kijun, senkou_a, senkou_b, chikou = calc_ichimoku(high, low, close)

    # Latest values
    c = close.iloc[-1]
    e10 = ema10.iloc[-1]
    e20 = ema20.iloc[-1]
    e50 = ema50.iloc[-1]
    e200 = ema200.iloc[-1]
    r = rsi.iloc[-1]
    a = atr.iloc[-1]
    adx_val = adx.iloc[-1]
    t = tenkan.iloc[-1]
    k = kijun.iloc[-1]
    sa = senkou_a.iloc[-1] if not pd.isna(senkou_a.iloc[-1]) else 0
    sb = senkou_b.iloc[-1] if not pd.isna(senkou_b.iloc[-1]) else 0
    vol_ma = volume.rolling(20).mean().iloc[-1]

    # --- STOCK SCORE (max 50) ---
    stock_score = 0
    stock_score += 8 if e10 > e20 else 0
    stock_score += 8 if e20 > e50 else 0
    stock_score += 8 if c > e50 else 0
    stock_score += 8 if c > e200 else 0
    stock_score += 8 if r > 50 else 0
    # Momentum: RSI rising 3 bars
    if len(rsi) >= 3:
        stock_score += 10 if (rsi.iloc[-1] > rsi.iloc[-2] > rsi.iloc[-3]) else 0

    # --- ICHIMOKU SCORE (max 15) ---
    ichi_score = 0
    kumo_top = max(sa, sb)
    ichi_score += 5 if c > kumo_top else 0
    ichi_score += 5 if t > k else 0
    if len(close) > 26:
        ichi_score += 3 if c > close.iloc[-27] else 0
    ichi_score += 2 if sa > sb else 0

    # --- EMA TREND FLAGS ---
    ema_trend = e10 > e20 and e20 > e50 and c > e50
    ema_stack_full = e10 > e20 and e20 > e50 and e50 > e200

    # --- ENTRY SIGNALS ---
    # Order block (bullish)
    bull_ob = False
    if len(low) >= 6:
        recent_low = low.iloc[-6:-1].min()
        bull_ob = low.iloc[-1] <= recent_low and c > df["Open"].iloc[-1]

    # EMA crossover
    ema_cross_up = (ema10.iloc[-1] > ema20.iloc[-1]) and (ema10.iloc[-2] <= ema20.iloc[-2]) and c > e50

    # EMA reclaim
    ema_reclaim = c > e10 and close.iloc[-2] <= ema10.iloc[-2]

    # Momentum up
    momentum_up = False
    if len(rsi) >= 3:
        momentum_up = rsi.iloc[-1] > rsi.iloc[-2] > rsi.iloc[-3]

    has_entry = ema_trend and (bull_ob or ema_cross_up or ema_reclaim or momentum_up)

    # Candle patterns
    body = abs(c - df["Open"].iloc[-1])
    upper_shadow = high.iloc[-1] - max(c, df["Open"].iloc[-1])
    lower_shadow = min(c, df["Open"].iloc[-1]) - low.iloc[-1]
    bull_engulfing = (close.iloc[-2] < df["Open"].iloc[-2] and
                      c > df["Open"].iloc[-1] and
                      c > df["Open"].iloc[-2] and
                      df["Open"].iloc[-1] < close.iloc[-2])
    hammer = lower_shadow > body * 2 and upper_shadow < body * 0.5 and c > df["Open"].iloc[-1]
    candle_trigger = (bull_engulfing or hammer) and volume.iloc[-1] > vol_ma * 1.2

    # SL and entry zone
    sl_dist = a * 1.5
    entry_zone = c
    sl_level = c - sl_dist
    tp1_level = c + sl_dist * 2.0
    tp2_level = c + sl_dist * 3.0

    return {
        "stock_score": stock_score,
        "ichi_score": ichi_score,
        "ema_trend": ema_trend,
        "ema_stack": ema_stack_full,
        "has_entry": has_entry,
        "candle_trigger": candle_trigger,
        "close": round(c, 2),
        "ema10": round(e10, 2),
        "ema20": round(e20, 2),
        "ema50": round(e50, 2),
        "ema200": round(e200, 2),
        "rsi": round(r, 1),
        "atr": round(a, 2),
        "adx": round(adx_val, 1),
        "entry_zone": round(entry_zone, 2),
        "sl_level": round(sl_level, 2),
        "tp1_2R": round(tp1_level, 2),
        "tp2_3R": round(tp2_level, 2),
    }


def score_market_regime(spy_df):
    """Score SPY market regime (max 30)."""
    if spy_df is None or len(spy_df) < 200:
        return 0

    c = spy_df["Close"].iloc[-1]
    ema50 = calc_ema(spy_df["Close"], 50).iloc[-1]
    ema200 = calc_ema(spy_df["Close"], 200).iloc[-1]
    rsi = calc_rsi(spy_df["Close"], 14).iloc[-1]
    atr = calc_atr(spy_df["High"], spy_df["Low"], spy_df["Close"], 14).iloc[-1]
    atr_pct = (atr / c) * 100 if c > 0 else 5

    score = 0
    score += 10 if c > ema50 else 0
    score += 10 if c > ema200 else 0
    score += 5 if rsi > 50 else 0
    score += 5 if 0.3 < atr_pct < 4.0 else 0
    return score


def score_sector(sector_df):
    """Score sector ETF (max 30)."""
    if sector_df is None or len(sector_df) < 200:
        return 0

    c = sector_df["Close"].iloc[-1]
    ema50 = calc_ema(sector_df["Close"], 50).iloc[-1]
    ema200 = calc_ema(sector_df["Close"], 200).iloc[-1]
    rsi = calc_rsi(sector_df["Close"], 14).iloc[-1]

    score = 0
    score += 10 if c > ema50 else 0
    score += 10 if c > ema200 else 0
    score += 10 if rsi > 50 else 0
    return score


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(tickers, period="1y", interval="1d"):
    """Fetch OHLCV data for multiple tickers."""
    print(f"  Fetching data for {len(tickers)} tickers...")
    data = {}
    ticker_list = list(tickers.keys())

    # Batch download
    try:
        raw = yf.download(ticker_list, period=period, interval=interval,
                          group_by="ticker", progress=False, threads=True)
        for ticker in ticker_list:
            try:
                if len(ticker_list) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()
                df = df.dropna(subset=["Close"])
                if len(df) >= 50:
                    data[ticker] = df
            except (KeyError, TypeError):
                pass
    except Exception as e:
        print(f"  Batch download failed: {e}")
        # Fallback: individual downloads
        for ticker in ticker_list:
            try:
                df = yf.download(ticker, period=period, interval=interval, progress=False)
                df = df.dropna(subset=["Close"])
                if len(df) >= 50:
                    data[ticker] = df
            except Exception:
                pass

    print(f"  Got data for {len(data)}/{len(tickers)} tickers")
    return data


# =============================================================================
# MAIN SCREENER
# =============================================================================

def run_screener(markets=None, min_score=0):
    """Run the full screener."""
    if markets is None:
        markets = list(MARKETS.keys())

    print("=" * 70)
    print("  WOLF x SHADOW SCREENER v1.0")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # 1. Fetch SPY for market regime
    print("\n[1/4] Market Regime (SPY)...")
    spy_data = fetch_data({"SPY": "S&P 500 ETF"}, period="1y")
    spy_df = spy_data.get("SPY")
    market_score = score_market_regime(spy_df)
    print(f"  Market Score: {market_score}/30")

    # 2. Fetch sector ETFs
    print("\n[2/4] Sector ETFs...")
    sector_etfs = {v: k for k, v in SECTOR_MAP.items()}
    sector_data = fetch_data({etf: name for etf, name in sector_etfs.items()}, period="1y")
    sector_scores = {}
    for etf, df in sector_data.items():
        sector_scores[etf] = score_sector(df)
        status = "OK" if sector_scores[etf] == 30 else "WEAK"
        print(f"  {etf}: {sector_scores[etf]}/30 [{status}]")

    # 3. Scan all selected markets
    all_results = []

    for market_name in markets:
        if market_name not in MARKETS:
            print(f"\n  Unknown market: {market_name}")
            continue

        tickers = MARKETS[market_name]
        print(f"\n[3/4] Scanning {market_name.upper()} ({len(tickers)} tickers)...")

        stock_data = fetch_data(tickers, period="1y")

        for ticker, df in stock_data.items():
            result = score_stock(df)
            if result is None:
                continue

            # Determine sector ETF for this stock
            # Default to XLK for tech, XLE for energy, etc.
            sector_etf_score = 15  # default mid-score if unknown

            # Try to detect sector from ticker
            if ticker in ["XLE", "XOM", "CVX", "COP", "EOG", "DVN", "FANG",
                          "OXY", "MPC", "VLO", "PSX", "HES", "CTRA", "APA",
                          "EQT", "EQNR.OL", "AKRBP.OL", "VAR.OL"]:
                sector_etf_score = sector_scores.get("XLE", 15)
            elif ticker in ["NEM", "GOLD", "FNV", "WPM", "RGLD", "AGI",
                            "KGC", "EQX", "CDE", "HL", "PAAS", "GDX",
                            "GDXJ", "SIL", "MAG", "AG", "GLD", "SLV",
                            "BOL.ST", "LUND-B.ST", "NHY.OL"]:
                sector_etf_score = sector_scores.get("XLB", 15)
            elif ticker in ["XLK", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
                            "AVGO", "AMD", "QCOM", "TXN", "AMAT", "ADBE",
                            "CRM", "ORCL", "NOW", "INTU", "NOD.OL"]:
                sector_etf_score = sector_scores.get("XLK", 15)
            elif ticker in ["XLV", "LLY", "UNH", "JNJ", "ABBV", "MRK",
                            "TMO", "ABT", "PFE", "AMGN", "GILD", "BMY",
                            "ISRG", "SYK", "MDT", "ZTS", "AZN.ST",
                            "GETI-B.ST"]:
                sector_etf_score = sector_scores.get("XLV", 15)
            elif ticker in ["XLF", "JPM", "BAC", "WFC", "GS", "MS",
                            "BLK", "SCHW", "CB", "CME", "SPGI",
                            "SEB-A.ST", "SHB-A.ST", "SWED-A.ST",
                            "DNB.OL"]:
                sector_etf_score = sector_scores.get("XLF", 15)

            # Total regime score
            total_score = market_score + sector_etf_score + result["stock_score"] + result["ichi_score"]

            preset = get_preset_for_ticker(ticker)
            preset_name = (
                ticker if ticker in PRESET_PARAMS
                else "Oil Sector" if ticker in {"XOM","CVX","COP","EOG","DVN","FANG","OXY","MPC","VLO","PSX","CTRA","APA","EQT","XLE","EQNR.OL","AKRBP.OL","VAR.OL"}
                else "Gold Miners" if ticker in {"NEM","GOLD","FNV","WPM","RGLD","AGI","KGC","EQX","CDE","HL","PAAS","GDX","GDXJ","SIL","MAG","AG","GLD","SLV","BOL.ST","LUND-B.ST","NHY.OL"}
                else "Universal"
            )
            row = {
                "Market": market_name.upper(),
                "Ticker": ticker,
                "Name": tickers.get(ticker, ticker),
                "Preset": preset_name,
                "Close": result["close"],
                "Total Score": total_score,
                "Market(30)": market_score,
                "Sector(30)": sector_etf_score,
                "Stock(50)": result["stock_score"],
                "Ichi(15)": result["ichi_score"],
                "EMA Trend": "YES" if result["ema_trend"] else "no",
                "EMA Stack": "FULL" if result["ema_stack"] else "no",
                "Entry Signal": "YES" if result["has_entry"] else "no",
                "Candle Trig": "YES" if result["candle_trigger"] else "no",
                "RSI": result["rsi"],
                "ADX": result["adx"],
                "ATR": result["atr"],
                "Entry Zone": result["entry_zone"],
                "SL (1.5 ATR)": result["sl_level"],
                "TP1 (2R)": result["tp1_2R"],
                "TP2 (3R)": result["tp2_3R"],
            }
            all_results.append(row)

    # 4. Build results DataFrame
    if not all_results:
        print("\nNo results found!")
        return None

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("Total Score", ascending=False)

    # Filter by min score
    if min_score > 0:
        df_results = df_results[df_results["Total Score"] >= min_score]

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"wolf_shadow_scan_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Market Regime (SPY): {market_score}/30")
    print(f"  Total stocks scanned: {len(all_results)}")
    print(f"  Stocks with score >= 70: {len(df_results[df_results['Total Score'] >= 70])}")
    print(f"  Stocks with entry signal: {len(df_results[df_results['Entry Signal'] == 'YES'])}")

    # Top 20
    print(f"\n  TOP 20 BY REGIME SCORE:")
    print("-" * 70)
    top = df_results.head(20)
    cols_display = ["Ticker", "Name", "Preset", "Total Score", "Stock(50)", "Ichi(15)",
                    "EMA Stack", "Entry Signal", "RSI", "ADX", "Close"]
    print(top[cols_display].to_string(index=False))

    # Entry signals
    entries = df_results[df_results["Entry Signal"] == "YES"]
    if len(entries) > 0:
        print(f"\n  ACTIVE ENTRY SIGNALS ({len(entries)}):")
        print("-" * 70)
        entry_cols = ["Ticker", "Name", "Total Score", "Entry Zone",
                      "SL (1.5 ATR)", "TP1 (2R)", "TP2 (3R)", "RSI"]
        print(entries[entry_cols].head(20).to_string(index=False))

    print(f"\n  Full results saved to: {csv_path}")
    print("=" * 70)

    return df_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="WOLF x SHADOW Screener")
    parser.add_argument("--market", type=str, default=None,
                        help="Market to scan: commodity, sp500, stockholm, oslo (default: all)")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Minimum regime score to show (default: 0)")
    args = parser.parse_args()

    markets = [args.market] if args.market else None
    run_screener(markets=markets, min_score=args.min_score)


if __name__ == "__main__":
    main()
