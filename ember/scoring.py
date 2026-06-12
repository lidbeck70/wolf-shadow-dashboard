"""
ember/scoring.py
EMBER strategy — macro and sentiment scoring.

MacroScore   (0-100): copper/gold ratio, DXY 4W, FRED T10Y2Y, cycle phase.
SentimentScore (0-100): short float, analyst ratio, put/call — US only.

DATA_GAP sources contribute 0 pts and are flagged for the UI.
Nordic tickers (.ST/.OL/.CO/.HE) return is_data_gap=True for sentiment.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ember.config import (
    MACRO_W_COPPER_GOLD, MACRO_W_DXY, MACRO_W_YIELD_CURVE, MACRO_W_CYCLE,
    SENTIMENT_W_SHORT, SENTIMENT_W_ANALYST, SENTIMENT_W_OPTIONS,
    FRED_T10Y2Y_URL, FRED_TIMEOUT, DXY_PRIMARY, DXY_FALLBACK,
    CYCLE_BONUS_TIDIG, CYCLE_BONUS_MITTEN, CYCLE_BONUS_SEN, CYCLE_BONUS_TOPP,
)

logger = logging.getLogger(__name__)

_SHORT_HIGH = 20.0   # > 20% short float → max contrarian signal
_SHORT_MED  = 10.0   # > 10% → moderate signal


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class MacroScore:
    total: float = 0.0
    copper_gold: float = 0.0
    dxy: float = 0.0
    yield_curve: float = 0.0
    cycle: float = 0.0
    copper_gold_detail: str = ""
    dxy_detail: str = ""
    yield_curve_detail: str = ""
    cycle_detail: str = ""
    data_gaps: list[str] = field(default_factory=list)


@dataclass
class SentimentScore:
    total: float = 0.0
    short_float: float = 0.0
    analyst: float = 0.0
    put_call: float = 0.0
    short_float_detail: str = ""
    analyst_detail: str = ""
    put_call_detail: str = ""
    is_data_gap: bool = False
    data_gap_reason: str = ""


# ── Download helper ───────────────────────────────────────────────────────────

def _close_simple(ticker: str, period: str = "3mo") -> pd.Series:
    try:
        import yfinance as yf
        try:
            df = yf.download(
                ticker, period=period, auto_adjust=True,
                progress=False, show_errors=False, multi_level_index=False,
            )
        except TypeError:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Close"].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s.dropna()
    except Exception as exc:
        logger.debug("_close_simple(%s): %s", ticker, exc)
        return pd.Series(dtype=float)


# ── Macro sub-scores ──────────────────────────────────────────────────────────

def _score_copper_gold(ratios_dict: Optional[dict]) -> tuple[float, str]:
    if ratios_dict is None:
        return 0.0, "DATA_GAP — ratios ej tillgängliga"
    r = ratios_dict.get("copper_gold")
    if r is None or getattr(r, "status", "DATA_GAP") == "DATA_GAP":
        return 0.0, "DATA_GAP — copper/gold ratio"
    try:
        vals = r.sparkline_values
        if len(vals) < 4:
            return 50.0, f"Copper/Gold {r.current:.4f} (otillräcklig historik)"
        change = (vals[-1] - vals[-4]) / abs(vals[-4]) * 100 if vals[-4] != 0 else 0.0
        score  = 100.0 if change > 0 else 0.0
        label  = "COPPER LEDER ✓" if score > 0 else "COPPER SVAG"
        return score, f"Copper/Gold {r.current:.4f} ({change:+.1f}% 4V) → {label}"
    except Exception as exc:
        return 0.0, f"DATA_GAP: {exc}"


def _score_dxy() -> tuple[float, str]:
    try:
        c = _close_simple(DXY_PRIMARY, "3mo")
        if c.empty:
            c = _close_simple(DXY_FALLBACK, "3mo")
        if len(c) < 20:
            return 0.0, "DATA_GAP — DXY ej tillgänglig"
        change = (float(c.iloc[-1]) / float(c.iloc[-20]) - 1) * 100
        score  = 100.0 if change < 0 else 0.0
        label  = "FALLANDE → MEDVIND ✓" if score > 0 else "STIGANDE → MOTVIND"
        return score, f"DXY 4V {change:+.1f}% → {label}"
    except Exception as exc:
        return 0.0, f"DATA_GAP: {exc}"


def _score_yield_curve() -> tuple[float, str]:
    try:
        import requests
        resp = requests.get(FRED_T10Y2Y_URL, timeout=FRED_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), skiprows=1, names=["date", "value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).tail(30)
        if len(df) < 20:
            return 50.0, f"T10Y2Y otillräcklig historik ({len(df)} datapunkter)"
        vals   = df["value"].values
        change = float(vals[-1]) - float(vals[-20])
        score  = 100.0 if change > 0 else 0.0
        label  = "BRANTARE ✓" if score > 0 else "FLATARE"
        return score, f"T10Y2Y {vals[-1]:.2f}% (1M {change:+.2f}pp) → {label}"
    except Exception as exc:
        logger.debug("_score_yield_curve: %s", exc)
        return 0.0, f"DATA_GAP — FRED: {exc}"


def _score_cycle(cykel_label: Optional[str]) -> tuple[float, str]:
    if not cykel_label:
        return 50.0, "Cykelfas okänd — neutralt"
    _MAP = {"TIDIG": 100.0, "MITTEN": 75.0, "SEN": 25.0, "TOPP": 0.0, "DATA_GAP": 50.0}
    score = _MAP.get(cykel_label.upper(), 50.0)
    return score, f"Cykelfas: {cykel_label} → {score:.0f}/100"


# ── MacroScore ────────────────────────────────────────────────────────────────

def compute_macro_score(
    ratios_dict: Optional[dict],
    cykel_label: Optional[str],
) -> MacroScore:
    ms = MacroScore()

    cg, cg_d = _score_copper_gold(ratios_dict)
    dx, dx_d = _score_dxy()
    yc, yc_d = _score_yield_curve()
    cy, cy_d = _score_cycle(cykel_label)

    ms.copper_gold        = round(cg, 1)
    ms.dxy                = round(dx, 1)
    ms.yield_curve        = round(yc, 1)
    ms.cycle              = round(cy, 1)
    ms.copper_gold_detail = cg_d
    ms.dxy_detail         = dx_d
    ms.yield_curve_detail = yc_d
    ms.cycle_detail       = cy_d

    w = MACRO_W_COPPER_GOLD + MACRO_W_DXY + MACRO_W_YIELD_CURVE + MACRO_W_CYCLE
    ms.total = round(
        (cg * MACRO_W_COPPER_GOLD + dx * MACRO_W_DXY
         + yc * MACRO_W_YIELD_CURVE + cy * MACRO_W_CYCLE) / w,
        1,
    )

    for flag, src in [(cg_d, "Copper/Gold"), (dx_d, "DXY"), (yc_d, "T10Y2Y")]:
        if "DATA_GAP" in flag:
            ms.data_gaps.append(src)

    return ms


# ── SentimentScore ────────────────────────────────────────────────────────────

def compute_sentiment_score(ticker: str) -> SentimentScore:
    ss = SentimentScore()

    # Nordic tickers have no yfinance sentiment data
    if any(ticker.upper().endswith(s) for s in (".ST", ".OL", ".CO", ".HE")):
        ss.is_data_gap = True
        ss.data_gap_reason = f"Nordisk ticker {ticker} — sentiment ej tillgängligt"
        ss.total = float("nan")
        return ss

    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        info = t.info or {}

        # Short float
        short_raw = info.get("shortPercentOfFloat")
        if short_raw is not None:
            pct = short_raw * 100 if short_raw < 1.0 else float(short_raw)
            if pct > _SHORT_HIGH:
                ss.short_float = 100.0
            elif pct > _SHORT_MED:
                ss.short_float = 60.0
            else:
                ss.short_float = 20.0
            ss.short_float_detail = f"Short float: {pct:.1f}%"
        else:
            ss.short_float_detail = "DATA_GAP — short float ej tillgänglig"

        # Analyst consensus
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                row = recs.iloc[-1]
                buy_cols  = [c for c in recs.columns
                             if "buy" in str(c).lower() or "strong" in str(c).lower()]
                sell_cols = [c for c in recs.columns
                             if "sell" in str(c).lower() or "underperform" in str(c).lower()]
                hold_cols = [c for c in recs.columns if "hold" in str(c).lower()]
                all_cols  = buy_cols + sell_cols + hold_cols
                if all_cols:
                    buys  = sum(int(row.get(c, 0) or 0) for c in buy_cols)
                    total = sum(int(row.get(c, 0) or 0) for c in all_cols)
                    if total > 0:
                        ratio = buys / total
                        ss.analyst = round(ratio * 100, 1)
                        ss.analyst_detail = f"KÖP-andel {buys}/{total} = {ratio*100:.0f}%"
                    else:
                        ss.analyst_detail = "DATA_GAP — summa 0"
                else:
                    ss.analyst_detail = "DATA_GAP — kolumnformat okänt"
            else:
                ss.analyst_detail = "DATA_GAP — inga rekommendationer"
        except Exception as exc:
            ss.analyst_detail = f"DATA_GAP — recs: {exc}"

        # Put/call ratio
        try:
            chain = t.option_chain()
            if chain is not None:
                p_vol = float((chain.puts["volume"].sum()
                               if "volume" in chain.puts.columns else 0) or 0)
                c_vol = float((chain.calls["volume"].sum()
                               if "volume" in chain.calls.columns else 0) or 0)
                if c_vol > 0:
                    pc = p_vol / c_vol
                    ss.put_call = min(100.0, round(pc * 50, 1))
                    ss.put_call_detail = f"Put/Call {pc:.2f} (hög = kontrariansk möjlighet)"
                else:
                    ss.put_call_detail = "DATA_GAP — optionsvolym = 0"
            else:
                ss.put_call_detail = "DATA_GAP — option chain ej tillgänglig"
        except Exception as exc:
            ss.put_call_detail = f"DATA_GAP — options: {exc}"

    except Exception as exc:
        logger.debug("compute_sentiment_score(%s): %s", ticker, exc)
        ss.is_data_gap = True
        ss.data_gap_reason = f"yfinance-fel: {exc}"
        ss.total = float("nan")
        return ss

    # Weighted total — skip DATA_GAP sources
    pts, wsum = 0.0, 0
    for val, det, w in [
        (ss.short_float, ss.short_float_detail, SENTIMENT_W_SHORT),
        (ss.analyst,     ss.analyst_detail,     SENTIMENT_W_ANALYST),
        (ss.put_call,    ss.put_call_detail,     SENTIMENT_W_OPTIONS),
    ]:
        if "DATA_GAP" not in det:
            pts  += val * w
            wsum += w

    if wsum > 0:
        ss.total = round(pts / wsum, 1)
    else:
        ss.is_data_gap = True
        ss.data_gap_reason = "Alla sentimentkällor är DATA_GAP"
        ss.total = float("nan")

    return ss


# ── Cycle asymmetry bonus ─────────────────────────────────────────────────────

def cycle_asymmetry_bonus(cykel_label: Optional[str]) -> float:
    """Extra ranking points based on cycle phase (TIDIG = most attractive)."""
    _MAP = {
        "TIDIG":    CYCLE_BONUS_TIDIG,
        "MITTEN":   CYCLE_BONUS_MITTEN,
        "SEN":      CYCLE_BONUS_SEN,
        "TOPP":     CYCLE_BONUS_TOPP,
        "DATA_GAP": 0.0,
    }
    return _MAP.get((cykel_label or "").upper(), 0.0)
