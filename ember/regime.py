"""
ember/regime.py
EMBER strategy — master environment gauge for commodity swing trading.

Five pillars (each GREEN / AMBER / RED with actual values shown):
  1. DOLLAR       — DXY 4-week trend
  2. TILLVÄXTPULS — Copper/Gold ratio 3-month trend
  3. RÄNTEKURVA   — T10Y2Y from FRED, steepening 4W
  4. TEMA-BREDD   — share of Theme Board themes TIDIG/MITTEN + positive 3m momentum
  5. RISKAPTIT    — GDX/SPY 3m + HYG/TLT 1m relative performance

Regime verdict:
  ≥4 GREEN → PÅ       (full position sizing)
   3 GREEN → SELEKTIV  (half position sizing)
  ≤2 GREEN → AV        (no new trades)

Session state key: "ember_regime" (EmberRegimeResult).
DATA_GAP is never treated as GREEN — it counts as AMBER for the verdict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from ember.config import (
    DXY_PRIMARY, DXY_FALLBACK,
    FRED_T10Y2Y_URL, FRED_TIMEOUT,
    BG, BG2, EMBER, GOLD, GREEN, RED, AMBER, TEXT, DIM,
)

logger = logging.getLogger(__name__)

# ── Cache (graceful without Streamlit) ───────────────────────────────────────
try:
    import streamlit as st
    def _cache_1h(fn):
        return st.cache_data(ttl=3600, show_spinner=False)(fn)
except ImportError:
    def _cache_1h(fn):
        return fn

# ── Regime threshold constants (imported by rules_page.py) ───────────────────
DXY_FLAT_PCT      = 0.5    # 4W DXY change below this → not clearly falling (AMBER boundary)
DXY_SURGE_REGIME  = 2.0    # 4W DXY change above this → RED
CG_FLAT_PCT       = 2.0    # |Copper/Gold 3M change| < 2% → flat (AMBER)
YC_STEEPEN_PP     = 0.05   # T10Y2Y 4W change > 0.05pp → steepening (GREEN)
YC_INVERT_PP      = -0.20  # T10Y2Y 4W change < -0.20pp → inverting deeper (RED)
TEMA_GREEN_PCT    = 50.0   # > 50% themes TIDIG/MITTEN + positive 3m → GREEN
TEMA_AMBER_PCT    = 25.0   # 25–50% → AMBER

# ── Verdict constants ─────────────────────────────────────────────────────────
VERDICT_PA       = "PÅ"
VERDICT_SELEKTIV = "SELEKTIV"
VERDICT_AV       = "AV"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PillarResult:
    name:   str            # Swedish display name
    status: str            # "GREEN" | "AMBER" | "RED" | "DATA_GAP"
    value:  str            # actual value string shown in card
    detail: str            # one-line Swedish explanation


@dataclass
class EmberRegimeResult:
    pillars:     list[PillarResult]
    green_count: int
    verdict:     str            # VERDICT_PA | VERDICT_SELEKTIV | VERDICT_AV
    action_text: str
    timestamp:   datetime = field(default_factory=datetime.now)


@dataclass
class ComplexRegimeResult:
    key:         str            # "energi" | "adelmetaller" | "basmetaller" | "agri"
    label:       str            # Swedish display label
    pillars:     list[PillarResult]
    green_count: int
    verdict:     str            # VERDICT_PA | VERDICT_SELEKTIV | VERDICT_AV
    action_text: str
    timestamp:   datetime = field(default_factory=datetime.now)


# ── Download helper ───────────────────────────────────────────────────────────

def _close(ticker: str, period: str) -> pd.Series:
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
        logger.debug("_close(%s, %s): %s", ticker, period, exc)
        return pd.Series(dtype=float)


# ── Commodity complex map ─────────────────────────────────────────────────────

TICKER_COMPLEX_MAP: dict[str, str] = {
    # ENERGI — olja, gas, kol
    "XOM": "energi", "CVX": "energi", "COP": "energi", "EOG": "energi",
    "SLB": "energi", "MPC": "energi", "VLO": "energi", "PSX": "energi",
    "OXY": "energi", "HAL": "energi", "DVN": "energi", "BKR": "energi",
    "FANG": "energi", "APA": "energi", "MRO": "energi", "SHEL": "energi",
    "EQNR.OL": "energi", "AKRBP.OL": "energi", "VAR.OL": "energi", "TGS.OL": "energi",
    "XLE": "energi", "XOP": "energi", "USO": "energi", "UNG": "energi",
    "BP.L": "energi", "SHEL.L": "energi",
    "SU.TO": "energi", "CNQ.TO": "energi", "CVE.TO": "energi", "IMO.TO": "energi",
    "WCP.TO": "energi", "ARX.TO": "energi", "BTE.TO": "energi", "TOU.TO": "energi",
    "BTU": "energi", "ARCH": "energi", "CEIX": "energi", "AMR": "energi",
    # ÄDELMETALLER — guld, silver, PGM, sällsynta jordartsmetaller
    "NEM": "adelmetaller", "GOLD": "adelmetaller", "AEM": "adelmetaller",
    "WPM": "adelmetaller", "KGC": "adelmetaller", "AGI": "adelmetaller",
    "AU": "adelmetaller", "GFI": "adelmetaller", "BTG": "adelmetaller",
    "EGO": "adelmetaller", "SSRM": "adelmetaller", "OR": "adelmetaller",
    "SA": "adelmetaller", "HMY": "adelmetaller", "DRD": "adelmetaller",
    "AG": "adelmetaller", "HL": "adelmetaller", "PAAS": "adelmetaller",
    "CDE": "adelmetaller", "FSM": "adelmetaller", "EXK": "adelmetaller",
    "MAG": "adelmetaller", "MUX": "adelmetaller", "GPL": "adelmetaller",
    "SVM": "adelmetaller", "ASM": "adelmetaller", "NGD": "adelmetaller",
    "GLD": "adelmetaller", "GDX": "adelmetaller", "GDXJ": "adelmetaller",
    "SLV": "adelmetaller", "SIL": "adelmetaller", "SILJ": "adelmetaller",
    "MP": "adelmetaller", "REMX": "adelmetaller",
    "ABX.TO": "adelmetaller", "K.TO": "adelmetaller", "ERO.TO": "adelmetaller",
    "AGI.TO": "adelmetaller", "BTO.TO": "adelmetaller", "EDV.TO": "adelmetaller",
    "WPM.TO": "adelmetaller", "FNV.TO": "adelmetaller",
    "RIO.L": "adelmetaller", "BHP.L": "adelmetaller", "AAL.L": "adelmetaller",
    "FRES.L": "adelmetaller", "ANTO.L": "adelmetaller",
    # BASMETALLER — koppar, nickel, zink, litium
    "FCX": "basmetaller", "SCCO": "basmetaller", "TECK": "basmetaller",
    "COPX": "basmetaller", "PICK": "basmetaller", "XME": "basmetaller",
    "LIT": "basmetaller", "FM.TO": "basmetaller", "LUN.TO": "basmetaller",
    "GLEN.L": "basmetaller",
    # AGRI & ÖVRIGT — jordbruk + uran (diversified)
    "MOS": "agri", "NTR": "agri", "CF": "agri", "UAN": "agri", "ADM": "agri",
    "NTR.TO": "agri", "DBA": "agri",
    "CCJ": "agri", "NXE": "agri", "DNN": "agri", "UUUU": "agri",
    "LEU": "agri", "UEC": "agri", "URA": "agri", "URNM": "agri",
    "CCO.TO": "agri", "DML.TO": "agri", "NXE.TO": "agri",
}


def detect_complex(ticker: str) -> str:
    """Map a ticker to its commodity complex key. Defaults to 'energi'."""
    return TICKER_COMPLEX_MAP.get(ticker.upper(), "energi")


# ── Shared generic pillar helpers ─────────────────────────────────────────────

def _p_50w_ema(ticker: str, label: str) -> PillarResult:
    """Is price above its 50-week EMA?"""
    try:
        c = _close(ticker, "5y")
        if len(c) < 52:
            return PillarResult(name=label, status="DATA_GAP",
                                value="DATA_GAP", detail=f"{ticker} ej tillgänglig")
        cw = c.resample("W").last().dropna()
        if len(cw) < 52:
            return PillarResult(name=label, status="DATA_GAP",
                                value="DATA_GAP", detail="För lite veckodata")
        ema50w = float(cw.ewm(span=50, adjust=False).mean().iloc[-1])
        price  = float(cw.iloc[-1])
        pct    = (price / ema50w - 1) * 100
        if price > ema50w:
            return PillarResult(name=label, status="GREEN",
                                value=f"{price:.2f} (EMA50V {ema50w:.2f})",
                                detail=f"{ticker} {pct:+.1f}% över 50V EMA → upptrend ✓")
        return PillarResult(name=label, status="RED",
                            value=f"{price:.2f} (EMA50V {ema50w:.2f})",
                            detail=f"{ticker} {pct:+.1f}% under 50V EMA → nedtrend ⛔")
    except Exception as exc:
        return PillarResult(name=label, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


def _p_rs_3m(ticker: str, bench: str, label: str) -> PillarResult:
    """3-month (63 trading day) relative strength of ticker vs benchmark."""
    try:
        t = _close(ticker, "6mo")
        b = _close(bench,  "6mo")
        if len(t) < 64 or len(b) < 64:
            return PillarResult(name=label, status="DATA_GAP",
                                value="DATA_GAP", detail=f"{ticker}/{bench} ej tillgänglig")
        t3m = float(t.iloc[-1]) / float(t.iloc[-64]) - 1
        b3m = float(b.iloc[-1]) / float(b.iloc[-64]) - 1
        rs  = t3m - b3m
        val = f"{ticker} {t3m*100:+.1f}% vs {bench} {b3m*100:+.1f}% (3M)"
        if rs > 0.02:
            return PillarResult(name=label, status="GREEN", value=val,
                                detail=f"{ticker} outperformar {bench} 3M → relativ styrka ✓")
        if rs < -0.02:
            return PillarResult(name=label, status="RED", value=val,
                                detail=f"{ticker} underperformar {bench} 3M → svag RS ⛔")
        return PillarResult(name=label, status="AMBER", value=val,
                            detail=f"{ticker} i linje med {bench} 3M → neutral RS")
    except Exception as exc:
        return PillarResult(name=label, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


# ── Pillar 1: DOLLAR (DXY 4-week trend) ──────────────────────────────────────

def _pillar_dxy() -> PillarResult:
    name = "DOLLAR (DXY)"
    try:
        c = _close(DXY_PRIMARY, "3mo")
        if len(c) < 22:
            c = _close(DXY_FALLBACK, "3mo")
        if len(c) < 22:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP",
                                detail="DXY ej tillgänglig via yfinance")
        cur        = float(c.iloc[-1])
        change_pct = (cur / float(c.iloc[-21]) - 1) * 100
        if change_pct < -DXY_FLAT_PCT:
            status = "GREEN"
            detail = f"DXY fallande {change_pct:+.1f}% 4V → råvarumedvind ✓"
        elif change_pct > DXY_SURGE_REGIME:
            status = "RED"
            detail = f"DXY rally {change_pct:+.1f}% 4V → råvarumotvind ⛔"
        else:
            status = "AMBER"
            detail = f"DXY sidorörelsig {change_pct:+.1f}% 4V → neutral"
        return PillarResult(name=name, status=status,
                            value=f"{cur:.2f} ({change_pct:+.1f}% 4V)", detail=detail)
    except Exception as exc:
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


# ── Pillar 2: TILLVÄXTPULS (Copper/Gold 3-month sparkline) ───────────────────

def _pillar_copper_gold() -> PillarResult:
    name = "TILLVÄXTPULS (Copper/Gold)"
    try:
        from alpha_regime.commodity_ratios import fetch_all_ratios
        ratios = fetch_all_ratios()
        r = ratios.get("copper_gold")
        if r is None or getattr(r, "status", "DATA_GAP") == "DATA_GAP":
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP",
                                detail="Copper/Gold ratio ej tillgänglig")
        vals = r.sparkline_values
        if len(vals) < 14:
            return PillarResult(name=name, status="DATA_GAP",
                                value=f"{r.current:.4f}",
                                detail="Otillräcklig sparkline-historik")
        n          = min(13, len(vals) - 1)
        change_pct = (vals[-1] - vals[-n]) / abs(vals[-n]) * 100 if vals[-n] != 0 else 0.0
        if change_pct > CG_FLAT_PCT:
            status = "GREEN"
            detail = "Copper/Gold stigande → industriell efterfrågan ✓"
        elif change_pct < -CG_FLAT_PCT:
            status = "RED"
            detail = "Copper/Gold fallande → tillväxtoro"
        else:
            status = "AMBER"
            detail = f"Copper/Gold sidostabil → neutral"
        return PillarResult(name=name, status=status,
                            value=f"{r.current:.4f} ({change_pct:+.1f}% 3M)", detail=detail)
    except Exception as exc:
        logger.debug("_pillar_copper_gold: %s", exc)
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


# ── Pillar 3: RÄNTEKURVA (T10Y2Y FRED 4-week change) ────────────────────────

def _pillar_yield_curve() -> PillarResult:
    name = "RÄNTEKURVA (T10Y2Y)"
    try:
        from ember.fred_cache import fetch_t10y2y_values
        vals_list, is_stale = fetch_t10y2y_values(FRED_T10Y2Y_URL)
        if len(vals_list) < 25:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP",
                                detail=f"T10Y2Y: för lite historik ({len(vals_list)} rader)")
        current   = float(vals_list[-1])
        change_4w = current - float(vals_list[-21])   # ~20 business days ≈ 4 weeks
        stale_sfx = " (cachad)" if is_stale else ""
        if change_4w > YC_STEEPEN_PP:
            status = "GREEN"
            detail = f"T10Y2Y brantnar {change_4w:+.2f}pp 4V → tillväxtmedvind ✓{stale_sfx}"
        elif change_4w < YC_INVERT_PP:
            status = "RED"
            detail = f"T10Y2Y inverterar djupare {change_4w:+.2f}pp 4V → recession-risk ⛔{stale_sfx}"
        else:
            status = "AMBER"
            detail = f"T10Y2Y flackar {change_4w:+.2f}pp 4V → neutral{stale_sfx}"
        return PillarResult(name=name, status=status,
                            value=f"{current:.2f}% ({change_4w:+.2f}pp 4V)", detail=detail)
    except Exception as exc:
        logger.debug("_pillar_yield_curve: %s", exc)
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=f"FRED: {exc}")


# ── Pillar 4: TEMA-BREDD (Theme Board TIDIG/MITTEN + 3m momentum) ────────────

def _pillar_tema_breadth() -> PillarResult:
    name = "TEMA-BREDD"
    try:
        from blindspot.theme_board import build_theme_board
        themes = build_theme_board()
        if not themes:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="Tema-tavlan ej tillgänglig")

        total    = len(themes)
        pos_count = 0
        for t in themes:
            if t.error:
                continue
            in_cycle = t.cykel_label in ("TIDIG", "MITTEN")
            if not in_cycle:
                continue
            # 3m momentum: compare sparkline week[-1] vs week[-13]
            if t.sparkline_values and len(t.sparkline_values) >= 14:
                n     = min(13, len(t.sparkline_values) - 1)
                v_now = t.sparkline_values[-1]
                v_3m  = t.sparkline_values[-n]
                if v_3m > 0 and v_now > v_3m:
                    pos_count += 1

        ratio_pct = (pos_count / total * 100) if total > 0 else 0.0

        if ratio_pct > TEMA_GREEN_PCT:
            status = "GREEN"
            detail = f"{pos_count}/{total} teman tidig/mitten + positiv 3M trend ✓"
        elif ratio_pct > TEMA_AMBER_PCT:
            status = "AMBER"
            detail = f"{pos_count}/{total} teman positiva — blandat cykelmönster"
        else:
            status = "RED"
            detail = f"Bara {pos_count}/{total} teman positiva — brett cykelnedtryck ⛔"

        return PillarResult(name=name, status=status,
                            value=f"{pos_count}/{total} ({ratio_pct:.0f}%)", detail=detail)
    except Exception as exc:
        logger.debug("_pillar_tema_breadth: %s", exc)
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


# ── Pillar 5: RISKAPTIT (GDX/SPY 3m + HYG/TLT 1m) ──────────────────────────

def _pillar_risk_appetite() -> PillarResult:
    name = "RISKAPTIT"
    try:
        # GDX/SPY 3-month relative performance (63 trading days)
        gdx = _close("GDX", "6mo")
        spy = _close("SPY", "6mo")
        gdx_spy_pos = False
        gdx_spy_str = "DATA_GAP"
        if len(gdx) >= 64 and len(spy) >= 64:
            gdx_3m = float(gdx.iloc[-1]) / float(gdx.iloc[-64]) - 1
            spy_3m = float(spy.iloc[-1]) / float(spy.iloc[-64]) - 1
            gdx_spy_pos = gdx_3m > spy_3m
            gdx_spy_str = f"GDX {gdx_3m*100:+.1f}% vs SPY {spy_3m*100:+.1f}% (3M)"

        # HYG/TLT 1-month relative performance (21 trading days)
        hyg = _close("HYG", "3mo")
        tlt = _close("TLT", "3mo")
        hyg_tlt_pos = False
        hyg_tlt_str = "DATA_GAP"
        if len(hyg) >= 22 and len(tlt) >= 22:
            hyg_1m = float(hyg.iloc[-1]) / float(hyg.iloc[-22]) - 1
            tlt_1m = float(tlt.iloc[-1]) / float(tlt.iloc[-22]) - 1
            hyg_tlt_pos = hyg_1m > tlt_1m
            hyg_tlt_str = f"HYG {hyg_1m*100:+.1f}% vs TLT {tlt_1m*100:+.1f}% (1M)"

        data_ok = gdx_spy_str != "DATA_GAP" or hyg_tlt_str != "DATA_GAP"
        if not data_ok:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP",
                                detail="GDX/SPY + HYG/TLT ej tillgängliga")

        if gdx_spy_pos and hyg_tlt_pos:
            status = "GREEN"
            detail = "Gruvbolag + kredit leder → stark riskaptit ✓"
        elif gdx_spy_pos or hyg_tlt_pos:
            status = "AMBER"
            detail = "Blandat — en signal positiv, en negativ"
        else:
            status = "RED"
            detail = "Gruvbolag + kredit underpresterar → svag riskaptit ⛔"

        parts = [s for s in [gdx_spy_str, hyg_tlt_str] if s != "DATA_GAP"]
        return PillarResult(name=name, status=status,
                            value=" · ".join(parts) if parts else "DATA_GAP",
                            detail=detail)
    except Exception as exc:
        logger.debug("_pillar_risk_appetite: %s", exc)
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


# ── Regime computation ────────────────────────────────────────────────────────

@_cache_1h
def compute_ember_regime() -> EmberRegimeResult:
    """Compute all five EMBER pillars and return the regime verdict. Cached 1h."""
    pillars = [
        _pillar_dxy(),
        _pillar_copper_gold(),
        _pillar_yield_curve(),
        _pillar_tema_breadth(),
        _pillar_risk_appetite(),
    ]

    # DATA_GAP treated as AMBER (not GREEN) for conservative verdict
    green_count = sum(1 for p in pillars if p.status == "GREEN")

    if green_count >= 4:
        verdict     = VERDICT_PA
        action_text = ("Full positionsstorlek tillåten. "
                       "Alla topp-rankade elitcase är giltiga.")
    elif green_count == 3:
        verdict     = VERDICT_SELEKTIV
        action_text = ("Halverad positionsstorlek. "
                       "Handla endast topp-1 och topp-2 i EMBER TOPP 3.")
    else:
        verdict     = VERDICT_AV
        action_text = ("Inga nya EMBER-trades. "
                       "Bevaka befintliga positioner och planera nästa setup.")

    return EmberRegimeResult(
        pillars=pillars,
        green_count=green_count,
        verdict=verdict,
        action_text=action_text,
    )


def get_cached_regime() -> Optional[EmberRegimeResult]:
    """Return the EmberRegimeResult stored in session_state, or None."""
    try:
        import streamlit as st
        return st.session_state.get("ember_regime")
    except Exception:
        return None


# ── ENERGI pillars ────────────────────────────────────────────────────────────

def _ep_dxy() -> PillarResult:
    return _pillar_dxy()


def _ep_xle_rs() -> PillarResult:
    return _p_rs_3m("XLE", "SPY", "XLE/SPY RELATIV STYRKA")


def _ep_oil_trend() -> PillarResult:
    return _p_50w_ema("CL=F", "OLJA (CL=F) vs 50V EMA")


def _ep_gas_trend() -> PillarResult:
    return _p_50w_ema("NG=F", "GAS (NG=F) vs 50V EMA")


def _ep_energy_breadth() -> PillarResult:
    name = "ENERGI-BREDD (XLE/XOP/OIH)"
    above, results = 0, []
    for etf in ("XLE", "XOP", "OIH"):
        try:
            c = _close(etf, "5y")
            cw = c.resample("W").last().dropna()
            if len(cw) >= 52:
                ok = float(cw.iloc[-1]) > float(cw.ewm(span=50, adjust=False).mean().iloc[-1])
                above += int(ok)
                results.append(f"{etf} {'✓' if ok else '✗'}")
            else:
                results.append(f"{etf} ?")
        except Exception:
            results.append(f"{etf} ?")
    val = f"{above}/3 ({', '.join(results)})"
    if above >= 2:
        return PillarResult(name=name, status="GREEN", value=val,
                            detail="≥2/3 energi-ETF:er ovan 50V EMA → bred upptrend ✓")
    if above == 1:
        return PillarResult(name=name, status="AMBER", value=val,
                            detail="1/3 energi-ETF:er ovan 50V EMA → selektiv")
    return PillarResult(name=name, status="RED", value=val,
                        detail="Inga energi-ETF:er ovan 50V EMA → undvik energi ⛔")


# ── ÄDELMETALLER pillars ──────────────────────────────────────────────────────

def _am_dxy() -> PillarResult:
    return _pillar_dxy()


def _am_gdx_rs() -> PillarResult:
    return _p_rs_3m("GDX", "SPY", "GDX/SPY RELATIV STYRKA")


def _am_gold_blowoff() -> PillarResult:
    name = "GULD BLOW-OFF GUARD"
    try:
        from alpha_regime.commodity_ratios import fetch_context_gauges
        g = fetch_context_gauges().get("gold_usd")
        if g is None:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="Guld-percentil ej tillgänglig")
        pct = g.percentile
        if pct > 95:
            return PillarResult(name=name, status="AMBER",
                                value=f"Percentil {pct:.0f}% (10å)",
                                detail=f"Guld i topp-5% ({pct:.0f}%) — blow-off risk, halvera position")
        detail = (f"Guld {pct:.0f}% percentil — ej blow-off ✓"
                  if pct > 80 else f"Guld {pct:.0f}% percentil — ej överköpt ✓")
        return PillarResult(name=name, status="GREEN",
                            value=f"Percentil {pct:.0f}% (10å)", detail=detail)
    except Exception as exc:
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


def _am_silver_trend() -> PillarResult:
    return _p_50w_ema("SLV", "SILVER (SLV) vs 50V EMA")


def _am_tip_trend() -> PillarResult:
    name = "REALRÄNTA (TIP 3M)"
    try:
        c = _close("TIP", "6mo")
        if len(c) < 64:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="TIP ej tillgänglig")
        chg = float(c.iloc[-1]) / float(c.iloc[-64]) - 1
        val = f"TIP {chg*100:+.1f}% 3M"
        if chg > 0.01:
            return PillarResult(name=name, status="GREEN", value=val,
                                detail=f"TIP +{chg*100:.1f}% 3M → realräntor fallande = guld-medvind ✓")
        if chg < -0.01:
            return PillarResult(name=name, status="RED", value=val,
                                detail=f"TIP {chg*100:.1f}% 3M → realräntor stigande = guld-motvind ⛔")
        return PillarResult(name=name, status="AMBER", value=val,
                            detail=f"TIP {chg*100:+.1f}% 3M → realräntor neutrala")
    except Exception as exc:
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


# ── BASMETALLER pillars ───────────────────────────────────────────────────────

def _bm_copper_gold() -> PillarResult:
    return _pillar_copper_gold()


def _bm_copx_rs() -> PillarResult:
    return _p_rs_3m("COPX", "SPY", "COPX/SPY RELATIV STYRKA")


def _bm_copper_trend() -> PillarResult:
    return _p_50w_ema("HG=F", "KOPPAR (HG=F) vs 50V EMA")


def _bm_china_demand() -> PillarResult:
    name = "KINA-EFTERFRÅGAN (MCHI/FXI)"
    try:
        mchi = _close("MCHI", "6mo")
        fxi  = _close("FXI",  "6mo")
        ticker = "MCHI" if len(mchi) >= 64 else ("FXI" if len(fxi) >= 64 else None)
        c = mchi if ticker == "MCHI" else (fxi if ticker == "FXI" else None)
        if c is None:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="MCHI/FXI ej tillgängliga")
        chg = float(c.iloc[-1]) / float(c.iloc[-64]) - 1
        val = f"{ticker} {chg*100:+.1f}% 3M"
        if chg > 0.03:
            return PillarResult(name=name, status="GREEN", value=val,
                                detail=f"{ticker} +{chg*100:.1f}% 3M → Kina-efterfrågan expanderar ✓")
        if chg < -0.03:
            return PillarResult(name=name, status="RED", value=val,
                                detail=f"{ticker} {chg*100:.1f}% 3M → Kina-osäkerhet tynger basmetaller ⛔")
        return PillarResult(name=name, status="AMBER", value=val,
                            detail=f"{ticker} {chg*100:+.1f}% 3M → neutral Kina-signal")
    except Exception as exc:
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


def _bm_yield_curve() -> PillarResult:
    return _pillar_yield_curve()


# ── AGRI & ÖVRIGT pillars ────────────────────────────────────────────────────

def _ag_dba_trend() -> PillarResult:
    name = "DBA vs 50V EMA + LUTNING"
    try:
        c = _close("DBA", "5y")
        if len(c) < 52:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="DBA ej tillgänglig")
        cw = c.resample("W").last().dropna()
        if len(cw) < 52:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="För lite veckodata")
        ema50w = cw.ewm(span=50, adjust=False).mean()
        ema20w = cw.ewm(span=20, adjust=False).mean()
        price  = float(cw.iloc[-1])
        e50    = float(ema50w.iloc[-1])
        e20_now = float(ema20w.iloc[-1])
        e20_4w  = float(ema20w.iloc[-5]) if len(ema20w) >= 5 else e20_now
        above, slope_up = price > e50, e20_now > e20_4w
        if above and slope_up:
            return PillarResult(name=name, status="GREEN",
                                value=f"{price:.2f} (50V EMA {e50:.2f})",
                                detail="DBA ovan 50V EMA och 20V EMA stiger → agri-upptrend ✓")
        if above:
            return PillarResult(name=name, status="AMBER",
                                value=f"{price:.2f} (50V EMA {e50:.2f})",
                                detail="DBA ovan 50V EMA men 20V EMA sjunker → avvakta")
        return PillarResult(name=name, status="RED",
                            value=f"{price:.2f} (50V EMA {e50:.2f})",
                            detail="DBA under 50V EMA → agri-nedtrend ⛔")
    except Exception as exc:
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


def _ag_dba_rs() -> PillarResult:
    return _p_rs_3m("DBA", "SPY", "DBA/SPY RELATIV STYRKA")


def _ag_dxy() -> PillarResult:
    return _pillar_dxy()


def _ag_agri_theme() -> PillarResult:
    name = "AGRI-TEMA CYKEL"
    try:
        from blindspot.theme_board import build_theme_board
        agri = next((t for t in build_theme_board() if t.key == "agri"), None)
        if agri is None:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="Agri-temat ej i Theme Board")
        lbl = agri.cykel_label
        if lbl == "DATA_GAP":
            return PillarResult(name=name, status="DATA_GAP",
                                value=f"Cykel: {lbl}", detail="Agri-tema cykeldata ej tillgänglig")
        if lbl in ("TIDIG", "MITTEN"):
            return PillarResult(name=name, status="GREEN",
                                value=f"Cykel: {lbl}",
                                detail=f"Agri-tema: {lbl} — rätt cykelfas för entries ✓")
        status = "AMBER" if lbl == "SEN" else "RED"
        return PillarResult(name=name, status=status,
                            value=f"Cykel: {lbl}",
                            detail=f"Agri-tema: {lbl} — avancerad cykel, öka selektivitet")
    except Exception as exc:
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


def _ag_dba_volume() -> PillarResult:
    name = "DBA VOLYM-TREND (3M)"
    try:
        import yfinance as yf
        try:
            df = yf.download("DBA", period="1y", auto_adjust=True,
                             progress=False, show_errors=False, multi_level_index=False)
        except TypeError:
            df = yf.download("DBA", period="1y", auto_adjust=True, progress=False)
        if df is None or df.empty or "Volume" not in df.columns:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="DBA volym ej tillgänglig")
        vol = df["Volume"].squeeze()
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        vol = pd.to_numeric(vol, errors="coerce").dropna()
        if len(vol) < 126:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP", detail="För lite volymhistorik")
        avg_3m = float(vol.iloc[-63:].mean())
        avg_6m = float(vol.iloc[-126:-63].mean())
        ratio  = avg_3m / avg_6m if avg_6m > 0 else 1.0
        val = f"Vol ratio 3M/prior3M: {ratio:.2f}x"
        if ratio > 1.05:
            return PillarResult(name=name, status="GREEN", value=val,
                                detail=f"DBA-volym {ratio:.2f}x — stigande intresse ✓")
        if ratio < 0.90:
            return PillarResult(name=name, status="RED", value=val,
                                detail=f"DBA-volym {ratio:.2f}x — minskande intresse ⛔")
        return PillarResult(name=name, status="AMBER", value=val,
                            detail=f"DBA-volym stabil {ratio:.2f}x → neutral")
    except Exception as exc:
        return PillarResult(name=name, status="DATA_GAP",
                            value="DATA_GAP", detail=str(exc))


# ── Complex compute helpers ───────────────────────────────────────────────────

def _complex_verdict(green_count: int, key: str) -> tuple[str, str]:
    lbl = {"energi": "ENERGI", "adelmetaller": "ÄDELMETALLER",
           "basmetaller": "BASMETALLER", "agri": "AGRI & ÖVRIGT"}.get(key, key.upper())
    if green_count >= 4:
        return (VERDICT_PA,
                f"Full positionsstorlek tillåten för {lbl}. Alla topp-rankade elitcase är giltiga.")
    if green_count == 3:
        return (VERDICT_SELEKTIV,
                f"Halverad positionsstorlek för {lbl}. Handla endast topp-1 och topp-2 i screener.")
    return (VERDICT_AV,
            f"Inga nya {lbl}-trades. Bevaka befintliga positioner och planera nästa setup.")


@_cache_1h
def compute_energi_regime() -> ComplexRegimeResult:
    pillars = [_ep_dxy(), _ep_xle_rs(), _ep_oil_trend(), _ep_gas_trend(), _ep_energy_breadth()]
    gc = sum(1 for p in pillars if p.status == "GREEN")
    v, a = _complex_verdict(gc, "energi")
    return ComplexRegimeResult(key="energi", label="ENERGI",
                               pillars=pillars, green_count=gc, verdict=v, action_text=a)


@_cache_1h
def compute_adelmetaller_regime() -> ComplexRegimeResult:
    pillars = [_am_dxy(), _am_gdx_rs(), _am_gold_blowoff(), _am_silver_trend(), _am_tip_trend()]
    gc = sum(1 for p in pillars if p.status == "GREEN")
    v, a = _complex_verdict(gc, "adelmetaller")
    return ComplexRegimeResult(key="adelmetaller", label="ÄDELMETALLER",
                               pillars=pillars, green_count=gc, verdict=v, action_text=a)


@_cache_1h
def compute_basmetaller_regime() -> ComplexRegimeResult:
    pillars = [_bm_copper_gold(), _bm_copx_rs(), _bm_copper_trend(),
               _bm_china_demand(), _bm_yield_curve()]
    gc = sum(1 for p in pillars if p.status == "GREEN")
    v, a = _complex_verdict(gc, "basmetaller")
    return ComplexRegimeResult(key="basmetaller", label="BASMETALLER",
                               pillars=pillars, green_count=gc, verdict=v, action_text=a)


@_cache_1h
def compute_agri_regime() -> ComplexRegimeResult:
    pillars = [_ag_dba_trend(), _ag_dba_rs(), _ag_dxy(), _ag_agri_theme(), _ag_dba_volume()]
    gc = sum(1 for p in pillars if p.status == "GREEN")
    v, a = _complex_verdict(gc, "agri")
    return ComplexRegimeResult(key="agri", label="AGRI & ÖVRIGT",
                               pillars=pillars, green_count=gc, verdict=v, action_text=a)


def compute_all_complex_regimes() -> dict[str, ComplexRegimeResult]:
    """Compute all 4 complex regimes. Caches result in session_state."""
    results: dict[str, ComplexRegimeResult] = {
        "energi":       compute_energi_regime(),
        "adelmetaller": compute_adelmetaller_regime(),
        "basmetaller":  compute_basmetaller_regime(),
        "agri":         compute_agri_regime(),
    }
    try:
        import streamlit as st
        st.session_state["ember_complex_regimes"] = results
    except Exception:
        pass
    return results


# ── Palette helpers for UI ────────────────────────────────────────────────────

_STATUS_COLOR = {
    "GREEN":    GREEN,
    "AMBER":    AMBER,
    "RED":      RED,
    "DATA_GAP": DIM,
}
_STATUS_LABEL_SV = {
    "GREEN":    "GRÖN ✓",
    "AMBER":    "AMBER ~",
    "RED":      "RÖD ⛔",
    "DATA_GAP": "DATA GAP",
}
_VERDICT_BG     = {VERDICT_PA: "#0f1f15", VERDICT_SELEKTIV: "#1a1509", VERDICT_AV: "#1a0909"}
_VERDICT_BORDER = {VERDICT_PA: GREEN,      VERDICT_SELEKTIV: AMBER,      VERDICT_AV: RED}
_VERDICT_ICON   = {VERDICT_PA: "🟢",       VERDICT_SELEKTIV: "🟡",       VERDICT_AV: "🔴"}


# ── Streamlit UI helpers ──────────────────────────────────────────────────────

_COMPLEX_ICON: dict[str, str] = {
    "energi": "⚡", "adelmetaller": "🥇", "basmetaller": "🔧", "agri": "🌾",
}


def _render_pillar_cards(pillars: list[PillarResult]) -> None:
    cols = st.columns(5)
    for col, pillar in zip(cols, pillars):
        sc  = _STATUS_COLOR.get(pillar.status, DIM)
        slv = _STATUS_LABEL_SV.get(pillar.status, pillar.status)
        with col:
            st.markdown(
                f"<div style='background:{BG2};border:1px solid {sc}55;"
                f"border-top:3px solid {sc};border-radius:8px;"
                f"padding:14px 12px;min-height:140px;'>"
                f"<div style='color:{DIM};font-size:0.6rem;text-transform:uppercase;"
                f"letter-spacing:0.07em;margin-bottom:6px;'>{pillar.name}</div>"
                f"<div style='color:{sc};font-size:1.0rem;font-weight:700;"
                f"margin-bottom:6px;'>{slv}</div>"
                f"<div style='color:{TEXT};font-size:0.72rem;margin-bottom:6px;"
                f"word-break:break-word;'>{pillar.value}</div>"
                f"<div style='color:{DIM};font-size:0.64rem;line-height:1.4;'>"
                f"{pillar.detail}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_complex_tab(result: ComplexRegimeResult) -> None:
    vbrd = _VERDICT_BORDER.get(result.verdict, DIM)
    vbg  = _VERDICT_BG.get(result.verdict, BG2)
    vico = _VERDICT_ICON.get(result.verdict, "○")
    icon = _COMPLEX_ICON.get(result.key, "🌍")

    st.markdown(
        f"<div style='background:{vbg};border:1px solid {vbrd};"
        f"border-left:5px solid {vbrd};border-radius:8px;"
        f"padding:20px 24px;margin-bottom:20px;'>"
        f"<div style='font-size:1.5rem;font-weight:700;color:{vbrd};"
        f"letter-spacing:0.1em;margin-bottom:6px;'>"
        f"{vico} {icon} {result.label}: {result.verdict}</div>"
        f"<div style='color:{TEXT};font-size:0.9rem;margin-bottom:6px;'>"
        f"{result.action_text}</div>"
        f"<div style='color:{DIM};font-size:0.7rem;'>"
        f"Gröna pelare: {result.green_count}/5 · "
        f"Uppdaterad: {result.timestamp.strftime('%H:%M:%S')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    _render_pillar_cards(result.pillars)


def _render_ticker_analysis(
    ticker: str,
    all_regimes: dict[str, ComplexRegimeResult],
) -> None:
    """Show complex regime + EMBER trend gate analysis for a specific ticker."""
    complex_key = detect_complex(ticker)
    result = all_regimes.get(complex_key)
    if result is None:
        st.warning(f"Ingen regimdata för komplex '{complex_key}'")
        return

    complex_label = {
        "energi": "⚡ ENERGI", "adelmetaller": "🥇 ÄDELMETALLER",
        "basmetaller": "🔧 BASMETALLER", "agri": "🌾 AGRI & ÖVRIGT",
    }.get(complex_key, complex_key.upper())

    vbrd = _VERDICT_BORDER.get(result.verdict, DIM)
    vico = _VERDICT_ICON.get(result.verdict, "○")

    st.markdown(
        f"<div style='background:{BG2};border:2px solid {EMBER}55;"
        f"border-radius:8px;padding:16px 20px;margin-bottom:16px;'>"
        f"<div style='color:{DIM};font-size:0.65rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin-bottom:6px;'>IDENTIFIERAT KOMPLEX FÖR {ticker}</div>"
        f"<div style='color:{EMBER};font-size:1.1rem;font-weight:700;'>{complex_label}</div>"
        f"<div style='color:{vbrd};font-size:0.85rem;font-weight:700;margin-top:4px;'>"
        f"{vico} Regim: {result.verdict}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    try:
        from ember.gates import _download_robust, compute_trend_gates
        from ember.config import EMBER_SECTOR_ETF, TICKER_THEME_MAP, DEFAULT_SECTOR_ETF

        with st.spinner(f"Hämtar trendgates för {ticker}…"):
            df_daily  = _download_robust(ticker, "2y")
            df_weekly = _download_robust(ticker, "5y")

        close_d: pd.Series = pd.Series(dtype=float)
        close_w: pd.Series = pd.Series(dtype=float)

        if not df_daily.empty and "Close" in df_daily.columns:
            cd = df_daily["Close"].squeeze()
            if isinstance(cd, pd.DataFrame):
                cd = cd.iloc[:, 0]
            close_d = cd.dropna()

        if not df_weekly.empty and "Close" in df_weekly.columns:
            cw = df_weekly["Close"].squeeze()
            if isinstance(cw, pd.DataFrame):
                cw = cw.iloc[:, 0]
            close_w = cw.dropna().resample("W").last().dropna()

        theme_key  = TICKER_THEME_MAP.get(ticker.upper())
        sector_etf = (EMBER_SECTOR_ETF.get(theme_key, DEFAULT_SECTOR_ETF)
                      if theme_key else DEFAULT_SECTOR_ETF)

        gates = compute_trend_gates(close_d, close_w, sector_etf) if len(close_d) >= 60 else []
        all_pass = all(g.passed for g in gates if g.is_blocker)

        if gates:
            st.markdown(
                f"<div style='color:{DIM};font-size:0.63rem;text-transform:uppercase;"
                f"letter-spacing:0.1em;margin:12px 0 8px;'>TREND GATES — {ticker}</div>",
                unsafe_allow_html=True,
            )
            gcols = st.columns(len(gates))
            for gcol, g in zip(gcols, gates):
                gc = GREEN if g.passed else (RED if g.is_blocker else AMBER)
                gi = "✓" if g.passed else "✗"
                with gcol:
                    st.markdown(
                        f"<div style='background:{BG2};border:1px solid {gc}55;"
                        f"border-top:3px solid {gc};border-radius:6px;padding:10px 12px;'>"
                        f"<div style='color:{DIM};font-size:0.6rem;text-transform:uppercase;"
                        f"letter-spacing:0.06em;margin-bottom:4px;'>{g.name}</div>"
                        f"<div style='color:{gc};font-size:0.9rem;font-weight:700;"
                        f"margin-bottom:4px;'>{gi}</div>"
                        f"<div style='color:{TEXT};font-size:0.65rem;word-break:break-word;'>"
                        f"{g.detail}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info(f"Kunde inte beräkna trendgates för {ticker} — otillräcklig historik")

        # Swedish action box
        if result.verdict == VERDICT_AV:
            ac, ai = RED,   "⛔"
            at = f"NEJ — regimen för {result.label} är AV. Inga nya entries."
        elif result.verdict == VERDICT_SELEKTIV and all_pass:
            ac, ai = AMBER, "🟡"
            at = f"SELEKTIV — regim SELEKTIV och trendfilter gröna. Halverad position."
        elif result.verdict == VERDICT_SELEKTIV:
            ac, ai = AMBER, "🟡"
            at = f"AVVAKTA — regim SELEKTIV och trendfilter ej klara."
        elif all_pass:
            ac, ai = GREEN, "🟢"
            at = f"KÖP-LÄGE — regim PÅ och trendfilter gröna."
        else:
            ac, ai = AMBER, "🟡"
            at = f"AVVAKTA — regim PÅ men trendfilter ej klara."

        st.markdown(
            f"<div style='background:{BG2};border-left:5px solid {ac};"
            f"border:1px solid {ac}55;border-radius:8px;"
            f"padding:16px 20px;margin-top:16px;'>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{ac};"
            f"letter-spacing:0.08em;'>{ai} {at}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.warning(f"Trendgate-analys misslyckades: {exc}")

    # ── Discipline fields (shared session_state keys with signal cards) ───────
    st.markdown("<br>", unsafe_allow_html=True)
    fd1, fd2 = st.columns(2)
    with fd1:
        inv_text = st.text_input(
            "Ogiltigförklaras om",
            value=st.session_state.get(f"ember_inv_{ticker}", ""),
            key=f"ember_regime_inv_input_{ticker}",
            placeholder="ex: stänger under 50D EMA",
        )
        st.session_state[f"ember_inv_{ticker}"] = inv_text
    with fd2:
        cat_text = st.text_input(
            "Trolig trigger",
            value=st.session_state.get(f"ember_cat_{ticker}", ""),
            key=f"ember_regime_cat_input_{ticker}",
            placeholder="ex: FED pivot, Kina-stimulus",
        )
        st.session_state[f"ember_cat_{ticker}"] = cat_text

    # Numeric invalidation price + INVALIDATED badge
    inv_price_col, badge_col = st.columns([2, 1])
    with inv_price_col:
        inv_price = st.number_input(
            "Invalideringspris (0 = ej satt)",
            min_value=0.0, value=float(st.session_state.get(f"ember_inv_price_{ticker}", 0.0)),
            step=0.01, format="%.2f",
            key=f"ember_regime_inv_price_{ticker}",
        )
        st.session_state[f"ember_inv_price_{ticker}"] = inv_price

    # Fetch current price for badge check
    try:
        from ember.gates import _download_robust as _dr
        _px_df = _dr(ticker, "5d")
        _cur_price: Optional[float] = None
        if not _px_df.empty and "Close" in _px_df.columns:
            _px = _px_df["Close"].squeeze()
            if hasattr(_px, "iloc"):
                _cur_price = float(_px.dropna().iloc[-1])
    except Exception:
        _cur_price = None

    with badge_col:
        if inv_price > 0 and _cur_price is not None and _cur_price <= inv_price:
            st.markdown(
                f"<div style='background:#2d0a0a;border:2px solid {RED};"
                f"border-radius:6px;padding:10px 14px;margin-top:22px;"
                f"text-align:center;'>"
                f"<div style='color:{RED};font-size:0.9rem;font-weight:700;"
                f"letter-spacing:0.08em;'>⛔ INVALIDERAD</div>"
                f"<div style='color:{DIM};font-size:0.62rem;'>"
                f"Pris {_cur_price:.2f} ≤ {inv_price:.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif inv_price > 0 and _cur_price is not None:
            st.markdown(
                f"<div style='background:{BG2};border:1px solid {GREEN}44;"
                f"border-radius:6px;padding:10px 14px;margin-top:22px;"
                f"text-align:center;'>"
                f"<div style='color:{GREEN};font-size:0.85rem;font-weight:700;'>✓ AKTIV</div>"
                f"<div style='color:{DIM};font-size:0.62rem;'>"
                f"Pris {_cur_price:.2f} &gt; {inv_price:.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Case-context block ────────────────────────────────────────────────────
    st.markdown(
        f"<div style='color:{DIM};font-size:0.63rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:16px 0 8px;border-top:1px solid {EMBER}22;"
        f"padding-top:12px;'>CASE-KONTEXT — {ticker}</div>",
        unsafe_allow_html=True,
    )

    ctx_col1, ctx_col2 = st.columns(2)

    # Left: theme phase + HAT
    with ctx_col1:
        try:
            from blindspot.theme_board import build_theme_board
            from ember.config import TICKER_THEME_MAP as _TTM
            _theme_key = _TTM.get(ticker.upper())
            _theme_data = None
            if _theme_key:
                for _tr in build_theme_board():
                    if _tr.key == _theme_key:
                        _theme_data = _tr
                        break

            if _theme_data:
                _cykel = _theme_data.cykel_label or "DATA_GAP"
                _pct   = _theme_data.percentile_10y
                _hat   = _theme_data.hat_score
                _pct_s = f" · {_pct:.0f}:e percentilen 10å" if _pct else ""
                _hat_s = f"{_hat:.0f}/100 — marknaden ignorerar" if _hat else "—"
                _cykel_c = {
                    "TIDIG": GREEN, "MITTEN": GOLD, "SEN": AMBER, "TOPP": RED,
                }.get(_cykel, DIM)
                st.markdown(
                    f"<div style='background:{BG2};border:1px solid {EMBER}33;"
                    f"border-radius:6px;padding:12px 14px;'>"
                    f"<div style='color:{DIM};font-size:0.6rem;text-transform:uppercase;"
                    f"letter-spacing:0.07em;margin-bottom:6px;'>CYKELFAS</div>"
                    f"<div style='color:{_cykel_c};font-size:1.0rem;font-weight:700;"
                    f"margin-bottom:4px;'>{_cykel}{_pct_s}</div>"
                    f"<div style='color:{DIM};font-size:0.6rem;text-transform:uppercase;"
                    f"letter-spacing:0.07em;margin:8px 0 4px;'>MARKNADEN OGILLAR (HAT)</div>"
                    f"<div style='color:{AMBER};font-size:0.8rem;'>{_hat_s}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='color:{DIM};font-size:0.75rem;'>"
                    f"Ingen temadata för {ticker} — lägg till i TICKER_THEME_MAP"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.markdown(f"<div style='color:{DIM};font-size:0.7rem;'>Temadata: {exc}</div>",
                        unsafe_allow_html=True)

    # Right: macro score breakdown
    with ctx_col2:
        try:
            from ember.scoring import compute_macro_score
            from alpha_regime.commodity_ratios import fetch_all_ratios
            from ember.config import TICKER_THEME_MAP as _TTM2
            from blindspot.theme_board import build_theme_board as _btb2
            _theme_key2 = _TTM2.get(ticker.upper())
            # Resolve cykel_label for this ticker's theme
            _cykel_label2: Optional[str] = None
            if _theme_key2:
                try:
                    for _tr2 in _btb2():
                        if _tr2.key == _theme_key2:
                            _cykel_label2 = _tr2.cykel_label
                            break
                except Exception:
                    pass
            _ratios: Optional[dict] = None
            try:
                _ratios = fetch_all_ratios()
            except Exception:
                pass
            _macro = compute_macro_score(_ratios, _cykel_label2)
            st.markdown(
                f"<div style='background:{BG2};border:1px solid {EMBER}33;"
                f"border-radius:6px;padding:12px 14px;'>"
                f"<div style='color:{DIM};font-size:0.6rem;text-transform:uppercase;"
                f"letter-spacing:0.07em;margin-bottom:8px;'>"
                f"MAKROSCORE — {_macro.total:.0f}/100</div>"
                + "".join(
                    f"<div style='margin-bottom:5px;'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:0.7rem;'>"
                    f"<span style='color:{TEXT};'>{lbl}</span>"
                    f"<span style='color:{GREEN if sc >= 50 else RED};font-weight:700;'>{sc:.0f}</span>"
                    f"</div><div style='color:{DIM};font-size:0.6rem;'>{det}</div></div>"
                    for lbl, sc, det in [
                        ("Copper/Gold", _macro.copper_gold, _macro.copper_gold_detail),
                        ("DXY",         _macro.dxy,         _macro.dxy_detail),
                        ("Räntekurva",  _macro.yield_curve, _macro.yield_curve_detail),
                        ("Cykelfas",    _macro.cycle,       _macro.cycle_detail),
                    ]
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.markdown(f"<div style='color:{DIM};font-size:0.7rem;'>Makroscore: {exc}</div>",
                        unsafe_allow_html=True)


# ── Streamlit page (v2) ────────────────────────────────────────────────────────

def render_ember_regime_page() -> None:
    """EMBER Regime v2 — per-complex regimes + ticker search."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        f"<div style='text-align:center;padding:14px 0 10px 0;'>"
        f"<h2 style='color:{EMBER};letter-spacing:0.12em;margin:0;'>🌍 EMBER REGIME</h2>"
        f"<p style='color:{DIM};font-size:0.75rem;letter-spacing:0.08em;margin:4px 0 0;'>"
        f"Per-komplex miljögauge · Energi · Ädelmetaller · Basmetaller · Agri · Cachad 1h"
        f"</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{EMBER}33;margin:0 0 16px 0;'>",
                unsafe_allow_html=True)

    # Ticker search bar
    tc1, tc2 = st.columns([3, 1])
    with tc1:
        search_ticker = st.text_input(
            "TICKER SÖKNING", placeholder="T.ex. XOM, GDX, FCX, DBA…",
            key="ember_regime_ticker_search",
        ).strip().upper()
    with tc2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        analyse_btn = st.button("🔍 ANALYSERA", key="ember_regime_analyse", width="stretch")

    col_r, _ = st.columns([1, 3])
    with col_r:
        if st.button("↺ Tvinga uppdatering", key="ember_regime_refresh"):
            for fn in (compute_energi_regime, compute_adelmetaller_regime,
                       compute_basmetaller_regime, compute_agri_regime):
                if hasattr(fn, "clear"):
                    fn.clear()
            st.rerun()

    with st.spinner("Beräknar EMBER per-komplex regimer…"):
        try:
            all_regimes = compute_all_complex_regimes()
        except Exception as exc:
            st.error(f"Kunde inte beräkna EMBER Regime: {exc}")
            return

    # Backward-compat: expose first complex result under the legacy key
    first = next(iter(all_regimes.values()), None)
    if first:
        st.session_state["ember_regime"] = first

    # Ticker analysis panel — show when ticker is entered (button re-runs anyway)
    if search_ticker:
        st.markdown("---")
        _render_ticker_analysis(search_ticker, all_regimes)
        st.markdown("---")

    # 4 complex tabs
    tab_e, tab_a, tab_b, tab_g = st.tabs([
        "⚡ ENERGI", "🥇 ÄDELMETALLER", "🔧 BASMETALLER", "🌾 AGRI & ÖVRIGT",
    ])
    with tab_e:
        _render_complex_tab(all_regimes["energi"])
    with tab_a:
        _render_complex_tab(all_regimes["adelmetaller"])
    with tab_b:
        _render_complex_tab(all_regimes["basmetaller"])
    with tab_g:
        _render_complex_tab(all_regimes["agri"])

    with st.expander("ℹ Hur per-komplex regimmodellen fungerar", expanded=False):
        st.markdown(
            f"<div style='color:{TEXT};font-size:0.83rem;line-height:1.65;'>"
            f"<b style='color:{EMBER};'>4 komplex — varje med 5 pelare — "
            f"≥4 gröna = PÅ, 3 = SELEKTIV, ≤2 = AV</b><br/><br/>"
            f"<b style='color:{GREEN};'>⚡ ENERGI</b> — DXY 4V · XLE/SPY RS 3M · "
            f"CL=F 50V EMA · NG=F 50V EMA · Bredd XLE/XOP/OIH<br/>"
            f"<b style='color:{GOLD};'>🥇 ÄDELMETALLER</b> — DXY 4V · GDX/SPY RS 3M · "
            f"Guld blow-off guard · SLV 50V EMA · TIP 3M (realränta)<br/>"
            f"<b style='color:{AMBER};'>🔧 BASMETALLER</b> — Copper/Gold 3M · COPX/SPY RS 3M · "
            f"HG=F 50V EMA · MCHI/FXI 3M (Kina) · T10Y2Y<br/>"
            f"<b style='color:{GREEN};'>🌾 AGRI & ÖVRIGT</b> — DBA 50V EMA+lutning · "
            f"DBA/SPY RS 3M · DXY 4V · Agri-tema cykel · DBA volym 3M<br/><br/>"
            f"Ticker-sökning identifierar automatiskt rätt komplex och visar "
            f"regimen + EMBER trend-gates."
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='color:{DIM};font-size:0.69rem;margin-top:8px;'>"
        f"Regimstatus per komplex används av EMBER-screener: "
        f"halverad position vid SELEKTIV, varningsbanner vid AV."
        f"</div>",
        unsafe_allow_html=True,
    )
