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

import io
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
        import requests
        resp = requests.get(FRED_T10Y2Y_URL, timeout=FRED_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), skiprows=1, names=["date", "value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if len(df) < 25:
            return PillarResult(name=name, status="DATA_GAP",
                                value="DATA_GAP",
                                detail=f"T10Y2Y: för lite historik ({len(df)} rader)")
        vals      = df["value"].values
        current   = float(vals[-1])
        change_4w = current - float(vals[-21])   # ~20 business days ≈ 4 weeks
        if change_4w > YC_STEEPEN_PP:
            status = "GREEN"
            detail = f"T10Y2Y brantnar {change_4w:+.2f}pp 4V → tillväxtmedvind ✓"
        elif change_4w < YC_INVERT_PP:
            status = "RED"
            detail = f"T10Y2Y inverterar djupare {change_4w:+.2f}pp 4V → recession-risk ⛔"
        else:
            status = "AMBER"
            detail = f"T10Y2Y flackar {change_4w:+.2f}pp 4V → neutral"
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


# ── Streamlit page ────────────────────────────────────────────────────────────

def render_ember_regime_page() -> None:
    """Render the EMBER Regime master environment gauge page."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        f"<div style='text-align:center;padding:14px 0 10px 0;'>"
        f"<h2 style='color:{EMBER};letter-spacing:0.12em;margin:0;'>🌍 EMBER REGIME</h2>"
        f"<p style='color:{DIM};font-size:0.75rem;letter-spacing:0.08em;margin:4px 0 0;'>"
        f"Miljögauge för råvaru-swings · Fem pelare · Cachad 1h"
        f"</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<hr style='border-color:{EMBER}33;margin:0 0 16px 0;'>",
        unsafe_allow_html=True,
    )

    if st.button("↺ Tvinga uppdatering", key="ember_regime_refresh"):
        if hasattr(compute_ember_regime, "clear"):
            compute_ember_regime.clear()
        st.rerun()

    with st.spinner("Beräknar EMBER Regime-pelare…"):
        try:
            result = compute_ember_regime()
        except Exception as exc:
            st.error(f"Kunde inte beräkna EMBER Regime: {exc}")
            return

    # Persist for the EMBER screener
    st.session_state["ember_regime"] = result

    # ── 5 pillar cards ────────────────────────────────────────────────────────
    cols = st.columns(5)
    for col, pillar in zip(cols, result.pillars):
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

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Verdict box ───────────────────────────────────────────────────────────
    vbg  = _VERDICT_BG.get(result.verdict, BG2)
    vbrd = _VERDICT_BORDER.get(result.verdict, DIM)
    vico = _VERDICT_ICON.get(result.verdict, "○")

    st.markdown(
        f"<div style='background:{vbg};border:1px solid {vbrd};"
        f"border-left:5px solid {vbrd};border-radius:8px;padding:20px 24px;'>"
        f"<div style='font-size:1.3rem;font-weight:700;color:{vbrd};"
        f"letter-spacing:0.1em;margin-bottom:8px;'>"
        f"{vico} EMBER REGIME: {result.verdict}</div>"
        f"<div style='color:{TEXT};font-size:0.9rem;margin-bottom:8px;'>"
        f"{result.action_text}</div>"
        f"<div style='color:{DIM};font-size:0.7rem;'>"
        f"Gröna pelare: {result.green_count}/5 · "
        f"Uppdaterad: {result.timestamp.strftime('%H:%M:%S')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pillar model explanation ──────────────────────────────────────────────
    with st.expander("ℹ Hur pelarmodellen fungerar", expanded=False):
        st.markdown(
            f"<div style='color:{TEXT};font-size:0.83rem;line-height:1.65;'>"
            f"<b style='color:{GOLD};'>5 pelare — ≥4 gröna = PÅ, 3 = SELEKTIV, ≤2 = AV</b>"
            f"<br/><br/>"
            f"<b style='color:{GREEN};'>DOLLAR (DXY)</b> — 4V (20 handelsdagar) förändring. "
            f"Fallande DXY = valutor stärks mot USD = råvaror billigare för globala köpare. "
            f"RED om DXY stigit &gt;{DXY_SURGE_REGIME}% på 4V.<br/>"
            f"<b style='color:{GREEN};'>TILLVÄXTPULS (Copper/Gold)</b> — Copper/Gold-ratio 3M "
            f"trend via veckosparklinen. Stigande = industriell efterfrågan expanderar = "
            f"makromedvind för bas- och ädelmetaller.<br/>"
            f"<b style='color:{GREEN};'>RÄNTEKURVA (T10Y2Y)</b> — FRED 10Y minus 2Y Treasury-spread, "
            f"4V förändring i procentenheter. Brantnar ({YC_STEEPEN_PP:+.2f}pp) = expansivt klimat. "
            f"RED om kurvan inverterar djupare (&lt;{YC_INVERT_PP:.2f}pp).<br/>"
            f"<b style='color:{GREEN};'>TEMA-BREDD</b> — Andel av de 9 råvarukategorierna som är "
            f"TIDIG/MITTEN i cykeln OCH har positiv 3M sparkline-trend. "
            f"Bredden bekräftar att cykeln är driven av fundamenta, inte ett enskilt tema.<br/>"
            f"<b style='color:{GREEN};'>RISKAPTIT</b> — GDX/SPY 3M (gruvbolag vs breda börsen) + "
            f"HYG/TLT 1M (HY-kredit vs lång statsobligation). Båda positiva = "
            f"investerarna tar risk aktivt.<br/><br/>"
            f"<b style='color:{AMBER};'>DATA_GAP</b> räknas alltid som AMBER — "
            f"ett okänt pelare ger aldrig grön signal."
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='color:{DIM};font-size:0.69rem;margin-top:8px;'>"
        f"Regimstatus ({result.verdict}) används automatiskt av EMBER-screener: "
        f"halverad positionsstorlek vid SELEKTIV, varningsbanner per setup-kort vid AV."
        f"</div>",
        unsafe_allow_html=True,
    )
