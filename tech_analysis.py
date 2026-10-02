"""
tech_analysis.py — INTELLIGENCE → 🕯️ Teknisk analys.

En sökbar teknisk översikt för valfri ticker, byggd på samma indikatorer
och grafer som Viking Regime (ovtlyr/indicators, ovtlyr/ui/charts) — men
fristående, utan strategibeslut:

  pris + EMA 10/20/50/200 + order blocks + volym
  order blocks: aktiva zoner, närmaste bullish/bearish, var kursen står
  entry patterns (bullish) och exit warnings (bearish) med när de kom
  risk (volatilitet) och momentum
  volatilitetsfördelning · oscillatorns riktning · Bull List %

Data: Yahoo via den delade priscachen, Börsdata som reserv.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from ui.components import note, page_header
from ui.tokens import AMBER, BG_CARD, CYAN, DIM, GREEN, RED, TEXT

PERIODS = {"6 mån": "6mo", "1 år": "1y", "2 år": "2y"}
BULL_LIST_ETFS = ("XLE", "XLF", "XLK", "XLV", "XLI", "XLB", "XLC", "XLY", "XLP", "XLU", "XLRE")
PATTERN_LOOKBACK = 5


# ── Data ────────────────────────────────────────────────────────────────────
def _bd_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    try:
        from borsdata_api import BorsdataAPI
        api = BorsdataAPI()
        if not api.is_configured:
            return None
        df = api.get_stockprices_df(ticker, max_count=520)
        return None if df is None or df.empty else df
    except Exception:
        return None


def load_ohlcv(ticker: str, period: str, getter=None, bd=None) -> tuple:
    """(DataFrame med Date-kolumn | None, källa). Yahoo först, Börsdata som reserv."""
    if getter is None:
        from market_prices import ohlcv as getter
    df, src = None, ""
    try:
        df = getter(ticker, period)
        src = f"Yahoo Finance {ticker}"
    except Exception:
        df = None
    if df is None or len(df) < 30:
        df = (bd or _bd_ohlcv)(ticker)
        src = f"Börsdata {ticker}" if df is not None else ""
    if df is None or len(df) < 30:
        return None, ""
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    return out, src


def _indexed(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("Date")


# ── Mönstren ────────────────────────────────────────────────────────────────
def grouped_patterns(patterns: list, dates: pd.Series) -> list:
    """Samma mönster flera dagar i rad visas en gång: [(mönster, antal, dagar sedan senast, datum)]."""
    groups: dict = {}
    for p in patterns or []:
        g = groups.setdefault(p.name, [p, 0, p.bar_index])
        g[1] += 1
        if p.bar_index < g[2]:
            g[0], g[2] = p, p.bar_index
    out = []
    for p, n, ago in groups.values():
        d = dates.iloc[-1 - ago] if 0 <= ago < len(dates) else None
        out.append((p, n, ago, str(pd.Timestamp(d).date()) if d is not None else ""))
    rank = {"Strong": 0, "Moderate": 1, "Weak": 2}
    return sorted(out, key=lambda x: (x[2], rank.get(x[0].confidence, 3)))


def _when(ago: int) -> str:
    return "i dag" if ago == 0 else "i går" if ago == 1 else f"för {ago} dagar sedan"


def patterns_html(title: str, color: str, groups: list, empty: str) -> str:
    rows = "".join(
        f"<div style='display:flex;justify-content:space-between;margin:4px 0;'>"
        f"<span style='color:{TEXT};font-size:0.85rem;'>{p.visual} {p.name}"
        + (f" <span style='color:{DIM};font-size:0.72rem;'>×{n}</span>" if n > 1 else "")
        + f"</span><span style='color:{color};font-size:0.75rem;font-weight:600;'>{p.confidence}</span></div>"
        f"<div style='color:{DIM};font-size:0.68rem;margin-bottom:6px;'>{p.description} · {_when(ago)} ({d})</div>"
        for p, n, ago, d in groups) or f"<div style='color:{DIM};font-size:0.8rem;padding:6px 0;'>{empty}</div>"
    return (f"<div style='background:{BG_CARD};border:2px solid {color};border-radius:8px;padding:12px 14px;"
            f"margin:6px 0;'><div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
            f"<span style='color:{color};font-weight:700;letter-spacing:0.08em;'>{title}</span>"
            f"<span style='color:{DIM};font-size:0.72rem;'>{len(groups)} st · senaste {PATTERN_LOOKBACK} dagarna</span>"
            f"</div>{rows}</div>")


# ── Order blocks ────────────────────────────────────────────────────────────
def orderblock_rows(obs: list, price: float) -> list:
    """Aktiva block sorterade på avstånd: [(typ, låg, hög, avstånd %, datum, styrka, inne?)]."""
    rows = []
    for ob in obs or []:
        if getattr(ob, "status", "") != "Active":
            continue
        mid = (ob.high + ob.low) / 2
        inside = ob.low <= price <= ob.high
        rows.append((ob.type, ob.low, ob.high, 0.0 if inside else (mid / price - 1) * 100, ob.date,
                     ob.vol_strength, inside))
    return sorted(rows, key=lambda r: abs(r[3]))


def _ob_html(rows: list, oa: dict, price: float) -> str:
    bias = str(oa.get("signal_bias", "HOLD")).upper()
    bc = GREEN if bias == "BUY" else RED if bias in ("SELL", "REDUCE") else AMBER
    near = []
    if oa.get("approaching_bullish"):
        near.append(f"<span style='color:{GREEN};'>nära ett bullish block (stöd)</span>")
    if oa.get("approaching_bearish"):
        near.append(f"<span style='color:{RED};'>nära ett bearish block (motstånd)</span>")
    body = "".join(
        f"<tr><td style='text-align:left;color:{GREEN if t == 'bullish' else RED};'>"
        f"{'▲ bullish (stöd)' if t == 'bullish' else '▼ bearish (motstånd)'}</td>"
        f"<td>{lo:,.2f}–{hi:,.2f}</td><td>{'inne i blocket' if ins else f'{d:+.1f} %'}</td>"
        f"<td>{str(dt)[:10]}</td><td>{vs:.1f}×</td></tr>" for t, lo, hi, d, dt, vs, ins in rows[:10])
    return (f"<div style='font-size:0.85rem;color:{TEXT};margin:4px 0 6px;'>Kurs <b>{price:,.2f}</b> · bias "
            f"<b style='color:{bc};'>{bias}</b>" + (f" · {' · '.join(near)}" if near else "") + "</div>"
            + (f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
               f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'>Block</th><th>Zon</th>"
               f"<th>Avstånd</th><th>Bildat</th><th>Volym</th></tr>{body}</table></div>"
               if rows else f"<div style='color:{DIM};font-size:0.8rem;'>Inga aktiva order blocks.</div>"))


# ── Sidan ───────────────────────────────────────────────────────────────────
def _quick_picks() -> list:
    picks = []
    data = st.session_state.get("vn_screen_rows") or {}
    picks += [r["ticker"] for r in (data.get("rows") or []) if r.get("nine") is not None][:15]
    try:
        import positions
        picks += [r["ticker"] for r in positions.all_positions() if r.get("ticker")][:15]
    except Exception:
        pass
    return list(dict.fromkeys(picks))


def _section(title: str, sub: str = "") -> None:
    st.markdown(f"<div style='color:{CYAN};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"margin:16px 0 6px;'>{title}" + (f" <span style='color:{DIM};text-transform:none;letter-spacing:0;'>"
                                                 f"— {sub}</span>" if sub else "") + "</div>", unsafe_allow_html=True)


def render_tech_analysis_page() -> None:
    page_header("🕯️ Teknisk analys", "Pris, order blocks, entry- och exitmönster, risk, momentum och "
                                      "marknadsbredd för valfri ticker. Underlag — inga köpbeslut.")
    st.session_state.setdefault("ta_ticker", "SPY")
    picks = _quick_picks()
    c1, c2, c3 = st.columns([2, 2, 1])
    if picks:
        pick = c2.selectbox("Snabbval (Viking Nine-skanningen och dina innehav)", ["—"] + picks, key="ta_pick")
        if pick != st.session_state.get("ta_last_pick"):
            st.session_state["ta_last_pick"] = pick
            if pick != "—":
                st.session_state["ta_ticker"] = pick
    ticker = (c1.text_input("🔎 Ticker", key="ta_ticker", help="Yahoo-format: NVDA, VOLV-B.ST, EQNR.OL, ^OMX")
              or "").strip().upper()
    period = c3.selectbox("Period", list(PERIODS), index=1, key="ta_period")
    if not ticker:
        return
    df, src = load_ohlcv(ticker, PERIODS[period])
    if df is None:
        note(f"DATA UNAVAILABLE — ingen kurshistorik för {ticker} (Yahoo och Börsdata).")
        return
    _render(ticker, df, src)


def _render(ticker: str, df: pd.DataFrame, src: str) -> None:
    from ovtlyr.indicators.candlesticks import detect_patterns
    from ovtlyr.indicators.momentum import compute_momentum
    from ovtlyr.indicators.orderblocks import classify_price_vs_ob, detect_orderblocks
    from ovtlyr.indicators.trend import compute_trend
    from ovtlyr.indicators.volatility import compute_volatility
    from ovtlyr.indicators.volume import compute_volume
    from ovtlyr.ui import charts

    price = float(df["Close"].iloc[-1])

    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default
    trend = _safe(lambda: compute_trend(df), {})
    volume = _safe(lambda: compute_volume(df), {})
    vol = _safe(lambda: compute_volatility(df), {})
    mom = _safe(lambda: compute_momentum(df), {})
    obs = _safe(lambda: detect_orderblocks(df), [])
    oa = _safe(lambda: classify_price_vs_ob(price, obs), {"signal_bias": "HOLD"}) if obs else {"signal_bias": "HOLD"}
    pats = _safe(lambda: detect_patterns(df, lookback=PATTERN_LOOKBACK), {"bullish": [], "bearish": []})

    chg = (price / float(df["Close"].iloc[-2]) - 1) * 100 if len(df) > 1 else 0.0
    st.markdown(f"<div style='font-size:1.05rem;color:{TEXT};'><b>{ticker}</b> {price:,.2f} "
                f"<span style='color:{GREEN if chg >= 0 else RED};'>{chg:+.2f} %</span> "
                f"<span style='color:{DIM};font-size:0.75rem;'>· {str(df['Date'].iloc[-1].date())} · {src}</span>"
                f"</div>", unsafe_allow_html=True)

    _section("📈 PRIS · EMA · ORDER BLOCKS")
    fig = _safe(lambda: charts.build_price_chart(df.copy(), trend, obs, volume), None)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="ta_price")

    _section("🧱 ORDER BLOCKS", "gröna = stöd där köpare tog över · röda = motstånd där säljare tog över")
    st.markdown(_ob_html(orderblock_rows(obs, price), oa, price), unsafe_allow_html=True)

    _section("🕯️ MÖNSTER")
    l, r = st.columns(2)
    l.markdown(patterns_html("ENTRY PATTERNS", GREEN, grouped_patterns(pats.get("bullish"), df["Date"]),
                             "Inga bullish mönster just nu"), unsafe_allow_html=True)
    r.markdown(patterns_html("EXIT WARNINGS", RED, grouped_patterns(pats.get("bearish"), df["Date"]),
                             "Inga bearish varningar just nu"), unsafe_allow_html=True)
    note("Ett mönster som syns flera dagar i rad visas en gång (×antal) med senaste dagen. Mönster är "
         "bekräftelser i rätt läge — vid stöd/motstånd och med volym — inte signaler på egen hand.")

    _section("⚖️ RISK · MOMENTUM")
    a, b = st.columns(2)
    fig = _safe(lambda: charts.build_risk_gauge(int(vol.get("risk_score", 50))), None)
    if fig is not None:
        a.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="ta_risk")
    rsi = float(mom.get("rsi", 50) or 50)
    b.markdown(f"<div style='background:{BG_CARD};border-left:3px solid {CYAN};border-radius:6px;padding:10px 12px;"
               f"font-size:0.85rem;color:{TEXT};line-height:1.8;'>RSI 14 <b style='float:right;'>{rsi:.1f}</b><br>"
               f"State <b style='float:right;color:{CYAN};'>{str(mom.get('ob_os_flag', 'neutral')).upper()}</b><br>"
               f"Vol ratio <b style='float:right;'>{float(volume.get('ratio', 1.0) or 1.0):.2f}x</b><br>"
               f"ATR 14 <b style='float:right;'>{float(vol.get('atr', 0) or 0):.2f}</b><br>"
               f"Trend <b style='float:right;'>{str(trend.get('trend_state', '—'))}</b></div>",
               unsafe_allow_html=True)

    _section("🔬 AVANCERAT")
    _advanced(ticker, df, charts)


def _advanced(ticker: str, df: pd.DataFrame, charts) -> None:
    from ovtlyr.indicators.advanced import (compute_bull_list_pct, compute_oscillator_direction,
                                            compute_volatility_histogram)
    try:
        long_df, _src = load_ohlcv(ticker, "2y")
        vh = compute_volatility_histogram(long_df if long_df is not None else df)
        st.plotly_chart(charts.build_volatility_histogram(vh), use_container_width=True,
                        config={"displayModeBar": False}, key="ta_vh")
        note(f"Upp-dagar {vh.get('up_pct', 50):.0f} % · snitt upp +{vh.get('mean_up', 0):.2f} % · snitt ner "
             f"{vh.get('mean_down', 0):.2f} % · {vh.get('years_analyzed', 0):.1f} års data")
    except Exception as exc:
        note(f"Volatilitetsfördelningen kunde inte räknas: {exc}")
    try:
        osc = compute_oscillator_direction(df)
        st.plotly_chart(charts.build_oscillator_direction(osc), use_container_width=True,
                        config={"displayModeBar": False}, key="ta_osc")
        note(f"{osc.get('timing', '')} ({osc.get('days_in_direction', 0)} dagar) — RSI {osc.get('rsi', 0):.0f} "
             f"({osc.get('rsi_change_5d', 0):+.0f} på 5 dagar). Tidigt i en uppgång är bättre än sent.")
    except Exception as exc:
        note(f"Oscillatorn kunde inte räknas: {exc}")
    try:
        from market_prices import ohlcv
        etfs = {t: ohlcv(t, "3mo") for t in BULL_LIST_ETFS}
        etfs = {t: d for t, d in etfs.items() if d is not None and len(d)}
        if etfs:
            bl = compute_bull_list_pct(etfs)
            st.plotly_chart(charts.build_bull_list_gauge(bl), use_container_width=True,
                            config={"displayModeBar": False}, key="ta_bl")
            note(f"Bull List % = andel av {len(etfs)} amerikanska sektor-ETF:er över EMA50 — marknadens bredd, "
                 f"inte aktiens. Samma mätare som i Viking Regime.")
    except Exception as exc:
        note(f"Bull List kunde inte räknas: {exc}")
