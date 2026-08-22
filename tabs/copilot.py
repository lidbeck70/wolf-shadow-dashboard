"""
tabs/copilot.py — AI Trading Copilot
======================================
Interaktiv kandidatanalys med deterministisk regelkontroll och AI-kommentar.

Flöde:
    1. Välj strategi + ange ticker
    2. Hämta live-data automatiskt (ATR-stop, ATR-target, teknisk kontext)
    3. Fyll i / justera prisdata (entry, stop, target)
    4. Regelkontroll per regel (PASS / MANUAL / FAIL) med auto-detektering
    5. Kandidatkort med rekommendation
    6. AI-kommentar (GPT med rik teknisk kontext, fallback om nyckel saknas)
    7. Watchlist — bevaka flera kandidater per session
    8. Snabb journal-logg
"""
from __future__ import annotations

import json
import os
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import streamlit as st

from strategy_rules import PLAYBOOKS, Playbook
from ui.theme import section_title, PALETTE as _P

# ── Palette shortcuts ─────────────────────────────────────────────────────────
_GREEN  = _P.get("green",    "#2d8a4e")
_RED    = _P.get("red",      "#c44545")
_AMBER  = _P.get("amber",    "#d4943a")
_CYAN   = _P.get("gold",     "#00E5FF")
_DIM    = _P.get("text_dim", "#6B7280")
_BG     = "#1A1F25"
_BORDER = "rgba(255,255,255,0.06)"

# ── Journal storage ───────────────────────────────────────────────────────────
_JOURNAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".copilot_journal.json",
)

# ── Watchlist storage key ─────────────────────────────────────────────────────
_WATCHLIST_KEY = "copilot_watchlist"


def _load_journal() -> list[dict]:
    if os.path.exists(_JOURNAL_PATH):
        try:
            with open(_JOURNAL_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_journal(entries: list[dict]) -> None:
    try:
        with open(_JOURNAL_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        st.warning(f"Kunde inte spara journal: {exc}")


# ── Nivå 1: Live-data fetcher ─────────────────────────────────────────────────

@dataclass
class LiveData:
    """Teknisk snapshot hämtad från yfinance."""
    ticker: str
    close: float
    atr14: float
    # Suggested levels
    suggested_stop: float      # 1 ATR under swing low (senaste 10-bars low)
    suggested_target_2r: float # close + 2 ATR
    suggested_target_3r: float # close + 3 ATR
    # Trend
    ema20: float
    ema50: float
    ema200: float
    trend_state: str           # "Bullish" | "Bearish" | "Neutral"
    price_above_ema200: bool
    ema50_above_ema200: bool
    # Momentum
    rsi: float
    ob_os_flag: str            # "Overbought" | "Oversold" | "Neutral"
    roc_10: float
    # Volume
    volume_ratio: float        # current / 20d avg
    volume_trend: str          # "Rising" | "Falling" | "Flat"
    # Candlestick patterns (last 3 bars)
    bullish_patterns: list[str] = field(default_factory=list)
    bearish_patterns: list[str] = field(default_factory=list)
    # ADX
    adx: float = 0.0
    error: str = ""


def _compute_adx(df, period: int = 14) -> float:
    """Simple ADX calculation (no external lib needed)."""
    try:
        import pandas as pd
        import numpy as np
        high = df["High"].astype(float)
        low  = df["Low"].astype(float)
        close = df["Close"].astype(float)
        if len(close) < period * 2:
            return float("nan")
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        dm_plus  = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0.0)
        dm_minus = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0.0)
        atr_s = tr.ewm(span=period, adjust=False).mean()
        di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean()  / atr_s.replace(0, float("nan"))
        di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr_s.replace(0, float("nan"))
        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, float("nan")))
        adx = dx.ewm(span=period, adjust=False).mean()
        return float(adx.iloc[-1])
    except Exception:
        return float("nan")


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_live_data(ticker: str) -> LiveData:
    """Hämtar OHLCV + beräknar alla tekniska indikatorer. Cachas 5 min."""
    try:
        import yfinance as yf
        from ovtlyr.indicators.volatility import compute_volatility
        from ovtlyr.indicators.trend import compute_trend
        from ovtlyr.indicators.momentum import compute_momentum
        from ovtlyr.indicators.volume import compute_volume
        from ovtlyr.indicators.candlesticks import detect_patterns

        raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 30:
            return LiveData(ticker=ticker, close=0, atr14=0, suggested_stop=0,
                            suggested_target_2r=0, suggested_target_3r=0,
                            ema20=0, ema50=0, ema200=0, trend_state="Neutral",
                            price_above_ema200=False, ema50_above_ema200=False,
                            rsi=50, ob_os_flag="Neutral", roc_10=0,
                            volume_ratio=1, volume_trend="Flat",
                            error=f"Ingen data hittades för '{ticker}'")

        # Flatten multi-level columns if needed
        if isinstance(raw.columns, __import__("pandas").MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        vol   = compute_volatility(raw)
        trend = compute_trend(raw)
        mom   = compute_momentum(raw)
        volm  = compute_volume(raw)
        pats  = detect_patterns(raw, lookback=3)
        adx   = _compute_adx(raw)

        close     = float(raw["Close"].iloc[-1])
        atr14     = float(vol["atr14"]) if vol["atr14"] == vol["atr14"] else 0.0
        swing_low = float(raw["Low"].iloc[-10:].min())

        # EMA 20 (not in compute_trend, compute manually)
        import pandas as pd
        ema20_series = raw["Close"].ewm(span=20, adjust=False).mean()
        ema20 = float(ema20_series.iloc[-1])
        ema50 = float(trend["ema50"].iloc[-1]) if not trend["ema50"].empty else 0.0
        ema200 = float(trend["ema200"].iloc[-1]) if not trend["ema200"].empty else 0.0

        return LiveData(
            ticker=ticker,
            close=close,
            atr14=atr14,
            suggested_stop=round(swing_low - atr14 * 0.5, 2),
            suggested_target_2r=round(close + atr14 * 2, 2),
            suggested_target_3r=round(close + atr14 * 3, 2),
            ema20=round(ema20, 2),
            ema50=round(ema50, 2),
            ema200=round(ema200, 2),
            trend_state=trend["trend_state"],
            price_above_ema200=bool(trend["price_above_200"]),
            ema50_above_ema200=bool(trend["ema50_above_200"]),
            rsi=round(float(mom["rsi"]), 1),
            ob_os_flag=mom["ob_os_flag"],
            roc_10=round(float(mom["roc_10"]) if mom["roc_10"] == mom["roc_10"] else 0.0, 1),
            volume_ratio=round(float(volm["volume_ratio"]) if volm["volume_ratio"] == volm["volume_ratio"] else 1.0, 2),
            volume_trend=volm["volume_trend"],
            bullish_patterns=[f"{p.visual} {p.name} ({p.confidence})" for p in pats["bullish"]],
            bearish_patterns=[f"{p.visual} {p.name} ({p.confidence})" for p in pats["bearish"]],
            adx=round(adx if adx == adx else 0.0, 1),
        )
    except Exception as exc:
        return LiveData(ticker=ticker, close=0, atr14=0, suggested_stop=0,
                        suggested_target_2r=0, suggested_target_3r=0,
                        ema20=0, ema50=0, ema200=0, trend_state="Neutral",
                        price_above_ema200=False, ema50_above_ema200=False,
                        rsi=50, ob_os_flag="Neutral", roc_10=0,
                        volume_ratio=1, volume_trend="Flat",
                        error=str(exc))


# ── Rule check logic ──────────────────────────────────────────────────────────


@dataclass
class RuleResult:
    number: int
    text: str
    status: str          # "PASS" | "MANUAL" | "FAIL"
    note: str = ""
    hard: bool = False


def _check_rules(
    pb: Playbook,
    entry: float,
    stop: float,
    target: float,
    live: Optional[LiveData] = None,
) -> list[RuleResult]:
    """
    Deterministisk regelkontroll.

    Med live-data (LiveData) konverteras MANUAL-regler automatiskt till
    PASS/FAIL baserade på EMA-trend, ADX, volym, RSI och candlestick-mönster.
    Utan live-data faller allt tillbaka till MANUAL-status.
    """
    results: list[RuleResult] = []

    risk_pct = abs(entry - stop) / entry * 100 if entry else 0
    rr = abs(target - entry) / abs(entry - stop) if (entry and stop and entry != stop) else 0

    for r in pb.entry:
        text_lower = r.text.lower()

        # ── R:R-kontroll ──────────────────────────────────────────────────────
        if "1:2" in r.text or "r:r" in text_lower or "reward" in text_lower:
            if rr >= 2.0:
                results.append(RuleResult(r.number, r.text, "PASS",
                                           f"R:R = {rr:.1f}x ✓", r.hard))
            else:
                results.append(RuleResult(r.number, r.text, "FAIL",
                                           f"R:R = {rr:.1f}x — kräver ≥ 2,0", r.hard))

        # ── Stop-nivå ─────────────────────────────────────────────────────────
        elif "stop" in text_lower and entry and stop:
            results.append(RuleResult(r.number, r.text, "PASS",
                                       f"Stop = {risk_pct:.1f} % från entry", r.hard))

        # ── Trend/regime-kontroll (auto med live-data) ────────────────────────
        elif live and live.close and (
            "trend" in text_lower or "regime" in text_lower
            or "bullish" in text_lower or "riktning" in text_lower
            or "ema" in text_lower or "ma " in text_lower
        ):
            if live.price_above_ema200 and live.ema50_above_ema200:
                results.append(RuleResult(r.number, r.text, "PASS",
                    f"Trend: {live.trend_state} · pris {live.close:.2f} > EMA200 {live.ema200:.2f} ✓",
                    r.hard))
            elif live.trend_state == "Bearish":
                results.append(RuleResult(r.number, r.text, "FAIL",
                    f"Trend: Bearish · pris {live.close:.2f} < EMA200 {live.ema200:.2f}",
                    r.hard))
            else:
                results.append(RuleResult(r.number, r.text, "MANUAL",
                    f"Trend: Neutral · EMA50 {live.ema50:.2f} / EMA200 {live.ema200:.2f}",
                    r.hard))

        # ── Konsolidering/ADX-kontroll ────────────────────────────────────────
        elif live and live.close and (
            "konsolid" in text_lower or "consolidat" in text_lower
            or "adx" in text_lower or "direkt" in text_lower
        ):
            if live.adx >= 25:
                results.append(RuleResult(r.number, r.text, "PASS",
                    f"ADX = {live.adx:.1f} (≥ 25 = trend aktiv) ✓", r.hard))
            elif live.adx > 0:
                results.append(RuleResult(r.number, r.text, "FAIL",
                    f"ADX = {live.adx:.1f} (< 25 = konsolidering)", r.hard))
            else:
                results.append(RuleResult(r.number, r.text, "MANUAL",
                    "ADX ej beräknat — kontrollera i panelen", r.hard))

        # ── Volymkonfirmation ─────────────────────────────────────────────────
        elif live and live.close and (
            "volym" in text_lower or "volume" in text_lower
        ):
            if live.volume_ratio >= 1.2:
                results.append(RuleResult(r.number, r.text, "PASS",
                    f"Volym = {live.volume_ratio:.1f}× snitt · {live.volume_trend} ✓", r.hard))
            elif live.volume_ratio < 0.8:
                results.append(RuleResult(r.number, r.text, "FAIL",
                    f"Volym = {live.volume_ratio:.1f}× snitt — låg volym", r.hard))
            else:
                results.append(RuleResult(r.number, r.text, "MANUAL",
                    f"Volym = {live.volume_ratio:.1f}× snitt — gränsfall", r.hard))

        # ── Candlestick trigger ───────────────────────────────────────────────
        elif live and live.close and (
            "candlestick" in text_lower or "candle" in text_lower
            or "ljus" in text_lower or "trigger" in text_lower
            or "engulf" in text_lower or "hammer" in text_lower
            or "pinbar" in text_lower or "pin bar" in text_lower
        ):
            if live.bullish_patterns:
                results.append(RuleResult(r.number, r.text, "PASS",
                    f"Mönster: {live.bullish_patterns[0]} ✓", r.hard))
            elif live.bearish_patterns:
                results.append(RuleResult(r.number, r.text, "FAIL",
                    f"Bearish mönster detekterat: {live.bearish_patterns[0]}", r.hard))
            else:
                results.append(RuleResult(r.number, r.text, "MANUAL",
                    "Inget candlestick-mönster detekterat — kontrollera manuellt", r.hard))

        # ── RSI/momentum ──────────────────────────────────────────────────────
        elif live and live.close and (
            "rsi" in text_lower or "momentum" in text_lower or "overbought" in text_lower
            or "överköpt" in text_lower or "impulse" in text_lower
        ):
            if live.ob_os_flag == "Overbought":
                results.append(RuleResult(r.number, r.text, "FAIL",
                    f"RSI = {live.rsi:.0f} — överköpt, undvik entry", r.hard))
            elif live.ob_os_flag == "Oversold":
                results.append(RuleResult(r.number, r.text, "PASS",
                    f"RSI = {live.rsi:.0f} — översålt, möjlig reversal ✓", r.hard))
            elif 40 <= live.rsi <= 65:
                results.append(RuleResult(r.number, r.text, "PASS",
                    f"RSI = {live.rsi:.0f} — hälsosamt momentum ✓", r.hard))
            else:
                results.append(RuleResult(r.number, r.text, "MANUAL",
                    f"RSI = {live.rsi:.0f} — kontrollera momentum-riktning", r.hard))

        # ── Allt annat kräver manuell koll ────────────────────────────────────
        else:
            results.append(RuleResult(r.number, r.text, "MANUAL",
                                       "Kontrollera i panelen: " + r.panel_guide[:80],
                                       r.hard))

    return results


# ── Candidate card ────────────────────────────────────────────────────────────

def _status_color(status: str) -> str:
    return {"PASS": _GREEN, "MANUAL": _AMBER, "FAIL": _RED}.get(status, _DIM)


def _render_rule_row(res: RuleResult) -> None:
    col_s, col_t, col_n = st.columns([1, 4, 4])
    color = _status_color(res.status)
    badge = (
        f'<span style="background:{color}22;color:{color};font-size:10px;'
        f'font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:1px;">'
        f'{res.status}</span>'
    )
    with col_s:
        st.markdown(badge, unsafe_allow_html=True)
    with col_t:
        hard_marker = " 🔒" if res.hard else ""
        st.markdown(
            f'<span style="font-size:12px;color:#E8EDF2;">{res.text}{hard_marker}</span>',
            unsafe_allow_html=True,
        )
    with col_n:
        st.markdown(
            f'<span style="font-size:11px;color:{_DIM};">{res.note}</span>',
            unsafe_allow_html=True,
        )


def _overall_status(results: list[RuleResult]) -> str:
    if any(r.status == "FAIL" and r.hard for r in results):
        return "REJECT"
    if any(r.status == "FAIL" for r in results):
        return "REJECT"
    if any(r.status == "MANUAL" for r in results):
        return "WATCH"
    return "BUY CANDIDATE"


def _status_to_color(s: str) -> str:
    return {
        "BUY CANDIDATE": _GREEN,
        "WATCH":         _AMBER,
        "REJECT":        _RED,
    }.get(s, _DIM)


# ── AI comment (OpenAI GPT, Nivå 3 — rik teknisk kontext) ────────────────────

def _ai_comment(
    ticker: str,
    pb: Playbook,
    results: list[RuleResult],
    entry: float,
    stop: float,
    target: float,
    live: Optional[LiveData] = None,
) -> str:
    """
    Anropar OpenAI Chat Completions med rik teknisk kontext och returnerar
    en analys på svenska (entry + exit + grannsteg-förslag).

    Kräver OPENAI_API_KEY i st.secrets eller miljövariabel.
    Faller tillbaka till en deterministisk kommentar om nyckeln saknas.
    """
    passed = [r.text for r in results if r.status == "PASS"]
    failed = [r.text for r in results if r.status == "FAIL"]
    manual = [r.text for r in results if r.status == "MANUAL"]
    rr     = abs(target - entry) / abs(entry - stop) if (entry and stop and entry != stop) else 0
    risk_p = abs(entry - stop) / entry * 100 if entry else 0

    api_key: Optional[str] = (
        st.secrets.get("OPENAI_API_KEY")
        if hasattr(st, "secrets")
        else None
    ) or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return _fallback_comment(ticker, pb, passed, failed, manual, rr, risk_p, live)

    # ── Bygg rik teknisk kontext-sträng ──────────────────────────────────────
    tech_ctx = ""
    if live and live.close:
        bull_str = ", ".join(live.bullish_patterns[:2]) or "inga"
        bear_str = ", ".join(live.bearish_patterns[:2]) or "inga"
        tech_ctx = (
            f"\nTeknisk snapshot ({ticker}):\n"
            f"  Aktuellt pris: {live.close:.2f}\n"
            f"  Trend: {live.trend_state} · EMA20={live.ema20:.2f} · EMA50={live.ema50:.2f} · EMA200={live.ema200:.2f}\n"
            f"  Pris > EMA200: {'Ja' if live.price_above_ema200 else 'Nej'} · EMA50 > EMA200: {'Ja' if live.ema50_above_ema200 else 'Nej'}\n"
            f"  ADX: {live.adx:.1f} ({'trend aktiv' if live.adx >= 25 else 'konsolidering'})\n"
            f"  RSI(14): {live.rsi:.0f} ({live.ob_os_flag}) · ROC(10): {live.roc_10:+.1f}%\n"
            f"  Volym: {live.volume_ratio:.1f}× snitt · {live.volume_trend}\n"
            f"  ATR(14): {live.atr14:.2f}\n"
            f"  Bullish mönster: {bull_str}\n"
            f"  Bearish mönster: {bear_str}\n"
            f"  Föreslaget stop (ATR): {live.suggested_stop:.2f}\n"
            f"  Föreslaget target 2R: {live.suggested_target_2r:.2f} · 3R: {live.suggested_target_3r:.2f}\n"
        )

    prompt = (
        f"Du är en professionell swing-trading-analytiker.\n\n"
        f"Ticker: {ticker}\n"
        f"Strategi: {pb.name} ({pb.tagline})\n"
        f"Entry: {entry:.2f} · Stop: {stop:.2f} · Target: {target:.2f}\n"
        f"R:R: {rr:.1f}x · Risk från entry: {risk_p:.1f}%\n"
        f"Godkända regler ({len(passed)}): {', '.join(passed) or 'inga'}\n"
        f"Manuella regler ({len(manual)}): {', '.join(manual) or 'inga'}\n"
        f"Misslyckade regler ({len(failed)}): {', '.join(failed) or 'inga'}\n"
        f"{tech_ctx}\n"
        f"Ge en analys (5–7 meningar) på svenska med dessa fyra delar:\n"
        f"1. ENTRY: Är entry-tillfället tekniskt motiverat? Stöds det av trend, volym och candlestick?\n"
        f"2. EXIT: Finns det tekniska motstånd (EMA200, previous high, runda tal) nära target {target:.2f}?\n"
        f"3. RISK: Om affären rör sig mot dig — var är rätt punkt att halvera positionen?\n"
        f"4. SLUTSATS: Rekommendation i ett ord (BUY CANDIDATE / WATCH / REJECT) och motivering.\n"
    )

    try:
        import openai  # noqa: PLC0415 — lazy import to keep startup fast
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4,
        )
        return response.choices[0].message.content or "Inget svar från AI."
    except Exception as exc:  # pragma: no cover
        return (
            f"{_fallback_comment(ticker, pb, passed, failed, manual, rr, risk_p, live)}\n\n"
            f"_⚠️ OpenAI-anropet misslyckades: {exc}_"
        )


def _fallback_comment(
    ticker: str,
    pb: Playbook,
    passed: list[str],
    failed: list[str],
    manual: list[str],
    rr: float,
    risk_p: float,
    live: Optional[LiveData] = None,
) -> str:
    """Deterministisk fallback-kommentar när OpenAI ej är tillgängligt."""
    lines = [f"**{ticker}** analyseras mot **{pb.name}**.", ""]

    if live and live.close:
        trend_icon = "🟢" if live.trend_state == "Bullish" else ("🔴" if live.trend_state == "Bearish" else "🟡")
        lines.append(f"{trend_icon} **Trend:** {live.trend_state} · EMA20={live.ema20:.2f} · EMA50={live.ema50:.2f} · EMA200={live.ema200:.2f}")
        adx_note = "trend aktiv" if live.adx >= 25 else "konsolidering"
        lines.append(f"📊 **ADX:** {live.adx:.1f} ({adx_note}) · **RSI:** {live.rsi:.0f} ({live.ob_os_flag})")
        lines.append(f"📦 **Volym:** {live.volume_ratio:.1f}× snitt ({live.volume_trend})")
        if live.bullish_patterns:
            lines.append(f"🕯️ **Mönster:** {live.bullish_patterns[0]}")
        lines.append("")

    if passed:
        lines.append(f"✅ Automatiska kontroller godkända: {len(passed)} regler klarade.")
    if manual:
        lines.append(f"⚠️ {len(manual)} regler kräver manuell kontroll i panelen.")
    if failed:
        lines.append(f"❌ {len(failed)} hårda regler misslyckades — affären bör undvikas.")

    lines += [
        "",
        f"Risk per affär: **{risk_p:.1f} %** av entry.",
        f"R:R-förhållande: **{rr:.1f}x** (kräver ≥ 2,0 för PASS).",
    ]

    if rr >= 3.0:
        lines.append("📈 Utmärkt R:R — affären erbjuder bra asymmetri.")
    elif rr >= 2.0:
        lines.append("📊 Godkänt R:R — klara alla MANUAL-regler innan du agerar.")
    else:
        lines.append("🚫 R:R under 2,0 — leta en bättre entry eller ett vidare target.")

    lines += [
        "",
        "_💡 Sätt OPENAI_API_KEY i Streamlit Secrets för att aktivera GPT-analys._",
    ]

    return "\n".join(lines)


# ── Nivå 4: Watchlist ─────────────────────────────────────────────────────────

def _get_watchlist() -> list[dict]:
    return st.session_state.get(_WATCHLIST_KEY, [])


def _save_watchlist(wl: list[dict]) -> None:
    st.session_state[_WATCHLIST_KEY] = wl


def _add_to_watchlist(
    ticker: str, strategy_key: str, entry: float, stop: float, target: float,
    overall: str, live: Optional[LiveData]
) -> None:
    wl = _get_watchlist()
    # Remove existing entry for same ticker+strategy
    wl = [w for w in wl if not (w["ticker"] == ticker and w["strategy"] == strategy_key)]
    wl.append({
        "ticker": ticker,
        "strategy": strategy_key,
        "entry": entry,
        "stop": stop,
        "target": target,
        "status": overall,
        "rsi": live.rsi if live else None,
        "trend": live.trend_state if live else None,
        "adx": live.adx if live else None,
        "added": datetime.datetime.utcnow().isoformat(),
    })
    _save_watchlist(wl)


def _render_watchlist() -> None:
    """Visa alla bevakade kandidater med uppdaterade R:R och status."""
    section_title("👁 Watchlist", "")
    wl = _get_watchlist()
    if not wl:
        st.caption("Inga kandidater i watchlistan ännu — lägg till via knappen nedan.")
        return

    st.caption(f"{len(wl)} kandidat(er) bevakad(e) denna session.")
    for w in reversed(wl):
        pb_name = PLAYBOOKS[w["strategy"]].name if w["strategy"] in PLAYBOOKS else w["strategy"]
        rr_val  = abs(w.get("target", 0) - w.get("entry", 0)) / abs(
            w.get("entry", 0) - (w.get("stop", 0) or 1e-9)
        ) if w.get("entry") else 0
        s_color = {"BUY CANDIDATE": _GREEN, "WATCH": _AMBER, "REJECT": _RED}.get(w["status"], _DIM)

        # Breakeven reminder
        be_note = ""
        if w.get("entry") and w.get("stop"):
            be_pct = abs(w["entry"] - w["stop"]) / w["entry"] * 100
            be_note = f" · Flytta SL till BE om pris rör sig {be_pct:.1f}% i rätt riktning"

        trend_badge = ""
        if w.get("trend"):
            t_col = {"Bullish": _GREEN, "Bearish": _RED, "Neutral": _AMBER}.get(w["trend"], _DIM)
            trend_badge = (
                f'<span style="background:{t_col}22;color:{t_col};font-size:10px;'
                f'font-weight:700;padding:1px 6px;border-radius:3px;margin-left:6px;">'
                f'{w["trend"]}</span>'
            )

        adx_str = f" · ADX {w['adx']:.0f}" if w.get("adx") else ""
        rsi_str = f" · RSI {w['rsi']:.0f}" if w.get("rsi") else ""

        st.markdown(
            f'<div style="background:{_BG};border:1px solid {_BORDER};'
            f'border-left:3px solid {s_color};border-radius:6px;'
            f'padding:10px 14px;margin-bottom:6px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div>'
            f'<span style="font-size:14px;font-weight:700;color:#E8EDF2;">{w["ticker"]}</span>'
            f'{trend_badge}'
            f'<span style="font-size:11px;color:{_DIM};margin-left:8px;">{pb_name}</span>'
            f'</div>'
            f'<span style="background:{s_color}22;color:{s_color};font-size:10px;'
            f'font-weight:700;padding:2px 8px;border-radius:4px;">{w["status"]}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:{_DIM};margin-top:4px;">'
            f'Entry {w.get("entry","—")} · Stop {w.get("stop","—")} · '
            f'Target {w.get("target","—")} · R:R {rr_val:.1f}x'
            f'{adx_str}{rsi_str}'
            f'<span style="color:{_AMBER};">{be_note}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.button("🗑 Rensa watchlist", key="clear_watchlist"):
        _save_watchlist([])
        st.rerun()


def _render_journal_log(ticker: str, strategy_key: str,
                        entry: float, stop: float, target: float) -> None:
    """Snabb loggning av trade-kandidat till journal."""
    section_title("📓 Logga kandidat", "")
    st.caption("Spara kandidaten till din lokala journal för uppföljning.")

    with st.form("copilot_log_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            log_date   = st.date_input("Datum", value=datetime.date.today())
            log_ticker = st.text_input("Ticker", value=ticker.upper())
        with col2:
            log_strat  = st.selectbox(
                "Strategi",
                options=list(PLAYBOOKS.keys()),
                index=list(PLAYBOOKS.keys()).index(strategy_key) if strategy_key in PLAYBOOKS else 0,
                format_func=lambda k: PLAYBOOKS[k].name,
            )
            log_note   = st.text_input("Anteckning (valfri)")

        submitted = st.form_submit_button("💾 Spara till journal", type="primary")
        if submitted:
            entries = _load_journal()
            entries.append({
                "date":     str(log_date),
                "ticker":   log_ticker.upper(),
                "strategy": log_strat,
                "entry":    entry,
                "stop":     stop,
                "target":   target,
                "note":     log_note,
                "logged_at": datetime.datetime.utcnow().isoformat(),
            })
            _save_journal(entries)
            st.success(f"✅ {log_ticker.upper()} loggad!")


def _render_journal_history() -> None:
    """Visa senaste journal-poster."""
    entries = _load_journal()
    if not entries:
        st.caption("Inga poster ännu.")
        return

    recent = list(reversed(entries[-20:]))
    for e in recent:
        rr_val = abs(e.get("target", 0) - e.get("entry", 0)) / abs(
            e.get("entry", 0) - e.get("stop", 1e-9) or 1e-9
        ) if e.get("entry") else 0
        st.markdown(
            f'<div style="background:{_BG};border:1px solid {_BORDER};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:13px;font-weight:700;color:#E8EDF2;">'
            f'{e.get("ticker","?")} '
            f'<span style="font-size:11px;color:{_DIM};font-weight:400;">'
            f'— {PLAYBOOKS.get(e.get("strategy",""), type("",(),{"name":e.get("strategy","")})()).name}'  # type: ignore[attr-defined]
            f'</span></span>'
            f'<span style="font-size:11px;color:{_DIM};">'
            f'Entry {e.get("entry","—")} · Stop {e.get("stop","—")} · '
            f'Target {e.get("target","—")} · R:R {rr_val:.1f}x'
            f'</span>'
            f'<span style="font-size:11px;color:{_DIM};">{e.get("date","")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Main render function ──────────────────────────────────────────────────────

def render_copilot_page() -> None:
    """Entry point — anropas från wolf_panel.py."""
    section_title("AI Trading Copilot", "🤖")
    st.markdown(
        f'<p style="color:{_DIM};font-size:0.82rem;margin:-8px 0 24px;">'
        f'Kandidatanalys · Regelkontroll · Riskkontroll · Journal · Watchlist</p>',
        unsafe_allow_html=True,
    )

    # ── Inmatning ─────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([2, 3])
    with col_a:
        strategy_key = st.selectbox(
            "Strategi",
            options=list(PLAYBOOKS.keys()),
            format_func=lambda k: PLAYBOOKS[k].name,
            key="copilot_strategy",
        )
    with col_b:
        ticker = st.text_input(
            "Ticker (yfinance-format, t.ex. ABB.ST, ERIC-B.ST)",
            placeholder="t.ex. ABB.ST, ERIC-B.ST, TSLA",
            key="copilot_ticker",
        ).strip().upper()

    pb: Playbook = PLAYBOOKS[strategy_key]

    st.markdown(
        f'<div style="background:{pb.color}11;border-left:3px solid {pb.color};'
        f'border-radius:6px;padding:10px 14px;margin:8px 0 16px;">'
        f'<span style="font-size:12px;color:{pb.color};font-weight:700;">{pb.name}</span>'
        f'<span style="font-size:11px;color:{_DIM};margin-left:10px;">{pb.tagline}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not ticker:
        st.info("👆 Ange en ticker ovan för att starta analysen.", icon="ℹ️")
        _render_watchlist()
        return

    # ── Nivå 1: Hämta live-data ───────────────────────────────────────────────
    live: Optional[LiveData] = None
    fetch_col, _ = st.columns([2, 5])
    with fetch_col:
        if st.button("🔄 Hämta live-data", key="fetch_live", type="primary"):
            with st.spinner(f"Hämtar data för {ticker}…"):
                live = _fetch_live_data(ticker)
                st.session_state["copilot_live"] = live
        elif "copilot_live" in st.session_state:
            cached: LiveData = st.session_state["copilot_live"]
            if cached.ticker == ticker:
                live = cached

    # Visa teknisk snapshot om live-data finns
    if live:
        if live.error:
            st.warning(f"⚠️ Kunde inte hämta data: {live.error}")
            live = None
        else:
            _render_tech_snapshot(live)

    # ── Nivå 1: Prisdata med auto-fill ────────────────────────────────────────
    with st.expander("📐 Prisdata & R:R-beräkning", expanded=True):
        # Auto-fill suggestions
        default_entry  = live.close              if live and live.close else 0.0
        default_stop   = live.suggested_stop     if live and live.suggested_stop else 0.0
        default_target = live.suggested_target_2r if live and live.suggested_target_2r else 0.0

        if live and live.close:
            st.caption(
                f"💡 Förslag (ATR-baserat): Entry ≈ {default_entry:.2f} · "
                f"Stop ≈ {default_stop:.2f} · Target 2R ≈ {default_target:.2f} / "
                f"3R ≈ {live.suggested_target_3r:.2f}"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            entry  = st.number_input("Entry-kurs", min_value=0.0,
                                     value=float(default_entry),
                                     step=0.5, format="%.2f", key="copilot_entry")
        with c2:
            stop   = st.number_input("Stop-kurs", min_value=0.0,
                                     value=float(default_stop),
                                     step=0.5, format="%.2f", key="copilot_stop")
        with c3:
            target = st.number_input("Target-kurs", min_value=0.0,
                                     value=float(default_target),
                                     step=0.5, format="%.2f", key="copilot_target")

        if entry and stop and entry != stop:
            rr      = abs(target - entry) / abs(entry - stop) if target else 0
            risk_p  = abs(entry - stop) / entry * 100
            rr_col  = _GREEN if rr >= 2.0 else (_AMBER if rr >= 1.5 else _RED)
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:22px;font-weight:700;color:{rr_col};">'
                    f'{rr:.1f}x</div>'
                    f'<div style="font-size:11px;color:{_DIM};">R:R-förhållande</div></div>',
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:22px;font-weight:700;color:{_AMBER};">'
                    f'{risk_p:.1f} %</div>'
                    f'<div style="font-size:11px;color:{_DIM};">Risk från entry</div></div>',
                    unsafe_allow_html=True,
                )
            with mc3:
                upside = abs(target - entry) / entry * 100 if target else 0
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:22px;font-weight:700;color:{_GREEN};">'
                    f'+{upside:.1f} %</div>'
                    f'<div style="font-size:11px;color:{_DIM};">Potentiell vinst</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Regelkontroll (Nivå 2) ─────────────────────────────────────────────────
    if not pb.entry:
        st.warning("Denna strategi har inga definierade entry-regler ännu.")
        return

    results = _check_rules(pb, entry, stop, target, live)
    overall = _overall_status(results)
    overall_color = _status_to_color(overall)

    # Kandidatkort header
    st.markdown(
        f'<div style="background:{_BG};border:1px solid {_BORDER};'
        f'border-top:3px solid {overall_color};border-radius:10px;'
        f'padding:18px 20px;margin-bottom:20px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<span style="font-size:22px;font-weight:800;color:#E8EDF2;">{ticker}</span>'
        f'<span style="font-size:13px;color:{_DIM};margin-left:12px;">{pb.name}</span>'
        f'</div>'
        f'<div style="background:{overall_color}22;color:{overall_color};'
        f'font-size:12px;font-weight:700;padding:6px 16px;border-radius:6px;'
        f'letter-spacing:2px;">{overall}</div>'
        f'</div>'
        f'<div style="margin-top:12px;display:flex;gap:24px;flex-wrap:wrap;">'
        f'<span style="font-size:12px;color:{_DIM};">Entry <b style="color:#E8EDF2;">'
        f'{entry:.2f}</b></span>'
        f'<span style="font-size:12px;color:{_DIM};">Stop <b style="color:{_RED};">'
        f'{stop:.2f}</b></span>'
        f'<span style="font-size:12px;color:{_DIM};">Target <b style="color:{_GREEN};">'
        f'{target:.2f}</b></span>'
        f'<span style="font-size:12px;color:{_DIM};">Risk '
        f'<b style="color:{_AMBER};">{pb.risk.risk_per_trade}</b></span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Regelrader
    section_title("Regelkontroll — Entry", "✅")
    auto_note = " · Auto-kontrollerat med live-data" if live and live.close else ""
    st.markdown(
        f'<p style="font-size:11px;color:{_DIM};margin:-6px 0 10px;">'
        f'🔒 = hård regel (aldrig bruten) · MANUAL = kräver kontroll i panelen{auto_note}</p>',
        unsafe_allow_html=True,
    )
    for res in results:
        _render_rule_row(res)

    passed_count = sum(1 for r in results if r.status == "PASS")
    manual_count = sum(1 for r in results if r.status == "MANUAL")
    fail_count   = sum(1 for r in results if r.status == "FAIL")
    st.markdown(
        f'<div style="margin:14px 0 24px;font-size:12px;color:{_DIM};">'
        f'<span style="color:{_GREEN};">✅ {passed_count} PASS</span> &nbsp;·&nbsp; '
        f'<span style="color:{_AMBER};">⚠️ {manual_count} MANUAL</span> &nbsp;·&nbsp; '
        f'<span style="color:{_RED};">❌ {fail_count} FAIL</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── AI-kommentar (Nivå 3) ─────────────────────────────────────────────────
    section_title("AI-kommentar", "💬")
    with st.spinner("Hämtar AI-kommentar…"):
        comment = _ai_comment(ticker, pb, results, entry, stop, target, live)
    st.markdown(comment)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:28px 0;'>",
                unsafe_allow_html=True)

    # ── Watchlist (Nivå 4) ────────────────────────────────────────────────────
    wl_col1, wl_col2 = st.columns([3, 2])
    with wl_col1:
        if st.button(f"⭐ Lägg till {ticker} i watchlist", key="add_watchlist"):
            _add_to_watchlist(ticker, strategy_key, entry, stop, target, overall, live)
            st.success(f"✅ {ticker} tillagd i watchlistan!")

    _render_watchlist()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:28px 0;'>",
                unsafe_allow_html=True)

    # ── Journal ───────────────────────────────────────────────────────────────
    _render_journal_log(ticker, strategy_key, entry, stop, target)

    with st.expander("📋 Senaste journal-poster", expanded=False):
        _render_journal_history()


def _render_tech_snapshot(live: LiveData) -> None:
    """Visa teknisk snapshot-panel med live-data."""
    trend_color = {"Bullish": _GREEN, "Bearish": _RED, "Neutral": _AMBER}.get(
        live.trend_state, _DIM)
    adx_color = _GREEN if live.adx >= 25 else (_AMBER if live.adx >= 18 else _RED)
    rsi_color = (_RED if live.ob_os_flag == "Overbought"
                 else (_GREEN if live.ob_os_flag == "Oversold" else _AMBER))
    vol_color = _GREEN if live.volume_ratio >= 1.2 else (_AMBER if live.volume_ratio >= 0.8 else _RED)

    bull_str = " · ".join(live.bullish_patterns[:2]) if live.bullish_patterns else "—"
    bear_str = " · ".join(live.bearish_patterns[:1]) if live.bearish_patterns else "—"

    st.markdown(
        f'<div style="background:{_BG};border:1px solid {_BORDER};border-radius:8px;'
        f'padding:12px 16px;margin:8px 0 16px;">'
        f'<div style="font-size:11px;font-weight:700;color:{_DIM};'
        f'letter-spacing:1px;margin-bottom:8px;">📡 TEKNISK SNAPSHOT — {live.ticker}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:16px;">'
        # Pris
        f'<div><div style="font-size:18px;font-weight:800;color:#E8EDF2;">{live.close:.2f}</div>'
        f'<div style="font-size:10px;color:{_DIM};">Aktuellt pris</div></div>'
        # Trend
        f'<div><div style="font-size:14px;font-weight:700;color:{trend_color};">{live.trend_state}</div>'
        f'<div style="font-size:10px;color:{_DIM};">Trend · EMA200={live.ema200:.2f}</div></div>'
        # EMA
        f'<div><div style="font-size:12px;color:#E8EDF2;">EMA20={live.ema20:.2f} · EMA50={live.ema50:.2f}</div>'
        f'<div style="font-size:10px;color:{_DIM};">EMA-nivåer</div></div>'
        # ADX
        f'<div><div style="font-size:14px;font-weight:700;color:{adx_color};">{live.adx:.1f}</div>'
        f'<div style="font-size:10px;color:{_DIM};">ADX ({'aktiv' if live.adx >= 25 else 'konsolid.'})</div></div>'
        # RSI
        f'<div><div style="font-size:14px;font-weight:700;color:{rsi_color};">{live.rsi:.0f}</div>'
        f'<div style="font-size:10px;color:{_DIM};">RSI · {live.ob_os_flag}</div></div>'
        # Volym
        f'<div><div style="font-size:14px;font-weight:700;color:{vol_color};">{live.volume_ratio:.1f}×</div>'
        f'<div style="font-size:10px;color:{_DIM};">Volym vs snitt ({live.volume_trend})</div></div>'
        # ATR
        f'<div><div style="font-size:12px;color:#E8EDF2;">{live.atr14:.2f}</div>'
        f'<div style="font-size:10px;color:{_DIM};">ATR(14)</div></div>'
        f'</div>'
        # Candlestick patterns
        f'<div style="margin-top:8px;font-size:11px;">'
        f'<span style="color:{_GREEN};">🕯️ Bullish: {bull_str}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:{_RED};">⚠️ Bearish: {bear_str}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
