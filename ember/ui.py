"""
ember/ui.py
EMBER strategy — Streamlit page.

Renders: scan controls, TOPP 3 section, ranked setup cards (18 fields each),
MANUELL BEDÖMNING checkboxes, near-miss section, and INGA ELITCASE fallback.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from ember.config import (
    BG, BG2, BG3, EMBER, GOLD, BRONZE, GREEN, RED, AMBER, TEXT, DIM,
    EMBER_ETF_UNIVERSE, EMBER_STOCK_UNIVERSE, RISK_PCT,
)
from ember.universe import (
    ALL_SOURCES, SOURCE_CURATED, SOURCE_AUTO, SOURCE_BOTH,
    US_INTL_CURATED, UniverseStats,
)
from ember.regime import VERDICT_PA, VERDICT_SELEKTIV, VERDICT_AV, detect_complex

try:
    from ember.engine import run_ember_scan, EmberSetupResult, EmberScanResult
    _ENGINE_OK = True
except ImportError:
    _ENGINE_OK = False


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _stat_box(label: str, value: str, color: str = GOLD) -> str:
    return (
        f"<div style='background:{BG2};border:1px solid {color}33;"
        f"border-radius:6px;padding:10px 14px;text-align:center;'>"
        f"<div style='color:{DIM};font-size:0.62rem;text-transform:uppercase;"
        f"letter-spacing:0.08em;margin-bottom:4px;'>{label}</div>"
        f"<div style='color:{color};font-size:1.05rem;font-weight:700;'>{value}</div>"
        f"</div>"
    )


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}22;border:1px solid {color}55;"
        f"color:{color};border-radius:4px;padding:2px 8px;"
        f"font-size:0.68rem;font-weight:700;white-space:nowrap;'>{text}</span>"
    )


def _field(label: str, value: str, color: str = TEXT) -> str:
    return (
        f"<div style='display:flex;gap:8px;margin-bottom:5px;font-size:0.78rem;'>"
        f"<span style='color:{DIM};min-width:170px;flex-shrink:0;'>{label}</span>"
        f"<span style='color:{color};font-weight:500;'>{value}</span>"
        f"</div>"
    )


def _gate_row(name: str, passed: bool, detail: str, is_blocker: bool) -> str:
    icon  = "✅" if passed else ("❌" if is_blocker else "○")
    color = TEXT if passed else (RED if is_blocker else DIM)
    return (
        f"<div style='display:flex;align-items:flex-start;gap:7px;"
        f"margin-bottom:5px;font-size:0.76rem;'>"
        f"<span style='flex-shrink:0;'>{icon}</span>"
        f"<div><span style='color:{color};'>{name}</span>"
        f"<div style='color:{DIM};font-size:0.65rem;'>{detail}</div>"
        f"</div></div>"
    )


def _section_label(text: str) -> str:
    return (
        f"<div style='color:{DIM};font-size:0.63rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:10px 0 6px;'>{text}</div>"
    )


def _cykel_color(label: str) -> str:
    return {"TIDIG": GREEN, "MITTEN": GOLD, "SEN": AMBER, "TOPP": RED}.get(
        (label or "").upper(), DIM
    )


# ── TOPP 3 section ────────────────────────────────────────────────────────────

def _render_top3(eligible: list[EmberSetupResult]) -> None:
    st.markdown(
        f"<div style='border-bottom:2px solid {EMBER}44;"
        f"padding-bottom:6px;margin-bottom:18px;'>"
        f"<h3 style='color:{EMBER};margin:0;letter-spacing:0.08em;'>TOPP 3 JUST NU</h3>"
        f"<span style='color:{DIM};font-size:0.7rem;'>"
        f"Rangordnat efter asymmetri (RR × makro) + cykelbonus</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    top3   = eligible[:3]
    labels = ["Säkrast", "Högst Potential", "Mest Konträr"]
    cols   = st.columns(len(top3)) if top3 else []

    for i, (col, r) in enumerate(zip(cols, top3)):
        with col:
            rr_str    = f"{r.rr:.1f}R"        if r.rr    else "—"
            macro_str = f"{r.macro.total:.0f}" if r.macro else "—"
            cy_color  = _cykel_color(r.cykel_label)
            e_str     = f"{r.entry:.2f}"       if r.entry else "—"
            s_str     = f"{r.stop:.2f}"        if r.stop  else "—"
            st.markdown(
                f"<div style='background:{BG2};border:1px solid {EMBER}44;"
                f"border-top:3px solid {EMBER};border-radius:8px;padding:16px 18px;'>"
                f"<div style='color:{DIM};font-size:0.6rem;letter-spacing:0.1em;"
                f"text-transform:uppercase;margin-bottom:4px;'>{labels[i]}</div>"
                f"<div style='color:{EMBER};font-size:1.3rem;font-weight:700;"
                f"margin-bottom:3px;'>{r.ticker}</div>"
                f"<div style='color:{TEXT};font-size:0.78rem;margin-bottom:10px;'>"
                f"{r.sektor} · {r.typ}</div>"
                f"<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;'>"
                + _badge(f"RR {rr_str}", GREEN)
                + " "
                + _badge(f"MAKRO {macro_str}/100", GOLD)
                + " "
                + _badge(r.cykel_label, cy_color)
                + f"</div>"
                f"<div style='color:{DIM};font-size:0.72rem;'>"
                f"Entry <b style='color:{GREEN};'>{e_str}</b>"
                f"&nbsp;·&nbsp;Stop <b style='color:{RED};'>{s_str}</b>"
                f"</div></div>",
                unsafe_allow_html=True,
            )


# ── Setup card ────────────────────────────────────────────────────────────────

def _render_setup_card(r: EmberSetupResult, idx: int) -> None:
    rr_str    = f"{r.rr:.1f}R"        if r.rr    else "—"
    macro_str = f"{r.macro.total:.0f}/100" if r.macro else "—"
    title = (
        f"#{idx+1}  {r.ticker}  ·  {r.sektor}  "
        f"[{r.cykel_label}]  RR {rr_str}  Makro {macro_str}"
    )

    # Read per-complex regime verdict from session state
    _complex_key   = detect_complex(r.ticker)
    _all_regimes   = st.session_state.get("ember_complex_regimes", {})
    _regime        = _all_regimes.get(_complex_key) or st.session_state.get("ember_regime")
    _rv            = _regime.verdict if _regime else ""

    with st.expander(title, expanded=(idx < 1)):
        # Regime banner (AV = hard warning; SELEKTIV = soft advisory)
        if _rv == VERDICT_AV:
            st.markdown(
                f"<div style='background:#2d0a0a;border:1px solid {RED}55;"
                f"border-radius:6px;padding:9px 14px;margin-bottom:12px;"
                f"color:{RED};font-size:0.8rem;font-weight:700;'>"
                f"⛔ REGIMEN ÄR AV — inga nya entries. "
                f"Bevaka för exit om du har en öppen position.</div>",
                unsafe_allow_html=True,
            )
        elif _rv == VERDICT_SELEKTIV:
            st.markdown(
                f"<div style='background:#1a1004;border:1px solid {AMBER}55;"
                f"border-radius:6px;padding:9px 14px;margin-bottom:12px;"
                f"color:{AMBER};font-size:0.78rem;'>"
                f"🟡 SELEKTIV REGIME — halverad positionsstorlek gäller för detta setup.</div>",
                unsafe_allow_html=True,
            )

        left, right = st.columns([3, 2])

        with left:
            cy_color = _cykel_color(r.cykel_label)

            # Auto why-interesting
            why_parts = []
            if r.cykel_label in ("TIDIG", "MITTEN"):
                why_parts.append("Tidig/mitten cykelposition — optimalt timing")
            if r.macro and r.macro.copper_gold >= 100:
                why_parts.append("Copper/Gold stiger → industriell efterfrågan")
            if r.macro and r.macro.dxy >= 100:
                why_parts.append("DXY fallande → råvarumedvind")
            if r.candle_pattern not in ("NONE", ""):
                why_parts.append(f"Bullish candlestick: {r.candle_pattern}")
            why_auto = "; ".join(why_parts) if why_parts else "Vänta på bekräftelse"

            hat_txt = (f"Hat score {r.hat_score:.0f}/100 — marknaden ignorerar"
                       if r.hat_score is not None else "—")
            nec_txt = (f"Nödvändighetspoäng {r.necessity}/100"
                       if r.necessity is not None else "—")

            underval = "—"
            if r.percentile_10y is not None:
                if r.percentile_10y <= 20:
                    underval = f"10å-percentil {r.percentile_10y:.0f}% — historiskt billig"
                elif r.percentile_10y <= 50:
                    underval = f"10å-percentil {r.percentile_10y:.0f}% — under historisk median"

            trend_txt = "✅ TREND OK" if r.trend_pass else "❌ TREND EJ OK"
            t_color   = GREEN if r.trend_pass else RED

            pct_10y_txt = (f" ({r.percentile_10y:.0f}:e percentilen 10å)"
                           if r.percentile_10y else "")

            st.markdown(
                _field("Instrument",              r.ticker,       EMBER)
                + _field("Typ",                   r.typ)
                + _field("Sektor",                r.sektor)
                + _field("Var i cykeln",          r.cykel_label + pct_10y_txt, cy_color)
                + _field("Trendstatus",           trend_txt,      t_color)
                + _field("Varför intressant nu",  why_auto,       GOLD)
                + _field("Marknaden ogillar",     hat_txt,        AMBER)
                + _field("Varför ändå behövs",    nec_txt,        GOLD)
                + _field("Tecken på undervärdering", underval,    GREEN)
                + _field("Setup (mönster)",       r.candle_pattern),
                unsafe_allow_html=True,
            )

            # Price levels
            p_s  = f"{r.price:.2f}"  if r.price  else "—"
            e_s  = f"{r.entry:.2f}" if r.entry  else "—"
            st_s = f"{r.stop:.2f}"  if r.stop   else "—"
            t1_s = f"{r.t1:.2f}"    if r.t1     else "—"
            t2_s = f"{r.t2:.2f}"    if r.t2     else "—"
            rr_s = f"1:{r.rr:.1f}"  if r.rr     else "—"
            sh_s = f"{r.shares} st" if r.shares  else "—"

            # Adjust position size display based on regime
            if _rv == VERDICT_SELEKTIV and r.shares:
                sh_s       = f"{max(1, r.shares // 2)} st"
                pos_label  = f"Position ({RISK_PCT*100:.0f}% risk · ÷2 SELEKTIV)"
                pos_color  = AMBER
            else:
                pos_label  = f"Position ({RISK_PCT*100:.0f}% risk)"
                pos_color  = TEXT

            st.markdown(
                "<hr style='border-color:rgba(255,107,61,0.15);margin:10px 0;'>"
                + _field("Pris",            p_s)
                + _field("Entry",           e_s,  GREEN)
                + _field("Stop",            st_s, RED)
                + _field("T1 (1:2)",        t1_s, GOLD)
                + _field("T2 (1:3)",        t2_s, GOLD)
                + _field("R:R",             rr_s, GOLD)
                + _field("Tidsram",         "Swing (dagar till veckor)")
                + _field(pos_label,         sh_s, pos_color),
                unsafe_allow_html=True,
            )

        with right:
            # Trend gates
            st.markdown(_section_label("Trendgates"), unsafe_allow_html=True)
            st.markdown(
                "".join(_gate_row(g.name, g.passed, g.detail, g.is_blocker)
                        for g in r.trend_gates),
                unsafe_allow_html=True,
            )

            # Entry gates
            st.markdown(_section_label("Entrygates"), unsafe_allow_html=True)
            st.markdown(
                "".join(_gate_row(g.name, g.passed, g.detail, g.is_blocker)
                        for g in r.entry_gates),
                unsafe_allow_html=True,
            )

            # No-trade flags (active ones only)
            active_flags = [f for f in r.notrade_flags if f.passed]
            if active_flags:
                st.markdown(_section_label("Ingen-handel-flaggor"), unsafe_allow_html=True)
                st.markdown(
                    "".join(_gate_row(f.name, False, f.detail, True)
                            for f in active_flags),
                    unsafe_allow_html=True,
                )

            # Macro score breakdown
            if r.macro:
                st.markdown(
                    _section_label(f"Makroscore — {r.macro.total:.0f}/100"),
                    unsafe_allow_html=True,
                )
                for label, score, detail in [
                    ("Copper/Gold", r.macro.copper_gold, r.macro.copper_gold_detail),
                    ("DXY",         r.macro.dxy,         r.macro.dxy_detail),
                    ("Räntekurva",  r.macro.yield_curve, r.macro.yield_curve_detail),
                    ("Cykelfas",    r.macro.cycle,        r.macro.cycle_detail),
                ]:
                    bar_c = GREEN if score >= 50 else RED
                    st.markdown(
                        f"<div style='margin-bottom:7px;'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:0.72rem;'>"
                        f"<span style='color:{TEXT};'>{label}</span>"
                        f"<span style='color:{bar_c};font-weight:700;'>{score:.0f}</span>"
                        f"</div><div style='color:{DIM};font-size:0.62rem;'>{detail}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Sentiment
            if r.sentiment:
                if r.sentiment.is_data_gap:
                    st.markdown(
                        f"<div style='color:{DIM};font-size:0.7rem;margin-top:8px;'>"
                        f"⚠ Sentiment: DATA_GAP<br/>"
                        f"<span style='font-size:0.62rem;'>{r.sentiment.data_gap_reason}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        _section_label(f"Sentimentpoäng — {r.sentiment.total:.0f}/100"),
                        unsafe_allow_html=True,
                    )
                    for lbl, det in [
                        ("Short float",  r.sentiment.short_float_detail),
                        ("Analytiker",   r.sentiment.analyst_detail),
                        ("Put/Call",     r.sentiment.put_call_detail),
                    ]:
                        st.markdown(
                            f"<div style='font-size:0.7rem;margin-bottom:4px;'>"
                            f"<span style='color:{TEXT};'>{lbl}:</span>"
                            f"<span style='color:{DIM};'> {det}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        # ── Discipline fields (session-persisted per ticker) ──────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fd1, fd2, fd3 = st.columns(3)
        with fd1:
            inv = st.text_input(
                "Ogiltigförklaras om",
                value=st.session_state.get(f"ember_inv_{r.ticker}", ""),
                key=f"ember_inv_input_{r.ticker}",
                placeholder="ex: stänger under 50D EMA",
            )
            st.session_state[f"ember_inv_{r.ticker}"] = inv

        with fd2:
            cat = st.text_input(
                "Trolig trigger",
                value=st.session_state.get(f"ember_cat_{r.ticker}", ""),
                key=f"ember_cat_input_{r.ticker}",
                placeholder="ex: FED pivot, Kina-stimulus",
            )
            st.session_state[f"ember_cat_{r.ticker}"] = cat

        with fd3:
            st.markdown(
                f"<div style='color:{DIM};font-size:0.63rem;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:6px;'>Manuell bedömning</div>",
                unsafe_allow_html=True,
            )
            st.checkbox(
                "LME-inventarier sjunker",
                key=f"ember_lme_{r.ticker}",
            )
            st.checkbox(
                "Capex-narrativ försämrad (brist byggs)",
                key=f"ember_capex_{r.ticker}",
            )


# ── Near-miss section ─────────────────────────────────────────────────────────

def _render_near_misses(near_misses: list[EmberSetupResult]) -> None:
    if not near_misses:
        return
    with st.expander(
        f"📋 Nästan-kandidater ({len(near_misses)}) — vad de saknar", expanded=False
    ):
        for r in near_misses[:12]:
            failed_trend  = [g.name for g in r.trend_gates   if not g.passed and g.is_blocker]
            failed_entry  = [g.name for g in r.entry_gates   if not g.passed and g.is_blocker]
            active_notrad = [g.name for g in r.notrade_flags if g.passed and g.is_blocker]
            missing = (failed_trend + failed_entry + active_notrad)[:3]
            missing_str = "; ".join(missing) if missing else "—"
            st.markdown(
                f"<div style='background:{BG2};border-left:2px solid {AMBER}44;"
                f"border-radius:4px;padding:8px 12px;margin-bottom:6px;font-size:0.78rem;'>"
                f"<span style='color:{AMBER};font-weight:700;'>{r.ticker}</span>"
                f"<span style='color:{DIM};'> · {r.sektor}</span>"
                f"<br/><span style='color:{DIM};font-size:0.67rem;'>Saknar: "
                f"<span style='color:{TEXT};'>{missing_str}</span></span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Universe summary (pre-scan) ───────────────────────────────────────────────

def _render_universe_info(source: str) -> None:
    if source == SOURCE_CURATED:
        body = (
            f"<b style='color:{GOLD};'>Universum:</b> "
            f"{len(EMBER_ETF_UNIVERSE)} ETF:er + {len(EMBER_STOCK_UNIVERSE)} aktier<br/>"
            f"<span style='font-size:0.68rem;'>"
            f"ETF:er: {', '.join(EMBER_ETF_UNIVERSE)}<br/>"
            f"Aktier: {', '.join(EMBER_STOCK_UNIVERSE)}"
            f"</span>"
        )
    elif source == SOURCE_AUTO:
        body = (
            f"<b style='color:{GOLD};'>Universum:</b> "
            f"Norden (Börsdata råvarufilter) + {len(US_INTL_CURATED)} US/INTL-tickers<br/>"
            f"<span style='font-size:0.68rem;color:{DIM};'>"
            f"Förfilter tillämpas: omsättning &gt; 5 MSEK/dag + pris &gt; SMA200"
            f"</span>"
        )
    else:  # BOTH
        curated_n = len(EMBER_ETF_UNIVERSE) + len(EMBER_STOCK_UNIVERSE)
        body = (
            f"<b style='color:{GOLD};'>Universum:</b> "
            f"Norden + {len(US_INTL_CURATED)} US/INTL + {curated_n} kurerade<br/>"
            f"<span style='font-size:0.68rem;color:{DIM};'>"
            f"Förfilter tillämpas: omsättning &gt; 5 MSEK/dag + pris &gt; SMA200"
            f"</span>"
        )

    st.markdown(
        f"<div style='margin-top:16px;color:{TEXT};font-size:0.8rem;'>"
        + body
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_universe_stats(stats: Optional[UniverseStats]) -> None:
    """Show universe build stats in the metadata bar after a scan."""
    if stats is None or stats.source == SOURCE_CURATED:
        return
    parts = []
    if stats.nordic_raw > 0:
        parts.append(f"Norden: {stats.nordic_raw} råvarubolag")
    if stats.us_intl_raw > 0:
        parts.append(f"US/INTL: {stats.us_intl_raw} tickers")
    if not stats.borsdata_available and stats.borsdata_error:
        parts.append(f"⚠ Börsdata: {stats.borsdata_error}")
    detail = " · ".join(parts) if parts else ""
    st.markdown(
        f"<div style='font-size:0.7rem;color:{DIM};margin-top:6px;'>"
        f"Universum: <b style='color:{TEXT};'>{stats.total_before_prefilter}</b> tickers "
        f"→ förfilter: <b style='color:{GOLD};'>{stats.passed_prefilter}</b> kvar"
        + (f"<br/>{detail}" if detail else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# ── Main page ─────────────────────────────────────────────────────────────────


def _render_ember_cached_preview() -> None:
    """Show pre-computed scheduled EMBER results (from Gist) as an instant preview."""
    try:
        from ember.cache import load_ember_results
        saved = load_ember_results()
    except Exception:
        saved = {}
    eligible = saved.get("eligible", []) if saved else []
    near = saved.get("near_misses", []) if saved else []
    ts = saved.get("timestamp", "") if saved else ""
    if not eligible and not near:
        return False
    ts_disp = str(ts)[:16].replace("T", " ")
    import streamlit as st
    st.markdown(
        f"<div style='background:#1A1F25;border-left:3px solid #FF6B3D;border-radius:8px;"
        f"padding:10px 14px;margin-bottom:12px;'>"
        f"<b style='color:#E8EDF2;'>Schemalagt EMBER-resultat</b> "
        f"<span style='color:#6B7280;'>&middot; senast uppdaterad {ts_disp} "
        f"&middot; {len(eligible)} elitcase, {len(near)} nastan</span><br>"
        f"<span style='font-size:0.8rem;color:#9aa4b0;'>Tryck <b>SKANNA</b> for live-analys.</span></div>",
        unsafe_allow_html=True,
    )
    import pandas as pd
    rows = []
    for r in (eligible + near)[:30]:
        rows.append({
            "Ticker": r.get("ticker", ""),
            "Typ": r.get("typ", ""),
            "Sektor": r.get("sektor", ""),
            "Elitcase": "JA" if r.get("eligible") else "nastan",
            "Cykel": r.get("cykel_label", ""),
            "Entry": r.get("entry", ""),
            "Stop": r.get("stop", ""),
            "RR": r.get("rr", ""),
            "Asymmetri": round(r.get("asymmetry_score", 0), 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    return True


def render_ember_page() -> None:
    if not _ENGINE_OK:
        st.error("EMBER-motorn ej tillgänglig — kontrollera ember/engine.py")
        return

    # Header
    st.markdown(
        f"<div style='text-align:center;padding:14px 0 10px 0;'>"
        f"<h2 style='color:{EMBER};letter-spacing:0.12em;margin:0;'>🔥 EMBER</h2>"
        f"<p style='color:{DIM};font-size:0.75rem;letter-spacing:0.08em;margin:4px 0 0;'>"
        f"Commodity swing trading · Cykelfiltrar · Trendgates · Makroscore · ATR-riskmodell"
        f"</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<hr style='border-color:{EMBER}33;margin:0 0 16px 0;'>",
        unsafe_allow_html=True,
    )

    # ── Universe source toggle ────────────────────────────────────────────────
    source = st.radio(
        "Universum",
        ALL_SOURCES,
        index=ALL_SOURCES.index(
            st.session_state.get("ember_universe_source", SOURCE_CURATED)
        ),
        horizontal=True,
        key="ember_universe_source",
        help=(
            "Kurerad lista: fast statisk lista (25 ticker, snabb). "
            "Auto: Börsdata Norden + 150+ US/INTL råvaror med förfilter. "
            "Båda: union av alla, förfiltrat."
        ),
    )

    # Control panel
    c1, c2, c3 = st.columns([2, 3, 1])
    with c1:
        account_size = st.number_input(
            "Kontokapital (kr)",
            min_value=10_000, max_value=10_000_000,
            value=st.session_state.get("ember_account_sz", 500_000),
            step=50_000, key="ember_account_sz",
        )
    with c2:
        custom_raw = st.text_area(
            "Extra tickers (kommaseparerade)",
            value=st.session_state.get("ember_extra_tickers", ""),
            height=70, key="ember_extra_tickers",
            placeholder="ex: UUUU, SU, NHY.OL",
        )
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn  = st.button("🔥 SKANNA", key="ember_scan",  type="primary",
                               use_container_width=True)
        clear_btn = st.button("↺ Rensa",   key="ember_clear",
                               use_container_width=True)

    if clear_btn:
        st.session_state.pop("ember_result", None)
        st.rerun()

    if scan_btn:
        extra = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
        spinner_msg = {
            SOURCE_CURATED: "Skannar kurerad lista…",
            SOURCE_AUTO:    "Bygger råvaruuniversum (Norden + US/INTL) och förfiltrerar — ca 1–2 min…",
            SOURCE_BOTH:    "Bygger fullständigt universum och förfiltrerar — ca 2 min…",
        }.get(source, "Skannar…")

        with st.spinner(spinner_msg):
            if extra:
                # Extra tickers provided — add them to whichever universe was built
                from ember.universe import build_universe as _bu
                base_tickers, u_stats = _bu(source, use_prefilter=(source != SOURCE_CURATED))
                tickers = list(dict.fromkeys(base_tickers + extra))
                result  = run_ember_scan(
                    tickers=tickers,
                    account_size=float(account_size),
                )
                result.universe_stats = u_stats
            else:
                result = run_ember_scan(
                    account_size=float(account_size),
                    universe_source=source,
                    use_prefilter=(source != SOURCE_CURATED),
                )
        st.session_state["ember_result"] = result

    result: Optional[EmberScanResult] = st.session_state.get("ember_result")
    if result is None:
        _had_cache = _render_ember_cached_preview()
        if not _had_cache:
            st.markdown(
                f"<div style='text-align:center;padding:40px 0;color:{DIM};'>"
                f"Tryck <b style='color:{EMBER};'>SKANNA</b> för att köra EMBER-analysen."
                f"</div>",
                unsafe_allow_html=True,
            )
        _render_universe_info(source)
        return

    # Metadata bar
    ts    = result.timestamp.strftime("%H:%M:%S")
    tot   = len(result.all_results)
    eli   = len(result.eligible)
    nm    = len(result.near_misses)
    u_stats = result.universe_stats

    # Pre-filter stat: show "X av Y" if we ran a prefilter
    if u_stats is not None and u_stats.source != SOURCE_CURATED:
        prefilter_val = f"{u_stats.passed_prefilter} av {u_stats.total_before_prefilter}"
    else:
        prefilter_val = str(tot)

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (lbl, val, col_) in zip(
        [m1, m2, m3, m4, m5],
        [("UNIVERSUM",   prefilter_val,  GOLD),
         ("SKANNADE",    str(tot),        GOLD),
         ("ELITCASE",    str(eli),        EMBER if eli > 0 else DIM),
         ("NÄSTAN",      str(nm),         AMBER),
         ("UPPDATERAD",  ts,              DIM)],
    ):
        with col:
            st.markdown(_stat_box(lbl, val, col_), unsafe_allow_html=True)

    _render_universe_stats(u_stats)

    # ── Regime status banner ──────────────────────────────────────────────────
    _regime = st.session_state.get("ember_regime")
    if _regime:
        _vbrd = {VERDICT_PA: GREEN, VERDICT_SELEKTIV: AMBER, VERDICT_AV: RED}.get(_regime.verdict, DIM)
        _vico = {VERDICT_PA: "🟢", VERDICT_SELEKTIV: "🟡", VERDICT_AV: "🔴"}.get(_regime.verdict, "○")
        st.markdown(
            f"<div style='background:{_vbrd}11;border:1px solid {_vbrd}44;"
            f"border-radius:6px;padding:9px 16px;margin:10px 0 4px 0;font-size:0.82rem;'>"
            f"<span style='color:{_vbrd};font-weight:700;'>"
            f"{_vico} EMBER REGIME: {_regime.verdict}</span>"
            f"<span style='color:{DIM};'> — {_regime.action_text}</span>"
            f"<span style='color:{DIM};font-size:0.68rem;'>"
            f" (konfigurera i REGIME → 🌍 EMBER Regime)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if not result.eligible:
        st.markdown(
            f"<div style='text-align:center;padding:32px 20px;"
            f"background:{BG2};border:1px solid {EMBER}33;"
            f"border-radius:8px;margin:16px 0;'>"
            f"<h3 style='color:{EMBER};letter-spacing:0.08em;margin-bottom:8px;'>"
            f"INGA ELITCASE just nu</h3>"
            f"<p style='color:{DIM};font-size:0.85rem;margin:0;'>"
            f"Disciplin är position. Vänta på rätt setup — råvarocykeln är tålmodig."
            f"</p></div>",
            unsafe_allow_html=True,
        )
        _render_near_misses(result.near_misses)
        return

    _render_top3(result.eligible)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='border-bottom:1px solid {EMBER}33;padding-bottom:5px;"
        f"margin-bottom:14px;'>"
        f"<h4 style='color:{GOLD};margin:0;'>Alla elitcase — rankade</h4>"
        f"</div>",
        unsafe_allow_html=True,
    )
    for idx, r in enumerate(result.eligible):
        _render_setup_card(r, idx)

    _render_near_misses(result.near_misses)
