"""
asymmetry/ui.py — fliken "🐺 Wolf Asymmetry" under GRANSKNING.

Läser samma ark som Durrett och Confidence-caset (data/confidence.json)
och visar Commodity Leverage, Margin of Safety, break-even-marginal,
scenarier, stressmatris, thesis killers och datakvalitet för ett bolag.
Inga egna inmatningar: fyll arket i Durrett 10-steg eller Confidence-case.
Varje tal har en "Varför?"-förklaring; saknat underlag är DATA_MISSING.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

import storage_ui
from confidence import reports
from confidence import store as cs
from confidence import ui as cui
from confidence.data.models import CompanyInput
from ui.components import badge as _badge, page_header
from ui.tokens import AMBER, DIM, GREEN, RED, TEXT

from asymmetry import ASYMMETRY_CONFIG as CFG, AsymmetryResult, analyze
from asymmetry import charts
from asymmetry import config as acfg

SUBS = ("Översikt", "Varför?", "Scenarier", "Stressmatris", "Thesis killers", "Data")
_SEV_COLOR = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": AMBER, "LOW": DIM}
_BAND_COLOR = {acfg.BREAK_EVEN_STRONG: GREEN, acfg.BREAK_EVEN_MODERATE: AMBER, acfg.BREAK_EVEN_WEAK: RED}


def _load() -> dict:
    fn = getattr(cui, "load_store", None) or getattr(cui, "_load")
    return fn()


def _f(v, fmt: str = "{:,.0f}", na: str = "DATA_MISSING") -> str:
    return na if v is None else fmt.format(v)


def _pct(v, na: str = "DATA_MISSING") -> str:
    return na if v is None else f"{v:+.0f} %"


def _chart(fig, key: str) -> None:
    """Ritar figuren, eller säger varför den saknas — aldrig ett tomt diagram med nollor."""
    if fig is None:
        st.caption("Diagrammet kan inte ritas: underlag saknas (DATA_MISSING).")
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def _score_color(score: Optional[float], maximum: float) -> str:
    if score is None:
        return DIM
    share = score / maximum if maximum else 0
    return GREEN if share >= 0.7 else AMBER if share >= 0.4 else RED


# ── sidan ────────────────────────────────────────────────────────────────────
def render_asymmetry_page() -> None:
    data = _load()
    storage_ui.save_bar(cs.STORE, "Wolf Asymmetry", key="save_asymmetry")
    page_header("Wolf Asymmetry", "Hur mycket hävstång mot råvarupriset, hur mycket får gå fel "
                "innan caset spricker, och vad uppsidan är värd efter confidence. Läser "
                "arket från Durrett 10-steg / Confidence-case — inget matas in här. "
                "DATA_MISSING är aldrig noll.")
    tickers = list(cs.companies(data))
    if not tickers:
        st.info("Inga bolag i arket än. Lägg in ett bolag under GRANSKNING → 🧭 Durrett & Confidence.")
        return
    c1, c2 = st.columns([2, 4])
    with c1:
        last = st.session_state.get("asym_last")
        choice = st.selectbox("Bolag", tickers, key="asym_pick",
                              index=tickers.index(last) if last in tickers else 0)
    st.session_state["asym_last"] = choice
    company = cs.get(data, choice)
    if company is None:
        st.warning("Bolaget hittades inte.")
        return
    with c2:
        sub = st.radio("", list(SUBS), horizontal=True, label_visibility="collapsed", key="asym_sub")
    st.markdown("---")

    conf = reports.analyze(company, cs.overrides(data), None, date.today())
    r = analyze(company, conf.confidence.total)
    _kpis(company, r, conf)
    if sub == "Översikt":
        _overview(r)
    elif sub == "Varför?":
        _why(r)
    elif sub == "Scenarier":
        _scenarios(r, conf)
    elif sub == "Stressmatris":
        _matrix(r)
    elif sub == "Thesis killers":
        _killers(company, r, conf)
    else:
        _data_quality(r, conf)


# ── KPI-raden ────────────────────────────────────────────────────────────────
def _kpis(company: CompanyInput, r: AsymmetryResult, conf: reports.Analysis) -> None:
    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>"
        f"<span style='color:{TEXT};font-size:1.1rem;font-weight:700;'>{company.ticker} · {company.name}</span>"
        f"{_badge(company.stage.upper(), DIM)}{_badge(company.commodity, DIM)}"
        f"{_badge(conf.recommendation, DIM)}</div>", unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("Commodity Leverage", r.leverage.label,
                f"{r.leverage.metric} {r.leverage.response_pct:+.0f} % vid +{r.leverage.probe_pct:g} % pris"
                if r.leverage.response_pct is not None else "DATA_MISSING")
    m[1].metric("Margin of Safety", r.safety.label, "fem delar om 0–2")
    m[2].metric("Break-even-marginal", _pct(r.break_even.margin_pct), r.break_even.band)
    m[3].metric("Confidence", f"{conf.confidence.total:g}", conf.confidence.band)
    m[4].metric("Uppsida (Base)", _pct(r.base_upside_pct), "mot börsvärde")
    m[5].metric("Justerad uppsida", _pct(r.adjusted_upside_pct), "× confidence/100")
    for flag in r.leverage.flags:
        st.markdown(_badge(flag, RED), unsafe_allow_html=True)
    if r.missing:
        st.caption("⚠ Saknade fält (DATA_MISSING): " + ", ".join(r.missing[:12])
                   + (" …" if len(r.missing) > 12 else ""))


# ── Översikt ─────────────────────────────────────────────────────────────────
def _grid_frame(r: AsymmetryResult) -> pd.DataFrame:
    rows = []
    for p in r.leverage.grid:
        rows.append({"Pris %": f"{p.price_pct:+g}", "Pris": _f(p.price, "{:,.4g}", "–"),
                     "Intäkt MUSD": _f(p.revenue_musd, na="–"), "EBITDA MUSD": _f(p.ebitda_musd, na="–"),
                     "FCF MUSD": _f(p.fcf_musd, na="–"), "Marginal %": _f(p.margin_pct, "{:.0f}", "–"),
                     "Equity MUSD": _f(p.equity_musd, na="–"), "Uppsida %": _f(p.upside_pct, "{:+.0f}", "–")})
    return pd.DataFrame(rows)


def _overview(r: AsymmetryResult) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Prisgrid")
        st.caption("Intäkt, EBITDA, FCF och equity per prissteg. Poängen mäts vid "
                   f"+{r.leverage.probe_pct:g} %, nedsidan kontrolleras vid {CFG['commodity_leverage']['downside_probe_pct']:+g} %.")
        _chart(charts.price_grid_chart(r), f"asym_ch_grid_{r.ticker}")
        st.dataframe(_grid_frame(r), hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Margin of Safety")
        _chart(charts.safety_chart(r), f"asym_ch_mos_{r.ticker}")
        for c in r.safety.components:
            if c.not_applicable:
                st.markdown(f"{_badge('EJ TILLÄMPLIGT', DIM)} {c.label}", unsafe_allow_html=True)
            elif c.points is None:
                st.markdown(f"{_badge('DATA_MISSING', DIM)} {c.label}", unsafe_allow_html=True)
            else:
                st.markdown(f"{_badge(f'{c.points:g}/{c.max:g}', _score_color(c.points, c.max))} {c.label}",
                            unsafe_allow_html=True)
        st.markdown("#### Break-even")
        st.markdown(f"{_badge(r.break_even.band, _BAND_COLOR.get(r.break_even.band, DIM))} "
                    f"pris {_f(r.break_even.price, '{:,.4g}', '–')} · break-even {_f(r.break_even.break_even, '{:,.4g}', '–')} "
                    f"· marginal {_pct(r.break_even.margin_pct)}", unsafe_allow_html=True)


# ── Varför? ──────────────────────────────────────────────────────────────────
def _steps(steps: list) -> None:
    for s in steps:
        st.caption("· " + s)


def _why(r: AsymmetryResult) -> None:
    st.caption("Varje poäng med sina steg och den tabell som gav den. Trösklarna ligger i asymmetry/config.py.")
    with st.expander(f"Commodity Leverage {r.leverage.label}", expanded=True):
        _steps(r.leverage.steps)
        for flag in r.leverage.flags:
            st.markdown(_badge(flag, RED), unsafe_allow_html=True)
    for c in r.safety.components:
        head = ("ej tillämpligt" if c.not_applicable else "DATA_MISSING" if c.points is None
                else f"{c.points:g}/{c.max:g}")
        with st.expander(f"{c.label} — {head}"):
            _steps(c.steps)
    with st.expander(f"Break-even-marginal — {r.break_even.band}"):
        _steps(r.break_even.steps)
    with st.expander("Justerad uppsida", expanded=True):
        st.caption(f"· Base-uppsida {_pct(r.base_upside_pct)} × confidence "
                   f"{_f(r.confidence, '{:g}')}/100 = {_pct(r.adjusted_upside_pct)} "
                   f"(formel: {CFG['adjusted_upside']['formula']})")
        _chart(charts.adjusted_upside_chart(r), f"asym_ch_adj_{r.ticker}")


# ── Scenarier ────────────────────────────────────────────────────────────────
def _scenarios(r: AsymmetryResult, conf: reports.Analysis) -> None:
    st.caption("Bear / Base / Bull / Super Bull med samma punkt-modell som prisgriden "
               "(pris- och capex-stegen ur confidence.config.SCENARIOS).")
    _chart(charts.scenario_chart(r), f"asym_ch_scen_{r.ticker}")
    rows = []
    for s in r.scenarios:
        rows.append({"Scenario": s.label, "Pris %": f"{s.price_change_pct:+g}", "CapEx %": f"{s.capex_change_pct:+g}",
                     "Pris": _f(s.price, "{:,.4g}", "–"), "EBITDA MUSD": _f(s.ebitda_musd, na="–"),
                     "FCF MUSD": _f(s.fcf_musd, na="–"), "Värde MUSD": _f(s.value_musd, na="–"),
                     "Equity MUSD": _f(s.equity_musd, na="–"), "Aktie": _f(s.share_price, "{:,.3g}", "–"),
                     "Uppsida %": _f(s.upside_pct, "{:+.0f}", "–")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    asym = r.asymmetry
    c1, c2 = st.columns(2)
    if asym:
        c1.metric("Asymmetri (Bull / |Bear|)",
                  f"{asym.ratio:.1f}×" if asym.ratio not in (None, float("inf")) else "∞", asym.band)
        c2.metric("Sannolikhetsvägd uppsida", _pct(asym.expected_pct),
                  " · ".join(f"{k} {v:.0f} %" for k, v in asym.probs.items()) if asym.probs else "")
    else:
        c1.metric("Asymmetri (Bull / |Bear|)", "DATA_MISSING", "Bear eller Bull kan inte räknas")
    for p in conf.scenarios.paths:
        with st.expander(f"{p.target:g}× — vad krävs?"):
            st.caption(f"· råvarupris {_f(p.required_price, '{:,.4g}', '–')} ({_pct(p.price_change_pct, '–')})"
                       + (f" · produktion × {p.production_factor:.1f}" if p.production_factor else "")
                       + (f" · P/NAV {p.required_p_nav:.1f}×" if p.required_p_nav else ""))
            _steps(p.steps)
    for s in r.scenarios:
        with st.expander(f"{s.label}: steg", expanded=False):
            _steps(s.steps)


# ── Stressmatris ─────────────────────────────────────────────────────────────
def _matrix_frame(r: AsymmetryResult, attr: str = "upside_pct") -> pd.DataFrame:
    table = {}
    for cp in r.matrix.capex_pct:
        col = {}
        for pp in r.matrix.price_pct:
            p = r.matrix.cell(pp, cp)
            v = getattr(p, attr) if p else None
            col[f"pris {pp:+g} %"] = "–" if v is None else (f"{v:+.0f} %" if attr == "upside_pct" else f"{v:,.0f}")
        table[f"capex {cp:+g} %"] = col
    return pd.DataFrame(table)


def _matrix(r: AsymmetryResult) -> None:
    st.caption("Uppsida mot börsvärde när pris och capex rör sig samtidigt. Rader = pris, kolumner = capex.")
    if r.matrix.note:
        st.caption("ℹ " + r.matrix.note)
    _chart(charts.matrix_chart(r), f"asym_ch_matrix_{r.ticker}")
    st.markdown("#### Uppsida %")
    st.dataframe(_matrix_frame(r, "upside_pct"), use_container_width=True)
    st.markdown("#### Equity MUSD")
    st.dataframe(_matrix_frame(r, "equity_musd"), use_container_width=True)
    worst = r.matrix.cell(min(r.matrix.price_pct), max(r.matrix.capex_pct))
    if worst and worst.upside_pct is not None:
        st.caption(f"Värsta rutan (pris {min(r.matrix.price_pct):+g} %, capex {max(r.matrix.capex_pct):+g} %): "
                   f"uppsida {worst.upside_pct:+.0f} %.")
        with st.expander("Steg för värsta rutan"):
            _steps(worst.steps)
    else:
        st.caption("Matrisen kan inte räknas: " + (", ".join(r.missing) if r.missing else "underlag saknas") + ".")


# ── Thesis killers ───────────────────────────────────────────────────────────
def _durrett_flags(company: CompanyInput) -> list:
    try:
        from engines.durrett.engine import analyze as durrett_analyze
        a = durrett_analyze(company)
        return [f for f in a.red_flags if f.severity in ("CRITICAL", "HIGH")]
    except Exception:            # motorn saknas eller faller — fliken ska ändå ritas
        return []


def _killers(company: CompanyInput, r: AsymmetryResult, conf: reports.Analysis) -> None:
    st.caption("Vad som dödar caset: risker ur Confidence-caset, röda flaggor ur Durrett och "
               "nedsidesflaggorna ur hävstångsanalysen.")
    for flag in r.leverage.flags:
        st.markdown(f"{_badge('HÄVSTÅNG', RED)} {flag}", unsafe_allow_html=True)
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        risks = [x for x in conf.risks if x.level == sev]
        for x in risks:
            st.markdown(f"{_badge(sev, _SEV_COLOR[sev])} **{x.name}** "
                        f"<span style='color:{DIM};font-size:0.8rem;'>{x.why}</span>", unsafe_allow_html=True)
    flags = _durrett_flags(company)
    if flags:
        st.markdown("#### Durrett red flags")
        for f in flags:
            st.markdown(f"{_badge(f.severity, _SEV_COLOR[f.severity])} **{f.flag}** "
                        f"<span style='color:{DIM};font-size:0.8rem;'>{f.reason}</span>", unsafe_allow_html=True)
    if not conf.risks and not flags and not r.leverage.flags:
        st.caption("Inga thesis killers i det som är angivet — vilket oftast betyder att underlaget är tunt.")


# ── Data ─────────────────────────────────────────────────────────────────────
def _data_quality(r: AsymmetryResult, conf: reports.Analysis) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown(f"#### Confidence {conf.confidence.total:g} — "
                    f"{_badge(conf.confidence.band, DIM)}", unsafe_allow_html=True)
        for p in conf.confidence.parts:
            st.caption(f"· {p.label}: {p.points:g}/{p.max:g}")
        for key, cap, text in conf.confidence.caps_applied:
            st.markdown(f"{_badge(f'TAK {cap:g}', AMBER)} {text}", unsafe_allow_html=True)
        if r.missing:
            st.markdown("#### Saknade fält")
            for k in r.missing:
                st.caption(f"· {k} — DATA_MISSING")
    with right:
        st.markdown("#### Antaganden i beräkningen")
        for a in r.assumptions:
            st.caption("· " + a.text())


__all__ = ["render_asymmetry_page", "SUBS"]
