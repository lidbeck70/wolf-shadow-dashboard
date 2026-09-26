"""
asymmetry/ui.py — fliken "🐺 Wolf Asymmetry" under GRANSKNING.

Ett fristående verktyg med eget ark (data/asymmetry.json): lägg in ett
bolag, tagga vilken strategi det kompletterar, fyll talen med källa och
datum, och se Commodity Leverage, Margin of Safety, break-even-marginal,
scenarier, stressmatris, thesis killers och datakvalitet. Motorerna är
samma rena funktioner som Confidence-lagret använder, men lagret är eget
— oberoende av Durrett-arket. Saknat underlag är DATA_MISSING, aldrig noll.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

import storage
import storage_ui
from confidence import reports
from confidence import ui as cui
from confidence.data.models import CompanyInput
from ui.components import badge as _badge, confirm_delete, page_header
from ui.tokens import AMBER, DIM, GREEN, RED, TEXT

from asymmetry import ASYMMETRY_CONFIG as CFG, AsymmetryResult, analyze
from asymmetry import charts
from asymmetry import config as acfg
from asymmetry import fetch
from asymmetry import store as ast

SUBS = ("Översikt", "Varför?", "Scenarier", "Stressmatris", "Thesis killers", "Data",
        "Confidence", "Råvaror", "Signaler", "Ark")
_CONF_SUBS = ("Confidence", "Råvaror", "Signaler")     # Confidence-caset, inbyggt (egna KPI:er)
_SEV_COLOR = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": AMBER, "LOW": DIM}
_BAND_COLOR = {acfg.BREAK_EVEN_STRONG: GREEN, acfg.BREAK_EVEN_MODERATE: AMBER, acfg.BREAK_EVEN_WEAK: RED}


def _load() -> dict:
    data = ast.normalize(storage.session_load(ast.STORE, ast.default()))
    st.session_state[ast.STORE] = data
    return data


def _save(data: dict) -> None:
    st.session_state[ast.STORE] = data          # persistensen sker via 💾 Spara


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
    storage_ui.save_bar(ast.STORE, "Wolf Asymmetry", key="save_asymmetry")
    page_header("Wolf Asymmetry", "Hur mycket hävstång mot råvarupriset, hur mycket får gå fel "
                "innan caset spricker, och vad uppsidan är värd efter confidence. Eget ark: "
                "lägg in bolaget, tagga strategin det kompletterar, fyll talen med källa. "
                "DATA_MISSING är aldrig noll.")
    _new_company(data)
    _fetch_sections(data)
    all_tickers = list(ast.companies(data))
    if not all_tickers:
        st.info("Inga bolag i arket än. Lägg in ett under ➕ Nytt bolag.")
        return
    c0, c1, c2 = st.columns([1.2, 1.6, 4])
    with c0:
        used = sorted({ast.strategy(data, t) for t in all_tickers} - {ast.NO_STRATEGY})
        strat = st.selectbox("Strategi", ["Alla"] + used, key="asym_strategy_filter")
    tickers = ast.tickers_for(data, strat) or all_tickers
    with c1:
        last = st.session_state.get("asym_last")
        choice = st.selectbox("Bolag", tickers, key="asym_pick",
                              index=tickers.index(last) if last in tickers else 0,
                              format_func=lambda t: f"{t} · {ast.strategy(data, t)}"
                              if ast.strategy(data, t) != ast.NO_STRATEGY else t)
    st.session_state["asym_last"] = choice
    company = ast.get(data, choice)
    if company is None:
        st.warning("Bolaget hittades inte.")
        return
    with c2:
        sub = st.radio("", list(SUBS), horizontal=True, label_visibility="collapsed", key="asym_sub")
    st.markdown("---")
    if sub == "Ark":
        _sheet(data, company)
        return
    if sub in _CONF_SUBS:
        _confidence(data, company, sub)
        return

    conf = reports.analyze(company, ast.overrides(data), None, date.today())
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


# ── Ark: nytt bolag, strategi, fält ──────────────────────────────────────────
def _new_company(data: dict) -> None:
    with st.expander("➕ Nytt bolag", expanded=not ast.companies(data)):
        with st.form("asym_new"):
            ident = cui.identity_widgets("asym_new", None)
            tags = list(ast.strategy_tags())
            strat = st.selectbox("Komplement till strategi", tags, key="asym_new_strategy",
                                 help="Vilken strategi bolaget granskas för. Filtrerar listan; ändras i Ark.")
            if st.form_submit_button("Lägg till"):
                t = ident["ticker"].strip().upper()
                if not t:
                    st.error("Ticker krävs.")
                elif ast.get(data, t):
                    st.error(f"{t} finns redan i arket.")
                else:
                    ast.put(data, CompanyInput(ticker=t, **{k: v for k, v in ident.items() if k != "ticker"}), strat)
                    _save(data)
                    st.session_state["asym_last"] = t
                    st.session_state["asym_pick"] = t
                    st.session_state["asym_sub"] = "Ark"
                    st.rerun()


def _sheet(data: dict, company: CompanyInput) -> None:
    st.caption("Wolf Asymmetrys eget ark. Talen här påverkar inte Durrett-arket och tvärtom.")
    with st.expander("Identitet och strategi", expanded=False):
        with st.form(f"asym_ident_{company.ticker}"):
            ident = cui.identity_widgets(f"asym_id_{company.ticker}", company)
            tags = list(ast.strategy_tags())
            cur = ast.strategy(data, company.ticker)
            strat = st.selectbox("Komplement till strategi", tags, index=tags.index(cur) if cur in tags else 0,
                                 key=f"asym_strat_{company.ticker}")
            if st.form_submit_button("Uppdatera"):
                for k, v in ident.items():
                    if k != "ticker":
                        setattr(company, k, v)
                ast.put(data, company, strat)
                _save(data)
                st.rerun()
        if confirm_delete("Ta bort bolaget ur Wolf Asymmetry", key=f"asym_del_{company.ticker}"):
            ast.remove(data, company.ticker)
            _save(data)
            st.session_state.pop("asym_last", None)
            st.rerun()
    _render_refresh(data, company)
    cui.render_prefill(data, company, store=ast.STORE)
    cui.render_extractor(data, company, store=ast.STORE, sheet_label="Wolf Asymmetry")
    cui.render_inputs(data, company, tools=False, store=ast.STORE, identity=False)


# ── Hämta: registret och Durrett-arket (bolagsskal / kopior) ─────────────────
def _fetch_sections(data: dict) -> None:
    c1, c2 = st.columns(2)
    with c1:
        _fetch_register(data)
    with c2:
        _fetch_durrett(data)


def _fetch_register(data: dict) -> None:
    with st.expander("📥 Hämta från registret (Holdings)", expanded=False):
        try:
            import positions
            rows = positions.open_positions()
        except Exception as exc:                          # pragma: no cover
            st.caption(f"Kunde inte läsa registret: {exc}")
            return
        cands = fetch.register_candidates(rows, data)
        if not cands:
            st.caption("Inga öppna positioner som saknas i arket.")
            return
        st.caption("Bolagsskal med ticker, namn och strategi-tagg. Råvara, stage och talen fyller du i Ark "
                   "— eller låter Börsdata och extraktorn göra det.")
        for t, name, strat in cands:
            a, b = st.columns([4, 1])
            a.markdown(f"<span style='color:{TEXT};'>{t}</span> <span style='color:{DIM};'>{name}</span> "
                       f"{_badge(strat, DIM)}", unsafe_allow_html=True)
            if b.button("Lägg in", key=f"asym_reg_{t}"):
                fetch.add_from_register(data, t, name, strat)
                _save(data)
                st.session_state["asym_last"] = t
                st.session_state["asym_pick"] = t
                st.rerun()
        if len(cands) > 1 and st.button(f"Lägg in alla ({len(cands)})", key="asym_reg_all"):
            for t, name, strat in cands:
                fetch.add_from_register(data, t, name, strat)
            _save(data)
            st.rerun()


def _fetch_durrett(data: dict) -> None:
    with st.expander("📥 Hämta från Durrett-arket (kopia)", expanded=False):
        try:
            from confidence import store as cs
            conf = cs.normalize(storage.session_load(cs.STORE, cs.default()))
        except Exception as exc:                          # pragma: no cover
            st.caption(f"Kunde inte läsa Durrett-arket: {exc}")
            return
        cands = fetch.durrett_candidates(conf, data)
        if not cands:
            st.caption("Inget i Durrett-arket som saknas här.")
            return
        st.caption("Oberoende kopia med alla fält och källor. Ändringar efteråt påverkar inte Durrett-arket.")
        for t, name, stage in cands:
            a, b = st.columns([4, 1])
            a.markdown(f"<span style='color:{TEXT};'>{t}</span> <span style='color:{DIM};'>{name} · {stage}</span>",
                       unsafe_allow_html=True)
            if b.button("Kopiera", key=f"asym_dur_{t}"):
                fetch.copy_from_durrett(conf, data, t)
                _save(data)
                st.session_state["asym_last"] = t
                st.session_state["asym_pick"] = t
                st.rerun()


# ── Börsdata-förslag ur sifferuppdateringen ──────────────────────────────────
def _render_refresh(data: dict, company: CompanyInput) -> None:
    try:
        import refresh_ui
        from confidence import config as ccfg
    except Exception:                                     # pragma: no cover
        return
    blob = refresh_ui.load_refresh()
    props = fetch.refresh_proposals(blob, company)
    t = company.ticker
    if not props:
        if fetch.refresh_row(blob, t):
            st.caption("🤖 Börsdata: sifferuppdateringen har inga nya tal för raden.")
        else:
            st.caption("🤖 Börsdata: inga tal än — sifferuppdateringen (sheets_refresh.py) läser arket "
                       "schemalagt när det är sparat.")
        return
    asof = props[0][1].pub_date or ""
    with st.expander(f"🤖 Börsdata {asof}: {len(props)} förslag ur sifferuppdateringen", expanded=True):
        for key, point, cur in props:
            label = ccfg.FIELD_BY_KEY[key].label
            a, b = st.columns([4, 1])
            val = point.value if not isinstance(point.value, (int, float)) else f"{point.value:,.4g}"
            a.markdown(f"<span style='color:{TEXT};'>{label}: {val}{(' ' + point.unit) if point.unit else ''}</span>"
                       + (f" <span style='color:{AMBER};font-size:0.78rem;'>ersätter {cur:,.4g}</span>"
                          if isinstance(cur, (int, float)) else
                          (f" <span style='color:{AMBER};font-size:0.78rem;'>ersätter {cur}</span>" if cur else ""))
                       + (f"<br><span style='color:{DIM};font-size:0.74rem;'>{point.note}</span>" if point.note else ""),
                       unsafe_allow_html=True)
            if b.button("Använd", key=f"asym_rf_{t}_{key}"):
                company.set(key, point)
                ast.put(data, company)
                _save(data)
                st.rerun()
        if st.button("Använd alla", key=f"asym_rf_all_{t}"):
            for key, point, _cur in props:
                company.set(key, point)
            ast.put(data, company)
            _save(data)
            st.rerun()


# ── Confidence-caset, inbyggt ────────────────────────────────────────────────
def _confidence(data: dict, company: CompanyInput, sub: str) -> None:
    """Case Score, Confidence Score, Thesis Killer, scenarier, Why Now (Confidence),
    råvaruöverstyrningar (Råvaror) och signalerna (Signaler) — samma vyer som
    Confidence-caset hade, räknade på Wolf Asymmetrys ark."""
    if sub == "Confidence":
        st.caption("Case Score 0–100 (hur bra är caset) och Confidence Score 0–100 (hur säkra är vi), "
                   "med källa per poäng. Tomt = DATA_MISSING, aldrig 0.")
        cui.render_analysis(data, company)
    elif sub == "Råvaror":
        cui.render_commodities(data, company, store=ast.STORE)
    else:
        cui.render_signals(company)


__all__ = ["render_asymmetry_page", "SUBS"]
