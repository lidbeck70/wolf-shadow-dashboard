"""
asymmetry/ui.py — fliken "🐺 Wolf Asymmetry" under GRANSKNING. En sida.

Samma ark som Durrett (data/confidence.json), en strategi-tagg per bolag så
fliken fungerar som komplement till vilken strategi som helst. Två lägen:

  Analys   KPI-raden → verdikt och thesis killers → prisgrid + margin of
           safety → scenarier + stressmatris → stegen bakom talen (hopfällt)
  Ark      nytt bolag, hämta från registret, Börsdata-förslag, extraktor,
           fälten, råvaror, signaler — allt som är inmatning

Motorerna är asymmetry/ och confidence/. DATA_MISSING är aldrig noll.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

import storage
import storage_ui
from confidence import commodities as com
from confidence import config as ccfg
from confidence import reports
from confidence import ui as cui
from confidence.data.models import CompanyInput
from confidence.store import companies as _companies, get as _get, overrides as _overrides, put as _put, remove as _remove
from ui.components import badge as _badge, confirm_delete, page_header
from ui.tokens import AMBER, DIM, GREEN, RED, TEXT

from asymmetry import ASYMMETRY_CONFIG as CFG, AsymmetryResult, analyze
from asymmetry import charts
from asymmetry import config as acfg
from asymmetry import fetch
from asymmetry import store as ast

MODES = ("Analys", "Ark")
_SEV_COLOR = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": AMBER, "LOW": DIM}
_BAND_COLOR = {acfg.BREAK_EVEN_STRONG: GREEN, acfg.BREAK_EVEN_MODERATE: AMBER, acfg.BREAK_EVEN_WEAK: RED}
_REC_COLOR = {"BUY CANDIDATE": GREEN, "WATCH": AMBER, "PASS": DIM, "REJECT": RED}


# ── lagret ───────────────────────────────────────────────────────────────────
def _load() -> dict:
    data = ast.normalize(storage.session_load(ast.STORE, cs_default()))
    legacy = storage.session_load(ast.LEGACY_STORE, None)
    if legacy and not st.session_state.get("asym_migrated"):
        moved = ast.merge_legacy(data, legacy)
        st.session_state["asym_migrated"] = True
        if moved:
            st.info(f"Flyttade {', '.join(moved)} från det gamla Wolf Asymmetry-arket in i det gemensamma arket. "
                    "Spara med 💾 så är flytten klar.")
    st.session_state[ast.STORE] = data
    return data


def cs_default() -> dict:
    from confidence import store as cs
    return cs.default()


def _save(data: dict) -> None:
    st.session_state[ast.STORE] = data          # persistensen sker via 💾 Spara


def _f(v, fmt: str = "{:,.0f}", na: str = "DATA_MISSING") -> str:
    return na if v is None else fmt.format(v)


def _pct(v, na: str = "DATA_MISSING") -> str:
    return na if v is None else f"{v:+.0f} %"


def _chart(fig, key: str) -> None:
    if fig is None:
        st.caption("Diagrammet kan inte ritas: underlag saknas (DATA_MISSING).")
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def _steps(steps: list) -> None:
    for s in steps:
        st.caption("· " + s)


def _goto(ticker: str) -> None:
    """Öppna ett bolag på nästa körning. Väljaren är redan ritad när knappen
    trycks, så dess nyckel får inte skrivas nu — den sätts överst nästa gång."""
    st.session_state["asym_goto"] = ticker


def _apply_goto() -> None:
    t = st.session_state.pop("asym_goto", None)
    if t:
        st.session_state["asym_pick"] = t
        st.session_state["asym_last"] = t
        st.session_state["asym_strategy_filter"] = "Alla"


# ── sidan ────────────────────────────────────────────────────────────────────
def render_asymmetry_page() -> None:
    data = _load()
    _apply_goto()
    storage_ui.save_bar(ast.STORE, "Wolf Asymmetry", key="save_asymmetry")
    page_header("Wolf Asymmetry", "Hur mycket hävstång mot råvarupriset, hur mycket får gå fel innan "
                "caset spricker, och vad uppsidan är värd efter confidence. Samma ark som Durrett; "
                "taggen säger vilken strategi bolaget kompletterar. DATA_MISSING är aldrig noll.")
    all_tickers = list(_companies(data))
    c0, c1, c2 = st.columns([1.3, 2, 1.5])
    with c0:
        used = sorted({ast.strategy(data, t) for t in all_tickers} - {ast.NO_STRATEGY})
        strat = st.selectbox("Strategi", ["Alla"] + used, key="asym_strategy_filter")
    tickers = ast.tickers_for(data, strat) or all_tickers
    with c1:
        last = st.session_state.get("asym_last")
        choice = st.selectbox("Bolag", tickers or ["—"], key="asym_pick",
                              index=(tickers.index(last) if last in tickers else 0),
                              format_func=lambda t: f"{t} · {ast.strategy(data, t)}"
                              if t in tickers and ast.strategy(data, t) != ast.NO_STRATEGY else t)
    with c2:
        mode = st.radio("", list(MODES), horizontal=True, label_visibility="collapsed", key="asym_mode",
                        index=0 if all_tickers else 1)
    st.markdown("---")
    company = _get(data, choice) if all_tickers else None
    if company is not None:
        st.session_state["asym_last"] = company.ticker
    if mode == "Ark" or company is None:
        if company is None:
            st.info("Inga bolag i arket än. Lägg in ett nedan, eller hämta från registret.")
        _sheet(data, company)
        return
    _analysis(data, company)


# ── Analys: en sida i läsordning ─────────────────────────────────────────────
def _analysis(data: dict, company: CompanyInput) -> None:
    signals = getattr(cui, "_signals_for", None)
    sig = signals(com.get(company.commodity, _overrides(data))) if signals else None
    conf = reports.analyze(company, _overrides(data), sig, date.today())
    r = analyze(company, conf.confidence.total)

    # 1. vem
    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>"
        f"<span style='color:{TEXT};font-size:1.1rem;font-weight:700;'>{company.ticker} · {company.name}</span>"
        f"{_badge(company.stage.upper(), DIM)}{_badge(company.commodity, DIM)}"
        f"{_badge(ast.strategy(data, company.ticker), DIM) if ast.strategy(data, company.ticker) != ast.NO_STRATEGY else ''}"
        f"</div>", unsafe_allow_html=True)

    # 2. KPI-raden
    asym = r.asymmetry
    m = st.columns(6)
    m[0].metric("Commodity Leverage", r.leverage.label,
                f"{r.leverage.metric} {r.leverage.response_pct:+.0f} % vid +{r.leverage.probe_pct:g} % pris"
                if r.leverage.response_pct is not None else "DATA_MISSING")
    m[1].metric("Margin of Safety", r.safety.label, "fem delar om 0–2")
    m[2].metric("Break-even-marginal", _pct(r.break_even.margin_pct), r.break_even.band)
    m[3].metric("Confidence", f"{conf.confidence.total:g}", conf.confidence.band)
    m[4].metric("Asymmetri", f"{asym.ratio:.1f}×" if asym and asym.ratio not in (None, float("inf")) else
                ("∞" if asym else "DATA_MISSING"), asym.band if asym else "Bear/Bull saknas")
    m[5].metric("Justerad uppsida", _pct(r.adjusted_upside_pct),
                f"Base {_pct(r.base_upside_pct)} × {conf.confidence.total:g}/100")

    # 3. verdikt och thesis killers
    st.markdown(f"{_badge(conf.recommendation, _REC_COLOR.get(conf.recommendation, DIM))} "
                f"<span style='color:{DIM};font-size:0.84rem;'>{conf.recommendation_why}</span>",
                unsafe_allow_html=True)
    for _k, cap, text in conf.confidence.caps_applied:
        st.markdown(f"{_badge(f'TAK {cap:g}', AMBER)} <span style='color:{DIM};font-size:0.84rem;'>{text}</span>",
                    unsafe_allow_html=True)
    for flag in r.leverage.flags:
        st.markdown(f"{_badge('HÄVSTÅNG', RED)} <span style='color:{TEXT};font-size:0.9rem;'>{flag}</span>",
                    unsafe_allow_html=True)
    killers = [x for x in conf.risks if x.level in ("CRITICAL", "HIGH")]
    for x in killers:
        st.markdown(f"{_badge(x.level, _SEV_COLOR[x.level])} <b style='color:{TEXT};'>{x.name}</b> "
                    f"<span style='color:{DIM};font-size:0.84rem;'>{x.why}</span>", unsafe_allow_html=True)
    if r.missing:
        st.caption("⚠ DATA_MISSING: " + ", ".join(r.missing[:12]) + (" …" if len(r.missing) > 12 else "")
                   + " — fyll i under Ark.")

    # 4. diagrammen
    a, b = st.columns([3, 2])
    with a:
        _chart(charts.price_grid_chart(r), f"asym_ch_grid_{r.ticker}")
    with b:
        _chart(charts.safety_chart(r), f"asym_ch_mos_{r.ticker}")
    a, b = st.columns(2)
    with a:
        _chart(charts.scenario_chart(r), f"asym_ch_scen_{r.ticker}")
    with b:
        _chart(charts.matrix_chart(r), f"asym_ch_matrix_{r.ticker}")
        if r.matrix.note:
            st.caption("ℹ " + r.matrix.note)

    # 5. stegen bakom talen — hopfällt
    with st.expander("Varför? — stegen bakom varje tal", expanded=False):
        st.markdown(f"**Commodity Leverage {r.leverage.label}**")
        _steps(r.leverage.steps)
        for c in r.safety.components:
            head = ("ej tillämpligt" if c.not_applicable else "DATA_MISSING" if c.points is None
                    else f"{c.points:g}/{c.max:g}")
            st.markdown(f"**{c.label} — {head}**")
            _steps(c.steps)
        st.markdown(f"**Break-even-marginal — {r.break_even.band}**")
        _steps(r.break_even.steps)
        st.markdown("**Justerad uppsida**")
        st.caption(f"· Base {_pct(r.base_upside_pct)} × confidence {conf.confidence.total:g}/100 = "
                   f"{_pct(r.adjusted_upside_pct)} (formel: {CFG['adjusted_upside']['formula']})")
        _chart(charts.adjusted_upside_chart(r), f"asym_ch_adj_{r.ticker}")
        st.markdown("**Scenarier**")
        for s in r.scenarios:
            st.caption(f"{s.label}: " + " · ".join(s.steps[:3]))
        st.markdown("**Antaganden**")
        for x in r.assumptions:
            st.caption("· " + x.text())

    with st.expander(f"Confidence-caset — Case Score {conf.case.total:g} ({conf.case.rating}) · "
                     f"Confidence {conf.confidence.total:g} ({conf.confidence.band})", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**Case Score — pelarna**")
            for p in conf.case.pillars:
                st.caption(f"· {p.label}: {p.points:g}/{p.max:g}" + (f" — {p.notes[0]}" if p.notes else ""))
            st.caption(f"Why Now {conf.why_now.score:g} ({conf.why_now.band}) · regional knapphet "
                       f"{conf.regional.score:g} ({conf.regional.band}) · time-to-money {conf.ttm.years:g} år")
        with right:
            st.markdown("**Confidence — delarna**")
            for p in conf.confidence.parts:
                st.caption(f"· {p.label}: {p.points:g}/{p.max:g}")
            for f in conf.confidence.flags:
                st.caption(("🔴 " if f.startswith("KILL") else "• ") + f)
        rest = [x for x in conf.risks if x.level not in ("CRITICAL", "HIGH")]
        if rest:
            st.markdown("**Övriga risker**")
            for x in rest:
                st.caption(f"· {x.level}: {x.name} — {x.why}")
        for p in conf.scenarios.paths:
            st.caption(f"{p.target:g}×: " + " · ".join(p.steps[:2]))

    with st.expander("Tabeller — prisgrid, scenarier, stressmatris", expanded=False):
        st.dataframe(_grid_frame(r), hide_index=True, use_container_width=True)
        st.dataframe(_scenario_frame(r), hide_index=True, use_container_width=True)
        st.dataframe(_matrix_frame(r), use_container_width=True)


def _grid_frame(r: AsymmetryResult) -> pd.DataFrame:
    return pd.DataFrame([{"Pris %": f"{p.price_pct:+g}", "Pris": _f(p.price, "{:,.4g}", "–"),
                          "Intäkt MUSD": _f(p.revenue_musd, na="–"), "EBITDA MUSD": _f(p.ebitda_musd, na="–"),
                          "FCF MUSD": _f(p.fcf_musd, na="–"), "Marginal %": _f(p.margin_pct, "{:.0f}", "–"),
                          "Equity MUSD": _f(p.equity_musd, na="–"), "Uppsida %": _f(p.upside_pct, "{:+.0f}", "–")}
                         for p in r.leverage.grid])


def _scenario_frame(r: AsymmetryResult) -> pd.DataFrame:
    return pd.DataFrame([{"Scenario": s.label, "Pris %": f"{s.price_change_pct:+g}", "CapEx %": f"{s.capex_change_pct:+g}",
                          "EBITDA MUSD": _f(s.ebitda_musd, na="–"), "FCF MUSD": _f(s.fcf_musd, na="–"),
                          "Värde MUSD": _f(s.value_musd, na="–"), "Equity MUSD": _f(s.equity_musd, na="–"),
                          "Aktie": _f(s.share_price, "{:,.3g}", "–"), "Uppsida %": _f(s.upside_pct, "{:+.0f}", "–")}
                         for s in r.scenarios])


def _matrix_frame(r: AsymmetryResult) -> pd.DataFrame:
    table = {}
    for cp in r.matrix.capex_pct:
        col = {}
        for pp in r.matrix.price_pct:
            p = r.matrix.cell(pp, cp)
            col[f"pris {pp:+g} %"] = "–" if p is None or p.upside_pct is None else f"{p.upside_pct:+.0f} %"
        table[f"capex {cp:+g} %"] = col
    return pd.DataFrame(table)


# ── Ark: allt som är inmatning ───────────────────────────────────────────────
def _sheet(data: dict, company: Optional[CompanyInput]) -> None:
    a, b = st.columns(2)
    with a:
        _new_company(data)
    with b:
        _fetch_register(data)
    if company is None:
        return
    st.markdown(f"#### {company.ticker} · {company.name}")
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
                _put(data, company)
                ast.set_strategy(data, company.ticker, strat)
                _save(data)
                st.rerun()
        if confirm_delete("Ta bort bolaget ur arket", key=f"asym_del_{company.ticker}"):
            _remove(data, company.ticker)
            ast.set_strategy(data, company.ticker, ast.NO_STRATEGY)
            _save(data)
            st.session_state.pop("asym_last", None)
            st.rerun()
    _render_refresh(data, company)
    cui.render_prefill(data, company)
    cui.render_extractor(data, company, sheet_label="Wolf Asymmetry")
    cui.render_inputs(data, company, tools=False, identity=False)
    with st.expander("Råvaran — utbud, lager, kostnadskurva (delas av alla bolag i samma råvara)", expanded=False):
        cui.render_commodities(data, company)
    with st.expander("Signaler till Why Now (rotation, teman, kvoter)", expanded=False):
        cui.render_signals(company)


def _new_company(data: dict) -> None:
    with st.expander("➕ Nytt bolag", expanded=not _companies(data)):
        with st.form("asym_new"):
            ident = cui.identity_widgets("asym_new", None)
            tags = list(ast.strategy_tags())
            strat = st.selectbox("Komplement till strategi", tags, key="asym_new_strategy")
            if st.form_submit_button("Lägg till"):
                t = ident["ticker"].strip().upper()
                if not t:
                    st.error("Ticker krävs.")
                elif _get(data, t):
                    st.error(f"{t} finns redan i arket.")
                else:
                    _put(data, CompanyInput(ticker=t, **{k: v for k, v in ident.items() if k != "ticker"}))
                    ast.set_strategy(data, t, strat)
                    _save(data)
                    st.session_state["asym_last"] = t
                    _goto(t)
                    st.rerun()


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
        st.caption("Bolagsskal med ticker, namn och strategi-tagg. Råvara, stage och talen fyller du här "
                   "— eller låter Börsdata och extraktorn göra det.")
        for t, name, strat in cands:
            a, b = st.columns([4, 1])
            a.markdown(f"<span style='color:{TEXT};'>{t}</span> <span style='color:{DIM};'>{name}</span> "
                       f"{_badge(strat, DIM)}", unsafe_allow_html=True)
            if b.button("Lägg in", key=f"asym_reg_{t}"):
                fetch.add_from_register(data, t, name, strat)
                _save(data)
                st.session_state["asym_last"] = t
                _goto(t)
                st.rerun()
        if len(cands) > 1 and st.button(f"Lägg in alla ({len(cands)})", key="asym_reg_all"):
            for t, name, strat in cands:
                fetch.add_from_register(data, t, name, strat)
            _save(data)
            st.rerun()


def _render_refresh(data: dict, company: CompanyInput) -> None:
    try:
        import refresh_ui
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
                _put(data, company)
                _save(data)
                st.rerun()
        if st.button("Använd alla", key=f"asym_rf_all_{t}"):
            for key, point, _cur in props:
                company.set(key, point)
            _put(data, company)
            _save(data)
            st.rerun()


__all__ = ["render_asymmetry_page", "MODES"]
