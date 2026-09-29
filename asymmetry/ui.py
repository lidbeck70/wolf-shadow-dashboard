"""
asymmetry/ui.py — fliken "🐺 Wolf Asymmetry" under GRANSKNING. Tre poäng, ett kort.

  🚀 Commodity Leverage · 🛡️ Margin of Safety · 🎯 Confidence

Analys: de tre poängen med band och en rad varför, en tunn rad (asymmetri,
justerad uppsida, break-even), verdikt och thesis killers, två hopfällda
block (diagram, stegen). Ark: bara de fält poängen läser (asymmetry/fields),
Börsdata-förslag, terminspris, extraktor. Samma ark som Durrett
(data/confidence.json) med en strategi-tagg. DATA_MISSING är aldrig noll.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import streamlit as st

import storage
import storage_ui
from confidence import config as ccfg
from confidence import reports
from confidence import ui as cui
from confidence.data.models import CompanyInput
from confidence.data.provenance import dp
from confidence.store import companies as _companies, get as _get, overrides as _overrides, put as _put, remove as _remove
from ui.components import badge as _badge, confirm_delete, page_header
from ui.tokens import AMBER, DIM, GREEN, RED, TEXT

from asymmetry import ASYMMETRY_CONFIG as CFG, AsymmetryResult, analyze
from asymmetry import charts, fetch, fields
from asymmetry import config as acfg
from asymmetry import store as ast

QUICK = "⚡ Snabbkoll"
MODES = (QUICK, "Analys", "Ark")
_SEV_COLOR = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": AMBER, "LOW": DIM}
_BAND_COLOR = {acfg.BREAK_EVEN_STRONG: GREEN, acfg.BREAK_EVEN_MODERATE: AMBER, acfg.BREAK_EVEN_WEAK: RED}
_REC_COLOR = {"BUY CANDIDATE": GREEN, "WATCH": AMBER, "PASS": DIM, "REJECT": RED}


# ── lagret ───────────────────────────────────────────────────────────────────
def _load() -> dict:
    from confidence import store as cs
    data = ast.normalize(storage.session_load(ast.STORE, cs.default()))
    legacy = storage.session_load(ast.LEGACY_STORE, None)
    if legacy and not st.session_state.get("asym_migrated"):
        moved = ast.merge_legacy(data, legacy)
        st.session_state["asym_migrated"] = True
        if moved:
            st.info(f"Flyttade {', '.join(moved)} från det gamla Wolf Asymmetry-arket in i det gemensamma arket. "
                    "Spara med 💾 så är flytten klar.")
    st.session_state[ast.STORE] = data
    return data


def _save(data: dict) -> None:
    st.session_state[ast.STORE] = data          # persistensen sker via 💾 Spara


def _goto(ticker: str) -> None:
    """Öppna ett bolag på nästa körning (väljaren är redan ritad när knappen trycks)."""
    st.session_state["asym_goto"] = ticker


def _apply_goto() -> None:
    t = st.session_state.pop("asym_goto", None)
    if t:
        st.session_state["asym_pick"] = t
        st.session_state["asym_last"] = t


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


# ── sidan ────────────────────────────────────────────────────────────────────
def render_asymmetry_page() -> None:
    data = _load()
    _apply_goto()
    storage_ui.save_bar(ast.STORE, "Wolf Asymmetry", key="save_asymmetry")
    page_header("Wolf Asymmetry", "⚡ Snabbkoll: skriv en ticker — Survival, Margin of Safety och "
                "Confidence helt automatiskt. Analys och Ark: arkets bolag med Commodity Leverage och "
                "scenarier. DATA_MISSING är aldrig noll.")
    tickers = list(_companies(data))
    mode = st.radio("", list(MODES), horizontal=True, label_visibility="collapsed", key="asym_mode")
    if mode == QUICK:
        from asymmetry.quick_ui import render_quick
        render_quick(add_to_sheet=lambda t, name: _add_quick(data, t, name))
        return
    c1, c2 = st.columns([3, 1.5])
    with c1:
        last = st.session_state.get("asym_last")
        choice = st.selectbox("Bolag", tickers or ["—"], key="asym_pick",
                              index=(tickers.index(last) if last in tickers else 0),
                              format_func=lambda t: f"{t} · {ast.strategy(data, t)}"
                              if t in tickers and ast.strategy(data, t) != ast.NO_STRATEGY else t)
    st.markdown("---")
    company = _get(data, choice) if tickers else None
    if company is not None:
        st.session_state["asym_last"] = company.ticker
    if mode == "Ark" or company is None:
        if company is None:
            st.info("Inga bolag i arket än. Lägg in ett nedan, eller hämta från registret.")
        _sheet(data, company)
        return
    _analysis(data, company)


def _add_quick(data: dict, ticker: str, name: str) -> bool:
    """Snabbkollens bolag in i arket (samma ark som Durrett). False om det redan fanns."""
    t = str(ticker or "").strip().upper()
    if not t or _get(data, t):
        return False
    _put(data, CompanyInput(ticker=t, name=name))
    _save(data)
    return True


# ── Analys: tre poäng, ett kort ──────────────────────────────────────────────
def _analysis(data: dict, company: CompanyInput) -> None:
    conf = reports.analyze(company, _overrides(data), None, date.today())
    r = analyze(company, conf.confidence.total)
    tag = ast.strategy(data, company.ticker)
    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>"
        f"<span style='color:{TEXT};font-size:1.1rem;font-weight:700;'>{company.ticker} · {company.name}</span>"
        f"{_badge(company.stage.upper(), DIM)}{_badge(company.commodity, DIM)}"
        f"{_badge(tag, DIM) if tag != ast.NO_STRATEGY else ''}</div>", unsafe_allow_html=True)

    # de tre poängen
    lev, mos = r.leverage, r.safety
    a, b, c = st.columns(3)
    a.metric("🚀 Commodity Leverage", lev.label,
             f"{lev.metric} {lev.response_pct:+.0f} % vid +{lev.probe_pct:g} % pris"
             if lev.response_pct is not None else "DATA_MISSING")
    b.metric("🛡️ Margin of Safety", mos.label,
             " · ".join(f"{x.label.split(' ')[0]} {x.points:g}" for x in mos.components if x.points is not None)
             or "DATA_MISSING")
    c.metric("🎯 Confidence", f"{conf.confidence.total:g}", conf.confidence.band)
    # den tunna raden
    asym = r.asymmetry
    a, b, c = st.columns(3)
    a.metric("Asymmetri Bull/|Bear|", f"{asym.ratio:.1f}×" if asym and asym.ratio not in (None, float("inf"))
             else ("∞" if asym else "DATA_MISSING"), asym.band if asym else "Bear/Bull saknas")
    b.metric("Justerad uppsida", _pct(r.adjusted_upside_pct),
             f"Base {_pct(r.base_upside_pct)} × {conf.confidence.total:g}/100")
    c.metric("Break-even-marginal", _pct(r.break_even.margin_pct), r.break_even.band)

    # verdikt och thesis killers
    st.markdown(f"{_badge(conf.recommendation, _REC_COLOR.get(conf.recommendation, DIM))} "
                f"<span style='color:{DIM};font-size:0.84rem;'>{conf.recommendation_why}</span>",
                unsafe_allow_html=True)
    for _k, cap, text in conf.confidence.caps_applied:
        st.markdown(f"{_badge(f'TAK {cap:g}', AMBER)} <span style='color:{DIM};font-size:0.84rem;'>{text}</span>",
                    unsafe_allow_html=True)
    for flag in lev.flags:
        st.markdown(f"{_badge('HÄVSTÅNG', RED)} <span style='color:{TEXT};font-size:0.9rem;'>{flag}</span>",
                    unsafe_allow_html=True)
    for x in (x for x in conf.risks if x.level in ("CRITICAL", "HIGH")):
        st.markdown(f"{_badge(x.level, _SEV_COLOR[x.level])} <b style='color:{TEXT};'>{x.name}</b> "
                    f"<span style='color:{DIM};font-size:0.84rem;'>{x.why}</span>", unsafe_allow_html=True)
    missing = [k for k in fields.all_keys(company.stage) if k in set(r.missing) | set(conf.confidence.missing)]
    if missing:
        st.caption("⚠ DATA_MISSING: " + ", ".join(ccfg.FIELD_BY_KEY[k].label for k in missing[:10])
                   + (" …" if len(missing) > 10 else "") + " — fyll i under Ark.")

    with st.expander("Diagram", expanded=False):
        x, y = st.columns([3, 2])
        with x:
            _chart(charts.price_grid_chart(r), f"asym_ch_grid_{r.ticker}")
        with y:
            _chart(charts.safety_chart(r), f"asym_ch_mos_{r.ticker}")
        x, y = st.columns(2)
        with x:
            _chart(charts.scenario_chart(r), f"asym_ch_scen_{r.ticker}")
        with y:
            _chart(charts.matrix_chart(r), f"asym_ch_matrix_{r.ticker}")
            if r.matrix.note:
                st.caption("ℹ " + r.matrix.note)
        _chart(charts.adjusted_upside_chart(r), f"asym_ch_adj_{r.ticker}")

    with st.expander("Varför? — stegen bakom varje tal", expanded=False):
        st.markdown(f"**🚀 Commodity Leverage {lev.label}**")
        _steps(lev.steps)
        st.markdown(f"**🛡️ Margin of Safety {mos.label}**")
        for comp in mos.components:
            head = ("ej tillämpligt" if comp.not_applicable else "DATA_MISSING" if comp.points is None
                    else f"{comp.points:g}/{comp.max:g}")
            st.caption(f"**{comp.label} — {head}**")
            _steps(comp.steps)
        st.markdown(f"**🎯 Confidence {conf.confidence.total:g} — {conf.confidence.band}**")
        for p in conf.confidence.parts:
            st.caption(f"· {p.label}: {p.points:g}/{p.max:g}" + (f" — {p.notes[0]}" if p.notes else ""))
        for f in conf.confidence.flags:
            st.caption(("🔴 " if f.startswith("KILL") else "• ") + f)
        st.markdown(f"**Break-even-marginal — {r.break_even.band}**")
        _steps(r.break_even.steps)
        st.markdown("**Justerad uppsida**")
        st.caption(f"· Base {_pct(r.base_upside_pct)} × confidence {conf.confidence.total:g}/100 = "
                   f"{_pct(r.adjusted_upside_pct)} (formel: {CFG['adjusted_upside']['formula']})")
        st.markdown("**Antaganden**")
        for x in r.assumptions:
            st.caption("· " + x.text())


# ── Ark: bara de fält poängen läser ──────────────────────────────────────────
def _sheet(data: dict, company: Optional[CompanyInput]) -> None:
    a, b = st.columns(2)
    with a:
        _new_company(data)
    with b:
        _fetch_register(data)
    if company is None:
        return
    st.markdown(f"#### {company.ticker}" + (f" · {company.name}" if company.name else ""))
    with st.expander("Identitet och strategi", expanded=False):
        with st.form(f"asym_ident_{company.ticker}"):
            ident = cui.identity_widgets(f"asym_id_{company.ticker}", company)
            tags = list(ast.strategy_tags())
            cur = ast.strategy(data, company.ticker)
            x, y = st.columns(2)
            strat = x.selectbox("Komplement till strategi", tags, index=tags.index(cur) if cur in tags else 0,
                                key=f"asym_strat_{company.ticker}")
            ins = y.text_input("Börsdata-id (ins_id)", value="" if company.ins_id is None else str(company.ins_id),
                               key=f"asym_insid_{company.ticker}",
                               help="Bara om Hämta nu inte hittar bolaget. Id:t står i Börsdatas URL.")
            if st.form_submit_button("Uppdatera"):
                for k, v in ident.items():
                    if k != "ticker":
                        setattr(company, k, v)
                company.ins_id = int(ins) if str(ins).strip().isdigit() else None
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
    _render_auto(data, company)
    _block(data, company, "econ", "Ekonomi — det 🚀 och 🛡️ läser", fields.economy(company.stage), expanded=True)
    _block(data, company, "core", "Confidence — kärnan (tungt vägd)", fields.confidence_core(company.stage))
    _block(data, company, "more", "Confidence — finlir (lätt vägd)", fields.confidence_more(company.stage))
    cui.render_extractor(data, company, sheet_label="Wolf Asymmetry")


def _block(data: dict, company: CompanyInput, key: str, title: str, specs: list, expanded: bool = False) -> None:
    filled = sum(1 for f in specs if company.has(f.key))
    with st.expander(f"{title} · {filled}/{len(specs)} ifyllda", expanded=expanded):
        with st.form(f"asym_{key}_{company.ticker}"):
            widgets = [(cui._field_widgets(company, f), f) for f in specs]
            if st.form_submit_button("Spara"):
                for w, f in widgets:
                    point = cui._point_from_widgets(w, f)
                    if point is None:
                        company.fields.pop(f.key, None)
                    else:
                        company.set(f.key, point)
                _put(data, company)
                _save(data)
                st.rerun()


def _render_auto(data: dict, company: CompanyInput) -> None:
    """Börsdata fyller: värdena som de är, förslagen ur sifferuppdateringen, terminspriset."""
    with st.expander("Börsdata fyller — börsvärde, kurs, aktier, kassa, skuld, råvarupris", expanded=True):
        parts = []
        for f in fields.auto():
            v = company.num(f.key) if company.has(f.key) else None
            parts.append(f"<span style='color:{TEXT};'>{f.label}</span> "
                         f"<span style='color:{DIM if v is None else TEXT};'>{'DATA_MISSING' if v is None else f'{v:,.4g}'}"
                         f"{(' ' + f.unit) if v is not None and f.unit else ''}</span>")
        st.markdown(" · ".join(parts), unsafe_allow_html=True)
        _render_refresh(data, company)
        _render_futures(data, company)
        with st.form(f"asym_auto_{company.ticker}"):
            widgets = [(cui._field_widgets(company, f), f) for f in fields.auto()]
            if st.form_submit_button("Spara ändringar"):
                for w, f in widgets:
                    point = cui._point_from_widgets(w, f)
                    if point is None:
                        company.fields.pop(f.key, None)
                    else:
                        company.set(f.key, point)
                _put(data, company)
                _save(data)
                st.rerun()


def _render_futures(data: dict, company: CompanyInput) -> None:
    name = fields.futures_name(company.commodity)
    cur = company.num("commodity_price") if company.has("commodity_price") else None
    if name is None:
        st.caption(f"Råvarupris: {('DATA_MISSING' if cur is None else f'{cur:,.4g}')} — {company.commodity} "
                   "har ingen termin på Yahoo; skriv in priset under Ekonomi.")
        return
    a, b = st.columns([4, 1])
    a.caption(f"Råvarupris nu: {('DATA_MISSING' if cur is None else f'{cur:,.4g}')} · terminen ({name}) "
              "kan hämtas och skrivs som ASSUMPTION med källa och datum.")
    if b.button("Hämta termin", key=f"asym_fut_{company.ticker}"):
        import commodity_prices
        q = commodity_prices.spot(name)
        if not q:
            st.warning("Kunde inte hämta terminspriset just nu.")
            return
        company.set("commodity_price", dp(q["price"], kind="ASSUMPTION", source=f"Yahoo {q['ticker']} (termin)",
                                          source_type="secondary", pub_date=q["asof"], unit=q["unit"]))
        _put(data, company)
        _save(data)
        st.rerun()


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
        for t, name, strat in cands:
            a, b = st.columns([4, 1])
            a.markdown(f"<span style='color:{TEXT};'>{t}</span> <span style='color:{DIM};'>{name}</span> "
                       f"{_badge(strat, DIM)}", unsafe_allow_html=True)
            if b.button("Lägg in", key=f"asym_reg_{t}"):
                fetch.add_from_register(data, t, name, strat)
                _save(data)
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
    t = company.ticker
    live_key = f"asym_bd_{t}"
    a, b = st.columns([4, 1])
    if b.button("Hämta nu", key=f"asym_bd_go_{t}", help="Börsdata först (nordiskt eller globalt), Yahoo som reserv. Utan att vänta på det nattliga jobbet."):
        with st.spinner("Hämtar från Börsdata …"):
            blob, props, msg = fetch.fetch_now(company)
        if blob is None:
            st.warning(msg)
        else:
            st.session_state[live_key] = blob
            st.rerun()
    blob = st.session_state.get(live_key) or refresh_ui.load_refresh()
    live = live_key in st.session_state
    props = fetch.refresh_proposals(blob, company)
    if not props:
        a.caption("🤖 Börsdata: " + ("inga nya tal för raden." if fetch.refresh_row(blob, t)
                                    else "inga tal än — tryck Hämta nu, eller vänta på det nattliga jobbet."))
        return
    asof = props[0][1].pub_date or ""
    a.caption(f"🤖 Börsdata {asof}: {len(props)} förslag" + (" (hämtade nu)" if live else " ur sifferuppdateringen"))
    for key, point, cur in props:
        label = ccfg.FIELD_BY_KEY[key].label
        a, b = st.columns([4, 1])
        val = point.value if not isinstance(point.value, (int, float)) else f"{point.value:,.4g}"
        a.markdown(f"<span style='color:{TEXT};'>{label}: {val}{(' ' + point.unit) if point.unit else ''}</span>"
                   + (f" <span style='color:{AMBER};font-size:0.78rem;'>ersätter {cur:,.4g}</span>"
                      if isinstance(cur, (int, float)) else
                      (f" <span style='color:{AMBER};font-size:0.78rem;'>ersätter {cur}</span>" if cur else "")),
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
