"""
engines/durrett/ui.py — fliken "🐺 DURRETT ANALYSIS" under GRANSKNING (SPEC §31–35, §41).

Delar samma lager som Confidence score (data/confidence.json): ett bolag
matas in en gång. Överst: bolag, ticker, råvara, typ, börsvärde, FD-börs-
värde, EV. Sedan dashboarden (DURRETT SCORE · RISK SCORE · UPSIDE ·
CONFIDENCE), scorecard med text-staplar och siffror, red flags per
allvar, Bear/Base/Bull med spårbara antaganden och justerbara parametrar,
developer-checklista / explorer-profil, investeringstes, katalysatorer,
peer-jämförelse sida vid sida (ingen ranking) och logg.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Optional

import streamlit as st

import storage_ui
from confidence import store as cs
from confidence import ui as cui
from engines.durrett import config as dc
from engines.durrett.engine import analyze, to_engine_result
from engines.durrett.models import DurrettAnalysis, Score, to_jsonable

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"
_SEV_COLOR = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": AMBER, "LOW": DIM}
_LEVEL_COLOR = {"LOW": GREEN, "MODERATE": AMBER, "ELEVATED": AMBER, "HIGH": RED, "UNKNOWN": DIM}
_SCEN_KEY = "durrett_scenario_overrides"


def _f(v, fmt="{:g}", na="N/A") -> str:
    return na if v is None else fmt.format(v)


def _bar(value: Optional[float], width: int = 10) -> str:
    if value is None:
        return "░" * width
    n = int(round(value / 100 * width))
    return "█" * n + "░" * (width - n)


def _badge(text: str, color: str) -> str:
    return (f"<span style='background:{color}22;color:{color};border:1px solid {color};"
            f"border-radius:4px;padding:2px 8px;font-size:0.78rem;font-weight:700;'>{text}</span>")


# ── sidan ────────────────────────────────────────────────────────────────────
def render_durrett_page() -> None:
    data = cui.load_store()
    storage_ui.save_bar(cs.STORE, "Confidence score / Durrett", key="save_durrett")
    st.markdown(
        f"<div style='text-align:center;padding:10px 0 4px;'>"
        f"<h2 style='color:{GOLD};letter-spacing:0.12em;margin:0;'>🐺 DURRETT ANALYSIS</h2>"
        f"<p style='color:{DIM};font-size:0.78rem;margin:6px 0 0;'>Don Durretts 10-stegsmetod. Alla poäng 0–100 "
        f"(50 = neutralt), N/A när det inte går att räkna. Risk Score 100 = lägst risk. Confidence är inte "
        f"attraktivitet. Ingen köp- eller säljrekommendation.</p></div>", unsafe_allow_html=True)

    tickers = list(cs.companies(data))
    c1, c2 = st.columns([2, 3])
    with c1:
        choice = st.selectbox("Bolag", ["➕ Nytt bolag"] + tickers, key="durrett_pick",
                              index=(tickers.index(st.session_state.get("durrett_last")) + 1
                                     if st.session_state.get("durrett_last") in tickers else (1 if tickers else 0)))
    if choice == "➕ Nytt bolag":
        cui.new_company_form(data)
        return
    st.session_state["durrett_last"] = choice
    company = cs.get(data, choice)
    if company is None:
        st.warning("Bolaget hittades inte.")
        return
    with c2:
        sub = st.radio("", ["Analys", "Scenarier", "Indata", "Katalysatorer", "Peers"], horizontal=True,
                       label_visibility="collapsed", key="durrett_sub")
    st.markdown("---")
    overrides = st.session_state.get(_SCEN_KEY, {}).get(company.ticker)
    a = analyze(company, scenario_overrides=overrides, today=date.today())
    if sub == "Analys":
        _render_analysis(a)
    elif sub == "Scenarier":
        _render_scenarios(a, company)
    elif sub == "Indata":
        st.caption("Samma fält som Confidence score — Durrett-grupperna ligger sist. Fyll aktiestruktur, "
                   "reserver/resurser, produktion och management för full analys.")
        _render_momentum_fetch(data, company)
        cui.render_inputs(data, company)
    elif sub == "Katalysatorer":
        _render_catalysts(data, company, a)
    else:
        _render_peers(data)


# ── Analys ───────────────────────────────────────────────────────────────────
def _render_analysis(a: DurrettAnalysis) -> None:
    cl = a.classification
    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>"
        f"<span style='color:{TEXT};font-size:1.1rem;font-weight:700;'>{a.name or a.ticker} · {a.ticker}</span>"
        f"<span style='color:{DIM};'>{a.commodity}</span>{_badge(cl.company_type, GOLD)}"
        f"<span style='color:{DIM};font-size:0.78rem;'>klassificering {cl.confidence}</span></div>",
        unsafe_allow_html=True)
    with st.expander("Varför denna typ?", expanded=cl.company_type == dc.UNKNOWN):
        for r in cl.reasons:
            st.caption("· " + r)
    m = st.columns(3)
    m[0].metric("Market Cap", _f(a.market_cap_musd, "{:,.0f} MUSD"))
    m[1].metric("FD Market Cap", _f(a.fd_market_cap_musd, "{:,.0f} MUSD"))
    m[2].metric("Enterprise Value", _f(a.enterprise_value_musd, "{:,.0f} MUSD"),
                _f(a.fd_enterprise_value_musd, "FD {:,.0f}", ""))

    d = st.columns(4)
    d[0].metric("DURRETT SCORE", _f(a.quality_score.value), a.quality_score.reason or "Quality 0–100")
    d[1].metric("RISK SCORE", _f(a.risk_score.value), f"Risk Level: {a.risk_level} (100 = lägst risk)")
    d[2].metric("UPSIDE", _f(a.upside_multiple, "{:.1f}×"), _f(a.potential_upside_pct, "{:+.0f} % (Base)", "Base N/A"))
    d[3].metric("CONFIDENCE", _f(a.confidence_score.value, "{:g} %"), _f(a.data_confidence, "data {:g}", ""))
    if a.missing_data:
        st.caption(f"⚠ {len(a.missing_data)} fält saknas: " + ", ".join(a.missing_data[:14])
                   + (" …" if len(a.missing_data) > 14 else ""))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Scorecard")
        rows = []
        for s in a.scores():
            rows.append(f"{s.label:<14} {_bar(s.value)}  {_f(s.value, '{:>5.0f}', '  N/A')}")
        rows.append(f"{'Quality':<14} {_bar(a.quality_score.value)}  {_f(a.quality_score.value, '{:>5.0f}', '  N/A')}")
        rows.append(f"{'Risk':<14} {_bar(a.risk_score.value)}  {_f(a.risk_score.value, '{:>5.0f}', '  N/A')}")
        rows.append(f"{'Confidence':<14} {_bar(a.confidence_score.value)}  {_f(a.confidence_score.value, '{:>5.0f}', '  N/A')}")
        st.code("\n".join(rows), language=None)
        with st.expander("Förklara en poäng (explain_score)", expanded=False):
            keys = [s.key for s in a.scores()] + ["quality", "risk", "confidence"]
            k = st.selectbox("Poäng", keys, key=f"durrett_explain_{a.ticker}")
            st.code(a.explain_score(k), language=None)
    with right:
        st.markdown("#### 🚨 Red flags")
        if not a.red_flags:
            st.caption("Inga red flags i det som är angivet.")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            fl = [f for f in a.red_flags if f.severity == sev]
            if not fl:
                continue
            st.markdown(f"{_badge(sev, _SEV_COLOR[sev])}", unsafe_allow_html=True)
            for f in fl:
                src = f" <span style='color:{DIM};font-size:0.74rem;'>({f.source}{', ' + f.date if f.date else ''})</span>" if f.source else ""
                st.markdown(f"<span style='color:{TEXT};font-size:0.86rem;'>• <b>{f.flag}</b> — {f.reason}{src}</span>",
                            unsafe_allow_html=True)

    st.markdown("#### Bear / Base / Bull")
    cols = st.columns(3)
    for col, case in zip(cols, (a.bear_case, a.base_case, a.bull_case)):
        with col:
            if case is None or not case.available:
                st.metric(case.label if case else "—", "N/A", (case.reason if case else "") or "Insufficient data")
                continue
            st.metric(case.label, _f(case.fair_value_per_share, "{:,.2f}/aktie", "FV N/A (aktier saknas)"),
                      f"Upside {case.upside_pct:+.0f} % · {case.commodity_price:g} · {case.multiple:g}×")
    st.caption("Alla siffror spåras till antagandena under Scenarier.")

    if a.developer_checklist:
        ck = a.developer_checklist
        st.markdown(f"#### {ck['label']}")
        for name, ok, why in ck["items"]:
            mark = "✅" if ok else ("❌" if ok is False else "❔")
            st.caption(f"{mark} {name} — {why}")
    if a.explorer_profile:
        p = a.explorer_profile
        st.markdown(f"#### Explorer: {p['play']} · Lassonde {p['lassonde_label']} ({p['lassonde_source']})")
        st.caption(f"Optionality Score {_f(p['optionality_score'])} — {p['optionality']['reason']}")
        st.caption(f"Discovery Score {_f(p['discovery_score'])} — {p['discovery']['reason']}")

    st.markdown("#### Investment thesis")
    th = a.thesis
    t1, t2 = st.columns(2)
    for col, keys in ((t1, ("why_it_could_work", "key_catalysts", "what_must_go_right", "next_milestones")),
                      (t2, ("why_it_could_fail", "key_risks", "what_would_invalidate"))):
        with col:
            for k in keys:
                st.markdown(f"**{k.replace('_', ' ').upper()}**")
                for line in th.get(k, []):
                    st.caption("· " + line)
    st.caption(th.get("note", ""))

    with st.expander("Nyckeltal och logg", expanded=False):
        st.json({k: v for k, v in a.metrics.items()})
        st.code("\n".join(a.log), language=None)
    st.download_button("⬇ EngineResult (JSON)", json.dumps(to_engine_result(a).as_dict(), ensure_ascii=False, indent=1),
                       file_name=f"durrett_{a.ticker}_{a.generated}.json", mime="application/json", key="durrett_dl")


# ── Momentum ur kurshistorik ─────────────────────────────────────────────────
def _render_momentum_fetch(data: dict, company) -> None:
    from engines.durrett import momentum_fetch as mf

    with st.expander("📈 Hämta momentum ur kurshistorik (yfinance)", expanded=False):
        st.caption("Räknar kurs mot MA200, 6-månadersutveckling och volymtrend och skriver in dem som "
                   "datapunkter med källa yfinance och dagens datum. RS-rank, sektor-/råvarumomentum och "
                   "nyhetsflöde sätter du själv.")
        c1, c2 = st.columns([2, 1])
        yft = c1.text_input("Yahoo-ticker", value=st.session_state.get(f"durrett_yft_{company.ticker}", company.ticker),
                            key=f"durrett_yft_in_{company.ticker}", help="T.ex. ABC.TO, ABC.V, ABC.AX, ABC")
        if c2.button("Hämta", key=f"durrett_mom_{company.ticker}"):
            st.session_state[f"durrett_yft_{company.ticker}"] = yft
            with st.spinner("Hämtar två års kurshistorik …"):
                pts, msg = mf.fetch_momentum(yft.strip())
            if not pts:
                st.error(msg)
                return
            for k, p in pts.items():
                company.set(k, p)
            cs.put(data, company)
            cui.save_store(data)
            st.success(msg + " — inskrivna i sessionen, spara med 💾")
            st.rerun()
        for k in ("price_vs_ma200_pct", "momentum_6m_pct", "volume_trend"):
            p = company.get(k)
            if p is not None and not p.missing:
                st.caption(f"· {k}: {p.value:g} ({p.kind}, {p.source}, {p.pub_date})")


# ── Scenarier ────────────────────────────────────────────────────────────────
def _render_scenarios(a: DurrettAnalysis, company) -> None:
    st.caption("Justera per scenario. Tomt = default ur config (syns som ASSUMPTION). Ändringar gäller sessionen.")
    ov_all = st.session_state.setdefault(_SCEN_KEY, {})
    cur = ov_all.get(company.ticker, {})
    with st.form(f"durrett_scen_{company.ticker}"):
        cols = st.columns(3)
        new: dict = {}
        for col, key, case in zip(cols, ("bear", "base", "bull"), (a.bear_case, a.base_case, a.bull_case)):
            with col:
                st.markdown(f"**{case.label}**")
                o = cur.get(key, {})
                vals = {}
                for pk, label in (("commodity_price", f"Råvarupris ({a.commodity})"), ("production", "Produktion"),
                                  ("aisc", "AISC"), ("capex_musd", "CapEx MUSD"), ("multiple", "Multipel"),
                                  ("recovery_pct", "Recovery %"), ("mine_life_years", "Gruvliv år"),
                                  ("fx_to_usd", "FX → USD")):
                    default = getattr(case, pk, None) if pk in ("commodity_price", "production", "aisc", "capex_musd", "multiple") else None
                    vals[pk] = st.text_input(label, value=("" if pk not in o else f"{o[pk]:g}"),
                                             placeholder=("" if default is None else f"{default:,.4g}"),
                                             key=f"ds_{company.ticker}_{key}_{pk}")
                new[key] = vals
        if st.form_submit_button("Räkna om"):
            parsed = {}
            for key, vals in new.items():
                d = {}
                for pk, raw in vals.items():
                    try:
                        if str(raw).strip():
                            d[pk] = float(str(raw).replace(",", ".").replace(" ", ""))
                    except ValueError:
                        st.error(f"{key} {pk}: inte ett tal ({raw!r})")
                if d:
                    parsed[key] = d
            ov_all[company.ticker] = parsed
            st.rerun()
    for case in (a.bear_case, a.base_case, a.bull_case):
        with st.expander(f"{case.label}: " + (f"fair value {_f(case.fair_value_per_share, '{:,.2f}')} · upside {case.upside_pct:+.0f} %"
                                              if case.available else f"N/A — {case.reason}"), expanded=case.key == "base"):
            for x in case.assumptions:
                st.caption("A · " + x.text())
            for s in case.steps:
                st.caption("· " + s)


# ── Katalysatorer ────────────────────────────────────────────────────────────
def _render_catalysts(data: dict, company, a: DurrettAnalysis) -> None:
    if a.catalysts:
        st.dataframe([c.as_dict() for c in a.catalysts], hide_index=True, use_container_width=True)
    else:
        st.caption("Inga katalysatorer registrerade.")
    with st.form(f"durrett_cat_{company.ticker}"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Händelse")
        typ = c2.selectbox("Typ", list(dc.CATALYST_TYPES))
        exp = c3.text_input("Väntat (ÅÅÅÅ-MM eller ÅÅÅÅ-Qn)")
        c4, c5, c6 = st.columns(3)
        imp = c4.selectbox("Vikt", list(dc.CATALYST_IMPORTANCE), index=1)
        conf = c5.selectbox("Konfidens", ["high", "medium", "low"], index=1)
        src = c6.text_input("Källa")
        impact = st.text_input("Påverkar (resurs, värdering, risk, upside, confidence …)")
        if st.form_submit_button("Lägg till") and name.strip():
            company.catalysts.append({"name": name.strip(), "type": typ, "expected": exp.strip(), "importance": imp,
                                      "impact": impact.strip(), "confidence": conf, "source": src.strip()})
            cs.put(data, company)
            cui.save_store(data)
            st.rerun()
    if a.catalysts and st.button("Ta bort sista katalysatorn", key=f"durrett_cat_del_{company.ticker}"):
        company.catalysts = company.catalysts[:-1]
        cs.put(data, company)
        cui.save_store(data)
        st.rerun()


# ── Peers ────────────────────────────────────────────────────────────────────
_PEER_ROWS = (("Typ", lambda a: a.classification.company_type), ("Market Cap MUSD", lambda a: a.market_cap_musd),
              ("EV MUSD", lambda a: a.enterprise_value_musd),
              ("Resurs", lambda a: (a.metrics.get("resource_total") or {}).get("value")),
              ("Reserv", lambda a: (a.metrics.get("reserve_total") or {}).get("value")),
              ("Produktion", lambda a: (a.metrics.get("production_growth_multiple") or {}).get("note")),
              ("AISC", lambda a: (a.metrics.get("operating_margin_per_unit") or {}).get("note")),
              ("NPV/CAPEX", lambda a: (a.metrics.get("npv_capex") or {}).get("value")),
              ("EV/oz resurs", lambda a: (a.metrics.get("ev_per_resource_unit") or {}).get("value")),
              ("EV/NPV", lambda a: (a.metrics.get("ev_npv") or {}).get("value")),
              ("Growth", lambda a: a.growth_score.value), ("Dilution", lambda a: a.dilution_score.value),
              ("Risk", lambda a: a.risk_score.value), ("Durrett Score", lambda a: a.quality_score.value),
              ("Upside ×", lambda a: a.upside_multiple), ("Confidence", lambda a: a.confidence_score.value))


def _render_peers(data: dict) -> None:
    tickers = list(cs.companies(data))
    pick = st.multiselect("Bolag att jämföra (sida vid sida — ingen ranking)", tickers,
                          default=tickers[:4], key="durrett_peers")
    if not pick:
        return
    analyses = {t: analyze(cs.get(data, t), today=date.today()) for t in pick}
    table = []
    for label, fn in _PEER_ROWS:
        row = {"Nyckeltal": label}
        for t, a in analyses.items():
            v = fn(a)
            row[t] = "N/A" if v is None else (f"{v:,.4g}" if isinstance(v, (int, float)) else str(v))
        table.append(row)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("N/A = kunde inte beräknas (otillräcklig data). Jämförelsen rankar inte.")


def analysis_json(a: DurrettAnalysis) -> dict:
    return to_jsonable(a)


__all__ = ["render_durrett_page", "analysis_json", "Score"]
