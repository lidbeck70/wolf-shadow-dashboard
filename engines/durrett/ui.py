"""
engines/durrett/ui.py — fliken "🐺 Durrett" under GRANSKNING, byggd som
granskningsarken (Rick Rule, Poängmodellen, Tiggre):

  ➕ Ny kandidat        ticker, bolag, råvara, typ
  🕸 Håven — Durrett    screens_scan.py:s Durrett-screener (Börsdata) med
                        "Lägg in" per träff — arket äger raden efteråt
  Rader                 ett expander per bolag: Arket (manuell inmatning av
                        Durretts tal), Analys, Scenarier, Mer
  🔬 Jämför             peers sida vid sida, ingen ranking

Lagret är data/confidence.json (delat med Confidence score): ett bolag
matas in en gång. Varje tal lagras som Datapoint med källa och datum.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Optional

import streamlit as st

import storage_ui
from confidence import commodities as com
from confidence import config as ccfg
from confidence import store as cs
from confidence import ui as cui
from confidence.data.models import CompanyInput
from confidence.data.provenance import dp
from engines.durrett import config as dc
from engines.durrett.engine import analyze, to_engine_result
from engines.durrett.models import DurrettAnalysis, Score, to_jsonable

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"
_SEV_COLOR = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": AMBER, "LOW": DIM}
_SCEN_KEY = "durrett_scenario_overrides"
_SCREEN_KEY = "durrett"                 # screens_scan.py / reference.py
_SOURCE_TYPES = ("mixed", "primary", "independent", "secondary", "weak", "unsupported")
_TYPE_PRESET = {"Producent": ("producer", "production"), "Utvecklare": ("developer", "pea"),
                "Explorer": ("explorer", "exploration"), "Royalty/stream": ("royalty", "production")}

# Arkets fält: (nyckel, etikett) i tre spalter. Nycklarna är confidence.config.FIELDS.
_SHEET_MARKET = (("market_cap_musd", "Börsvärde (MUSD)"), ("share_price", "Kurs"), ("basic_shares_m", "Aktier basic (M)"),
                 ("options_m", "Optioner (M)"), ("warrants_m", "Warranter (M)"), ("shares_3y_ago_m", "Aktier 3 år sedan (M)"),
                 ("cash_musd", "Kassa (MUSD)"), ("debt_musd", "Skuld (MUSD)"), ("insider_ownership_pct", "Insynsägande %"),
                 ("revenue_musd", "Omsättning 12 mån (MUSD)"))
_SHEET_OPS = (("commodity_price", "Råvarupris"), ("production_current", "Produktion nu"),
              ("production_future", "Produktion plan"), ("production_future_year", "År för planen"),
              ("aisc", "AISC / kostnad per enhet"), ("mine_life_years", "Gruvliv (år)"),
              ("reserve_proven", "Reserv Proven"), ("reserve_probable", "Reserv Probable"),
              ("resource_measured", "Resurs Measured"), ("resource_indicated", "Resurs Indicated"),
              ("resource_inferred", "Resurs Inferred"))
_SHEET_PROJECT = (("npv_musd", "NPV efter skatt (MUSD)"), ("capex_musd", "Initial CapEx (MUSD)"), ("irr_pct", "IRR %"),
                  ("first_cashflow_year", "Första kassaflöde (år)"), ("committed_financing_musd", "Åtagen finansiering (MUSD)"),
                  ("quarterly_burn_musd", "Burn per kvartal (MUSD)"), ("net_debt_ebitda", "Nettoskuld/EBITDA"),
                  ("recovery_pct", "Recovery %"), ("grade", "Halt"))


def _cui(public: str, private: str):
    """Confidence-flikens funktioner via de publika aliasen, med fallback till de
    privata namnen — så fliken fungerar även när en redan laddad confidence.ui
    (utan aliasen) ligger kvar i processen tills appen startas om."""
    fn = getattr(cui, public, None) or getattr(cui, private, None)
    if fn is None:
        raise RuntimeError(f"confidence.ui saknar {public}/{private} — starta om appen (Manage app → Reboot)")
    return fn


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


def _num_or_none(s) -> Optional[float]:
    s = str(s or "").strip().replace(",", ".").replace(" ", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load() -> dict:
    return _cui("load_store", "_load")()


def _save(data: dict) -> None:
    _cui("save_store", "_save")(data)


# ── sidan ────────────────────────────────────────────────────────────────────
def render_durrett_page() -> None:
    data = _load()
    storage_ui.save_bar(cs.STORE, "Durrett / Confidence score", key="save_durrett")
    st.markdown(
        f"<div style='text-align:center;padding:10px 0 4px;'>"
        f"<h2 style='color:{GOLD};letter-spacing:0.12em;margin:0;'>🐺 DURRETT ANALYSIS</h2>"
        f"<p style='color:{DIM};font-size:0.78rem;margin:6px 0 0;'>Don Durretts 10-stegsmetod. Håven håvar in "
        f"screenerns träffar, arket tar dina tal. Alla poäng 0–100 (50 = neutralt), N/A när det inte går att "
        f"räkna. Risk Score 100 = lägst risk. Ingen köp- eller säljrekommendation.</p></div>",
        unsafe_allow_html=True)
    _new_candidate(data)
    _screen_section(data)
    _rows(data)
    _peers_section(data)


# ── ny kandidat ──────────────────────────────────────────────────────────────
def _new_candidate(data: dict) -> None:
    keys = com.all_keys()
    with st.expander("➕ Ny kandidat", expanded=not cs.companies(data)):
        c1, c2, c3, c4 = st.columns([1, 2, 1.2, 1.2])
        ticker = c1.text_input("Ticker", key="durrett_new_t")
        name = c2.text_input("Bolag", key="durrett_new_n")
        commodity = c3.selectbox("Råvara", keys, index=keys.index("gold"),
                                 format_func=lambda k: com.REGISTRY[k].label, key="durrett_new_c")
        ctype = c4.selectbox("Typ", list(_TYPE_PRESET), key="durrett_new_type")
        if st.button("Lägg till", key="durrett_new_add"):
            t = ticker.strip().upper()
            if not t:
                st.warning("Ticker krävs.")
            elif cs.get(data, t):
                st.warning(f"{t} finns redan i arket.")
            else:
                stage, maturity = _TYPE_PRESET[ctype]
                cs.put(data, CompanyInput(ticker=t, name=name.strip(), commodity=commodity, stage=stage,
                                          maturity=maturity))
                _save(data)
                st.rerun()


# ── håven ────────────────────────────────────────────────────────────────────
def _company_from_screen(fields: dict, generated: str) -> CompanyInput:
    """Håvträff → bolag. Guld om inte namnet säger silver; producent (screenern kräver omsättning)."""
    name = fields.get("name", "")
    commodity = "silver" if "silver" in name.lower() else "gold"
    c = CompanyInput(ticker=fields["ticker"], name=name, commodity=commodity, stage="producer",
                     maturity="production", ins_id=fields.get("ins_id"))
    if fields.get("mcap") is not None:
        c.set("market_cap_musd", dp(float(fields["mcap"]), kind="ACTUAL", source="Börsdata-håven (Durrett)",
                                    source_type="secondary", pub_date=(generated or None), unit="MUSD"))
    return c


def _screen_section(data: dict) -> None:
    try:
        import screens_ui
    except Exception:
        return
    existing = set(cs.companies(data))
    blob = screens_ui.load_screens()
    generated = str((blob or {}).get("generated") or "")[:10]

    def _add(fields: dict) -> None:
        cs.put(data, _company_from_screen(fields, generated))
        _save(data)

    screens_ui.render_screen_section(_SCREEN_KEY, existing, _add, key_prefix="durrett")


# ── raderna ──────────────────────────────────────────────────────────────────
def _rows(data: dict) -> None:
    companies = cs.companies(data)
    if not companies:
        st.caption("Inga kandidater ännu. Håven fylls av den schemalagda Durrett-screenern i Börsdata "
                   "(kriterierna står i RULES → 📚 SNABBREFERENS); ➕ Ny kandidat för manuell inläggning.")
        return
    overrides_all = st.session_state.get(_SCEN_KEY, {})
    analyses = {t: analyze(c, scenario_overrides=overrides_all.get(t), today=date.today())
                for t, c in companies.items()}
    order = sorted(companies, key=lambda t: (-(analyses[t].quality_score.value
                                              if analyses[t].quality_score.value is not None else -1), t))
    for t in order:
        company, a = companies[t], analyses[t]
        head = (f"{t} · {com.REGISTRY[company.commodity].label if company.commodity in com.REGISTRY else company.commodity}"
                f" · {a.classification.company_type} · Durrett {_f(a.quality_score.value, '{:.0f}', '–')}"
                f" · Risk {_f(a.risk_score.value, '{:.0f}', '–')} {a.risk_level}"
                f" · Upside {_f(a.upside_multiple, '{:.1f}×', '–')} · Confidence {_f(a.confidence_score.value, '{:.0f} %', '–')}")
        with st.expander(head, expanded=False):
            sub = st.radio("", ["Arket", "Analys", "Scenarier", "Mer"], horizontal=True,
                           label_visibility="collapsed", key=f"durrett_row_sub_{t}")
            if sub == "Arket":
                _render_sheet(data, company, a)
            elif sub == "Analys":
                _render_analysis(a)
            elif sub == "Scenarier":
                _render_scenarios(a, company)
            else:
                _render_more(data, company, a)


def _sheet_widget(company: CompanyInput, key: str, label: str, wkey: str):
    spec = ccfg.FIELD_BY_KEY[key]
    p = company.get(key)
    cur = "" if p is None or p.missing else (str(p.value) if not isinstance(p.value, (int, float)) else f"{p.value:g}")
    if spec.kind == "choice":
        opts = ["—"] + list(spec.choices)
        return st.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0, key=wkey, help=spec.hint or None)
    return st.text_input(label, value=cur, key=wkey, placeholder=spec.unit or "", help=spec.hint or None)


def _render_sheet(data: dict, company: CompanyInput, a: DurrettAnalysis) -> None:
    t = company.ticker
    st.markdown(f"<b style='color:{TEXT};'>Arket</b> <span style='color:{DIM};font-size:0.78rem;'>— Durretts tal. "
                f"Tomt = okänt (N/A), aldrig 0. Enheten för resurser/produktion väljs här; kostnad och pris i "
                f"råvarans prisenhet ({a.commodity and com.REGISTRY.get(company.commodity).unit if company.commodity in com.REGISTRY else 'USD/enhet'}).</span>",
                unsafe_allow_html=True)
    with st.form(f"durrett_sheet_{t}"):
        h1, h2, h3, h4 = st.columns([2, 1.2, 1.2, 1])
        source = h1.text_input("Källa för det du fyller i nu", value="Arket (manuell)", key=f"ds_src_{t}",
                               help="T.ex. 'MD&A Q2 2026' eller '43-101 2025'. Skrivs på varje ändrat fält.")
        stype = h2.selectbox("Källtyp", list(_SOURCE_TYPES), key=f"ds_stype_{t}")
        kind = h3.selectbox("Datatyp", list(ccfg.KINDS), index=1, key=f"ds_kind_{t}")
        when = h4.text_input("Datum", value=date.today().isoformat(), key=f"ds_date_{t}")
        cols = st.columns(3)
        widgets = {}
        with cols[0]:
            st.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>MARKNAD & AKTIER</span>", unsafe_allow_html=True)
            widgets["market_currency"] = _sheet_widget(company, "market_currency", "Valuta", f"ds_{t}_market_currency")
            widgets["fx_to_usd"] = _sheet_widget(company, "fx_to_usd", "Växelkurs → USD", f"ds_{t}_fx_to_usd")
            for k, lbl in _SHEET_MARKET:
                widgets[k] = _sheet_widget(company, k, lbl, f"ds_{t}_{k}")
        with cols[1]:
            st.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>DRIFT, RESERVER & RESURSER</span>", unsafe_allow_html=True)
            widgets["resource_unit"] = _sheet_widget(company, "resource_unit", "Enhet (resurs/produktion)", f"ds_{t}_resource_unit")
            for k, lbl in _SHEET_OPS:
                widgets[k] = _sheet_widget(company, k, lbl, f"ds_{t}_{k}")
        with cols[2]:
            st.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>PROJEKT & FINANSIERING</span>", unsafe_allow_html=True)
            for k, lbl in _SHEET_PROJECT:
                widgets[k] = _sheet_widget(company, k, lbl, f"ds_{t}_{k}")
            widgets["mgmt_track_verified"] = _sheet_widget(company, "mgmt_track_verified", "Management track record",
                                                           f"ds_{t}_mgmt_track_verified")
            widgets["team_mines_built"] = _sheet_widget(company, "team_mines_built", "Gruvor teamet byggt", f"ds_{t}_team_mines_built")
        if st.form_submit_button("Uppdatera arket"):
            changed = _apply_sheet(company, widgets, source.strip() or "Arket (manuell)", stype, kind, when.strip() or None)
            cs.put(data, company)
            _save(data)
            st.success(f"{changed} fält uppdaterade — spara med 💾 när du är klar.")
            st.rerun()
    _sheet_summary(a)


def _apply_sheet(company: CompanyInput, widgets: dict, source: str, stype: str, kind: str, when: Optional[str]) -> int:
    changed = 0
    for key, raw in widgets.items():
        spec = ccfg.FIELD_BY_KEY[key]
        cur = company.get(key)
        if spec.kind == "choice":
            if raw == "—":
                if cur is not None:
                    company.fields.pop(key, None)
                    changed += 1
                continue
            if cur is not None and not cur.missing and str(cur.value) == raw:
                continue
            company.set(key, dp(raw, kind=kind, source=source, source_type=stype, pub_date=when))
            changed += 1
            continue
        if not str(raw).strip():
            if cur is not None and not cur.missing:
                company.fields.pop(key, None)
                changed += 1
            continue
        v = _num_or_none(raw)
        if v is None:
            continue                                   # valideringen visar felet under Mer → Alla fält
        if spec.kind == "int" and float(v).is_integer():
            v = int(v)
        if cur is not None and not cur.missing:
            try:
                if float(cur.value) == float(v):
                    continue
            except (TypeError, ValueError):
                pass
        company.set(key, dp(v, kind=kind, source=source, source_type=stype, pub_date=when, unit=spec.unit))
        changed += 1
    return changed


def _sheet_summary(a: DurrettAnalysis) -> None:
    m = st.columns(5)
    m[0].metric("Durrett Score", _f(a.quality_score.value), a.quality_score.reason or "Quality 0–100")
    m[1].metric("Risk", _f(a.risk_score.value), a.risk_level)
    m[2].metric("Upside", _f(a.upside_multiple, "{:.1f}×"), _f(a.potential_upside_pct, "{:+.0f} % Base", "Base N/A"))
    m[3].metric("Confidence", _f(a.confidence_score.value, "{:g} %"), _f(a.data_confidence, "data {:g}", ""))
    fe = a.metrics.get("mcap_future_earnings", {}).get("value")
    m[4].metric("MCap/framtida vinst", _f(fe, "{:.1f}×"), f"köpregel < {dc.DURRETT_CONFIG['mcap_future_earnings_buy_max']:g}×")
    if a.missing_data:
        st.caption("⚠ Saknas för full analys: " + ", ".join(a.missing_data[:12]) + (" …" if len(a.missing_data) > 12 else ""))
    crit = [f for f in a.red_flags if f.severity in ("CRITICAL", "HIGH")]
    if crit:
        st.markdown(" ".join(_badge(f"{f.severity} {f.flag}", _SEV_COLOR[f.severity]) for f in crit[:5]), unsafe_allow_html=True)


# ── Analys ───────────────────────────────────────────────────────────────────
def _render_analysis(a: DurrettAnalysis) -> None:
    cl = a.classification
    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>"
        f"<span style='color:{TEXT};font-size:1.05rem;font-weight:700;'>{a.name or a.ticker} · {a.ticker}</span>"
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
        rows = [f"{s.label:<14} {_bar(s.value)}  {_f(s.value, '{:>5.0f}', '  N/A')}" for s in a.scores()]
        for lbl, s in (("Quality", a.quality_score), ("Risk", a.risk_score), ("Confidence", a.confidence_score)):
            rows.append(f"{lbl:<14} {_bar(s.value)}  {_f(s.value, '{:>5.0f}', '  N/A')}")
        st.code("\n".join(rows), language=None)
        keys = [s.key for s in a.scores()] + ["quality", "risk", "confidence"]
        k = st.selectbox("Förklara en poäng (explain_score)", keys, key=f"durrett_explain_{a.ticker}")
        st.code(a.explain_score(k), language=None)
    with right:
        st.markdown("#### 🚨 Red flags")
        if not a.red_flags:
            st.caption("Inga red flags i det som är angivet.")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            fl = [f for f in a.red_flags if f.severity == sev]
            if not fl:
                continue
            st.markdown(_badge(sev, _SEV_COLOR[sev]), unsafe_allow_html=True)
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
            st.caption(f"{'✅' if ok else ('❌' if ok is False else '❔')} {name} — {why}")
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
        st.json(a.metrics)
        st.code("\n".join(a.log), language=None)
    st.download_button("⬇ EngineResult (JSON)", json.dumps(to_engine_result(a).as_dict(), ensure_ascii=False, indent=1),
                       file_name=f"durrett_{a.ticker}_{a.generated}.json", mime="application/json",
                       key=f"durrett_dl_{a.ticker}")


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
                                             key=f"dsc_{company.ticker}_{key}_{pk}")
                new[key] = vals
        if st.form_submit_button("Räkna om"):
            parsed = {}
            for key, vals in new.items():
                d = {}
                for pk, raw in vals.items():
                    v = _num_or_none(raw)
                    if str(raw).strip() and v is None:
                        st.error(f"{key} {pk}: inte ett tal ({raw!r})")
                    elif v is not None:
                        d[pk] = v
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


# ── Mer: momentum, extraktion, förslag, alla fält, katalysatorer, ta bort ────
def _render_more(data: dict, company: CompanyInput, a: DurrettAnalysis) -> None:
    _render_momentum_fetch(data, company)
    for public, private in (("render_extractor", "_render_extractor"), ("render_prefill", "_render_prefill")):
        fn = getattr(cui, public, None) or getattr(cui, private, None)
        if fn:
            fn(data, company)
    with st.expander("Alla fält (Confidence score + Durrett) med proveniens", expanded=False):
        _cui("render_inputs", "_render_inputs")(data, company)
    _render_catalysts(data, company, a)
    if st.button("🗑 Ta bort bolaget ur arket", key=f"durrett_del_{company.ticker}"):
        cs.remove(data, company.ticker)
        _save(data)
        st.rerun()


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
            _save(data)
            st.success(msg + " — inskrivna i sessionen, spara med 💾")
            st.rerun()
        for k in ("price_vs_ma200_pct", "momentum_6m_pct", "volume_trend"):
            p = company.get(k)
            if p is not None and not p.missing:
                st.caption(f"· {k}: {p.value:g} ({p.kind}, {p.source}, {p.pub_date})")


def _render_catalysts(data: dict, company, a: DurrettAnalysis) -> None:
    with st.expander(f"📅 Katalysatorer ({len(a.catalysts)})", expanded=False):
        if a.catalysts:
            st.dataframe([c.as_dict() for c in a.catalysts], hide_index=True, use_container_width=True)
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
                _save(data)
                st.rerun()
        if a.catalysts and st.button("Ta bort sista katalysatorn", key=f"durrett_cat_del_{company.ticker}"):
            company.catalysts = company.catalysts[:-1]
            cs.put(data, company)
            _save(data)
            st.rerun()


# ── Peers ────────────────────────────────────────────────────────────────────
_PEER_ROWS = (("Typ", lambda a: a.classification.company_type), ("Market Cap MUSD", lambda a: a.market_cap_musd),
              ("EV MUSD", lambda a: a.enterprise_value_musd),
              ("Resurs", lambda a: (a.metrics.get("resource_total") or {}).get("value")),
              ("Reserv", lambda a: (a.metrics.get("reserve_total") or {}).get("value")),
              ("Produktion", lambda a: (a.metrics.get("production_growth_multiple") or {}).get("note")),
              ("AISC-marginal", lambda a: (a.metrics.get("operating_margin_per_unit") or {}).get("note")),
              ("NPV/CAPEX", lambda a: (a.metrics.get("npv_capex") or {}).get("value")),
              ("EV/enhet resurs", lambda a: (a.metrics.get("ev_per_resource_unit") or {}).get("value")),
              ("EV/NPV", lambda a: (a.metrics.get("ev_npv") or {}).get("value")),
              ("MCap/framtida vinst", lambda a: (a.metrics.get("mcap_future_earnings") or {}).get("value")),
              ("Growth", lambda a: a.growth_score.value), ("Dilution", lambda a: a.dilution_score.value),
              ("Risk", lambda a: a.risk_score.value), ("Durrett Score", lambda a: a.quality_score.value),
              ("Upside ×", lambda a: a.upside_multiple), ("Confidence", lambda a: a.confidence_score.value))


def _peers_section(data: dict) -> None:
    tickers = list(cs.companies(data))
    if len(tickers) < 2:
        return
    with st.expander("🔬 Jämför sida vid sida (ingen ranking)", expanded=False):
        pick = st.multiselect("Bolag", tickers, default=tickers[:4], key="durrett_peers")
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
        st.caption("N/A = kunde inte beräknas (otillräcklig data).")


def analysis_json(a: DurrettAnalysis) -> dict:
    return to_jsonable(a)


__all__ = ["render_durrett_page", "analysis_json", "Score"]
