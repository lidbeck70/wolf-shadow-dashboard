"""
confidence/ui.py — fliken "🧭 Confidence score" under GRANSKNING.

Fyra delar: Analys (Investment Card, Case, Confidence, Why Now, regional
knapphet, scenarier, Thesis Killer, rapport), Indata (fält per pelare med
proveniens, förslag ur granskningsarken, extraktion ur presentation),
Råvaror (registrets sourcade värden) och Signaler (rotation, blindspot,
kvoter, ember — hämtas på knapp och skickas in som data).

Lagring: data/confidence.json via storage.session_load + 💾 Spara (samma
mekanik som arken). Inget beräknat lagras — allt räknas om ur inmatningarna.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import streamlit as st

import storage
import storage_ui
from confidence import commodities as com
from confidence import config as cfg
from confidence import prefill
from confidence import reports
from confidence import store as cs
from confidence.data.models import CompanyInput
from confidence.data.provenance import Datapoint, describe, dp
from confidence.data.validation import validate
from confidence.why_now import Signals, signals_from_sources

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"
_SIG = "conf_signals"
_BAND_COLOR = {"ELITE": GOLD, "PICK": GREEN, "STRONG CANDIDATE": GREEN, "WATCHLIST": AMBER,
               "SPECULATIVE": AMBER, "PASS": RED, "VERIFIED": GOLD, "HIGH CONFIDENCE": GREEN,
               "GOOD CONFIDENCE": GREEN, "MODERATE": AMBER, "LOW CONFIDENCE": RED,
               "BUY CANDIDATE": GREEN, "WATCH": AMBER, "REJECT": RED}
_SOURCE_TYPES = ("", "primary", "independent", "secondary", "mixed", "weak", "unsupported")
_PILLAR_LABEL = {**{k: l for k, l, _m in cfg.PILLARS}, **{k: l for k, l, _m in cfg.CONFIDENCE_PARTS},
                 "scenarios": "Scenarier", "kill": "Kill switches (bedömningar med belägg)",
                 "durrett_shares": "Durrett · aktiestruktur & valuta", "durrett_resources": "Durrett · reserver & resurser",
                 "durrett_production": "Durrett · produktion & kostnader", "durrett_balance": "Durrett · kassa & skuld",
                 "durrett_management": "Durrett · management", "durrett_jurisdiction": "Durrett · projektrisk",
                 "durrett_explorer": "Durrett · explorer", "durrett_momentum": "Durrett · momentum"}


# ── lagring ──────────────────────────────────────────────────────────────────
def _load() -> dict:
    data = storage.session_load(cs.STORE, cs.default())
    data = cs.normalize(data)
    st.session_state[cs.STORE] = data
    return data


def _save(data: dict) -> None:
    st.session_state[cs.STORE] = data          # persistensen sker via 💾 Spara


# ── hjälpare ─────────────────────────────────────────────────────────────────
from ui.components import badge as _badge, confirm_delete, page_header  # noqa: E402


def _num_or_none(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", ".").replace(" ", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Ja" if v else "Nej"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)


def _signals_cache() -> dict:
    return st.session_state.setdefault(_SIG, {"rotation": None, "themes": None, "ratios": None,
                                              "complexes": None, "log": []})


def _signals_for(commodity) -> Signals:
    c = _signals_cache()
    return signals_from_sources(commodity, c.get("rotation"), c.get("themes"), c.get("ratios"),
                                c.get("complexes"))


# ── sidan ────────────────────────────────────────────────────────────────────
def render_confidence_page() -> None:
    data = _load()
    storage_ui.save_bar(cs.STORE, "Confidence score")
    page_header("Confidence-case", "Case Score 0–100 (hur bra är caset) och Confidence Score "
                "0–100 (hur säkra är vi). Varje poäng har en förklaringsrad med källa. "
                "Tomt = okänt (DATA_MISSING), aldrig 0 och aldrig ett gissat tal. "
                "Samma ark som Durrett 10-steg.")

    tickers = list(cs.companies(data))
    c1, c2 = st.columns([2, 3])
    with c1:
        choice = st.selectbox("Bolag", ["➕ Nytt bolag"] + tickers, key="conf_pick",
                              index=(tickers.index(st.session_state.get("conf_last")) + 1
                                     if st.session_state.get("conf_last") in tickers else (1 if tickers else 0)))
    if choice == "➕ Nytt bolag":
        _new_company_form(data)
        return
    st.session_state["conf_last"] = choice
    company = cs.get(data, choice)
    if company is None:
        st.warning("Bolaget hittades inte.")
        return
    with c2:
        sub = st.radio("", ["Analys", "Indata", "Råvaror", "Signaler"], horizontal=True,
                       label_visibility="collapsed", key="conf_sub")
    st.markdown("---")
    if sub == "Analys":
        _render_analysis(data, company)
    elif sub == "Indata":
        _render_inputs(data, company)
    elif sub == "Råvaror":
        _render_commodities(data, company)
    else:
        _render_signals(company)


# ── nytt bolag ───────────────────────────────────────────────────────────────
def _identity_widgets(prefix: str, c: Optional[CompanyInput]) -> dict:
    labels = {k: f"{v.label} ({k})" for k, v in com.REGISTRY.items()}
    keys = list(labels)
    a, b, d = st.columns(3)
    out = {}
    out["ticker"] = a.text_input("Ticker", value=c.ticker if c else "", key=f"{prefix}_ticker",
                                 disabled=c is not None)
    out["name"] = b.text_input("Namn", value=c.name if c else "", key=f"{prefix}_name")
    cur = keys.index(c.commodity) if c and c.commodity in keys else 0
    out["commodity"] = d.selectbox("Råvara", keys, index=cur, format_func=lambda k: labels[k],
                                   key=f"{prefix}_commodity")
    a, b, d = st.columns(3)
    out["country"] = a.text_input("Land (kod eller namn)", value=c.country if c else "", key=f"{prefix}_country",
                                  help="CA, US, AU, SE … — repots jurisdiktionstabell")
    out["jurisdiction"] = b.text_input("Region/stat/provins", value=c.jurisdiction if c else "",
                                       key=f"{prefix}_jur", help="Quebec, Nevada, Western Australia …")
    out["exchange"] = d.text_input("Börs", value=c.exchange if c else "", key=f"{prefix}_exch",
                                   help="TSX, TSXV, ASX, NYSE …")
    a, b = st.columns(2)
    out["stage"] = a.selectbox("Stage", list(cfg.STAGES), index=cfg.STAGES.index(c.stage) if c else 1,
                               key=f"{prefix}_stage")
    out["maturity"] = b.selectbox("Mognad", list(cfg.MATURITY), index=cfg.MATURITY.index(c.maturity) if c else 1,
                                  format_func=lambda k: cfg.MATURITY_LABEL[k], key=f"{prefix}_mat")
    return out


def _new_company_form(data: dict) -> None:
    st.markdown("#### Nytt bolag")
    with st.form("conf_new"):
        ident = _identity_widgets("conf_new", None)
        if st.form_submit_button("Lägg till"):
            t = ident["ticker"].strip().upper()
            if not t:
                st.error("Ticker krävs.")
            elif cs.get(data, t):
                st.error(f"{t} finns redan.")
            else:
                cs.put(data, CompanyInput(ticker=t, **{k: v for k, v in ident.items() if k != "ticker"}))
                _save(data)
                st.session_state["conf_last"] = t
                st.session_state["conf_pick"] = t
                st.rerun()


# ── Analys ───────────────────────────────────────────────────────────────────
def _render_analysis(data: dict, company: CompanyInput) -> None:
    commodity = com.get(company.commodity, cs.overrides(data))
    a = reports.analyze(company, cs.overrides(data), _signals_for(commodity), date.today())
    card = reports.investment_card(a)

    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>"
        f"<span style='color:{TEXT};font-size:1.1rem;font-weight:700;'>{company.ticker} · {company.name}</span>"
        f"{_badge(a.recommendation, _BAND_COLOR.get(a.recommendation, DIM))}"
        f"<span style='color:{DIM};font-size:0.8rem;'>{a.recommendation_why}</span></div>",
        unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("Case Score", f"{a.case.total:g}", a.case.rating)
    m[1].metric("Confidence", f"{a.confidence.total:g}", a.confidence.band)
    m[2].metric("Why Now", f"{a.why_now.score:g}", f"{a.why_now.band} · täckning {a.why_now.coverage * 100:.0f} %")
    m[3].metric("Regional knapphet", f"{a.regional.score:g}", a.regional.band)
    m[4].metric("Time-to-money", card["time_to_money"], a.ttm.basis)
    m[5].metric("Asymmetri", card["asymmetry"],
                f"Base {card['upside_base_pct']:+.0f} %" if card["upside_base_pct"] is not None else "DATA_MISSING")
    if a.case.missing:
        st.caption(f"⚠ {len(a.case.missing)} fält saknas (DATA_MISSING) — Case Score är ett golv, inte ett betyg: "
                   + ", ".join(a.case.missing[:12]) + (" …" if len(a.case.missing) > 12 else ""))
    errs = [i for i in a.issues if i.level == "error"]
    for i in errs:
        st.error(f"{i.field}: {i.message}")

    left, right = st.columns(2)
    with left:
        st.markdown(f"#### Case Score {a.case.total:g} — {_badge(a.case.rating, _BAND_COLOR.get(a.case.rating, DIM))}",
                    unsafe_allow_html=True)
        for p in a.case.pillars:
            _pillar_row(p)
        if a.case.discovery_option is not None:
            st.caption(f"Discovery Option (separat): {a.case.discovery_option:g}/{cfg.DISCOVERY_OPTION_MAX}")
    with right:
        st.markdown(f"#### Confidence {a.confidence.total:g} — "
                    f"{_badge(a.confidence.band, _BAND_COLOR.get(a.confidence.band, DIM))}",
                    unsafe_allow_html=True)
        for p in a.confidence.parts:
            _pillar_row(p)
        for f in a.confidence.flags:
            st.caption(("🔴 " if f.startswith("KILL") else "• ") + f)

    st.markdown("#### Thesis Killer")
    lvl_color = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}
    for r in a.risks:
        st.markdown(f"{_badge(r.level, lvl_color[r.level])} <b style='color:{TEXT};'>{r.name}</b> "
                    f"<span style='color:{DIM};font-size:0.84rem;'>{r.why}</span>", unsafe_allow_html=True)

    st.markdown("#### Scenarier")
    if not a.scenarios.scenarios:
        st.info("; ".join(a.scenarios.flags) or "Inga scenarier.")
    else:
        rows = [{"Scenario": s.label, "Pris": s.price, "EBITDA/marginal MUSD": s.ebitda_musd, "FCF MUSD": s.fcf_musd,
                 "EV/NAV MUSD": s.value_musd, "Equity MUSD": s.equity_musd, "Aktie": s.share_price,
                 "Upside %": None if s.upside_pct is None else round(s.upside_pct)} for s in a.scenarios.scenarios]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        with st.expander("Steg och antaganden", expanded=False):
            for s in a.scenarios.scenarios:
                st.markdown(f"**{s.label}**")
                for step in s.steps:
                    st.caption("· " + step)
            for p in a.scenarios.paths:
                st.markdown(f"**{p.target:g}× — vad måste hända**")
                for step in p.steps:
                    st.caption("· " + step)
            st.markdown("**Antaganden**")
            for x in a.scenarios.assumptions:
                st.caption("· " + x.text())
        for f in a.scenarios.flags:
            st.caption("⚠ " + f)

    with st.expander("Why Now · Regional knapphet · Time-to-money", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Why Now {a.why_now.score:g} — {a.why_now.band}**")
            for n in a.why_now.notes:
                st.caption("· " + n)
            for f in a.why_now.flags:
                st.caption("⚠ " + f)
        with c2:
            st.markdown(f"**Regional knapphet {a.regional.score:g} — {a.regional.band}**")
            for n in a.regional.notes:
                st.caption("· " + n)
            for f in a.regional.flags:
                st.caption("⚠ " + f)
        with c3:
            st.markdown(f"**Time-to-money {a.ttm.years:g} år — {a.ttm.confidence}**")
            for n in a.ttm.notes:
                st.caption("· " + n)
            for f in a.ttm.flags:
                st.caption("⚠ " + f)

    with st.expander("Investment Card", expanded=False):
        st.json(card)
    md = reports.report_markdown(a)
    st.download_button("⬇ Rapport (Markdown)", md, file_name=f"confidence_{company.ticker}_{a.today}.md",
                       mime="text/markdown", key="conf_dl_md")


def _pillar_row(p) -> None:
    pct = p.pct
    color = GREEN if pct >= 70 else (AMBER if pct >= 40 else RED)
    with st.expander(f"{p.label}  {p.points:g}/{p.max:g}", expanded=False):
        st.markdown(f"<div style='height:6px;background:#2a2a2a;border-radius:3px;'>"
                    f"<div style='height:6px;width:{pct:.0f}%;background:{color};border-radius:3px;'></div></div>",
                    unsafe_allow_html=True)
        for n in p.notes:
            st.caption(("🔻 " if n.startswith("TAK") else "· ") + n)
        if p.missing:
            st.caption("DATA_MISSING: " + ", ".join(p.missing))


# ── Indata ───────────────────────────────────────────────────────────────────
def _render_inputs(data: dict, company: CompanyInput) -> None:
    with st.expander("Identitet", expanded=False):
        with st.form(f"conf_ident_{company.ticker}"):
            ident = _identity_widgets(f"conf_id_{company.ticker}", company)
            if st.form_submit_button("Uppdatera"):
                for k, v in ident.items():
                    if k != "ticker":
                        setattr(company, k, v)
                cs.put(data, company)
                _save(data)
                st.rerun()
        if confirm_delete("Ta bort bolaget", key=f"conf_del_{company.ticker}"):
            cs.remove(data, company.ticker)
            _save(data)
            st.session_state.pop("conf_last", None)
            st.rerun()

    _render_prefill(data, company)
    _render_extractor(data, company)

    issues = validate(company)
    for i in issues:
        if i.level == "error":
            st.error(f"{i.field}: {i.message}")
    st.caption("Varje fält: värde · datatyp (ACTUAL/ESTIMATE/GUIDANCE/MODELLED/ASSUMPTION) · källa · "
               "källtyp · datum. Tomt värde = DATA_MISSING. Källtyp styr Data Quality.")

    groups: dict = {}
    for f in cfg.fields_for(company.stage):
        groups.setdefault(f.pillar, []).append(f)
    for pillar, fields in groups.items():
        filled = sum(1 for f in fields if company.has(f.key))
        with st.expander(f"{_PILLAR_LABEL.get(pillar, pillar)} · {filled}/{len(fields)} ifyllda", expanded=False):
            with st.form(f"conf_form_{company.ticker}_{pillar}"):
                widgets = [(_field_widgets(company, f), f) for f in fields]
                if st.form_submit_button("Uppdatera"):
                    for w, f in widgets:
                        point = _point_from_widgets(w, f)
                        if point is None:
                            company.fields.pop(f.key, None)
                        else:
                            company.set(f.key, point)
                    cs.put(data, company)
                    _save(data)
                    st.rerun()


def _field_widgets(company: CompanyInput, f: cfg.FieldSpec) -> dict:
    p = company.get(f.key)
    k = f"cf_{company.ticker}_{f.key}"
    c = st.columns([3, 1.4, 2.4, 1.4, 1.3])
    label = f.label + (f" ({f.unit})" if f.unit else "") + (f" 0–{f.max:g}" if f.max is not None else "")
    w = {}
    if f.kind == "bool":
        cur = {True: "Ja", False: "Nej"}.get(p.value if p else None, "—") if p and not p.missing else "—"
        if p and isinstance(p.value, str):
            cur = "Ja" if p.value.lower() in ("ja", "true", "yes", "1") else ("Nej" if p.value.lower() in ("nej", "false", "no", "0") else "—")
        w["value"] = c[0].selectbox(label, ["—", "Ja", "Nej"], index=["—", "Ja", "Nej"].index(cur), key=f"{k}_v",
                                    help=f.hint or None)
    elif f.kind == "choice":
        opts = ["—"] + list(f.choices)
        cur = str(p.value) if p and not p.missing and str(p.value) in f.choices else "—"
        w["value"] = c[0].selectbox(label, opts, index=opts.index(cur), key=f"{k}_v", help=f.hint or None)
    else:
        w["value"] = c[0].text_input(label, value=_fmt(p.value) if p and not p.missing else "", key=f"{k}_v",
                                     help=f.hint or None)
    kinds = list(cfg.KINDS)
    w["kind"] = c[1].selectbox("Typ", kinds, index=kinds.index(p.kind) if p and p.kind in kinds else 1,
                               key=f"{k}_k")
    w["source"] = c[2].text_input("Källa", value=p.source if p else "", key=f"{k}_s")
    st_ = list(_SOURCE_TYPES)
    w["source_type"] = c[3].selectbox("Källtyp", st_, index=st_.index(p.source_type) if p and p.source_type in st_ else 0,
                                      key=f"{k}_t")
    w["pub_date"] = c[4].text_input("Datum", value=(p.pub_date or "") if p else "", key=f"{k}_d",
                                    placeholder="ÅÅÅÅ-MM-DD")
    return w


def _point_from_widgets(w: dict, f: cfg.FieldSpec) -> Optional[Datapoint]:
    raw = w["value"]
    if f.kind == "bool":
        if raw == "—":
            return None
        value = raw == "Ja"
    elif f.kind == "choice":
        if raw == "—":
            return None
        value = raw
    elif f.kind in ("number", "int"):
        if not str(raw).strip():
            return None
        value = _num_or_none(raw)
        if value is None:
            value = raw                      # valideringen visar felet
        elif f.kind == "int" and float(value).is_integer():
            value = int(value)
    else:
        if not str(raw).strip():
            return None
        value = str(raw).strip()
    return dp(value, kind=w["kind"], source=w["source"].strip(), source_type=w["source_type"],
              pub_date=(w["pub_date"].strip() or None), unit=f.unit)


def _render_prefill(data: dict, company: CompanyInput) -> None:
    with st.expander("Förslag ur granskningsarken (Rick Rule/AQS/DS, Tiggre)", expanded=False):
        try:
            producers_data = storage.session_load("producers", {})
            import positions as _positions
            tiggre_data = {**storage.session_load("tiggre", {}),
                           "positions": _positions.view_rows("Tiggre")}   # registret
        except Exception as exc:                          # pragma: no cover
            st.caption(f"Kunde inte läsa arken: {exc}")
            return
        props = prefill.proposals(company, producers_data, tiggre_data)
        if not props:
            st.caption("Inget nytt att hämta — ingen rad med samma ticker, eller allt är redan inläst.")
            return
        for key, point, already in props:
            a, b = st.columns([4, 1])
            if key == "maturity":
                a.markdown(f"<span style='color:{TEXT};'>Mognad: {cfg.MATURITY_LABEL[point]}</span> "
                           f"<span style='color:{DIM};font-size:0.8rem;'>(Tiggre: FS klar)</span>",
                           unsafe_allow_html=True)
                if b.button("Använd", key=f"cf_pf_{company.ticker}_{key}"):
                    company.maturity = point
                    cs.put(data, company)
                    _save(data)
                    st.rerun()
                continue
            label = cfg.FIELD_BY_KEY[key].label
            a.markdown(f"<span style='color:{TEXT};'>{describe(point, label)}</span>"
                       + (f" <span style='color:{AMBER};font-size:0.78rem;'>ersätter {_fmt(company.get(key).value)}</span>"
                          if already else ""), unsafe_allow_html=True)
            if b.button("Använd", key=f"cf_pf_{company.ticker}_{key}"):
                company.set(key, point)
                cs.put(data, company)
                _save(data)
                st.rerun()


def _render_extractor(data: dict, company: CompanyInput) -> None:
    from ai import document as doc
    from ai import extract_prompt as xp
    from ai import openai_client as oc

    skey = f"cf_xt_{company.ticker}"
    with st.expander("🤖 Läs ur presentationen / tekniska rapporten (AI-förslag — inget skrivs in utan Använd)",
                     expanded=False):
        if not oc.configured():
            st.caption("OPENAI_API_KEY saknas i secrets — extraktionen är avstängd.")
            return
        c1, c2 = st.columns(2)
        up = c1.file_uploader("PDF", type=["pdf"], key=f"{skey}_pdf")
        pasted = c2.text_area("… eller klistra in text", height=120, key=f"{skey}_txt")
        docname = st.text_input("Dokumentets namn (blir källa)", value=(up.name if up is not None else ""),
                                key=f"{skey}_name", placeholder="DFS 2025 (Ausenco)")
        if st.button("Extrahera", key=f"{skey}_go"):
            try:
                if up is not None:
                    text = doc.pdf_to_text(up.getvalue())
                elif pasted.strip():
                    text = doc.text_with_pages(pasted)
                else:
                    st.warning("Ladda upp en PDF eller klistra in text först.")
                    return
                prompt = xp.build_extract_prompt("confidence", company.ticker, company.name, text)
                with st.spinner("Läser dokumentet …"):
                    reply = oc.complete(xp.SYSTEM_EXTRACT, prompt, max_output_tokens=2500, timeout=120.0,
                                        json_mode=True)
                parsed = xp.parse_extraction(reply.text)
                st.session_state[skey] = {"proposals": xp.proposals("confidence", parsed),
                                          "notes": [str(n) for n in (parsed.get("notes") or [])][:6],
                                          "model": reply.model, "doc": docname or "Presentation"}
            except (oc.AIError, xp.ExtractionError, RuntimeError) as exc:
                st.error(str(exc))
                return
        res = st.session_state.get(skey)
        if not res:
            return
        st.caption(f"{res['model']} · {len(res['proposals'])} förslag")
        for p in res["proposals"]:
            a, b = st.columns([4, 1])
            page = f" · sida {p['page']}" if p["page"] else ""
            a.markdown(f"<span style='color:{TEXT};font-weight:700;'>{p['label']}:</span> "
                       f"<span style='color:{TEXT};'>{_fmt(p['value'])} {p['unit']}</span> "
                       f"<span style='color:{DIM};font-size:0.78rem;'>{p['confidence']}{page}"
                       + (f" · ”{p['quote']}”" if p["quote"] else "") + "</span>", unsafe_allow_html=True)
            if p["apply"] and p["key"] in cfg.FIELD_BY_KEY and b.button("Använd", key=f"{skey}_use_{p['key']}"):
                spec = cfg.FIELD_BY_KEY[p["key"]]
                value = p["value"]
                if spec.kind == "int" and isinstance(value, float) and value.is_integer():
                    value = int(value)
                src = res["doc"] + (f" s.{p['page']}" if p["page"] else "") + (f": ”{p['quote']}”" if p["quote"] else "")
                company.set(p["key"], dp(value, kind="ESTIMATE", source=src, source_type="primary",
                                         confidence=p["confidence"], unit=p["unit"] or spec.unit,
                                         note="ur extraktionen — sätt datatyp och datum"))
                cs.put(data, company)
                _save(data)
                st.rerun()
        for n in res["notes"]:
            st.caption("📝 " + n)


# Publika alias för andra flikar som delar lagret (Durrett-fliken)
load_store = _load
save_store = _save
render_inputs = _render_inputs
identity_widgets = _identity_widgets
new_company_form = _new_company_form


# ── Råvaror ──────────────────────────────────────────────────────────────────
def _render_commodities(data: dict, company: CompanyInput) -> None:
    keys = com.all_keys()
    cur = keys.index(company.commodity) if company.commodity in keys else 0
    key = st.selectbox("Råvara", keys, index=cur, format_func=lambda k: com.REGISTRY[k].label, key="conf_com")
    c = com.get(key, cs.overrides(data))
    st.caption("Registret seedar bara strategisk betydelse (ur repots nödvändighetstabell, ESTIMATE). "
               "Allt annat är null tills du matar in ett värde med källa. Crosswalk: "
               + ", ".join(f"{k}={v}" for k, v in com.crosswalk(key).items() if v))
    fields = (("strategic_significance", "Strategisk betydelse 0–10"), ("demand_growth", "Efterfrågetillväxt 0–5"),
              ("geopolitical_scarcity", "Geopolitisk knapphet 0–5"),
              ("supply_balance_pct", "Utbudsbalans % av efterfrågan (underskott +, överskott −)"),
              ("supply_concentration_pct", "Största producentlands andel av utbudet %"),
              ("western_share_pct", "Andel ur västliga jurisdiktioner %"))
    with st.form(f"conf_com_{key}"):
        vals = {}
        for fkey, label in fields:
            p = getattr(c, fkey)
            cols = st.columns([3, 1.4, 2.4, 1.4, 1.3])
            vals[fkey] = {
                "value": cols[0].text_input(label, value=_fmt(p.value) if not p.missing else "", key=f"cc_{key}_{fkey}_v"),
                "kind": cols[1].selectbox("Typ", list(cfg.KINDS), index=list(cfg.KINDS).index(p.kind) if p.kind in cfg.KINDS else 1,
                                          key=f"cc_{key}_{fkey}_k"),
                "source": cols[2].text_input("Källa", value=p.source, key=f"cc_{key}_{fkey}_s"),
                "source_type": cols[3].selectbox("Källtyp", list(_SOURCE_TYPES),
                                                 index=list(_SOURCE_TYPES).index(p.source_type) if p.source_type in _SOURCE_TYPES else 0,
                                                 key=f"cc_{key}_{fkey}_t"),
                "pub_date": cols[4].text_input("Datum", value=p.pub_date or "", key=f"cc_{key}_{fkey}_d"),
            }
        top = st.text_input("Största producentland", value=c.top_supplier, key=f"cc_{key}_top")
        adj = st.multiselect("Justeringar (−1 p var; projektförseningar +1)", list(cfg.SUPPLY_ADJUSTMENTS),
                             default=[a for a in c.adjustments if a in cfg.SUPPLY_ADJUSTMENTS], key=f"cc_{key}_adj")
        if st.form_submit_button("Uppdatera"):
            for fkey, w in vals.items():
                v = _num_or_none(w["value"])
                if v is None:
                    if fkey != "strategic_significance":
                        cs.set_override(data, key, fkey, None)
                    continue
                cs.set_override(data, key, fkey, {"value": v, "kind": w["kind"], "source": w["source"].strip(),
                                                  "source_type": w["source_type"],
                                                  "pub_date": w["pub_date"].strip() or None})
            cs.set_override(data, key, "top_supplier", top.strip())
            cs.set_override(data, key, "adjustments", list(adj))
            _save(data)
            st.rerun()
    st.markdown("**Nu gällande**")
    for fkey, label in fields:
        st.caption("· " + describe(getattr(c, fkey), label))


# ── Signaler ─────────────────────────────────────────────────────────────────
def _render_signals(company: CompanyInput) -> None:
    cache = _signals_cache()
    st.caption("Why Now läser repots egna signaler. Hämta dem här (cachas i sessionen) — motorn räknar "
               "aldrig med nätverket själv.")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Läs rotationen", key="conf_sig_rot"):
        try:
            cache["rotation"] = storage.session_load("rotation", {})
            cache["log"].append("rotation: läst")
        except Exception as exc:
            cache["log"].append(f"rotation: {exc}")
    if c2.button("Hämta blindspot-teman", key="conf_sig_theme"):
        try:
            from blindspot.theme_board import build_theme_board
            with st.spinner("Hämtar 10-årshistorik …"):
                cache["themes"] = build_theme_board()
            cache["log"].append(f"teman: {len(cache['themes'])}")
        except Exception as exc:
            cache["log"].append(f"teman: {exc}")
    if c3.button("Hämta kvoter", key="conf_sig_ratio"):
        try:
            from alpha_regime.commodity_ratios import fetch_all_ratios
            with st.spinner("Hämtar kvoter …"):
                cache["ratios"] = fetch_all_ratios()
            cache["log"].append(f"kvoter: {len(cache['ratios'])}")
        except Exception as exc:
            cache["log"].append(f"kvoter: {exc}")
    if c4.button("Beräkna ember-komplex", key="conf_sig_ember"):
        try:
            from ember.regime import compute_all_complex_regimes
            with st.spinner("Beräknar komplexen …"):
                cache["complexes"] = compute_all_complex_regimes()
            cache["log"].append(f"komplex: {len(cache['complexes'])}")
        except Exception as exc:
            cache["log"].append(f"komplex: {exc}")
    for line in cache["log"][-6:]:
        st.caption("· " + line)
    commodity = com.get(company.commodity)
    s = _signals_for(commodity)
    st.markdown(f"**Signaler för {commodity.label if commodity else company.commodity}**")
    st.caption(f"· cykel: {s.cycle_label or 'DATA_MISSING'}"
               + (f" ({s.cycle_percentile:.0f}:e percentilen)" if s.cycle_percentile is not None else ""))
    st.caption("· Triple Signal: " + (f"{s.rotation_grade} ({s.rotation_month})" if s.rotation_grade else "DATA_MISSING"))
    st.caption(f"· gummiband: {s.ratio_status or 'DATA_MISSING'}" + (f" ({s.ratio_key})" if s.ratio_key else ""))
    st.caption(f"· komplex: {s.complex_verdict or 'DATA_MISSING'}" + (f" ({s.complex_key})" if s.complex_key else ""))
