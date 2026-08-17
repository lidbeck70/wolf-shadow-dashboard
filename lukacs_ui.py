"""
lukacs_ui.py — FV-modulen som sektion i CSM-blocket.

Ligger i CSM-expandern eftersom frågorna hänger ihop: CSM säger om bolaget
överlever varje scenario, FV vad det är värt där. Muterar raden och returnerar
True om något ändrats, precis som de andra kontrollsektionerna.
"""

from __future__ import annotations

import streamlit as st

import controls as ctl
import lukacs as fv

TEXT, DIM = "#e8e4dc", "#8a8578"


def _class_label(q) -> str:
    band = (f"{q.lo:g}–{q.hi:g} %" if q.lo is not None
            else "värderas ej på FCF")
    return f"{q.code} — {q.desc} Target-yield {band}"


def render_fv(row: dict, key: str, prefix: str = "fv") -> bool:
    """Lukacs FV-modulen för ett kandidatkort."""
    changed = False
    data = row.setdefault("fv", {})

    st.markdown(
        f"<b style='color:{TEXT};'>Lukacs FV — fair value per aktie</b> "
        f"<span style='color:{DIM};font-size:0.78rem;'>— CSM säger om bolaget "
        f"överlever scenariot, det här vad det är värt där.</span>",
        unsafe_allow_html=True)

    # ── Kvalitetsklassen, som låser yield-bandet ─────────────────────────────
    codes = list(fv.CLASS_CODES)
    cur_code = str(row.get("fcf_kvalitet") or "C").upper()[:1]
    idx = codes.index(cur_code) if cur_code in codes else codes.index("C")
    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    code = c1.selectbox(
        "FCF-kvalitet", codes, index=idx, key=f"{prefix}_{key}_class",
        format_func=lambda c: f"{c} — {fv.CLASS_BY_CODE[c].name}",
        help="\n\n".join(_class_label(q) for q in fv.QUALITY_CLASSES))
    if code != row.get("fcf_kvalitet"):
        row["fcf_kvalitet"] = code
        changed = True

    q = fv.quality_class(code)
    band = fv.yield_band(code)
    locked = band is None

    shares = c2.number_input(
        "Framtida antal aktier (miljoner)", min_value=0.0, step=1.0,
        value=float(fv.num(row.get("framtida_antal_aktier"), 0.0) or 0.0),
        key=f"{prefix}_{key}_shares",
        help="Prognos inklusive återköp och utspädning — återköp är DS:ens "
             "spegelbild.")
    price = c3.number_input(
        "Aktuell kurs", min_value=0.0, step=0.1,
        value=float(fv.num(row.get("aktuell_kurs"), 0.0) or 0.0),
        key=f"{prefix}_{key}_price",
        help="Samma valuta som forward FCF anges i.")
    if (shares != row.get("framtida_antal_aktier")
            or price != row.get("aktuell_kurs")):
        row["framtida_antal_aktier"], row["aktuell_kurs"] = shares, price
        changed = True

    if locked:
        st.info(f"{fv.NOT_FCF_VALUED}. Target-yield är låst — ett "
                f"extraordinärt kassaflöde ger ingen meningsfull fair value. "
                f"Byt klass om kassaflödet visar sig vara uthålligt.")
    else:
        st.caption(f"Klass {q.code}: target-yield {band[0]:g}–{band[1]:g} %. "
                   f"Bandet är låst till klassen — annars räknar man fram den "
                   f"uppsida man redan bestämt sig för.")

    # ── Scenarierna ──────────────────────────────────────────────────────────
    for s in fv.FV_SCENARIOS:
        sc = data.setdefault(s, {})
        cols = st.columns([0.6, 1.2, 1, 1, 1])
        cols[0].markdown(f"<div style='padding-top:8px;color:{TEXT};"
                         f"font-weight:600;'>{s}</div>", unsafe_allow_html=True)
        fcf = cols[1].number_input(
            f"Forward FCF MUSD ({s})", step=1.0,
            value=float(fv.num(sc.get("forward_fcf_musd"), 0.0) or 0.0),
            key=f"{prefix}_{key}_{s}_fcf", label_visibility="collapsed",
            help="Normaliserat råvarupris, aldrig toppår.")
        ty = cols[2].number_input(
            f"Target-yield % ({s})", min_value=0.0, step=0.5,
            value=float(fv.num(sc.get("target_yield"), 0.0) or 0.0),
            key=f"{prefix}_{key}_{s}_yield", label_visibility="collapsed",
            disabled=locked,
            help=(f"Låst till klass {q.code}: {band[0]:g}–{band[1]:g} %."
                  if band else fv.NOT_FCF_VALUED))
        vals = {"forward_fcf_musd": fcf, "target_yield": ty}
        if any(sc.get(k) != v for k, v in vals.items()):
            sc.update(vals)
            changed = True

        one = fv.fair_value_per_share(fcf, ty, shares) if not locked else None
        up = fv.upside_pct(one, price)
        cols[3].metric(f"FV ({s})", f"{one:,.2f}" if one is not None else "–",
                       label_visibility="collapsed")
        cols[4].metric(f"Uppsida ({s})", f"{up:+.0f} %" if up is not None else "–",
                       label_visibility="collapsed")

        err = fv.yield_error(code, ty)
        if err and err != fv.NOT_FCF_VALUED:
            st.error(err)

    st.caption("Kolumner: scenario · forward FCF (MUSD, normaliserat pris) · "
               "target-yield % · fair value per aktie · uppsida mot kurs.")

    # ── Sannolikheter ────────────────────────────────────────────────────────
    dev = st.checkbox(
        "Avvik från default 20/60/20", value=bool(row.get("prob_deviation")),
        key=f"{prefix}_{key}_probdev",
        help="Defaultvikterna är låsta. Avvikelser kräver skriftlig "
             "motivering — annars är en justerad sannolikhet bara en tumme "
             "på vågen.")
    if dev != bool(row.get("prob_deviation")):
        row["prob_deviation"] = dev
        changed = True

    if dev:
        stored = row.setdefault("probs", dict(fv.DEFAULT_PROBS))
        pcols = st.columns(3)
        newp = {}
        for s, col in zip(fv.FV_SCENARIOS, pcols):
            newp[s] = col.number_input(
                f"{s} %", min_value=0.0, max_value=100.0, step=5.0,
                value=float(fv.num(stored.get(s), fv.DEFAULT_PROBS[s])),
                key=f"{prefix}_{key}_p_{s}")
        if any(stored.get(s) != newp[s] for s in fv.FV_SCENARIOS):
            row["probs"] = newp
            changed = True
        mot = st.text_area(
            "Motivering till avvikelsen", value=row.get("prob_motivation", ""),
            key=f"{prefix}_{key}_probmot", height=68,
            placeholder="Varför är basfallet mindre/mer sannolikt än normalt?")
        if mot != row.get("prob_motivation", ""):
            row["prob_motivation"] = mot
            changed = True
    else:
        st.caption("Sannolikheter: Bear 20 % · Base 60 % · Bull 20 % (låsta).")

    # ── What must go right ───────────────────────────────────────────────────
    wmgr = st.text_area(
        "What must go right — Base", value=row.get("what_must_go_right", ""),
        key=f"{prefix}_{key}_wmgr", height=80,
        placeholder="Råvarupris · produktion · capex · amorteringstakt · återköp",
        help="Obligatoriskt för grönt i köpgrindens steg 5. Bryts ett "
             "antagande är det en modellförlust — inte ett väntläge.")
    if wmgr != row.get("what_must_go_right", ""):
        row["what_must_go_right"] = wmgr
        changed = True

    holding = st.checkbox(
        "Detta är ett innehav (inte bara en kandidat)",
        value=bool(row.get("is_holding")), key=f"{prefix}_{key}_holding",
        help="Slår på trimvarningen när uppsidan mot Base krympt.")
    if holding != bool(row.get("is_holding")):
        row["is_holding"] = holding
        changed = True

    # ── Deleveraging ─────────────────────────────────────────────────────────
    state = fv.deleveraging_state(row.get("nd_ebitda"),
                                  row.get("ar_till_lag_skuld"))
    if state["applies"]:
        st.warning(fv.DELEV_TEXT)
        yrs = st.number_input(
            "År till låg skuld", min_value=0.0, step=0.5,
            value=float(fv.num(row.get("ar_till_lag_skuld"), 0.0) or 0.0),
            key=f"{prefix}_{key}_delev",
            help="Nettoskuld ÷ årlig amortering. Kravet är under "
                 f"{fv.DELEV_YEARS_MAX:g} år.")
        if yrs != row.get("ar_till_lag_skuld"):
            row["ar_till_lag_skuld"] = yrs
            changed = True
        for gap in fv.deleveraging_state(row.get("nd_ebitda"), yrs)["gaps"]:
            st.error(gap)

    # ── Resultatet ───────────────────────────────────────────────────────────
    ev = fv.evaluate(row)
    _summary(ev)
    return changed


def _summary(ev: dict) -> None:
    """Säkerhetsmarginal, expected value och säljregeln."""
    if ev["not_fcf_valued"]:
        return

    color = fv.MOS_COLOR.get(ev["mos_band"], DIM)
    mos = ev["mos"]
    if mos is None:
        st.caption("Säkerhetsmarginalen kräver fair value för Base och en "
                   "kurs.")
    else:
        st.markdown(
            f"<div style='border:1px solid {color}55;background:{color}0d;"
            f"border-radius:8px;padding:10px 14px;margin:8px 0;'>"
            f"<span style='color:{color};font-weight:700;font-size:1.05rem;'>"
            f"Säkerhetsmarginal {mos:.1f} %</span>"
            f"<span style='color:{color};font-weight:700;margin-left:12px;'>"
            f"{ev['mos_band']}</span></div>", unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    e1.metric("Expected value / aktie",
              f"{ev['expected_value']:,.2f}"
              if ev["expected_value"] is not None else "–",
              help="Σ(scenariovärde × sannolikhet).")
    e2.metric("EV-uppsida mot kurs",
              f"{ev['ev_upside']:+.0f} %"
              if ev["ev_upside"] is not None else "–")

    for err in ev["prob_errors"]:
        st.error(err)
    if ev["trim"]:
        st.markdown(
            f"<div style='border:1px solid {fv.ORANGE}88;"
            f"background:{fv.ORANGE}14;border-radius:8px;padding:8px 12px;"
            f"margin:6px 0;color:{fv.ORANGE};font-weight:700;'>"
            f"⚠️ {fv.TRIM_TEXT} — uppsidan mot Base är "
            f"{ev['upside_base']:.0f} %, under {fv.TRIM_UPSIDE_MIN:g} %.</div>",
            unsafe_allow_html=True)
    if mos is not None and not fv.mos_passes_gate(mos):
        st.caption(f"Köpgrindens steg 5 kräver {fv.MOS_BUY_MIN:g} % "
                   f"säkerhetsmarginal.")
    if not ev["what_must_go_right"]:
        st.caption(f"{fv.WMGR_MISSING} — steg 5 kan inte bli grönt utan det.")
