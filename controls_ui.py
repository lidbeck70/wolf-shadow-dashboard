"""
controls_ui.py — DS, AQS och CSM som återanvändbara sektioner.

Skrivna en gång och anropade från Poängmodellen, Lobo-arket och Rick Rule,
så att en ändrad tröskel inte behöver hittas på tre ställen. All logik ligger i
controls.py; det här är bara ytan.

Varje render-funktion muterar raden den får och returnerar True om något
ändrats, så anropande flik själv avgör när den sparar.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

import controls as ctl

TEXT, DIM = "#e8e4dc", "#8a8578"


def _band_badge(label: str, value, maximum: int, band: Optional[str],
                color: str) -> str:
    return (f"<div style='border:1px solid {color}55;background:{color}0d;"
            f"border-radius:8px;padding:8px 12px;margin:6px 0;'>"
            f"<span style='color:{color};font-weight:700;'>{label} "
            f"{value if value is not None else '–'}/{maximum}</span>"
            f"<span style='color:{color};font-weight:700;margin-left:10px;'>"
            f"{band or 'ej bedömd'}</span></div>")


def _score_row(row: dict, fields, prefix: str, key: str,
               prefilled: tuple = ()) -> bool:
    """Ett fält per kolumn, med kriterierna som hjälptext."""
    changed = False
    cols = st.columns(len(fields))
    opts = ["–", 0, 1, 2]
    for f, col in zip(fields, cols):
        cur = row.get(f.key)
        cur = None if cur in (None, "") else int(cur)
        idx = opts.index(cur) if cur in (0, 1, 2) else 0
        label = f.label + (" ↩" if f.key in prefilled else "")
        v = col.selectbox(
            label, opts, index=idx, key=f"{prefix}_{key}_{f.key}",
            help=f"{f.help}\n\n0: {f.zero}\n\n1: {f.one}\n\n2: {f.two}")
        new = None if v == "–" else int(v)
        if new != cur:
            if new is None:
                row.pop(f.key, None)
            else:
                row[f.key] = new
            changed = True
    return changed


# ── DS ───────────────────────────────────────────────────────────────────────
def render_ds(row: dict, key: str, runway_years=None,
              prefix: str = "ds") -> bool:
    """Dilution Score. runway_years ger ett förslag på första fältet."""
    changed = False
    st.markdown(
        f"<b style='color:{TEXT};'>DS — Dilution Score</b> "
        f"<span style='color:{DIM};font-size:0.78rem;'>— riskpoäng 0–"
        f"{ctl.DS_MAX}, <b>lägre är bättre</b>. Hur mycket av uppsidan späds "
        f"bort innan tesen hinner spela ut?</span>", unsafe_allow_html=True)

    suggestion = ctl.ds_runway_suggestion(runway_years)
    if suggestion is not None and row.get("ds_runway") is None:
        st.caption(f"Runwayen motsvarar {suggestion} riskpoäng på första "
                   f"fältet — förslag, inte ifyllt åt dig.")

    changed |= _score_row(row, ctl.DS_FIELDS, prefix, key)

    total = ctl.ds_total(row)
    band = ctl.ds_band(total)
    st.markdown(_band_badge("DS", total, ctl.DS_MAX, band,
                            ctl.DS_BAND_COLOR.get(band, DIM)),
                unsafe_allow_html=True)

    i1, i2 = st.columns(2)
    for (ikey, ilabel), col in zip(ctl.DS_INFO_FIELDS, (i1, i2)):
        cur = row.get(ikey)
        v = col.number_input(ilabel, min_value=0.0, step=1.0,
                             value=float(cur) if cur not in (None, "") else 0.0,
                             key=f"{prefix}_{key}_{ikey}")
        if v != cur:
            row[ikey] = v
            changed = True

    note = ctl.ds_note(row)
    if note:
        if ctl.ds_blocks_buy(row):
            st.error(note)
            f1, f2 = st.columns([2, 1])
            txt = f1.text_input("Finansieringskatalysator",
                                value=row.get("fin_catalyst_text", ""),
                                key=f"{prefix}_{key}_fincat",
                                placeholder="t.ex. Riktad emission fulltecknad")
            dat = f2.text_input("När", value=row.get("fin_catalyst_date", ""),
                                key=f"{prefix}_{key}_findate",
                                placeholder="ÅÅÅÅ-MM")
            if (txt != row.get("fin_catalyst_text", "")
                    or dat != row.get("fin_catalyst_date", "")):
                row["fin_catalyst_text"], row["fin_catalyst_date"] = txt, dat
                changed = True
        else:
            st.warning(f"{note} — {row.get('fin_catalyst_text', '')} "
                       f"({row.get('fin_catalyst_date', '')})")
    return changed


# ── AQS ──────────────────────────────────────────────────────────────────────
def render_aqs(row: dict, key: str, prefill: Optional[dict] = None,
               prefix: str = "aqs") -> bool:
    """Asset Quality Score. prefill fyller jurisdiktion/management från den
    flik som redan frågat, i stället för att fråga igen."""
    changed = False
    st.markdown(
        f"<b style='color:{TEXT};'>AQS — Asset Quality Score</b> "
        f"<span style='color:{DIM};font-size:0.78rem;'>— 0–{ctl.AQS_MAX}. "
        f"Tillgångens kvalitet, oberoende av vad den kostar.</span>",
        unsafe_allow_html=True)

    for k, v in (prefill or {}).items():
        if row.get(k) is None:
            row[k] = v
            changed = True
    if prefill:
        st.caption("↩ = hämtad från fliken som redan frågat. Skriv över om du "
                   "vill — den frågan ställs inte två gånger.")

    half = len(ctl.AQS_FIELDS) // 2
    changed |= _score_row(row, ctl.AQS_FIELDS[:half], prefix, key,
                          ctl.AQS_PREFILLED)
    changed |= _score_row(row, ctl.AQS_FIELDS[half:], prefix, key,
                          ctl.AQS_PREFILLED)

    total = ctl.aqs_total(row)
    band = ctl.aqs_band(total)
    st.markdown(_band_badge("AQS", total, ctl.AQS_MAX, band,
                            ctl.AQS_BAND_COLOR.get(band, DIM)),
                unsafe_allow_html=True)
    return changed


# ── CSM ──────────────────────────────────────────────────────────────────────
def render_csm(row: dict, key: str, kind: str = ctl.PRODUCER,
               prefix: str = "csm") -> bool:
    """Commodity Sensitivity Matrix. Tre scenarier, fem för kärninnehav."""
    changed = False
    matrix = row.setdefault("csm", {})

    st.markdown(
        f"<b style='color:{TEXT};'>CSM — råvarukänslighet</b> "
        f"<span style='color:{DIM};font-size:0.78rem;'>— överlever caset "
        f"Bear, och hur mycket hävstång ger Bull?</span>",
        unsafe_allow_html=True)

    k1, k2 = st.columns(2)
    kinds = [ctl.PRODUCER, ctl.DEVELOPER]
    new_kind = k1.selectbox("Typ", kinds,
                            index=kinds.index(kind if kind in kinds
                                              else ctl.PRODUCER),
                            key=f"{prefix}_{key}_kind")
    is_core = k2.checkbox("Kärninnehav (fem scenarier)",
                          value=bool(row.get("is_core")),
                          key=f"{prefix}_{key}_core")
    if new_kind != row.get("csm_kind") or is_core != bool(row.get("is_core")):
        row["csm_kind"], row["is_core"] = new_kind, is_core
        changed = True

    for s in ctl.scenarios(is_core):
        sc = matrix.setdefault(s, {})
        cols = st.columns(4)
        cols[0].markdown(f"<div style='padding-top:8px;color:{TEXT};"
                         f"font-weight:600;'>{s}</div>",
                         unsafe_allow_html=True)
        price = cols[1].number_input(f"Råvarupris ({s})", min_value=0.0,
                                     step=1.0,
                                     value=float(sc.get("price") or 0.0),
                                     key=f"{prefix}_{key}_{s}_price",
                                     label_visibility="collapsed")
        if new_kind == ctl.PRODUCER:
            fcf = cols[2].number_input(f"FCF MUSD ({s})", step=1.0,
                                       value=float(sc.get("fcf_musd") or 0.0),
                                       key=f"{prefix}_{key}_{s}_fcf",
                                       label_visibility="collapsed")
            vals = {"price": price, "fcf_musd": fcf}
        else:
            nav = cols[2].number_input(f"NAV MUSD ({s})", step=1.0,
                                       value=float(sc.get("nav_musd") or 0.0),
                                       key=f"{prefix}_{key}_{s}_nav",
                                       label_visibility="collapsed")
            need = cols[3].number_input(f"Finansieringsbehov ({s})",
                                        min_value=0.0, step=1.0,
                                        value=float(sc.get("financing_need") or 0.0),
                                        key=f"{prefix}_{key}_{s}_need",
                                        label_visibility="collapsed")
            vals = {"price": price, "nav_musd": nav, "financing_need": need}
        if any(sc.get(k) != v for k, v in vals.items()):
            sc.update(vals)
            changed = True

    st.caption("Kolumner: scenario · råvarupris · "
               + ("fritt kassaflöde" if new_kind == ctl.PRODUCER
                  else "NAV · finansieringsbehov"))

    secured = st.checkbox("Kassan räcker genom Bear (säkrad finansiering)",
                          value=bool(row.get("secured_cash")),
                          key=f"{prefix}_{key}_secured")
    if secured != bool(row.get("secured_cash")):
        row["secured_cash"] = secured
        changed = True

    suggestion = ctl.bear_survival_suggestion(new_kind, matrix.get(ctl.BEAR, {}),
                                              secured)
    opts = ["– (följ förslaget)", "Ja", "Nej"]
    cur = matrix.get("bear_survival")
    idx = 1 if cur is True else (2 if cur is False else 0)
    pick = st.selectbox(
        "Överlever Bear?", opts, index=idx,
        key=f"{prefix}_{key}_bearsurv",
        help=f"Förslag ur matrisen: "
             f"{'Ja' if suggestion else 'Nej' if suggestion is False else '–'}")
    new_val = None if pick.startswith("–") else (pick == "Ja")
    if new_val != cur:
        if new_val is None:
            matrix.pop("bear_survival", None)
        else:
            matrix["bear_survival"] = new_val
        changed = True

    if ctl.csm_red_flag(new_kind, matrix, secured):
        st.error(ctl.CSM_BEAR_FAIL)

    lev = ctl.leverage_ratio(matrix)
    if lev is not None:
        st.metric("Hävstångskvot (Bull/Bear FCF)", f"{lev:.2f}×",
                  help="Skiljer billigt från billigt MED hävstång.")
    elif new_kind == ctl.PRODUCER:
        st.caption("Hävstångskvoten kräver ett positivt Bear-kassaflöde — "
                   "en kvot mot noll är inte oändlig hävstång.")
    return changed


def render_all(row: dict, key: str, position_pct=None, strategy: str = "",
               runway_years=None, aqs_prefill: Optional[dict] = None,
               kind: str = ctl.PRODUCER) -> bool:
    """Alla kontroller som proportionalitetsregeln kräver för positionen."""
    req = ctl.required_sections(position_pct, strategy,
                                bool(row.get("dilution_risk")))
    changed = False
    if ctl.SEC_DS in req:
        with st.expander("🧪 DS — utspädningsrisk", expanded=False):
            changed |= render_ds(row, key, runway_years)
    if ctl.SEC_AQS in req:
        with st.expander("⛏ AQS — tillgångskvalitet", expanded=False):
            changed |= render_aqs(row, key, aqs_prefill)
    if ctl.SEC_CSM in req:
        with st.expander("📉 CSM — råvarukänslighet", expanded=False):
            changed |= render_csm(row, key, kind)
    if req == {ctl.SEC_STRATEGY}:
        st.caption("Proportionalitetsregeln: den här strategin behöver ingen "
                   "AQS eller CSM. Kryssa emissionsrisk om DS ska visas.")
    return changed
