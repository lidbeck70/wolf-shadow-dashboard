"""
controls_ui.py — DS, AQS och CSM som återanvändbara sektioner.

Skrivna en gång och anropade från Poängmodellen, Lobo-arket och Rick Rule,
så att en ändrad tröskel inte behöver hittas på tre ställen. All logik ligger i
controls.py; det här är bara ytan.

Varje render-funktion muterar raden den får och returnerar True om något
ändrats, så anropande flik själv avgör när den sparar.
"""

from __future__ import annotations

import math
from typing import Optional

import streamlit as st

import storage

import controls as ctl
import lukacs
import lukacs_ui


def _opt(v) -> Optional[float]:
    """Nummerfältets startvärde: tomt när inget är ifyllt — 0 är ett svar."""
    return ctl._num(v)

TEXT, DIM = "#e8e4dc", "#8a8578"


def _band_badge(label: str, value, maximum: int, band: Optional[str],
                color: str) -> str:
    return (f"<div style='border:1px solid {color}55;background:{color}0d;"
            f"border-radius:8px;padding:8px 12px;margin:6px 0;'>"
            f"<span style='color:{color};font-weight:700;'>{label} "
            f"{value if value is not None else '–'}/{maximum}</span>"
            f"<span style='color:{color};font-weight:700;margin-left:10px;'>"
            f"{band or 'ej bedömd'}</span></div>")


def score_option(v, f) -> str:
    """Rullgardinens text: '1 — 12–24 mån' — kriteriet syns i valet, inte bara i hjälpen (mobilen har ingen
    hovring)."""
    return v if v == "–" else f"{v} — {(f.zero, f.one, f.two)[int(v)]}"


def _score_row(row: dict, fields, prefix: str, key: str,
               prefilled: tuple = ()) -> bool:
    """Ett fält per kolumn, med kriteriet i varje val och alla tre i hjälptexten."""
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
            format_func=lambda o, f=f: score_option(o, f),
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
def _shares_year(sheet: str, row: dict, key: str) -> Optional[int]:
    """Året aktieantalet i sifferuppdateringen kommer ur (senaste årsrapporten) — None i en äldre blob."""
    try:
        import refresh_ui
        sug = refresh_ui.suggestion(refresh_ui.load_refresh(), sheet, row, "shares_year", row_id=key)
        return int(sug[0]) if sug is not None else None
    except Exception:
        return None


def _runway_months(years: float) -> str:
    """Runwayen i hela månader för DS-texten — med decimaler bara när avrundningen skulle hamna på andra sidan
    fältets gränser (12 och 24 mån, ctl.ds_runway_suggestion): 11,96 mån står inte som '12 mån' bredvid '< 12 mån'."""
    m = years * 12.0
    for d in range(4):
        r = round(m, d)
        if (r >= 12) == (m >= 12) and (r > 24) == (m > 24):
            return f"{r:.{d}f}".replace(".", ",")
    return f"≈{m:.0f}"


def _musd_txt(v) -> str:
    """MUSD med en decimal och svenska tecken: 1 250,0."""
    return f"{float(v):,.1f}".replace(",", " ").replace(".", ",")


def refresh_runway(r) -> tuple:
    """(runway i år, källtext) ur sifferuppdateringens rad för arkraden: kassa ÷ burn, där burn är negativt FCF
    12 mån — samma scoring.runway_years som Sprott-kortet räknar med. (None, "") utan talen. Burn 0 (positivt
    FCF 12 mån) ger (math.inf, källtext): ingen förbränning, alltså runway över 24 mån (ägarens beslut) — kassan
    behövs inte då."""
    if not isinstance(r, dict):
        return None, ""
    asof = str(r.get("asof") or "")[:10]
    src = f"Börsdata{' ' + asof if asof else ''}"
    burn = ctl._num(r.get("burn_musd"))
    if burn == 0:
        cash = ctl._num(r.get("cash_musd"))
        return math.inf, (f"{src}: positivt FCF 12 mån, burn 0 MUSD/år"
                          + (f", kassa {_musd_txt(cash)} MUSD" if cash is not None and cash >= 0 else ""))
    try:
        import scoring
        years = scoring.runway_years(r.get("cash_musd"), r.get("burn_musd"))
    except Exception:
        return None, ""
    if years is None:
        return None, ""
    return years, (f"{src}: kassa {_musd_txt(r['cash_musd'])} MUSD ÷ burn {_musd_txt(r['burn_musd'])} MUSD/år "
                   f"(negativt FCF 12 mån) = runway {_runway_months(years)} mån")


def card_runway(cash, burn) -> tuple:
    """(runway i år, underlag) ur kortets egen kassa och burn (Poängmodellens Sprott-del) — samma
    scoring.runway_years som rutan 'Runway'. (None, "") utan båda talen och vid burn 0, som rutan."""
    try:
        import scoring
        years = scoring.runway_years(cash, burn)
    except Exception:
        return None, ""
    if years is None:
        return None, ""
    return years, (f"kortets kassa {_musd_txt(cash)} MUSD ÷ burn {_musd_txt(burn)} MUSD/år "
                   f"= runway {_runway_months(years)} mån")


def _match_txt(sheet: str, row: dict, row_id, blob=None) -> str:
    """' (matchat: Genovis AB)' när sifferuppdateringens bolag för arkraden inte liknar radens namn
    (refresh_ui.matched_note, samma som Börsdatas andra förslag på kortet). Oescapad: refresh_ui.offer escapar
    underlaget själv ('AT&T' blir annars 'AT&amp;amp;T'). Tom sträng annars, och utan sifferuppdatering."""
    try:
        import html
        import refresh_ui
        if blob is None:
            blob = refresh_ui.load_refresh()
        return html.unescape(refresh_ui.matched_note(blob, sheet, row, row_id))
    except Exception:
        return ""


def sheet_runway(sheet: str, row_id, row: Optional[dict] = None) -> tuple:
    """refresh_runway för arkraden '<sheet>:<id>' (Tiggre, Rick Rule) — (None, "") utan sifferuppdatering. Med row
    slutar källtexten med '(matchat: …)' när nattjobbets bolag inte liknar radens namn — ett klick sätter DS-poäng."""
    try:
        import refresh_ui
        blob = refresh_ui.load_refresh() or {}
        r = (blob.get("rows") or {}).get(f"{sheet}:{row_id}")
    except Exception:
        blob, r = {}, None
    years, note = refresh_runway(r)
    if years is not None and row is not None:
        note += _match_txt(sheet, row, row_id, blob)
    return years, note


def _crit(f, points: int) -> str:
    """Fältets tryckta kriterium för poängen, med liten bokstav mitt i en mening: 'serieutspädare', '> 24 mån'."""
    t = (f.zero, f.one, f.two)[points]
    return t[:1].lower() + t[1:]


def _ds_offer(row: dict, key: str, prefix: str, field: str, points, basis: str) -> None:
    """Ett DS-förslag med Använd (refresh_ui.offer): '🤖 DS: Runway 12–24 mån → 1 riskpoäng — <underlaget>'.
    Underlaget (tal, källa, datum) står alltid med — ett klick sätter DS-poäng, och DS ≥ 6 låser köp
    (ctl.ds_blocks_buy). Klicket skriver fältet och rullgardinen; att rita förslaget skriver ingenting."""
    f = next((x for x in ctl.DS_FIELDS if x.key == field), None)
    if f is None or points not in (0, 1, 2):
        return
    try:
        import refresh_ui
    except Exception:
        return
    refresh_ui.offer(row, field, points, f.label, key=f"ds_{prefix}_{key}_{field}", source="DS",
                     widget_key=f"{prefix}_{key}_{field}", cast=int, note=basis,
                     fmt=lambda p, f=f: f"{_crit(f, p)} → {p} riskpoäng")


def _pct_txt(g: float, limits: tuple = ()) -> str:
    """Förändringen i hela procent med tecken — med decimaler bara när avrundningen skulle hamna på andra sidan en
    av fältets gränser eller på 0 (scorecard._edge_digits, kantregeln): +25,4 står inte som '+25' bredvid
    '> +25 %', och -0,4 inte som '-0'. Varje gräns prövas med både >= och >, eftersom 3-årsbanden bryter vid < 10
    och <= 25 men 5-årsbanden vid >= 20 och > 50."""
    import scorecard
    g = 0.0 if g == 0 else float(g)                                   # -0,0 ur round(-0,04, 1) blir 0
    d = scorecard._edge_digits(g, 0, [(x, op) for x in (0.0, *limits) for op in (">=", ">")])
    return f"≈{g:+.0f}" if d is None else f"{g:+.{d}f}".replace(".", ",")


def _shares_basis(sug: tuple, year: Optional[int], back: int, limits: tuple = (), match: str = "") -> str:
    """Underlaget för 'Aktier 3 år' och 'Historik': Börsdatas datum, årsrapportens år och förändringen (_pct_txt mot
    fältets gränser limits). match = '(matchat: …)' när nattjobbets bolag inte liknar radens namn. Antalet ur
    årsrapporten släpar (senare emissioner syns inte), och en omvänd split ser ut som en minskning."""
    g, asof = sug[0], sug[1]
    pct = _pct_txt(g, limits)
    if year:
        span = f"i årsrapporten {year} är {pct} % mot {year - back}"
    else:
        span = f"är {pct} % " + ("mot för tre år sedan" if back == 3 else "på fem år")
    return (f"Börsdata{' ' + asof if asof else ''}: antalet aktier {span}{match}"
            + (" — en minskning kan vara en omvänd split" if g < 0 else ""))


def _refresh_pct(sheet: str, row: dict, field: str, row_id=None):
    """(värde, asof, valuta) ur sifferuppdateringen för arkraden, eller None (ingen blob, inget tal)."""
    try:
        import refresh_ui
        return refresh_ui.suggestion(refresh_ui.load_refresh(), sheet, row, field, row_id=row_id)
    except Exception:
        return None


def render_ds(row: dict, key: str, runway_years=None,
              prefix: str = "ds", sheet: str = "", runway_note: str = "") -> bool:
    """Dilution Score. runway_years ger ett förslag på första fältet (runway_note säger varifrån talet kommer;
    math.inf = ingen förbränning); sheet gör att "Aktier 3 år" och "Historik" kan föreslås ur Börsdatas
    aktiehistorik. Förslagen har Använd och visas bara för en obesvarad delfråga — inget fylls i när sidan ritas."""
    changed = False
    st.markdown(
        f"<b style='color:{TEXT};'>DS — Dilution Score</b> "
        f"<span style='color:{DIM};font-size:0.78rem;'>— riskpoäng 0–"
        f"{ctl.DS_MAX}, <b>lägre är bättre</b>. Hur mycket av uppsidan späds "
        f"bort innan tesen hinner spela ut?</span>", unsafe_allow_html=True)

    suggestion = ctl.ds_runway_suggestion(runway_years)
    if suggestion is not None and row.get("ds_runway") is None:
        y = ctl._num(runway_years)
        basis = runway_note or ("ingen förbränning" if math.isinf(y) else f"runway {_runway_months(y)} mån")
        _ds_offer(row, key, prefix, "ds_runway", suggestion, basis)
    # '(matchat: …)' på aktieförslagen när nattjobbets bolag inte liknar radens namn (runwayen har den i
    # runway_note ur sheet_runway) — som Börsdatas andra förslag på kortet.
    mm = _match_txt(sheet, row, key) if sheet else ""
    if sheet and row.get("ds_aktier_3ar") is None:
        sug = _refresh_pct(sheet, row, "shares_growth_3y_pct", row_id=key)
        if sug is not None:
            # gränserna i ctl.ds_shares_suggestion: < 10 → 0, ≤ 25 → 1, > 25 → 2
            _ds_offer(row, key, prefix, "ds_aktier_3ar", ctl.ds_shares_suggestion(sug[0]),
                      _shares_basis(sug, _shares_year(sheet, row, key), 3, (10.0, 25.0), mm))
    if sheet and row.get("ds_historik") is None:
        sug5 = _refresh_pct(sheet, row, "shares_growth_5y_pct", row_id=key)
        if sug5 is not None:
            import judgment_hints as jh
            _ds_offer(row, key, prefix, "ds_historik", jh.ds_history_points(sug5[0]),
                      _shares_basis(sug5, _shares_year(sheet, row, key), 5,
                                    (jh.DILUTION_5Y_MIXED_PCT, jh.DILUTION_5Y_SERIAL_PCT), mm)
                      + f". Grova gränser på 5 år: +{jh.DILUTION_5Y_MIXED_PCT:g} % blandad, "
                        f"över +{jh.DILUTION_5Y_SERIAL_PCT:g} % serieutspädare")

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
                             value=_opt(cur),
                             key=f"{prefix}_{key}_{ikey}")
        if storage.differs(v, cur):
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
def _aqs_offers(row: dict, key: str, prefill: Optional[dict], prefix: str, strategy: str) -> None:
    """Förslagen på jurisdiktion/management ur frågorna fliken redan ställt, med Använd (refresh_ui.offer) — bara
    för en obedömd delfråga. Klicket skriver fältet och rullgardinen; att rita förslaget skriver ingenting."""
    if not prefill:
        return
    try:
        import refresh_ui
    except Exception:
        return
    fields = {f.key: f for f in ctl.AQS_FIELDS}
    notes = ctl.AQS_PREFILL_NOTE.get((strategy or "").strip().lower(), {})
    for k, v in prefill.items():
        f = fields.get(k)
        if f is None or row.get(k) not in (None, ""):
            continue
        refresh_ui.offer(row, k, v, f.label, key=f"aqs_{prefix}_{key}_{k}", source="Kortet",
                         widget_key=f"{prefix}_{key}_{k}", cast=int, note=notes.get(k, ""),
                         fmt=lambda o, f=f: f"{o} ({(f.zero, f.one, f.two)[o]})" if o in (0, 1, 2) else str(o))


def _aqs_life_offer(row: dict, key: str, prefix: str, life_years) -> None:
    """'Livslängd / resurs' ur kortets gruvlivslängd (ctl.aqs_life_suggestion, fältets tryckta gränser), med
    Använd — bara för en obedömd delfråga. Talet är gruvplanens/reservernas livslängd, inte resursens: en lång
    resurs är inte en lång reserv, och det står i texten."""
    pts = ctl.aqs_life_suggestion(life_years)
    if pts is None or row.get("aqs_livslangd") not in (None, ""):
        return
    try:
        import refresh_ui
    except Exception:
        return
    f = next(x for x in ctl.AQS_FIELDS if x.key == "aqs_livslangd")
    yrs = f"{ctl._num(life_years):g}".replace(".", ",")
    refresh_ui.offer(row, f.key, pts, f.label, key=f"aqs_{prefix}_{key}_{f.key}", source="Kortet",
                     widget_key=f"{prefix}_{key}_{f.key}", cast=int,
                     note=f"ur Gruvlivslängd {yrs} år: reservernas livslängd (LOM), inte hela resursen",
                     fmt=lambda o, f=f: f"{o} ({(f.zero, f.one, f.two)[o]})")


def render_aqs(row: dict, key: str, prefill: Optional[dict] = None,
               prefix: str = "aqs", strategy: str = "", life_years=None) -> bool:
    """Asset Quality Score. prefill = förslag på jurisdiktion/management ur den flik som redan frågat (strategy
    säger vilken, för underlaget i texten), life_years = kortets gruvlivslängd för 'Livslängd / resurs' — visas
    med Använd, inget fylls i när sektionen ritas."""
    changed = False
    st.markdown(
        f"<b style='color:{TEXT};'>AQS — Asset Quality Score</b> "
        f"<span style='color:{DIM};font-size:0.78rem;'>— 0–{ctl.AQS_MAX}. "
        f"Tillgångens kvalitet, oberoende av vad den kostar.</span>",
        unsafe_allow_html=True)

    _aqs_offers(row, key, prefill, prefix, strategy)
    _aqs_life_offer(row, key, prefix, life_years)

    half = len(ctl.AQS_FIELDS) // 2
    changed |= _score_row(row, ctl.AQS_FIELDS[:half], prefix, key)
    changed |= _score_row(row, ctl.AQS_FIELDS[half:], prefix, key)

    total = ctl.aqs_total(row)
    band = ctl.aqs_band(total)
    st.markdown(_band_badge("AQS", total, ctl.AQS_MAX, band,
                            ctl.AQS_BAND_COLOR.get(band, DIM)),
                unsafe_allow_html=True)
    return changed


# ── CSM ──────────────────────────────────────────────────────────────────────
NAV_PRICE_SCALE_MAX = 10.0      # NAV-priset mer än så många gånger från spot står nog i en annan enhet


def _price_txt(v: float) -> str:
    """Pris med decimalkomma: hela tal från 100 (4 012), två decimaler under (4,52)."""
    return f"{v:,.{0 if v >= 100 else 2}f}".replace(",", " ").replace(".", ",")


def spot_context(ref, row: dict) -> str:
    """Kontextraden över CSM-scenarierna: 'Spot i dag X USD/oz (datum) · NAV räknad på Y'. Spot är nattens
    råvarupris på producentraden (commodity_price/commodity_unit, Yahoo-termin), Y kortets eget 'Metallpris i NAV'.
    Inget fylls i. Tom sträng utan spot, och när nattens pris inte gäller kortets råvara (råvaran bytt sedan natten,
    eller ingen Yahoo-termin: commodity_prices.night_fits) — USD/lb jämförs aldrig med USD/oz. Ett NAV-pris mer än
    NAV_PRICE_SCALE_MAX× från spot visas inte som jämförelse."""
    r = ref if isinstance(ref, dict) else {}
    spot, unit = ctl._num(r.get("commodity_price")), str(r.get("commodity_unit") or "").strip()
    if spot is None or spot <= 0 or not unit:
        return ""
    try:
        import commodity_prices
        fits = commodity_prices.night_fits(r, (row or {}).get("commodity"))
    except Exception:
        fits = False
    if not fits:
        return ""
    asof = str(r.get("commodity_asof") or "")[:10]
    txt = f"🤖 Spot i dag {_price_txt(spot)} {unit}{f' ({asof})' if asof else ''}"
    nav = ctl._num((row or {}).get("nav_price"))
    if nav is not None and nav > 0:
        if nav / spot > NAV_PRICE_SCALE_MAX or spot / nav > NAV_PRICE_SCALE_MAX:
            txt += f" · NAV-priset {_price_txt(nav)} ser ut att stå i en annan enhet än {unit}"
        else:
            txt += f" · NAV räknad på {_price_txt(nav)}"
    return txt


def render_csm(row: dict, key: str, kind: str = ctl.PRODUCER,
               prefix: str = "csm", with_fv: bool = False, sheet: str = "") -> bool:
    """Commodity Sensitivity Matrix. Tre scenarier, fem för kärninnehav.

    with_fv lägger Lukacs FV-modulen sist i sektionen: CSM säger om bolaget
    överlever varje scenario, FV vad det är värt där.
    """
    changed = False
    # Läggs i raden först när något fylls i — att öppna arket ska inte skriva en tom matris.
    matrix = row.get("csm") if isinstance(row.get("csm"), dict) else {}

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
    cur_kind = (row.get("csm_kind") if row.get("csm_kind") in kinds
                else (kind if kind in kinds else ctl.PRODUCER))
    if new_kind != cur_kind or is_core != bool(row.get("is_core")):
        row["csm_kind"], row["is_core"] = new_kind, is_core
        changed = True

    if sheet == "producers":                            # spot och kortets NAV-pris — bara visning, inga rutor fylls
        ctx = spot_context(lukacs_ui._refresh_row(sheet, key), row)
        if ctx:
            st.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>{ctx}</span>", unsafe_allow_html=True)
    if new_kind == ctl.PRODUCER and sheet:
        lukacs_ui.bd_fcf_line(row, key, sheet)          # referens vid FCF-rutorna, ingen knapp
    for s in ctl.scenarios(is_core):
        sc = matrix.get(s) if isinstance(matrix.get(s), dict) else {}
        # Synliga korta etiketter med scenariot: på mobilen staplas kolumnerna och rutorna måste säga vad de är
        cols = st.columns(4, vertical_alignment="bottom")
        cols[0].markdown(f"<div style='padding-bottom:8px;color:{TEXT};"
                         f"font-weight:600;'>{s}</div>",
                         unsafe_allow_html=True)
        price = cols[1].number_input(f"Råvarupris ({s})", min_value=0.0,
                                     step=1.0,
                                     value=_opt(sc.get("price")),
                                     key=f"{prefix}_{key}_{s}_price")
        if new_kind == ctl.PRODUCER:
            fcf = cols[2].number_input(f"FCF MUSD ({s})", step=1.0,
                                       value=_opt(sc.get("fcf_musd")),
                                       key=f"{prefix}_{key}_{s}_fcf")
            vals = {"price": price, "fcf_musd": fcf}
        else:
            nav = cols[2].number_input(f"NAV MUSD ({s})", step=1.0,
                                       value=_opt(sc.get("nav_musd")),
                                       key=f"{prefix}_{key}_{s}_nav")
            need = cols[3].number_input(f"Fin.behov MUSD ({s})",
                                        min_value=0.0, step=1.0,
                                        value=_opt(sc.get("financing_need")),
                                        key=f"{prefix}_{key}_{s}_need")
            vals = {"price": price, "nav_musd": nav, "financing_need": need}
        if any(storage.differs(v, sc.get(k))
               for k, v in vals.items()):
            sc.update(vals)
            matrix[s] = sc
            row["csm"] = matrix
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
            row["csm"] = matrix
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

    if with_fv and new_kind == ctl.PRODUCER:
        st.markdown("<hr style='border-color:rgba(138,133,120,0.2);"
                    "margin:14px 0 8px;'>", unsafe_allow_html=True)
        changed |= lukacs_ui.render_fv(row, key, sheet=sheet)
    elif with_fv:
        st.caption("Lukacs FV värderar producenters kassaflöde — en utvecklare "
                   "värderas på NAV och finansieringsbehov i matrisen ovan.")
    return changed


# Vilket ark (sifferuppdateringens nyckel) en strategi granskas i.
SHEET_BY_STRATEGY = {"producenter": "producers", "royalty": "producers",
                     "tiggre": "tiggre", "sprott": "scoring", "durrett": "scoring",
                     "insider": "insider"}


def render_all(row: dict, key: str, position_pct=None, strategy: str = "",
               runway_years=None, aqs_prefill: Optional[dict] = None,
               kind: str = ctl.PRODUCER, runway_note: str = "", life_years=None) -> bool:
    """Alla kontroller som proportionalitetsregeln kräver för positionen. life_years = kortets gruvlivslängd
    (Rick Rule) till AQS-förslaget på 'Livslängd / resurs'."""
    req = ctl.required_sections(position_pct, strategy,
                                bool(row.get("dilution_risk")))
    changed = False
    if ctl.SEC_DS in req:
        with st.expander("🧪 DS — utspädningsrisk", expanded=False):
            changed |= render_ds(row, key, runway_years,
                                 sheet=SHEET_BY_STRATEGY.get(strategy, ""), runway_note=runway_note)
    if ctl.SEC_AQS in req:
        with st.expander("⛏ AQS — tillgångskvalitet", expanded=False):
            changed |= render_aqs(row, key, aqs_prefill, strategy=strategy, life_years=life_years)
    fv_req = lukacs.fv_required(position_pct, strategy)
    if ctl.SEC_CSM in req:
        with st.expander("📉 CSM — råvarukänslighet"
                         + (" · Lukacs FV" if fv_req else ""), expanded=False):
            changed |= render_csm(row, key, kind, with_fv=fv_req,
                                  sheet=SHEET_BY_STRATEGY.get(strategy, ""))
    elif lukacs.fv_applicable(strategy):
        # Under 2 % är FV frivillig — proportionalitetsregeln. Men den ska gå
        # att öppna ändå: en liten position i ett case man vill räkna på är
        # inte ett skäl att stänga verktyget.
        open_fv = st.checkbox(
            "Räkna Lukacs FV ändå", value=bool(row.get("fv_manual")),
            key=f"fvman_{key}",
            help=f"FV krävs först över {ctl.FULL_WORK_MIN_PCT:g} % av totalen. "
                 f"Under det är den frivillig — och räknas inte in i "
                 f"köpgrinden.")
        if open_fv != bool(row.get("fv_manual")):
            row["fv_manual"] = open_fv
            changed = True
        if open_fv:
            with st.expander("💵 Lukacs FV — fair value", expanded=True):
                changed |= lukacs_ui.render_fv(row, key, sheet=SHEET_BY_STRATEGY.get(strategy, ""))
    if req == {ctl.SEC_STRATEGY}:
        st.caption("Proportionalitetsregeln: den här strategin behöver ingen "
                   "AQS eller CSM. Kryssa emissionsrisk om DS ska visas.")
    return changed
