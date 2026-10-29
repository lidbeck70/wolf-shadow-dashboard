"""
portfolio/metals.py — fysiskt guld och silver: vad du köpt, för hur mycket, och vad det är värt i dag.

Raderna sparas i data/metals.json — krypterat (private_data.py), eftersom repot är publikt. Värdet räknas på
spotpriset: terminen (GC=F/SI=F, USD per troy ounce) gånger USDSEK, per gram finvikt. Handlarens återköpspris
ligger oftast några procent under spot, och köppriset innehöll en premie — avkastningen mot spot är därför en
övre gräns för vad en försäljning ger.
"""
from __future__ import annotations

import uuid
from typing import Callable, Optional

import pandas as pd

STORE = "metals"
DEFAULTS: dict = {"rows": []}
TROY_OZ_G = 31.1034768
UNITS = {"g": 1.0, "oz": TROY_OZ_G, "kg": 1000.0}
METALS = {"guld": "GC=F", "silver": "SI=F"}
PURITY_DEFAULT = {"guld": 0.9999, "silver": 0.999}


def fine_grams(row: dict) -> float:
    """Finvikt i gram: antal × vikt × enhet × finhalt."""
    return (float(row.get("count") or 1) * float(row.get("weight") or 0) * UNITS.get(row.get("unit") or "g", 1.0)
            * float(row.get("purity") or PURITY_DEFAULT.get(row.get("metal"), 1.0)))


def spot_sek_per_gram(getter: Optional[Callable] = None) -> dict:
    """{metall: (SEK per gram, datum)} ur terminen och USDSEK — tomt för det som inte gick att hämta."""
    if getter is None:
        from market_prices import close as getter
    out = {}
    try:
        fx = getter("USDSEK=X", "1mo")
    except Exception:
        fx = None
    if fx is None or not len(fx):
        return out
    usdsek = float(fx.iloc[-1])
    for metal, sym in METALS.items():
        try:
            px = getter(sym, "1mo")
        except Exception:
            px = None
        if px is not None and len(px):
            out[metal] = (float(px.iloc[-1]) * usdsek / TROY_OZ_G, str(pd.Timestamp(px.index[-1]).date()))
    return out


def valuation(rows: list, spot: dict, today=None) -> list:
    """En rad per innehav: finvikt, köp, värde, vinst kr och %, årstakt (från ett år) och break-even per gram."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    out = []
    for r in rows or []:
        g = fine_grams(r)
        cost = float(r.get("cost_sek") or 0)
        px = (spot.get(r.get("metal")) or (None, None))[0]
        value = g * px if px is not None else None
        years = (today - pd.Timestamp(r["date"])).days / 365.25 if r.get("date") else None
        gain = value - cost if value is not None else None
        pct = (value / cost - 1) * 100 if value is not None and cost > 0 else None
        cagr = ((value / cost) ** (1 / years) - 1) * 100 if value and cost > 0 and years and years >= 1 else None
        out.append({"id": r.get("id"), "Metall": r.get("metal"), "Beskrivning": r.get("desc") or "",
                    "Antal": r.get("count"), "Vikt": f"{r.get('weight')} {r.get('unit')}", "Finhalt": r.get("purity"),
                    "Finvikt (g)": round(g, 2), "Köpt": r.get("date") or "—", "Köp (kr)": round(cost),
                    "Värde spot (kr)": round(value) if value is not None else None,
                    "Vinst (kr)": round(gain) if gain is not None else None,
                    "Avkastning %": round(pct, 1) if pct is not None else None,
                    "Årstakt %": round(cagr, 1) if cagr is not None else None,
                    "Break-even (kr/g)": round(cost / g, 1) if g else None})
    return out


def totals(vals: list) -> dict:
    """{metall: {köp, värde, vinst, %}} och "alla"."""
    out = {}
    for key in sorted({v["Metall"] for v in vals}) + ["alla"]:
        sel = [v for v in vals if key == "alla" or v["Metall"] == key]
        cost = sum(v["Köp (kr)"] or 0 for v in sel)
        known = [v for v in sel if v["Värde spot (kr)"] is not None]
        value = sum(v["Värde spot (kr)"] for v in known) if known and len(known) == len(sel) else None
        out[key] = {"cost": cost, "value": value, "gain": value - cost if value is not None else None,
                    "pct": round((value / cost - 1) * 100, 1) if value is not None and cost > 0 else None,
                    "grams": round(sum(v["Finvikt (g)"] for v in sel), 2)}
    return out


def new_row(metal: str, desc: str, count: int, weight: float, unit: str, purity: float, cost_sek: float,
            date) -> dict:
    if metal not in METALS or unit not in UNITS or not weight or weight <= 0 or count < 1 or cost_sek < 0:
        raise ValueError("ogiltig rad")
    if not 0 < purity <= 1:
        raise ValueError("finhalten anges som andel, t.ex. 0,9999")
    return {"id": uuid.uuid4().hex[:8], "metal": metal, "desc": desc.strip(), "count": int(count),
            "weight": float(weight), "unit": unit, "purity": float(purity), "cost_sek": float(cost_sek),
            "date": str(pd.Timestamp(date).date())}


def render_metals_section() -> None:
    import streamlit as st

    import storage
    import storage_ui
    from thor import cards
    from ui.components import note
    from ui.tokens import CYAN, DIM, GOLD, GREEN, RED
    st.markdown("#### 🥇 Fysiska metaller")
    data = storage.session_load(STORE, dict(DEFAULTS))
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        data = dict(DEFAULTS)
        st.session_state[STORE] = data
    storage_ui.save_bar(STORE, "Metaller", key="save_metals")

    @st.cache_data(ttl=3600, show_spinner=False)
    def _spot():
        try:
            return spot_sek_per_gram()
        except Exception:
            return {}
    spot = _spot()
    vals = valuation(data["rows"], spot)
    if vals:
        tot = totals(vals)
        a = tot["alla"]
        col = GREEN if (a["gain"] or 0) >= 0 else RED
        kpis = [("Värde (spot)", f"{a['value']:,.0f} kr".replace(",", " ") if a["value"] is not None else "—", GOLD,
                 f"köpt för {a['cost']:,.0f} kr".replace(",", " ")),
                ("Avkastning", f"{a['pct']:+.1f} %" if a["pct"] is not None else "—", col,
                 f"{a['gain']:+,.0f} kr".replace(",", " ") if a["gain"] is not None else "")]
        for m in ("guld", "silver"):
            if m in tot:
                px = spot.get(m)
                kpis.append((m.capitalize(), f"{tot[m]['grams']:,.0f} g".replace(",", " "), CYAN,
                             f"spot {px[0]:,.1f} kr/g · {px[1]}".replace(",", " ") if px else "spot saknas"))
        st.markdown(cards.kpi_row(kpis), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != "id"} for r in vals]), hide_index=True,
                     width="stretch")
        labels = {r["id"]: f"{r['metal']} · {r.get('desc') or ''} · {r.get('date')}" for r in data["rows"]}
        pick = st.selectbox("Ta bort ett innehav", ["—"] + list(labels), format_func=lambda k: labels.get(k, k),
                            key="metals_remove_pick")
        if pick != "—" and st.button("Ta bort", key="metals_remove_btn"):
            data["rows"] = [r for r in data["rows"] if r.get("id") != pick]
            st.session_state[STORE] = data
            st.rerun()
    else:
        note("Inga metaller inlagda än.")
    if not spot:
        note("Spotpriset kunde inte hämtas just nu — värdet visas när det går.")
    with st.expander("➕ Lägg till guld eller silver", expanded=not vals):
        c1, c2, c3 = st.columns(3)
        metal = c1.selectbox("Metall", list(METALS), key="metals_metal")
        desc = c2.text_input("Beskrivning", placeholder="t.ex. 1 oz Krugerrand, 100 g tacka", key="metals_desc")
        count = c3.number_input("Antal", min_value=1, value=1, step=1, key="metals_count")
        c4, c5, c6 = st.columns(3)
        weight = c4.number_input("Vikt per styck", min_value=0.0, value=1.0, step=1.0, key="metals_weight")
        unit = c5.selectbox("Enhet", list(UNITS), index=1, key="metals_unit",
                            help="oz = troy ounce (31,1 g)")
        # Nyckel per metall: annars behöll fältet guldets 0,9999 när man valde silver
        purity = c6.number_input("Finhalt (andel)", min_value=0.0, max_value=1.0,
                                 value=PURITY_DEFAULT[metal], step=0.001, format="%.4f",
                                 key=f"metals_purity_{metal}",
                                 help="0,9999 för investeringsguld, 0,9167 för Krugerrand (vikten är då bruttovikt), "
                                      "0,999 för silvertackor, 0,925 för sterling")
        c7, c8 = st.columns(2)
        cost = c7.number_input("Köppris totalt (kr, inkl. premie)", min_value=0.0, value=0.0, step=500.0,
                               key="metals_cost")
        date = c8.date_input("Köpdatum", key="metals_date")
        if st.button("Lägg till", key="metals_add", type="primary"):
            try:
                data["rows"].append(new_row(metal, desc, int(count), float(weight), unit, float(purity), float(cost),
                                            date))
                st.session_state[STORE] = data
                st.rerun()
            except ValueError as exc:
                st.error(f"Kunde inte läggas till: {exc}")
    note("Värdet är spotvärde: finvikt × dagens terminspris (GC=F/SI=F) × USDSEK. Handlarens återköp ligger oftast "
         "några procent under spot. Årstakt visas från ett års innehav. Sparas krypterat (💾 Spara).",)
    st.markdown(f"<div style='color:{DIM};font-size:0.7rem;'>Spot: "
                + " · ".join(f"{m} {v[0]:,.1f} kr/g ({v[1]})".replace(",", " ") for m, v in spot.items())
                + "</div>", unsafe_allow_html=True)
