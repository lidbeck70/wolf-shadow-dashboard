"""
gold_silver/ui.py — REGIME → Råvaror → 🥇🥈 Guld/Silver.

Överst dagens läge, sedan historiken, revalveringsmotorn, matrisen
guldpris × kvot och de tre referenskvoterna (geologisk, produktion,
ovanjord) — var och en med källa. Ingen förutsägelse: sidan visar vad som
händer OM kvoten ändras. Varje uträkning står utskriven.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from gold_silver import config as gc
from gold_silver import data as gd
from gold_silver import engine as ge
from ui.charts import PLOTLY_LAYOUT
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

SILVER_C = "#c0c6cc"
_CACHE = "gs_cache"
_TTL_S = 3600


def _section(title: str, sub: str = "") -> None:
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.85rem;"
                f"margin:18px 0 6px;'>{title}" + (f" <span style='color:{DIM};letter-spacing:0;'>— {sub}</span>"
                                                  if sub else "") + "</div>", unsafe_allow_html=True)


def _load() -> dict:
    import time
    c = st.session_state.get(_CACHE)
    if c and time.time() - c.get("t", 0) < _TTL_S:
        return c["data"]
    data = gd.fetch()
    st.session_state[_CACHE] = {"t": time.time(), "data": data}
    return data


def _fmt(v: Optional[float], dec: int = 2) -> str:
    return "—" if v is None else f"{v:,.{dec}f}"


def render_gold_silver_page() -> None:
    page_header("🥇🥈 Guld/Silver", "Kvoten, historiken och vilket silverpris varje kvot motsvarar — en "
                                   "scenariomotor, ingen förutsägelse.")
    c_r, _ = st.columns([1, 4])
    if c_r.button("🔄 Uppdatera", key="gs_refresh"):
        st.session_state.pop(_CACHE, None)
    data = _load()
    if data.get("error"):
        note(f"⚠ {data['error']}")

    # ── Egna värden (spec §13) ────────────────────────────────────────────────
    live_g = (data.get("gold") or {}).get("value")
    live_s = (data.get("silver") or {}).get("value")
    with st.expander("✏️ Justera guld, silver, referens och målkvoter", expanded=live_g is None or live_s is None):
        a, b, c = st.columns(3)
        gold = a.number_input("Guldpris (USD/oz)", min_value=0.0, value=float(live_g or 0.0), step=50.0,
                              key="gs_gold")
        silver = b.number_input("Silverpris (USD/oz)", min_value=0.0, value=float(live_s or 0.0), step=0.5,
                                key="gs_silver")
        ref = c.number_input("Referenskvot", min_value=0.0, value=gc.REFERENCE_RATIO, step=1.0, key="gs_ref")
        raw = st.text_input("Målkvoter (kommaseparerade)", ", ".join(f"{r:g}" for r in gc.TARGET_RATIOS),
                            key="gs_targets")
        targets = []
        for part in raw.replace(";", ",").split(","):
            try:
                v = float(part.strip().replace(" ", ""))
            except ValueError:
                continue
            if v > 0:
                targets.append(v)
        note("Tomt eller 0 räknas som saknat — ingen uträkning görs på ogiltiga tal.")
    targets = targets or list(gc.TARGET_RATIOS)
    cur = ge.ratio(gold, silver)
    series = data.get("series")
    stats = ge.all_periods(series) if series is not None else []
    long = next((s for s in stats if s.years == gc.POSITION_PERIOD_YEARS), stats[-1] if stats else None)

    # ── Överst (spec §4, §22) ─────────────────────────────────────────────────
    gap = ge.geological_gap(gold, silver, ref)
    r1 = st.columns(3)
    r1[0].markdown(big_card("🥇 GULD", f"${_fmt(ge._pos(gold), 0)}", _src(data.get("gold"), gold, live_g), GOLD),
                   unsafe_allow_html=True)
    r1[1].markdown(big_card("🥈 SILVER", f"${_fmt(ge._pos(silver))}", _src(data.get("silver"), silver, live_s),
                            SILVER_C), unsafe_allow_html=True)
    r1[2].markdown(big_card("GULD / SILVER", _fmt(cur, 1),
                            f"1 uns guld = {_fmt(cur, 1)} uns silver" if cur else "priser saknas", CYAN),
                   unsafe_allow_html=True)
    r2 = st.columns(3)
    r2[0].markdown(big_card("🌍 GEOLOGISK REFERENS", f"~{ref:g}",
                            "Approximate geological abundance reference · not a market equilibrium value",
                            GREY), unsafe_allow_html=True)
    r2[1].markdown(big_card("AVSTÅND TILL REFERENSEN", "—" if gap is None else f"{gap.pct_of_current:.0f} %",
                            "—" if gap is None else f"({gap.current:g} − {gap.reference:g}) / {gap.current:g} · "
                                                    f"skillnad {gap.difference:g}", AMBER), unsafe_allow_html=True)
    r2[2].markdown(big_card("HISTORISK MEDIAN", "—" if long is None else f"{long.median:g}",
                            "—" if long is None else f"{long.label} ({long.start}–{long.end})", CYAN),
                   unsafe_allow_html=True)

    _history(series, stats, cur, ref, long)
    _position(cur, long)
    _gap(gap)
    _references(cur, ref)
    _revaluation(gold, silver, targets, ref)
    _matrix(ref)
    try:
        from gold_silver.miners import render_miner_section
        render_miner_section(gold, silver, cur)
    except ImportError:
        pass
    _section("⚠️ VAD KVOTEN BEROR PÅ")
    note("19 är en referens för hur mycket silver det ungefär finns per guld i jordskorpan — inte rätt "
         "värdering. Marknadskvoten beror på: " + ", ".join(gc.MARKET_DRIVERS) + ". Sidan visar vad som "
         "händer om marknadskvoten ändras, inte att den kommer att göra det.")


def _src(point: Optional[dict], used: float, live: Optional[float]) -> str:
    if point and live is not None and abs(float(used) - float(live)) < 1e-9:
        return f"{point['source']} · {point['date']} · {point['kind']}"
    if ge._pos(used) is not None:
        return "egen inmatning · ASSUMPTION"
    return "saknas"


# ── 📈 Historik (spec §5, §6, §18) ────────────────────────────────────────────
def ratio_chart(series, cur: Optional[float], ref: float, median: Optional[float]) -> Optional[go.Figure]:
    if series is None or len(series) < 2:
        return None
    w = series.resample("W").last().dropna()
    fig = go.Figure(go.Scatter(x=list(w.index), y=[float(v) for v in w.values], name="Guld/silver",
                               line=dict(color=CYAN, width=1.6)))
    lines = [(ref, f"geologisk referens {ref:g}", GREY, "dash"),
             (gc.GUIDE_ACCUMULATE, f"guidens ackumuleringszon {gc.GUIDE_ACCUMULATE:g}", GREEN, "dot"),
             (gc.GUIDE_LATE, f"guidens sena zon {gc.GUIDE_LATE:g}", RED, "dot")]
    if median:
        lines.append((median, f"median {median:g}", GOLD, "dot"))
    if cur:
        lines.append((cur, f"nu {cur:.1f}", TEXT, "solid"))
    for y, label, color, dash in lines:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash=dash), annotation_text=label,
                      annotation_font=dict(color=color, size=10), annotation_position="top left")
    layout = {k: v for k, v in PLOTLY_LAYOUT.items()}
    layout.update(title=dict(text="GULD / SILVER — VECKOVIS", font=dict(size=12, color=CYAN)), height=360,
                  showlegend=False, yaxis=dict(title="Kvot", gridcolor="rgba(255,255,255,0.05)"),
                  xaxis=dict(title="Datum", gridcolor="rgba(255,255,255,0.05)"))
    fig.update_layout(**layout)
    return fig


def _history(series, stats: list, cur, ref: float, long) -> None:
    _section("📈 HISTORISK KVOT")
    if not stats:
        note("Historik saknas — Yahoo levererade ingen serie för GC=F/SI=F. Revalveringsmotorn nedan fungerar "
             "ändå med dagens (eller egna) priser.")
        return
    fig = ratio_chart(series, cur, ref, long.median if long else None)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="gs_ratio_chart")
    head = "".join(f"<th>{h}</th>" for h in ("Period", "Nu", "Medel", "Median", "P10", "P25", "P75", "P90",
                                             "Min", "Max", "Percentil"))
    rows = "".join(
        f"<tr><td style='text-align:left;'>{s.label}<br><span style='color:{DIM};font-size:0.68rem;'>"
        f"{s.start[:4]}–{s.end[:4]}</span></td><td>{s.current:g}</td><td>{s.mean:g}</td><td>{s.median:g}</td>"
        f"<td>{s.p10:g}</td><td>{s.p25:g}</td><td>{s.p75:g}</td><td>{s.p90:g}</td><td>{s.min:g}</td>"
        f"<td>{s.max:g}</td><td>{s.percentile:.0f}</td></tr>" for s in stats)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'>{head}</tr>{rows}</table></div>",
                unsafe_allow_html=True)
    first = stats[-1]
    note(f"Terminerna GC=F/SI=F från Yahoo, dagliga stängningar från {first.start}. Längre perioder än "
         f"historiken visas inte — 50 år kräver en annan källa. Percentil = andel av dagarna i perioden med "
         f"lägre eller samma kvot som i dag.")


def _position(cur, long) -> None:
    _section("📊 HISTORISK POSITION", "neutral — ingen värdering")
    if cur is None or long is None:
        note("Kräver dagens kvot och historik.")
        return
    diff = (cur / long.median - 1) * 100
    zone = ge.guide_zone(cur)
    st.markdown(
        f"<div style='font-size:0.85rem;color:{TEXT};line-height:1.7;'>Nu <b>{cur:.1f}</b> · {long.label}-median "
        f"<b>{long.median:g}</b> · nu mot median <b>{diff:+.0f} %</b> · percentil <b>{long.percentile:.0f}</b><br>"
        f"<b style='color:{CYAN};'>{ge.position(cur, long.median)}</b> (inom ±{gc.NEAR_MEDIAN_PCT:g} % = nära)<br>"
        f"<span style='color:{AMBER};'>{zone}</span></div>", unsafe_allow_html=True)
    note("Läget säger var kvoten står mot sin egen historik — inte att den är billig eller dyr. Guidens zoner "
         "kommer från masterguiden och Råvarubiblioteket.")


def _gap(gap) -> None:
    _section("🌍 GEOLOGISKT GAP")
    if gap is None:
        note("Kräver guld, silver och en referenskvot > 0.")
        return
    st.markdown(
        f"<div style='font-size:0.85rem;color:{TEXT};line-height:1.7;'>Nu <b>{gap.current:g}</b> · referens "
        f"<b>{gap.reference:g}</b> · skillnad <b>{gap.difference:g}</b> · <b>{gap.pct_of_current:.0f} %</b> av dagens "
        f"kvot<br>Om marknadskvoten skulle röra sig mot {gap.reference:g}, vid dagens guldpris, motsvarar det "
        f"ett silverpris på <b>${gap.implied_silver:,.2f}</b> ({gap.upside_pct:+.0f} %).</div>",
        unsafe_allow_html=True)


def _references(cur, ref: float) -> None:
    _section("⚖️ FYRA OLIKA KVOTER", "blandas aldrig ihop")
    geo, prod, above = (gc.REFERENCES[k] for k in ("geological", "production", "above_ground"))
    pr, ar = ge.production_ratio(prod), ge.above_ground_ratio(above)
    cards = [
        ("📈 MARKNADSKVOT", _fmt(cur, 1), "guldpris / silverpris — vad marknaden betalar i dag", CYAN),
        ("🌍 GEOLOGISK", f"~{ref:g}", f"{geo['note']} Källa: {geo['source']} · {geo['kind']}", GREY),
        ("⛏️ GRUVPRODUKTION", "DATA UNAVAILABLE" if pr is None else f"{pr:.1f} : 1",
         f"silver {prod['silver_t']:,.0f} t / guld {prod['gold_t']:,.0f} t. {prod['note']} Källa: {prod['source']} · "
         f"{prod['kind']}", AMBER),
        ("🏦 OVAN JORD", "Partial data" if ar is None else f"{ar:.1f} : 1",
         f"guld ~{above['gold_t']:,.0f} t · silver okänt. {above['note']} Källa: {above['source']}", GREY),
    ]
    cols = st.columns(2)
    for i, (t, big, sub, color) in enumerate(cards):
        cols[i % 2].markdown(big_card(t, big, sub, color), unsafe_allow_html=True)
    note("Referensvärdena är uppskattningar med källa och datum (gold_silver/config.py) — uppdatera när en ny "
         "rapport kommer. Ingen av dem är ett fair value.")


def _revaluation(gold, silver, targets: list, ref: float) -> None:
    _section("🚀 SILVERREVALVERING", "vilket silverpris motsvarar varje kvot?")
    rows = ge.revaluation_table(gold, silver, targets, ref)
    if not rows:
        note("Kräver ett guldpris > 0.")
        return
    body = "".join(
        f"<tr style='{'font-weight:700;color:' + CYAN + ';' if r.is_current else ''}'>"
        f"<td style='text-align:left;'>{r.ratio:g}{' (nu)' if r.is_current else ''}"
        f"{' · REFERENCE SCENARIO' if r.is_reference else ''}</td>"
        f"<td>{_fmt(r.silver)}</td><td>{'—' if r.change is None else f'{r.change:+,.2f}'}</td>"
        f"<td style='color:{GREEN if (r.change_pct or 0) > 0 else RED if (r.change_pct or 0) < 0 else TEXT};'>"
        f"{'—' if r.change_pct is None else f'{r.change_pct:+.0f} %'}</td>"
        f"<td style='color:{DIM};'>{r.formula}</td></tr>" for r in rows)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.78rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'>Kvot</th>"
                f"<th>Silver (USD)</th><th>Förändring</th><th>Upp-/nedsida</th><th>Uträkning</th></tr>{body}"
                f"</table></div>", unsafe_allow_html=True)
    note(f"Silver = guld / kvot, vid guldpriset {_fmt(ge._pos(gold), 0)} USD. Referenskvoten är ett "
         f"REFERENCE SCENARIO, inte ett fair value.")


def _matrix(ref: float) -> None:
    _section("💥 GULDPRIS × KVOT", "silverpriset för varje par")
    rows = ge.matrix()
    if not rows:
        return
    ratios = list(rows[0][1])
    head = "".join(f"<th>{r:g}{' ⓡ' if r == ref else ''}</th>" for r in ratios)
    body = "".join(f"<tr><td style='text-align:left;color:{GOLD};'>${g:,.0f}</td>"
                   + "".join(f"<td>{vals[r]:,.0f}</td>" for r in ratios) + "</tr>" for g, vals in rows)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.78rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'>Guld \\ kvot</th>"
                f"{head}</tr>{body}</table></div>", unsafe_allow_html=True)
    g0, r0 = rows[1][0] if len(rows) > 1 else rows[0][0], 40.0
    note(f"Exempel: guld ${g0:,.0f} och kvot {r0:g} → silver = {ge.formula(g0, r0)} USD. ⓡ = referenskvoten.")
