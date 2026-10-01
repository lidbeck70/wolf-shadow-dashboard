"""
gold_ratios/ui.py — REGIME → Råvaror → 🥇 Guldkvoter.

Samma upplägg som Guld/Silver, för fler par: översikt över alla kvoter,
sedan för valt par dagens läge, historiken, revalveringen och matrisen.
Referensen är kvotens egen historiska median och målkvoterna dess egna
percentiler. Ingen förutsägelse: sidan visar vad som händer OM kvoten
ändras. Varje uträkning står utskriven.
"""

from __future__ import annotations

import time
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from gold_ratios import config as rc
from gold_ratios import data as grd
from gold_ratios import engine as gre
from gold_silver import config as gc
from gold_silver import engine as ge
from ui.charts import PLOTLY_LAYOUT
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

_CACHE = "gr_cache"
_TTL_S = 3600


def _section(title: str, sub: str = "") -> None:
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.85rem;"
                f"margin:18px 0 6px;'>{title}" + (f" <span style='color:{DIM};letter-spacing:0;'>— {sub}</span>"
                                                  if sub else "") + "</div>", unsafe_allow_html=True)


def _load() -> dict:
    c = st.session_state.get(_CACHE)
    if c and time.time() - c.get("t", 0) < _TTL_S:
        return c["data"]
    with st.spinner("Hämtar guld och alla par …"):
        data = grd.fetch_all()
    st.session_state[_CACHE] = {"t": time.time(), "data": data}
    return data


def _fmt(v: Optional[float], dec: Optional[int] = None) -> str:
    if v is None:
        return "—"
    return f"{v:,.{gre.decimals(v) if dec is None else dec}f}"


def _names(pair: dict) -> tuple:
    """(täljarens namn, nämnarens namn) i text."""
    return ("Guld", pair["label"]) if pair["kind"] == rc.COMMODITY else (pair["label"], "Guld")


def render_gold_ratios_page() -> None:
    page_header("🥇 Guldkvoter", "Råvaror och index mätta i guld — kvoten, historiken och vilket pris varje "
                                "kvot motsvarar. En scenariomotor, ingen förutsägelse.")
    c_r, _ = st.columns([1, 4])
    if c_r.button("🔄 Uppdatera", key="gr_refresh"):
        st.session_state.pop(_CACHE, None)
    all_data = _load()
    _overview(all_data)

    keys = [p["key"] for p in rc.PAIRS]
    key = st.selectbox("Par", keys, index=keys.index(rc.DEFAULT_PAIR), key="gr_pair",
                       format_func=lambda k: f"{rc.PAIR_BY_KEY[k]['emoji']} {_title(rc.PAIR_BY_KEY[k])}")
    _render_pair(all_data[key])


def _title(pair: dict) -> str:
    num, den = _names(pair)
    return f"{num} / {den}"


# ── Översikt ────────────────────────────────────────────────────────────────
def _overview(all_data: dict) -> None:
    _section("🧭 ALLA KVOTER", "nu mot egen historik — neutral, ingen värdering")
    rows = []
    for p in rc.PAIRS:
        d = all_data.get(p["key"]) or {}
        r = gre.position_row(p, d.get("series"))
        diff = r["diff_pct"]
        color = TEXT if not diff else GREEN if diff > 0 else RED
        diff_txt = "—" if diff is None else f"{diff:+.0f} %"
        pctl = "—" if r["pctl10"] is None else f"{r['pctl10']:.0f}"
        where = r["position"] or ("historik saknas" if r["current"] is None else "kortare än 10 år")
        rows.append(
            f"<tr><td style='text-align:left;'>{p['emoji']} {_title(p)}</td><td>{_fmt(r['current'])}</td>"
            f"<td>{_fmt(r['median10'])}</td><td style='color:{color};'>{diff_txt}</td><td>{pctl}</td>"
            f"<td>{_fmt(r['median_max'])}</td><td style='text-align:left;color:{DIM};'>{where}</td></tr>")
    head = "".join(f"<th>{h}</th>" for h in ("Kvot nu", "Median 10 år", "Nu mot median", "Percentil 10 år",
                                             "Median max"))
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'>Par</th>{head}"
                f"<th style='text-align:left;'>Läge</th></tr>{''.join(rows)}</table></div>",
                unsafe_allow_html=True)
    note("Råvaror: guld ÷ råvara — hög kvot = ett uns guld köper mycket av råvaran. Index: index ÷ guld — hög "
         "kvot = indexet kostar många uns guld. Percentil = andel av dagarna med lägre eller samma kvot.")


# ── Valt par ────────────────────────────────────────────────────────────────
def _step(v) -> float:
    v = ge._pos(v)
    if v is None:
        return 1.0
    return float(10 ** (len(str(int(v))) - 3)) if v >= 100 else 0.01 if v < 10 else 0.1


def _parse_targets(raw: str) -> list:
    out = []
    for part in raw.replace(";", ",").split(","):
        try:
            v = float(part.strip().replace(" ", ""))
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    return out


def _src(point: Optional[dict], used: float, live: Optional[float]) -> str:
    if point and live is not None and abs(float(used) - float(live)) < 1e-9:
        return f"{point['source']} · {point['date']} · {point['kind']}"
    if ge._pos(used) is not None:
        return "egen inmatning · ASSUMPTION"
    return "saknas"


def _render_pair(d: dict) -> None:
    pair = d["pair"]
    k = pair["key"]
    is_c = pair["kind"] == rc.COMMODITY
    if d.get("error"):
        note(f"⚠ {d['error']}")
    series = d.get("series")
    stats = ge.all_periods(series) if series is not None else []
    full = gre.full_stats(series) if series is not None else None
    tq = gre.targets(full)
    live_g = (d.get("gold") or {}).get("value")
    live_o = (d.get("other") or {}).get("value")
    g_unit, o_unit = "USD/oz", (pair["unit"] if is_c else "punkter")

    with st.expander("✏️ Justera priser, referens och målkvoter", expanded=live_g is None or live_o is None):
        a, b, c = st.columns(3)
        gold = a.number_input(f"Guldpris ({g_unit})", min_value=0.0, value=float(live_g or 0.0),
                              step=_step(live_g), key=f"gr_gold_{k}")
        other = b.number_input(f"{pair['label']} ({o_unit})", min_value=0.0, value=float(live_o or 0.0),
                               step=_step(live_o), key=f"gr_other_{k}", format="%.4f" if (live_o or 0) < 10 else "%.2f")
        ref = c.number_input("Referenskvot (egen median)", min_value=0.0,
                             value=float(full.median if full else 0.0), step=_step(full.median if full else None),
                             key=f"gr_ref_{k}")
        raw = st.text_input("Målkvoter (kommaseparerade)", ", ".join(f"{v:g}" for _l, v in tq), key=f"gr_targets_{k}")
        note("Förvalda målkvoter = kvotens egna percentiler P10, P25, median, P75 och P90 över hela historiken. "
             "Tomt eller 0 räknas som saknat — ingen uträkning görs på ogiltiga tal.")
    targets = _parse_targets(raw) or [v for _l, v in tq]
    num, den = gre.orient(pair, gold, other)
    cur = ge.ratio(num, den)
    long = next((s for s in stats if s.years == rc.POSITION_PERIOD_YEARS), stats[-1] if stats else None)
    gap = ge.geological_gap(num, den, ref)

    # ── Överst ───────────────────────────────────────────────────────────────
    r1 = st.columns(3)
    r1[0].markdown(big_card("🥇 GULD", f"${_fmt(ge._pos(gold), 0)}", _src(d.get("gold"), gold, live_g), GOLD),
                   unsafe_allow_html=True)
    o_big = f"${_fmt(ge._pos(other))}" if is_c else _fmt(ge._pos(other), 0)
    r1[1].markdown(big_card(f"{pair['emoji']} {pair['label'].upper()}", o_big,
                            f"{o_unit} · {_src(d.get('other'), other, live_o)}", CYAN), unsafe_allow_html=True)
    sub = ("priser saknas" if cur is None else
           f"1 uns guld = {_fmt(cur)} {gre.unit_word(pair)} {pair['label'].lower()}" if is_c else
           f"{pair['label']} = {_fmt(cur)} uns guld")
    r1[2].markdown(big_card(_title(pair).upper(), _fmt(cur), sub, CYAN), unsafe_allow_html=True)
    r2 = st.columns(3)
    r2[0].markdown(big_card("📏 REFERENS", _fmt(ge._pos(ref)),
                            "kvotens egen median över hela historiken" + (f" ({full.start[:4]}–{full.end[:4]})"
                                                                         if full else "") + " · inte ett fair value",
                            GREY), unsafe_allow_html=True)
    r2[1].markdown(big_card("AVSTÅND TILL REFERENSEN", "—" if gap is None else f"{gap.pct_of_current:.0f} %",
                            "—" if gap is None else f"({_fmt(gap.current)} − {_fmt(gap.reference)}) / "
                                                    f"{_fmt(gap.current)}", AMBER), unsafe_allow_html=True)
    r2[2].markdown(big_card("HISTORISK MEDIAN", "—" if long is None else _fmt(long.median),
                            "—" if long is None else f"{long.label} ({long.start}–{long.end})", CYAN),
                   unsafe_allow_html=True)

    _history(pair, series, stats, cur, ref, full, d)
    _position(pair, cur, long)
    _revaluation(pair, num, den, targets, ref)
    _matrix(pair, num, targets, ref)
    if is_c:
        try:
            from gold_ratios.miners import render_miner_section
            render_miner_section(pair, gold, other, full)
        except ImportError:
            pass
    _section("⚠️ VAD KVOTEN BEROR PÅ")
    note(f"{gre.meaning(pair)}. Kvoten speglar två marknader samtidigt — guldets och "
         f"{'råvarans' if is_c else 'aktiemarknadens'} — med egna drivkrafter (utbud, efterfrågan, räntor, "
         f"dollarn, cykeln). Referensen är bara kvotens egen median: sidan visar vad som händer om kvoten "
         f"ändras, inte att den kommer att göra det.")


def ratio_chart(pair: dict, series, cur: Optional[float], ref: float, full) -> Optional[go.Figure]:
    if series is None or len(series) < 2:
        return None
    w = series.resample("W").last().dropna()
    fig = go.Figure(go.Scatter(x=list(w.index), y=[float(v) for v in w.values], name=_title(pair),
                               line=dict(color=CYAN, width=1.6)))
    lines = []
    if ge._pos(ref):
        lines.append((ref, f"referens {_fmt(ref)}", GOLD, "dash"))
    if full is not None:
        lines += [(full.p10, f"P10 {_fmt(full.p10)}", GREY, "dot"), (full.p90, f"P90 {_fmt(full.p90)}", GREY, "dot")]
    if cur:
        lines.append((cur, f"nu {_fmt(cur)}", TEXT, "solid"))
    for y, label, color, dash in lines:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash=dash), annotation_text=label,
                      annotation_font=dict(color=color, size=10), annotation_position="top left")
    layout = {k: v for k, v in PLOTLY_LAYOUT.items()}
    layout.update(title=dict(text=f"{_title(pair).upper()} — VECKOVIS", font=dict(size=12, color=CYAN)), height=360,
                  showlegend=False, yaxis=dict(title="Kvot", gridcolor="rgba(255,255,255,0.05)"),
                  xaxis=dict(title="Datum", gridcolor="rgba(255,255,255,0.05)"))
    fig.update_layout(**layout)
    return fig


def _history(pair: dict, series, stats: list, cur, ref: float, full, d: dict) -> None:
    _section("📈 HISTORISK KVOT")
    if not stats:
        note(f"Historik saknas — ingen serie för {rc.GOLD_TICKER}/{pair['ticker']}. Revalveringen nedan fungerar "
             f"ändå med dagens (eller egna) priser.")
        return
    fig = ratio_chart(pair, series, cur, ref, full)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                        key=f"gr_chart_{pair['key']}")
    head = "".join(f"<th>{h}</th>" for h in ("Period", "Nu", "Medel", "Median", "P10", "P25", "P75", "P90",
                                             "Min", "Max", "Percentil"))
    rows = "".join(
        f"<tr><td style='text-align:left;'>{s.label}<br><span style='color:{DIM};font-size:0.68rem;'>"
        f"{s.start[:4]}–{s.end[:4]}</span></td><td>{_fmt(s.current)}</td><td>{_fmt(s.mean)}</td>"
        f"<td>{_fmt(s.median)}</td><td>{_fmt(s.p10)}</td><td>{_fmt(s.p25)}</td><td>{_fmt(s.p75)}</td>"
        f"<td>{_fmt(s.p90)}</td><td>{_fmt(s.min)}</td><td>{_fmt(s.max)}</td><td>{s.percentile:.0f}</td></tr>"
        for s in stats)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'>{head}</tr>{rows}</table></div>",
                unsafe_allow_html=True)
    src = [x["source"] for x in (d.get("gold"), d.get("other")) if x]
    scale = (" Yahoo noterar priset i US-cent — omräknat till USD." if pair.get("scale", 1.0) != 1.0 else "")
    note(f"Dagliga stängningar från {stats[-1].start} ({' · '.join(src) or 'källa saknas'}).{scale} Längre "
         f"perioder än historiken visas inte. Percentil = andel av dagarna i perioden med lägre eller samma "
         f"kvot som i dag.")


def _position(pair: dict, cur, long) -> None:
    _section("📊 HISTORISK POSITION", "neutral — ingen värdering")
    if cur is None or long is None:
        note("Kräver dagens kvot och historik.")
        return
    diff = (cur / long.median - 1) * 100
    st.markdown(
        f"<div style='font-size:0.85rem;color:{TEXT};line-height:1.7;'>Nu <b>{_fmt(cur)}</b> · {long.label}-median "
        f"<b>{_fmt(long.median)}</b> · nu mot median <b>{diff:+.0f} %</b> · percentil <b>{long.percentile:.0f}</b><br>"
        f"<b style='color:{CYAN};'>{ge.position(cur, long.median)}</b> (inom ±{gc.NEAR_MEDIAN_PCT:g} % = nära)<br>"
        f"<span style='color:{DIM};'>{gre.meaning(pair)}.</span></div>", unsafe_allow_html=True)


def _revaluation(pair: dict, num, den, targets: list, ref: float) -> None:
    is_c = pair["kind"] == rc.COMMODITY
    solved = pair["label"] if is_c else "Guld"
    s_unit = pair["unit"] if is_c else "USD/oz"
    _section(f"🚀 {solved.upper()} PER KVOT", f"vilket {solved.lower()}pris motsvarar varje kvot?")
    rows = ge.revaluation_table(num, den, targets, ref)
    if not rows:
        note(f"Kräver {'ett guldpris' if is_c else 'en indexnivå'} > 0.")
        return
    body = "".join(
        f"<tr style='{'font-weight:700;color:' + CYAN + ';' if r.is_current else ''}'>"
        f"<td style='text-align:left;'>{_fmt(r.ratio)}{' (nu)' if r.is_current else ''}"
        f"{' · REFERENS' if r.is_reference else ''}</td>"
        f"<td>{_fmt(r.silver)}</td><td>{'—' if r.change is None else f'{r.change:+,.2f}'}</td>"
        f"<td style='color:{GREEN if (r.change_pct or 0) > 0 else RED if (r.change_pct or 0) < 0 else TEXT};'>"
        f"{'—' if r.change_pct is None else f'{r.change_pct:+.0f} %'}</td>"
        f"<td style='color:{DIM};'>{r.formula}</td></tr>" for r in rows)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.78rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'>Kvot</th>"
                f"<th>{solved} ({s_unit})</th><th>Förändring</th><th>Upp-/nedsida</th><th>Uträkning</th></tr>"
                f"{body}</table></div>", unsafe_allow_html=True)
    if is_c:
        note(f"{pair['label']} = guld / kvot, vid guldpriset {_fmt(ge._pos(num), 0)} USD. Referensen är kvotens "
             f"egen median — inte ett fair value.")
    else:
        note(f"Guld = {pair['label']} / kvot, vid {pair['label']} {_fmt(ge._pos(num), 0)}. Indexet hålls fast; "
             f"samma kvot kan också nås med ett lägre index. Referensen är kvotens egen median — inte ett fair value.")


def _matrix(pair: dict, num, targets: list, ref: float) -> None:
    is_c = pair["kind"] == rc.COMMODITY
    num_name, den_name = _names(pair)
    _section(f"💥 {num_name.upper()} × KVOT", f"{den_name.lower()}priset för varje par")
    grid = gre.anchor_grid(num)
    rows = ge.matrix(grid, sorted(set(targets), reverse=True)) if targets else []
    if not rows:
        note(f"Kräver {'ett guldpris' if is_c else 'en indexnivå'} > 0 och minst en målkvot.")
        return
    ratios = list(rows[0][1])
    head = "".join(f"<th>{_fmt(r)}{' ⓡ' if abs(r - ref) < 1e-9 else ''}</th>" for r in ratios)
    pre = "$" if is_c else ""
    body = "".join(f"<tr><td style='text-align:left;color:{GOLD};'>{pre}{g:,.0f}</td>"
                   + "".join(f"<td>{_fmt(vals[r])}</td>" for r in ratios) + "</tr>" for g, vals in rows)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.78rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'>{num_name} \\ kvot</th>"
                f"{head}</tr>{body}</table></div>", unsafe_allow_html=True)
    g0, r0 = rows[0][0], ratios[len(ratios) // 2]
    note(f"Raderna = dagens {num_name.lower()} × {', '.join(f'{m:g}' for m in rc.ANCHOR_GRID)} (avrundat). "
         f"Exempel: {num_name.lower()} {g0:,.0f} och kvot {_fmt(r0)} → {den_name.lower()} = {ge.formula(g0, r0)}. "
         f"ⓡ = referenskvoten.")
