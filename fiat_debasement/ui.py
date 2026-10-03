"""
fiat_debasement/ui.py — REGIME → Makro → 🐺 Fiat Debasement.

Sektioner:
  1. FIAT OVERVIEW        en rad per valuta: penningmängd, KPI, real BNP,
                          Monetary Gap, statsskuld och köpkraft i guld
  2. MONEY SUPPLY         M2-tillväxt och Monetary Gap för vald valuta
  3. INFLATION            KPI, kärn-KPI och PURCHASING POWER INDEX
  4. WHAT HAPPENED TO 100 UNITS?  kontanter mot KPI, guld och silver
  5. FIAT VS GOLD         SEK/EUR/USD mätta i guld (start = 100), 1/5/10 år,
                          guldets och valutans köpkraft som två serier
  6. FIAT VS SILVER       samma för silver
  7. REAL ASSET PROTECTION  guld, silver, koppar, olja, bitcoin mot KPI i vald valuta
  8. GOLD/SILVER RATIO    kvoten mot historiken och mot valutans köpkraft
  9. METHODOLOGY          vad varje mått mäter, antaganden och datakällorna

Färgerna är bara visuell hjälp — siffran står alltid bredvid. Saknad data
visas som DATA UNAVAILABLE, aldrig 0. Sidan säger aldrig köp eller sälj.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fiat_debasement import config as cfg
from fiat_debasement import data as fd
from fiat_debasement import engine as fe
from fiat_debasement import snapshot as fs
from ui.charts import PLOTLY_LAYOUT
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, PURPLE, RED, TEXT

NA = "DATA UNAVAILABLE"
CCY_COLOR = {"SEK": CYAN, "EUR": PURPLE, "USD": GREEN}
START_CHOICES = ("1970", "1980", "1990", "2000", "2010", "2015", "2020", "Eget datum")
MAX_START_SHIFT_DAYS = 366          # en serie som börjar senare än så räknas som saknad för startåret

# Visuell hjälp: (gräns grön, gräns gul, högre = sämre?) — dokumenterat i metodpanelen
SCALES = {
    "m2_yoy": (3.0, 7.0, True), "cpi_yoy": (2.5, 5.0, True), "monetary_gap": (2.0, 4.0, True),
    "debt_gdp": (60.0, 100.0, True), "gdp_yoy": (2.0, 0.0, False), "gold_5y": (0.0, -30.0, False),
}

OVERVIEW_COLS = (("m2_yoy", "M2-tillväxt", "%"), ("cpi_yoy", "KPI", "%"), ("gdp_yoy", "Real BNP", "%"),
                 ("monetary_gap", "Monetary Gap", "pe"), ("debt_gdp", "Statsskuld/BNP", "%"),
                 ("gold_5y", "Köpkraft i guld 5 år", "%"))


# ── Rena hjälpare (testbara) ────────────────────────────────────────────────
def color_for(key: str, value: Optional[float]) -> str:
    if value is None or key not in SCALES:
        return GREY
    a, b, higher_worse = SCALES[key]
    if higher_worse:
        return GREEN if value <= a else AMBER if value <= b else RED
    return GREEN if value >= a else AMBER if value >= b else RED


def fmt(value: Optional[float], unit: str = "%", sign: bool = True) -> str:
    if value is None:
        return NA
    num = f"{value:+.1f}" if sign else f"{value:.1f}"
    return f"{num} {unit}".strip()


def overview_rows(snaps: dict) -> list:
    """[{valuta, kolumner: [(nyckel, värde, text, färg, källa, datum, inaktuell)]}]."""
    rows = []
    for ccy in cfg.CURRENCIES:
        snap = snaps.get(ccy)
        cells = []
        for key, _label, unit in OVERVIEW_COLS:
            m = snap.get(key) if snap is not None else fs.Metric()
            cells.append({"key": key, "value": m.value, "text": fmt(m.value, unit, sign=key != "debt_gdp"),
                          "color": color_for(key, m.value), "source": m.source, "as_of": m.as_of,
                          "stale": m.stale})
        rows.append({"currency": ccy, "cells": cells})
    return rows


def principle_sentence(snap: Optional[fs.Snapshot]) -> str:
    """Specens formulering: beskriver vad indikatorerna visar — aldrig vad man ska göra."""
    if snap is None:
        return ""
    gap = snap.get("monetary_gap")
    if gap.value is None:
        return f"{snap.currency}: Monetary Gap kan inte beräknas ({NA})."
    side = "över" if gap.value > 0 else "under"
    return (f"{snap.currency}: Enligt valda indikatorer växer penningmängden {side} den reala produktionen "
            f"({gap.value:+.1f} procentenheter, {gap.as_of}).")


def start_date(choice: str, custom=None) -> pd.Timestamp:
    if choice == "Eget datum" and custom is not None:
        return pd.Timestamp(custom)
    return pd.Timestamp(f"{choice}-01-01")


def usable_from(s: Optional[pd.Series], start: pd.Timestamp) -> tuple:
    """(True, första datum) om serien har data inom MAX_START_SHIFT_DAYS efter start."""
    if s is None or not len(s):
        return False, None
    later = s[s.index >= start]
    if not len(later):
        return False, None
    first = later.index[0]
    return (first - start).days <= MAX_START_SHIFT_DAYS, first


def cpi_for_power(ccy: str, loader=None) -> fd.Loaded:
    """Lång KPI-serie där den finns (SEK sedan 1914, USD sedan 1913), annars ordinarie KPI."""
    load = loader or fd.load
    long = load(cfg.CPI_LONG, ccy)
    return long if long.ok else load(cfg.CPI, ccy)


def power_lines(start: pd.Timestamp, loader=None) -> dict:
    """valuta → (köpkraftsserie | None, förklaring)."""
    out = {}
    for ccy in cfg.CURRENCIES:
        ld = cpi_for_power(ccy, loader)
        ok, first = usable_from(ld.values, start)
        if not ok:
            since = ld.data.first if ld.ok else None
            out[ccy] = (None, f"{ccy}: {NA} för {start.date()}" + (f" — serien börjar {since}" if since else ""))
            continue
        label = ld.data.label or ld.data.source
        out[ccy] = (fe.purchasing_power(ld.values, start), f"{ccy}: {label} [{ld.data.source} {ld.data.series_id}]"
                    + (f", från {first.date()}" if first > start else ""))
    return out


def hundred_units(ccy: str, start: pd.Timestamp, loader=None, asset_loader=None) -> dict:
    """'What happened to 100 units?' → {start, lines: {namn: serie}, notes: [...], segments: [...]}."""
    load = loader or fd.load
    load_asset = asset_loader or fd.load_asset
    notes, lines, segments = [], {}, []
    fx = load(cfg.FX, ccy) if ccy != "USD" else None
    assets = {}
    for name in (cfg.GOLD, cfg.SILVER):
        ld = load_asset(name)
        price = fe.price_in(ld.values, ccy, fx.values if fx is not None else None)
        ok, first = usable_from(price, start)
        if ok:
            assets[name] = (price, first, ld)
        else:
            notes.append(f"{cfg.CONCEPT_LABEL[name]}: {NA} för {start.date()}"
                         + (f" (pris i {ccy} finns från {price.index[0].date()})" if price is not None else ""))
    eff = max([start] + [first for _p, first, _ld in assets.values()])
    if eff > start:
        notes.append(f"Startdatum justerat till {eff.date()} — första gemensamma noteringen för guld/silver i {ccy}.")
    cpi = cpi_for_power(ccy, load)
    pp = fe.purchasing_power(cpi.values, eff) if usable_from(cpi.values, eff)[0] else None
    if pp is not None:
        lines["Kontanter, köpkraft (KPI)"] = pp
    else:
        notes.append(f"KPI för {ccy}: {NA} för {eff.date()}")
    for name, (price, _first, ld) in assets.items():
        lines[f"I {cfg.CONCEPT_LABEL[name].lower()}"] = fe.units_of_100(price, eff)
        segments += [dict(s, asset=name) for s in ld.segments if s.get("kind") == cfg.FUTURES]
        if ld.note:
            notes.append(f"{cfg.CONCEPT_LABEL[name]}: {ld.note}")
    return {"start": eff, "lines": lines, "notes": notes, "segments": segments}


WINDOWS = (("1 år", 1), ("5 år", 5), ("10 år", 10))
REAL_ASSETS = (cfg.GOLD, cfg.SILVER, cfg.COPPER, cfg.OIL, cfg.BTC)
ASSET_COLOR = {cfg.GOLD: GOLD, cfg.SILVER: "#c0c0c0", cfg.COPPER: "#b87333", cfg.OIL: "#4a90d9", cfg.BTC: "#f7931a"}
GS_PERIODS = (("1 år", 1), ("5 år", 5), ("10 år", 10), ("20 år", 20), ("Hela historiken", None))


def asset_price(name: str, ccy: str, loader=None, asset_loader=None) -> tuple:
    """(tillgångens pris i valutan | None, Loaded för tillgången)."""
    load, load_asset = loader or fd.load, asset_loader or fd.load_asset
    ld = load_asset(name)
    fx = load(cfg.FX, ccy) if ccy != "USD" else None
    return fe.price_in(ld.values, ccy, fx.values if fx is not None else None), ld


def fiat_vs_lines(name: str, start: pd.Timestamp, loader=None, asset_loader=None) -> dict:
    """valuta → (valutans värde i tillgången, start = 100 | None, förklaring)."""
    out = {}
    for ccy in cfg.CURRENCIES:
        price, ld = asset_price(name, ccy, loader, asset_loader)
        ok, first = usable_from(price, start)
        if not ok:
            out[ccy] = (None, f"{ccy}/{cfg.CONCEPT_LABEL[name]}: {NA} för {start.date()}")
            continue
        out[ccy] = (fe.fiat_vs_asset(price, start), f"{ccy}/{cfg.CONCEPT_LABEL[name]} från {first.date()}")
    return out


def window_rows(lines: dict) -> list:
    """Förändring (%) i valutans värde mätt i tillgången: 1/5/10 år och sedan start."""
    rows = []
    for ccy, (s, _txt) in lines.items():
        row = {"Valuta": ccy}
        for label, yrs in WINDOWS:
            row[label] = fe.change_pct(s, yrs)
        row["Sedan start"] = None if s is None else float(s.iloc[-1]) - 100
        rows.append(row)
    return rows


def power_pair(ccy: str, start: pd.Timestamp, name: str = cfg.GOLD, loader=None, asset_loader=None) -> dict:
    """Två separata serier, start = 100: valutans köpkraft (KPI) och metallens köpkraft
    (metallpriset i valutan delat med KPI — hur mycket varor en uns köper)."""
    out = {}
    cpi = cpi_for_power(ccy, loader)
    price, _ld = asset_price(name, ccy, loader, asset_loader)
    if cpi.ok and usable_from(price, start)[0]:
        eff = max(start, price[price.index >= start].index[0])
        pp = fe.purchasing_power(cpi.values, eff)
        if pp is not None:
            out[f"{ccy}: köpkraft (KPI)"] = pp
        real = fe.normalize_100(fe.real_price(price, cpi.values, eff), eff)
        if real is not None:
            out[f"{cfg.CONCEPT_LABEL[name]}: köpkraft i {ccy}"] = real
    return out


def real_asset_lines(ccy: str, start: pd.Timestamp, loader=None, asset_loader=None) -> tuple:
    """({namn: pris i valutan, start = 100}, [förklaringar]) — plus KPI som prisnivå."""
    lines, notes = {}, []
    for name in REAL_ASSETS:
        price, ld = asset_price(name, ccy, loader, asset_loader)
        ok, first = usable_from(price, start)
        if not ok:
            since = f" — finns från {price.index[0].date()}" if price is not None and len(price) else ""
            notes.append(f"{cfg.CONCEPT_LABEL[name]}: {NA} för {start.date()}{since}")
            continue
        lines[cfg.CONCEPT_LABEL[name]] = fe.normalize_100(price, start)
        if first > start + pd.Timedelta(days=31):
            notes.append(f"{cfg.CONCEPT_LABEL[name]}: börjar {first.date()}")
    cpi = cpi_for_power(ccy, loader)
    if cpi.ok and usable_from(cpi.values, start)[0]:
        lines[f"KPI {ccy} (prisnivå)"] = fe.normalize_100(cpi.values, start)
    notes.append(f"Fastigheter: {NA} — ingen jämförbar daglig källa för SEK, EUR och USD är vald än.")
    return lines, notes


def gs_ratio(asset_loader=None) -> tuple:
    """(kvotserie, [{period, nu, snitt, median, min, max, percentil}]) — samma uträkning som 🥇🥈 Guld/Silver."""
    from gold_silver import engine as ge
    load_asset = asset_loader or fd.load_asset
    g, s = load_asset(cfg.GOLD), load_asset(cfg.SILVER)
    if not (g.ok and s.ok):
        return None, []
    ratio = ge.ratio_series(g.values, s.values)
    rows = []
    for label, yrs in GS_PERIODS:
        st_ = ge.period_stats(ratio, label, yrs)
        if st_ is not None and (st_.complete or yrs is None):
            rows.append({"Period": label, "Från": st_.start, "Snitt": st_.mean, "Median": st_.median,
                         "Min": st_.min, "Max": st_.max, "Nu mot perioden (percentil)": st_.percentile})
    return (ratio if len(ratio) else None), rows


# ── Rendering ───────────────────────────────────────────────────────────────
def _section(title: str, sub: str = "") -> None:
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.85rem;"
                f"margin:22px 0 6px;'>{title}" + (f" <span style='color:{DIM};letter-spacing:0;'>— {sub}</span>"
                                                  if sub else "") + "</div>", unsafe_allow_html=True)


def _why(text: str) -> None:
    with st.expander("Why? — vad måttet faktiskt mäter"):
        note(text)


LOG_TICKS = [m * 10 ** e for e in range(-1, 7) for m in (1, 2, 5)]


def _layout(title: str, height: int = 320, ytitle: str = "", log: bool = False) -> dict:
    lay = dict(PLOTLY_LAYOUT)
    lay.update(height=height, title=dict(text=title, font=dict(size=12, color=CYAN)),
               legend=dict(orientation="h", y=-0.15),
               yaxis=dict(title=ytitle, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                          type="log" if log else "linear",
                          **({"tickmode": "array", "tickvals": LOG_TICKS, "ticktext": [f"{v:g}" for v in LOG_TICKS]}
                             if log else {})))
    return lay


def _chart(fig: go.Figure, key: str) -> None:
    try:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
    except Exception:
        note("Grafen kunde inte ritas.")


def _snaps() -> dict:
    return {ccy: fs.snapshot(ccy) for ccy in cfg.CURRENCIES}


def render_fiat_debasement_page() -> None:
    page_header("🐺 Fiat Debasement", "Penningmängd, inflation, köpkraft och fiatvalutor mot reala tillgångar — "
                                      "SEK, EUR och USD. Data och uträkningar, ingen rekommendation.")
    c_r, _ = st.columns([1, 4])
    if c_r.button("🔄 Uppdatera data", key="fd_refresh"):
        fd.clear_cache()
    with st.spinner("Hämtar penningmängd, KPI, BNP, statsskuld, växelkurser och guld …"):
        snaps = _snaps()
    _overview(snaps)
    ccy = st.radio("Valuta", list(cfg.CURRENCIES), horizontal=True, key="fd_ccy")
    _money_supply(snaps[ccy])
    _inflation(snaps[ccy])
    _hundred_units()
    _fiat_vs(cfg.GOLD, ccy)
    _fiat_vs(cfg.SILVER, ccy)
    _real_assets(ccy)
    _gs_ratio(ccy)
    _methodology(snaps)


def _overview(snaps: dict) -> None:
    _section("FIAT OVERVIEW", "senaste värdet per valuta · håll över en siffra för källa och datum")
    head = "".join(f"<th>{label}</th>" for _k, label, _u in OVERVIEW_COLS)
    body = ""
    for row in overview_rows(snaps):
        cells = "".join(
            f"<td title='{c['source'] or NA} · {c['as_of'] or '—'}'>"
            f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{c['color']};"
            f"margin-right:6px;'></span><span style='color:{TEXT if c['value'] is not None else DIM};'>"
            f"{c['text']}</span>{' ⚠' if c['stale'] else ''}</td>" for c in row["cells"])
        body += (f"<tr><td style='text-align:left;color:{CCY_COLOR[row['currency']]};font-weight:700;'>"
                 f"{row['currency']}</td>{cells}</tr>")
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.74rem;color:{TEXT};"
                f"text-align:right;white-space:nowrap;'><tr style='color:{DIM};'><th style='text-align:left;'>Valuta</th>{head}</tr>"
                f"{body}</table></div>", unsafe_allow_html=True)
    note("Färgen är bara visuell hjälp (gränserna står i metodpanelen) — siffran gäller. M2-tillväxt och KPI "
         "är årsförändring; Monetary Gap = M2-tillväxt − real BNP-tillväxt i procentenheter; köpkraft i guld = "
         "hur mycket mer eller mindre guld en valutaenhet köper än för fem år sedan. ⚠ = inaktuell data.")
    for ccy in cfg.CURRENCIES:
        line = principle_sentence(snaps.get(ccy))
        if line:
            note(line)


def _cards(items: list) -> None:
    cols = st.columns(len(items))
    for col, (title, big, sub, color) in zip(cols, items):
        col.markdown(big_card(title, big, sub, color), unsafe_allow_html=True)


def _money_supply(snap: fs.Snapshot) -> None:
    ccy = snap.currency
    _section("MONEY SUPPLY", f"{ccy} · Money Supply Growth — inte konsumentinflation")
    m2 = snap.loaded.get(cfg.M2)
    g = snap.get
    since = st.selectbox("CAGR sedan", START_CHOICES[:-1], index=3, key="fd_m2_since")
    cs = fe.cagr_since(m2.values if m2 is not None else None, start_date(since))
    _cards([("M2 YoY", fmt(g("m2_yoy").value), g("m2_yoy").as_of or NA, color_for("m2_yoy", g("m2_yoy").value)),
            ("M2 CAGR 5 ÅR", fmt(g("m2_cagr5").value), "per år", CYAN),
            ("M2 CAGR 10 ÅR", fmt(g("m2_cagr10").value), "per år", CYAN),
            (f"M2 CAGR SEDAN {since}", fmt(cs), "per år", CYAN)])
    fig = go.Figure()
    for c in cfg.CURRENCIES:
        y = fe.yoy(fd.load(cfg.M2, c).values)
        if y is not None:
            y = y[y.index >= start_date(since)]
            fig.add_trace(go.Scatter(x=y.index, y=y.values, name=c, line=dict(color=CCY_COLOR[c], width=1.6)))
    fig.add_hline(y=0, line=dict(color=DIM, dash="dot", width=1))
    fig.update_layout(**_layout("M2 — ÅRSFÖRÄNDRING (%)", ytitle="%"))
    _chart(fig, "fd_m2_chart")
    gap, gap5 = g("monetary_gap"), g("monetary_gap5")
    _cards([("MONETARY GAP", fmt(gap.value, "pe"), gap.as_of or NA, color_for("monetary_gap", gap.value)),
            ("MONETARY GAP 5 ÅR", fmt(gap5.value, "pe"), "årstakt", color_for("monetary_gap", gap5.value)),
            ("REAL BNP YoY", fmt(g("gdp_yoy").value), g("gdp_yoy").as_of or NA, color_for("gdp_yoy", g("gdp_yoy").value)),
            ("REAL BNP CAGR 10 ÅR", fmt(g("gdp_cagr10").value), "per år", CYAN)])
    gs = fe.gap_series(m2.values if m2 is not None else None,
                       snap.loaded[cfg.GDP].values if cfg.GDP in snap.loaded else None)
    if gs is not None:
        gs = gs[gs.index >= start_date(since)]
        fig = go.Figure(go.Bar(x=gs.index, y=gs.values, marker_color=[RED if v > 0 else GREEN for v in gs.values],
                               name="Monetary Gap"))
        fig.update_layout(**_layout(f"MONETARY GAP {ccy} — M2-TILLVÄXT MINUS REAL BNP-TILLVÄXT (PE, KVARTAL)",
                                    height=260, ytitle="pe"))
        _chart(fig, "fd_gap_chart")
    note(f"Källor: {g('m2_yoy').source or NA} · {snap.loaded[cfg.GDP].data.label if snap.loaded[cfg.GDP].ok else NA}. "
         f"Senast hämtat: {m2.data.last_updated if m2 is not None and m2.ok else NA}.")
    _why((m2.definition if m2 is not None and m2.definition else "") + " Monetary Gap beskriver penningmängdens "
         "tillväxt i förhållande till den reala produktionen (\"money supply growth relative to real economic "
         "output\"). Det är inte faktisk inflation: omloppshastigheten, efterfrågan på pengar och var pengarna "
         "hamnar (tillgångar eller konsumtion) avgör om tillväxten syns i KPI.")


def _inflation(snap: fs.Snapshot) -> None:
    ccy = snap.currency
    _section("INFLATION & PURCHASING POWER", f"{ccy} · KPI och köpkraft")
    g = snap.get
    _cards([("KPI YoY", fmt(g("cpi_yoy").value), g("cpi_yoy").as_of or NA, color_for("cpi_yoy", g("cpi_yoy").value)),
            ("KÄRN-KPI YoY", fmt(g("core_yoy").value), g("core_yoy").as_of or NA, CYAN),
            ("KPI CAGR 5 ÅR", fmt(g("cpi_cagr5").value), "per år", CYAN),
            ("KPI CAGR 10 ÅR", fmt(g("cpi_cagr10").value), "per år", CYAN)])
    c1, c2 = st.columns([2, 2])
    choice = c1.selectbox("PURCHASING POWER INDEX — start", START_CHOICES, index=3, key="fd_pp_start")
    custom = c2.date_input("Eget startdatum", value=pd.Timestamp("2000-01-01"), key="fd_pp_custom",
                           min_value=pd.Timestamp("1914-01-01"), max_value=pd.Timestamp.today()) \
        if choice == "Eget datum" else None
    start = start_date(choice, custom)
    lines = power_lines(start)
    cpi_ld = cpi_for_power(ccy)
    cum = fe.cumulative_change(cpi_ld.values, start)
    remain = 100 / (1 + cum / 100) if cum is not None and cum > -100 else None
    _cards([(f"KPI SEDAN {start.date()}", fmt(cum), cpi_ld.data.label if cpi_ld.ok else NA, AMBER),
            ("ESTIMATED PURCHASING POWER REMAINING", NA if remain is None else f"{remain:.0f} av 100",
             "enligt vald KPI-serie", RED if remain is not None and remain < 70 else AMBER)])
    fig = go.Figure()
    for c, (s, _txt) in lines.items():
        if s is not None:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=c, line=dict(color=CCY_COLOR[c], width=1.8)))
    fig.add_hline(y=100, line=dict(color=DIM, dash="dot", width=1))
    fig.update_layout(**_layout(f"PURCHASING POWER INDEX — {start.date()} = 100", height=340))
    _chart(fig, "fd_pp_chart")
    for _c, (_s, txt) in lines.items():
        note(txt)
    _why("KPI mäter prisutvecklingen för en korg av varor och tjänster som hushållen köper. Köpkraftsindex = "
         "100 × KPI vid start / KPI i dag: stiger KPI med 25 % återstår 100 / 1,25 = 80 — en valutaenhet köper "
         "ungefär 80 % av vad den gjorde vid start, enligt den valda KPI-serien. EUR (HICP) finns från 1996; "
         "före det visas ingen eurolinje i stället för en konstruerad serie.")


def _hundred_units() -> None:
    _section("WHAT HAPPENED TO 100 UNITS?", "kontanter jämfört med KPI, guld och silver")
    c1, c2 = st.columns(2)
    ccy = c1.radio("Valuta", list(cfg.CURRENCIES), horizontal=True, key="fd_units_ccy")
    choice = c2.selectbox("Start", START_CHOICES[3:-1], index=0, key="fd_units_start")
    log = st.toggle("Logaritmisk skala", value=True, key="fd_units_log",
                    help="Lika stora procentuella rörelser blir lika stora i grafen — alla linjer syns.")
    res = hundred_units(ccy, start_date(choice))
    if not res["lines"]:
        note(f"{NA} — " + " ".join(res["notes"]))
        return
    fig = go.Figure()
    colors = {"Kontanter, köpkraft (KPI)": RED, "I guld": GOLD, "I silver": GREY}
    fig.add_hline(y=100, line=dict(color=DIM, dash="dot", width=1),
                  annotation_text=f"100 {ccy} som kontanter (nominellt)", annotation_font_color=DIM,
                  annotation_position="top left")
    for name, s in res["lines"].items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, line=dict(color=colors.get(name, TEXT), width=1.8)))
    for seg in res["segments"]:
        if pd.Timestamp(seg["to"]) >= res["start"]:
            fig.add_vrect(x0=max(pd.Timestamp(seg["from"]), res["start"]), x1=pd.Timestamp(seg["to"]),
                          fillcolor="rgba(201,168,76,0.08)", line_width=0,
                          annotation_text="terminspris", annotation_font_color=DIM)
    fig.update_layout(**_layout(f"100 {ccy} FRÅN {res['start'].date()} — I DAGENS {ccy}", height=380,
                                ytitle=ccy, log=log))
    _chart(fig, "fd_units_chart")
    last = {n: float(s.iloc[-1]) for n, s in res["lines"].items() if s is not None and len(s)}
    if last:
        _cards([(n.upper(), f"{v:,.0f}", f"{ccy} i dag av 100", colors.get(n, TEXT)) for n, v in last.items()])
    for n in res["notes"]:
        note(n)
    _why("Röd linje: vad 100 enheter kontanter köper i dag mätt med KPI (köpkraft). Guld/silver: vad 100 enheter "
         "växlade till metallen vid start är värda i dag, i samma valuta. Visar historisk utveckling — inte vad "
         "som kommer att hända, och inte att guld alltid skyddar (perioder med fallande guldpris finns).")


def _fiat_vs(name: str, ccy: str) -> None:
    label = cfg.CONCEPT_LABEL[name]
    _section(f"FIAT VS {'GOLD' if name == cfg.GOLD else 'SILVER'}",
             f"hur mycket {label.lower()} en valutaenhet köper · start = 100")
    choice = st.selectbox("Start", START_CHOICES[3:-1], index=0, key=f"fd_{name}_start")
    start = start_date(choice)
    lines = fiat_vs_lines(name, start)
    fig = go.Figure()
    for c, (s, _t) in lines.items():
        if s is not None:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{c}/{label}", line=dict(color=CCY_COLOR[c], width=1.6)))
    fig.add_hline(y=100, line=dict(color=DIM, dash="dot", width=1))
    fig.update_layout(**_layout(f"VALUTA MÄTT I {label.upper()} ({start.year} = 100)", log=True))
    _chart(fig, f"fd_{name}_chart")
    rows = window_rows(lines)
    head = "".join(f"<th>{k}</th>" for k in rows[0]) if rows else ""
    body = "".join("<tr>" + "".join(
        f"<td style='text-align:left;color:{CCY_COLOR[r['Valuta']]};font-weight:700;'>{v}</td>" if k == "Valuta"
        else f"<td style='color:{GREEN if (v or 0) > 0 else RED if v is not None else DIM};'>{fmt(v)}</td>"
        for k, v in r.items()) + "</tr>" for r in rows)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
                f"text-align:right;white-space:nowrap;'><tr style='color:{DIM};'>{head}</tr>{body}</table></div>",
                unsafe_allow_html=True)
    for _c, (_s, txt) in lines.items():
        if NA in txt:
            note(txt)
    pair = power_pair(ccy, start, name)
    if pair:
        fig = go.Figure()
        for i, (n, s) in enumerate(pair.items()):
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=n,
                                     line=dict(color=CCY_COLOR[ccy] if i == 0 else ASSET_COLOR[name], width=1.8)))
        fig.add_hline(y=100, line=dict(color=DIM, dash="dot", width=1))
        fig.update_layout(**_layout(f"KÖPKRAFT: {label.upper()} VS {ccy} (START = 100)", height=300, log=True))
        _chart(fig, f"fd_{name}_pair")
    _why(f"Linjen faller när valutan köper mindre {label.lower()} än vid start, och stiger när den köper mer. "
         f"Skillnaden mellan SEK, EUR och USD är växelkursens rörelse, eftersom {label.lower()} prissätts i USD. "
         f"Den nedre grafen visar två separata saker: valutans köpkraft enligt KPI, och hur mycket varor en "
         f"uns {label.lower()} köper (priset i {ccy} delat med KPI). Stigande metallpris betyder inte "
         f"nödvändigtvis att valutan kollapsar.")


def _real_assets(ccy: str) -> None:
    _section("REAL ASSET PROTECTION", f"reala tillgångar i {ccy} mot KPI · start = 100")
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Start", START_CHOICES[3:-1], index=1, key="fd_ra_start")
    log = c2.toggle("Log-skala", value=True, key="fd_ra_log")
    start = start_date(choice)
    lines, notes = real_asset_lines(ccy, start)
    if not lines:
        note(f"{NA} — " + " ".join(notes))
        return
    fig = go.Figure()
    by_label = {cfg.CONCEPT_LABEL[n]: ASSET_COLOR[n] for n in REAL_ASSETS}
    for n, s in lines.items():
        dash = "dot" if n.startswith("KPI") else None
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=n,
                                 line=dict(color=by_label.get(n, RED), width=1.6, dash=dash)))
    fig.update_layout(**_layout(f"REALA TILLGÅNGAR I {ccy} — {start.date()} = 100", height=360, log=log))
    _chart(fig, "fd_ra_chart")
    body = "".join(f"<tr><td style='text-align:left;'>{n}</td><td>{float(s.iloc[-1]):,.0f}</td>"
                   f"<td>{fmt(fe.change_pct(s, 1))}</td><td>{fmt(fe.change_pct(s, 5))}</td></tr>"
                   for n, s in lines.items() if s is not None and len(s))
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
                f"text-align:right;white-space:nowrap;'><tr style='color:{DIM};'><th style='text-align:left;'>"
                f"Tillgång</th><th>Nu (start=100)</th><th>1 år</th><th>5 år</th></tr>{body}</table></div>",
                unsafe_allow_html=True)
    for n in notes:
        note(n)
    _why("Över den prickade KPI-linjen har tillgången stigit mer än prisnivån i vald valuta under perioden, under "
         "den mindre. Historik, ingen prognos — reala tillgångar kan falla kraftigt och länge.")


def _gs_ratio(ccy: str) -> None:
    _section("GOLD/SILVER RATIO", "samma kvot som REGIME → Råvaror → 🥇🥈 Guld/Silver")
    ratio, rows = gs_ratio()
    if ratio is None:
        note(f"Guld/silver-kvoten: {NA}")
        return
    cur = float(ratio.iloc[-1])
    avg = next((r["Snitt"] for r in rows if r["Period"] == "Hela historiken"), None)
    _cards([("GULD/SILVER NU", f"{cur:.1f}", str(ratio.index[-1].date()), GOLD),
            ("HISTORISKT SNITT", "—" if avg is None else f"{avg:.1f}", f"sedan {ratio.index[0].date()}", CYAN)])
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    cpi = cpi_for_power(ccy)
    pp = fe.purchasing_power(cpi.values, ratio.index[0]) if cpi.ok else None
    fig = go.Figure(go.Scatter(x=ratio.index, y=ratio.values, name="Guld/silver", line=dict(color=GOLD, width=1.4)))
    if pp is not None:
        fig.add_trace(go.Scatter(x=pp.index, y=pp.values, name=f"{ccy} köpkraft (KPI, start = 100)", yaxis="y2",
                                 line=dict(color=CCY_COLOR[ccy], width=1.6)))
    lay = _layout(f"GULD/SILVER OCH {ccy} KÖPKRAFT", height=320)
    lay["yaxis2"] = dict(overlaying="y", side="right", showgrid=False, title="köpkraft")
    fig.update_layout(**lay)
    _chart(fig, "fd_gs_chart")
    _why("Kvoten = guldpris / silverpris. Hög kvot = silver billigt relativt guld, låg = dyrt. Grafen visar om "
         "silver blivit billigare eller dyrare mot guld samtidigt som valutans köpkraft ändrats — två serier "
         "bredvid varandra, inget orsakssamband. Före 2006 bygger kvoten på terminspris.")


METHODOLOGY = (
    ("1. KPI", "Konsumentprisindex mäter priset på en korg av varor och tjänster. SEK: SCB:s KPI (skuggindex "
               "2020=100) och kärnmåttet KPIF-XE. EUR: HICP. USD: CPI-U (BLS)."),
    ("2. M2", "Penningmängden: sedlar, mynt och inlåning upp till en viss bindningstid. Definitionerna skiljer "
              "sig (Fed respektive ECB/SCB) — jämför nivåer med försiktighet, tillväxttakter bättre."),
    ("3. Real BNP", "Produktionen i fasta priser (volym). Kvartalsdata, publiceras med ungefär en månads "
                    "eftersläpning och revideras."),
    ("4. Monetary Gap", "M2-tillväxt minus real BNP-tillväxt (procentenheter, årstakt per kvartal). Mäter "
                        "penningmängd relativt real produktion — inte inflation."),
    ("5. Köpkraft", "100 × KPI vid start / KPI i dag. Lång historik: SCB:s levnadskostnadsindex (1914–) och "
                    "CPI-U ej säsongsjusterad (1913–)."),
    ("6. Guld och silver", "Börsdata spotpris från 2006; före det Yahoo-terminer, märkta terminspris (ingen "
                           "nivåjustering vid skarven). Priset i SEK/EUR räknas med Riksbankens respektive ECB:s "
                           "växelkurs samma dag — dagar utan växelkurs används inte."),
    ("7. Wolf Debasement Index", "Byggs i en senare version: komponenterna normaliseras mot sin egen historik "
                                 "(percentil) innan viktning. Det blir en modellbaserad indikator, inte ett "
                                 "officiellt mått."),
    ("8. Antaganden", "Årsförändring jämförs med samma månad/kvartal året innan på kalenderdatum. CAGR kräver "
                      "en observation exakt n år bakåt, annars visas DATA UNAVAILABLE. Färggränser (visuell "
                      "hjälp): M2 3/7 %, KPI 2,5/5 %, Monetary Gap 2/4 pe, statsskuld 60/100 %, real BNP 2/0 %, "
                      "köpkraft i guld 0/−30 %."),
    ("9. Begränsningar", "Revideringar i BNP och penningmängd, olika M2-definitioner, ombasning av HICP till "
                         "2025=100, USA:s offentliga skuld bara årlig (IMF), guld före 2006 är terminspris, "
                         "euroområdets sammansättning har ändrats över tid. Koppar och olja (Börsdata) finns "
                         "från 2006, bitcoin från 2014; fastigheter saknas än. Guld/silver-kvoten före 2006 "
                         "bygger på terminspris."),
)
PRINCIPLES = ("Inflation ≠ M2-tillväxt", "Valutaförsvagning ≠ KPI-inflation",
              "Penningmängdstillväxt ≠ automatisk förlust av köpkraft",
              "Stigande guldpris ≠ nödvändigtvis fiatkollaps")


def source_rows(snaps: dict) -> list:
    """Varje försök bakom sidan: begrepp, valuta, källa, serie, status, från–till, hämtat, SCB-val."""
    rows, seen = [], set()
    for ccy, snap in snaps.items():
        for concept, ld in (snap.loaded or {}).items():
            for sd in ld.attempts:
                key = (concept, ld.currency, sd.source, sd.series_id)
                if key in seen:
                    continue
                seen.add(key)
                status = "används" if ld.data is sd or (ld.ok and ld.data.series_id == sd.series_id
                                                       and ld.data.source == sd.source) else \
                    ("reserv" if sd.ok else "FEL")
                rows.append({"Begrepp": cfg.CONCEPT_LABEL.get(concept, concept), "Valuta": ld.currency,
                             "Källa": sd.source, "Serie": sd.series_id, "Status": status,
                             "Från": sd.first or "—", "Till": sd.last or "—", "Frekvens": sd.frequency or "—",
                             "Enhet": sd.unit, "Hämtat": sd.last_updated or "—",
                             "Inaktuell": "⚠" if fd.is_stale(sd) else "",
                             "Val (SCB)": "; ".join(f"{k}={v}" for k, v in (sd.meta.get("chosen") or {}).items()),
                             "Fel": sd.error or ""})
    return rows


def _methodology(snaps: dict) -> None:
    _section("METHODOLOGY", "vad måtten mäter, antaganden och data")
    with st.expander("Metod, antaganden och begränsningar"):
        for title, text in METHODOLOGY:
            st.markdown(f"<div style='color:{GOLD};font-size:0.8rem;font-weight:700;margin-top:6px;'>{title}</div>",
                        unsafe_allow_html=True)
            note(text)
        st.markdown(f"<div style='color:{CYAN};font-size:0.8rem;margin-top:10px;'>"
                    + " · ".join(PRINCIPLES) + "</div>", unsafe_allow_html=True)
    with st.expander("Datakällor och datakvalitet"):
        rows = source_rows(snaps)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        note("Primärkällan används när den fungerar; reserver bara om den inte gör det. ⚠ = senaste "
             "observationen är äldre än vad frekvensen tillåter. Felaktiga hämtningar visas här i stället för som 0.")
