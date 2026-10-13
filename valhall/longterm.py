"""
valhall/longterm.py — fas 4b: hjälper cykeln de långsiktiga köpen? (backtest 2008–idag, utan facit)

Två nivåer, båda med cykelläget PÅ KÖPDAGEN (cykelmotorns dagliga serie, senast kända rad):

  Segment   varje månadsskifte: segmentets (temats) läge — status, fas, cykelläge, cykelstart — och
            avkastningen 12/24/36 månader framåt, mot marknaden (hela Norden / S&P 500) för
            aktiesegment och absolut för råvaruteman. Plus "sälj på RÖD": samma köp men sålt vid
            första månadsskiftet då statusen är RÖD.
  Aktier    varje kvartalsskifte: likvida nordiska aktier som klarar Quality Contrarians hårda grind
            PUNKT FÖR PUNKT ur Börsdatas ROIC-historik (KPI 37, samma nyckeltal som skärmen):
              Kvalitet  senast kända ROIC > 15 %                          (quality-läget)
              Djup      senast kända ROIC > 8 % eller medianen > 8 %      (deep_contrarian)
            En årsrapport räknas som känd fyra månader efter räkenskapsårets slut (LAG_MONTHS).
            Framåtavkastningen mot hela Norden, grupperad på segmentets nordiska läge, och en
            kvartalsportfölj (likaviktad, alla som klarar filtret) per filter.

Kurshopp större än 60 % på en dag räknas som datafel (clean). Begränsningar: aktieuniversumet är dagens noterade bolag (avnoterade saknas — överlevnadsbias, lika för
alla filter); kvalitetsgrinden är bara ROIC (skärmens fulla poäng har fler delar utan historik);
överlappande 12–36-månadersfönster gör observationerna beroende — jämför filtren, inte t-värden.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from berserk import signals as bsg
from valhall import cycle

HORIZONS = (12, 24, 36)
LAG_MONTHS = 4
GATE_QUALITY, GATE_DEEP = 15.0, 8.0                    # contrarian_alpha.quality.GATE_ROIC_*
PERIODS = (("2008–2020", "2020-12-31"), ("2021–", None))


def period_of(date) -> str:
    return "2008–2020" if pd.Timestamp(date) <= pd.Timestamp("2020-12-31") else "2021–"


def month_ends(index: pd.DatetimeIndex, start, step: int = 1) -> list:
    """Sista handelsdagen varje (step:e) månad från start."""
    s = pd.Series(index, index=index)
    s = s[s.index >= pd.Timestamp(start)]
    if s.empty:
        return []
    last = s.groupby([s.index.year, s.index.month]).max().tolist()
    return last[::step]


def _asof(series: pd.Series, date) -> Optional[float]:
    k = int(series.index.searchsorted(pd.Timestamp(date), side="right")) - 1
    if k < 0:
        return None
    v = series.iloc[k]
    return None if pd.isna(v) else float(v)


def fwd(series: Optional[pd.Series], date, months: int) -> Optional[float]:
    """Avkastning i % från datumet till samma dag months månader senare (None när slutet saknas)."""
    if series is None or series.empty:
        return None
    end = pd.Timestamp(date) + pd.DateOffset(months=months)
    if end > series.index[-1]:
        return None
    a, b = _asof(series, date), _asof(series, end)
    return (b / a - 1) * 100 if a and b is not None else None


_RED: dict = {}


def red_months(frame: pd.DataFrame) -> np.ndarray:
    """Månadsskiftena (sista handelsdagen) då statusen är RÖD — beräknat en gång per serie."""
    hit = _RED.get(id(frame))
    if hit is not None and hit[0] is frame:
        return hit[1]
    st = frame["status"]
    last = st.groupby([st.index.year, st.index.month]).tail(1)
    arr = last[last == cycle.ROD].index.values
    _RED[id(frame)] = (frame, arr)
    return arr


def red_exit(frame: pd.DataFrame, series: pd.Series, date, months: int) -> Optional[float]:
    """Som fwd, men sålt vid första månadsskiftet (efter köpet) då statusen är RÖD."""
    end = pd.Timestamp(date) + pd.DateOffset(months=months)
    if series is None or series.empty or end > series.index[-1]:
        return None
    a = _asof(series, date)
    if not a:
        return None
    reds = red_months(frame) if frame is not None and not frame.empty else np.array([], dtype="datetime64[ns]")
    k = int(np.searchsorted(reds, np.datetime64(pd.Timestamp(date)), side="right"))
    sell = pd.Timestamp(reds[k]) if k < len(reds) and pd.Timestamp(reds[k]) <= end else end
    b = _asof(series, sell)
    return (b / a - 1) * 100 if b is not None else None


def _row(frame: Optional[pd.DataFrame], date) -> dict:
    if frame is None or frame.empty:
        return {}
    k = int(frame.index.searchsorted(pd.Timestamp(date), side="right")) - 1
    return {} if k < 0 else frame.iloc[k].to_dict()


def _state(frame, date) -> dict:
    r = _row(frame, date)
    return {"status": r.get("status", cycle.NODATA), "phase": r.get("phase", cycle.NODATA),
            "cykel": r.get("cykel", cycle.NODATA), "cycle_start": bool(r.get("cycle_start", False))}


def _returns(series, bench, date, frame=None) -> dict:
    """{r12, x12, red12, xred12, …}: absolut, mot riktmärket och samma med sälj på RÖD."""
    out = {}
    for h in HORIZONS:
        r, b = fwd(series, date, h), fwd(bench, date, h) if bench is not None else 0.0
        out[f"r{h}"] = r
        out[f"x{h}"] = r - b if r is not None and b is not None else None
        if frame is not None:
            rr = red_exit(frame, series, date, h)
            out[f"red{h}"] = rr
            # sälj på RÖD mot att ligga kvar i riktmärket hela perioden (kassan antas stå still)
            out[f"xred{h}"] = rr - b if rr is not None and b is not None else None
    return out


# ── Segmentnivå ─────────────────────────────────────────────────────────────
def segment_rows(frames: dict, bench: Optional[pd.Series], start, where: str, step: int = 1) -> list:
    """En rad per segment och månadsskifte. frames = {nyckel: cycle_frame} (kolumnen close = serien)."""
    out = []
    for key, f in (frames or {}).items():
        if f is None or f.empty:
            continue
        s = f["close"].astype(float)
        for d in month_ends(f.index, start, step):
            st = _state(f, d)
            if st["status"] == cycle.NODATA:
                continue
            out.append({"where": where, "key": key, "date": str(d.date()), "period": period_of(d), **st,
                        **_returns(s, bench, d, f)})
    return out


# ── Aktienivå ───────────────────────────────────────────────────────────────
def clean(s: pd.Series, max_abs_ret: float = bsg.JUMP_MAX) -> pd.Series:
    """Kursserien utan datafel: dagsrörelser större än max_abs_ret (Yahoos split- och tickerhopp) räknas
    som 0 % — samma spärr som de nordiska sektorindexen (nordic_index.equal_weight_index)."""
    s = s.astype(float).dropna()
    if len(s) < 2:
        return s
    r = s.pct_change(fill_method=None)
    r = r.where(r.abs() <= max_abs_ret, 0.0).fillna(0.0)
    return float(s.iloc[0]) * (1 + r).cumprod()


def known_roic(values: list, date) -> list:
    """ROIC-värdena (år, %) som var kända på datumet — årsrapporten känd LAG_MONTHS efter årets slut."""
    out = []
    for v in values or []:
        y, x = v.get("y"), v.get("v")
        if y is None or x is None:
            continue
        known = pd.Timestamp(int(y) + 1, 1, 1) + pd.DateOffset(months=LAG_MONTHS)
        if known <= pd.Timestamp(date):
            out.append((int(y), float(x)))
    return [x for _y, x in sorted(out)]


def gates(roic: list) -> dict:
    """Quality Contrarians hårda grind ur kända ROIC-värden (senaste sist)."""
    if not roic:
        return {"q": False, "d": False}
    last, med = roic[-1], float(np.median(roic[-5:]))
    return {"q": last > GATE_QUALITY, "d": last > GATE_DEEP or med > GATE_DEEP}


def stock_rows(closes: dict, turnover: dict, universe: dict, roic: dict, nordic_frames: dict,
               bench: Optional[pd.Series], start, step: int = 3, min_turnover: float = 1e6,
               days: int = 60) -> list:
    """En rad per likvid aktie och kvartalsskifte: grindarna, segmentets nordiska läge och avkastningen
    framåt (3 mån för portföljen + HORIZONS). universe = {ticker: {segment, ins_id}}, roic = {ins_id: [{y, v}]}."""
    if bench is None or bench.empty:
        return []
    dates = month_ends(bench.index, start, step)
    out = []
    for t, s in closes.items():
        info = universe.get(t) or {}
        seg, vals = info.get("segment"), roic.get(info.get("ins_id"))
        if not seg or not vals or s is None or s.empty:
            continue
        s = clean(s)
        tv = turnover.get(t)
        liq = tv.rolling(days, min_periods=days // 2).mean().shift(1) if tv is not None else None
        frame = nordic_frames.get(seg)
        for d in dates:
            if d < s.index[0] or (liq is not None and not (_asof(liq, d) or 0) >= min_turnover):
                continue
            g = gates(known_roic(vals, d))
            if not (g["q"] or g["d"]):
                continue
            out.append({"ticker": t, "segment": seg, "date": str(d.date()), "period": period_of(d), **g,
                        **_state(frame, d), "r3": fwd(s, d, 3), **_returns(s, bench, d, frame)})
    return out


# ── Sammanställning ─────────────────────────────────────────────────────────
def summarize(rows: list, by: tuple, keys=None) -> list:
    """[{grupp…, period, n, för varje horisont: snitt/median/träff mot riktmärket, absolut snitt och
    sälj på RÖD}] — per grupp × period (alla / 2008–2020 / 2021–)."""
    keys = keys or HORIZONS
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in by), []).append(r)
    out = []
    for g, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        for per, _end in (("alla", None),) + PERIODS:
            sel = rs if per == "alla" else [r for r in rs if r["period"] == per]
            if not sel:
                continue
            row = {**dict(zip(by, g)), "period": per, "n": len(sel)}
            for h in keys:
                x = [r[f"x{h}"] for r in sel if r.get(f"x{h}") is not None]
                a = [r[f"r{h}"] for r in sel if r.get(f"r{h}") is not None]
                red = [r[f"xred{h}"] for r in sel if r.get(f"xred{h}") is not None]
                row[f"x{h}"] = round(float(np.mean(x)), 1) if x else None
                row[f"x{h}_med"] = round(float(np.median(x)), 1) if x else None
                row[f"hit{h}"] = round(sum(1 for v in x if v > 0) / len(x) * 100, 0) if x else None
                row[f"r{h}"] = round(float(np.mean(a)), 1) if a else None
                row[f"xred{h}"] = round(float(np.mean(red)), 1) if red else None
                row[f"n{h}"] = len(x)
            out.append(row)
    return out


STOCK_FILTERS = {
    "Kvalitet — alla": lambda r: r["q"],
    "Kvalitet — segment GRÖN": lambda r: r["q"] and r["status"] == cycle.GRON,
    "Kvalitet — segment ej RÖD": lambda r: r["q"] and r["status"] in (cycle.GRON, cycle.GUL),
    "Kvalitet — segment RÖD (hatat)": lambda r: r["q"] and r["status"] == cycle.ROD,
    "Kvalitet — cykel TIDIG": lambda r: r["q"] and r["cykel"] == cycle.TIDIG,
    "Kvalitet — cykelstart (VÄNDER)": lambda r: r["q"] and r["cycle_start"],
    "Djup — alla": lambda r: r["d"],
    "Djup — segment RÖD (hatat)": lambda r: r["d"] and r["status"] == cycle.ROD,
    "Djup — cykel TIDIG": lambda r: r["d"] and r["cykel"] == cycle.TIDIG,
    "Djup — segment ej RÖD": lambda r: r["d"] and r["status"] in (cycle.GRON, cycle.GUL),
}


def filter_table(rows: list, filters: Optional[dict] = None) -> list:
    """summarize per filter (alla perioder)."""
    out = []
    for name, f in (filters or STOCK_FILTERS).items():
        sel = [dict(r, filter=name) for r in rows if f(r)]
        out += summarize(sel, ("filter",)) if sel else [{"filter": name, "period": "alla", "n": 0}]
    return out


def portfolio(rows: list, f: Callable, bench: Optional[pd.Series] = None, start=None, end=None) -> dict:
    """Kvartalsportfölj: vid varje kvartalsskifte likavikt i alla som klarar filtret, hålls tre månader.
    Tomt kvartal = kassa (0 %). CAGR, max drawdown (kvartalsvis), snittantal och riktmärkets CAGR."""
    by_date: dict = {}
    for r in rows:
        by_date.setdefault(r["date"], [])
        if f(r) and r.get("r3") is not None:
            by_date[r["date"]].append(r["r3"])
    dates = sorted(d for d in by_date if (start is None or d >= start) and (end is None or d <= end))
    if not dates:
        return {"quarters": 0}
    eq, peak, dd, sizes, beq = 1.0, 1.0, 0.0, [], 1.0
    for d in dates:
        rs = by_date[d]
        sizes.append(len(rs))
        eq *= 1 + (float(np.mean(rs)) / 100 if rs else 0.0)
        peak = max(peak, eq)
        dd = min(dd, eq / peak - 1)
        b = fwd(bench, d, 3) if bench is not None else None
        beq *= 1 + (b or 0.0) / 100
    years = len(dates) / 4
    return {"quarters": len(dates), "cagr": round((eq ** (1 / years) - 1) * 100, 1),
            "dd": round(dd * 100, 1), "avg_n": round(float(np.mean(sizes)), 1),
            "bench_cagr": round((beq ** (1 / years) - 1) * 100, 1) if bench is not None else None}


def portfolios(rows: list, bench: Optional[pd.Series], filters: Optional[dict] = None) -> list:
    out = []
    for name, f in (filters or STOCK_FILTERS).items():
        for per, end in (("alla", None),) + PERIODS:
            start = "2021-01-01" if per == "2021–" else None
            out.append({"filter": name, "period": per, **portfolio(rows, f, bench, start, end)})
    return out
