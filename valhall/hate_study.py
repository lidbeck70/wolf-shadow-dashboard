"""
valhall/hate_study.py — hatets varaktighet: hur länge sektorns storbolag (och råvarorna) är hatade, om det
lönar sig att vänta, och om hat-bredden är en rotationssignal. Förregistrerat i
docs/forskning/hatets_varaktighet_spec.md (version 1) — parametrarna här är låsta av den sidan.

Rent och utan nätverk (hate_duration_study.py hämtar data). Cykelmotorn (valhall.cycle) är oförändrad;
"hatad" är samma villkor som långsidans E1 (valhall.long_study.cheap_hated).

  D1  hatperioder (start, slut, censurerad) och Kaplan–Meier-längd per grupp
  D2  köp vid start, efter 6 och 12 månader — hålls 2 år, relativt marknaden, parvisa skillnader
  D3  hat-bredd i sektorns topp 10: B1 ≥ 50 % (massivt hatad), B2 ≤ 30 % efter B1 (hatet släpper)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from valhall import long_study as lst

SPEC_VERSION = 1
SPEC_PATH = "docs/forskning/hatets_varaktighet_spec.md"
GAP, MONTH = 21, 21
DELAYS, HOLD = (0, 126, 252), 504
START = "2008-01-01"
TOP_N, TOP_VARIANT = 10, 5
B1_LEVEL, B2_LEVEL, B2_WINDOW, DECLUSTER, MIN_MEMBERS = 0.50, 0.30, 756, 252, 5
BREADTH_H, MAIN_H = (252, 504), 504
VAL_PCT, VAL_MIN_YEARS, LAG_MONTHS = 30.0, 5, 4
N_BOOT, SEED = 5000, 20261025
D2_MIN_N, D2_MIN_YEARS, D3_MIN_N, D3_MIN_YEARS = 30, 5, 15, 5


# ── Hatad och hatperioder ───────────────────────────────────────────────────
def hated(frame: Optional[pd.DataFrame]) -> pd.Series:
    """1/0 per dag (NaN innan percentilen kan räknas): tioårspercentil ≤ 30 och hatpoäng ≥ 60."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    h = lst.cheap_hated(frame).astype(float)
    return h.where(frame["pct10"].notna())


def episodes(h: pd.Series, gap: int = GAP) -> list:
    """[{start, end, length, censored}] — en period slutar när bolaget inte varit hatat `gap` dagar i rad.
    Perioder som börjar första dagen läget alls finns stryks (okänd start)."""
    if h is None or not len(h.dropna()):
        return []
    idx = h.index
    valid = np.flatnonzero(h.notna().to_numpy())
    first_valid, last = int(valid[0]), len(h) - 1
    pos = np.flatnonzero((h == 1).to_numpy())
    out = []
    if not len(pos):
        return out
    s = e = int(pos[0])
    for p in list(pos[1:]) + [None]:
        if p is not None and p - e <= gap:
            e = int(p)
            continue
        if s != first_valid:
            out.append({"start": idx[s], "end": idx[e], "length": e - s + 1, "censored": last - e < gap})
        if p is not None:
            s = e = int(p)
    return out


def km(lengths: list, censored: list) -> dict:
    """Kaplan–Meier: median och kvartiler för längden (handelsdagar) och andel kvar efter 12/24 månader."""
    if not lengths:
        return {"n": 0}
    t = np.asarray(lengths, dtype=float)
    ev = ~np.asarray(censored, dtype=bool)
    times = np.unique(t[ev])
    s, surv = 1.0, []
    for x in times:
        at_risk = int((t >= x).sum())
        d = int(((t == x) & ev).sum())
        if at_risk:
            s *= 1 - d / at_risk
        surv.append((x, s))

    def q(level):
        return next((float(x) for x, v in surv if v <= level), None)

    def s_at(x):
        v = [v for tt, v in surv if tt <= x]
        return round((v[-1] if v else 1.0) * 100, 1)

    def m(v):
        return None if v is None else round(v / MONTH, 1)
    return {"n": int(len(t)), "completed": int(ev.sum()), "ongoing": int((~ev).sum()),
            "median_m": m(q(0.5)), "q25_m": m(q(0.75)), "q75_m": m(q(0.25)),
            "left_12m": s_at(12 * MONTH), "left_24m": s_at(24 * MONTH)}


# ── D2: lönar det sig att vänta? ────────────────────────────────────────────
def delay_rows(close: pd.Series, market: pd.Series, eps: list, start=START) -> list:
    """{start, year, rel: {fördröjning: relativ 2-årsavkastning}} per period med hel horisont för alla fördröjningar."""
    c = close.astype(float).dropna()
    mk = market.reindex(c.index).ffill()
    out = []
    for e in eps:
        if e["start"] < pd.Timestamp(start) or e["start"] not in c.index:
            continue
        i0 = c.index.get_loc(e["start"])
        rel = {}
        for d in DELAYS:
            a, b = i0 + d + 1, i0 + d + 1 + HOLD
            if b >= len(c) or pd.isna(mk.iloc[a]) or pd.isna(mk.iloc[b]):
                break
            rel[d] = float(c.iloc[b] / c.iloc[a] - 1) - float(mk.iloc[b] / mk.iloc[a] - 1)
        if len(rel) == len(DELAYS):
            out.append({"start": str(e["start"].date()), "year": int(e["start"].year), "rel": rel,
                        "undervalued": e.get("undervalued")})
    return out


def cluster_ci(values: list, years: list, rng: np.random.Generator, n: int = N_BOOT) -> list:
    """95 %-intervall för snittet: drar hela kalenderår med återläggning."""
    by: dict = {}
    for v, y in zip(values, years):
        by.setdefault(y, []).append(v)
    ys = list(by)
    if not ys:
        return [None, None]
    sums = np.array([sum(by[y]) for y in ys])
    cnts = np.array([len(by[y]) for y in ys])
    pick = rng.integers(0, len(ys), (n, len(ys)))
    means = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _pct(v):
    return None if v is None else round(float(v) * 100, 2)


def d2_summary(rows: list, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """Nivåerna per fördröjning och V6/V12 med intervall och utfall."""
    out = {"n": len(rows), "years": len({r["year"] for r in rows}),
           "levels": {d: {"mean": _pct(np.mean([r["rel"][d] for r in rows])) if rows else None,
                          "median": _pct(np.median([r["rel"][d] for r in rows])) if rows else None} for d in DELAYS}}
    for name, d in (("V6", 126), ("V12", 252)):
        x = [r["rel"][d] - r["rel"][0] for r in rows]
        ci = cluster_ci(x, [r["year"] for r in rows], rng, n_boot) if x else [None, None]
        med = float(np.median(x)) if x else None
        enough = len(rows) >= D2_MIN_N and out["years"] >= D2_MIN_YEARS
        if enough and ci[0] is not None and ci[0] > 0 and med > 0:
            word = "VÄNTA LÖNAR SIG"
        elif enough and ci[1] is not None and ci[1] < 0 and med < 0:
            word = "KÖP DIREKT"
        else:
            word = "INGEN SKILLNAD BEVISAD"
        out[name] = {"mean": _pct(np.mean(x)) if x else None, "median": _pct(med), "ci": [_pct(v) for v in ci],
                     "verdict": word, "enough": enough}
    return out


# ── Nordens storbolag: urval, index, bredd ──────────────────────────────────
def top_by_year(mcap: dict, info: dict, segment_of: dict, years, n: int = TOP_N) -> dict:
    """{segment: {år: [insId]}} — de n största efter föregående års börsvärde i kronor, ett aktieslag per bolag."""
    from valhall import nordic_index as ni
    from valhall import picks as vp
    out: dict = {}
    for y in years:
        by: dict = {}
        for iid, hist in mcap.items():
            v, seg = hist.get(y - 1), segment_of.get(iid)
            if v is None or not seg or v <= 0:
                continue
            i = info[iid]
            fx = ni.FX_TO_SEK.get(str(i.get("reportCurrency") or i.get("stockPriceCurrency") or "SEK").upper(), 1.0)
            by.setdefault(seg, []).append((v * fx, str(i.get("ticker") or "").upper().endswith("B"), iid))
        for seg, rows in by.items():
            seen, picked = set(), []
            for _v, _b, iid in sorted(rows, key=lambda r: (-r[0], not r[1])):
                k = vp.company_key({"name": info[iid].get("name"), "ticker": info[iid].get("ticker")})
                if k in seen:
                    continue
                seen.add(k)
                picked.append(iid)
                if len(picked) == n:
                    break
            out.setdefault(seg, {})[y] = picked
    return out


def restrict(members: dict, have: set, n: int) -> dict:
    """{år: [id]} — de n första med kursdata."""
    return {y: [i for i in ids if i in have][:n] for y, ids in members.items()}


def universe_index(closes: dict, members_by_year: dict) -> pd.Series:
    """Likaviktat index (start 100) av årets medlemmar: dagens snitt av dagsavkastningen, kedjat."""
    if not closes:
        return pd.Series(dtype=float)
    df = pd.DataFrame(closes).sort_index()
    rets = df.pct_change(fill_method=None)
    years = pd.Series(df.index.year, index=df.index)
    r = pd.Series(np.nan, index=df.index)
    for y, ids in members_by_year.items():
        cols = [i for i in ids if i in rets.columns]
        if cols:
            m = (years == int(y)).to_numpy()
            r[m] = rets.loc[m, cols].mean(axis=1)
    r = r.fillna(0.0)
    r.iloc[0] = 0.0
    first = np.flatnonzero(df.notna().any(axis=1).to_numpy())
    return (100 * (1 + r).cumprod()).iloc[first[0]:] if len(first) else pd.Series(dtype=float)


def breadth(states: dict, members_by_year: dict, calendar: pd.DatetimeIndex) -> pd.Series:
    """Andelen hatade av årets medlemmar per dag (NaN när färre än MIN_MEMBERS har ett läge)."""
    df = pd.DataFrame({k: v.reindex(calendar) for k, v in states.items()})
    years = pd.Series(calendar.year, index=calendar)
    out = pd.Series(np.nan, index=calendar)
    for y, ids in members_by_year.items():
        cols = [i for i in ids if i in df.columns]
        if not cols:
            continue
        m = (years == int(y)).to_numpy()
        sub = df.loc[m, cols]
        cnt = sub.notna().sum(axis=1)
        out[m] = (sub.sum(axis=1) / cnt).where(cnt >= MIN_MEMBERS)
    return out


def breadth_events(b: pd.Series) -> dict:
    """{"B1": [datum], "B2": [datum]} ur hat-bredden."""
    s = b.dropna()
    if not len(s):
        return {"B1": [], "B2": []}
    hi = s >= B1_LEVEL
    raw = list(s.index[(hi & ~hi.shift(1, fill_value=True)).to_numpy()])
    from valhall import event_study as es
    b1 = es.decluster(raw, s.index, DECLUSTER)
    b2 = []
    for d in b1:
        k = s.index.get_loc(d)
        after = s.iloc[k + 1:k + 1 + B2_WINDOW]
        low = after[after <= B2_LEVEL]
        if len(low):
            b2.append(low.index[0])
    return {"B1": b1, "B2": sorted(set(b2))}


def d3_rows(sector: str, rel_fwd: dict, events: dict, start=START) -> list:
    """{sector, event, date, year, excess: {h}} — relativ framåtavkastning minus seriens snitt sedan start."""
    base = {h: float(f[f.index >= pd.Timestamp(start)].dropna().mean()) if len(f.dropna()) else None
            for h, f in rel_fwd.items()}
    rows = []
    for kind, dates in events.items():
        for d in dates:
            if d < pd.Timestamp(start):
                continue
            exc = {h: float(rel_fwd[h].get(d)) - base[h] for h in rel_fwd
                   if base[h] is not None and pd.notna(rel_fwd[h].get(d))}
            rows.append({"sector": sector, "event": kind, "date": str(d.date()), "year": int(d.year), "excess": exc})
    return rows


def d3_summary(rows: list, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    out = {}
    for kind in ("B1", "B2"):
        sel = [r for r in rows if r["event"] == kind]
        res = {}
        for h in BREADTH_H:
            x = [r["excess"][h] for r in sel if h in r["excess"]]
            ys = [r["year"] for r in sel if h in r["excess"]]
            ci = cluster_ci(x, ys, rng, n_boot) if x else [None, None]
            res[h] = {"n": len(x), "years": len(set(ys)), "mean": _pct(np.mean(x)) if x else None,
                      "median": _pct(np.median(x)) if x else None, "ci": [_pct(v) for v in ci]}
        m = res[MAIN_H]
        ok = (m["n"] >= D3_MIN_N and m["years"] >= D3_MIN_YEARS and m["ci"][0] is not None and m["ci"][0] > 0
              and m["median"] is not None and m["median"] > 0)
        out[kind] = {**res, "pass": ok}
    return out


# ── Undervärderad ───────────────────────────────────────────────────────────
def cheap_years(vals: list, positive_only: bool = False) -> dict:
    """{år: billig?} — årets värde bland de 30 % lägsta i historiken fram till året (minst 5 år)."""
    pts = sorted((int(v["y"]), float(v["v"])) for v in vals or [] if v.get("y") is not None and v.get("v") is not None)
    if positive_only:
        pts = [(y, x) for y, x in pts if x > 0]
    out = {}
    for k in range(len(pts)):
        hist = [x for _y, x in pts[:k + 1]]
        if len(hist) >= VAL_MIN_YEARS:
            out[pts[k][0]] = pts[k][1] <= float(np.percentile(hist, VAL_PCT))
    return out


def undervalued_at(ev_ebit: list, pb: list, date) -> Optional[bool]:
    """Undervärderad på datumet: senast kända år (4 månaders fördröjning) billigt på EV/EBIT eller P/B."""
    d = pd.Timestamp(date)
    known = lambda y: pd.Timestamp(int(y) + 1, 1, 1) + pd.DateOffset(months=LAG_MONTHS) <= d   # noqa: E731
    flags = []
    for vals, pos in ((ev_ebit, True), (pb, False)):
        ys = {y: f for y, f in cheap_years(vals, pos).items() if known(y)}
        if ys:
            flags.append(ys[max(ys)])
    return any(flags) if flags else None


def current(name: str, h: pd.Series, eps: list) -> Optional[dict]:
    """Pågående hatperiod i dag: {namn, sedan, månader} — None när serien inte är hatad nu."""
    if not eps or not eps[-1]["censored"] or h is None or not len(h.dropna()):
        return None
    e = eps[-1]
    days = int(len(h.loc[e["start"]:]))
    return {"name": name, "since": str(e["start"].date()), "months": round(days / MONTH, 1)}


# ── Hatat och billigt (docs/forskning/hatat_och_billigt_spec.md, version 1) ─
V_MIN_N, V_MIN_YEARS = 30, 5
MID_RANKS = (10, 30)                     # rank 11–30 (0-indexerat 10..29)


def ranks(members: dict, have: set, lo: int, hi: int) -> dict:
    """{år: [id]} — plats lo..hi-1 bland dem med kursdata (0 = störst)."""
    return {y: [i for i in ids if i in have][lo:hi] for y, ids in members.items()}


def start_rows(close: pd.Series, market: pd.Series, eps: list, start=START) -> list:
    """{year, rel, undervalued} — köp dagen efter hatets start, håll HOLD dagar, relativt marknaden."""
    c = close.astype(float).dropna()
    mk = market.reindex(c.index).ffill()
    out = []
    for e in eps:
        if e["start"] < pd.Timestamp(start) or e["start"] not in c.index:
            continue
        a = c.index.get_loc(e["start"]) + 1
        b = a + HOLD
        if b >= len(c) or pd.isna(mk.iloc[a]) or pd.isna(mk.iloc[b]):
            continue
        out.append({"year": int(e["start"].year), "start": str(e["start"].date()),
                    "rel": float(c.iloc[b] / c.iloc[a] - 1) - float(mk.iloc[b] / mk.iloc[a] - 1),
                    "undervalued": e.get("undervalued")})
    return out


def _diff_ci(a: list, b: list, rng: np.random.Generator, n: int = N_BOOT) -> list:
    """95 %-intervall för snitt(a) − snitt(b) med samma dragna år; a, b = [(år, värde)]."""
    years = sorted({y for y, _v in a} | {y for y, _v in b})
    if not a or not b:
        return [None, None]

    def agg(rows):
        s = np.zeros(len(years))
        c = np.zeros(len(years))
        for y, v in rows:
            k = years.index(y)
            s[k] += v
            c[k] += 1
        return s, c
    sa, ca = agg(a)
    sb, cb = agg(b)
    pick = rng.integers(0, len(years), (n, len(years)))
    na, nb = ca[pick].sum(axis=1), cb[pick].sum(axis=1)
    ok = (na > 0) & (nb > 0)
    d = sa[pick].sum(axis=1)[ok] / na[ok] - sb[pick].sum(axis=1)[ok] / nb[ok]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [None, None]


def value_summary(rows: list, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """V1 (billigt − dyrt bland de hatade) och V2 (hatat och billigt mot marknaden) med utfall."""
    cheap = [(r["year"], r["rel"]) for r in rows if r.get("undervalued") is True]
    dear = [(r["year"], r["rel"]) for r in rows if r.get("undervalued") is False]

    def grp(x):
        v = [b for _a, b in x]
        return {"n": len(v), "years": len({a for a, _b in x}), "mean": _pct(np.mean(v)) if v else None,
                "median": _pct(np.median(v)) if v else None}
    gc, gd = grp(cheap), grp(dear)
    ci1 = _diff_ci(cheap, dear, rng, n_boot)
    v1 = (gc["n"] >= V_MIN_N and gd["n"] >= V_MIN_N and min(gc["years"], gd["years"]) >= V_MIN_YEARS
          and ci1[0] is not None and ci1[0] > 0 and gc["median"] is not None and gd["median"] is not None
          and gc["median"] > gd["median"])
    ci2 = cluster_ci([b for _a, b in cheap], [a for a, _b in cheap], rng, n_boot) if cheap else [None, None]
    v2 = (gc["n"] >= V_MIN_N and gc["years"] >= V_MIN_YEARS and ci2[0] is not None and ci2[0] > 0
          and gc["median"] is not None and gc["median"] > 0)
    return {"cheap": gc, "dear": gd, "no_value": sum(1 for r in rows if r.get("undervalued") is None),
            "V1": {"diff": _pct(gc["mean"] / 100 - gd["mean"] / 100) if gc["mean"] is not None and gd["mean"] is not None else None,
                   "ci": [_pct(v) for v in ci1], "pass": bool(v1)},
            "V2": {"ci": [_pct(v) for v in ci2], "pass": bool(v2)}}
