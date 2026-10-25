"""
valhall/long_study.py — långsidans cykeltest: ger ett inträde nära cykelbotten i en hatad sektor högre
avkastning över 1–3 år än en slumpvis dag? Förregistrerat i docs/forskning/lang_cykel_spec.md (version 1) —
parametrarna här är låsta av den sidan.

Rent och utan nätverk (valhall_long_study.py hämtar serierna). Återanvänder eventtestets byggstenar
(valhall.event_study) och cykelmotorn (valhall.cycle), båda oförändrade.

  E1  billig och hatad            tioårspercentil ≤ 30 och hat ≥ 60, första dagen
  E2  + liten trend upp           E1 någon dag de senaste 126 dagarna och drivarens turn (EMA20 stigande,
                                  över EMA50) slår om till sann i dag
  E3  bekräftad vändning          som nivå 1 (sektor RÖD→GRÖN 10 dagar, annars bekräftad VÄNDER)

Serierna kommer in som framåtavkastning per horisont (så att bolagskorgar och vanliga serier mäts lika):
{name, group, fwd: {h: Series}, mae: {h: Series} | None, events: {inträde: [datum]}}.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from berserk import signals as bsg
from valhall import cycle
from valhall import event_study as es

SPEC_VERSION = 1
SPEC_PATH = "docs/forskning/lang_cykel_spec.md"
HORIZONS = (252, 504, 756)
MAIN_H, LONG_H = 504, 756
DECLUSTER = 252
CHEAP_PCT, HAT_MIN, RECENT = cycle.CYKEL_TIDIG, 60.0, 126
START = "2008-01-01"
SPLITS = (("2008–2014", "2008-01-01", "2014-12-31"), ("2015–", "2015-01-01", None))
N_BOOT, SEED = 5000, 20261025
MIN_EVENTS, MIN_YEARS = 20, 5
E1, E2, E3 = "E1 billig+hatad", "E2 +liten trend", "E3 bekräftad"
ENTRIES = (E1, E2, E3)
PRODUCENT, SEKTOR, BOLAG, RAVARA = "producent", "sektor", "bolag", "råvara"
GROUPS = (PRODUCENT, SEKTOR, BOLAG, RAVARA)
DELISTED_MIN, DELISTED_DAYS = 20, 365


# ── Inträden ────────────────────────────────────────────────────────────────
def _first_true(cond: pd.Series, valid_before: pd.Series) -> list:
    cond = cond.fillna(False).astype(bool)
    first = cond & ~cond.shift(1, fill_value=False) & valid_before.shift(1, fill_value=False).astype(bool)
    return list(cond.index[first.to_numpy()])


def cheap_hated(frame: pd.DataFrame) -> pd.Series:
    """Billig och hatad per dag: tioårspercentil ≤ 30 och hatpoäng ≥ 60."""
    return (frame["pct10"] <= CHEAP_PCT) & (frame["hat"] >= HAT_MIN)


def e1_events(frame: Optional[pd.DataFrame]) -> list:
    if frame is None or frame.empty:
        return []
    return _first_true(cheap_hated(frame), frame["pct10"].notna())


def e2_events(frame: Optional[pd.DataFrame]) -> list:
    """Första dagen drivarens turn gäller medan billig och hatad gällt någon dag de senaste 126 dagarna."""
    if frame is None or frame.empty:
        return []
    turn = bsg.driver_frame(frame["close"])["turn"].reindex(frame.index).fillna(False).astype(bool)
    recent = cheap_hated(frame).astype(float).rolling(RECENT, min_periods=1).max().astype(bool)
    first = turn & ~turn.shift(1, fill_value=False) & recent & frame["pct10"].notna()
    return list(frame.index[first.to_numpy()])


def e3_events(frame: Optional[pd.DataFrame], kind: str) -> list:
    return es.segment_events(frame) if kind == cycle.SEGMENT else es.ravara_events(frame)


def entry_events(frame: Optional[pd.DataFrame], kind: str) -> dict:
    return {E1: e1_events(frame), E2: e2_events(frame), E3: e3_events(frame, kind)}


# ── Framåt ──────────────────────────────────────────────────────────────────
def mae(close: pd.Series, h: int) -> pd.Series:
    """Största nedgången från inträdet (stängningen dagen efter) inom horisonten: min(close[t+1..t+1+h]) / close[t+1] − 1."""
    lo = close[::-1].rolling(h + 1, min_periods=h + 1).min()[::-1]
    return lo.shift(-1) / close.shift(-1) - 1


def series_from_close(name: str, group: str, close: pd.Series, events: dict) -> dict:
    c = close.astype(float).dropna()
    return {"name": name, "group": group, "fwd": {h: es.forward(c, h) for h in HORIZONS},
            "mae": {h: mae(c, h) for h in HORIZONS}, "events": events}


def basket_forward(closes: dict, members_by_year: dict, h: int) -> pd.Series:
    """Likaviktad korgavkastning per dag: korgen för år Y är members_by_year[Y]. Ett bolag räknas från dagen
    efter (inträdet) och måste ha handlats då; avnoteras det under horisonten gäller sista kursen."""
    cl = {t: s.astype(float).dropna() for t, s in (closes or {}).items() if s is not None and len(s.dropna())}
    if not cl:
        return pd.Series(dtype=float)
    cal = pd.DatetimeIndex(sorted(set().union(*[s.index for s in cl.values()])))
    fwd = {}
    for t, s in cl.items():
        c = s.reindex(cal).ffill()
        listed = pd.Series(cal <= s.index[-1], index=cal) & pd.Series(cal >= s.index[0], index=cal)
        r = c.shift(-(1 + h)) / c.shift(-1) - 1
        has_end = pd.Series(np.arange(len(cal)) + 1 + h < len(cal), index=cal)
        fwd[t] = r.where(listed.shift(-1, fill_value=False) & has_end)
    years = pd.Series(cal.year, index=cal)
    out = pd.Series(np.nan, index=cal)
    for y, members in (members_by_year or {}).items():
        cols = [fwd[t] for t in members if t in fwd]
        if not cols:
            continue
        m = years == int(y)
        out[m] = pd.concat(cols, axis=1)[m].mean(axis=1)
    return out


# ── Statistik ───────────────────────────────────────────────────────────────
def _period_mask(index: pd.DatetimeIndex, start, end) -> np.ndarray:
    m = np.ones(len(index), dtype=bool)
    if start:
        m &= index >= pd.Timestamp(start)
    if end:
        m &= index <= pd.Timestamp(end)
    return m


def _on_or_before(idx: pd.DatetimeIndex, d):
    """Seriens sista handelsdag på eller före d (bolagskorgen handlas i Norden, signalen kommer från USA) —
    inträdet blir då dagen efter, alltid efter signalen."""
    if d in idx:
        return d
    k = int(idx.searchsorted(pd.Timestamp(d), side="right")) - 1
    return idx[k] if k >= 0 else None


def event_rows(s: dict, start=START, end=None) -> list:
    """{series, group, entry, date, year, ret: {h}, excess: {h}, mae: {h}, base50/100} per rensad händelse."""
    base = {}
    for h in HORIZONS:
        f = s["fwd"][h]
        x = f[_period_mask(f.index, start, end)].dropna()
        base[h] = (float(x.mean()), float((x > 0.5).mean()), float((x > 1.0).mean())) if len(x) else None
    rows = []
    for entry, dates in s["events"].items():
        idx = s["fwd"][HORIZONS[0]].index
        dd = sorted({_on_or_before(idx, d) for d in dates} - {None})
        for d in es.decluster(dd, idx, DECLUSTER):
            if not _period_mask(pd.DatetimeIndex([d]), start, end)[0]:
                continue
            ret, exc, dd_ = {}, {}, {}
            for h in HORIZONS:
                v = s["fwd"][h].get(d)
                if v is not None and pd.notna(v) and base[h] is not None:
                    ret[h], exc[h] = float(v), float(v) - base[h][0]
                    m = (s.get("mae") or {}).get(h)
                    if m is not None and pd.notna(m.get(d)):
                        dd_[h] = float(m.get(d))
            rows.append({"series": s["name"], "group": s["group"], "entry": entry, "date": str(pd.Timestamp(d).date()),
                         "year": int(pd.Timestamp(d).year), "ret": ret, "excess": exc, "mae": dd_,
                         "base": {h: base[h] for h in HORIZONS if base[h] is not None}})
    return rows


def cluster_boot(rows: list, h: int, rng: np.random.Generator, n: int = N_BOOT) -> list:
    """95 %-intervall för snitt-överavkastningen: drar hela kalenderår (med återläggning)."""
    by_year: dict = {}
    for r in rows:
        if h in r["excess"]:
            by_year.setdefault(r["year"], []).append(r["excess"][h])
    years = list(by_year)
    if not years:
        return [None, None]
    sums = np.array([sum(by_year[y]) for y in years])
    cnts = np.array([len(by_year[y]) for y in years])
    pick = rng.integers(0, len(years), (n, len(years)))
    means = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def cluster_boot_diff(a: list, b: list, h: int, rng: np.random.Generator, n: int = N_BOOT) -> list:
    """95 %-intervall för skillnaden i snitt-överavkastning a − b, samma dragna år för båda."""
    def agg(rows):
        d: dict = {}
        for r in rows:
            if h in r["excess"]:
                s, c = d.get(r["year"], (0.0, 0))
                d[r["year"]] = (s + r["excess"][h], c + 1)
        return d
    da, db = agg(a), agg(b)
    years = sorted(set(da) | set(db))
    if not da or not db:
        return [None, None]
    sa = np.array([da.get(y, (0.0, 0))[0] for y in years])
    ca = np.array([da.get(y, (0.0, 0))[1] for y in years])
    sb = np.array([db.get(y, (0.0, 0))[0] for y in years])
    cb = np.array([db.get(y, (0.0, 0))[1] for y in years])
    pick = rng.integers(0, len(years), (n, len(years)))
    na, nb = ca[pick].sum(axis=1), cb[pick].sum(axis=1)
    ok = (na > 0) & (nb > 0)
    diff = sa[pick].sum(axis=1)[ok] / na[ok] - sb[pick].sum(axis=1)[ok] / nb[ok]
    if not len(diff):
        return [None, None]
    return [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]


def _pct(v) -> Optional[float]:
    return None if v is None else round(float(v) * 100, 2)


def summarize(rows: list, h: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    sel = [r for r in rows if h in r["excess"]]
    if not sel:
        return {"n": 0, "years": 0}
    x = np.array([r["excess"][h] for r in sel])
    ret = np.array([r["ret"][h] for r in sel])
    dd = [r["mae"][h] for r in sel if h in r["mae"]]
    b50 = np.mean([r["base"][h][1] for r in sel])
    b100 = np.mean([r["base"][h][2] for r in sel])
    return {"n": len(sel), "years": len({r["year"] for r in sel}), "mean_ret": _pct(ret.mean()),
            "median_ret": _pct(np.median(ret)), "mean_excess": _pct(x.mean()), "median_excess": _pct(np.median(x)),
            "hit": round(float((x > 0).mean()) * 100, 1),
            "ci": [_pct(v) for v in cluster_boot(sel, h, rng, n_boot)],
            "over50": round(float((ret > 0.5).mean()) * 100, 1), "over50_base": round(float(b50) * 100, 1),
            "over100": round(float((ret > 1.0).mean()) * 100, 1), "over100_base": round(float(b100) * 100, 1),
            "median_mae": _pct(np.median(dd)) if dd else None}


def verdict(main: dict, long: dict) -> dict:
    """L1–L3: ≥ 20 händelser från ≥ 5 år, intervallets nedre gräns > 0, median > 0 (504 d) och snitt > 0 (756 d)."""
    reasons = []
    if (main.get("n") or 0) < MIN_EVENTS:
        reasons.append(f"{main.get('n') or 0} händelser (< {MIN_EVENTS})")
    if (main.get("years") or 0) < MIN_YEARS:
        reasons.append(f"{main.get('years') or 0} kalenderår (< {MIN_YEARS})")
    lo = (main.get("ci") or [None])[0]
    if lo is None or lo <= 0:
        reasons.append(f"intervallets nedre gräns {lo} ≤ 0")
    if main.get("median_excess") is None or main["median_excess"] <= 0:
        reasons.append(f"median {main.get('median_excess')} ≤ 0")
    if long.get("mean_excess") is None or long["mean_excess"] <= 0:
        reasons.append(f"snitt på {LONG_H} dagar {long.get('mean_excess')} ≤ 0")
    return {"pass": not reasons, "reasons": reasons}


def diff_verdict(a: list, b: list, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    xa = [r["excess"][MAIN_H] for r in a if MAIN_H in r["excess"]]
    xb = [r["excess"][MAIN_H] for r in b if MAIN_H in r["excess"]]
    ci = cluster_boot_diff(a, b, MAIN_H, rng, n_boot)
    reasons = []
    if min(len(xa), len(xb)) < MIN_EVENTS:
        reasons.append(f"för få händelser ({len(xa)} / {len(xb)})")
    if ci[0] is None or ci[0] <= 0:
        reasons.append(f"intervallets nedre gräns {_pct(ci[0])} ≤ 0")
    return {"diff": _pct(np.mean(xa) - np.mean(xb)) if xa and xb else None, "ci": [_pct(v) for v in ci],
            "pass": not reasons, "reasons": reasons}


def run_study(series: list, deciding: tuple = (PRODUCENT, SEKTOR), n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Hela testet. deciding = de avgörande grupperna (bolag bara om avnoteringskontrollen godkänts)."""
    rng = np.random.default_rng(seed)
    rows = [r for s in series for r in event_rows(s)]
    split_rows = {p: [r for s in series for r in event_rows(s, a, b)] for p, a, b in SPLITS}
    groups = [g for g in GROUPS if any(s["group"] == g for s in series)]
    res: dict = {}
    for g in groups:
        res[g] = {}
        for e in ENTRIES:
            sel = [r for r in rows if r["group"] == g and r["entry"] == e]
            res[g][e] = {"all": {h: summarize(sel, h, rng, n_boot) for h in HORIZONS},
                         "split": {p: summarize([r for r in split_rows[p] if r["group"] == g and r["entry"] == e],
                                                MAIN_H, rng, n_boot) for p, _a, _b in SPLITS}}
    hyp: dict = {}
    for g in groups:
        if g not in deciding:
            continue
        for k, e in (("L1", E1), ("L2", E2), ("L3", E3)):
            hyp[f"{k} {g}"] = {"group": g, "entry": e, **verdict(res[g][e]["all"][MAIN_H], res[g][e]["all"][LONG_H])}
        sel = {e: [r for r in rows if r["group"] == g and r["entry"] == e] for e in ENTRIES}
        hyp[f"L4 {g}"] = {"group": g, "entry": f"{E2} − {E1}", **diff_verdict(sel[E2], sel[E1], rng, n_boot)}
        hyp[f"L5 {g}"] = {"group": g, "entry": f"{E2} − {E3}", **diff_verdict(sel[E2], sel[E3], rng, n_boot)}
    strong = [e for e in ENTRIES
              if sum(1 for k, v in hyp.items() if k[:2] in ("L1", "L2", "L3") and v["entry"] == e and v["pass"]) >= 2]
    per_series = []
    for s in series:
        for e in ENTRIES:
            x = [r["excess"][MAIN_H] for r in rows if r["series"] == s["name"] and r["entry"] == e and MAIN_H in r["excess"]]
            per_series.append({"series": s["name"], "group": s["group"], "entry": e, "n": len(x),
                               "mean_excess": _pct(np.mean(x)) if x else None})
    return {"spec_version": SPEC_VERSION, "spec": SPEC_PATH, "deciding": list(deciding),
            "params": {"horizons": list(HORIZONS), "main_h": MAIN_H, "decluster": DECLUSTER, "start": START,
                       "cheap_pct": CHEAP_PCT, "hat_min": HAT_MIN, "recent": RECENT, "n_boot": n_boot, "seed": seed,
                       "min_events": MIN_EVENTS, "min_years": MIN_YEARS},
            "groups": res, "hypotheses": hyp, "strong": strong, "per_series": per_series, "events": rows}


def delisted_check(nordic_ids: set, last_dates: dict, today) -> dict:
    """Avnoteringskontrollen: hur många nordiska instrument i listan har senaste kurs äldre än 365 dagar."""
    cut = pd.Timestamp(today) - pd.Timedelta(days=DELISTED_DAYS)
    old = [i for i in nordic_ids if last_dates.get(i) is not None and pd.Timestamp(last_dates[i]) < cut]
    return {"nordic": len(nordic_ids), "old": len(old), "covered": len(old) >= DELISTED_MIN}


def report_lines(result: dict) -> list:
    lines = [f"LÅNGSIDANS CYKELTEST — spec version {result['spec_version']} ({result['spec']})",
             f"Avgörande grupper: {', '.join(result['deciding'])}", "== HYPOTESER"]
    for k, v in result["hypotheses"].items():
        extra = f" skillnad {v['diff']} %-enheter, 95 % {v['ci']}" if "diff" in v else ""
        lines.append(f"  {k:<14} {v['entry']:<34} {'GODKÄND' if v['pass'] else 'EJ GODKÄND'}{extra}")
        lines += [f"      {r}" for r in v["reasons"]]
    lines.append(f"  Starkt (godkänt i ≥ 2 grupper): {', '.join(result['strong']) or 'inget'}")
    for g, entries in result["groups"].items():
        lines.append(f"== {g.upper()}")
        for e, d in entries.items():
            for h, s in d["all"].items():
                if not s.get("n"):
                    lines.append(f"  {e:<18} {h:>3}d n=0")
                    continue
                lines.append(f"  {e:<18} {h:>3}d n={s['n']:>3} år={s['years']:>2} avk={s['mean_ret']} median={s['median_ret']} "
                             f"över={s['mean_excess']} medianöver={s['median_excess']} 95%={s['ci']} träff={s['hit']} "
                             f">50%={s['over50']}/{s['over50_base']} >100%={s['over100']}/{s['over100_base']} "
                             f"djupast={s['median_mae']}")
            for p, s in d["split"].items():
                if s.get("n"):
                    lines.append(f"  {e:<18} {p:<10} {MAIN_H}d n={s['n']:>3} över={s['mean_excess']} 95%={s['ci']}")
    lines.append(f"== PER SERIE (överavkastning {MAIN_H}d, beskrivande)")
    for r in result["per_series"]:
        if r["n"]:
            lines.append(f"  {r['group']:<9} {r['series']:<22} {r['entry']:<18} n={r['n']:>2} över={r['mean_excess']}")
    return lines
