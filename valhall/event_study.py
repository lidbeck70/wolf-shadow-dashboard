"""
valhall/event_study.py — eventtest nivå 1: ger en bekräftad cykelvändning högre avkastning framåt än en
slumpvis dag på samma serie? Förregistrerat i docs/forskning/vandning_eventtest_spec.md (version 1) —
parametrarna här är låsta av den sidan och ändras bara med en ny version där.

Rent och utan nätverk: serierna kommer in färdiga (valhall_event_study.py hämtar dem). Cykelmotorn
(valhall.cycle) är oförändrad; vändningsdagarna är exakt de dagar turn_state kallar confirmed_since.

  Händelser    segment: tionde GRÖNA dagen efter RÖD · råvara: första bekräftade dagen i en VÄNDER-period ·
               trend: första dagen över EMA50 och SMA200 (jämförelsesignal)
  Avkastning   close[t+1+h] / close[t+1] − 1 (inträde dagen efter signalen)
  Baslinje     snittet över alla dagar i samma serie och period, samma inträde och horisont
  Test         bootstrap-intervall över händelserna och slumpdagar per serie (p-värde)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from valhall import cycle

SPEC_VERSION = 1
SPEC_PATH = "docs/forskning/vandning_eventtest_spec.md"
HORIZONS = (21, 63, 126)
MAIN_H = 63                                  # huvudhorisonten — bara den avgör
DECLUSTER = 63                               # stryk händelse inom 63 handelsdagar efter föregående (samma signal/serie)
PERIODS = (("2008–2020", "2008-01-01", "2020-12-31"), ("2021–", "2021-01-01", None))
N_BOOT, N_PERM, SEED = 5000, 2000, 20261025
MIN_EVENTS, ALPHA = 30, 0.05
TURN, TREND = "vändning", "trend"
SEGMENT, RAVARA, KOMPLEX = "segment", "råvara", "komplex"
DECIDING = (SEGMENT, RAVARA)                 # komplexen är bara beskrivande (samma drivare som råvarorna)


def _runs(s: pd.Series) -> pd.Series:
    return s.ne(s.shift()).cumsum()


def segment_events(frame: Optional[pd.DataFrame]) -> list:
    """Dagar då en GRÖN period efter RÖD når TURN_CONFIRM_DAYS (turn_state SEGMENT → confirmed_since)."""
    if frame is None or frame.empty:
        return []
    st = frame["status"]
    out, prev = [], None
    for _k, grp in st.groupby(_runs(st), sort=True):
        val = grp.iloc[0]
        if val == cycle.GRON and prev == cycle.ROD and len(grp) >= cycle.TURN_CONFIRM_DAYS:
            out.append(grp.index[cycle.TURN_CONFIRM_DAYS - 1])
        prev = val
    return out


def ravara_events(frame: Optional[pd.DataFrame]) -> list:
    """Första bekräftade dagen i varje VÄNDER-period (turn_state RAVARA → confirmed_since)."""
    if frame is None or frame.empty or "confirmed" not in frame:
        return []
    ph, conf = frame["phase"], frame["confirmed"].astype(bool)
    out = []
    for _k, grp in ph.groupby(_runs(ph), sort=True):
        if grp.iloc[0] != cycle.VANDER:
            continue
        c = conf.loc[grp.index]
        if c.any():
            out.append(c[c].index[0])
    return out


def trend_events(close: pd.Series) -> list:
    """Första dagen stängningen ligger över både EMA50 och SMA200 efter en dag då den inte gjorde det."""
    c = close.astype(float).dropna()
    sma = c.rolling(200).mean()
    ok = (c > c.ewm(span=50, adjust=False).mean()) & (c > sma) & sma.notna()
    first = ok & ~ok.shift(1, fill_value=False) & sma.shift(1).notna()
    return list(c.index[first.to_numpy()])


def decluster(dates: list, index: pd.DatetimeIndex, gap: int = DECLUSTER) -> list:
    """Stryker en händelse som kommer inom gap handelsdagar efter föregående händelse (rå, samma signal/serie)."""
    pos = sorted(int(index.get_loc(d)) for d in dates)
    keep = [p for i, p in enumerate(pos) if i == 0 or p - pos[i - 1] > gap]
    return [index[p] for p in keep]


def forward(close: pd.Series, h: int) -> pd.Series:
    """Avkastning h dagar framåt med inträde dagen efter: close[t+1+h] / close[t+1] − 1 (NaN utan hel horisont)."""
    return close.shift(-(1 + h)) / close.shift(-1) - 1


def period_of(date) -> Optional[str]:
    d = pd.Timestamp(date)
    for name, start, end in PERIODS:
        if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end)):
            return name
    return None


def _in_period(index: pd.DatetimeIndex, name: str) -> np.ndarray:
    _n, start, end = next(p for p in PERIODS if p[0] == name)
    m = np.ones(len(index), dtype=bool)
    if start:
        m &= index >= pd.Timestamp(start)
    if end:
        m &= index <= pd.Timestamp(end)
    return m


def series_rows(name: str, group: str, close: pd.Series, events: dict) -> tuple:
    """(händelserader, pooler) för en serie. events = {signal: [datum]} (orensade).
    Rad: {series, group, signal, date, period, ret: {h}, excess: {h}}.
    Pool: {(period, h): överavkastningen för alla dagar i perioden} — slumptestets dragningsunderlag."""
    c = close.astype(float).dropna()
    fwd = {h: forward(c, h) for h in HORIZONS}
    base, pools = {}, {}
    for pname, _s, _e in PERIODS:
        m = _in_period(c.index, pname)
        for h in HORIZONS:
            x = fwd[h][m].dropna()
            if len(x):
                base[(pname, h)] = float(x.mean())
                pools[(pname, h)] = (x - x.mean()).to_numpy()
    rows = []
    for signal, dates in events.items():
        for d in decluster([d for d in dates if d in c.index], c.index):
            p = period_of(d)
            if p is None:
                continue
            ret = {h: float(fwd[h].loc[d]) for h in HORIZONS if pd.notna(fwd[h].loc[d])}
            exc = {h: r - base[(p, h)] for h, r in ret.items() if (p, h) in base}
            rows.append({"series": name, "group": group, "signal": signal, "date": str(pd.Timestamp(d).date()),
                         "period": p, "ret": ret, "excess": exc})
    return rows, pools


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = N_BOOT) -> list:
    if len(x) == 0:
        return [None, None]
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_diff_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = N_BOOT) -> list:
    if len(a) == 0 or len(b) == 0:
        return [None, None]
    da = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (n, len(b)))].mean(axis=1)
    return [float(np.percentile(da - db, 2.5)), float(np.percentile(da - db, 97.5))]


def perm_p(observed: float, counts: dict, pools: dict, rng: np.random.Generator, n: int = N_PERM) -> Optional[float]:
    """Slumptestet: lika många slumpdagar per serie som händelser (utan återläggning), n omgångar.
    counts = {serie: antal händelser}, pools = {serie: överavkastning alla dagar}. p = (1 + #≥obs) / (1 + n)."""
    total = sum(counts.values())
    if not total:
        return None
    sums = np.zeros(n)
    for s, k in counts.items():
        pool = pools.get(s)
        if pool is None or not len(pool) or not k:
            continue
        if k <= len(pool):
            idx = np.stack([rng.choice(len(pool), k, replace=False) for _ in range(n)])
        else:
            idx = rng.integers(0, len(pool), (n, k))
        sums += pool[idx].sum(axis=1)
    null = sums / total
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + n))


def summarize(rows: list, pools: dict, group: str, signal: str, period: str, h: int,
              rng: np.random.Generator, n_boot: int = N_BOOT, n_perm: int = N_PERM) -> dict:
    """Sammanvägt för en grupp/signal/period/horisont: n, snitt och median, träffandel, intervall och p."""
    sel = [r for r in rows if r["group"] == group and r["signal"] == signal and r["period"] == period
           and h in r["excess"]]
    x = np.array([r["excess"][h] for r in sel], dtype=float)
    if not len(x):
        return {"n": 0, "mean_ret": None, "mean_excess": None, "median_excess": None, "hit": None,
                "ci": [None, None], "p": None}
    counts: dict = {}
    for r in sel:
        counts[r["series"]] = counts.get(r["series"], 0) + 1
    obs = float(x.mean())
    return {"n": int(len(x)), "mean_ret": round(float(np.mean([r["ret"][h] for r in sel])) * 100, 2),
            "mean_excess": round(obs * 100, 2), "median_excess": round(float(np.median(x)) * 100, 2),
            "hit": round(float((x > 0).mean()) * 100, 1),
            "ci": [None if v is None else round(v * 100, 2) for v in boot_ci(x, rng, n_boot)],
            "p": perm_p(obs, counts, {s: pools[s].get((period, h)) for s in counts}, rng, n_perm)}


def verdict(by_period: dict, need_p: bool = True) -> dict:
    """Godkänd bara om varje period har ≥ MIN_EVENTS, intervallets nedre gräns > 0 och (H1/H2) p < ALPHA."""
    reasons = []
    for pname, _s, _e in PERIODS:
        s = by_period.get(pname) or {}
        if (s.get("n") or 0) < MIN_EVENTS:
            reasons.append(f"{pname}: {s.get('n') or 0} händelser (< {MIN_EVENTS})")
        lo = (s.get("ci") or [None])[0]
        if lo is None or lo <= 0:
            reasons.append(f"{pname}: intervallets nedre gräns {lo} ≤ 0")
        if need_p and (s.get("p") is None or s["p"] >= ALPHA):
            reasons.append(f"{pname}: p = {s.get('p')} (≥ {ALPHA})")
    return {"pass": not reasons, "reasons": reasons}


def run_study(series: list, n_boot: int = N_BOOT, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """series = [{name, group, close, events: {signal: [datum]}}] → hela resultatet (grupper, hypoteser, per serie)."""
    rng = np.random.default_rng(seed)
    rows, pools = [], {}
    for s in series:
        r, p = series_rows(s["name"], s["group"], s["close"], s["events"])
        rows += r
        pools[s["name"]] = p
    groups = sorted({s["group"] for s in series}, key=lambda g: (SEGMENT, RAVARA, KOMPLEX).index(g)
                    if g in (SEGMENT, RAVARA, KOMPLEX) else 9)
    res: dict = {}
    for g in groups:
        res[g] = {sig: {p: {h: summarize(rows, pools, g, sig, p, h, rng, n_boot, n_perm) for h in HORIZONS}
                        for p, _s, _e in PERIODS} for sig in (TURN, TREND)}
    hyp = {}
    for hk, g in (("H1", SEGMENT), ("H2", RAVARA)):
        if g in res:
            hyp[hk] = {"group": g, **verdict({p: res[g][TURN][p][MAIN_H] for p, _s, _e in PERIODS})}
    for g in DECIDING:
        if g not in res:
            continue
        diff = {}
        for p, _s, _e in PERIODS:
            a = np.array([r["excess"][MAIN_H] for r in rows if r["group"] == g and r["signal"] == TURN
                          and r["period"] == p and MAIN_H in r["excess"]])
            b = np.array([r["excess"][MAIN_H] for r in rows if r["group"] == g and r["signal"] == TREND
                          and r["period"] == p and MAIN_H in r["excess"]])
            ci = boot_diff_ci(a, b, rng, n_boot)
            diff[p] = {"n": int(min(len(a), len(b))),
                       "diff": round(float(a.mean() - b.mean()) * 100, 2) if len(a) and len(b) else None,
                       "ci": [None if v is None else round(v * 100, 2) for v in ci]}
        hyp[f"H3 {g}"] = {"group": g, "by_period": diff, **verdict(diff, need_p=False)}
    per_series = []
    for s in series:
        for sig in (TURN, TREND):
            for p, _s, _e in PERIODS:
                x = [r["excess"][MAIN_H] for r in rows if r["series"] == s["name"] and r["signal"] == sig
                     and r["period"] == p and MAIN_H in r["excess"]]
                per_series.append({"series": s["name"], "group": s["group"], "signal": sig, "period": p, "n": len(x),
                                   "mean_excess": round(float(np.mean(x)) * 100, 2) if x else None})
    return {"spec_version": SPEC_VERSION, "spec": SPEC_PATH,
            "params": {"horizons": list(HORIZONS), "main_h": MAIN_H, "decluster": DECLUSTER,
                       "periods": [list(p) for p in PERIODS], "n_boot": n_boot, "n_perm": n_perm, "seed": seed,
                       "min_events": MIN_EVENTS, "alpha": ALPHA},
            "groups": res, "hypotheses": hyp, "per_series": per_series, "events": rows,
            "level2": any(hyp.get(k, {}).get("pass") for k in ("H1", "H2"))}


def report_lines(result: dict) -> list:
    """Loggens rader: hypoteserna först, sedan grupperna per signal/period/horisont och serierna."""
    lines = [f"EVENTTEST NIVÅ 1 — spec version {result['spec_version']} ({result['spec']})", "== HYPOTESER"]
    for k, v in result["hypotheses"].items():
        lines.append(f"  {k:<14} {v['group']:<8} {'GODKÄND' if v['pass'] else 'EJ GODKÄND — ingen bevisad kant'}")
        lines += [f"      {r}" for r in v["reasons"]]
        for p, d in (v.get("by_period") or {}).items():
            lines.append(f"      {p}: vändning − trend {d['diff']} %-enheter, 95 % {d['ci']}")
    lines.append(f"  Nivå 2: {'körs för godkänd grupp' if result['level2'] else 'körs inte'}")
    for g, sigs in result["groups"].items():
        lines.append(f"== {g.upper()}")
        for sig, per in sigs.items():
            for p, hs in per.items():
                for h, s in hs.items():
                    lines.append(f"  {sig:<9} {p:<10} {h:>3}d n={s['n']:>4} avk={s['mean_ret']} över={s['mean_excess']} "
                                 f"median={s['median_excess']} träff={s['hit']} 95%={s['ci']} p={s['p']}")
    lines.append(f"== PER SERIE (överavkastning {MAIN_H}d, beskrivande)")
    for r in result["per_series"]:
        if r["n"]:
            lines.append(f"  {r['group']:<8} {r['series']:<22} {r['signal']:<9} {r['period']:<10} n={r['n']:>3} "
                         f"över={r['mean_excess']}")
    return lines
