"""
odin/rotation_backtest.py — historiskt test av sektorrotationen (docs/forskning/sektorrotation_spec.md, version 1).

Signal på månadens sista stängning (odin.rotation_engine), handel på stängningen nästa handelsdag, kostnad per
sida av omsättningen. Strategin "Rotation topp 3" jämförs med SPY, likaviktade sektorer, ren 12-månaders
momentum och rent trendfilter; utfallet prövas mot de låsta kriterierna R1, R2, R3 och S. Rent, utan nätverk.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from odin import rotation_engine as re_

COST_SIDE = 0.0045                 # 45 bp per sida (backtest_costs: utland 90 bp tur och retur)
TOP_N = 3
MIN_FUNDS = 8
SPLIT = "2015-01-01"
PERIODS = (("2000–2014", None, "2014-12-31"), ("2015–", SPLIT, None))
N_BOOT, BLOCK, SEED = 5000, 12, 20261025
ROTATION, SPY, EQUAL, MOM12, TREND = "Rotation topp 3", "SPY", "Likaviktade sektorer", "Ren 12-mån momentum", "Rent trendfilter"


# ── Vikter per ombalansering ────────────────────────────────────────────────
def w_rotation(sig: dict, i: int, n: int = TOP_N) -> dict:
    ok = sig["eligible"].iloc[i] & sig["f1"].iloc[i] & sig["f2"].iloc[i] & sig["f3"].iloc[i]
    s = sig["S"].iloc[i][ok[ok].index].dropna().sort_values(ascending=False)
    return {t: 1.0 / n for t in s.index[:n]}


def w_equal(sig: dict, i: int) -> dict:
    el = sig["eligible"].iloc[i]
    names = list(el[el].index)
    return {t: 1.0 / len(names) for t in names} if names else {}


def w_mom12(sig: dict, i: int, n: int = TOP_N) -> dict:
    el = sig["eligible"].iloc[i]
    s = sig["rs"][12].iloc[i][el[el].index].dropna().sort_values(ascending=False)
    return {t: 1.0 / n for t in s.index[:n]}


def w_trend(sig: dict, i: int) -> dict:
    el = sig["eligible"].iloc[i]
    names = list(el[el].index)
    up = [t for t in names if bool(sig["f1"].iloc[i][t])]
    return {t: 1.0 / len(names) for t in up} if names else {}


# ── Simuleringen ────────────────────────────────────────────────────────────
def exec_days(sig: dict, calendar: pd.DatetimeIndex) -> list:
    """[(månadsrad i, handelsdag)] — handel på första handelsdagen efter månadens sista stängning."""
    out = []
    for i, me in enumerate(sig["S"].index):
        k = int(calendar.searchsorted(me, side="right"))
        if k < len(calendar):
            out.append((i, calendar[k]))
    return out


def start_index(sig: dict) -> int:
    """Första månaden med minst MIN_FUNDS fonder som har hela underlaget."""
    n = sig["eligible"].sum(axis=1)
    ok = np.flatnonzero(n.to_numpy() >= MIN_FUNDS)
    return int(ok[0]) if len(ok) else len(n)


def simulate(daily: pd.DataFrame, sig: dict, weight_fn: Callable, cost: float = COST_SIDE,
             calendar: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """En rad per innehavsperiod: start, slut, brutto, netto, omsättning och investerad andel."""
    cal = calendar if calendar is not None else daily.index
    days = [x for x in exec_days(sig, cal) if x[0] >= start_index(sig)]
    rows, drift = [], {}
    for (i, d0), (_j, d1) in zip(days[:-1], days[1:]):
        w = weight_fn(sig, i)
        names = set(w) | set(drift)
        turnover = sum(abs(w.get(t, 0.0) - drift.get(t, 0.0)) for t in names)
        r = {t: float(daily[t].loc[d1] / daily[t].loc[d0] - 1) for t in w}
        gross = sum(w[t] * r[t] for t in w)
        net = (1 - turnover * cost) * (1 + gross) - 1
        tot = 1 + gross
        drift = {t: w[t] * (1 + r[t]) / tot for t in w} if tot > 0 else {}
        rows.append({"start": d0, "end": d1, "gross": gross, "net": net, "turnover": turnover,
                     "invested": sum(w.values())})
    return pd.DataFrame(rows)


def buy_hold(series: pd.Series, periods: pd.DataFrame, cost: float = COST_SIDE) -> pd.DataFrame:
    """Köp och behåll över samma perioder (en köpkostnad i början)."""
    out = periods[["start", "end"]].copy()
    r = [float(series.loc[e] / series.loc[s] - 1) for s, e in zip(out["start"], out["end"])]
    out["gross"] = r
    out["net"] = r
    if len(out):
        out.loc[out.index[0], "net"] = (1 - cost) * (1 + r[0]) - 1
    out["turnover"] = 0.0
    out["invested"] = 1.0
    return out


# ── Mått ────────────────────────────────────────────────────────────────────
def stats(p: pd.DataFrame) -> dict:
    if p is None or not len(p):
        return {"n": 0}
    r = p["net"].to_numpy()
    years = (pd.Timestamp(p["end"].iloc[-1]) - pd.Timestamp(p["start"].iloc[0])).days / 365.25
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    vol = float(np.std(r, ddof=1) * np.sqrt(12)) if len(r) > 1 else None
    return {"n": int(len(r)), "years": round(years, 1),
            "cagr": round((float(eq[-1]) ** (1 / years) - 1) * 100, 2) if years > 0 else None,
            "max_dd": round(float(np.max(1 - eq / peak)) * 100, 1),
            "vol": round(vol * 100, 1) if vol else None,
            "sharpe": round(float(np.mean(r) * 12) / vol, 2) if vol else None,
            "turnover_yr": round(float(p["turnover"].sum()) / years, 2) if years > 0 else None,
            "invested": round(float(p["invested"].mean()) * 100, 1)}


def split(p: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    m = pd.Series(True, index=p.index)
    if start:
        m &= p["start"] >= pd.Timestamp(start)
    if end:
        m &= p["start"] <= pd.Timestamp(end)
    return p[m]


def block_boot(active: np.ndarray, rng: np.random.Generator, n: int = N_BOOT, block: int = BLOCK) -> list:
    """95 %-intervall för årssnittet av aktiv avkastning: cirkulär blockbootstrap (block om 12 månader)."""
    T = len(active)
    if T < block:
        return [None, None]
    k = int(np.ceil(T / block))
    starts = rng.integers(0, T, (n, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T] % T
    means = active[idx].mean(axis=1) * 12
    return [round(float(np.percentile(means, 2.5)) * 100, 2), round(float(np.percentile(means, 97.5)) * 100, 2)]


def verdict(by: dict, ci: list) -> dict:
    """R1–R3 och S ur måtten per strategi och period. by = {period: {strategi: stats}} med "Hela" för hela tiden."""
    def g(p, s, k):
        return ((by.get(p) or {}).get(s) or {}).get(k)
    periods = [p for p, _a, _b in PERIODS]
    r1 = all(g(p, ROTATION, "cagr") is not None and g(p, ROTATION, "cagr") > max(g(p, SPY, "cagr") or -1e9, g(p, EQUAL, "cagr") or -1e9)
             for p in periods)
    r2 = all(g(p, ROTATION, "sharpe") is not None and g(p, ROTATION, "sharpe") > (g(p, SPY, "sharpe") or -1e9) for p in periods) \
        and (g("Hela", ROTATION, "max_dd") or 1e9) <= (g("Hela", SPY, "max_dd") or -1e9)
    last = periods[-1]
    r3 = (g(last, ROTATION, "cagr") or -1e9) >= max(g(last, MOM12, "cagr") or -1e9, g(last, TREND, "cagr") or -1e9)
    s = ci[0] is not None and ci[0] > 0
    if r1 and r2:
        use = "sektorgrind för swing (fas 2 = handla, fas 4 = inte) — ägarens beslut"
    elif r2:
        use = "riskfilter (minska vid fas 4), inte avkastningsmotor — ägarens beslut"
    else:
        use = "information i fliken, märkt ej validerad — ingen grind"
    return {"R1": r1, "R2": r2, "R3": r3, "S": s, "use": use}


def run_backtest(daily: pd.DataFrame, spy: pd.Series, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Hela testet: huvudstrategin, jämförelserna, robustheten och utfallet."""
    daily = daily.sort_index()
    spy = spy.reindex(daily.index).ffill()
    sig = re_.compute(daily, spy)
    cal = daily.index
    runs = {ROTATION: simulate(daily, sig, w_rotation, calendar=cal),
            EQUAL: simulate(daily, sig, w_equal, calendar=cal),
            MOM12: simulate(daily, sig, w_mom12, calendar=cal),
            TREND: simulate(daily, sig, w_trend, calendar=cal)}
    runs[SPY] = buy_hold(spy, runs[ROTATION])
    by = {"Hela": {k: stats(v) for k, v in runs.items()}}
    for p, a, b in PERIODS:
        by[p] = {k: stats(split(v, a, b)) for k, v in runs.items()}
    active = (runs[ROTATION]["net"].to_numpy() - runs[SPY]["net"].to_numpy())
    ci = block_boot(active, np.random.default_rng(seed), n_boot)
    robust = {}
    variants = {"Lika vikter (1/3)": dict(weights={12: 1 / 3, 6: 1 / 3, 3: 1 / 3}), "12–1 (utan senaste månaden)": dict(skip=1),
                "200-dagars snitt": dict(trend="200d")}
    for name, kw in variants.items():
        s2 = re_.compute(daily, spy, **kw)
        robust[name] = stats(simulate(daily, s2, w_rotation, calendar=cal))
    for n in (2, 4):
        robust[f"Topp {n}"] = stats(simulate(daily, sig, lambda s, i, n=n: w_rotation(s, i, n), calendar=cal))
    v = verdict(by, ci)
    first = runs[ROTATION]["start"].iloc[0] if len(runs[ROTATION]) else None
    return {"spec_version": re_.SPEC_VERSION, "spec": re_.SPEC_PATH, "start": str(pd.Timestamp(first).date()) if first is not None else None,
            "params": {"cost_side": COST_SIDE, "top_n": TOP_N, "split": SPLIT, "n_boot": n_boot, "block": BLOCK, "seed": seed},
            "by_period": by, "active_ci": ci, "active_mean": round(float(np.mean(active)) * 12 * 100, 2) if len(active) else None,
            "robust": robust, "verdict": v, "latest": re_.table(sig),
            "climb_1m": re_.climbers(sig, 1), "climb_3m": re_.climbers(sig, 3)}


def report_lines(res: dict) -> list:
    lines = [f"SEKTORROTATION — spec version {res['spec_version']} ({res['spec']}) · från {res['start']}"]
    for p, rows in res["by_period"].items():
        lines.append(f"== {p}")
        for k, s in rows.items():
            if s.get("n"):
                lines.append(f"  {k:<22} CAGR {s['cagr']:>6} % · max DD {s['max_dd']:>5} % · vol {s['vol']} % · Sharpe {s['sharpe']} "
                             f"· omsättning/år {s['turnover_yr']} · investerad {s['invested']} %")
    lines.append(f"== AKTIV AVKASTNING MOT SPY: {res['active_mean']} %/år · 95 % {res['active_ci']} (blockbootstrap 12 mån)")
    lines.append("== ROBUSTHET (beskrivande, hela tiden)")
    for k, s in res["robust"].items():
        if s.get("n"):
            lines.append(f"  {k:<28} CAGR {s['cagr']} % · max DD {s['max_dd']} % · Sharpe {s['sharpe']}")
    v = res["verdict"]
    lines.append("== UTFALL")
    for k, text in (("R1", "slår SPY och likaviktat i båda perioderna"), ("R2", "Sharpe över SPY i båda, DD högst SPY:s"),
                    ("R3", "minst som ren momentum och rent trendfilter 2015–"), ("S", "aktiv avkastning statistiskt > 0")):
        lines.append(f"  {k} {'GODKÄND' if v[k] else 'EJ GODKÄND'} — {text}")
    lines.append(f"  Användning: {v['use']}")
    lines.append("== SENASTE MÅNADEN")
    for r in res["latest"]:
        lines.append(f"  {r['ticker']:<5} S {r['S']:>5} · R12 {r['R12']} R6 {r['R6']} R3 {r['R3']} · filter "
                     f"{'✓' if r['f1'] else '✗'}{'✓' if r['f2'] else '✗'}{'✓' if r['f3'] else '✗'} · {r['phase']}")
    return lines
