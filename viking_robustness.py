"""
viking_robustness.py — försök fälla Viking Nine-backtestet (PR A, inga regeländringar).

  cost_table      courtage + spread (baspunkter tur och retur) och extra slippage på
                  stoppar: när blir expectancy och portföljen negativ?
  monte_carlo     10 000 simuleringar på portföljens FAKTISKA affärer: bootstrap,
                  block-bootstrap (behåller kluster), slumpad ordning och bootstrap
                  med slumpad extra kostnad — CAGR- och drawdown-percentiler,
                  P(förlust), P(DD > 10/15/20 %)
  expectancy_ci   bootstrap-konfidensintervall (95 %) för expectancy i R
  concentration   hur mycket av vinsten de fem bästa affärerna står för
  breakdown       var kanten kommer ifrån: år, exittyp, sektor, aktie och mått på
                  signaldagen (ATR %, relativ volym, RSI, CLV, övre veke, RS63, gap,
                  avstånd till EMA20) samt innehavstid
  benchmark       köp och behåll indexet samma period: CAGR, max DD, CAGR/DD

Monte Carlo räknar drawdown på stängda affärer i följd — den dagsvärderade
drawdownen i portföljläget är högre (öppna förluster), så percentilerna är en
undre gräns för risken.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

import viking_backtest as vb
import viking_portfolio as vp

COST_LEVELS_BPS = (0, 10, 20, 30, 50)
GAP_SLIPPAGE_BPS = 25              # extra slippage på stoppar (gap och snabba fall)
STOP_REASONS = ("stopp", "breakeven-stopp", "nödstopp 2 ATR", "ATR-steg")
MC_SIMS = 10_000
MC_BLOCK = 5                       # affärer per block i block-bootstrap
MC_SLIP_MAX_BPS = 30               # slumpad extra kostnad 0–30 bp per affär
DD_LEVELS = (10, 15, 20)
RUIN_DD = 50                       # "ruin" = drawdown över 50 %
CI_SIMS = 5_000
TOP_N = 5

BUCKETS = {
    "atr_pct": ("ATR % av kursen", (1.5, 2.5, 3.5), "%"),
    "rvol": ("Relativ volym", (1.5, 2.0, 3.0), "×"),
    "rsi": ("RSI14", (60.0, 70.0), ""),
    "clv": ("Stängning i dagens spann (CLV)", (0.6, 0.8), ""),
    "upper_wick": ("Övre veke / spann", (0.1, 0.3), ""),
    "rs63": ("RS63 mot index (procentenheter)", (0.0, 5.0, 15.0), ""),
    "gap_atr": ("Gap vid entry (× ATR)", (0.0, 0.25, 0.5), ""),
    "dist_ema20_atr": ("Avstånd till EMA20 (× ATR)", (1.0, 2.0, 3.0), ""),
}
DAY_BUCKETS = (2, 5, 10, 20)


# ── Kostnader ───────────────────────────────────────────────────────────────
def with_costs(trades: list, bps: float, gap_bps: float = 0.0) -> list:
    """Kopior av affärerna med R efter kostnad: bps tur och retur på positionen,
    plus gap_bps på stoppar. Kostnaden i R = kostnad i % / stoppavstånd i %."""
    out = []
    for t in trades:
        if t.r is None or not t.entry or not t.risk:
            out.append(t)
            continue
        cost = bps + (gap_bps if t.exit_reason in STOP_REASONS else 0.0)
        cost_r = (cost / 10_000) / (t.risk / t.entry)
        out.append(dataclasses.replace(t, r=round(t.r - cost_r, 3),
                                       exit=round(t.exit - cost / 10_000 * t.entry, 4) if t.exit is not None
                                       else t.exit))
    return out


def cost_table(trades: list, years: Optional[float] = None, pc: vp.PortfolioConfig = vp.PortfolioConfig(),
               levels=COST_LEVELS_BPS, gap_bps: float = GAP_SLIPPAGE_BPS) -> list:
    """En rad per kostnadsnivå: expectancy, profit factor, summa R och portföljen."""
    rows = []
    for bps in levels:
        for gap in ((0.0, gap_bps) if bps else (0.0,)):
            ts = with_costs(trades, bps, gap)
            m = vb.metrics(ts)
            p = vp.simulate(ts, years=years, pc=pc)
            rows.append({"Kostnad bp": bps, "Gap-slippage bp": gap, "Expectancy R": m.get("expectancy"),
                         "Profit factor": m.get("profit_factor"), "Summa R": m.get("total_r"),
                         "Avkastning %": p["return_pct"], "CAGR %": p["cagr_pct"], "Max DD %": p["max_dd_pct"]})
    return rows


def breakeven_cost_bps(trades: list) -> Optional[float]:
    """Kostnad (bp tur och retur) där expectancy blir noll — linjärt, utan gap-slippage."""
    closed = [t for t in trades if not t.open and t.r is not None and t.entry and t.risk]
    if not closed:
        return None
    exp = float(np.mean([t.r for t in closed]))
    per_bp = float(np.mean([(1 / 10_000) / (t.risk / t.entry) for t in closed]))
    return round(exp / per_bp, 0) if per_bp > 0 and exp > 0 else 0.0


# ── Monte Carlo ─────────────────────────────────────────────────────────────
def _paths(sims: np.ndarray, years: Optional[float]) -> dict:
    eq = np.cumprod(1 + sims / 100, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(eq), 1)), eq], axis=1), axis=1)[:, 1:]
    dd = np.max(1 - eq / peak, axis=1) * 100
    final = eq[:, -1]
    out = {"Avkastning p50 %": round(float(np.percentile((final - 1) * 100, 50)), 1)}
    if years and years > 0:
        cagr = (np.clip(final, 1e-9, None) ** (1 / years) - 1) * 100
        out.update({f"CAGR p{q} %": round(float(np.percentile(cagr, q)), 1) for q in (5, 50, 95)})
    out.update({f"DD p{q} %": round(float(np.percentile(dd, q)), 1) for q in (50, 75, 90, 95, 99)})
    out["P(förlust) %"] = round(float(np.mean(final < 1)) * 100, 1)
    out.update({f"P(DD>{lvl}%) %": round(float(np.mean(dd > lvl)) * 100, 1) for lvl in DD_LEVELS})
    out[f"P(ruin, DD>{RUIN_DD}%) %"] = round(float(np.mean(dd > RUIN_DD)) * 100, 2)
    return out


def monte_carlo(rows: list, years: Optional[float] = None, sims: int = MC_SIMS, seed: int = 7,
                block: int = MC_BLOCK, slip_max_bps: float = MC_SLIP_MAX_BPS) -> list:
    """rows = portföljens rader (viking_portfolio.simulate()["rows"]) — avkastning i % av kontot per
    affär, i exitordning. En rad per metod."""
    closed = sorted((r for r in rows if r.get("return_pct") is not None),
                    key=lambda r: (r["trade"].exit_date, r["trade"].ticker))
    if len(closed) < 5:
        return []
    ret = np.array([r["return_pct"] for r in closed], dtype=float)
    pos = np.array([r["position_pct"] for r in closed], dtype=float)
    n = len(ret)
    rng = np.random.default_rng(seed)
    out = [{"Metod": "Historiskt (faktisk ordning)", **_paths(ret[None, :], years)}]
    idx = rng.integers(0, n, size=(sims, n))
    out.append({"Metod": "Bootstrap (slumpat urval)", **_paths(ret[idx], years)})
    b = max(1, min(block, n))
    starts = rng.integers(0, n - b + 1, size=(sims, math.ceil(n / b)))
    bidx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(sims, -1)[:, :n]
    out.append({"Metod": f"Block-bootstrap ({b} affärer)", **_paths(ret[bidx], years)})
    perm = np.argsort(rng.random((sims, n)), axis=1)
    out.append({"Metod": "Slumpad ordning (samma affärer)", **_paths(ret[perm], years)})
    slip = rng.uniform(0, slip_max_bps, size=(sims, n)) / 10_000 * pos[idx]      # % av kontot
    out.append({"Metod": f"Bootstrap + kostnad 0–{slip_max_bps:g} bp", **_paths(ret[idx] - slip, years)})
    return out


# ── Statistisk säkerhet och koncentration ───────────────────────────────────
def expectancy_ci(trades: list, sims: int = CI_SIMS, seed: int = 11) -> Optional[dict]:
    rs = np.array([t.r for t in trades if not t.open and t.r is not None], dtype=float)
    if len(rs) < 5:
        return None
    rng = np.random.default_rng(seed)
    boot = rs[rng.integers(0, len(rs), size=(sims, len(rs)))].mean(axis=1)
    return {"mean": round(float(rs.mean()), 3), "low": round(float(np.percentile(boot, 2.5)), 3),
            "high": round(float(np.percentile(boot, 97.5)), 3), "n": int(len(rs)),
            "p_le_zero": round(float(np.mean(boot <= 0)) * 100, 1),
            "se": round(float(rs.std(ddof=1) / math.sqrt(len(rs))), 3)}


def concentration(trades: list, top: int = TOP_N) -> Optional[dict]:
    closed = sorted((t for t in trades if not t.open and t.r is not None), key=lambda t: -t.r)
    if not closed:
        return None
    total = sum(t.r for t in closed)
    best = closed[:top]
    gross = sum(t.r for t in closed if t.r > 0)
    rest = closed[top:]
    return {"total_r": round(total, 2), "top_r": round(sum(t.r for t in best), 2),
            "top_share_of_gross": round(sum(t.r for t in best) / gross * 100, 1) if gross > 0 else None,
            "total_without_top": round(total - sum(t.r for t in best), 2),
            "expectancy_without_top": round(sum(t.r for t in rest) / len(rest), 3) if rest else None,
            "top": [(t.ticker, t.entry_date, t.r) for t in best]}


# ── Var kommer kanten ifrån? ────────────────────────────────────────────────
def _bucket(v, edges, unit="") -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    lo = None
    for e in edges:
        if v < e:
            return f"< {e:g}{unit}" if lo is None else f"{lo:g}–{e:g}{unit}"
        lo = e
    return f"≥ {edges[-1]:g}{unit}"


def group(trades: list, key: Callable) -> list:
    """Nyckeltal i R per grupp (stängda affärer)."""
    groups = {}
    for t in trades:
        if t.open or t.r is None:
            continue
        g = key(t)
        if g is not None:
            groups.setdefault(g, []).append(t)
    rows = []
    for g, ts in groups.items():
        m = vb.metrics(ts)
        rows.append({"Grupp": g, "Affärer": m["trades"], "Win rate %": m["win_rate"],
                     "Expectancy R": m["expectancy"], "Summa R": m["total_r"],
                     "Snittvinnare R": m["avg_winner"], "Stoppade %": round(
                         sum(1 for t in ts if t.exit_reason == "stopp") / len(ts) * 100, 0)})
    return rows


def _ordered(rows: list, edges, unit="") -> list:
    labels = [_bucket(e - 1e-9, edges, unit) for e in edges] + [_bucket(edges[-1], edges, unit)]
    order = {lbl: k for k, lbl in enumerate(labels)}
    return sorted(rows, key=lambda r: order.get(r["Grupp"], 99))


def breakdown(trades: list) -> dict:
    """{rubrik: [rader]} — år, exittyp, sektor, aktie (bästa/sämsta), mått på signaldagen, innehavstid."""
    out = {"År": sorted(group(trades, lambda t: t.signal_date[:4]), key=lambda r: r["Grupp"]),
           "Exittyp": sorted(group(trades, lambda t: t.exit_reason), key=lambda r: -r["Summa R"]),
           "Sektor": sorted(group(trades, lambda t: t.sector or "okänd"), key=lambda r: -r["Summa R"])}
    tick = sorted(group(trades, lambda t: t.ticker), key=lambda r: -r["Summa R"])
    out["Aktie (10 bästa och 10 sämsta)"] = tick if len(tick) <= 20 else tick[:10] + tick[-10:]
    for key, (title, edges, unit) in BUCKETS.items():
        rows = group(trades, lambda t, key=key, edges=edges, unit=unit:
                     _bucket((t.features or {}).get(key), edges, unit))
        if rows:
            out[title] = _ordered(rows, edges, unit)
    out["Innehavstid (handelsdagar)"] = _ordered(group(trades, lambda t: _bucket(t.days, DAY_BUCKETS, " d")),
                                                 DAY_BUCKETS, " d")
    return out


# ── Köp och behåll indexet ──────────────────────────────────────────────────
def benchmark(close: pd.Series) -> Optional[dict]:
    s = close.dropna().astype(float) if close is not None else None
    if s is None or len(s) < 2:
        return None
    years = (s.index[-1] - s.index[0]).days / 365.25
    total = float(s.iloc[-1] / s.iloc[0])
    dd = float((1 - s / s.cummax()).max()) * 100
    cagr = (total ** (1 / years) - 1) * 100 if years > 0 else None
    return {"Avkastning %": round((total - 1) * 100, 1), "CAGR %": None if cagr is None else round(cagr, 1),
            "Max DD %": round(dd, 1), "CAGR/DD": round(cagr / dd, 2) if cagr is not None and dd > 0 else None,
            "Från": str(s.index[0].date()), "Till": str(s.index[-1].date())}


def strategy_vs_benchmarks(portfolio: dict, benchmarks: dict) -> list:
    """Strategins portfölj mot köp och behåll av indexen (prisindex utan utdelning)."""
    cagr, dd = portfolio.get("cagr_pct"), portfolio.get("max_dd_pct")
    rows = [{"Vad": "Viking Nine (portfölj)", "Avkastning %": portfolio.get("return_pct"), "CAGR %": cagr,
             "Max DD %": dd, "CAGR/DD": round(cagr / dd, 2) if cagr is not None and dd else None,
             "Snittexponering %": (portfolio.get("mtm") or {}).get("avg_exposure_pct")}]
    for name, s in (benchmarks or {}).items():
        b = benchmark(s)
        if b:
            rows.append({"Vad": f"Köp och behåll {name}", "Avkastning %": b["Avkastning %"], "CAGR %": b["CAGR %"],
                         "Max DD %": b["Max DD %"], "CAGR/DD": b["CAGR/DD"], "Snittexponering %": 100.0})
    return rows
