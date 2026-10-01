"""
commodity_leverage/beta.py — kurshävstång: hur mycket aktien rört sig när
råvaran rört sig. Veckoavkastning (fredagsstängning), OLS-lutning (beta),
R², och separat för veckor då råvaran steg respektive föll.

Historiskt mått — säger hur aktien har rört sig, inte hur den kommer att
göra det. Fungerar utan resultat (juniorer, prospektering). Rena funktioner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from commodity_leverage import config as cc


@dataclass
class StockBeta:
    beta: float
    r2: float
    weeks: int
    up_beta: Optional[float]           # råvaran upp
    down_beta: Optional[float]         # råvaran ned
    up_weeks: int
    down_weeks: int
    start: str
    end: str
    asymmetric: bool = False           # upp-beta klart över ned-beta
    weak: bool = False                 # R² under BETA_WEAK_R2


def _ols(xs: list, ys: list) -> Optional[tuple]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / sxx, (sxy * sxy / (sxx * syy)) if syy > 0 else 0.0


def weekly_returns(stock, commodity, years: int = cc.BETA_YEARS):
    """DataFrame med kolumnerna s och c (veckoavkastning), de senaste `years` åren."""
    import pandas as pd
    if stock is None or commodity is None or len(stock) == 0 or len(commodity) == 0:
        return pd.DataFrame(columns=["s", "c"])
    s, c = stock.copy(), commodity.copy()
    for x in (s, c):
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
    df = pd.concat([s.resample("W-FRI").last().rename("s"), c.resample("W-FRI").last().rename("c")],
                   axis=1, join="inner").dropna()
    df = df[(df["s"] > 0) & (df["c"] > 0)]
    if len(df) == 0:
        return pd.DataFrame(columns=["s", "c"])
    df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    return df.pct_change().dropna()


def stock_move(b: Optional[StockBeta], commodity_pct: float = cc.STOCK_SHOCK_PCT) -> tuple:
    """(aktiens ungefärliga rörelse i %, vilket beta som användes) vid en
    råvarurörelse — linjärt, historiskt, inte en prognos. Golv −100 %."""
    if b is None:
        return None, ""
    side = b.down_beta if commodity_pct < 0 else b.up_beta
    used, label = (side, "ned-beta" if commodity_pct < 0 else "upp-beta") if side is not None else (b.beta, "beta")
    return round(max(used * commodity_pct, -100.0), 0), f"{label} {used:.2f}×"


def stock_beta(stock, commodity, years: int = cc.BETA_YEARS) -> tuple:
    """(StockBeta | None, fel | None)."""
    r = weekly_returns(stock, commodity, years)
    if len(r) < cc.BETA_MIN_WEEKS:
        return None, f"för kort kurshistorik ({len(r)} veckor < {cc.BETA_MIN_WEEKS})"
    xs, ys = [float(v) for v in r["c"]], [float(v) for v in r["s"]]
    fit = _ols(xs, ys)
    if fit is None:
        return None, "råvarans pris har inte rört sig"
    up = [(x, y) for x, y in zip(xs, ys) if x > 0]
    dn = [(x, y) for x, y in zip(xs, ys) if x < 0]

    def side(pts):
        if len(pts) < cc.BETA_MIN_SIDE_WEEKS:
            return None
        f = _ols([p[0] for p in pts], [p[1] for p in pts])
        return None if f is None else round(f[0], 2)
    ub, db = side(up), side(dn)
    b = StockBeta(round(fit[0], 2), round(fit[1], 3), len(r), ub, db, len(up), len(dn),
                  str(r.index[0])[:10], str(r.index[-1])[:10])
    b.asymmetric = ub is not None and db is not None and ub - db >= cc.ASYMMETRY_MIN_GAP
    b.weak = b.r2 < cc.BETA_WEAK_R2
    return b, None
