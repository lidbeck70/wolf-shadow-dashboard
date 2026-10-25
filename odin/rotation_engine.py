"""
odin/rotation_engine.py — Sektorrotation (ägarens "Sector Rotation Engine"): relativ styrka 12–6–3 månader,
trendfilter och fas per sektor vid varje månadsskifte. Låst i docs/forskning/sektorrotation_spec.md
(version 1) — ändra inte vikter eller gränser här utan en ny version där.

Rent och utan nätverk: in kommer dagskurser (kolumn per fond) och SPY. Varje månadsrad bygger bara på data
fram till månadens sista stängning.

  RS_k     (1 + sektorns avkastning k mån) / (1 + SPY:s) − 1, k = 3, 6, 12
  R_k      percentil 0–100 bland fonderna som är med den månaden
  S        0,40 × R12 + 0,35 × R6 + 0,25 × R3
  Filter   f1 över 10-månaders snittet · f2 snittet stiger (mot förra månaden) · f3 relativa linjen över sitt
           6-månaders snitt
  Fas      4 förlorat ledarskap · 3 mognande ledare · 2 bekräftad ledare · 1 tidig rotation · neutral
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

SPEC_VERSION = 1
SPEC_PATH = "docs/forskning/sektorrotation_spec.md"
HORIZONS = (3, 6, 12)
WEIGHTS = {12: 0.40, 6: 0.35, 3: 0.25}
SMA_MONTHS, RS_SMA_MONTHS = 10, 6
LEADER, WEAK, EARLY_R3, EARLY_R12, CLIMB = 60.0, 40.0, 60.0, 40.0, 3
F4, F3, F2, F1, NEUTRAL = "4 Förlorat ledarskap", "3 Mognande ledare", "2 Bekräftad ledare", "1 Tidig rotation", "Neutral"
PHASES = (F2, F1, F3, F4, NEUTRAL)


def month_ends(daily: pd.DataFrame) -> pd.DataFrame:
    """Månadens sista stängning per kolumn, med den faktiska handelsdagen som index."""
    d = daily.sort_index()
    last = d.groupby(d.index.to_period("M")).apply(lambda g: g.index[-1])
    return d.loc[pd.DatetimeIndex(last.values)]


def _pct_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Percentil 0–100 per rad (lägst 0, högst 100) bland icke-saknade värden."""
    r = df.rank(axis=1, method="average")
    n = df.notna().sum(axis=1)
    out = r.sub(1).div((n - 1).where(n > 1), axis=0) * 100
    single = (n == 1)
    out.loc[single] = df.loc[single].notna().astype(float).where(df.loc[single].notna()) * 100
    return out


def compute(daily: pd.DataFrame, spy: pd.Series, skip: int = 0, trend: str = "10m",
            weights: Optional[dict] = None) -> dict:
    """Alla månadsserier: rs, R, S, filter, eligible, place, climb och fas. skip = 1 ger 12–1 (robusthet),
    trend = "200d" byter filter 1–2 mot 200-dagarssnittet (robusthet)."""
    weights = weights or WEIGHTS
    daily = daily.sort_index()
    spy = spy.reindex(daily.index).ffill()
    m = month_ends(daily)
    sm = spy.reindex(m.index)
    rs = {}
    for k in HORIZONS:
        a = m.shift(skip) / m.shift(k) - 1
        b = sm.shift(skip) / sm.shift(k) - 1
        rs[k] = (1 + a).div(1 + b, axis=0) - 1
    if trend == "200d":
        sma_d = daily.rolling(200, min_periods=200).mean()
        f1 = (daily > sma_d).reindex(m.index)
        f2 = (sma_d > sma_d.shift(21)).reindex(m.index)
        trend_ok = sma_d.shift(21).reindex(m.index).notna()
    else:
        sma = m.rolling(SMA_MONTHS, min_periods=SMA_MONTHS).mean()
        f1, f2 = m > sma, sma > sma.shift(1)
        trend_ok = sma.shift(1).notna()
    line = m.div(sm, axis=0)
    line_sma = line.rolling(RS_SMA_MONTHS, min_periods=RS_SMA_MONTHS).mean()
    f3 = line > line_sma
    eligible = rs[12].notna() & trend_ok & line_sma.notna()
    R = {k: _pct_rank(rs[k].where(eligible)) for k in HORIZONS}
    S = sum(weights[k] * R[k] for k in HORIZONS)
    place = S.rank(axis=1, ascending=False, method="min")
    climb = place.shift(3) - place
    f1, f2, f3 = (f.where(eligible, False).astype(bool) for f in (f1, f2, f3))
    phase = pd.DataFrame(NEUTRAL, index=m.index, columns=m.columns)
    phase = phase.mask(((R[3] >= EARLY_R3) & (R[12] <= EARLY_R12)) | ((climb >= CLIMB) & (R[3] >= 50)), F1)
    phase = phase.mask((S >= LEADER) & f1 & f2 & f3, F2)
    phase = phase.mask((S >= LEADER) & ((R[3] <= WEAK) | (climb <= -CLIMB)), F3)
    phase = phase.mask(~f1 & ~f3 & eligible, F4)
    phase = phase.where(eligible)
    return {"m": m, "rs": rs, "R": R, "S": S, "f1": f1, "f2": f2, "f3": f3, "eligible": eligible,
            "place": place, "climb": climb, "phase": phase}


def table(sig: dict, i: int = -1, labels: Optional[dict] = None) -> list:
    """Raderna för en månad (standard den senaste), högst S först: fond, poäng, rankningar, filter, fas, klättring."""
    date = sig["S"].index[i]
    rows = []
    for t in sig["S"].columns:
        if not bool(sig["eligible"].iloc[i].get(t, False)):
            continue

        def v(df, nd=1):
            x = df.iloc[i].get(t)
            return None if x is None or pd.isna(x) else round(float(x), nd)
        rows.append({"ticker": t, "label": (labels or {}).get(t, t), "date": str(date.date()), "S": v(sig["S"]),
                     "R12": v(sig["R"][12]), "R6": v(sig["R"][6]), "R3": v(sig["R"][3]),
                     "rs12": v(sig["rs"][12] * 100), "rs6": v(sig["rs"][6] * 100), "rs3": v(sig["rs"][3] * 100),
                     "f1": bool(sig["f1"].iloc[i][t]), "f2": bool(sig["f2"].iloc[i][t]), "f3": bool(sig["f3"].iloc[i][t]),
                     "place": v(sig["place"], 0), "climb": v(sig["climb"], 0), "phase": sig["phase"].iloc[i][t]})
    return sorted(rows, key=lambda r: -(r["S"] if r["S"] is not None else -1))


def climbers(sig: dict, months: int = 1, i: int = -1) -> list:
    """[(fond, platser klättrade)] över `months` månader, störst förändring först (negativ = föll)."""
    p = sig["place"]
    if len(p) <= months:
        return []
    ch = (p.shift(months) - p).iloc[i].dropna()
    return [(t, int(v)) for t, v in ch.sort_values(ascending=False).items()]
