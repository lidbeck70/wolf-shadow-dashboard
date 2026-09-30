"""
gold_silver/engine.py — guld/silver-kvoten som en genomskinlig scenariomotor.

Rena funktioner. Ingen förutsägelse: motorn visar var kvoten står, var den
har stått, vad referensen säger och vilket silverpris varje kvot motsvarar.
Ogiltiga eller saknade tal ger None — aldrig noll, aldrig en gissning.

  kvot          = guldpris / silverpris
  silverpris    = guldpris / kvot
  avstånd       = (kvot − referens) / kvot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gold_silver import config as gc


def _pos(v) -> Optional[float]:
    """Positivt tal eller None (0, negativt, NaN, text → None)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f > 0 else None


def ratio(gold, silver) -> Optional[float]:
    g, s = _pos(gold), _pos(silver)
    return None if g is None or s is None else g / s


def implied_silver(gold, target_ratio) -> Optional[float]:
    g, r = _pos(gold), _pos(target_ratio)
    return None if g is None or r is None else g / r


def formula(gold, target_ratio) -> str:
    """'5 000 / 40 = 125,00' — uträkningen som text (ingen svart låda)."""
    s = implied_silver(gold, target_ratio)
    if s is None:
        return "ogiltigt guldpris eller kvot"
    return f"{gold:,.0f} / {target_ratio:g} = {s:,.2f}"


@dataclass
class RevaluationRow:
    ratio: float
    silver: Optional[float]
    change: Optional[float]          # USD mot dagens silver
    change_pct: Optional[float]
    formula: str
    is_current: bool = False
    is_reference: bool = False


def revaluation_table(gold, silver_now, targets=gc.TARGET_RATIOS,
                      reference=gc.REFERENCE_RATIO) -> list:
    """Dagens kvot först, sedan målkvoterna (dubbletter och ogiltiga bort)."""
    rows = []
    cur = ratio(gold, silver_now)
    ratios = ([cur] if cur else []) + [r for r in (_pos(t) for t in targets) if r is not None]
    seen = set()
    s0 = _pos(silver_now)
    for r in ratios:
        key = round(r, 4)
        if key in seen:
            continue
        seen.add(key)
        s = implied_silver(gold, r)
        rows.append(RevaluationRow(
            ratio=round(r, 2), silver=None if s is None else round(s, 2),
            change=None if s is None or s0 is None else round(s - s0, 2),
            change_pct=None if s is None or s0 is None else round((s / s0 - 1) * 100, 1),
            formula=formula(gold, r) if s is not None else "—",
            is_current=bool(cur) and r == cur, is_reference=_pos(reference) is not None and key == round(reference, 4)))
    return rows


def matrix(golds=gc.GOLD_GRID, ratios=gc.MATRIX_RATIOS) -> list:
    """[(guldpris, {kvot: silverpris})] — silverpriset för varje par."""
    out = []
    for g in golds:
        if _pos(g) is None:
            continue
        out.append((float(g), {float(r): round(implied_silver(g, r), 2)
                               for r in ratios if implied_silver(g, r) is not None}))
    return out


@dataclass
class Gap:
    current: float
    reference: float
    difference: float                # kvotenheter
    pct_of_current: float            # (kvot − ref) / kvot i %
    implied_silver: Optional[float]  # silver om kvoten gick mot referensen, vid dagens guld
    upside_pct: Optional[float]


def geological_gap(gold, silver, reference=gc.REFERENCE_RATIO) -> Optional[Gap]:
    cur, ref = ratio(gold, silver), _pos(reference)
    if cur is None or ref is None:
        return None
    s = implied_silver(gold, ref)
    return Gap(round(cur, 2), ref, round(cur - ref, 2), round((cur - ref) / cur * 100, 1),
               None if s is None else round(s, 2), None if s is None else round((s / float(silver) - 1) * 100, 1))


# ── Historik ─────────────────────────────────────────────────────────────────
def ratio_series(gold_close, silver_close):
    """Kvoten per dag där båda har ett positivt pris (pandas Series)."""
    import pandas as pd
    if gold_close is None or silver_close is None or len(gold_close) == 0 or len(silver_close) == 0:
        return pd.Series(dtype=float)
    df = pd.concat([gold_close.rename("g"), silver_close.rename("s")], axis=1, join="inner").dropna()
    df = df[(df["g"] > 0) & (df["s"] > 0)]
    return (df["g"] / df["s"]).rename("ratio")


def _quantile(vals: list, q: float) -> float:
    s = sorted(vals)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class PeriodStats:
    label: str
    years: Optional[int]
    start: str
    end: str
    n: int
    current: float
    mean: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float
    min: float
    max: float
    percentile: float                # dagens kvot mot perioden (% av dagarna ≤ nu)
    complete: bool                   # perioden täcks helt av historiken


def period_stats(series, label: str, years: Optional[int]) -> Optional[PeriodStats]:
    """Statistik för de senaste `years` åren (None = allt). None utan data."""
    if series is None or len(series) < 2:
        return None
    end = series.index[-1]
    sub = series if years is None else series[series.index >= end - _years(years)]
    if len(sub) < 2:
        return None
    vals = [float(v) for v in sub.values]
    cur = float(series.iloc[-1])
    first = series.index[0]
    return PeriodStats(label, years, str(sub.index[0])[:10], str(end)[:10], len(vals), round(cur, 2),
                       round(sum(vals) / len(vals), 2), round(_quantile(vals, 0.5), 2),
                       round(_quantile(vals, 0.10), 2), round(_quantile(vals, 0.25), 2),
                       round(_quantile(vals, 0.75), 2), round(_quantile(vals, 0.90), 2),
                       round(min(vals), 2), round(max(vals), 2),
                       round(sum(1 for v in vals if v <= cur) / len(vals) * 100, 0),
                       complete=years is None or first <= end - _years(years))


def _years(n: int):
    import pandas as pd
    return pd.DateOffset(years=n)


def all_periods(series) -> list:
    out = []
    for label, years in gc.PERIODS:
        st = period_stats(series, label, years)
        if st is not None and (st.complete or years is None):
            out.append(st)
    return out


def position(current, median) -> Optional[str]:
    """Neutral text: under / nära / över historisk median. Ingen värdering."""
    c, m = _pos(current), _pos(median)
    if c is None or m is None:
        return None
    diff = (c / m - 1) * 100
    if abs(diff) <= gc.NEAR_MEDIAN_PCT:
        return "Nära historisk median"
    return "Över historisk median" if diff > 0 else "Under historisk median"


def guide_zone(current) -> Optional[str]:
    """Guidens egna zoner (masterguiden / Råvarubiblioteket), ordagrant."""
    c = _pos(current)
    if c is None:
        return None
    if c >= gc.GUIDE_ACCUMULATE:
        return f"Över {gc.GUIDE_ACCUMULATE:g} — guidens ackumuleringszon för silver (Durretts silvervariant)"
    if c <= gc.GUIDE_LATE:
        return f"Under {gc.GUIDE_LATE:g} — guidens sencykliska zon (trimma silverbolag)"
    return f"Mellan {gc.GUIDE_LATE:g} och {gc.GUIDE_ACCUMULATE:g} — ingen av guidens zoner"


# ── Referenskvoter ──────────────────────────────────────────────────────────
def production_ratio(ref: dict = None) -> Optional[float]:
    ref = ref or gc.REFERENCES["production"]
    return ratio(ref.get("silver_t"), ref.get("gold_t"))


def above_ground_ratio(ref: dict = None) -> Optional[float]:
    """None när en sida saknas — då visas 'Partial data', aldrig en gissad kvot."""
    ref = ref or gc.REFERENCES["above_ground"]
    return ratio(ref.get("silver_t"), ref.get("gold_t"))
