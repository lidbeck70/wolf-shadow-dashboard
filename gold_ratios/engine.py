"""
gold_ratios/engine.py — Guldkvoternas motor: ett tunt lager ovanpå
gold_silver.engine (som lämnas orörd).

Guld/Silver-motorns funktioner är generella: kvot = täljare / nämnare och
nämnare = täljare / kvot. Här väljs bara vilken sida som är täljare:
  råvara  guld ÷ råvara   → råvarupriset räknas ut ur guldet
  index   index ÷ guld    → guldpriset räknas ut ur indexet
Referensen och målkvoterna är kvotens egna percentiler — ingen geologi.
"""

from __future__ import annotations

import math
from typing import Optional

from gold_ratios import config as rc
from gold_silver import engine as ge


def orient(pair: dict, gold, other) -> tuple:
    """(täljare, nämnare) för paret."""
    return (gold, other) if pair["kind"] == rc.COMMODITY else (other, gold)


def ratio_series(pair: dict, gold_close, other_close):
    num, den = orient(pair, gold_close, other_close)
    return ge.ratio_series(num, den)


def full_stats(series):
    """Statistik över hela historiken (referens och målkvoter)."""
    return ge.period_stats(series, "Max", rc.REFERENCE_PERIOD)


def targets(stats) -> list:
    """[(etikett, kvot)] — kvotens egna percentiler P10 … P90."""
    if stats is None:
        return []
    return [(label, getattr(stats, attr)) for label, attr in rc.TARGET_QUANTILES]


def _round_sig(v: float, sig: int = 2) -> float:
    if not v or v <= 0:
        return 0.0
    digits = sig - 1 - int(math.floor(math.log10(v)))
    return round(v, digits)


def anchor_grid(num_now, multiples=rc.ANCHOR_GRID) -> tuple:
    """Matrisens rader: täljaren (guld resp. index) som multiplar av dagens värde."""
    v = ge._pos(num_now)
    if v is None:
        return ()
    return tuple(dict.fromkeys(_round_sig(v * m) for m in multiples))


def position_row(pair: dict, series) -> dict:
    """En rad i översikten: dagens kvot mot 10 år och mot hela historiken."""
    out = {"pair": pair, "current": None, "median10": None, "diff_pct": None, "pctl10": None,
           "median_max": None, "position": None, "start": None}
    if series is None or len(series) < 2:
        return out
    stats = ge.all_periods(series)
    long = next((s for s in stats if s.years == rc.POSITION_PERIOD_YEARS), None)
    full = full_stats(series)
    cur = float(series.iloc[-1])
    out.update(current=round(cur, 2), median_max=full.median if full else None,
               start=full.start if full else None)
    if long is not None:
        out.update(median10=long.median, diff_pct=round((cur / long.median - 1) * 100, 0),
                   pctl10=long.percentile, position=ge.position(cur, long.median))
    return out


def meaning(pair: dict) -> str:
    """Vad en hög kvot betyder — i ord, utan värdering."""
    if pair["kind"] == rc.COMMODITY:
        return f"Hög kvot = ett uns guld köper mycket {pair['label'].lower()}"
    return f"Hög kvot = {pair['label']} kostar många uns guld"


def unit_word(pair: dict) -> str:
    """'USD/lb' → 'lb'."""
    return pair["unit"].split("/")[-1] if "/" in pair["unit"] else pair["unit"]


def decimals(v) -> int:
    v = ge._pos(v)
    if v is None:
        return 2
    return 2 if v < 100 else 1 if v < 1000 else 0
