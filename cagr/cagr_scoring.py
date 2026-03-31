"""
cagr_scoring.py
Combines fundamental, cycle, and technical scores into a single signal.

Supports two scoring models:
  • Börsdata mode:  fund 0-10 + cycle 0-3 + tech 0-4 = 0-17
  • yfinance mode:  fund 0-6  + cycle 0-3 + tech 0-4 = 0-13

Signals use percentage thresholds (works for both scales):
  BUY   >= 65%
  HOLD   40-64%
  SELL  < 40%
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Signal thresholds (percentage-based for universal scaling)
# ---------------------------------------------------------------------------

BUY_THRESHOLD_PCT  = 0.65   # >= 65% of max
HOLD_THRESHOLD_PCT = 0.40   # >= 40% of max

SignalType = Literal["BUY", "HOLD", "SELL"]

# Cyberpunk colour mapping used by the Streamlit layer
SIGNAL_COLORS: dict[SignalType, str] = {
    "BUY":  "#00ff88",   # neon green
    "HOLD": "#ffdd00",   # gold / yellow
    "SELL": "#ff3355",   # neon red
}

SIGNAL_EMOJI: dict[SignalType, str] = {
    "BUY":  "🟢",
    "HOLD": "🟡",
    "SELL": "🔴",
}


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def calculate_total_score(
    fund: dict,
    cycle: dict,
    tech: dict,
) -> dict:
    """
    Aggregate sub-scores into a total score and trading signal.

    Parameters
    ----------
    fund  : dict — output of cagr_fundamentals.score_fundamentals()
                   Expected keys: "fund_score" (int), "fund_max" (int, 6 or 10)
    cycle : dict — output of cagr_cycle.score_cycle()
                   Expected key: "cycle_score" (int 0-3)
    tech  : dict — output of cagr_technical.score_technical()
                   Expected key: "tech_score" (int 0-4)

    Returns
    -------
    dict with keys:
      - total_score : int
      - max_score   : int (17 for Börsdata, 13 for yfinance)
      - signal      : str  ("BUY" | "HOLD" | "SELL")
      - fund_score  : int
      - fund_max    : int
      - cycle_score : int
      - tech_score  : int
      - signal_color: str  (hex colour for UI)
      - score_pct   : float (0.0-1.0)
    """
    fund_score = int(fund.get("fund_score", 0))
    fund_max   = int(fund.get("fund_max", 6))
    cycle_score = int(cycle.get("cycle_score", 0))
    tech_score = int(tech.get("tech_score", 0))

    # Clamp individual scores
    fund_score  = max(0, min(fund_max, fund_score))
    cycle_score = max(0, min(3, cycle_score))
    tech_score  = max(0, min(4, tech_score))

    total = fund_score + cycle_score + tech_score
    max_total = fund_max + 3 + 4

    score_pct = total / max(max_total, 1)
    signal = _classify_signal(score_pct)

    return {
        "total_score":  total,
        "max_score":    max_total,
        "signal":       signal,
        "signal_color": SIGNAL_COLORS[signal],
        "fund_score":   fund_score,
        "fund_max":     fund_max,
        "cycle_score":  cycle_score,
        "tech_score":   tech_score,
        "score_pct":    round(score_pct, 3),
    }


def _classify_signal(score_pct: float) -> SignalType:
    """Map a score percentage (0-1) to a BUY / HOLD / SELL signal."""
    if score_pct >= BUY_THRESHOLD_PCT:
        return "BUY"
    if score_pct >= HOLD_THRESHOLD_PCT:
        return "HOLD"
    return "SELL"


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def score_batch(records: list[dict]) -> list[dict]:
    """
    Score a list of pre-computed sub-score dicts.

    Each record must contain keys:
      ticker, name, country, sector,
      fund (dict), cycle (dict), tech (dict)

    Returns the same list augmented with the total score fields.
    """
    results = []
    for rec in records:
        scores = calculate_total_score(
            fund=rec.get("fund", {}),
            cycle=rec.get("cycle", {}),
            tech=rec.get("tech", {}),
        )
        results.append({**rec, **scores})
    return results


def build_summary_stats(scored_records: list[dict]) -> dict:
    """
    Build KPI summary counts from a list of scored records.

    Returns dict with keys:
      total_scanned, buy_count, hold_count, sell_count, data_source
    """
    total = len(scored_records)
    buy  = sum(1 for r in scored_records if r.get("signal") == "BUY")
    hold = sum(1 for r in scored_records if r.get("signal") == "HOLD")
    sell = sum(1 for r in scored_records if r.get("signal") == "SELL")

    # Detect data source from first record
    data_source = "yfinance"
    if scored_records:
        first_max = scored_records[0].get("fund_max", 6)
        data_source = "Börsdata" if first_max == 10 else "yfinance"

    return {
        "total_scanned": total,
        "buy_count":     buy,
        "hold_count":    hold,
        "sell_count":    sell,
        "data_source":   data_source,
    }
