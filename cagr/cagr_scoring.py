"""
cagr_scoring.py
Combines fundamental, cycle, and technical scores into a single signal.

Supports two scoring models:
  • Börsdata mode:  fund 0-20 + cycle 0-3 + tech 0-7 = 0-30
  • yfinance mode:  fund 0-6  + cycle 0-3 + tech 0-7 = 0-16

Backward-compatible: old fund_max=10 callers still work (max_score = 17).

Signal classification uses hard-gate rules (the user's 10 long-term rules)
combined with score percentage thresholds:

  Sell triggers checked first:
    • STRONG SELL  — sell trigger fires AND score_pct < 55 %
    • SELL         — sell trigger fires (any score)

  If no sell trigger:
    • STRONG BUY   — score_pct >= 70 % AND all hard gates pass
    • BUY          — score_pct >= 55 % AND all hard gates pass
    • HOLD         — score_pct >= 35 % (gates fail or score 35–54 %)
    • SELL         — score_pct < 35 %
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Signal types and UI colours
# ---------------------------------------------------------------------------

SignalType = Literal["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]

SIGNAL_COLORS: dict[str, str] = {
    "STRONG BUY":  "#c9a84c",   # gold
    "BUY":         "#2d8a4e",   # forest green
    "HOLD":        "#d4943a",   # warm amber
    "SELL":        "#c44545",   # deep red
    "STRONG SELL": "#8b7340",   # bronze
}

SIGNAL_EMOJI: dict[str, str] = {
    "STRONG BUY":  "🟦",
    "BUY":         "🟢",
    "HOLD":        "🟡",
    "SELL":        "🔴",
    "STRONG SELL": "🟣",
}

# Score percentage thresholds
STRONG_BUY_PCT  = 0.70   # >= 70 %
BUY_PCT         = 0.55   # >= 55 %
HOLD_PCT        = 0.35   # >= 35 %
# Below HOLD_PCT → SELL

# Legacy constants kept for any importers that referenced them
BUY_THRESHOLD_PCT  = BUY_PCT
HOLD_THRESHOLD_PCT = HOLD_PCT


# ---------------------------------------------------------------------------
# Rule 1-5: Hard-gate check (BUY gates)
# ---------------------------------------------------------------------------

def _check_hard_gates(
    fund: dict,
    cycle: dict,
    tech: dict,
    fear_greed: dict | None = None,
) -> dict:
    """
    Evaluate the user's 5 per-stock BUY gates (rules 1-5).

    Rules 8-10 are portfolio-level and not evaluated here.
    Rules 6-7 are SELL triggers evaluated separately.

    Parameters
    ----------
    fund       : dict — fundamentals sub-score dict (unused by gates currently,
                 reserved for future fundamental hard gates)
    cycle      : dict — cycle sub-score dict; must contain 'cycle_score' (0-3)
    tech       : dict — technical sub-score dict; must contain 'details' sub-dict
    fear_greed : dict | None — optional dict with key 'fear_greed_score' (0-100)

    Returns
    -------
    dict with:
      - all_pass : bool  — True only when every gate passes
      - gates    : list of {rule: str, pass: bool, reason: str}
    """
    gates: list[dict] = []
    cycle_score  = int(cycle.get("cycle_score", 0))
    tech_details = tech.get("details", {})

    # Rule 1: Buy only in green regime (cycle_score >= 2)
    r1_pass = cycle_score >= 2
    gates.append({
        "rule":   "1. Green Regime",
        "pass":   r1_pass,
        "reason": f"Cycle score {cycle_score}/3" + (" ✓" if r1_pass else " — sector not bullish"),
    })

    # Rule 2: Price > EMA200
    r2_detail = tech_details.get("Price > EMA200", {})
    r2_pass   = r2_detail.get("pass", False)
    gates.append({
        "rule":   "2. Price > EMA200",
        "pass":   r2_pass,
        "reason": str(r2_detail.get("value", "N/A")) + (" ✓" if r2_pass else " ✗"),
    })

    # Rule 3: EMA50 > EMA200 (golden cross)
    r3_detail = tech_details.get("EMA50 > EMA200", {})
    r3_pass   = r3_detail.get("pass", False)
    gates.append({
        "rule":   "3. EMA50 > EMA200",
        "pass":   r3_pass,
        "reason": str(r3_detail.get("value", "N/A")) + (" ✓" if r3_pass else " ✗"),
    })

    # Rule 4: Sector must be green (equivalent to cycle >= 2)
    r4_pass = cycle_score >= 2
    gates.append({
        "rule":   "4. Sector Green",
        "pass":   r4_pass,
        "reason": f"Cycle {cycle_score}/3" + (" ✓" if r4_pass else " — sector bearish"),
    })

    # Rule 5: Fear & Greed < 60
    if fear_greed is not None:
        fg_score = fear_greed.get("fear_greed_score", 50)
        r5_pass  = int(fg_score) < 60
        r5_label = str(fg_score)
    else:
        r5_pass  = True   # no data → don't block
        r5_label = "N/A"
    gates.append({
        "rule":   "5. Fear & Greed < 60",
        "pass":   r5_pass,
        "reason": f"F&G: {r5_label}" + (" ✓" if r5_pass else " — too euphoric"),
    })

    return {
        "all_pass": all(g["pass"] for g in gates),
        "gates":    gates,
    }


# ---------------------------------------------------------------------------
# Rules 6-7: Sell trigger check
# ---------------------------------------------------------------------------

def _check_sell_triggers(tech: dict, cycle: dict) -> dict:
    """
    Evaluate the user's two per-stock SELL triggers (rules 6-7).

    Parameters
    ----------
    tech  : dict — technical sub-score dict; must contain 'details' sub-dict
    cycle : dict — cycle sub-score dict; must contain 'cycle_score' (0-3)

    Returns
    -------
    dict with:
      - should_sell : bool  — True if ANY trigger fires
      - triggers    : list of {rule: str, triggered: bool, reason: str}
    """
    triggers: list[dict] = []
    tech_details = tech.get("details", {})
    cycle_score  = int(cycle.get("cycle_score", 0))

    # Rule 6: Sell when price closes below EMA200
    r6_detail   = tech_details.get("Price > EMA200", {})
    r6_triggered = not r6_detail.get("pass", True)
    triggers.append({
        "rule":      "6. Price < EMA200",
        "triggered": r6_triggered,
        "reason":    str(r6_detail.get("value", "N/A")) + (" — below EMA200" if r6_triggered else ""),
    })

    # Rule 7: Sell when regime turns red (cycle_score == 0)
    r7_triggered = cycle_score == 0
    triggers.append({
        "rule":      "7. Regime Red",
        "triggered": r7_triggered,
        "reason":    f"Cycle score {cycle_score}/3" + (" — regime bearish" if r7_triggered else ""),
    })

    return {
        "should_sell": any(t["triggered"] for t in triggers),
        "triggers":    triggers,
    }


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def _classify_signal(
    score_pct:     float,
    gates_result:  dict,
    sell_triggers: dict,
) -> SignalType:
    """
    Apply the 5-level signal logic.

    Priority order:
      1. Sell triggers → SELL (or STRONG SELL when score also weak)
      2. Score + gates → STRONG BUY | BUY | HOLD | SELL
    """
    if sell_triggers.get("should_sell", False):
        if score_pct < BUY_PCT:           # score_pct < 55 % → STRONG SELL
            return "STRONG SELL"
        return "SELL"

    all_gates = gates_result.get("all_pass", False)

    if score_pct >= STRONG_BUY_PCT and all_gates:
        return "STRONG BUY"
    if score_pct >= BUY_PCT and all_gates:
        return "BUY"
    if score_pct >= HOLD_PCT:
        return "HOLD"   # good gates but bad score, or bad gates, or mid score
    return "SELL"


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def calculate_total_score(
    fund:        dict,
    cycle:       dict,
    tech:        dict,
    fear_greed:  dict | None = None,
) -> dict:
    """
    Aggregate sub-scores into a total score and 5-level trading signal.

    Parameters
    ----------
    fund       : dict — output of cagr_fundamentals.score_fundamentals()
                 Expected keys: 'fund_score' (int), 'fund_max' (int)
                 fund_max: 20 (Börsdata), 6 (yfinance), or 10 (legacy)
    cycle      : dict — output of cagr_cycle.score_cycle()
                 Expected key: 'cycle_score' (int 0-3)
    tech       : dict — output of cagr_technical.score_technical()
                 Expected keys: 'tech_score' (int 0-7), 'details' (dict)
    fear_greed : dict | None — optional; expected key 'fear_greed_score' (0-100)

    Returns
    -------
    dict with keys:
      total_score    : int
      max_score      : int   (30 Börsdata | 16 yfinance | 17 legacy fund_max=10)
      signal         : str   ('STRONG BUY'|'BUY'|'HOLD'|'SELL'|'STRONG SELL')
      signal_color   : str   (hex colour for UI)
      signal_emoji   : str
      fund_score     : int
      fund_max       : int
      cycle_score    : int
      tech_score     : int
      score_pct      : float (0.0–1.0)
      gates_result   : dict  (all_pass, gates list)
      sell_triggers  : dict  (should_sell, triggers list)
    """
    fund_score  = int(fund.get("fund_score", 0))
    fund_max    = int(fund.get("fund_max", 6))
    cycle_score = int(cycle.get("cycle_score", 0))
    tech_score  = int(tech.get("tech_score", 0))

    # Determine tech ceiling from actual data (7 for new system, 4 for legacy)
    tech_max = int(tech.get("tech_max", 7))

    # Clamp individual scores to their maximums
    fund_score  = max(0, min(fund_max,    fund_score))
    cycle_score = max(0, min(3,           cycle_score))
    tech_score  = max(0, min(tech_max,    tech_score))

    total     = fund_score + cycle_score + tech_score
    max_total = fund_max   + 3           + tech_max

    score_pct = total / max(max_total, 1)

    gates_result  = _check_hard_gates(fund, cycle, tech, fear_greed)
    sell_triggers = _check_sell_triggers(tech, cycle)
    signal        = _classify_signal(score_pct, gates_result, sell_triggers)

    return {
        "total_score":   total,
        "max_score":     max_total,
        "signal":        signal,
        "signal_color":  SIGNAL_COLORS[signal],
        "signal_emoji":  SIGNAL_EMOJI[signal],
        "fund_score":    fund_score,
        "fund_max":      fund_max,
        "cycle_score":   cycle_score,
        "tech_score":    tech_score,
        "score_pct":     round(score_pct, 3),
        "gates_result":  gates_result,
        "sell_triggers": sell_triggers,
    }


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def score_batch(records: list[dict]) -> list[dict]:
    """
    Score a list of pre-computed sub-score dicts.

    Each record must contain keys:
      ticker, name, country, sector,
      fund (dict), cycle (dict), tech (dict)

    Optional per-record key:
      fear_greed (dict)

    Returns the same list augmented with the total score fields.
    """
    results = []
    for rec in records:
        scores = calculate_total_score(
            fund=rec.get("fund", {}),
            cycle=rec.get("cycle", {}),
            tech=rec.get("tech", {}),
            fear_greed=rec.get("fear_greed"),
        )
        results.append({**rec, **scores})
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary_stats(scored_records: list[dict]) -> dict:
    """
    Build KPI summary counts from a list of scored records.

    Returns dict with keys:
      total_scanned, strong_buy_count, buy_count, hold_count,
      sell_count, strong_sell_count, data_source
    """
    total        = len(scored_records)
    strong_buy   = sum(1 for r in scored_records if r.get("signal") == "STRONG BUY")
    buy          = sum(1 for r in scored_records if r.get("signal") == "BUY")
    hold         = sum(1 for r in scored_records if r.get("signal") == "HOLD")
    sell         = sum(1 for r in scored_records if r.get("signal") == "SELL")
    strong_sell  = sum(1 for r in scored_records if r.get("signal") == "STRONG SELL")

    # Detect data source from first record's fund_max
    data_source = "yfinance"
    if scored_records:
        first_max = scored_records[0].get("fund_max", 6)
        if first_max == 20:
            data_source = "Börsdata"
        elif first_max == 10:
            data_source = "Börsdata (legacy)"

    return {
        "total_scanned":    total,
        "strong_buy_count": strong_buy,
        "buy_count":        buy,
        "hold_count":       hold,
        "sell_count":       sell,
        "strong_sell_count": strong_sell,
        "data_source":      data_source,
    }
