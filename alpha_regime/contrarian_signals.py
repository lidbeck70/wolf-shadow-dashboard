"""
alpha_regime/contrarian_signals.py
Pure staged accumulate/distribute logic for Deep Contrarian mode.
Based on Rule/Sprott cycle framework: buy in despair, distribute in euphoria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Phase sets ───────────────────────────────────────────────────────────────

# Maximum pain / capitulation — strongest accumulate signal
_PHASE_ACCUMULATE_1 = {"CAPITULATION", "DEPRESSION"}

# Disbelief / anger after the crash — still cheap, skepticism high
_PHASE_ACCUMULATE_2 = {"DISBELIEF", "DISBELIEF_NEW", "ANGER", "PANIC"}

# Early recovery — hope just returning, price recrossing 200D
_PHASE_ACCUMULATE_3 = {"HOPE"}

# Trend confirmed — momentum running, hold existing positions / monitor
_PHASE_HOLD = {"OPTIMISM", "BELIEF"}

# Late bull / excitement — start trimming into strength
_PHASE_DISTRIBUTE_1 = {"THRILL"}

# Euphoria / complacency — distribute aggressively
_PHASE_DISTRIBUTE_2 = {"EUPHORIA", "COMPLACENCY"}

# Trend breaking down — momentum gone, distribute defensively
_PHASE_DISTRIBUTE_3 = {"ANXIETY", "DENIAL"}


@dataclass
class ContrairianStageResult:
    stage: str           # e.g. "ACCUMULATE_2"
    label: str           # human-readable headline
    color: str           # CSS hex color
    rationale: list[str] = field(default_factory=list)
    sentiment_note: str = ""
    confidence: str = "MEDIUM"   # "HIGH" | "MEDIUM" | "LOW"


def get_contrarian_stage(
    phase: str,
    price_vs_ma200_pct: float,
    sentiment_score: Optional[float],
    cycle_confidence: float = 50.0,
) -> ContrairianStageResult:
    """
    Map market cycle phase + price position + sentiment to a staged signal.

    Parameters
    ----------
    phase               : winning phase from detect_market_cycle()
    price_vs_ma200_pct  : (price/ma200 - 1) * 100
    sentiment_score     : retail sentiment composite 0-100 (None = unavailable)
    cycle_confidence    : confidence % from detect_market_cycle()

    Returns
    -------
    ContrairianStageResult with stage, label, color, rationale list
    """
    rationale: list[str] = []
    sentiment_note = ""

    # ── Sentiment overlay ────────────────────────────────────────────────────
    # 0-100 scale: < 30 = extreme fear, > 70 = extreme greed
    sent_bearish = sentiment_score is not None and sentiment_score < 30
    sent_bullish = sentiment_score is not None and sentiment_score > 70
    sent_neutral = sentiment_score is None or 30 <= sentiment_score <= 70

    if sent_bearish:
        sentiment_note = f"Retail sentiment extremely bearish ({sentiment_score:.0f}/100) — contrarian BUY signal"
    elif sent_bullish:
        sentiment_note = f"Retail sentiment elevated bullish ({sentiment_score:.0f}/100) — contrarian SELL signal"
    elif sentiment_score is not None:
        sentiment_note = f"Retail sentiment neutral ({sentiment_score:.0f}/100)"
    else:
        sentiment_note = "Retail sentiment unavailable"

    conf_str = "HIGH" if cycle_confidence >= 60 else ("MEDIUM" if cycle_confidence >= 40 else "LOW")

    # ── Stage mapping ────────────────────────────────────────────────────────

    if phase in _PHASE_ACCUMULATE_1:
        rationale += [
            f"Market in {phase} — maximum pain / capitulation",
            f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA — distressed territory",
            "Rule/Sprott: greatest opportunity is at maximum pessimism",
        ]
        if sent_bearish:
            rationale.append("Extreme retail fear confirms contrarian setup")
            confidence = "HIGH"
        else:
            confidence = conf_str
        return ContrairianStageResult(
            stage="ACCUMULATE_1",
            label="ACCUMULATE · Phase 1",
            color="#1a7a3a",
            rationale=rationale,
            sentiment_note=sentiment_note,
            confidence=confidence,
        )

    if phase in _PHASE_ACCUMULATE_2:
        below_200 = price_vs_ma200_pct < 0
        rationale += [
            f"Market in {phase} — disbelief / anger after bear market",
            f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA",
            "Sprott: accumulate in stages while below 200D MA",
        ]
        if not below_200:
            rationale.append("Price recrossing 200D — transition to phase 3 watch")
        if sent_bearish:
            rationale.append("Extreme retail fear confirms contrarian thesis")
        return ContrairianStageResult(
            stage="ACCUMULATE_2",
            label="ACCUMULATE · Phase 2",
            color="#2e9e50",
            rationale=rationale,
            sentiment_note=sentiment_note,
            confidence=conf_str,
        )

    if phase in _PHASE_ACCUMULATE_3:
        at_crossover = -5 <= price_vs_ma200_pct <= 15
        rationale += [
            f"Market in {phase} — hope returning, early recovery",
            f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA",
        ]
        if at_crossover:
            rationale.append("Price near 200D MA crossover — final accumulation window")
        else:
            rationale.append("Price extended from 200D — consider waiting for pullback")
        return ContrairianStageResult(
            stage="ACCUMULATE_3",
            label="ACCUMULATE · Phase 3",
            color="#5aba70",
            rationale=rationale,
            sentiment_note=sentiment_note,
            confidence=conf_str,
        )

    if phase in _PHASE_HOLD:
        rationale += [
            f"Market in {phase} — trend confirmed, momentum running",
            f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA",
            "Hold existing positions · no new accumulation at these levels",
        ]
        if price_vs_ma200_pct > 20:
            rationale.append("Price extended >20% above 200D — watch for THRILL transition")
        return ContrairianStageResult(
            stage="HOLD",
            label="HOLD · Monitor",
            color="#3a8ac4",
            rationale=rationale,
            sentiment_note=sentiment_note,
            confidence=conf_str,
        )

    if phase in _PHASE_DISTRIBUTE_1:
        rationale += [
            f"Market in {phase} — excitement and FOMO driving momentum",
            f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA — extended",
            "Trim 25-33% of position into strength",
        ]
        if sent_bullish:
            rationale.append("Elevated retail sentiment confirms distribution window")
        return ContrairianStageResult(
            stage="DISTRIBUTE_1",
            label="DISTRIBUTE · Phase 1",
            color="#e8961e",
            rationale=rationale,
            sentiment_note=sentiment_note,
            confidence=conf_str,
        )

    if phase in _PHASE_DISTRIBUTE_2:
        rationale += [
            f"Market in {phase} — euphoria / complacency at cycle peak",
            f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA",
            "Distribute 50-75% of position · Rule: sell when others are greedy",
        ]
        if sent_bullish:
            rationale.append("Extreme retail greed — strongest distribution signal")
            confidence = "HIGH"
        else:
            confidence = conf_str
        return ContrairianStageResult(
            stage="DISTRIBUTE_2",
            label="DISTRIBUTE · Phase 2",
            color="#d45c00",
            rationale=rationale,
            sentiment_note=sentiment_note,
            confidence=confidence,
        )

    if phase in _PHASE_DISTRIBUTE_3:
        rationale += [
            f"Market in {phase} — trend breaking down, momentum deteriorating",
            f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA",
            "Exit remaining long exposure · preserve capital for next cycle",
        ]
        if price_vs_ma200_pct < 0:
            rationale.append("Price below 200D MA — trend confirmation of breakdown")
        return ContrairianStageResult(
            stage="DISTRIBUTE_3",
            label="DISTRIBUTE · Phase 3",
            color="#cc2200",
            rationale=rationale,
            sentiment_note=sentiment_note,
            confidence=conf_str,
        )

    # Fallback for DENIAL / PANIC handled below or unmapped phases
    if phase == "PANIC":
        rationale += [
            "Market in PANIC — fear-driven selling, volume spike",
            f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA",
            "Watch for capitulation bottom · begin cautious accumulation if conviction high",
        ]
        return ContrairianStageResult(
            stage="ACCUMULATE_2",
            label="ACCUMULATE · Phase 2 (Panic)",
            color="#2e9e50",
            rationale=rationale,
            sentiment_note=sentiment_note,
            confidence="LOW",
        )

    # Unknown / transitional phase
    rationale.append(f"Market phase {phase!r} — no clear directional signal")
    rationale.append(f"Price {price_vs_ma200_pct:+.1f}% vs 200D MA · await clearer cycle confirmation")
    return ContrairianStageResult(
        stage="HOLD",
        label="HOLD · Await Clarity",
        color="#607080",
        rationale=rationale,
        sentiment_note=sentiment_note,
        confidence="LOW",
    )


# Ordered cycle strip for UI rendering (psychology cycle order)
CYCLE_STRIP_ORDER = [
    "DISBELIEF", "HOPE", "OPTIMISM", "BELIEF", "THRILL", "EUPHORIA",
    "COMPLACENCY", "ANXIETY", "DENIAL", "PANIC", "CAPITULATION",
    "ANGER", "DEPRESSION", "DISBELIEF_NEW",
]

CYCLE_PHASE_COLORS = {
    "DISBELIEF":     "#4a7c59",
    "HOPE":          "#5a9a6a",
    "OPTIMISM":      "#78b44a",
    "BELIEF":        "#00E5FF",
    "THRILL":        "#e8a020",
    "EUPHORIA":      "#ff6030",
    "COMPLACENCY":   "#b06090",
    "ANXIETY":       "#cc8844",
    "DENIAL":        "#cc5533",
    "PANIC":         "#cc3333",
    "CAPITULATION":  "#aa2222",
    "ANGER":         "#882222",
    "DEPRESSION":    "#5a1020",
    "DISBELIEF_NEW": "#3a6a4a",
}
