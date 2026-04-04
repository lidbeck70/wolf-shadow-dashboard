"""
candlesticks.py — Candlestick pattern detection for OVTLYR.

Detects bullish (entry) and bearish (exit/warning) patterns.
Each pattern returns: name, type (bullish/bearish), confidence, description, unicode visual.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class CandlePattern:
    name: str           # "Bullish Engulfing", "Hammer", etc.
    type: str           # "bullish" or "bearish"
    bar_index: int      # which bar triggered it (0 = latest)
    confidence: str     # "Strong" / "Moderate" / "Weak"
    description: str    # Swedish explanation
    visual: str         # Unicode representation like "⬇⬆" or emoji


def detect_patterns(df: pd.DataFrame, lookback: int = 5) -> Dict[str, List[CandlePattern]]:
    """
    Detect candlestick patterns in the last `lookback` bars.

    Returns dict with:
      "bullish": list of bullish patterns (entry signals)
      "bearish": list of bearish patterns (exit/warning signals)

    Patterns detected:

    BULLISH (entry):
      1. Hammer — small body, long lower wick (>2x body), upper wick tiny
         Visual: "🔨", Description: "Hammare — köpare tog kontroll vid botten"
         Strong if: at support/OB level, volume > avg

      2. Bullish Engulfing — green candle fully engulfs previous red candle
         Visual: "⬇⬆", Description: "Bullish Engulfing — köpare övermannade säljare"
         Strong if: second candle has higher volume

      3. Morning Star — red candle, small body (doji-like), green candle
         Visual: "⬇•⬆", Description: "Morning Star — 3-bar reversal, ny upptrend"
         Strong if: 3rd candle closes above 1st candle midpoint

      4. Piercing Line — red candle, then green candle opens below prev low but closes above prev midpoint
         Visual: "⬇↗", Description: "Piercing Line — köpare pressade tillbaka"
         Moderate

      5. Bullish Harami — small green candle inside previous red candle
         Visual: "⬇·", Description: "Bullish Harami — konsolidering, möjlig vändning"
         Weak (needs confirmation)

      6. Three White Soldiers — three consecutive green candles with higher closes
         Visual: "⬆⬆⬆", Description: "Three White Soldiers — stark köpkraft i 3 dagar"
         Strong

    BEARISH (exit/warning):
      1. Shooting Star — small body, long upper wick (>2x body), at top of trend
         Visual: "💫", Description: "Shooting Star — säljare vid toppen"
         Strong if: at resistance/OB level

      2. Bearish Engulfing — red candle fully engulfs previous green candle
         Visual: "⬆⬇", Description: "Bearish Engulfing — säljare tog kontroll"
         Strong

      3. Evening Star — green candle, small body, red candle
         Visual: "⬆•⬇", Description: "Evening Star — 3-bar topp-reversal"
         Strong

      4. Dark Cloud Cover — green candle, then red opens above prev high but closes below prev midpoint
         Visual: "⬆↘", Description: "Dark Cloud Cover — säljtryck"
         Moderate

      5. Bearish Harami — small red candle inside previous green candle
         Visual: "⬆·", Description: "Bearish Harami — momentum avtar"
         Weak

      6. Three Black Crows — three consecutive red candles with lower closes
         Visual: "⬇⬇⬇", Description: "Three Black Crows — starkt säljtryck i 3 dagar"
         Strong

      7. Hanging Man — like hammer but at top of uptrend
         Visual: "🪤", Description: "Hanging Man — varning, möjlig topp"
         Moderate

      8. Doji at resistance — tiny body at top of move
         Visual: "✚", Description: "Doji — obeslutsam marknad, möjlig vändning"
         Weak
    """
    # Implementation checks the last `lookback` bars
    # Each pattern checks specific OHLC relationships
    # Returns patterns found, sorted by most recent first

    if df is None or df.empty or len(df) < 5:
        return {"bullish": [], "bearish": []}

    o = df["Open"].astype(float).values
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    c = df["Close"].astype(float).values

    n = len(df)
    bullish = []
    bearish = []

    # Check last `lookback` bars for patterns
    for offset in range(min(lookback, n - 3)):
        i = n - 1 - offset  # current bar (most recent first)

        body = abs(c[i] - o[i])
        upper_wick = h[i] - max(c[i], o[i])
        lower_wick = min(c[i], o[i]) - l[i]
        is_green = c[i] > o[i]
        is_red = c[i] < o[i]

        if i < 1:
            continue

        prev_body = abs(c[i - 1] - o[i - 1])
        prev_is_green = c[i - 1] > o[i - 1]
        prev_is_red = c[i - 1] < o[i - 1]
        prev_mid = (o[i - 1] + c[i - 1]) / 2

        avg_body = (
            np.mean([abs(c[j] - o[j]) for j in range(max(0, i - 20), i)])
            if i > 1
            else body
        )

        # ── BULLISH PATTERNS ──

        # 1. Hammer
        if is_green and lower_wick > body * 2 and upper_wick < body * 0.5 and body > 0:
            bullish.append(
                CandlePattern(
                    "Hammer", "bullish", offset, "Strong",
                    "Hammare — köpare tog kontroll vid botten", "🔨"
                )
            )

        # 2. Bullish Engulfing
        if (
            is_green and prev_is_red
            and c[i] > o[i - 1] and o[i] < c[i - 1]
            and body > prev_body
        ):
            bullish.append(
                CandlePattern(
                    "Bullish Engulfing", "bullish", offset, "Strong",
                    "Bullish Engulfing — köpare övermannade säljare", "⬇⬆"
                )
            )

        # 3. Morning Star (need i >= 2)
        if i >= 2:
            if (
                c[i - 2] < o[i - 2]
                and abs(c[i - 1] - o[i - 1]) < avg_body * 0.3
                and is_green
                and c[i] > prev_mid
            ):
                bullish.append(
                    CandlePattern(
                        "Morning Star", "bullish", offset, "Strong",
                        "Morning Star — 3-bar reversal, ny upptrend", "⬇•⬆"
                    )
                )

        # 4. Piercing Line
        if (
            is_green and prev_is_red
            and o[i] < l[i - 1]
            and c[i] > prev_mid
            and c[i] < o[i - 1]
        ):
            bullish.append(
                CandlePattern(
                    "Piercing Line", "bullish", offset, "Moderate",
                    "Piercing Line — köpare pressade tillbaka", "⬇↗"
                )
            )

        # 5. Bullish Harami
        if (
            is_green and prev_is_red
            and o[i] > c[i - 1] and c[i] < o[i - 1]
            and body < prev_body * 0.5
        ):
            bullish.append(
                CandlePattern(
                    "Bullish Harami", "bullish", offset, "Weak",
                    "Bullish Harami — konsolidering, möjlig vändning", "⬇·"
                )
            )

        # 6. Three White Soldiers (need i >= 2)
        if (
            i >= 2
            and all(c[i - j] > o[i - j] for j in range(3))
            and c[i] > c[i - 1] > c[i - 2]
        ):
            bullish.append(
                CandlePattern(
                    "Three White Soldiers", "bullish", offset, "Strong",
                    "Three White Soldiers — stark köpkraft i 3 dagar", "⬆⬆⬆"
                )
            )

        # ── BEARISH PATTERNS ──

        # 1. Shooting Star
        if is_red and upper_wick > body * 2 and lower_wick < body * 0.5 and body > 0:
            bearish.append(
                CandlePattern(
                    "Shooting Star", "bearish", offset, "Strong",
                    "Shooting Star — säljare vid toppen", "💫"
                )
            )

        # 2. Bearish Engulfing
        if (
            is_red and prev_is_green
            and c[i] < o[i - 1] and o[i] > c[i - 1]
            and body > prev_body
        ):
            bearish.append(
                CandlePattern(
                    "Bearish Engulfing", "bearish", offset, "Strong",
                    "Bearish Engulfing — säljare tog kontroll", "⬆⬇"
                )
            )

        # 3. Evening Star
        if i >= 2:
            if (
                c[i - 2] > o[i - 2]
                and abs(c[i - 1] - o[i - 1]) < avg_body * 0.3
                and is_red
                and c[i] < (o[i - 2] + c[i - 2]) / 2
            ):
                bearish.append(
                    CandlePattern(
                        "Evening Star", "bearish", offset, "Strong",
                        "Evening Star — 3-bar topp-reversal", "⬆•⬇"
                    )
                )

        # 4. Dark Cloud Cover
        if (
            is_red and prev_is_green
            and o[i] > h[i - 1]
            and c[i] < prev_mid
            and c[i] > o[i - 1]
        ):
            bearish.append(
                CandlePattern(
                    "Dark Cloud Cover", "bearish", offset, "Moderate",
                    "Dark Cloud Cover — säljtryck", "⬆↘"
                )
            )

        # 5. Bearish Harami
        if (
            is_red and prev_is_green
            and o[i] < c[i - 1] and c[i] > o[i - 1]
            and body < prev_body * 0.5
        ):
            bearish.append(
                CandlePattern(
                    "Bearish Harami", "bearish", offset, "Weak",
                    "Bearish Harami — momentum avtar", "⬆·"
                )
            )

        # 6. Three Black Crows
        if (
            i >= 2
            and all(c[i - j] < o[i - j] for j in range(3))
            and c[i] < c[i - 1] < c[i - 2]
        ):
            bearish.append(
                CandlePattern(
                    "Three Black Crows", "bearish", offset, "Strong",
                    "Three Black Crows — starkt säljtryck i 3 dagar", "⬇⬇⬇"
                )
            )

        # 7. Hanging Man (hammer shape but at top of trend)
        if is_red and lower_wick > body * 2 and upper_wick < body * 0.5 and body > 0:
            if i >= 5 and c[i] < max(c[i - j] for j in range(1, min(6, i + 1))):
                bearish.append(
                    CandlePattern(
                        "Hanging Man", "bearish", offset, "Moderate",
                        "Hanging Man — varning, möjlig topp", "🪤"
                    )
                )

        # 8. Doji at resistance — tiny body at top of move
        if body < avg_body * 0.1 and body > 0:
            if i >= 5 and h[i] >= max(h[i - j] for j in range(1, min(6, i + 1))) * 0.99:
                bearish.append(
                    CandlePattern(
                        "Doji", "bearish", offset, "Weak",
                        "Doji — obeslutsam marknad, möjlig vändning", "✚"
                    )
                )

    # Deduplicate (same pattern at same offset)
    seen: set = set()
    unique_bull: List[CandlePattern] = []
    for p in bullish:
        key = (p.name, p.bar_index)
        if key not in seen:
            seen.add(key)
            unique_bull.append(p)

    seen = set()
    unique_bear: List[CandlePattern] = []
    for p in bearish:
        key = (p.name, p.bar_index)
        if key not in seen:
            seen.add(key)
            unique_bear.append(p)

    return {"bullish": unique_bull, "bearish": unique_bear}
