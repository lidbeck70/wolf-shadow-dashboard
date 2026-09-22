"""
engines/durrett/confidence.py — CONFIDENCE (SPEC §28), integrerad med
Wolfpanels Confidence score (confidence.scoring.confidence_score).

Data Confidence  = Confidence Score ur confidence-motorn (datakomplett-
                   het, färskhet, källkvalitet, verifiering, konsistens,
                   kill-caps) — byggs inte om.
Model Confidence = data_weight × Data Confidence
                 + model_weight × scenariorobusthet (Bear/Base)
                 − avdrag för andel N/A bland Durretts delpoäng (modell-
                   överensstämmelse kan inte mätas när poäng saknas).
Confidence är INTE attraktivitet: "Durrett 88, Upside 14×, Confidence 61"
betyder stor potential med betydande osäkerhet.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from confidence.scoring import confidence_score
from engines.durrett._base import Ctx
from engines.durrett.models import Score

NA_PENALTY_MAX = 25.0      # VAL: alla delpoäng N/A → −25


def compute(ctx: Ctx, scores: list, robustness: Optional[float], today: Optional[date] = None) -> tuple:
    """(Model Confidence Score, Data Confidence 0–100 | None, confidence-motorns resultat)."""
    cf = confidence_score(ctx.c, today)
    data_conf = cf.total
    w = ctx.cfg["confidence"]
    s = Score("confidence", "Confidence", None)
    s.positive.append(f"+ Data Confidence {data_conf:g} ({cf.band}) ur Confidence score-motorn")
    for k, limit, text in cf.caps_applied:
        s.negative.append(f"− kill-cap {limit}: {text}")
    na = [x.label for x in scores if x.value is None]
    known = len(scores) - len(na)
    na_share = len(na) / len(scores) if scores else 1.0
    if robustness is None:
        model_part = data_conf                       # ingen scenariorobusthet → data bär allt
        s.unknown.append("? scenariorobusthet okänd (Bear/Base saknas) — Data Confidence bär hela vikten")
        value = data_conf
    else:
        value = w["data_weight"] * data_conf + w["model_weight"] * robustness
        model_part = robustness
        (s.positive if robustness >= 50 else s.negative).append(
            f"{'+' if robustness >= 50 else '−'} scenariorobusthet {robustness:g}: Bear behåller {robustness:.0f} % av Base-multipeln")
    penalty = NA_PENALTY_MAX * na_share
    if na:
        s.negative.append(f"− {len(na)}/{len(scores)} delpoäng N/A ({', '.join(na)}) → −{penalty:.0f}")
    s.value = round(max(0.0, min(100.0, value - penalty)), 1)
    s.components = {"data_confidence": (data_conf, w["data_weight"]), "robustness": (model_part, w["model_weight"]),
                    "known_scores": (float(known), 0)}
    return s, data_conf, cf
