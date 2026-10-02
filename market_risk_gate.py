"""
market_risk_gate.py — riskspärren ur 🌩️ Marknadsrisk.

  HÖG       Viking Nine: inga nya entries (NO TRADE med skäl)
            Wolf: halverad positionsstorlek + varning
  FÖRHÖJD   bara information (historiskt 1,3–1,4× normalt — för svagt för en spärr)
  LÅG       ingenting

Nordiska tickers (.ST .OL .CO .HE) mäts mot OMXS30, övriga mot SPY.
Dagens nivå räknas i lätt läge (två års data, ingen kalibrering) och delas
mellan alla sessioner i sex timmar. Okänd nivå spärrar inget men visas.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import market_risk as mr

logger = logging.getLogger(__name__)

HIGH, ELEVATED = "HÖG", "FÖRHÖJD"
WOLF_HIGH_SIZE_FACTOR = 0.5
TTL_S = 6 * 3600
_NORDIC = (".ST", ".OL", ".CO", ".HE")
_CACHE: dict = {}


def market_for(ticker: str) -> str:
    return "OMXS30" if str(ticker or "").upper().endswith(_NORDIC) else "SPY"


def summarize(r: mr.MarketRisk) -> Optional[dict]:
    """Det spärren och larmen behöver ur ett MarketRisk."""
    if r is None or r.error or r.level is None:
        return None
    return {"market": r.market, "label": r.label, "level": r.level, "points": r.points, "possible": r.possible,
            "date": r.date, "active": [s["label"] for s in r.signals if s["active"]]}


def current(market: str, evaluator: Optional[Callable] = None) -> Optional[dict]:
    """Dagens nivå för marknaden (cachad TTL_S i processen). None när den inte gick att räkna."""
    hit = _CACHE.get(market)
    if hit and time.time() - hit[0] < TTL_S:
        return hit[1]
    try:
        r = (evaluator or (lambda m: mr.evaluate(m, light=True)))(market)
        val = summarize(r)
    except Exception as exc:
        logger.warning("marknadsrisk %s: %s", market, exc)
        val = None
    _CACHE[market] = (time.time(), val)
    return val


def for_ticker(ticker: str, evaluator: Optional[Callable] = None) -> Optional[dict]:
    return current(market_for(ticker), evaluator)


def blocks_entry(risk: Optional[dict]) -> bool:
    return bool(risk) and risk.get("level") == HIGH


def size_factor(risk: Optional[dict]) -> float:
    return WOLF_HIGH_SIZE_FACTOR if blocks_entry(risk) else 1.0


def describe(risk: Optional[dict]) -> str:
    if not risk:
        return "Marknadsrisk okänd — ingen spärr"
    act = ", ".join(risk.get("active") or []) or "inga"
    return (f"Marknadsrisk {risk['label']}: {risk['level']} ({risk['points']} av {risk['possible']} varningar: "
            f"{act}) · {risk['date']}")
