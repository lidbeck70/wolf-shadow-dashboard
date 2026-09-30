"""
gold_silver/data.py — priserna till Guld/Silver-fliken via den delade
priscachen (market_prices), så samma serie inte hämtas två gånger.

Varje livepris bär proveniens: värde, källa, datum, typ, ACTUAL/ESTIMATE
och konfidens. Saknas en serie blir den None — aldrig ett gissat pris.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from gold_silver import config as gc
from gold_silver import engine as ge

logger = logging.getLogger(__name__)


def _close_default(ticker: str, period: str):
    from market_prices import close
    return close(ticker, period)


def _point(series, ticker: str) -> Optional[dict]:
    if series is None or len(series) == 0:
        return None
    v = float(series.iloc[-1])
    if not v > 0:
        return None
    return {"value": round(v, 2), "source": f"Yahoo Finance {ticker} (närmaste terminen)",
            "date": str(series.index[-1])[:10], "source_type": "market", "kind": "ACTUAL",
            "confidence": "hög"}


def fetch(getter: Optional[Callable] = None) -> dict:
    """{gold, silver: datapunkt | None, series: kvot per dag, error: text | None}."""
    getter = getter or _close_default
    out = {"gold": None, "silver": None, "series": None, "error": None}
    try:
        g = getter(gc.GOLD_TICKER, gc.HISTORY_PERIOD)
        s = getter(gc.SILVER_TICKER, gc.HISTORY_PERIOD)
    except Exception as exc:                            # pragma: no cover — getter sväljer normalt
        logger.warning("guld/silver: %s", exc)
        out["error"] = f"priserna kunde inte hämtas ({exc})"
        return out
    for key, series, ticker in (("gold", g, gc.GOLD_TICKER), ("silver", s, gc.SILVER_TICKER)):
        if series is not None and len(series) and getattr(series.index, "tz", None) is not None:
            series.index = series.index.tz_localize(None)
        out[key] = _point(series, ticker)
    out["series"] = ge.ratio_series(g, s)
    if out["gold"] is None or out["silver"] is None:
        out["error"] = "guld- eller silverpriset saknas just nu — skriv in priserna själv nedan"
    return out
