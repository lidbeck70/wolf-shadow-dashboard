"""
gold_ratios/data.py — priserna till Guldkvoter via den delade priscachen
(market_prices), med Börsdata som reserv för guld, platina, palladium,
koppar och olja när Yahoo inte levererar.

Varje livepris bär proveniens: värde, källa, datum, typ, ACTUAL och
konfidens. Saknas en serie blir den None — aldrig ett gissat pris.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from gold_ratios import config as rc
from gold_ratios import engine as gre

logger = logging.getLogger(__name__)


def _close_default(ticker: str, period: str):
    from market_prices import close
    return close(ticker, period)


_BD_API = None


def _bd_default(ins_id: int):
    """Börsdatas dagliga stängningar (upp till 20 år) eller None."""
    global _BD_API
    try:
        if _BD_API is None:
            from borsdata_api import BorsdataAPI
            _BD_API = BorsdataAPI()
        if not _BD_API.is_configured:
            return None
        df = _BD_API.get_stockprices_df(int(ins_id), max_count=5040)
        return None if df is None or df.empty else df["Close"].astype(float)
    except Exception as exc:
        logger.warning("guldkvoter: Börsdata %s: %s", ins_id, exc)
        return None


def _clean(series):
    if series is None or len(series) == 0:
        return None
    if getattr(series.index, "tz", None) is not None:
        series = series.copy()
        series.index = series.index.tz_localize(None)
    series = series.dropna()
    return series if len(series) else None


def _series(ticker: str, bd_id: Optional[int], scale: float, getter: Callable, bd_getter: Callable) -> tuple:
    """(serie i visad enhet | None, källtext)."""
    try:
        s = _clean(getter(ticker, rc.HISTORY_PERIOD))
    except Exception as exc:                            # pragma: no cover — getter sväljer normalt
        logger.warning("guldkvoter: %s: %s", ticker, exc)
        s = None
    if s is not None:
        return s * scale, f"Yahoo Finance {ticker}"
    if bd_id:
        b = _clean(bd_getter(bd_id))
        if b is not None:
            return b, f"Börsdata (Nymex, insId {bd_id}) — Yahoo {ticker} saknades"
    return None, ""


def _point(series, source: str) -> Optional[dict]:
    if series is None or len(series) == 0:
        return None
    v = float(series.iloc[-1])
    if not v > 0:
        return None
    return {"value": round(v, 4), "source": source, "date": str(series.index[-1])[:10],
            "source_type": "market", "kind": "ACTUAL", "confidence": "hög"}


def fetch_gold(getter: Optional[Callable] = None, bd_getter: Optional[Callable] = None) -> dict:
    getter, bd_getter = getter or _close_default, bd_getter or _bd_default
    s, src = _series(rc.GOLD_TICKER, rc.GOLD_BD_ID, 1.0, getter, bd_getter)
    return {"series": s, "point": _point(s, src)}


def fetch_pair(key: str, gold: dict, getter: Optional[Callable] = None,
               bd_getter: Optional[Callable] = None) -> dict:
    """{pair, gold, other: datapunkt | None, other_series, series: kvot per dag, error}."""
    getter, bd_getter = getter or _close_default, bd_getter or _bd_default
    pair = rc.PAIR_BY_KEY[key]
    s, src = _series(pair["ticker"], pair.get("bd_id"), pair.get("scale", 1.0), getter, bd_getter)
    out = {"pair": pair, "gold": gold.get("point"), "other": _point(s, src), "other_series": s,
           "series": gre.ratio_series(pair, gold.get("series"), s), "error": None}
    if out["gold"] is None or out["other"] is None:
        missing = "guldpriset" if out["gold"] is None else f"priset på {pair['label'].lower()}"
        out["error"] = f"{missing} saknas just nu — skriv in priserna själv nedan"
    return out


def fetch_all(getter: Optional[Callable] = None, bd_getter: Optional[Callable] = None) -> dict:
    """{nyckel: fetch_pair(...)} för alla par — guldet hämtas en gång."""
    gold = fetch_gold(getter, bd_getter)
    return {p["key"]: fetch_pair(p["key"], gold, getter, bd_getter) for p in rc.PAIRS}
