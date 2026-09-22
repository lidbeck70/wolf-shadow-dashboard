"""
engines/durrett/momentum_fetch.py — momentumfälten ur kurshistorik.

Rena beräkningar (momentum_from_history) på en lista av stängningskurser
och volymer: kurs mot MA200, 6-månadersutveckling, volymtrend 0–2.
fetch_momentum() hämtar historiken via yfinance — anropas bara från
fliken på knapp, aldrig från motorn. Resultatet blir Datapoints med
källa "yfinance", dagens datum och kind ACTUAL (volymtrend MODELLED,
regeln står i note). RS-rank, sektor-/råvarumomentum och nyhetsflöde
sätts inte här.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from confidence.data.provenance import Datapoint, dp

MA_DAYS = 200
SIX_MONTHS_DAYS = 126            # handelsdagar
VOLUME_SHORT_DAYS, VOLUME_LONG_DAYS = 63, 252
VOLUME_UP, VOLUME_DOWN = 1.2, 0.8    # VAL: snittvolym 3 mån / 12 mån ≥ 1,2 → 2, ≤ 0,8 → 0, annars 1


def momentum_from_history(closes: list, volumes: Optional[list] = None) -> dict:
    """{price_vs_ma200_pct, momentum_6m_pct, volume_trend, days} — None där historiken inte räcker."""
    c = [float(x) for x in closes if x is not None and x == x]
    out = {"price_vs_ma200_pct": None, "momentum_6m_pct": None, "volume_trend": None, "days": len(c)}
    if not c:
        return out
    last = c[-1]
    if len(c) >= MA_DAYS and last > 0:
        ma = sum(c[-MA_DAYS:]) / MA_DAYS
        out["price_vs_ma200_pct"] = (last / ma - 1.0) * 100.0 if ma > 0 else None
    if len(c) > SIX_MONTHS_DAYS and c[-SIX_MONTHS_DAYS - 1] > 0:
        out["momentum_6m_pct"] = (last / c[-SIX_MONTHS_DAYS - 1] - 1.0) * 100.0
    v = [float(x) for x in (volumes or []) if x is not None and x == x]
    if len(v) >= VOLUME_LONG_DAYS:
        short = sum(v[-VOLUME_SHORT_DAYS:]) / VOLUME_SHORT_DAYS
        long = sum(v[-VOLUME_LONG_DAYS:]) / VOLUME_LONG_DAYS
        if long > 0:
            r = short / long
            out["volume_trend"] = 2 if r >= VOLUME_UP else (0 if r <= VOLUME_DOWN else 1)
            out["volume_ratio"] = r
    return out


def as_datapoints(m: dict, ticker: str, today: Optional[date] = None) -> dict:
    """{fältnyckel: Datapoint} för det som gick att räkna."""
    today = (today or date.today()).isoformat()
    src = f"yfinance {ticker} ({m.get('days', 0)} handelsdagar)"
    out: dict = {}
    if m.get("price_vs_ma200_pct") is not None:
        out["price_vs_ma200_pct"] = dp(round(m["price_vs_ma200_pct"], 1), kind="ACTUAL", source=src,
                                       source_type="secondary", pub_date=today, unit="%",
                                       note=f"kurs / MA{MA_DAYS} − 1")
    if m.get("momentum_6m_pct") is not None:
        out["momentum_6m_pct"] = dp(round(m["momentum_6m_pct"], 1), kind="ACTUAL", source=src,
                                    source_type="secondary", pub_date=today, unit="%",
                                    note=f"kurs / kurs {SIX_MONTHS_DAYS} handelsdagar sedan − 1")
    if m.get("volume_trend") is not None:
        out["volume_trend"] = dp(int(m["volume_trend"]), kind="MODELLED", source=src, source_type="secondary",
                                 pub_date=today, unit="p",
                                 note=f"snittvolym 3 mån / 12 mån = {m.get('volume_ratio', 0):.2f} "
                                      f"(≥ {VOLUME_UP} → 2, ≤ {VOLUME_DOWN} → 0)")
    return out


def fetch_momentum(ticker: str, today: Optional[date] = None) -> tuple:
    """(datapoints, meddelande). Nätverk via yfinance — bara från fliken."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
    except Exception as exc:                                    # pragma: no cover
        return {}, f"yfinance: {exc}"
    if hist is None or len(hist) == 0:
        return {}, f"ingen kurshistorik för {ticker} — kontrollera tickerformatet (t.ex. ABC.TO, ABC.V, ABC.AX)"
    m = momentum_from_history(list(hist["Close"]), list(hist["Volume"]) if "Volume" in hist else None)
    pts = as_datapoints(m, ticker, today)
    missing = [k for k in ("price_vs_ma200_pct", "momentum_6m_pct", "volume_trend") if k not in pts]
    msg = f"{len(pts)} fält hämtade ur {m['days']} handelsdagar"
    if missing:
        msg += f" · saknas (för kort historik): {', '.join(missing)}"
    return pts, msg


__all__ = ["momentum_from_history", "as_datapoints", "fetch_momentum", "Datapoint"]
