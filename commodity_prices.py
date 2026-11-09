"""
commodity_prices.py — dagspris per råvara åt Rick Rule-arket.

Arket vill ha "råvarupris nu" i samma enhet som AISC/C1 (USD per oz, lb
eller fat). Yahoo har terminspriserna för ädelmetaller, koppar, olja och
gas; uran, kol, zink, järnmalm och litium handlas inte där (LME/spotindex)
och lämnas därför tomma — hellre inget förslag än fel enhet.

Namnen är rotation.COMMODITIES:s (Guld, Silver, …). sheets_refresh
hämtar en gång per råvara och lägger commodity_price på producentraderna.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# råvara → (Yahoo-ticker, enhet)
TICKERS: dict = {
    "guld":      ("GC=F", "USD/oz"),
    "silver":    ("SI=F", "USD/oz"),
    "platina":   ("PL=F", "USD/oz"),
    "palladium": ("PA=F", "USD/oz"),
    "koppar":    ("HG=F", "USD/lb"),
    "olja":      ("CL=F", "USD/fat"),
    "gas":       ("NG=F", "USD/MMBtu"),
}
NOT_ON_YAHOO: tuple = ("uran", "kol", "zink", "järnmalm", "jarnmalm", "litium", "royalty")


def _key(name: str) -> str:
    return str(name or "").strip().lower()


def ticker_for(name: str) -> Optional[tuple]:
    """(ticker, enhet) för råvarunamnet, eller None när Yahoo saknar den."""
    return TICKERS.get(_key(name))


def night_fits(night, name) -> bool:
    """Gäller nattens råvarupris (sheets_refresh: commodity_price på producentraden) råvaran som står i raden NU?
    Natten hämtar priset för råvaran som stod där då — byts den sedan (väljaren, Använd), gäller priset inte längre.
    Terminen jämförs när natten sparade den (Guld → Silver är båda USD/oz), annars enheten: USD/oz jämförs aldrig
    med USD/lb. False utan Yahoo-termin för råvaran ('—', Uran)."""
    spec = ticker_for(name)
    if not spec or not isinstance(night, dict):
        return False
    tick = str(night.get("commodity_ticker") or "").strip()
    if tick:
        return tick == spec[0]
    return spec[1] == str(night.get("commodity_unit") or "").strip()


def spot(name: str) -> Optional[dict]:
    """{price, unit, asof, ticker} — senaste stängning, eller None."""
    spec = ticker_for(name)
    if spec is None:
        return None
    ticker, unit = spec
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        if h is None or h.empty:
            return None
        return {"price": round(float(h["Close"].iloc[-1]), 2), "unit": unit,
                "asof": str(h.index[-1])[:10], "ticker": ticker}
    except Exception as exc:
        logger.warning("råvarupris %s (%s) misslyckades: %s", name, ticker, exc)
        return None


def spot_many(names) -> dict:
    """{råvarunamn: spot} för de namn som går att hämta — en hämtning per råvara."""
    out: dict = {}
    for n in {_key(x) for x in names or [] if x}:
        s = spot(n)
        if s:
            out[n] = s
    return out
