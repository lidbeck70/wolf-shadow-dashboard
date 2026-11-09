"""
commodity_prices.py — dagspris per råvara åt arken (Rick Rule, Poängmodellen, Durrett och Wolf Asymmetry).

Arket vill ha "råvarupris nu" i samma enhet som AISC/C1 (USD per oz, lb,
fat eller ton). Yahoo har terminspriserna för ädelmetaller, koppar, olja,
gas och järnmalm (TIO=F); uran, kol, zink och litium handlas inte där
(LME/spotindex) och lämnas därför tomma — hellre inget förslag än fel enhet.

Namnen är rotation.COMMODITIES:s (Guld, Silver, …) — Durrett-arkets
registernycklar ('gold') översätts med registry_name (rotation_key).
sheets_refresh hämtar en gång per termin och lägger commodity_price på
raderna. Terminen är varken spot eller realiserat pris och kan vara en dag
gammal.
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
    # Ägarens beslut (batch 5): TIO=F (62 % Fe CFR Kina, per torrton) — registrets järnmalm står i USD/t.
    "järnmalm": ("TIO=F", "USD/t"),
    "jarnmalm": ("TIO=F", "USD/t"),
}
NOT_ON_YAHOO: tuple = ("uran", "kol", "zink", "litium", "royalty")
# Vad terminen faktiskt är — visas vid förslaget, så att ingen läser den som realiserat pris
BASIS: dict = {"CL=F": "WTI", "NG=F": "Henry Hub", "TIO=F": "62 % Fe CFR Kina, per torrton"}
# Nattens enhet → arkets enhet som räknas som samma. Ägarens beslut (batch 5): ett fat räknas som en boe.
SAME_UNIT: dict = {"USD/fat": "USD/boe"}


def _key(name: str) -> str:
    return str(name or "").strip().lower()


def ticker_for(name: str) -> Optional[tuple]:
    """(ticker, enhet) för råvarunamnet, eller None när Yahoo saknar den."""
    return TICKERS.get(_key(name))


def registry_name(commodity) -> Optional[str]:
    """Durrett-/Wolf Asymmetry-arkets råvara (registernyckel 'gold', eller ett alias) → namnet i TICKERS ('guld')
    via registrets rotation_key. None utan råvara eller utan rotationsnamn (Sällsynta, Nickel …)."""
    if not str(commodity or "").strip():
        return None
    try:
        from confidence import commodities as com
        c = com.get(commodity)
    except Exception:
        return None
    return (c.rotation_key or None) if c else None


def unit_fits(night_unit, sheet_unit) -> bool:
    """Står nattens pris i arkets enhet? Samma text, eller SAME_UNIT (USD/fat → USD/boe). USD/lb mot USD/lb U3O8
    eller USD/oz mot USD/lb är aldrig samma."""
    n, s = str(night_unit or "").strip(), str(sheet_unit or "").strip()
    return bool(n and s) and (n == s or SAME_UNIT.get(n) == s)


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
    """{råvarunamn: spot} för de namn som går att hämta — en hämtning per termin, så 'Järnmalm' (Rick Rule) och
    'jarnmalm' (Durrett-arkets rotation_key) delar samma TIO=F."""
    by_ticker: dict = {}
    for n in sorted({_key(x) for x in names or [] if x}):
        spec = ticker_for(n)
        if spec:
            by_ticker.setdefault(spec[0], []).append(n)
    out: dict = {}
    for same in by_ticker.values():
        s = spot(same[0])
        if s:
            for n in same:
                out[n] = s
    return out
