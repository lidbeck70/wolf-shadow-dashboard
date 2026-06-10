"""
journal_import/ticker_map.py — ISIN / exchange+symbol → yfinance ticker.

Priority:
  1. ISIN_MAP  (hardcoded + user-maintained)
  2. Nordea exchange+symbol suffix (e.g. HOLMb + Nasdaq Stockholm → HOLM-B.ST)
  3. Returns None — caller reports unresolved tickers
"""
from __future__ import annotations

import re
from typing import Optional

# ── ISIN map — add entries here for tickers not resolved via exchange suffix ──
# Keys are ISINs; values are yfinance tickers.
ISIN_MAP: dict[str, str] = {
    # Sweden
    "SE0000115446": "ATCO-A.ST",   # Atlas Copco A
    "SE0011166610": "ATCO-B.ST",   # Atlas Copco B
    "SE0000107419": "VOLV-B.ST",   # Volvo B
    "SE0000667891": "SEB-A.ST",    # SEB A
    "SE0000193120": "HM-B.ST",     # H&M B
    "SE0000108656": "SHB-A.ST",    # Svenska Handelsbanken A
    "SE0000652216": "INVE-B.ST",   # Investor B
    "SE0005994832": "ALFA.ST",     # Alfa Laval
    "SE0000872305": "AZN.ST",      # AstraZeneca (Swedish listing)
    "SE0000148884": "SAND.ST",     # Sandvik
    "SE0000667925": "SCA-B.ST",    # SCA B
    "SE0000869646": "ERICB.ST",    # Ericsson B
    "SE0000112724": "SKF-B.ST",    # SKF B
    "SE0001116761": "SSAB-A.ST",   # SSAB A
    "SE0000171100": "BOL.ST",      # Boliden
    "SE0000101584": "SINCH.ST",    # Sinch
    "SE0000454746": "NDA-SE.ST",   # Nordea Bank (SE)
    # Norway
    "NO0010096985": "EQNR.OL",    # Equinor
    "NO0003054108": "TEL.OL",     # Telenor
    "NO0010721050": "MOWI.OL",    # Mowi
    "NO0003733800": "NHY.OL",     # Norsk Hydro
    "NO0010096922": "DNB.OL",     # DNB Bank
    "NO0003078900": "ORK.OL",     # Orkla
    "NO0010081235": "STL.OL",     # Storebrand
    "NO0003440709": "YAR.OL",     # Yara
    "NO0005052605": "AKSO.OL",    # Aker Solutions
    "NO0010140502": "SALM.OL",    # SalMar
    "NO0003399084": "GOGL.OL",    # Golden Ocean
    "NO0010209331": "RECSI.OL",   # REC Silicon
    "NO0003399001": "SRBANK.OL",  # SpareBank 1 SR-Bank
    # Finland
    "FI0009000681": "NOKIA.HE",   # Nokia
    "FI0009008403": "FORTUM.HE",  # Fortum
    "FI0009013403": "NESTE.HE",   # Neste
    "FI0009000202": "SAMPO.HE",   # Sampo A
    # Denmark
    "DK0060534915": "NOVO-B.CO",  # Novo Nordisk B
    "DK0010234467": "MAERSK-B.CO", # Maersk B
    "DK0060079531": "RBREW.CO",   # Royal Unibrew
    # US (common via Nordnet)
    "US0378331005": "AAPL",
    "US5949181045": "MSFT",
    "US88160R1014": "TSLA",
    "US02079K3059": "GOOGL",
    "US0231351067": "AMZN",
    "US67066G1040": "NVDA",
    "US30303M1027": "META",
}

# ── Exchange suffix map (Nordea "Børs" column → yfinance suffix) ──────────────
_EXCHANGE_SUFFIX: dict[str, str] = {
    "Nasdaq Stockholm": ".ST",
    "Nasdaq Helsinki":  ".HE",
    "Nasdaq Copenhagen": ".CO",
    "Oslo Børs":        ".OL",
    "Oslo Axess":       ".OL",
    "Euronext Oslo":    ".OL",
    "Xetra":            ".DE",
    "London Stock Exchange": ".L",
    "NYSE":             "",
    "Nasdaq":           "",
    "Nasdaq Global Select Market": "",
    "NYSE Arca":        "",
}


def _normalise_symbol(symbol: str, exchange: str) -> str:
    """
    Adjust broker symbol to yfinance convention.

    Examples:
      HOLMb + Nasdaq Stockholm → HOLM-B.ST
      AKSOA + Oslo Børs        → AKSO-A.OL   (but Aker Solutions is just AKSO.OL)
      NOKIa + Nasdaq Helsinki  → NOKIA.HE
    """
    if not symbol:
        return symbol

    # Lowercase trailing share-class suffix (e.g. 'b', 'a') → uppercase '-B', '-A'
    # Pattern: symbol ends with a single lowercase letter that is a share class
    m = re.match(r'^([A-Z0-9]+)([a-z])$', symbol)
    if m:
        base, cls = m.group(1), m.group(2).upper()
        # yfinance uses dash separator for Swedish share classes
        if exchange in ("Nasdaq Stockholm",):
            symbol = f"{base}-{cls}"
        else:
            # For other exchanges just uppercase
            symbol = f"{base}{cls}"

    return symbol.upper()


def resolve_ticker(isin: str, symbol: str = "", exchange: str = "") -> Optional[str]:
    """
    Return yfinance-compatible ticker for the given ISIN / symbol / exchange.
    Returns None if unresolvable.
    """
    # 1. ISIN lookup (most reliable)
    if isin and isin in ISIN_MAP:
        return ISIN_MAP[isin]

    # 2. Exchange + symbol (Nordea trades)
    if symbol and exchange:
        suffix = _EXCHANGE_SUFFIX.get(exchange)
        if suffix is not None:          # suffix="" is valid (US stocks)
            norm = _normalise_symbol(symbol, exchange)
            return f"{norm}{suffix}"

    # 3. Symbol alone (last resort — no suffix, likely wrong for non-US)
    if symbol:
        norm = _normalise_symbol(symbol, exchange)
        # Only return bare symbol for plausibly US tickers (no special chars)
        if re.match(r'^[A-Z]{1,5}$', norm):
            return norm

    return None
