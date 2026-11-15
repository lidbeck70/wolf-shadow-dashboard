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
# Kontrollerad mot Börsdatas instrumentlistor 2026-11-15 (ISIN → bolag → ticker). Den gamla listan kopplade
# de flesta nordiska ISIN till fel bolag (SEB A och Sandvik ombytta, Volvo B → Atlas Copco A, Mowi → Telenor …)
# och hade ISIN som byts ut (Atlas Copco efter splitten, Investor, Boliden, Novo, Sampo). Golden Ocean, REC Silicon
# och SpareBank 1 SR-Bank finns inte längre hos Börsdata och är borttagna. Lägg bara till rader som är
# kontrollerade mot en källa — en gissning här ger fel ticker på journalens rader.
ISIN_MAP: dict[str, str] = {
    # Sweden
    "SE0017486889": "ATCO-A.ST",   # Atlas Copco A
    "SE0017486897": "ATCO-B.ST",   # Atlas Copco B
    "SE0000115446": "VOLV-B.ST",   # Volvo B
    "SE0000148884": "SEB-A.ST",    # SEB A
    "SE0000106270": "HM-B.ST",     # H&M B
    "SE0007100599": "SHB-A.ST",    # Svenska Handelsbanken A
    "SE0015811963": "INVE-B.ST",   # Investor B
    "SE0000695876": "ALFA.ST",     # Alfa Laval
    "GB0009895292": "AZN.ST",      # AstraZeneca (samma ISIN noteras också i London)
    "SE0000667891": "SAND.ST",     # Sandvik
    "SE0000112724": "SCA-B.ST",    # SCA B
    "SE0000667925": "TELIA.ST",    # Telia Company
    "SE0000108656": "ERIC-B.ST",   # Ericsson B
    "SE0000108227": "SKF-B.ST",    # SKF B
    "SE0000171100": "SSAB-A.ST",   # SSAB A
    "SE0020050417": "BOL.ST",      # Boliden
    "SE0016101844": "SINCH.ST",    # Sinch
    "FI4000297767": "NDA-SE.ST",   # Nordea Bank (samma ISIN i Helsingfors och Köpenhamn)
    # Norway
    "NO0010096985": "EQNR.OL",    # Equinor
    "NO0010063308": "TEL.OL",     # Telenor
    "NO0003054108": "MOWI.OL",    # Mowi
    "NO0005052605": "NHY.OL",     # Norsk Hydro
    "NO0010161896": "DNB.OL",     # DNB Bank
    "NO0003733800": "ORK.OL",     # Orkla
    "NO0003053605": "STB.OL",     # Storebrand
    "NO0010208051": "YAR.OL",     # Yara International
    "NO0010716582": "AKSO.OL",    # Aker Solutions
    "NO0010310956": "SALM.OL",    # SalMar
    # Finland
    "FI0009000681": "NOKIA.HE",   # Nokia
    "FI0009007132": "FORTUM.HE",  # Fortum
    "FI0009013296": "NESTE.HE",   # Neste
    "FI4000552500": "SAMPO.HE",   # Sampo
    # Denmark
    "DK0062498333": "NOVO-B.CO",  # Novo Nordisk B
    "DK0010244508": "MAERSK-B.CO",  # A.P. Møller-Mærsk B
    "DK0060634707": "RBREW.CO",   # Royal Unibrew
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
