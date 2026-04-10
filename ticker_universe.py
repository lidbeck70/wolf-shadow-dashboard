"""
ticker_universe.py — Complete ticker universe for SweWolf Panel
All tickers from Borsdata Pro+ API (Nordic + Global = 17,495 instruments).
Hardcoded fallback lists for when API is unavailable.
"""
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Borsdata API imports ─────────────────────────────────────────────────
try:
    from borsdata_api import (
        get_all_instruments,
        get_global_instruments,
        get_complete_instrument_universe,
    )
    _HAS_BORSDATA = True
except ImportError:
    try:
        from .borsdata_api import (
            get_all_instruments,
            get_global_instruments,
            get_complete_instrument_universe,
        )
        _HAS_BORSDATA = True
    except ImportError:
        _HAS_BORSDATA = False

# ── Market suffix mapping (Borsdata marketId -> yfinance suffix) ─────────
# VERIFIED: maps every known marketId from /instruments and /instruments/global
MARKET_SUFFIX = {
    # Nordic (from /instruments)
    1: ".ST", 2: ".ST", 3: ".ST", 4: ".ST", 5: ".ST", 6: ".ST",       # Sweden
    9: ".OL", 10: ".OL", 11: ".OL", 12: ".OL", 27: ".OL", 78: ".OL",  # Norway
    14: ".CO", 15: ".CO", 16: ".CO", 17: ".CO", 30: ".CO",             # Denmark
    20: ".HE", 21: ".HE", 22: ".HE", 23: ".HE", 48: ".HE",            # Finland
    # Global (from /instruments/global)
    32: "",      # NYSE — no suffix
    33: "",      # Nasdaq — no suffix
    34: "",      # OTC — no suffix
    35: ".TO",   # Toronto
    36: ".V",    # TSX Venture
    37: ".CN",   # CSE Canada
    38: ".L",    # England/London
    39: ".DE",   # Tyskland/Xetra
    40: ".PA",   # Frankrike/Paris
    41: ".MC",   # Spanien/Madrid
    42: ".LS",   # Portugal/Lissabon
    43: ".MI",   # Italien/Milano
    44: ".SW",   # Schweiz
    45: ".BR",   # Belgien/Bryssel
    46: ".AS",   # Nederlanderna/Amsterdam
    50: ".WA",   # Polen/Warszawa
    52: ".TL",   # Estland/Tallinn
    53: ".TL",   # Lettland (Baltic)
    54: ".TL",   # Litauen (Baltic)
}

# Index marketIds to exclude (index instruments, not tradable stocks)
INDEX_MARKETS = {7, 8, 13, 19, 28, 31}
# Non-stock markets to exclude
NON_STOCK_MARKETS = {76, 77}  # Forex, Nymex

# ── Region definitions (by Borsdata countryId) ───────────────────────────
COUNTRY_REGIONS = {
    "Norden": [1, 2, 3, 4],                    # SE, NO, DK, FI
    "USA": [5],
    "Kanada": [6],
    "England": [7],
    "Tyskland": [8],
    "Frankrike": [9],
    "Sydeuropa": [10, 11, 12],                  # ES, PT, IT
    "Centraleuropa": [13, 14, 15],               # CH, BE, NL
    "Östeuropa & Baltikum": [17, 19, 20, 21],   # PL, EE, LV, LT
}

# Nordic countryIds (use /instruments endpoint)
_NORDIC_COUNTRY_IDS = {1, 2, 3, 4}

# ── Legacy alias: old REGIONS dict for backwards compatibility ───────────
# wolf_panel.py imports REGIONS as TU_REGIONS — point it to COUNTRY_REGIONS
REGIONS = COUNTRY_REGIONS

# ── Hardcoded FALLBACK lists (used when API is unavailable) ──────────────
# All verified working in yfinance as of April 2026

US_OIL_GAS = [
    "XOM", "CVX", "COP", "EOG", "DVN", "OXY", "MPC", "VLO", "PSX",
    "FANG", "HAL", "SLB", "BKR", "MRO", "APA", "CTRA", "OVV", "EQT",
    "AR", "RRC", "SM", "MTDR", "CHRD", "MGY", "DINO", "PBF", "DK",
]

US_GOLD_SILVER = [
    "NEM", "GOLD", "AEM", "KGC", "AGI", "FNV", "WPM", "RGLD",
    "HL", "AG", "PAAS", "CDE", "EGO", "SSRM", "BTG", "OR",
    "SA", "GFI", "AU", "HMY", "DRD",
]

US_URANIUM = [
    "CCJ", "UEC", "UUUU", "DNN", "NXE", "LEU", "SMR",
]

US_MINING_MATERIALS = [
    "FCX", "SCCO", "TECK", "CLF", "X", "NUE", "STLD",
    "MP", "LAC", "ALB", "LTHM", "SQM",
]

US_ETFS_COMMODITY = [
    "XLE", "XOP", "OIH", "GDX", "GDXJ", "SIL", "SILJ",
    "GLD", "SLV", "URNM", "URA", "COPX", "PICK", "XME",
    "REMX", "LIT", "USO", "BNO", "UNG", "PPLT", "PALL",
]

CANADA_OIL_GAS = [
    "SU.TO", "CNQ.TO", "CVE.TO", "IMO.TO", "WCP.TO",
    "ARX.TO", "BTE.TO", "OVV.TO", "TOU.TO", "VET.TO",
    "CPG.TO", "PEY.TO", "FRU.TO", "TVE.TO", "BIR.TO",
]

CANADA_MINING = [
    "ABX.TO", "NTR.TO", "FM.TO", "LUN.TO", "TKO.TO",
    "K.TO", "ERO.TO", "CS.TO", "WPM.TO", "FNV.TO",
    "AGI.TO", "MND.TO", "BTO.TO", "EDV.TO", "SII.TO",
    "CCO.TO", "DML.TO", "NXE.TO", "UEC.TO", "FCU.TO",
]

UK_COMMODITY = [
    "BP.L", "SHEL.L", "RIO.L", "BHP.L", "AAL.L",
    "ANTO.L", "GLEN.L", "FRES.L", "HBR.L", "TLW.L",
    "CNE.L", "KAZ.L", "HOC.L", "PHNX.L",
]

EU_COMMODITY = [
    "TTE.PA", "ENI.MI", "REP.MC", "MT.AS",
    "OMV.VI", "GALP.LS", "AKZA.AS",
]

# Fallback tickers grouped by region key (used when API fails)
FALLBACK_TICKERS = {
    "Norden": [],            # Nordic fallback handled by screener's own fallback
    "USA": US_OIL_GAS + US_GOLD_SILVER + US_URANIUM + US_MINING_MATERIALS + US_ETFS_COMMODITY,
    "Kanada": CANADA_OIL_GAS + CANADA_MINING,
    "England": UK_COMMODITY,
    "Tyskland": [],
    "Frankrike": [],
    "Sydeuropa": EU_COMMODITY,
    "Centraleuropa": [],
    "Östeuropa & Baltikum": [],
}

# Legacy REGION_TICKERS mapping (old key-based format, kept for any straggling imports)
REGION_TICKERS = {
    "us_oil": US_OIL_GAS,
    "us_gold": US_GOLD_SILVER,
    "us_uranium": US_URANIUM,
    "us_mining": US_MINING_MATERIALS,
    "us_etf": US_ETFS_COMMODITY,
    "ca_oil": CANADA_OIL_GAS,
    "ca_mining": CANADA_MINING,
    "uk_commodity": UK_COMMODITY,
    "eu_commodity": EU_COMMODITY,
}


# ── Public functions ─────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_nordic_tickers() -> list:
    """Get all Nordic tickers from Borsdata API with yfinance suffixes."""
    try:
        if _HAS_BORSDATA:
            df = get_all_instruments()
            if df is not None and not df.empty:
                tickers = []
                for _, row in df.iterrows():
                    mid = row.get("marketId")
                    if mid in INDEX_MARKETS or mid in NON_STOCK_MARKETS:
                        continue
                    suffix = MARKET_SUFFIX.get(mid, ".ST")
                    tickers.append(f"{row['ticker']}{suffix}")
                return sorted(set(tickers))
    except Exception:
        pass
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def _get_api_tickers_for_country_ids(country_ids_tuple: tuple) -> list:
    """Fetch tickers from Borsdata API for a set of countryIds.

    Nordic countries (1-4) use /instruments, others use /instruments/global.
    Returns list of yfinance-formatted ticker strings.
    """
    try:
        if not _HAS_BORSDATA:
            return []

        country_ids = set(country_ids_tuple)
        tickers = []

        # Nordic countries -> /instruments endpoint
        if country_ids & _NORDIC_COUNTRY_IDS:
            nordic_df = get_all_instruments()
            if nordic_df is not None and not nordic_df.empty:
                for _, row in nordic_df.iterrows():
                    mid = row.get("marketId")
                    if mid in INDEX_MARKETS or mid in NON_STOCK_MARKETS:
                        continue
                    cid = row.get("countryId")
                    if cid not in country_ids:
                        continue
                    suffix = MARKET_SUFFIX.get(mid, ".ST")
                    tickers.append(f"{row['ticker']}{suffix}")

        # Global countries -> /instruments/global endpoint
        global_countries = country_ids - _NORDIC_COUNTRY_IDS
        if global_countries:
            global_df = get_global_instruments()
            if global_df is not None and not global_df.empty:
                for _, row in global_df.iterrows():
                    mid = row.get("marketId")
                    if mid in INDEX_MARKETS or mid in NON_STOCK_MARKETS:
                        continue
                    cid = row.get("countryId")
                    if cid not in country_ids:
                        continue
                    suffix = MARKET_SUFFIX.get(mid, "")
                    tickers.append(f"{row['ticker']}{suffix}")

        return sorted(set(tickers))
    except Exception as exc:
        logger.warning("_get_api_tickers_for_country_ids failed: %s", exc)
        return []


def get_tickers_for_regions(selected_regions: list) -> list:
    """Get tickers with yfinance suffixes for selected regions.

    Tries Borsdata API first, falls back to hardcoded lists on failure.
    Accepts region names from COUNTRY_REGIONS keys.
    """
    try:
        # Collect countryIds for all selected regions
        country_ids = set()
        for region in selected_regions:
            country_ids.update(COUNTRY_REGIONS.get(region, []))

        if not country_ids:
            return []

        # Try API first
        if _HAS_BORSDATA:
            api_tickers = _get_api_tickers_for_country_ids(tuple(sorted(country_ids)))
            if api_tickers:
                return api_tickers

        # Fallback to hardcoded lists
        tickers = []
        for region in selected_regions:
            fb = FALLBACK_TICKERS.get(region, [])
            tickers.extend(fb)
        return sorted(set(tickers))
    except Exception:
        # Last resort fallback
        tickers = []
        for region in selected_regions:
            fb = FALLBACK_TICKERS.get(region, [])
            tickers.extend(fb)
        return sorted(set(tickers))


def get_all_international_tickers() -> list:
    """Get ALL international tickers (no Nordic)."""
    try:
        intl_regions = [r for r in COUNTRY_REGIONS if r != "Norden"]
        return get_tickers_for_regions(intl_regions)
    except Exception:
        return []


def get_complete_universe() -> list:
    """Get ALL available tickers (Nordic + international)."""
    try:
        return get_tickers_for_regions(list(COUNTRY_REGIONS.keys()))
    except Exception:
        return []
