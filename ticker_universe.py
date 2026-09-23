"""
ticker_universe.py — Complete ticker universe for Nordic Alpha Systems
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

# ── Marknadstabellen bor i markets.py ────────────────────────────────────
# Den tabell som stod här (avläst ur /instruments och /instruments/global)
# är nu markets.FALLBACK; Börsdatas /markets går före när API:t finns.
# Namnen nedan finns kvar för moduler som importerar dem.
import markets as _markets
import dead_tickers as _dead

MARKET_SUFFIX = {m.id: m.suffix for m in _markets.FALLBACK.values()
                 if m.kind == _markets.STOCK}
INDEX_MARKETS = {m.id for m in _markets.FALLBACK.values() if m.kind == _markets.INDEX}
NON_STOCK_MARKETS = {m.id for m in _markets.FALLBACK.values() if m.kind == _markets.OTHER}

# ── Region definitions (by Borsdata countryId) ───────────────────────────
# Australien har inget känt lands-id i den avlästa tabellen; regionen får
# sina id ur /countries när API:t finns (REGION_COUNTRY_HINTS), så ASX
# hamnar i universumet så fort Börsdata täcker det.
COUNTRY_REGIONS = {
    "Norden": [1, 2, 3, 4],                    # SE, NO, DK, FI
    "USA": [5],
    "Kanada": [6],
    "Australien": [],
    "England": [7],
    "Tyskland": [8],
    "Frankrike": [9],
    "Sydeuropa": [10, 11, 12],                  # ES, PT, IT
    "Centraleuropa": [13, 14, 15],               # CH, BE, NL
    "Östeuropa & Baltikum": [17, 19, 20, 21],   # PL, EE, LV, LT
}

REGION_COUNTRY_HINTS = {
    "Norden": ("sverige", "sweden", "norge", "norway", "danmark", "denmark", "finland"),
    "USA": ("usa", "united states", "förenta stater"),
    "Kanada": ("kanada", "canada"),
    "Australien": ("australi",),
    "England": ("england", "storbritannien", "united kingdom"),
    "Tyskland": ("tyskland", "germany"),
    "Frankrike": ("frankrike", "france"),
    "Sydeuropa": ("spanien", "spain", "portugal", "italien", "italy"),
    "Centraleuropa": ("schweiz", "switzerland", "belgien", "belgium", "nederländerna", "netherlands"),
    "Östeuropa & Baltikum": ("polen", "poland", "estland", "estonia", "lettland", "latvia",
                             "litauen", "lithuania"),
}


def region_country_ids(regions: list, countries=None) -> set:
    """Lands-id för regionerna: ur /countries (namn) när det finns, annars
    de avlästa id:na. Australien finns bara den första vägen."""
    out = set()
    for region in regions:
        ids = set(COUNTRY_REGIONS.get(region, []))
        if countries:
            ids |= _markets.country_ids(countries, REGION_COUNTRY_HINTS.get(region, ()))
        out |= {i for i in ids if i is not None}
    return out


# Nordic countryIds (use /instruments endpoint)
_NORDIC_COUNTRY_IDS = {1, 2, 3, 4}

# ── Legacy alias: old REGIONS dict for backwards compatibility ───────────
# wolf_panel.py imports REGIONS as TU_REGIONS — point it to COUNTRY_REGIONS
REGIONS = COUNTRY_REGIONS

# ── Hardcoded FALLBACK lists (used when API is unavailable) ──────────────
# Status per ticker (omdöpt/uppköpt) bor i dead_tickers.py och tillämpas i
# FALLBACK_TICKERS nedan — listorna här är råmaterialet.

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

# Fallback tickers grouped by region key (used when API fails).
# dead_tickers.alive(): omdöpta byts (GOLD → B), uppköpta och felplacerade
# släpps (MRO, X, LTHM, UEC.TO, FCU.TO, PHNX.L, AKZA.AS, SMR…).
FALLBACK_TICKERS = {
    "Norden": [],            # Nordic fallback handled by screener's own fallback
    "USA": _dead.alive(US_OIL_GAS + US_GOLD_SILVER + US_URANIUM + US_MINING_MATERIALS
                       + US_ETFS_COMMODITY),
    "Kanada": _dead.alive(CANADA_OIL_GAS + CANADA_MINING),
    "Australien": [],        # ASX kommer via /instruments/global när Börsdata täcker det
    "England": _dead.alive(UK_COMMODITY),
    "Tyskland": [],
    "Frankrike": [],
    "Sydeuropa": _dead.alive(EU_COMMODITY),
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
                table = _markets.current()
                tickers = []
                for _, row in df.iterrows():
                    mid = row.get("marketId")
                    if not _markets.is_stock(mid, table):
                        continue
                    tickers.append(_markets.to_yf(row["ticker"], mid, table))
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
        table = _markets.current()

        def _collect(df):
            for _, row in df.iterrows():
                mid = row.get("marketId")
                if not _markets.is_stock(mid, table) or row.get("countryId") not in country_ids:
                    continue
                # Suffix ur marknadstabellen, "TECK B" → "TECK-B.TO"
                tickers.append(_markets.to_yf(row["ticker"], mid, table))

        # Nordic countries -> /instruments endpoint
        if country_ids & _NORDIC_COUNTRY_IDS:
            nordic_df = get_all_instruments()
            if nordic_df is not None and not nordic_df.empty:
                _collect(nordic_df)

        # Global countries -> /instruments/global endpoint
        global_countries = country_ids - _NORDIC_COUNTRY_IDS
        if global_countries:
            global_df = get_global_instruments()
            if global_df is not None and not global_df.empty:
                _collect(global_df)

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
        # Lands-id: ur /countries när API:t finns (så Australien får sina),
        # annars de avlästa id:na.
        countries = None
        if _HAS_BORSDATA:
            try:
                from borsdata_api import get_api
                api = get_api()
                if api.is_configured:
                    countries = api.get_countries()
                    _markets.load(api=api)      # marknadstabellen ur Börsdata
            except Exception:
                countries = None
        country_ids = region_country_ids(selected_regions, countries)

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
