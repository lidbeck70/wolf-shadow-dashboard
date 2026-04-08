"""
ticker_universe.py — Complete ticker universe for SweWolf Panel
Nordic tickers from Börsdata API, international from curated lists.
"""
import streamlit as st
import pandas as pd

# ── Nordic (from Börsdata API, already implemented in borsdata_api.py) ──────
# Import the existing function
try:
    from borsdata_api import get_all_instruments
    _HAS_BORSDATA = True
except ImportError:
    _HAS_BORSDATA = False

# ── Market suffix mapping ──────────────────────────────────────────────────
# Börsdata marketId → yfinance suffix
BORSDATA_SUFFIX = {
    1: ".ST", 2: ".ST", 3: ".ST", 4: ".ST", 5: ".ST", 6: ".ST",  # Sweden
    9: ".OL", 10: ".OL", 11: ".OL", 12: ".OL", 27: ".OL",        # Norway
    14: ".CO", 15: ".CO", 16: ".CO", 17: ".CO", 30: ".CO",        # Denmark
    20: ".HE", 21: ".HE", 22: ".HE", 23: ".HE",                   # Finland
}

# ── International curated lists ────────────────────────────────────────────
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
    "TTE.PA", "ENI.MI", "REP.MC", "MT.AS",  # TotalEnergies, ENI, Repsol, ArcelorMittal
    "OMV.VI", "GALP.LS", "AKZA.AS",          # OMV, Galp, AkzoNobel
]

# ── Region groupings ──────────────────────────────────────────────────────
REGIONS = {
    "Norden (Börsdata)": "nordic",
    "USA — Olja & Gas": "us_oil",
    "USA — Guld & Silver": "us_gold",
    "USA — Uran": "us_uranium",
    "USA — Gruvor & Material": "us_mining",
    "USA — Råvaru-ETF": "us_etf",
    "Kanada — Olja & Gas": "ca_oil",
    "Kanada — Gruvor": "ca_mining",
    "UK — Råvaror": "uk_commodity",
    "Europa — Råvaror": "eu_commodity",
}

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

# ── Public functions ───────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_nordic_tickers() -> list:
    """Get all Nordic tickers from Börsdata API with yfinance suffixes."""
    try:
        if _HAS_BORSDATA:
            from borsdata_api import get_all_instruments
            df = get_all_instruments()
            if df is not None and not df.empty:
                tickers = []
                for _, row in df.iterrows():
                    mid = row.get("marketId")
                    suffix = BORSDATA_SUFFIX.get(mid, ".ST")
                    # Skip index instruments
                    if mid in (7, 8, 13, 19, 28, 31):
                        continue
                    tickers.append(f"{row['ticker']}{suffix}")
                return sorted(set(tickers))
    except Exception:
        pass
    return []  # Will use fallback in screener


def get_tickers_for_regions(selected_regions: list) -> list:
    """Get combined ticker list for selected regions."""
    try:
        tickers = []
        for region_key in selected_regions:
            if region_key == "nordic":
                tickers.extend(get_nordic_tickers())
            elif region_key in REGION_TICKERS:
                tickers.extend(REGION_TICKERS[region_key])
        return sorted(set(tickers))
    except Exception:
        return []


def get_all_international_tickers() -> list:
    """Get ALL international tickers (no Nordic)."""
    try:
        tickers = []
        for region_key, ticker_list in REGION_TICKERS.items():
            tickers.extend(ticker_list)
        return sorted(set(tickers))
    except Exception:
        return []


def get_complete_universe() -> list:
    """Get ALL available tickers (Nordic + international)."""
    try:
        nordic = get_nordic_tickers()
        intl = get_all_international_tickers()
        return sorted(set(nordic + intl))
    except Exception:
        return []
