"""
ember/universe.py
EMBER strategy — commodity universe builder.

Three modes
-----------
CURATED  — static ETF + seed stock lists from ember/config.py (fast, no prefilter)
AUTO     — Nordic (Börsdata branch filter) + US/INTL curated static lists
BOTH     — union of CURATED + AUTO

Nordic tickers are fetched from Börsdata API (requires BORSDATA_API_KEY).
US/INTL lists are static, annotated with source ETF and approximate date.

Pre-filter (applied to AUTO and BOTH)
--------------------------------------
- avg daily turnover (close × volume, 20 days) ≥ 5 MSEK / $5 M
- price > SMA200 when ≥ 200 bars are available
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ember.config import (
    EMBER_ETF_UNIVERSE, EMBER_STOCK_UNIVERSE,
    PREFILTER_MIN_TURNOVER, PREFILTER_BATCH_SIZE, PREFILTER_PERIOD,
)

logger = logging.getLogger(__name__)

# ── Universe source labels ─────────────────────────────────────────────────────
SOURCE_CURATED = "Kurerad lista"
SOURCE_AUTO    = "Auto: Norden + US-råvaror"
SOURCE_BOTH    = "Båda"
ALL_SOURCES    = [SOURCE_CURATED, SOURCE_AUTO, SOURCE_BOTH]


# ── Börsdata branch/sector keyword filter ─────────────────────────────────────
_COMMODITY_KEYWORDS: list[str] = [
    # Mining & metals
    "gruv", "gruvdrift", "mining", "mineral",
    "metall", "metal", "guld", "gold", "silver", "koppar", "copper",
    "zink", "zinc", "järn", "iron", "stål", "steel", "aluminium",
    "uranium", "uran", "nickel", "litium", "lithium", "kobolt", "cobalt",
    # Energy
    "olja", "oil", "gas", "naturgas", "petro", "energi", "energy",
    "kol", "coal",
    # Materials
    "material", "råvara", "fosfat", "fertilizer", "gödsel",
    # Forestry / Paper (commodity-driven)
    "skogs", "skog", "trä", "timber", "forest", "papper", "paper",
    "massa", "pulp",
    # Chemicals (raw material adjacent)
    "kemi", "chemical",
    # Agriculture
    "lantbruk", "agri",
]

# Market IDs and their yfinance suffixes (Nordic markets from borsdata_api.py)
_MARKET_SUFFIX: dict[int, str] = {
    1: ".ST", 2: ".ST", 3: ".ST",
    7: ".ST", 8: ".ST", 9: ".ST",
    18: ".ST", 19: ".ST",
    4: ".OL",
    14: ".OL",
    5: ".HE",
    16: ".HE",
    6: ".CO",
    15: ".CO",
}
_NORDIC_MARKET_IDS: set[int] = set(_MARKET_SUFFIX.keys())
# Market IDs that hold indices or non-stock instruments
_INDEX_MARKET_IDS: set[int] = {7, 8, 13, 19, 28, 31}


# ── US / INTL curated static lists ────────────────────────────────────────────
# Source ETFs noted; curated ~June 2026. Focus on liquid swing-tradable names.

# GDX (VanEck Gold Miners ETF) top 15 holdings
_GDX_NAMES: list[str] = [
    "NEM", "GOLD", "AEM", "WPM", "KGC", "AGI", "AU", "GFI",
    "BTG", "EGO", "SSRM", "OR", "SA", "HMY", "DRD",
]

# GDXJ (VanEck Junior Gold Miners ETF) key holdings
_GDXJ_NAMES: list[str] = [
    "AG", "HL", "PAAS", "CDE", "FSM", "EXK", "MAG", "MUX",
    "GPL", "SVM", "ASM", "NGD",
]

# SIL (Global X Silver Miners ETF) key holdings
_SIL_NAMES: list[str] = [
    "WPM", "PAAS", "HL", "AG", "CDE", "FSM", "EXK", "GPL",
]

# COPX (Global X Copper Miners ETF) key holdings
_COPX_NAMES: list[str] = [
    "FCX", "SCCO", "TECK",
]

# URA / URNM (uranium ETFs) key holdings
_URA_NAMES: list[str] = [
    "CCJ", "NXE", "DNN", "UUUU", "LEU", "UEC",
]

# REMX (VanEck Rare Earth/Strategic Metals ETF) key names
_REMX_NAMES: list[str] = [
    "MP",
]

# XLE / XOP (US energy ETFs) top holdings + Shell US-listed ADR
_XLE_NAMES: list[str] = [
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX",
    "OXY", "HAL", "DVN", "BKR", "FANG", "APA", "MRO", "SHEL",
]

# Norwegian energy — Oslo Børs listed
_NORWAY_ENERGY: list[str] = [
    "EQNR.OL",   # Equinor
    "AKRBP.OL",  # Aker BP
    "VAR.OL",    # Var Energi
    "TGS.OL",    # TGS seismic / data
]

# Coal / thermal energy
_COAL_NAMES: list[str] = [
    "BTU", "ARCH", "CEIX", "AMR",
]

# Agriculture / Fertilizers (DBA, MOS, NTR constituents)
_AGRI_NAMES: list[str] = [
    "MOS", "NTR", "CF", "UAN", "ADM",
]

# Canada mining — TSX-listed, most liquid names
_CANADA_MINING: list[str] = [
    "ABX.TO", "NTR.TO", "FM.TO", "LUN.TO",
    "K.TO", "ERO.TO", "AGI.TO", "BTO.TO", "EDV.TO",
    "CCO.TO", "DML.TO", "NXE.TO", "WPM.TO", "FNV.TO",
]

# Canada oil & gas — most liquid TSX names
_CANADA_OIL: list[str] = [
    "SU.TO", "CNQ.TO", "CVE.TO", "IMO.TO", "WCP.TO",
    "ARX.TO", "BTE.TO", "TOU.TO",
]

# UK London-listed commodity names
_UK_COMMODITY: list[str] = [
    "BP.L", "SHEL.L", "RIO.L", "BHP.L", "AAL.L",
    "ANTO.L", "GLEN.L", "FRES.L",
]

# Commodity ETFs to include in AUTO universe
_AUTO_ETFS: list[str] = [
    "GLD", "GDX", "GDXJ", "SLV", "SIL", "SILJ",
    "COPX", "URA", "URNM", "XLE", "XOP",
    "USO", "UNG", "BTU", "DBA", "REMX",
    "PICK", "XME", "LIT",
]

# Full US/INTL curated list (de-duplicated, insertion order preserved)
US_INTL_CURATED: list[str] = list(dict.fromkeys(
    _GDX_NAMES + _GDXJ_NAMES + _SIL_NAMES + _COPX_NAMES
    + _URA_NAMES + _REMX_NAMES + _XLE_NAMES + _COAL_NAMES
    + _AGRI_NAMES + _CANADA_MINING + _CANADA_OIL + _NORWAY_ENERGY
    + _UK_COMMODITY + _AUTO_ETFS
))


# ── Universe stats ─────────────────────────────────────────────────────────────

@dataclass
class UniverseStats:
    source: str
    total_before_prefilter: int = 0
    nordic_raw: int             = 0
    us_intl_raw: int            = 0
    passed_prefilter: int       = 0
    borsdata_available: bool    = False
    borsdata_error: str         = ""


# ── Nordic Börsdata fetch ─────────────────────────────────────────────────────

def _is_commodity(name: str) -> bool:
    text = name.lower()
    return any(kw in text for kw in _COMMODITY_KEYWORDS)


def fetch_nordic_commodity_tickers() -> tuple[list[str], bool, str]:
    """
    Fetch Nordic commodity tickers from Börsdata.
    Returns (tickers, borsdata_available, error_msg).
    """
    try:
        from borsdata_api import get_api  # type: ignore[import]
        api = get_api()
        if not getattr(api, "is_configured", False):
            return [], False, "BORSDATA_API_KEY ej konfigurerad"

        instruments = api.get_instruments()
        if not instruments:
            return [], False, "Inga instrument från Börsdata"

        # Branch id → name
        branch_map: dict[int, str] = {}
        try:
            for b in api.get_branches():
                bid = b.get("id") or b.get("branchId")
                if bid is not None:
                    branch_map[int(bid)] = b.get("name", "")
        except Exception as exc:
            logger.debug("get_branches: %s", exc)

        # Sector id → name
        sector_map: dict[int, str] = {}
        try:
            for s in api.get_sectors():
                sid = s.get("id") or s.get("sectorId")
                if sid is not None:
                    sector_map[int(sid)] = s.get("name", "")
        except Exception as exc:
            logger.debug("get_sectors: %s", exc)

        tickers: list[str] = []
        for inst in instruments:
            mid = int(inst.get("marketId", 0))
            if mid not in _NORDIC_MARKET_IDS or mid in _INDEX_MARKET_IDS:
                continue

            bname = branch_map.get(int(inst.get("branchId", 0) or 0), "")
            sname = sector_map.get(int(inst.get("sectorId",  0) or 0), "")
            cname = inst.get("name", "")

            if not (_is_commodity(bname) or _is_commodity(sname) or _is_commodity(cname)):
                continue

            raw_ticker = inst.get("ticker", "")
            if not raw_ticker:
                continue
            suffix = _MARKET_SUFFIX.get(mid, ".ST")
            tickers.append(f"{raw_ticker}{suffix}")

        return sorted(set(tickers)), True, ""

    except Exception as exc:
        logger.debug("fetch_nordic_commodity_tickers: %s", exc)
        return [], False, str(exc)


# ── Batch pre-filter ──────────────────────────────────────────────────────────

def batch_prefilter(
    tickers: list[str],
    min_turnover: float = PREFILTER_MIN_TURNOVER,
    period: str        = PREFILTER_PERIOD,
    batch_size: int    = PREFILTER_BATCH_SIZE,
) -> tuple[list[str], int]:
    """
    Batch-download 1y price data and apply coarse pre-filter.

    Returns (passed_tickers, n_successfully_downloaded).

    Filters applied:
    - avg(close × volume, 20d) ≥ min_turnover
    - price > SMA200 when ≥ 200 bars available
    """
    try:
        import yfinance as yf
    except ImportError:
        return tickers, len(tickers)

    if not tickers:
        return [], 0

    passed: list[str] = []
    total_downloaded  = 0

    for i in range(0, len(tickers), batch_size):
        batch      = tickers[i : i + batch_size]
        batch_data: dict[str, pd.DataFrame] = {}

        if len(batch) > 1:
            try:
                try:
                    raw = yf.download(
                        batch, period=period, auto_adjust=True,
                        threads=True, progress=False, multi_level_index=False,
                    )
                except TypeError:
                    raw = yf.download(
                        batch, period=period, auto_adjust=True,
                        threads=True, progress=False,
                    )

                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        for ticker in batch:
                            try:
                                df = raw.xs(ticker, level=1, axis=1).dropna(how="all")
                                if len(df) >= 20:
                                    if hasattr(df.index, "tz") and df.index.tz:
                                        df.index = df.index.tz_localize(None)
                                    batch_data[ticker] = df
                            except (KeyError, ValueError):
                                pass
                    else:
                        # Single-ticker MultiIndex collapsed — treat as one ticker
                        df = raw.dropna(how="all")
                        if len(df) >= 20 and len(batch) == 1:
                            if hasattr(df.index, "tz") and df.index.tz:
                                df.index = df.index.tz_localize(None)
                            batch_data[batch[0]] = df
            except Exception as exc:
                logger.debug("batch_prefilter multi-download batch %d: %s", i, exc)
        else:
            # Single ticker
            try:
                try:
                    raw = yf.download(
                        batch[0], period=period, auto_adjust=True,
                        progress=False, multi_level_index=False,
                    )
                except TypeError:
                    raw = yf.download(batch[0], period=period, auto_adjust=True, progress=False)

                if raw is not None and not raw.empty and len(raw) >= 20:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    if hasattr(raw.index, "tz") and raw.index.tz:
                        raw.index = raw.index.tz_localize(None)
                    batch_data[batch[0]] = raw.dropna(how="all")
            except Exception as exc:
                logger.debug("batch_prefilter single-download %s: %s", batch[0], exc)

        total_downloaded += len(batch_data)

        for ticker, df in batch_data.items():
            if "Close" not in df.columns:
                continue

            close = df["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) < 20:
                continue

            # Turnover filter
            if "Volume" in df.columns:
                vol = df["Volume"].squeeze()
                if isinstance(vol, pd.DataFrame):
                    vol = vol.iloc[:, 0]
                vol = vol.dropna()
                if len(vol) >= 20:
                    avg_turnover = float((close * vol).tail(20).mean())
                    if pd.notna(avg_turnover) and avg_turnover < min_turnover:
                        continue

            # SMA200 filter (only when enough bars)
            if len(close) >= 200:
                sma200 = float(close.rolling(200).mean().iloc[-1])
                if pd.notna(sma200) and float(close.iloc[-1]) < sma200:
                    continue

            passed.append(ticker)

    return passed, total_downloaded


# ── Universe builder ──────────────────────────────────────────────────────────

def build_universe(
    source: str,
    use_prefilter: bool = True,
) -> tuple[list[str], UniverseStats]:
    """
    Build the EMBER scan universe from the selected source.

    Parameters
    ----------
    source       : one of SOURCE_CURATED, SOURCE_AUTO, SOURCE_BOTH
    use_prefilter: apply turnover + SMA200 pre-filter for AUTO/BOTH

    Returns
    -------
    (tickers, stats)
    """
    stats = UniverseStats(source=source)

    if source == SOURCE_CURATED:
        tickers = list(dict.fromkeys(EMBER_ETF_UNIVERSE + EMBER_STOCK_UNIVERSE))
        stats.total_before_prefilter = len(tickers)
        stats.passed_prefilter       = len(tickers)
        return tickers, stats

    base: list[str] = []

    if source in (SOURCE_AUTO, SOURCE_BOTH):
        nordic, bd_ok, bd_err = fetch_nordic_commodity_tickers()
        stats.nordic_raw         = len(nordic)
        stats.borsdata_available = bd_ok
        stats.borsdata_error     = bd_err
        base.extend(nordic)

        stats.us_intl_raw = len(US_INTL_CURATED)
        base.extend(US_INTL_CURATED)

    if source == SOURCE_BOTH:
        curated = list(dict.fromkeys(EMBER_ETF_UNIVERSE + EMBER_STOCK_UNIVERSE))
        base.extend(curated)

    base = list(dict.fromkeys(base))
    stats.total_before_prefilter = len(base)

    if use_prefilter:
        passed, _ = batch_prefilter(base)
        stats.passed_prefilter = len(passed)
        return passed, stats

    stats.passed_prefilter = len(base)
    return base, stats
