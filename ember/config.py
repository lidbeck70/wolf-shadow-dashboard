"""
ember/config.py
EMBER strategy — named threshold constants, universe config, and palette.
"""
from __future__ import annotations

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0c0c12"
BG2    = "#14141e"
BG3    = "#1a1a28"
EMBER  = "#FF6B3D"
GOLD   = "#00E5FF"
BRONZE = "#00A8BF"
GREEN  = "#2d8a4e"
RED    = "#c44545"
AMBER  = "#d4943a"
TEXT   = "#e8e4dc"
DIM    = "#8a8578"

# ── Trend gate thresholds ─────────────────────────────────────────────────────
HIGHER_LOWS_MIN        = 3     # minimum higher lows required in lookback window
HIGHER_LOWS_LOOKBACK_W = 6     # weeks to look back for higher low pattern
RS_LOOKBACK_DAYS       = 63    # 3-month relative strength lookback

# ── Entry gate thresholds ─────────────────────────────────────────────────────
PULLBACK_EMA_PCT = 3.0   # price must be within ±3% of 20D EMA
RSI_ENTRY_MAX    = 45    # RSI(14) must be below this on entry
RSI_PERIOD       = 14
VOL_MIN_RATIO    = 1.0   # current volume ≥ 1.0× 20D average

# ── No-trade zone thresholds ──────────────────────────────────────────────────
ATR_SURGE_PCT       = 40.0  # ATR up > 40% vs 2 weeks ago → volatility spike
ATR_SURGE_LOOKBACK_W = 2    # weeks for surge detection
LATE_CYCLE_PCT      = 85.0  # 10y price percentile > 85 → late / top phase
DXY_SURGE_PCT       = 2.0   # DXY up > 2% in 2 weeks → commodity headwind
DXY_SURGE_LOOKBACK_W = 2

# ── Risk model ────────────────────────────────────────────────────────────────
RISK_PCT      = 0.02   # 2% account risk per trade
ATR_STOP_MULT = 2.5    # stop = entry − 2.5 × ATR(14)
ATR_PERIOD    = 14
MIN_RR        = 2.0    # minimum acceptable risk/reward ratio

# ── Portfolio limits ──────────────────────────────────────────────────────────
MAX_PER_SECTOR = 4
MAX_TOTAL      = 8

# ── Macro score weights (must sum to 100) ─────────────────────────────────────
MACRO_W_COPPER_GOLD  = 30   # copper/gold ratio direction
MACRO_W_DXY          = 25   # DXY 4-week trend
MACRO_W_YIELD_CURVE  = 25   # T10Y2Y steepening/flattening
MACRO_W_CYCLE        = 20   # theme board cycle phase

# ── Sentiment score weights (must sum to 100) ─────────────────────────────────
SENTIMENT_W_SHORT   = 35    # short % of float (contrarian signal)
SENTIMENT_W_ANALYST = 35    # analyst buy/hold/sell ratio
SENTIMENT_W_OPTIONS = 30    # put/call ratio

# ── Cycle asymmetry bonuses (added to ranking score) ─────────────────────────
CYCLE_BONUS_TIDIG  =  20.0
CYCLE_BONUS_MITTEN =  10.0
CYCLE_BONUS_SEN    =   0.0
CYCLE_BONUS_TOPP   = -10.0

# ── External data ─────────────────────────────────────────────────────────────
FRED_T10Y2Y_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
FRED_TIMEOUT    = 8    # seconds

DXY_PRIMARY  = "DX-Y.NYB"
DXY_FALLBACK = "UUP"

# ── ETF universe ──────────────────────────────────────────────────────────────
EMBER_ETF_UNIVERSE: list[str] = [
    "URA",            # Uranium
    "SLV", "SIL",     # Silver
    "GLD", "GDX", "GDXJ",  # Gold
    "COPX",           # Copper
    "XLE", "USO",     # Oil
    "UNG",            # Natural gas
    "BTU",            # Coal (proxy — US-listed)
    "DBA",            # Agriculture
    "REMX",           # Rare earth
]

# ── Seed stock universe (user can extend in UI) ───────────────────────────────
EMBER_STOCK_UNIVERSE: list[str] = [
    "CCJ", "NXE", "DNN",        # Uranium miners
    "PAAS", "HL", "AG",         # Silver/gold miners
    "OXY", "DVN",               # Oil producers
    "FCX", "SCCO",              # Copper miners
    "MOS", "NTR",               # Fertilizer / Agriculture
]

# ── Ticker → sector ETF for RS check ─────────────────────────────────────────
EMBER_SECTOR_ETF: dict[str, str] = {
    "uran":      "URA",
    "silver":    "SIL",
    "guld":      "GDX",
    "koppar":    "COPX",
    "olja":      "XLE",
    "naturgas":  "UNG",
    "kol":       "XLE",     # proxy — no dedicated coal ETF
    "agri":      "DBA",
    "sallsynta": "REMX",
}

DEFAULT_SECTOR_ETF = "GLD"

# ── Ticker → theme key map ────────────────────────────────────────────────────
TICKER_THEME_MAP: dict[str, str] = {
    "URA":  "uran",  "CCJ":  "uran",  "NXE":  "uran",  "DNN":  "uran",
    "SLV":  "silver", "SIL": "silver", "PAAS": "silver", "HL": "silver", "AG": "silver",
    "GLD":  "guld",  "GDX":  "guld",  "GDXJ": "guld",
    "COPX": "koppar", "FCX": "koppar", "SCCO": "koppar",
    "XLE":  "olja",  "USO":  "olja",  "OXY":  "olja",  "DVN": "olja",
    "UNG":  "naturgas",
    "BTU":  "kol",
    "DBA":  "agri",  "MOS":  "agri",  "NTR":  "agri",
    "REMX": "sallsynta",
}

_THEME_LABEL: dict[str, str] = {
    "uran":      "Uran",
    "silver":    "Silver",
    "guld":      "Guld",
    "koppar":    "Koppar",
    "olja":      "Olja",
    "naturgas":  "Naturgas",
    "kol":       "Kol",
    "agri":      "Agri",
    "sallsynta": "Sällsynta Jordartsmetaller",
}
