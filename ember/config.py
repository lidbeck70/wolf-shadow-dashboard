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

# ── Graderad setup-poäng (ersätter "alla nio grindar eller inget") ──────────
# Hårda grindar (blockerar): pris > 50V EMA, sen cykel > 85:e percentilen,
# regimen AV. Allt annat är poäng 0–100 med vikterna nedan (summa 100),
# minus avdrag för aktiva flaggor. Trösklarna är VAL — de syns i RULES.
SETUP_WEIGHTS = {
    "ema_cross":   15,   # 20D EMA > 50D EMA
    "rs":          20,   # relativ styrka vs sektor-ETF, graderad −RS_FULL … +RS_FULL
    "higher_lows": 15,   # stigande bottnar 6V: 2 = halva, ≥3 = full
    "pullback":    20,   # avstånd till 20D EMA: 0 % = full, ≥ PULLBACK_MAX_PCT = 0
    "rsi":         15,   # RSI(14): ≤ RSI_FULL = full, ≥ RSI_ZERO = 0
    "macd":         5,   # MACD-histogram stigande botten
    "volume":       4,   # volym ≥ 20D-snitt
    "candle":       3,   # bullish candle
    "atr_falling":  3,   # ATR faller i rekylen
}
RS_FULL_PCT      = 5.0    # RS ≥ +5 % → full poäng; ≤ −5 % → 0 (binärt tidigare)
HIGHER_LOWS_OK   = 2      # 2 stigande bottnar = halva poängen, HIGHER_LOWS_MIN (3) = full
PULLBACK_MAX_PCT = 6.0    # graderat 0–6 % (hård gräns 3 % tidigare)
RSI_FULL         = 40.0   # full poäng under 40 …
RSI_ZERO         = 55.0   # … noll vid 55 (hård gräns 45 tidigare)
PENALTY_ATR_SURGE = 15.0  # ATR-surge: avdrag, inte stopp
PENALTY_DXY_SURGE = 10.0  # DXY-rally: avdrag, inte stopp
VERDICT_BUY_MIN   = 70.0  # ≥ 70 → KÖPLÄGE
VERDICT_WATCH_MIN = 50.0  # 50–70 → BEVAKA, annars AVVAKTA
SETUP_KOP, SETUP_BEVAKA, SETUP_AVVAKTA = "KÖPLÄGE", "BEVAKA", "AVVAKTA"

# ── Regimverdikt per komplex: netto i stället för "räkna gröna" ─────────────
# Grön +1, gul 0, röd −1, DATA_GAP 0 (visas som varning). Netto ≥ REGIME_PA_MIN
# = PÅ, netto ≥ REGIME_SELEKTIV_MIN = SELEKTIV, annars AV. Minst
# REGIME_AV_RED_MIN röda = AV oavsett netto. Gul straffades som röd tidigare.
REGIME_PILLAR_POINTS = {"GREEN": 1, "AMBER": 0, "RED": -1, "DATA_GAP": 0}
REGIME_PA_MIN        = 3     # t.ex. 3 gröna + 2 gula, eller 4 gröna + 1 röd
REGIME_SELEKTIV_MIN  = 0     # 0–2: t.ex. 2 gröna + 3 gula, eller 3 gröna + 1 gul + 1 röd
REGIME_AV_RED_MIN    = 2     # två röda pelare = AV oavsett

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

# ── Pre-filter constants ──────────────────────────────────────────────────────
PREFILTER_MIN_TURNOVER = 5_000_000  # avg daily turnover (close×vol) in local currency
PREFILTER_BATCH_SIZE   = 50         # tickers per yf.download() batch call
PREFILTER_PERIOD       = "1y"       # download period for pre-filter (gives ~252 bars)

# ── Globalt universum (Börsdata /instruments/global) ──────────────────────────
# USA (NYSE, Nasdaq — inte OTC), Kanada (Toronto, TSX Venture, CSE) och
# Australien (ASX), råvarubranscher, börsvärde ≥ GLOBAL_MIN_MCAP_MUSD.
GLOBAL_COUNTRIES      = ("usa", "kanada", "canada", "australien", "australia")
GLOBAL_EXCLUDE_LISTS  = ("otc",)
GLOBAL_MIN_MCAP_MUSD  = 300.0       # swingbart: mindre bolag faller oftast i omsättningsfiltret ändå
# Börsdatas bransch-id: råvarugrindens (olja/gas/kol/uran/gruv) + skog 21 och
# jordbruk 60, som Norden-filtret också tar med (EMBER har agri- och skogsteman).
GLOBAL_EXTRA_BRANCH_IDS = (21, 60)

# ── External data ─────────────────────────────────────────────────────────────
FRED_T10Y2Y_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
FRED_TIMEOUT    = 20   # seconds (increased; disk cache in fred_cache.py handles daily data)

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

# Tema → råvarukomplex (ember/regime.py). EN mappning i stället för två
# handskrivna tickerkartor som sa olika saker om 16 av 114 tickers (uran låg
# under "agri", koppar under ädelmetaller). Uran är energi — det är ett
# bränsle, och komplexets pelare (olja, gas) säger mer om uran än DBA och
# SPY gör. Sällsynta jordartsmetaller är basmetaller i den här indelningen.
THEME_TO_COMPLEX: dict[str, str] = {
    "guld": "adelmetaller", "silver": "adelmetaller",
    "koppar": "basmetaller", "sallsynta": "basmetaller",
    "uran": "energi", "olja": "energi", "naturgas": "energi", "kol": "energi",
    "agri": "agri",
}

# ── Råvara i arket (confidence.commodities-nyckel) → komplex ─────────────────
# Så ett bolag i Durrett-/Wolf Asymmetry-arket får sitt komplex utan att stå
# i TICKER_THEME_MAP: VISC.ST med råvara copper → basmetaller.
COMMODITY_TO_COMPLEX: dict[str, str] = {
    "uranium": "energi", "oil_gas": "energi", "natural_gas": "energi", "coal": "energi",
    "gold": "adelmetaller", "silver": "adelmetaller", "platinum": "adelmetaller",
    "palladium": "adelmetaller", "diamonds": "adelmetaller",
    "copper": "basmetaller", "lithium": "basmetaller", "rare_earth": "basmetaller",
    "nickel": "basmetaller", "cobalt": "basmetaller", "graphite": "basmetaller", "tin": "basmetaller",
    "zinc": "basmetaller", "iron_ore": "basmetaller", "aluminum": "basmetaller", "steel": "basmetaller",
    "potash": "agri", "phosphate": "agri", "agri": "agri",
}
# Råvara i arket → EMBER-tema (sektor-ETF och cykelfas följer temat). Basmetaller
# utan egen ETF mäts mot COPX, sällsynta/batterimetaller mot REMX.
COMMODITY_TO_THEME: dict[str, str] = {
    "uranium": "uran", "oil_gas": "olja", "natural_gas": "naturgas", "coal": "kol",
    "gold": "guld", "silver": "silver", "platinum": "guld", "palladium": "guld", "diamonds": "guld",
    "copper": "koppar", "nickel": "koppar", "zinc": "koppar", "tin": "koppar", "iron_ore": "koppar",
    "aluminum": "koppar", "steel": "koppar",
    "lithium": "sallsynta", "rare_earth": "sallsynta", "cobalt": "sallsynta", "graphite": "sallsynta",
    "potash": "agri", "phosphate": "agri", "agri": "agri",
}
# Registrets sektortext (fri text i Holdings) → tema. Första träffen vinner.
SECTOR_KEYWORD_THEME: tuple = (
    ("uran", "uran"), ("uranium", "uran"),
    ("silver", "silver"),
    ("guld", "guld"), ("gold", "guld"), ("ädelmetall", "guld"), ("precious", "guld"),
    ("koppar", "koppar"), ("copper", "koppar"), ("basmetall", "koppar"), ("nickel", "koppar"), ("zink", "koppar"),
    ("litium", "sallsynta"), ("lithium", "sallsynta"), ("sällsynta", "sallsynta"), ("rare", "sallsynta"),
    ("batteri", "sallsynta"),
    ("naturgas", "naturgas"), ("gas", "naturgas"), ("lng", "naturgas"),
    ("olja", "olja"), ("oil", "olja"), ("energi", "olja"), ("energy", "olja"), ("offshore", "olja"),
    ("oljeservice", "olja"), ("petroleum", "olja"),
    ("kol", "kol"), ("coal", "kol"),
    ("agri", "agri"), ("jordbruk", "agri"), ("potash", "agri"), ("gödsel", "agri"), ("fertilizer", "agri"),
)
# Yahoo-bransch (sector/industry/namn, engelska) → tema. Läses automatiskt för
# tickers som inte finns i temakartan, arket eller registret. Första träffen vinner.
INDUSTRY_KEYWORD_THEME: tuple = (
    ("uranium", "uran"),
    ("silver", "silver"),
    ("gold", "guld"), ("precious", "guld"), ("platinum", "guld"), ("palladium", "guld"),
    ("copper", "koppar"), ("industrial metals", "koppar"), ("nickel", "koppar"), ("zinc", "koppar"),
    ("steel", "koppar"), ("aluminum", "koppar"), ("iron", "koppar"),
    ("lithium", "sallsynta"), ("rare earth", "sallsynta"), ("cobalt", "sallsynta"), ("graphite", "sallsynta"),
    ("coal", "kol"),
    ("natural gas", "naturgas"), ("lng", "naturgas"),
    ("oil", "olja"), ("petroleum", "olja"), ("offshore", "olja"), ("drilling", "olja"), ("energy", "olja"),
    ("agricultural", "agri"), ("fertilizer", "agri"), ("potash", "agri"), ("farm", "agri"),
)
# Valt komplex utan råvara → temat som bär komplexet (för sektor-ETF och cykel).
COMPLEX_DEFAULT_THEME: dict[str, str] = {"energi": "olja", "adelmetaller": "guld",
                                         "basmetaller": "koppar", "agri": "agri"}
COMPLEX_LABEL: dict[str, str] = {"energi": "ENERGI", "adelmetaller": "ÄDELMETALLER",
                                 "basmetaller": "BASMETALLER", "agri": "AGRI & ÖVRIGT"}
EMBER_STORE = "ember"           # data/ember.json: {"complex_overrides": {TICKER: komplex}}

# ── Ticker → theme key map ────────────────────────────────────────────────────
# Covers all universe members so cycle phase is never DATA_GAP for known tickers.
_TICKER_THEME_RAW: dict[str, str] = {
    # ── Guld (GDX / GDXJ names + key majors) ──────────────────────────────
    "GLD":  "guld", "GDX": "guld", "GDXJ": "guld",
    "NEM": "guld", "GOLD": "guld", "AEM": "guld", "WPM": "guld",
    "KGC": "guld", "AGI": "guld", "AU":  "guld", "GFI": "guld",
    "BTG": "guld", "EGO": "guld", "SSRM":"guld", "OR":  "guld",
    "SA":  "guld", "HMY": "guld", "DRD": "guld", "NGD": "guld",
    "MUX": "guld",
    # Canada gold
    "ABX.TO": "guld", "K.TO": "guld", "AGI.TO": "guld",
    "BTO.TO": "guld", "EDV.TO": "guld", "WPM.TO": "guld", "FNV.TO": "guld",
    # UK diversified miners — järnmalm och koppar, inte guld
    "RIO.L": "koppar", "BHP.L": "koppar",
    # ── Silver (SIL / SILJ names) ──────────────────────────────────────────
    "SLV": "silver", "SIL": "silver", "SILJ": "silver",
    "PAAS": "silver", "HL": "silver", "AG": "silver",
    "CDE": "silver", "FSM": "silver", "EXK": "silver",
    "MAG": "silver", "GPL": "silver", "SVM": "silver", "ASM": "silver",
    # Canada silver streaming
    "ERO.TO": "koppar",   # Ero Copper, not silver
    # UK silver major
    "FRES.L": "silver",
    # ── Koppar (COPX names) ────────────────────────────────────────────────
    "COPX": "koppar", "FCX": "koppar", "SCCO": "koppar", "TECK": "koppar",
    "PICK": "koppar", "XME": "koppar", "LIT": "koppar",
    # Canada copper
    "FM.TO": "koppar", "LUN.TO": "koppar",
    # UK copper majors
    "AAL.L": "koppar", "ANTO.L": "koppar", "GLEN.L": "koppar",
    # ── Uran ──────────────────────────────────────────────────────────────
    "URA": "uran", "URNM": "uran",
    "CCJ": "uran", "NXE": "uran", "DNN": "uran",
    "UUUU": "uran", "LEU": "uran", "UEC": "uran",
    # Canada uranium
    "CCO.TO": "uran", "DML.TO": "uran", "NXE.TO": "uran",
    # ── Olja ──────────────────────────────────────────────────────────────
    "XLE": "olja", "XOP": "olja", "USO": "olja",
    "XOM": "olja", "CVX": "olja", "COP": "olja", "EOG": "olja",
    "SLB": "olja", "MPC": "olja", "VLO": "olja", "PSX": "olja",
    "OXY": "olja", "HAL": "olja", "DVN": "olja", "BKR": "olja",
    "FANG": "olja", "APA": "olja", "MRO": "olja", "SHEL": "olja",
    # Norway energy
    "EQNR.OL": "olja", "AKRBP.OL": "olja", "VAR.OL": "olja", "TGS.OL": "olja",
    # Canada oil
    "SU.TO": "olja", "CNQ.TO": "olja", "CVE.TO": "olja", "IMO.TO": "olja",
    "WCP.TO": "olja", "ARX.TO": "olja", "BTE.TO": "olja", "TOU.TO": "olja",
    # UK oil
    "BP.L": "olja", "SHEL.L": "olja",
    # ── Naturgas ──────────────────────────────────────────────────────────
    "UNG": "naturgas",
    # ── Kol ───────────────────────────────────────────────────────────────
    "BTU": "kol", "ARCH": "kol", "CEIX": "kol", "AMR": "kol",
    # ── Agri ──────────────────────────────────────────────────────────────
    "DBA": "agri", "MOS": "agri", "NTR": "agri",
    "CF": "agri", "UAN": "agri", "ADM": "agri",
    "NTR.TO": "agri",
    # ── Sällsynta jordartsmetaller ─────────────────────────────────────────
    "REMX": "sallsynta", "MP": "sallsynta",
}

# Omdöpta tickers byts (GOLD → B), uppköpta släpps (MRO, MAG, ARCH, CEIX…).
try:
    from dead_tickers import alive_map as _alive_map
    TICKER_THEME_MAP: dict[str, str] = _alive_map(_TICKER_THEME_RAW)
except Exception:  # pragma: no cover
    TICKER_THEME_MAP = dict(_TICKER_THEME_RAW)

assert set(TICKER_THEME_MAP.values()) <= set(THEME_TO_COMPLEX), "tema utan komplex"

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
