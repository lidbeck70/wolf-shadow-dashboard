"""
gold_ratios/config.py — alla tal för Guldkvoter-fliken på ett ställe.

Samma motor som Guld/Silver (gold_silver.engine, oförändrad) men för fler
par. Det finns ingen geologisk referens för de här paren — referensen är
kvotens egen historiska median, och målkvoterna är dess egna percentiler.
Inget av det är ett "fair value".

Två slags par:
  råvara  kvot = guld ÷ råvara   → priset på råvaran räknas ut ur guldet
  index   kvot = index ÷ guld    → guldpriset räknas ut ur indexet
I båda fallen hålls täljaren fast och nämnaren räknas ut: nämnare = täljare / kvot.
"""

COMMODITY, INDEX = "råvara", "index"

GOLD_TICKER = "GC=F"
GOLD_BD_ID = 21031                 # Börsdata Nymex "Gold" (XAU) — reserv
HISTORY_PERIOD = "max"

# Paren. ticker = Yahoo; bd_id = Börsdatas Nymex-instrument (reserv när Yahoo
# inte levererar; id ur borsdata_probe 2026-10-01, ~20 års dagliga priser);
# scale = Yahoo-enhet → visad enhet (vete och kaffe noteras i US-cent);
# theme = asymmetry-temat för gruvbolagsdelen (None = ingen).
PAIRS = (
    {"key": "platina", "label": "Platina", "emoji": "⚪", "kind": COMMODITY, "ticker": "PL=F", "bd_id": 21033,
     "unit": "USD/oz", "scale": 1.0, "theme": "platina"},
    {"key": "palladium", "label": "Palladium", "emoji": "⚪", "kind": COMMODITY, "ticker": "PA=F",
     "bd_id": 21034, "unit": "USD/oz", "scale": 1.0, "theme": "palladium"},
    {"key": "koppar", "label": "Koppar", "emoji": "🟠", "kind": COMMODITY, "ticker": "HG=F", "bd_id": 21035,
     "unit": "USD/lb", "scale": 1.0, "theme": "koppar"},
    {"key": "olja", "label": "Olja (WTI)", "emoji": "🛢️", "kind": COMMODITY, "ticker": "CL=F", "bd_id": 21047,
     "unit": "USD/fat", "scale": 1.0, "theme": "olja"},
    {"key": "brent", "label": "Olja (Brent)", "emoji": "🛢️", "kind": COMMODITY, "ticker": "BZ=F",
     "bd_id": 21046, "unit": "USD/fat", "scale": 1.0, "theme": None},
    {"key": "naturgas", "label": "Naturgas", "emoji": "🔥", "kind": COMMODITY, "ticker": "NG=F", "bd_id": None,
     "unit": "USD/MMBtu", "scale": 1.0, "theme": "naturgas"},
    {"key": "vete", "label": "Vete", "emoji": "🌾", "kind": COMMODITY, "ticker": "ZW=F", "bd_id": None,
     "unit": "USD/bu", "scale": 0.01, "theme": "vete"},
    {"key": "kaffe", "label": "Kaffe", "emoji": "☕", "kind": COMMODITY, "ticker": "KC=F", "bd_id": None,
     "unit": "USD/lb", "scale": 0.01, "theme": "kaffe"},
    {"key": "kakao", "label": "Kakao", "emoji": "🍫", "kind": COMMODITY, "ticker": "CC=F", "bd_id": None,
     "unit": "USD/t", "scale": 1.0, "theme": "kakao"},
    {"key": "dow", "label": "Dow Jones", "emoji": "🏛️", "kind": INDEX, "ticker": "^DJI", "bd_id": None,
     "unit": "punkter", "scale": 1.0, "theme": None},
    {"key": "spx", "label": "S&P 500", "emoji": "📈", "kind": INDEX, "ticker": "^GSPC", "bd_id": None,
     "unit": "punkter", "scale": 1.0, "theme": None},
)
PAIR_BY_KEY = {p["key"]: p for p in PAIRS}
DEFAULT_PAIR = "platina"

# Målkvoter och matrisens kvoter: kvotens egna percentiler över hela historiken
TARGET_QUANTILES = (("P10", "p10"), ("P25", "p25"), ("Median", "median"), ("P75", "p75"), ("P90", "p90"))
REFERENCE_PERIOD = None            # referensen = medianen över allt som finns (None = Max)
POSITION_PERIOD_YEARS = 10         # "historisk position" mäts mot 10 år, som i Guld/Silver

# Matrisen: täljaren (guld resp. index) som multiplar av dagens värde
ANCHOR_GRID = (0.75, 1.0, 1.25, 1.5, 2.0)

# Gruvbolag per kvotscenario (PR B): (namn, percentilnyckel; None = dagens kvot).
# För guld ÷ råvara betyder lägre kvot dyrare råvara: P90 = svag råvara, P10 = stark.
MINER_SCENARIOS = (("BASE", None), ("SVAG (P90)", "p90"), ("MEDIAN", "median"),
                   ("STARK (P25)", "p25"), ("MYCKET STARK (P10)", "p10"))

# Exempel i tickerfältet per tema (bara platshållartext)
MINER_EXAMPLES = {"platina": "SBSW, IMP.JO, PLG", "palladium": "SBSW, NILSY", "koppar": "FCX, BOL.ST, LUN.TO",
                  "olja": "EQNR.OL, XOM, VAR.OL", "naturgas": "EQT, AR, TOU.TO", "vete": "ADM, BG",
                  "kaffe": "SJM, NSRGY", "kakao": "BARN.SW, HSY"}
