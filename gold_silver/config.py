"""
gold_silver/config.py — alla tal för Guld/Silver-fliken på ett ställe.

Referensvärdena (geologisk, produktion, ovanjord) är inte marknadsdata. De
står här med källa, datum och typ, så de syns som det de är — och kan
uppdateras när en ny rapport kommer. Inget av dem är ett "fair value".
"""

# Marknadspriser: terminerna (USD/oz). ETF:er (GLD/SLV) har andra enheter
# per andel och ger en skalad kvot — används inte som reserv för nivåer.
GOLD_TICKER = "GC=F"
SILVER_TICKER = "SI=F"
HISTORY_PERIOD = "max"             # Yahoo har terminerna från ~2000

# Geologisk referens. Skorpans halter: silver ~0,075 ppm, guld ~0,004 ppm
# (CRC Handbook of Chemistry and Physics) → ~19. Andra uppskattningar av
# skorpans sammansättning ger andra kvoter — värdet är ungefärligt.
REFERENCE_RATIO = 19.0

# Revalveringstabellen: dagens kvot visas alltid först, sedan de här.
TARGET_RATIOS = (60.0, 50.0, 40.0, 30.0, 25.0, 20.0, 19.0)

# Matrisen guldpris × kvot
GOLD_GRID = (4000.0, 5000.0, 6000.0, 7000.0, 8000.0)
MATRIX_RATIOS = (60.0, 50.0, 40.0, 30.0, 25.0, 20.0, 19.0)

# Historiska perioder (år; None = allt som finns)
PERIODS = (("1 år", 1), ("5 år", 5), ("10 år", 10), ("20 år", 20), ("Max", None))
POSITION_PERIOD_YEARS = 10         # "historisk position" mäts mot 10 år
NEAR_MEDIAN_PCT = 10.0             # inom ±10 % av medianen = nära

# Guidens zoner (strategy_rules_masterguide, commodity_book "silver"):
# över 85 aktiverar Durretts silvervariant, under 50 = sencykliskt, trimma.
GUIDE_ACCUMULATE = 85.0
GUIDE_LATE = 50.0

# Silverbolagsscenarier (PR B): (namn, målkvot; None = dagens kvot)
MINER_SCENARIOS = (("BASE", None), ("RATIO COMPRESSION", 50.0), ("STRONG SILVER", 40.0),
                   ("SILVER BULL", 30.0), ("REFERENCE SCENARIO", 19.0))

# Referensdata med proveniens. kind: ACTUAL | ESTIMATE; confidence: hög/medel/låg.
# Uppdatera värde, källa och datum när en ny rapport kommer.
REFERENCES = {
    "geological": {
        "label": "Geologisk kvot (skorpan)", "value": 19.0, "unit": ": 1 (massa)",
        "source": "CRC Handbook of Chemistry and Physics — skorpans halter, silver ~0,075 ppm, guld ~0,004 ppm",
        "date": "", "source_type": "secondary", "kind": "ESTIMATE", "confidence": "låg",
        "note": "Hur mycket silver det ungefär finns per guld i jordskorpan. Uppskattningarna varierar "
                "mellan källor. Inte ett marknadsjämviktsvärde.",
    },
    "production": {
        "label": "Gruvproduktion", "gold_t": 3300.0, "silver_t": 25000.0,
        "source": "USGS Mineral Commodity Summaries 2025 (världens gruvproduktion 2024, uppskattning)",
        "date": "2025-01", "source_type": "secondary", "kind": "ESTIMATE", "confidence": "medel",
        "note": "Silver ut ur gruvorna per guld, i vikt (samma kvot i troy ounce). Mycket av silvret är "
                "biprodukt från bly, zink och koppar.",
    },
    "above_ground": {
        "label": "Ovan jord", "gold_t": 216000.0, "silver_t": None,
        "source": "World Gold Council — ovanjordslager guld, slutet av 2024 (~216 000 t). Silver: ingen "
                  "tillförlitlig samlad siffra — uppskattningarna går isär kraftigt.",
        "date": "2025", "source_type": "secondary", "kind": "ESTIMATE", "confidence": "låg",
        "note": "Partial data — bara guldsidan har en etablerad siffra.",
    },
}

# Vad marknadskvoten faktiskt beror på (spec §26) — visas ordagrant.
MARKET_DRIVERS = ("utbud och efterfrågan", "ovanjordslager", "industriell efterfrågan (solceller, elektronik)",
                  "investeringsefterfrågan", "återvinning", "gruvutbud", "biproduktion (bly, zink, koppar)",
                  "monetär efterfrågan", "marknadsstruktur")
