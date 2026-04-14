"""
config.py — Configuration for the Retail Sentiment Engine.
"""

DEFAULT_TICKERS = [
    # US retail favorites
    "TSLA", "NVDA", "AAPL", "AMD", "PLTR", "GME", "AMC",
    "SOFI", "RIVN", "COIN", "HOOD", "SMCI", "MARA", "RIOT",
    "CCJ", "UEC", "GDX", "XLE", "XOP",
    # Nordic
    "EQNR.OL", "VOLV-B.ST", "EVO.ST", "NOVO-B.CO", "SSAB-A.ST",
    "NHY.OL", "BOL.ST", "AKRBP.OL",
]

BASE_WEIGHTS = {
    "reddit": 0.25,
    "twitter": 0.20,
    "retail_flow": 0.30,
    "yahoo": 0.10,
    "hype_overlap": 0.15,
}

CACHE_TTL = 900  # 15 minutes

# Nordic Gold palette
BG = "#0c0c12"
BG2 = "#14141e"
GOLD = "#c9a84c"
BRONZE = "#8b7340"
GREEN = "#2d8a4e"
RED = "#c44545"
TEXT = "#e8e4dc"
DIM = "#8a8578"


def detect_market(ticker: str) -> str:
    """Detect whether a ticker is Nordic or US based on suffix."""
    suffixes = (".ST", ".OL", ".CO", ".HE")
    return "NORDIC" if any(ticker.upper().endswith(s) for s in suffixes) else "US"
