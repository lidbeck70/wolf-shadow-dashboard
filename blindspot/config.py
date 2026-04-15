"""
config.py — Configuration for Odin's Blindspot Index.
Tickers, weights, necessity map, and Nordic Gold palette.
"""

BLINDSPOT_TICKERS = [
    # Uranium
    "CCJ", "UEC", "DNN", "NXE", "UUUU", "LEU",
    # Oil & Gas
    "OXY", "DVN", "EQNR.OL", "CVE", "SU", "PBR",
    # Copper & Metals
    "FCX", "SCCO", "TECK", "BOL.ST",
    # Steel
    "CLF", "X", "NUE", "SSAB-A.ST",
    # Fertilizers
    "MOS", "NTR", "CF", "YARA.OL",
    # Shipping
    "GOGL", "SBLK", "DAC",
    # Power
    "CEG", "VST", "NRG",
    # Defense
    "LMT", "RTX", "NOC", "SAAB-B.ST",
    # Semiconductors
    "INTC", "MU", "TXN", "ASML",
    # Infrastructure
    "VMC", "MLM", "URI",
    # Agriculture
    "ADM", "BG",
    # Nordic industrials
    "VOLV-B.ST", "NHY.OL", "AKER.OL",
]

BASE_WEIGHTS = {
    "hat": 0.35,
    "necessity": 0.25,
    "strength": 0.25,
    "catalyst": 0.15,
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
