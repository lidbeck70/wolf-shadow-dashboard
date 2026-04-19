SECTOR_ETF_NAMES = {
    "XLE": "XLE Energy",
    "XLB": "XLB Materials",
    "XLF": "XLF Financials",
    "XLK": "XLK Technology",
    "XLV": "XLV Healthcare",
    "XLI": "XLI Industrials",
    "XLY": "XLY Consumer Disc",
    "XLP": "XLP Consumer Staples",
    "XLRE": "XLRE Real Estate",
    "XLU": "XLU Utilities",
    "XLC": "XLC Communication",
}

SECTOR_ETF_LIST = list(SECTOR_ETF_NAMES.values())


def etf_from_display(display_name: str) -> str:
    """Convert 'XLE Energy' back to 'XLE'."""
    return display_name.split(" ")[0] if display_name else "XLE"


# Backtest parameter presets (21 presets: US sectors, Nordic exchanges, individual stocks)
PRESET_PARAMS_BT = {
    # US Sector ETFs
    "XLE":  {"atr_mult": 2.1, "adx_thresh": 5,  "tp1_r": 1.75, "tp1_pct": 0.10, "tp2_r": 5.5,  "tp2_pct": 0.10, "core_pct": 0.40},
    "XLB":  {"atr_mult": 1.5, "adx_thresh": 21, "tp1_r": 2.25, "tp1_pct": 0.10, "tp2_r": 6.0,  "tp2_pct": 0.15, "core_pct": 0.55},
    "XLF":  {"atr_mult": 3.2, "adx_thresh": 28, "tp1_r": 2.5,  "tp1_pct": 0.25, "tp2_r": 3.5,  "tp2_pct": 0.15, "core_pct": 0.55},
    "XLK":  {"atr_mult": 2.3, "adx_thresh": 3,  "tp1_r": 2.5,  "tp1_pct": 0.20, "tp2_r": 5.25, "tp2_pct": 0.05, "core_pct": 0.70},
    "XLV":  {"atr_mult": 2.0, "adx_thresh": 2,  "tp1_r": 4.0,  "tp1_pct": 0.05, "tp2_r": 4.5,  "tp2_pct": 0.20, "core_pct": 0.40},
    "XLI":  {"atr_mult": 1.8, "adx_thresh": 6,  "tp1_r": 3.75, "tp1_pct": 0.10, "tp2_r": 4.0,  "tp2_pct": 0.10, "core_pct": 0.60},
    "XLY":  {"atr_mult": 2.0, "adx_thresh": 2,  "tp1_r": 3.0,  "tp1_pct": 0.20, "tp2_r": 3.75, "tp2_pct": 0.25, "core_pct": 0.55},
    "XLP":  {"atr_mult": 2.3, "adx_thresh": 0,  "tp1_r": 2.25, "tp1_pct": 0.05, "tp2_r": 5.75, "tp2_pct": 0.25, "core_pct": 0.50},
    "XLRE": {"atr_mult": 2.5, "adx_thresh": 1,  "tp1_r": 3.5,  "tp1_pct": 0.05, "tp2_r": 5.25, "tp2_pct": 0.20, "core_pct": 0.40},
    "XLU":  {"atr_mult": 1.1, "adx_thresh": 13, "tp1_r": 3.5,  "tp1_pct": 0.15, "tp2_r": 4.5,  "tp2_pct": 0.20, "core_pct": 0.45},
    "XLC":  {"atr_mult": 3.1, "adx_thresh": 14, "tp1_r": 3.25, "tp1_pct": 0.25, "tp2_r": 3.5,  "tp2_pct": 0.05, "core_pct": 0.60},
    # Nordic Exchanges
    "OMX Stockholm":  {"atr_mult": 2.0, "adx_thresh": 18, "tp1_r": 3.15, "tp1_pct": 0.20, "tp2_r": 4.8,  "tp2_pct": 0.22, "core_pct": 0.46},
    "OMX Copenhagen": {"atr_mult": 2.1, "adx_thresh": 12, "tp1_r": 2.2,  "tp1_pct": 0.12, "tp2_r": 5.15, "tp2_pct": 0.12, "core_pct": 0.48},
    "Oslo OSEBX":     {"atr_mult": 2.3, "adx_thresh": 20, "tp1_r": 3.55, "tp1_pct": 0.16, "tp2_r": 4.8,  "tp2_pct": 0.19, "core_pct": 0.49},
    "OMX Helsinki":   {"atr_mult": 2.1, "adx_thresh": 11, "tp1_r": 3.05, "tp1_pct": 0.21, "tp2_r": 5.55, "tp2_pct": 0.15, "core_pct": 0.54},
    # Individual stocks
    "OXY":  {"atr_mult": 2.8, "adx_thresh": 27, "tp1_r": 3.0,  "tp1_pct": 0.20, "tp2_r": 5.5,  "tp2_pct": 0.25, "core_pct": 0.70},
    "GOLD": {"atr_mult": 1.5, "adx_thresh": 16, "tp1_r": 1.75, "tp1_pct": 0.20, "tp2_r": 5.75, "tp2_pct": 0.05, "core_pct": 0.50},
    "NEM":  {"atr_mult": 3.1, "adx_thresh": 17, "tp1_r": 3.0,  "tp1_pct": 0.05, "tp2_r": 4.25, "tp2_pct": 0.25, "core_pct": 0.60},
    "XOM":  {"atr_mult": 2.7, "adx_thresh": 27, "tp1_r": 3.5,  "tp1_pct": 0.10, "tp2_r": 5.5,  "tp2_pct": 0.10, "core_pct": 0.70},
    "GLD":  {"atr_mult": 2.6, "adx_thresh": 7,  "tp1_r": 1.75, "tp1_pct": 0.10, "tp2_r": 5.0,  "tp2_pct": 0.20, "core_pct": 0.60},
    # Universal fallback
    "Universal": {"atr_mult": 2.5, "adx_thresh": 19, "tp1_r": 2.6, "tp1_pct": 0.13, "tp2_r": 5.2, "tp2_pct": 0.17, "core_pct": 0.62},
}

PRESET_LABELS = [
    "Auto-detect", "Universal",
    "XLE Energy", "XLB Materials", "XLF Financials", "XLK Technology",
    "XLV Healthcare", "XLI Industrials", "XLY Consumer Disc",
    "XLP Consumer Staples", "XLRE Real Estate", "XLU Utilities", "XLC Communication",
    "OMX Stockholm", "OMX Copenhagen", "Oslo OSEBX", "OMX Helsinki",
    "OXY", "GOLD", "NEM", "XOM", "GLD",
]


def resolve_preset_key(preset_label: str, ticker: str) -> str:
    """Map a preset label + ticker to the PRESET_PARAMS_BT key."""
    if preset_label == "Auto-detect":
        if ticker in PRESET_PARAMS_BT:
            return ticker
        elif ticker.endswith(".ST"):
            return "OMX Stockholm"
        elif ticker.endswith(".OL"):
            return "Oslo OSEBX"
        elif ticker.endswith(".CO"):
            return "OMX Copenhagen"
        elif ticker.endswith(".HE"):
            return "OMX Helsinki"
        else:
            return "Universal"
    # Strip trailing ETF name suffix like "XLE Energy" → "XLE"
    key = preset_label.split(" ")[0] if " " in preset_label else preset_label
    return key if key in PRESET_PARAMS_BT else preset_label
