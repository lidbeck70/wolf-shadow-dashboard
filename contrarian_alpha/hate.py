"""
hate.py — Hat Score (0-100) for Contrarian Alpha Screener.

Measures how hated / neglected / sold-off a stock is.
Higher score = more contrarian opportunity. Threshold: HAT_THRESHOLD = 45.

7 kärnkomponenter (summa 100) — alla nåbara med Börsdata/yfinance/prisserien:
  1. Pris vs SMA200            max 15p  (under SMA200 = institutionellt övergiven)
  2. Nära 52v-lägsta           max 12p  (max smärta)
  3. Cykelposition             max 15p  (under eget flerårssnitt = trough)
  4. Blankning                 max 15p  (FI-registret via Börsdata / yfinance för US)
  5. Värderingsdepression      max 15p  (botten av egen KPI-historik)
  6. Volymtorka                max 13p  (bevakningen har dött)
  7. Sektorrelativ svaghet     max 15p  (svagare än egna sektorn = utflöde)

Poängen normaliseras mot NÅBART max (komponenter med riktig data), med
skyddsräcket att minst 50p av rymden måste vara mätbar. EODHD/StockTwits
(analytiker/sentiment) är en frivillig bonus, max +8 — aldrig ett krav.

Adapted from blindspot/scoring/hat.py + retail_sentiment/sources/twitter.py.

Value Trap flag:
  Hat Score > 85 AND Strength Score < 50 → "POTENTIAL_VALUE_TRAP"
  (hated + financially weak = falling knife, not contrarian gem)

Input dicts (all optional, use None/empty dict when unavailable):

  price_data:
    close          float  Current price
    sma200         float  200-day SMA
    high_52w       float  52-week high
    low_52w        float  52-week low
    avg_price_5y   float  Average closing price over 5 years (optional; from extended fetch)

  sentiment_data:  (from retail_sentiment/sources/twitter.fetch_ticker_sentiment)
    message_count  int    Number of StockTwits messages in feed
    bear_ratio     float  0-1 fraction of tagged-bearish messages
    bull_ratio     float  0-1 fraction of tagged-bullish messages
    watchlist_count int   Symbol watchlist count
    confidence     float  0-1 data quality

  analyst_data:    (from EODHD upgrades-downgrades endpoint or yfinance)
    downgrades_90d int    Downgrade actions in last 90 days
    upgrades_90d   int    Upgrade actions in last 90 days
    consensus      str    'Strong Buy'|'Buy'|'Hold'|'Underperform'|'Sell'

  short_data:      (from EODHD fundamentals SharesStats or shorts endpoint)
    short_float_pct float  % of float sold short  (e.g. 8.5 = 8.5%)
    days_to_cover   float  Short interest / avg daily volume

  sector_data:
    sector_vs_market_3m  float  Sector ETF perf vs SPY over 3 months (pp, e.g. -12.5)
    sector_vs_market_6m  float  Sector ETF perf vs SPY over 6 months (pp)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ─── Pipeline constants ───────────────────────────────────────────────────────

HAT_THRESHOLD = 45              # Hat Score >= 45 required to proceed
HAT_COMPOSITE_WEIGHT = 0.40     # 40% of Composite Score (placeholder; set in composite.py)

VALUE_TRAP_HAT_MIN      = 85    # Hat score above this...
VALUE_TRAP_STRENGTH_MAX = 50    # ...combined with strength below this → Value Trap

# ─── Component max points (must sum to 100) ───────────────────────────────────
# Omdesign: sju komponenter som ALLA går att nå med källor panelen redan har
# (Börsdata + yfinance + den egna prisserien). Tidigare krävde 58 av 100 poäng
# EODHD/StockTwits-data som aldrig hämtades, och saknad data gav "moderata
# defaultpoäng" — 21 fabricerade poäng av tröskelns 45, så grinden sorterade
# mest på brus. Nu ger saknad data 0 poäng och räknas BORT ur nåbart max i
# stället (se normaliseringen i calculate_hate_score).

_MAX_SMA200     = 15   # pris vs SMA200 — institutionellt övergiven
_MAX_52W_LOW    = 12   # nära 52v-lägsta — max smärta
_MAX_CYCLE      = 15   # under eget flerårssnitt — cykeltrough
_MAX_SHORT      = 15   # blankning — FI-registret (Börsdata) / yfinance för US
_MAX_VALUATION  = 15   # värderingsdepression — botten av egen KPI-historik
_MAX_VOLUME     = 13   # volymtorka — bevakningen har dött
_MAX_SECTOR     = 15   # svagare än egna sektorn — utflöde

# Frivillig förstärkning (EODHD/StockTwits när nyckeln finns) — begränsad
# bonus ovanpå kärnan, aldrig ett krav för att nå tröskeln.
_MAX_BONUS      = 8
_MAX_RETAIL_SIL =  5   # legacy-skalor för bonusdelen
_MAX_ANALYST    = 12
_MAX_BEAR_RATIO = 16

# ─── Result model ────────────────────────────────────────────────────────────

@dataclass
class HateResult:
    score: float
    breakdown: dict[str, float]     = field(default_factory=dict)
    flags: list[str]                = field(default_factory=list)
    confidence: float               = 1.0   # 0-1, fraction of components with real data

    @property
    def passes_threshold(self) -> bool:
        return self.score >= HAT_THRESHOLD

    @property
    def is_value_trap(self) -> bool:
        return "POTENTIAL_VALUE_TRAP" in self.flags


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ─── Component scorers ────────────────────────────────────────────────────────

def _score_sma200_gap(price_data: dict) -> tuple[float, bool]:
    """
    How far is the price below SMA200? (max 20p)
    Returns (points, has_real_data).
    0% below → 0p | 10% below → 10p | 20%+ below → 20p
    If price is above SMA200 → 0p (stock is in uptrend, not hated enough).
    """
    close  = price_data.get("close",  0.0)
    sma200 = price_data.get("sma200", 0.0)
    if sma200 <= 0 or close <= 0:
        return 0.0, False  # ingen data — utesluts ur nåbart max, inga låtsaspoäng
    pct_below = (sma200 - close) / sma200 * 100
    pts = _clamp(pct_below / 20.0 * _MAX_SMA200, 0.0, float(_MAX_SMA200))
    return pts, True


def _score_52w_low_proximity(price_data: dict) -> tuple[float, bool]:
    """
    How close is the price to the 52-week low? (max 15p)
    At 52w low → 15p | At mid-range → 0p | Above mid-range → 0p
    """
    close   = price_data.get("close",   0.0)
    high_52 = price_data.get("high_52w", 0.0)
    low_52  = price_data.get("low_52w",  0.0)
    if high_52 <= low_52 or close <= 0:
        return 0.0, False
    position = (close - low_52) / (high_52 - low_52)   # 0=at low, 1=at high
    pts = _clamp((0.5 - position) / 0.5 * _MAX_52W_LOW, 0.0, float(_MAX_52W_LOW))
    return pts, True


def _score_retail_silence(sentiment_data: dict | None) -> tuple[float, bool]:
    """
    How quiet is retail about this stock on StockTwits? (max 5p)
    Silence = forgotten = hated / irrelevant.
    0 messages → 5p | <5 → 4p | <20 → 2p | >=20 → 0p
    Weighted by confidence.
    """
    if not sentiment_data:
        return 0.0, False
    count = sentiment_data.get("message_count", 0) or 0
    conf  = sentiment_data.get("confidence", 0.0)
    if conf == 0.0:
        return 0.0, False
    if count == 0:
        raw = 5.0
    elif count < 5:
        raw = 4.0
    elif count < 20:
        raw = 2.0
    elif count < 50:
        raw = 1.0
    else:
        raw = 0.0
    return raw * min(1.0, conf + 0.1), True   # slight confidence boost for any data


def _score_analyst_downgrades(analyst_data: dict | None) -> tuple[float, bool]:
    """
    Analyst downgrade pressure over last 90 days. (max 15p)

    Logic:
    - Net downgrades (downgrades − upgrades) drives the base score
    - Consensus modifier adds/subtracts up to 3p
    - Cap at 15p

    Scoring:
      net=0  → 0p
      net=1  → 5p
      net=2  → 8p
      net=3  → 11p
      net=4+ → 14p  (before consensus modifier)

    Consensus modifier:
      'Strong Buy' → -3p  (contradicts hate signal)
      'Buy'        → -1p
      'Hold'       →  0p
      'Underperform' → +1p
      'Sell'       → +2p
    """
    if not analyst_data:
        return 0.0, False
    downs = analyst_data.get("downgrades_90d", 0) or 0
    ups   = analyst_data.get("upgrades_90d",   0) or 0
    net   = max(0, downs - ups)

    _tiers = {0: 0.0, 1: 5.0, 2: 8.0, 3: 11.0}
    base = _tiers.get(net, 14.0)

    consensus_modifiers = {
        "strong buy":    -3.0,
        "buy":           -1.0,
        "hold":           0.0,
        "underperform":   1.0,
        "sell":           2.0,
    }
    consensus = str(analyst_data.get("consensus", "hold")).lower()
    modifier = consensus_modifiers.get(consensus, 0.0)

    pts = _clamp(base + modifier, 0.0, float(_MAX_ANALYST))
    return pts, True


def _score_short_interest(short_data: dict | None) -> tuple[float, bool]:
    """
    Short interest as % of float (EODHD). (max 15p)
    High short % = active bet against stock = institutionally hated.

    <2%   → 0p  (negligible)
    2-5%  → 4p
    5-10% → 8p
    10-15%→ 11p
    15-20%→ 13p
    >20%  → 15p

    Källor: Börsdata /holdings/shorts (FI:s blankningsregister, hela
    universumet i ett anrop) för nordiska aktier; yfinance
    shortPercentOfFloat för US; EODHD kvar som frivillig väg. En nordisk
    aktie som SAKNAS i registret ligger under 0,5 %-golvet — motorn skickar
    då {"short_float_pct": 0.0}, vilket är RIKTIG data (0 p), inte saknad.
    """
    if not short_data:
        return 0.0, False
    pct = short_data.get("short_float_pct")
    if pct is None:
        return 0.0, False
    pct = float(pct)
    if pct < 2:    return 0.0,  True
    if pct < 5:    return 4.0,  True
    if pct < 10:   return 8.0,  True
    if pct < 15:   return 11.0, True
    if pct < 20:   return 13.0, True
    return float(_MAX_SHORT),    True


def _score_sector_outflow(sector_data: dict | None) -> tuple[float, bool]:
    """
    Relativ svaghet mot egna sektorn. (max 15p)

    Primär nyckel: stock_vs_sector_3m — aktiens 3-månadersavkastning minus
    sektormedianens, i procentenheter, räknad ur universumets EGNA prisdata
    (noll extra API-anrop). Negativt = svagare än sektorn = utflöde ur just
    den här aktien, inte bara branschen.

      >= 0 pe   → 0p
      till −5   → 4p
      till −12  → 8p
      till −20  → 12p
      < −20     → 15p

    Fallback: de gamla ETF-nycklarna (sector_vs_market_3m/6m) skalas till
    samma 15-poängsskala så EODHD-vägen fortfarande fungerar.
    """
    if not sector_data:
        return 0.0, False

    rel = sector_data.get("stock_vs_sector_3m")
    if rel is not None:
        rel = float(rel)
        if rel >= 0:      pts = 0.0
        elif rel > -5:    pts = 4.0
        elif rel > -12:   pts = 8.0
        elif rel > -20:   pts = 12.0
        else:             pts = 15.0
        return pts, True

    rel_3m = sector_data.get("sector_vs_market_3m")
    if rel_3m is None:
        return 0.0, False
    rel_3m = float(rel_3m)
    if rel_3m > 0:      base = 0.0
    elif rel_3m > -5:   base = 3.0
    elif rel_3m > -10:  base = 7.0
    elif rel_3m > -15:  base = 11.0
    else:               base = 13.0

    bonus = 0.0
    rel_6m = sector_data.get("sector_vs_market_6m")
    if rel_6m is not None and float(rel_6m) < -10:
        bonus = 2.0
    elif rel_6m is not None and float(rel_6m) < -5:
        bonus = 1.0

    pts = _clamp(base + bonus, 0.0, float(_MAX_SECTOR))
    return pts, True


def _score_valuation_depression(valuation_data: dict | None) -> tuple[float, bool]:
    """
    Värderingsdepression: dagens multipel mot aktiens EGEN historik. (max 15p)

    Ersätter analytikernedgraderingarna (ingen nordisk källa) med en ärligare
    kontrarisk signal: en aktie i botten av sin egen värderingshistorik är
    övergiven av marknaden — Dremen/Rule-logik, ur Börsdatas KPI-historik
    som redan är 24h-cachad.

    valuation_data = {"current": float, "history": [float, ...], "metric": str}
    Percentilrank = andel historiska värden UNDER dagens. Lågt = billig mot
    sig själv. Kräver minst 8 historikpunkter och positiv multipel (negativa
    multiplar är förlustår — ingen värderingssignal).

      rank <= 0.10 → 15p   (billigaste tiondelen av egen historik)
      rank <= 0.20 → 12p
      rank <= 0.35 →  8p
      rank <= 0.50 →  4p
      rank >  0.50 →  0p
    """
    if not valuation_data:
        return 0.0, False
    current = valuation_data.get("current")
    history = [h for h in (valuation_data.get("history") or [])
               if isinstance(h, (int, float)) and h > 0]
    if current is None or float(current) <= 0 or len(history) < 8:
        return 0.0, False
    current = float(current)
    rank = sum(1 for h in history if h < current) / len(history)
    if rank <= 0.10:   return 15.0, True
    if rank <= 0.20:   return 12.0, True
    if rank <= 0.35:   return 8.0,  True
    if rank <= 0.50:   return 4.0,  True
    return 0.0, True


def _score_volume_drought(price_data: dict) -> tuple[float, bool]:
    """
    Volymtorka: bevakningen har dött. (max 13p)

    Ersätter StockTwits-tystnaden (hämtades aldrig) med samma signal ur data
    vi redan har: 20-dagarsvolymen mot 6-månaderssnittet. En aktie vars
    omsättning torkat ihop är bortglömd — ingen säljer ens längre.

      kvot < 0.4  → 13p  (volymen mer än halverad)
      kvot < 0.6  → 10p
      kvot < 0.8  →  6p
      kvot < 1.0  →  3p
      kvot >= 1.0 →  0p  (aktiv handel — inte bortglömd)
    """
    v20 = price_data.get("avg_volume_20d")
    v6m = price_data.get("avg_volume_6m")
    if not v20 or not v6m or v6m <= 0:
        return 0.0, False
    ratio = float(v20) / float(v6m)
    if ratio < 0.4:    return 13.0, True
    if ratio < 0.6:    return 10.0, True
    if ratio < 0.8:    return 6.0,  True
    if ratio < 1.0:    return 3.0,  True
    return 0.0, True


def _score_stocktwits_bear(sentiment_data: dict | None) -> tuple[float, bool]:
    """
    StockTwits bear ratio. (max 20p)
    High bear ratio = retail capitulation / active bearishness = contrarian signal.

    bear_ratio 0.0  → 0p
    bear_ratio 0.3  → 9p   (30% bear)
    bear_ratio 0.5  → 15p  (majority bear)
    bear_ratio 0.7+ → 20p  (strong bear consensus)

    Scaled linearly 0-0.7+ capped at 20p, weighted by confidence.
    """
    if not sentiment_data:
        return 0.0, False
    conf       = sentiment_data.get("confidence", 0.0)
    bear_ratio = sentiment_data.get("bear_ratio", 0.0) or 0.0
    if conf == 0.0:
        return 0.0, False
    raw = _clamp(bear_ratio / 0.70 * _MAX_BEAR_RATIO, 0.0, float(_MAX_BEAR_RATIO))
    # Scale down by confidence — partial data should be treated with caution
    pts = raw * _clamp(conf, 0.5, 1.0)
    return round(pts, 1), True


def _score_cycle_position(price_data: dict) -> tuple[float, bool]:
    """
    Distance below 5-year average price: max 15p.
    Returns (points, has_real_data).

    Captures multi-year cycle compression beyond short-term pullbacks.
    A stock trading 30-50% below its 5-year average is likely in a
    sector trough — the core contrarian opportunity.

    Input: price_data["avg_price_5y"] — mean closing price over 5 years
           price_data["close"]        — current close (always present)

    > 40% below 5y avg → 15p  (deep trough)
    > 30% below        → 12p
    > 20% below        →  8p
    > 10% below        →  4p
    ≤ 10% below or above → 0p
    """
    close  = price_data.get("close", 0.0)
    avg_5y = price_data.get("avg_price_5y")

    if not avg_5y or avg_5y <= 0 or close <= 0:
        return 0.0, False

    pct_below = (avg_5y - close) / avg_5y * 100

    if pct_below > 40:    return 15.0, True
    elif pct_below > 30:  return 12.0, True
    elif pct_below > 20:  return  8.0, True
    elif pct_below > 10:  return  4.0, True
    else:                  return  0.0, True


# ─── Value Trap flag ─────────────────────────────────────────────────────────

def _check_value_trap(hat_score: float, strength_score: float | None) -> bool:
    """True if Hat is very high but Strength is very low — falling knife, not gem."""
    if strength_score is None:
        return False
    return hat_score > VALUE_TRAP_HAT_MIN and strength_score < VALUE_TRAP_STRENGTH_MAX


# ─── Main scoring function ───────────────────────────────────────────────────

def calculate_hate_score(
    price_data: dict,
    sentiment_data: dict | None = None,
    analyst_data: dict | None = None,
    short_data: dict | None = None,
    sector_data: dict | None = None,
    strength_score: float | None = None,
    valuation_data: dict | None = None,
) -> HateResult:
    """
    Calculate Hat Score (0-100) for a single instrument.

    Kärnan är sju komponenter (summa 100) som alla nås med Börsdata +
    yfinance + prisserien. Saknad data ger 0 poäng OCH räknas bort ur
    nåbart max — poängen normaliseras sedan mot det nåbara, så "Hat >= 45"
    betyder 45 % av det som faktiskt gick att mäta. Utan normaliseringen
    hade en rad med luckor aldrig kunnat nå tröskeln (den gamla buggen,
    fast utan de fabricerade defaultpoängen).

    Skyddsräcke: minst halva poängrymden (50 p) måste vara mätbar för att
    normaliseringen ska användas — annars gäller råpoängen, så två mätta
    komponenter aldrig ensamma kan blåsa upp ett 27/27-läge till 100.

    EODHD/StockTwits (analyst/sentiment) är frivillig förstärkning: en
    begränsad bonus (max +8) ovanpå kärnan, aldrig ett krav.

    Args:
        price_data:     Required. close, sma200, high_52w, low_52w.
                        Optional: avg_price_5y, avg_volume_20d, avg_volume_6m.
        sentiment_data: Optional bonus. StockTwits bear_ratio, message_count.
        analyst_data:   Optional bonus. downgrades_90d, upgrades_90d, consensus.
        short_data:     short_float_pct — FI-registret via Börsdata (Norden),
                        yfinance (US) eller EODHD.
        sector_data:    stock_vs_sector_3m (primär) eller ETF-nycklarna.
        valuation_data: {"current", "history", "metric"} — Börsdata KPI-historik.
        strength_score: Optional float (0-100) from strength.py — used for
                        Value Trap detection only.

    Returns:
        HateResult with score, breakdown, flags, confidence.
    """
    if not price_data:
        return HateResult(
            score=0.0,
            flags=["NO_PRICE_DATA"],
            confidence=0.0,
        )

    # Kärnkomponenterna (max-poäng, poäng, har riktig data)
    sma_pts,    sma_real    = _score_sma200_gap(price_data)
    low_pts,    low_real    = _score_52w_low_proximity(price_data)
    cycle_pts,  cycle_real  = _score_cycle_position(price_data)
    short_pts,  short_real  = _score_short_interest(short_data)
    val_pts,    val_real    = _score_valuation_depression(valuation_data)
    vol_pts,    vol_real    = _score_volume_drought(price_data)
    sector_pts, sector_real = _score_sector_outflow(sector_data)

    core = [
        (_MAX_SMA200,    sma_pts,    sma_real),
        (_MAX_52W_LOW,   low_pts,    low_real),
        (_MAX_CYCLE,     cycle_pts,  cycle_real),
        (_MAX_SHORT,     short_pts,  short_real),
        (_MAX_VALUATION, val_pts,    val_real),
        (_MAX_VOLUME,    vol_pts,    vol_real),
        (_MAX_SECTOR,    sector_pts, sector_real),
    ]
    raw       = sum(p for _m, p, _r in core)
    reachable = sum(m for m, _p, r in core if r)

    if reachable >= 50:
        total = raw / reachable * 100.0
    else:
        total = raw   # för lite mätt för att normalisera — råpoäng + låg confidence

    # Bonus: EODHD/StockTwits när de råkar finnas — skalas ihop, max +8.
    analyst_pts, analyst_real = _score_analyst_downgrades(analyst_data)
    sil_pts,     sil_real     = _score_retail_silence(sentiment_data)
    bear_pts,    bear_real    = _score_stocktwits_bear(sentiment_data)
    bonus_raw = (analyst_pts / _MAX_ANALYST * 5.0 if analyst_real else 0.0) \
        + (bear_pts / _MAX_BEAR_RATIO * 2.0 if bear_real else 0.0) \
        + (sil_pts / _MAX_RETAIL_SIL * 1.0 if sil_real else 0.0)
    bonus = _clamp(bonus_raw, 0.0, float(_MAX_BONUS))

    total = round(_clamp(total + bonus, 0.0, 100.0), 1)

    confidence = round(reachable / 100.0, 2)

    breakdown = {
        "sma200_gap":           round(sma_pts, 1),
        "low_52w_proximity":    round(low_pts, 1),
        "cycle_position":       round(cycle_pts, 1),
        "short_interest":       round(short_pts, 1),
        "valuation_depression": round(val_pts, 1),
        "volume_drought":       round(vol_pts, 1),
        "sector_outflow":       round(sector_pts, 1),
    }
    if bonus > 0:
        breakdown["sentiment_bonus"] = round(bonus, 1)

    flags: list[str] = []
    if not sma_real:    flags.append("PRICE_DATA_PARTIAL")
    if not short_real:  flags.append("SHORT_DATA_MISSING")
    if not val_real:    flags.append("VALUATION_DATA_MISSING")
    if not vol_real:    flags.append("VOLUME_DATA_MISSING")
    if not sector_real: flags.append("SECTOR_DATA_MISSING")
    if not cycle_real:  flags.append("CYCLE_DATA_MISSING")
    if reachable < 50:
        flags.append("HATE_LOW_COVERAGE")

    if _check_value_trap(total, strength_score):
        flags.append("POTENTIAL_VALUE_TRAP")

    return HateResult(
        score=total,
        breakdown=breakdown,
        flags=flags,
        confidence=confidence,
    )


# ─── EODHD data fetchers ─────────────────────────────────────────────────────

def _get_eodhd_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("EODHD_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("EODHD_API_KEY", "")


def fetch_analyst_data(ticker: str, api_key: str | None = None) -> dict:
    """
    Fetch analyst upgrades/downgrades from EODHD (last 90 days).
    Falls back to yfinance analyst info if no API key.

    Returns dict with downgrades_90d, upgrades_90d, consensus.
    """
    import requests

    api_key = api_key or _get_eodhd_key()
    cutoff  = datetime.now(tz=timezone.utc) - timedelta(days=90)

    if api_key:
        # Normalise ticker for EODHD
        eodhd_ticker = (
            ticker.replace(".ST", ".STO")
                  .replace(".OL", ".OSL")
                  .replace(".CO", ".CPH")
                  .replace(".HE", ".HEL")
        )
        try:
            url = f"https://eodhd.com/api/upgrades-downgrades/{eodhd_ticker}"
            r = requests.get(url, params={"api_token": api_key, "fmt": "json"}, timeout=15)
            if r.status_code == 200:
                rows = r.json()
                downs = ups = 0
                for row in rows:
                    try:
                        date = datetime.fromisoformat(row.get("date", "2000-01-01")).replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                    if date < cutoff:
                        continue
                    action = str(row.get("action", "")).lower()
                    if "down" in action:
                        downs += 1
                    elif "up" in action:
                        ups += 1
                return {"downgrades_90d": downs, "upgrades_90d": ups, "consensus": "Hold"}
        except Exception as e:
            logger.debug("EODHD analyst fetch failed for %s: %s", ticker, e)

    # yfinance fallback
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        consensus_map = {
            1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Underperform", 5: "Sell"
        }
        rec = info.get("recommendationMean")
        consensus = "Hold"
        if rec is not None:
            consensus = consensus_map.get(round(float(rec)), "Hold")
        return {
            "downgrades_90d": 0,
            "upgrades_90d":   0,
            "consensus":      consensus,
        }
    except Exception as e:
        logger.debug("yfinance analyst fetch failed for %s: %s", ticker, e)

    return {}


def fetch_short_data(ticker: str, api_key: str | None = None) -> dict:
    """
    Fetch short interest from EODHD fundamentals (SharesStats).
    Falls back to yfinance shortPercentOfFloat.

    Returns dict with short_float_pct, days_to_cover.
    """
    import requests

    api_key = api_key or _get_eodhd_key()

    if api_key:
        eodhd_ticker = (
            ticker.replace(".ST", ".STO")
                  .replace(".OL", ".OSL")
                  .replace(".CO", ".CPH")
                  .replace(".HE", ".HEL")
        )
        try:
            url = f"https://eodhd.com/api/fundamentals/{eodhd_ticker}"
            r = requests.get(url, params={"api_token": api_key, "fmt": "json"}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                stats = data.get("SharesStats", {}) or {}
                pct_raw = stats.get("PercentFloat") or stats.get("ShortPercentFloat")
                dtc_raw = stats.get("ShortRatio") or stats.get("DaystoCover")
                result: dict = {}
                if pct_raw is not None:
                    try:
                        pct = float(pct_raw)
                        # EODHD returns 0-1 or 0-100 depending on field
                        result["short_float_pct"] = pct * 100 if pct < 1.0 else pct
                    except (ValueError, TypeError):
                        pass
                if dtc_raw is not None:
                    try:
                        result["days_to_cover"] = float(dtc_raw)
                    except (ValueError, TypeError):
                        pass
                if result:
                    return result
        except Exception as e:
            logger.debug("EODHD short fetch failed for %s: %s", ticker, e)

    # yfinance fallback
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        result = {}
        pct = info.get("shortPercentOfFloat")
        if pct is not None:
            try:
                pct = float(pct)
                result["short_float_pct"] = pct * 100 if pct < 1.0 else pct
            except (ValueError, TypeError):
                pass
        dtc = info.get("shortRatio")
        if dtc is not None:
            try:
                result["days_to_cover"] = float(dtc)
            except (ValueError, TypeError):
                pass
        return result
    except Exception as e:
        logger.debug("yfinance short fetch failed for %s: %s", ticker, e)

    return {}


def fetch_sector_data(sector_etf: str, market_etf: str = "SPY") -> dict:
    """
    Compute sector ETF performance relative to market ETF over 3m and 6m.
    sector_etf: e.g. 'XLE' (energy), 'XME' (metals), 'GDX' (gold miners)
    """
    try:
        import yfinance as yf
        import numpy as np

        tickers_data = yf.download(
            [sector_etf, market_etf], period="1y",
            auto_adjust=True, progress=False, threads=True
        )
        closes = tickers_data.get("Close", tickers_data)
        if closes is None or closes.empty:
            return {}

        result: dict = {}
        for label, n_days in (("3m", 63), ("6m", 126)):
            key = f"sector_vs_market_{label}"
            try:
                sec = closes[sector_etf].dropna()
                mkt = closes[market_etf].dropna()
                if len(sec) < n_days or len(mkt) < n_days:
                    continue
                sec_perf = (sec.iloc[-1] / sec.iloc[-n_days] - 1) * 100
                mkt_perf = (mkt.iloc[-1] / mkt.iloc[-n_days] - 1) * 100
                result[key] = round(float(sec_perf - mkt_perf), 2)
            except Exception:
                pass
        return result
    except Exception as e:
        logger.debug("Sector relative perf failed for %s: %s", sector_etf, e)
        return {}


# ─── CLI diagnostics ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "_label": "Uranium miner (hated, strong fundamentals)",
            "_strength": 78.0,
            "price":    {"close": 12.5,  "sma200": 18.0,  "high_52w": 22.0, "low_52w": 10.5},
            "sentiment":{"message_count": 3, "bear_ratio": 0.62, "bull_ratio": 0.15, "confidence": 0.8},
            "analyst":  {"downgrades_90d": 3, "upgrades_90d": 0, "consensus": "Underperform"},
            "short":    {"short_float_pct": 12.5, "days_to_cover": 8.2},
            "sector":   {"sector_vs_market_3m": -14.0, "sector_vs_market_6m": -22.0},
        },
        {
            "_label": "Gold miner (mildly hated)",
            "_strength": 65.0,
            "price":    {"close": 35.0,  "sma200": 38.5,  "high_52w": 48.0, "low_52w": 32.0},
            "sentiment":{"message_count": 18, "bear_ratio": 0.38, "bull_ratio": 0.30, "confidence": 1.0},
            "analyst":  {"downgrades_90d": 1, "upgrades_90d": 1, "consensus": "Hold"},
            "short":    {"short_float_pct": 6.2,  "days_to_cover": 3.5},
            "sector":   {"sector_vs_market_3m": -5.5, "sector_vs_market_6m": -8.0},
        },
        {
            "_label": "Tech darling (loved, fails threshold)",
            "_strength": 72.0,
            "price":    {"close": 185.0, "sma200": 155.0, "high_52w": 195.0, "low_52w": 120.0},
            "sentiment":{"message_count": 210, "bear_ratio": 0.08, "bull_ratio": 0.75, "confidence": 1.0},
            "analyst":  {"downgrades_90d": 0, "upgrades_90d": 4, "consensus": "Strong Buy"},
            "short":    {"short_float_pct": 1.2,  "days_to_cover": 1.1},
            "sector":   {"sector_vs_market_3m": 12.0, "sector_vs_market_6m": 18.0},
        },
        {
            "_label": "Value Trap (hated + weak fundamentals)",
            "_strength": 28.0,
            "price":    {"close": 2.10,  "sma200": 6.50,  "high_52w": 9.80, "low_52w": 1.95},
            "sentiment":{"message_count": 1, "bear_ratio": 0.80, "bull_ratio": 0.05, "confidence": 0.6},
            "analyst":  {"downgrades_90d": 5, "upgrades_90d": 0, "consensus": "Sell"},
            "short":    {"short_float_pct": 28.0, "days_to_cover": 14.0},
            "sector":   {"sector_vs_market_3m": -18.0, "sector_vs_market_6m": -30.0},
        },
        {
            "_label": "Copper miner (no sentiment/short data)",
            "_strength": 61.0,
            "price":    {"close": 8.20, "sma200": 11.0, "high_52w": 14.5, "low_52w": 7.8},
            "sentiment": None,
            "analyst":  {"downgrades_90d": 2, "upgrades_90d": 0, "consensus": "Hold"},
            "short":    None,
            "sector":   {"sector_vs_market_3m": -9.0, "sector_vs_market_6m": -12.0},
        },
    ]

    print(f"\n{'─'*82}")
    print(f"  HAT SCORE  |  Threshold: {HAT_THRESHOLD}  |  Value Trap: Hat>{VALUE_TRAP_HAT_MIN} & Strength<{VALUE_TRAP_STRENGTH_MAX}")
    print(f"{'─'*82}")

    for case in test_cases:
        label    = case["_label"]
        strength = case["_strength"]
        result   = calculate_hate_score(
            price_data    = case["price"],
            sentiment_data= case["sentiment"],
            analyst_data  = case["analyst"],
            short_data    = case["short"],
            sector_data   = case["sector"],
            strength_score= strength,
        )
        gate = "PASS" if result.passes_threshold else "FAIL"
        trap = " *** VALUE TRAP ***" if result.is_value_trap else ""
        print(f"\n  {label}")
        print(f"    Score : {result.score:>5.1f}/100   [{gate}]   Conf: {result.confidence:.0%}   Strength: {strength}{trap}")
        print(f"    Break : {result.breakdown}")
        non_data_flags = [f for f in result.flags if "MISSING" not in f and "PARTIAL" not in f]
        if non_data_flags:
            print(f"    Flags : {non_data_flags}")

    print(f"\n{'─'*82}\n")
