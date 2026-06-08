"""
catalyst.py — Catalyst Score (0-100) for Contrarian Alpha Screener.

Detects early technical reversal signals in stocks that already pass
Necessity + Strength + Hate gates.  Without a catalyst the contrarian
thesis is "correct but early" — the most dangerous position.

Adapted from blindspot/scoring/catalyst.py (0-20 scale) and
expanded to 0-100 with 4 components + Viking Regime bonus.

6 components:
  1. Price vs SMA50           max 35p  (above = institutional re-interest)
  2. SMA50 slope positive     max 20p  (trend turning, not just bouncing)
  3. Volume surge > 150%      max 18p  (accumulation signal)
  4. Price reversal (3 green) max 12p  (3 consecutive up-closes after decline)
  5. Insider ownership %      max  8p  (aligned management = confidence signal)
  6. Net insider buying (12m) max  7p  (recent accumulation by insiders)

Viking Regime bonus (OVTLYR via ovtlyr/indicators/trend.compute_trend):
  regime_color == "green"  → VIKING_REGIME_BONUS = +5p to Composite Score
  This bonus is NOT folded into Catalyst Score — it's applied in composite.py.
  regime_color is: green (Bullish + low vol) | orange (neutral) | red (Bearish/high vol)

Composite weight: CATALYST_COMPOSITE_WEIGHT = 0.15 (15%)

Input dict keys:

  price_data:
    close          float   Current price
    sma50          float   50-day SMA
    sma50_slope    float   Linear regression slope of SMA50 (last 10 bars)
                           Same field computed by blindspot/sources/price.py
    current_volume float   Latest session volume
    avg_volume_20d float   20-session average volume
    close_history  list    Closes newest-first: [today, yesterday, ...]
                           Min 6 values needed for reversal check.

  ovtlyr_regime:   (optional — provide to avoid re-fetching price data)
    regime_color   str     "green" | "orange" | "red"
    trend_state    str     "Bullish" | "Bearish" | "Neutral"
    ovtlyr_nine    int     0-100 OVTLYR NINE score (optional)

  insider_data:    (optional — from borsdata_api.get_insider_transactions)
    insider_ownership_pct  float   % of shares held by insiders
    insider_net_bought_12m float   net shares bought in last 12 months
                                   positive = net buying, negative = net selling
    insider_buy_count      int     number of buy transactions in 12 months
    insider_sell_count     int     number of sell transactions in 12 months
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─── Pipeline constants ───────────────────────────────────────────────────────

CATALYST_COMPOSITE_WEIGHT = 0.15   # 15% of final Composite Score
VIKING_REGIME_BONUS       = 5.0    # Flat +5p added to Composite Score (not Catalyst)

# Component max points (must sum to 100)
_W_ABOVE_SMA50  = 35
_W_SMA50_SLOPE  = 20
_W_VOL_SURGE    = 18
_W_REVERSAL     = 12
_W_INSIDER_OWN  =  8   # insider ownership % — skin in the game
_W_INSIDER_BUY  =  7   # net insider buying in last 12 months

# Volume surge threshold from spec
VOL_SURGE_THRESHOLD = 1.50   # > 150% of 20-day average

# ─── Result model ────────────────────────────────────────────────────────────

@dataclass
class CatalystResult:
    score: float
    breakdown: dict[str, float]         = field(default_factory=dict)
    flags: list[str]                    = field(default_factory=list)
    viking_regime_green: bool           = False   # True → apply VIKING_REGIME_BONUS in composite
    viking_regime_color: str            = "unknown"
    ovtlyr_nine: int | None             = None    # Raw OVTLYR NINE score if available
    insider_ownership_pct: float | None = None    # % of shares held by insiders
    insider_net_buying: bool            = False   # True when net 12m insider buying is positive

    @property
    def composite_bonus(self) -> float:
        return VIKING_REGIME_BONUS if self.viking_regime_green else 0.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ─── Component scorers ────────────────────────────────────────────────────────

def _score_above_sma50(price_data: dict) -> tuple[float, bool]:
    """
    Price vs SMA50. (max 35p)

    Graduated: breaking above SMA50 is the primary entry signal.
    > SMA50 by >= 3%   → 35p  (clear breakout with cushion)
    > SMA50 by any %   → 28p  (just crossed — still valid)
    Within 1% below    → 10p  (coiling at SMA50 — potential)
    > 1% below SMA50   → 0p
    """
    close = price_data.get("close", 0.0)
    sma50 = price_data.get("sma50", 0.0)
    if sma50 <= 0 or close <= 0:
        return 0.0, False
    gap_pct = (close - sma50) / sma50 * 100
    if gap_pct >= 3.0:   return float(_W_ABOVE_SMA50), True
    if gap_pct > 0.0:    return 28.0, True
    if gap_pct >= -1.0:  return 10.0, True   # coiling just below
    return 0.0, True


def _score_sma50_slope(price_data: dict) -> tuple[float, bool]:
    """
    SMA50 slope direction and strength. (max 20p)

    slope is a linear regression slope over last 10 bars of SMA50,
    as computed by blindspot/sources/price.py and
    ovtlyr/data_fetch/yahoo.py (via np.polyfit).

    Normalise slope relative to price to get %-per-bar metric:
      rel_slope = slope / close * 100

    rel_slope >= +0.10%/bar → 20p   (strongly rising)
    rel_slope >= +0.03%/bar → 16p   (clearly rising)
    rel_slope >   0         → 11p   (barely positive — early)
    rel_slope == 0          →  0p
    rel_slope <   0         →  0p
    """
    slope = price_data.get("sma50_slope", None)
    close = price_data.get("close", 0.0)
    if slope is None:
        return 0.0, False
    if close <= 0:
        return 0.0, False
    rel = slope / close * 100
    if rel >= 0.10:  return float(_W_SMA50_SLOPE), True
    if rel >= 0.03:  return 16.0, True
    if rel > 0.0:    return 11.0, True
    return 0.0, True


def _score_volume_surge(price_data: dict) -> tuple[float, bool]:
    """
    Volume vs 20-day average. Threshold: > 150%. (max 20p)

    User spec: "volym > 150% av 20-dagars snitt"

    >= 3.0x avg → 20p  (strong institutional surge)
    >= 2.0x avg → 18p
    >= 1.5x avg → 15p  (spec threshold — clear accumulation)
    >= 1.2x avg →  8p  (notable but sub-threshold)
    < 1.2x avg  →  0p

    Volume z-score bonus: if std dev is available and z > 2.0, add 2p.
    """
    vol   = price_data.get("current_volume", 0.0)
    avg   = price_data.get("avg_volume_20d", 0.0)
    if avg <= 0 or vol <= 0:
        return 0.0, False

    ratio = vol / avg
    if ratio >= 3.0:   base = 20.0
    elif ratio >= 2.0: base = 18.0
    elif ratio >= 1.5: base = 15.0
    elif ratio >= 1.2: base =  8.0
    else:              return 0.0, True

    # z-score bonus
    std = price_data.get("std_volume_20d", 0.0)
    bonus = 0.0
    if std > 0:
        z = (vol - avg) / std
        if z > 2.0:
            bonus = 2.0
    return _clamp(base + bonus, 0.0, float(_W_VOL_SURGE)), True


def _score_price_reversal(price_data: dict) -> tuple[float, bool]:
    """
    3 consecutive green days (close > prev close) after a prior decline. (max 15p)

    Requires close_history (newest-first list, min 6 values).
    close_history[0] = today, [1] = yesterday, [2] = 2 days ago, ...

    Pattern check:
      Step 1: Last 3 bars all green: h[0]>h[1], h[1]>h[2], h[2]>h[3]
      Step 2: Prior context (bars 4-8) had at least 1 red bar (declining):
              any of h[4]<h[5], h[5]<h[6], h[6]<h[7], h[7]<h[8]

    Scoring:
      3 green + prior decline → 15p  (full reversal pattern)
      3 green, no decline context (but data available) → 7p  (uptrend extension)
      2 green days only → 5p  (partial pattern)
      < 2 green → 0p
    """
    closes = price_data.get("close_history") or []
    if len(closes) < 4:
        return 0.0, False

    h = closes  # alias for readability

    green_d0 = h[0] > h[1]   # today up
    green_d1 = h[1] > h[2]   # yesterday up
    green_d2 = len(h) > 3 and h[2] > h[3]   # 2 days ago up

    if not (green_d0 and green_d1):
        return 0.0, True

    if not green_d2:
        return 5.0, True   # Only 2 consecutive green days

    # We have 3 consecutive green days — check prior decline
    had_decline = False
    if len(h) >= 9:
        had_decline = any(h[i] < h[i + 1] for i in range(4, 8))
    elif len(h) >= 6:
        had_decline = any(h[i] < h[i + 1] for i in range(3, len(h) - 1))

    return (float(_W_REVERSAL) if had_decline else 7.0), True


def _score_insider_ownership(insider_data: dict | None) -> tuple[float, float | None, bool]:
    """
    Insider ownership as % of shares: max 8p.
    Returns (pts, ownership_pct, has_real_data).

    High insider ownership means management has significant personal wealth
    tied to the share price — a quality signal in a hated stock.

    > 20% → 8p  (management heavily invested)
    10-20% → 5p
     5-10% → 3p
      < 5% → 0p
    """
    if not insider_data:
        return 2.0, None, False   # small default — data often unavailable

    pct = insider_data.get("insider_ownership_pct")
    if pct is None:
        return 2.0, None, False

    pct = float(pct)
    if pct > 20:    pts = 8.0
    elif pct > 10:  pts = 5.0
    elif pct >  5:  pts = 3.0
    else:            pts = 0.0

    return pts, pct, True


def _score_insider_buying(insider_data: dict | None) -> tuple[float, bool, bool]:
    """
    Net insider buying over last 12 months: max 7p.
    Returns (pts, is_net_buying, has_real_data).

    Insiders buying their own depressed stock is the strongest possible
    alignment signal — they are putting personal money in at the trough.

    net positive + >= 2 buys → 7p  (active accumulation)
    net positive + 1 buy     → 4p  (initial buying)
    mixed / neutral          → 2p
    net selling              → 0p
    """
    if not insider_data:
        return 2.0, False, False   # small default

    net   = insider_data.get("insider_net_bought_12m", 0) or 0
    buys  = insider_data.get("insider_buy_count",  0) or 0
    sells = insider_data.get("insider_sell_count", 0) or 0

    if net is None:
        return 2.0, False, False

    is_net_buying = float(net) > 0

    if is_net_buying and buys >= 2:   pts = 7.0
    elif is_net_buying and buys >= 1: pts = 4.0
    elif buys == 0 and sells == 0:    pts = 2.0   # no activity — neutral
    else:                              pts = 0.0   # net selling

    return pts, is_net_buying, (buys > 0 or sells > 0)


# ─── Insider data fetcher ─────────────────────────────────────────────────────

def fetch_insider_data(ins_id: int | None, api=None) -> dict:
    """
    Fetch insider ownership and 12-month transaction summary for an instrument.

    Calls api.get_insider_transactions(ins_id) (Börsdata /insiders/{insId}).
    Falls back to empty dict if API unavailable or ins_id is None.

    Returns dict with:
      insider_ownership_pct  float   % of shares held by insiders (latest)
      insider_net_bought_12m float   net shares bought in 12 months
      insider_buy_count      int     buy transactions in 12 months
      insider_sell_count     int     sell transactions in 12 months
    """
    if ins_id is None or api is None:
        return {}

    try:
        from contrarian_alpha.cache import get_insider, set_insider
        _ckey = f"insider:{ins_id}"
        cached = get_insider(_ckey)
        if cached is not None:
            return cached
    except Exception:
        get_insider = set_insider = None
        _ckey = None

    try:
        transactions = api.get_insider_transactions(ins_id)
        if not transactions:
            return {}

        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=365)

        net_shares = 0.0
        buy_count = sell_count = 0
        ownership_pct: float | None = None

        for tx in transactions:
            # Börsdata insider fields: date, shares, transactionType, ownershipPct
            try:
                tx_date_str = tx.get("verificationDate") or tx.get("date") or ""
                tx_date = datetime.fromisoformat(tx_date_str.replace("Z", "+00:00"))
                if tx_date.tzinfo is None:
                    tx_date = tx_date.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                continue

            tx_type   = str(tx.get("transactionType", "")).lower()
            tx_shares = tx.get("numberOfShares") or tx.get("shares") or 0

            # Capture the most recent ownership % (any date)
            own = tx.get("ownershipChange") or tx.get("ownershipPct")
            if own is not None and ownership_pct is None:
                try:
                    ownership_pct = float(own)
                except (TypeError, ValueError):
                    pass

            if tx_date < cutoff:
                continue   # only count 12-month activity below

            try:
                shares = float(tx_shares)
            except (TypeError, ValueError):
                shares = 0.0

            if "buy" in tx_type or "köp" in tx_type:
                net_shares += shares
                buy_count  += 1
            elif "sell" in tx_type or "sälj" in tx_type:
                net_shares -= shares
                sell_count += 1

        result = {
            "insider_net_bought_12m": round(net_shares, 0),
            "insider_buy_count":      buy_count,
            "insider_sell_count":     sell_count,
        }
        if ownership_pct is not None:
            result["insider_ownership_pct"] = ownership_pct

        if get_insider is not None and _ckey:
            try:
                set_insider(_ckey, result)
            except Exception:
                pass

        return result

    except Exception as e:
        logger.debug("fetch_insider_data failed for ins_id=%s: %s", ins_id, e)
        return {}


# ─── Viking Regime (OVTLYR) ──────────────────────────────────────────────────

def compute_regime_color(df) -> str:
    """
    Compute OVTLYR regime_color from a pandas OHLCV DataFrame.
    Pure reimplementation of ovtlyr/indicators/trend.compute_trend() —
    kept inline to avoid cross-project import dependency.

    Returns "green" | "orange" | "red".

    green  : Bullish (price > EMA200 AND EMA50 > EMA200) + ATR/close < 3%
    orange : Price within 2% of EMA200 OR Neutral trend with moderate vol
    red    : Bearish OR ATR/close > 4%
    """
    try:
        import pandas as pd
        import numpy as np

        close = df["Close"].dropna()
        if len(close) < 50:
            return "orange"

        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        # ATR14
        high = df["High"].dropna()
        low  = df["Low"].dropna()
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()

        last_close  = float(close.iloc[-1])
        last_ema50  = float(ema50.iloc[-1])
        last_ema200 = float(ema200.iloc[-1])
        last_atr    = float(atr14.iloc[-1]) if not atr14.empty else 0.0

        bullish = (last_close > last_ema200) and (last_ema50 > last_ema200)
        bearish = (last_close < last_ema200) and (last_ema50 < last_ema200)
        near_200 = last_ema200 > 0 and abs(last_close - last_ema200) / last_ema200 < 0.02
        atr_ratio = last_atr / last_close if last_close > 0 else 0.0

        if near_200:
            return "orange"
        if bearish or atr_ratio > 0.04:
            return "red"
        if bullish and atr_ratio < 0.03:
            return "green"
        return "orange"

    except Exception as e:
        logger.debug("compute_regime_color failed: %s", e)
        return "orange"


def get_viking_regime(
    ticker: str | None = None,
    df=None,
    ovtlyr_regime: dict | None = None,
) -> tuple[bool, str, int | None]:
    """
    Determine if Viking (OVTLYR) Regime is GREEN for the given instrument.

    Priority:
      1. ovtlyr_regime dict (pre-computed, fastest)
      2. df (DataFrame, compute inline)
      3. ticker (fetch via yfinance, slowest)

    Returns (is_green: bool, regime_color: str, ovtlyr_nine: int | None).
    """
    # 1. Pre-computed result
    if ovtlyr_regime:
        color = str(ovtlyr_regime.get("regime_color", "orange")).lower()
        nine  = ovtlyr_regime.get("ovtlyr_nine")
        return color == "green", color, nine

    # 2. From DataFrame
    if df is not None and not df.empty:
        color = compute_regime_color(df)
        return color == "green", color, None

    # 3. Fetch via yfinance — check regime TTLCache (1 h) first
    if ticker:
        try:
            from contrarian_alpha.cache import get_regime, set_regime
            _rkey = f"viking:{ticker}"
            cached = get_regime(_rkey)
            if cached is not None:
                return cached
        except Exception:
            get_regime = set_regime = None
            _rkey = None

        try:
            from contrarian_alpha.cache import is_delisted
            if is_delisted(ticker):
                return False, "unknown", None
        except Exception:
            pass

        try:
            import yfinance as yf
            import logging as _logging
            _yf_log = _logging.getLogger("yfinance")
            _prev = _yf_log.level
            _yf_log.setLevel(_logging.CRITICAL)
            try:
                df_fetched = yf.Ticker(ticker).history(period="1y", auto_adjust=True, progress=False)
            finally:
                _yf_log.setLevel(_prev)
            if df_fetched is not None and not df_fetched.empty:
                df_fetched.columns = [c.capitalize() for c in df_fetched.columns]
                color = compute_regime_color(df_fetched)
                regime_tuple = (color == "green", color, None)
                if get_regime is not None and _rkey:
                    try:
                        set_regime(_rkey, regime_tuple)
                    except Exception:
                        pass
                return regime_tuple
        except Exception as e:
            logger.debug("get_viking_regime yfinance fetch failed for %s: %s", ticker, e)

    return False, "unknown", None


# ─── Main scoring function ───────────────────────────────────────────────────

def calculate_catalyst_score(
    price_data: dict,
    ticker: str | None = None,
    df=None,
    ovtlyr_regime: dict | None = None,
    insider_data: dict | None = None,
) -> CatalystResult:
    """
    Calculate Catalyst Score (0-100) and Viking Regime status.

    Args:
        price_data:    Required. close, sma50, sma50_slope, volume, close_history.
        ticker:        Optional. Used to fetch Viking Regime if df/ovtlyr_regime not given.
        df:            Optional. OHLCV DataFrame for Viking Regime computation.
        ovtlyr_regime: Optional. Pre-computed OVTLYR dict with regime_color key.
        insider_data:  Optional. Dict with insider_ownership_pct, insider_net_bought_12m,
                       insider_buy_count, insider_sell_count. Use fetch_insider_data().

    Returns:
        CatalystResult with score, breakdown, viking_regime_green, composite_bonus,
        insider_ownership_pct, insider_net_buying.

    Usage in pipeline:
        result = calculate_catalyst_score(price_data, ticker=ticker, insider_data=idata)
        composite += result.score * CATALYST_COMPOSITE_WEIGHT
        composite += result.composite_bonus        # flat +5 if Viking green
    """
    if not price_data:
        return CatalystResult(
            score=0.0,
            flags=["NO_PRICE_DATA"],
        )

    sma_pts,  sma_real           = _score_above_sma50(price_data)
    slp_pts,  slp_real           = _score_sma50_slope(price_data)
    vol_pts,  vol_real           = _score_volume_surge(price_data)
    rev_pts,  rev_real           = _score_price_reversal(price_data)
    own_pts,  own_pct, own_real  = _score_insider_ownership(insider_data)
    buy_pts,  is_buying, buy_real = _score_insider_buying(insider_data)

    total = _clamp(
        sma_pts + slp_pts + vol_pts + rev_pts + own_pts + buy_pts,
        0.0, 100.0,
    )

    breakdown = {
        "above_sma50":      round(sma_pts, 1),
        "sma50_slope":      round(slp_pts, 1),
        "vol_surge":        round(vol_pts, 1),
        "reversal_3d":      round(rev_pts, 1),
        "insider_ownership":round(own_pts, 1),
        "insider_buying":   round(buy_pts, 1),
    }

    flags: list[str] = []
    if not sma_real:  flags.append("SMA50_MISSING")
    if not slp_real:  flags.append("SMA50_SLOPE_MISSING")
    if not vol_real:  flags.append("VOLUME_DATA_MISSING")
    if not rev_real:  flags.append("CLOSE_HISTORY_SHORT")
    if not own_real:  flags.append("INSIDER_DATA_MISSING")

    # Viking Regime
    is_green, color, nine = get_viking_regime(
        ticker=ticker, df=df, ovtlyr_regime=ovtlyr_regime
    )
    if color == "unknown":
        flags.append("VIKING_REGIME_UNKNOWN")

    return CatalystResult(
        score=round(total, 1),
        breakdown=breakdown,
        flags=flags,
        viking_regime_green=is_green,
        viking_regime_color=color,
        ovtlyr_nine=nine,
        insider_ownership_pct=own_pct,
        insider_net_buying=is_buying,
    )


# ─── CLI diagnostics ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "_label":   "Uranium reversal — full catalyst, insider buying, Viking green",
            "_regime":  {"regime_color": "green", "ovtlyr_nine": 74},
            "_insider": {"insider_ownership_pct": 22.0, "insider_net_bought_12m": 850_000,
                         "insider_buy_count": 4, "insider_sell_count": 0},
            "close": 14.20, "sma50": 13.80, "sma50_slope": 0.018,
            "current_volume": 4_800_000, "avg_volume_20d": 2_100_000,
            "std_volume_20d": 500_000,
            "close_history": [14.20, 13.95, 13.70, 13.40, 13.20, 13.00, 12.80, 13.10, 13.30],
        },
        {
            "_label":   "Copper miner — above SMA50, slope flat, no surge, modest ownership",
            "_regime":  {"regime_color": "orange", "ovtlyr_nine": 48},
            "_insider": {"insider_ownership_pct": 8.0, "insider_net_bought_12m": 0,
                         "insider_buy_count": 0, "insider_sell_count": 0},
            "close": 9.10, "sma50": 8.95, "sma50_slope": 0.001,
            "current_volume": 900_000, "avg_volume_20d": 800_000,
            "close_history": [9.10, 8.90, 8.75, 8.60, 8.55, 8.70, 8.80],
        },
        {
            "_label":   "Oil stock — volume surge, price below SMA50, insiders selling",
            "_regime":  {"regime_color": "red", "ovtlyr_nine": 32},
            "_insider": {"insider_ownership_pct": 3.0, "insider_net_bought_12m": -500_000,
                         "insider_buy_count": 0, "insider_sell_count": 3},
            "close": 45.20, "sma50": 48.30, "sma50_slope": -0.05,
            "current_volume": 12_000_000, "avg_volume_20d": 3_500_000,
            "std_volume_20d": 1_200_000,
            "close_history": [45.20, 44.10, 44.50, 44.80, 46.20, 47.00, 47.50, 48.00],
        },
        {
            "_label":   "Gold miner — 3 green days after decline, Viking green, owner-operator",
            "_regime":  {"regime_color": "green", "ovtlyr_nine": 71},
            "_insider": {"insider_ownership_pct": 35.0, "insider_net_bought_12m": 1_200_000,
                         "insider_buy_count": 5, "insider_sell_count": 1},
            "close": 38.50, "sma50": 37.20, "sma50_slope": 0.035,
            "current_volume": 5_400_000, "avg_volume_20d": 3_200_000,
            "std_volume_20d": 800_000,
            "close_history": [38.50, 37.80, 37.10, 36.50, 36.00, 35.50, 35.20, 35.70, 36.10],
        },
        {
            "_label":   "Tech stock (no price history, no insider data)",
            "_regime":  None,
            "_insider": None,
            "close": 185.0, "sma50": 170.0,
        },
    ]

    print(f"\n{'─'*80}")
    print(f"  CATALYST SCORE  |  Composite weight: {CATALYST_COMPOSITE_WEIGHT:.0%}  |  Viking bonus: +{VIKING_REGIME_BONUS:.0f}p")
    print(f"  Components: SMA50({_W_ABOVE_SMA50}p) Slope({_W_SMA50_SLOPE}p) "
          f"Vol>{VOL_SURGE_THRESHOLD:.0%}({_W_VOL_SURGE}p) Rev3d({_W_REVERSAL}p) "
          f"InsOwn({_W_INSIDER_OWN}p) InsBuy({_W_INSIDER_BUY}p)")
    print(f"{'─'*80}")

    for case in test_cases:
        label   = case.pop("_label")
        regime  = case.pop("_regime")
        insider = case.pop("_insider")
        result  = calculate_catalyst_score(
            price_data    = case,
            ovtlyr_regime = regime,
            insider_data  = insider,
        )
        viking = f"VIKING={result.viking_regime_color.upper()}"
        bonus  = f"+{result.composite_bonus:.0f}p bonus" if result.viking_regime_green else "no bonus"
        own_s  = (f"own={result.insider_ownership_pct:.0f}%"
                  if result.insider_ownership_pct is not None else "own=n/a")
        buy_s  = "NET BUY" if result.insider_net_buying else "no-buy"
        data_flags = [f for f in result.flags if any(x in f for x in ("MISSING", "SHORT", "UNKNOWN"))]
        print(f"\n  {label}")
        print(f"    Score : {result.score:>5.1f}/100   {viking} ({bonus})   insider: {own_s} {buy_s}")
        print(f"    Break : {result.breakdown}")
        if data_flags:
            print(f"    Info  : {data_flags}")

    print(f"\n{'─'*80}\n")
