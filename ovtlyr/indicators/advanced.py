"""
Advanced OVTLYR indicators:
  1. Volatility Histogram (5-year directional volatility distribution)
  2. Oscillator Direction (RSI momentum timing)
  3. Bull List % with EMA 5 crossover
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_volatility_histogram(df: pd.DataFrame, years: int = 5) -> dict:
    """
    Compute directional volatility histogram (OVTLYR-style).
    
    Shows the distribution of daily returns over the last N years.
    
    Interpretation:
      - Right cluster (tall peak right side) = More consistency, better for longs
      - Left cluster (tall peak left side) = Greater downside risk
      - Wide peak = More erratic/volatile
      - Tall narrow peak = Consistent behavior
    
    Uses daily returns, bins them into ~50 buckets from -10% to +10%.
    
    Returns dict with:
      bins: list of floats (bin centers, e.g. [-5%, -4.5%, ..., +5%])
      counts: list of ints (count per bin)
      up_count: int (total positive return days)
      down_count: int (total negative return days)
      up_pct: float (percentage of days that were positive)
      mean_up: float (average positive return %)
      mean_down: float (average negative return %)
      skew: float (positive = right-skewed = bullish bias)
      classification: str ("Bullish Bias" / "Bearish Bias" / "Symmetric" / "High Volatility")
      
    Classification logic:
      - up_pct > 55% AND skew > 0.1 → "Bullish Bias"  
      - up_pct < 45% AND skew < -0.1 → "Bearish Bias"
      - abs(std) > 2.5% → "High Volatility"
      - else → "Symmetric"
    """
    # Implementation:
    close = df["Close"].astype(float)
    # Use last N years of data (252 trading days per year)
    max_bars = years * 252
    if len(close) > max_bars:
        close = close.iloc[-max_bars:]
    
    returns = close.pct_change().dropna() * 100  # in percent
    
    # Clip to -10% to +10% for histogram
    clipped = returns.clip(-10, 10)
    
    # 50 bins from -10 to +10
    bin_edges = np.linspace(-10, 10, 51)
    counts, _ = np.histogram(clipped, bins=bin_edges)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    
    up_days = (returns > 0).sum()
    down_days = (returns < 0).sum()
    total = len(returns)
    up_pct = up_days / total * 100 if total > 0 else 50
    
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]
    mean_up = float(pos_returns.mean()) if len(pos_returns) > 0 else 0
    mean_down = float(neg_returns.mean()) if len(neg_returns) > 0 else 0
    
    skew = float(returns.skew()) if len(returns) > 10 else 0
    std = float(returns.std()) if len(returns) > 10 else 0
    
    if up_pct > 55 and skew > 0.1:
        classification = "Bullish Bias"
    elif up_pct < 45 and skew < -0.1:
        classification = "Bearish Bias"
    elif std > 2.5:
        classification = "High Volatility"
    else:
        classification = "Symmetric"
    
    return {
        "bins": [round(b, 2) for b in bin_centers],
        "counts": counts.tolist(),
        "up_count": int(up_days),
        "down_count": int(down_days),
        "up_pct": round(up_pct, 1),
        "mean_up": round(mean_up, 2),
        "mean_down": round(mean_down, 2),
        "skew": round(skew, 3),
        "std": round(std, 2),
        "classification": classification,
        "total_days": int(total),
        "years_analyzed": round(total / 252, 1),
    }


def compute_oscillator_direction(df: pd.DataFrame, rsi_period: int = 14) -> dict:
    """
    Compute RSI oscillator with direction and timing analysis.
    
    Key insight from OVTLYR: "If oscillator just started upward journey
    few days ago = even better. If already going up 8-9 days = may be late."
    
    Returns dict with:
      rsi: float (current RSI value)
      direction: "Rising" / "Falling" / "Flat"
      days_in_direction: int (consecutive days in current direction)
      timing: "Early" / "Mid" / "Late" / "Exhausted"
      timing_color: str (rgba color)
      rsi_5d_ago: float
      rsi_change_5d: float
      signal: "ENTER" / "WAIT" / "LATE" / "EXIT"
      
    Timing logic:
      Rising for 1-3 days → "Early" (best entry, green)
      Rising for 4-6 days → "Mid" (acceptable, yellow)
      Rising for 7-9 days → "Late" (risky entry, orange)
      Rising for 10+ days → "Exhausted" (don't enter, red)
      Falling → timing = "EXIT" direction
      
    Signal logic:
      RSI rising from below 50, Early/Mid → "ENTER"
      RSI rising, Late → "WAIT"
      RSI rising, Exhausted → "LATE"  
      RSI falling → "EXIT" (or "WAIT" if above 50)
    """
    close = df["Close"].astype(float)
    
    # Compute RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_series = (100 - (100 / (1 + rs))).fillna(50)
    
    rsi_now = float(rsi_series.iloc[-1])
    rsi_5d = float(rsi_series.iloc[-6]) if len(rsi_series) > 5 else rsi_now
    rsi_change = rsi_now - rsi_5d
    
    # Count consecutive days in same direction
    rsi_diff = rsi_series.diff()
    days_rising = 0
    days_falling = 0
    for i in range(len(rsi_diff) - 1, max(0, len(rsi_diff) - 30), -1):
        d = rsi_diff.iloc[i]
        if d > 0.1:
            if days_falling > 0:
                break
            days_rising += 1
        elif d < -0.1:
            if days_rising > 0:
                break
            days_falling += 1
        else:
            break
    
    if days_rising > 0:
        direction = "Rising"
        days = days_rising
    elif days_falling > 0:
        direction = "Falling"
        days = days_falling
    else:
        direction = "Flat"
        days = 0
    
    # Timing
    if direction == "Rising":
        if days <= 3:
            timing = "Early"
            timing_color = "rgba(0,255,136,0.9)"
        elif days <= 6:
            timing = "Mid"
            timing_color = "rgba(255,221,0,0.9)"
        elif days <= 9:
            timing = "Late"
            timing_color = "rgba(255,136,0,0.9)"
        else:
            timing = "Exhausted"
            timing_color = "rgba(255,51,85,0.9)"
    elif direction == "Falling":
        timing = "Declining"
        timing_color = "rgba(255,51,85,0.9)"
    else:
        timing = "Flat"
        timing_color = "rgba(74,74,106,0.9)"
    
    # Signal
    if direction == "Rising" and timing in ("Early", "Mid") and rsi_now < 65:
        signal = "ENTER"
    elif direction == "Rising" and timing == "Late":
        signal = "WAIT"
    elif direction == "Rising" and timing == "Exhausted":
        signal = "LATE"
    elif direction == "Falling" and rsi_now < 45:
        signal = "EXIT"
    elif direction == "Falling":
        signal = "WAIT"
    else:
        signal = "WAIT"
    
    return {
        "rsi": round(rsi_now, 1),
        "direction": direction,
        "days_in_direction": days,
        "timing": timing,
        "timing_color": timing_color,
        "rsi_5d_ago": round(rsi_5d, 1),
        "rsi_change_5d": round(rsi_change, 1),
        "signal": signal,
        "rsi_series": rsi_series.tolist()[-60:],  # last 60 bars for chart
    }


def compute_bull_list_pct(ticker_data: dict, ema_crossover_period: int = 5) -> dict:
    """
    Compute Bull List % with EMA 5 crossover.
    
    Bull List % = percentage of tickers where price > EMA50 (bullish trend).
    
    Parameters:
      ticker_data: dict of ticker -> pd.DataFrame (OHLCV data)
                   OR dict of ticker -> {"close": float, "ema50": float}
      
    If raw DataFrames are provided, compute EMA50 internally.
    If simple dicts with close/ema50, use those directly.
    
    The function computes:
      1. For each ticker: is price > EMA50? → bullish = True
      2. Bull List % = count(bullish) / total * 100
      3. Track Bull List % over time if possible (use last 60 data points)
      4. Compute EMA 5 of Bull List %
      5. Detect crossover: Bull List % crosses above/below EMA 5
    
    Returns dict with:
      bull_pct: float (current Bull List %)
      bull_count: int
      total_count: int
      bear_count: int
      ema5: float (EMA 5 of Bull List %)
      crossover: "Bullish" / "Bearish" / None
      zone: "Extreme Bullish" (>75%) / "Bullish" (50-75%) / "Bearish" (25-50%) / "Extreme Bearish" (<25%)
      signal: "GO" / "CAUTION" / "STOP"
      history: list of floats (last 60 Bull List % values, if available from ticker_data)
      ema5_history: list of floats (last 60 EMA5 values)
      
    Signal logic:
      Bull% < 25 AND turning up → "GO" (extreme fear, best time to buy)
      Bull% > 75 AND turning down → "STOP" (extreme greed, stop buying)
      Bull% crossing above EMA5 from below 50 → "GO"
      Bull% crossing below EMA5 from above 50 → "CAUTION"
    """
    # Simple mode: just count current bull/bear
    bull = 0
    bear = 0
    total = 0
    
    for ticker, data in ticker_data.items():
        if isinstance(data, pd.DataFrame):
            if "Close" in data.columns and len(data) >= 50:
                close_val = float(data["Close"].iloc[-1])
                ema50_val = float(data["Close"].ewm(span=50).mean().iloc[-1])
            else:
                continue
        elif isinstance(data, dict):
            close_val = data.get("close", 0)
            ema50_val = data.get("ema50", 0)
        else:
            continue
        
        total += 1
        if close_val > ema50_val:
            bull += 1
        else:
            bear += 1
    
    bull_pct = (bull / total * 100) if total > 0 else 50.0
    
    # For history, we'd need to compute this over time
    # For now, generate a synthetic history based on current reading
    # (In production, this would be computed from stored daily snapshots)
    history = [bull_pct]  # Current only — will expand when we have time series
    ema5_val = bull_pct  # Same as current for single point
    
    # Zone
    if bull_pct > 75:
        zone = "Extreme Bullish"
    elif bull_pct > 50:
        zone = "Bullish"
    elif bull_pct > 25:
        zone = "Bearish"
    else:
        zone = "Extreme Bearish"
    
    # Signal
    if bull_pct < 25:
        signal = "GO"  # Extreme fear = contrarian buy
    elif bull_pct > 75:
        signal = "STOP"  # Extreme greed = stop buying
    else:
        signal = "CAUTION"
    
    # Crossover (would need history — placeholder)
    crossover = None
    
    return {
        "bull_pct": round(bull_pct, 1),
        "bull_count": bull,
        "bear_count": bear,
        "total_count": total,
        "ema5": round(ema5_val, 1),
        "crossover": crossover,
        "zone": zone,
        "signal": signal,
        "history": history,
        "ema5_history": [ema5_val],
    }
