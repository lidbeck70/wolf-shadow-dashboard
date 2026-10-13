"""
valhall/cycle.py — cykelmotorn: var i cykeln står ett segment, dag för dag och utan facit.

Två delar, båda återanvända:
  BERSERK-fasen   berserk.signals.driver_frame → BAISSE · VÄNDER / BAISSE / STARK / UPPTREND / SVAG
                  (samma regler som 🪓 BERSERK Regime)
  Odins Blindspot tioårspercentil, 200-veckors EMA-lutning → TIDIG / MITTEN / SEN / TOPP; hat
                  (avstånd från femårstoppen 40 %, volymtrend 30 %, relativ styrka mot SPY 30 %) och
                  blindspot-poäng = nödvändighet × hat × billighet. Blindspots egen tavla mäter mot
                  hela tioårsfönstret i efterhand; här räknas varje dag bara på data fram till dagen,
                  med samma gränser och vikter (testerna håller dem synkade med blindspot/theme_board).

Status — bestämd i förväg, prövas i fas 3 (backtest) och i papperskontona:
  GRÖN  fasen är BAISSE · VÄNDER, STARK eller UPPTREND och cykeln är TIDIG eller MITTEN
  GUL   fasen är STARK eller UPPTREND men cykeln är SEN (sent i uppgången)
  RÖD   fasen är BAISSE eller SVAG, eller cykeln är i TOPP
Cykelstart = fasen är BAISSE · VÄNDER: segmentet har varit i baisse och vänder — där contrarian-
strategin letar.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from berserk import signals as bsg

# Blindspots gränser och vikter (blindspot/theme_board.py — testerna kontrollerar att de är lika)
CYKEL_TIDIG, CYKEL_SEN, CYKEL_TOPP = 30.0, 70.0, 90.0
HAT_W_5Y_HIGH, HAT_W_VOL, HAT_W_RS = 40, 30, 30
PCT_WINDOW, PCT_MIN = 2520, 504          # tio år, minst två (Blindspot använder den historik som finns)
WEEKS_EMA, WEEKS_SLOPE, WEEKS_MIN = 200, 4, 204

VANDER, BAISSE, STARK, UPPTREND, SVAG, NODATA = "BAISSE · VÄNDER", "BAISSE", "STARK", "UPPTREND", "SVAG", "DATA SAKNAS"
TIDIG, MITTEN, SEN, TOPP = "TIDIG", "MITTEN", "SEN", "TOPP"
GRON, GUL, ROD = "GRÖN", "GUL", "RÖD"


def _clean(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None:
        return None
    s = s.astype(float).dropna()
    if getattr(s.index, "tz", None) is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s if len(s) else None


def phase_series(close: pd.Series) -> pd.Series:
    """BERSERK-fasen per dag (samma ordning som berserk.live.phase_of). DATA SAKNAS före SMA200."""
    f = bsg.driver_frame(close)
    ready = f["sma200"].notna()
    out = np.select([f["bear"] & f["turn"], f["bear"], f["strong"] & f["uptrend"], f["uptrend"]],
                    [VANDER, BAISSE, STARK, UPPTREND], default=SVAG)
    return pd.Series(np.where(ready, out, NODATA), index=f.index)


def weekly_slope(close: pd.Series) -> pd.Series:
    """200-veckors EMA:ns lutning över fyra veckor (%), lagd på dagarna från veckans sista handelsdag."""
    w = close.resample("W-FRI").last().dropna()
    ema = w.ewm(span=WEEKS_EMA, adjust=False).mean()
    slope = (ema / ema.shift(WEEKS_SLOPE) - 1) * 100
    slope[np.arange(len(w)) < WEEKS_MIN - 1] = np.nan
    return slope.reindex(slope.index.union(close.index)).ffill().reindex(close.index)


def cykel_label(pct: float, slope: Optional[float]) -> str:
    """Blindspots cykelläge ur tioårspercentilen (0–100) och 200-veckorslutningen."""
    if pct is None or pd.isna(pct):
        return NODATA
    if pct >= CYKEL_TOPP:
        return TOPP
    if pct >= CYKEL_SEN:
        return SEN
    if pct <= CYKEL_TIDIG:
        return TIDIG
    return MITTEN if slope is not None and not pd.isna(slope) and slope > 0 else TIDIG


def hat_frame(close: pd.Series, volume: Optional[pd.Series] = None,
              spy: Optional[pd.Series] = None) -> pd.DataFrame:
    """Blindspots hat-poäng per dag (0–100, högre = mer hatat) och dess tre delar."""
    idx = close.index
    high5 = close.rolling(1260, min_periods=1).max()
    from_high = ((high5 - close) / high5 * 100).clip(lower=0)
    hat_high = (from_high / 60.0).clip(upper=1.0) * 100
    hat_vol = pd.Series(50.0, index=idx)
    vol_chg = pd.Series(np.nan, index=idx)
    v = _clean(volume)
    if v is not None:
        v = v.reindex(idx)
        now, prev = v.rolling(252, min_periods=200).mean(), v.shift(252).rolling(252, min_periods=200).mean()
        vol_chg = (now / prev.replace(0, np.nan) - 1) * 100
        hat_vol = (50.0 - vol_chg).clip(0, 100).fillna(50.0)
    hat_rs = pd.Series(50.0, index=idx)
    rs = pd.Series(np.nan, index=idx)
    s = bsg.align(_clean(spy), idx) if spy is not None else None
    if s is not None:
        rs = ((close / close.shift(252) - 1) - (s / s.shift(252) - 1)) * 100
        hat_rs = (50.0 - rs / 0.6).clip(0, 100).fillna(50.0)
    hat = hat_high * HAT_W_5Y_HIGH / 100 + hat_vol * HAT_W_VOL / 100 + hat_rs * HAT_W_RS / 100
    return pd.DataFrame({"hat": hat, "from_high5": from_high, "vol_chg": vol_chg, "rs_spy": rs}, index=idx)


def status_of(phase: str, cykel: str) -> str:
    if phase == NODATA or cykel == NODATA:
        return NODATA
    if phase in (BAISSE, SVAG) or cykel == TOPP:
        return ROD
    if phase in (STARK, UPPTREND) and cykel == SEN:
        return GUL
    return GRON


def cycle_frame(close: pd.Series, volume: Optional[pd.Series] = None, spy: Optional[pd.Series] = None,
                necessity: int = 50) -> pd.DataFrame:
    """Hela cykelbilden per dag: fas, tioårspercentil, veckolutning, cykelläge, hat, blindspot-poäng,
    status och cykelstart. Varje rad bygger bara på data fram till den dagen."""
    c = _clean(close)
    if c is None:
        return pd.DataFrame()
    out = pd.DataFrame(index=c.index)
    out["close"] = c
    out["phase"] = phase_series(c)
    out["pct10"] = c.rolling(PCT_WINDOW, min_periods=PCT_MIN).rank(pct=True) * 100
    out["slope200w"] = weekly_slope(c)
    out["cykel"] = [cykel_label(p, s) for p, s in zip(out["pct10"], out["slope200w"])]
    hf = hat_frame(c, volume, spy)
    out = out.join(hf)
    out["blindspot"] = (necessity / 100.0) * (out["hat"] / 100.0) * ((100 - out["pct10"]) / 100.0) * 100
    out["status"] = [status_of(p, k) for p, k in zip(out["phase"], out["cykel"])]
    out["cycle_start"] = out["phase"] == VANDER
    return out


def state(frame: pd.DataFrame) -> dict:
    """Dagens läge ur cycle_frame (sista raden) — tomt läge när serien saknas."""
    if frame is None or frame.empty:
        return {"status": NODATA, "phase": NODATA, "cykel": NODATA, "date": None}
    last = frame.iloc[-1]

    def num(v, nd=1):
        return None if v is None or pd.isna(v) else round(float(v), nd)

    since = None
    st = frame["status"]
    changed = st.ne(st.shift()).cumsum()
    run = st[changed == changed.iloc[-1]]
    if len(run):
        since = str(run.index[0].date())
    return {"date": str(frame.index[-1].date()), "close": num(last["close"], 4), "phase": last["phase"],
            "cykel": last["cykel"], "status": last["status"], "status_since": since,
            "cycle_start": bool(last["cycle_start"]), "pct10": num(last["pct10"]),
            "slope200w": num(last["slope200w"], 2), "hat": num(last["hat"]), "blindspot": num(last["blindspot"]),
            "from_high5": num(last["from_high5"]), "rs_spy": num(last["rs_spy"]), "vol_chg": num(last["vol_chg"])}
