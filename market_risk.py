"""
market_risk.py — 🌩️ Marknadsrisk: en riskmodell, ingen kristallkula.

Frågan modellen svarar på: "När de här varningssignalerna har lyst tidigare,
hur ofta föll marknaden minst 10 % inom tre månader — jämfört med normalt?"

  Mål       framtida lägsta stängning inom 63 handelsdagar ≤ dagens × 0,90
  Signaler  nio på/av-signaler med förklaring, satta i förväg (inte optimerade):
              trend: under SMA200 · dödskors (SMA50 < SMA200)
              bredd: nära toppen men sektorbredden har rasat (bara SPY)
              volatilitet: VIX-spik från låg nivå · VIX över VIX3M
              kredit: high yield (HYG) tappar mot statsobligationer (IEF)
              räntekurva: T10Y2Y positiv igen efter inversion (FRED)
              rotation: defensiva sektorer drar ifrån offensiva
              eufori: indexet långt över SMA200
  Poäng     antal aktiva signaler. Nivå: LÅG 0–1 · FÖRHÖJD 2–3 · HÖG ≥ 4
  Kalibrering  varje dag sedan alla signaler har data: träffandel per nivå mot
               basfrekvensen, plus nedgångsepisoder (varnade / missade) och
               larm (träffar / falsklarm). Bara data fram till dagen — inget
               look-ahead i signalerna; målet tittar framåt per definition.

OMXS30 får samma globala signaler (VIX, kredit, kurva, rotation) — de
amerikanska riskmåtten är globala — men ingen bredd (bolagsdata saknas).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKETS = {"SPY": {"label": "S&P 500 (SPY)", "ticker": "SPY", "breadth": True},
           "OMXS30": {"label": "OMXS30", "ticker": "^OMX", "breadth": False}}
HORIZON = 63
DRAWDOWN = 0.10
LEVELS = (("LÅG", 0), ("FÖRHÖJD", 2), ("HÖG", 4))      # (namn, lägsta poäng)
BREADTH_ETFS = ("XLE", "XLF", "XLK", "XLV", "XLI", "XLB", "XLY", "XLP", "XLU")   # finns sedan 1998
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

SIGNALS = {
    "below_200": ("Under SMA200", "Indexet stänger under sitt 200-dagars glidande medel."),
    "death_cross": ("Dödskors", "SMA50 under SMA200 — den medellånga trenden har vänt ner."),
    "breadth_div": ("Breddivergens", "Indexet inom 3 % från årshögsta men andelen sektorer över EMA50 har "
                                     "fallit minst 30 procentenheter från sin topp de senaste tre månaderna."),
    "vix_spike": ("VIX-spik", "VIX minst 50 % över sin lägsta nivå de senaste 20 dagarna och över 20."),
    "vix_inverted": ("VIX inverterad", "VIX över VIX3M — oron på kort sikt är större än på tre månader."),
    "credit": ("Kreditstress", "HYG/IEF under sitt SMA50 och ner mer än 2 % på en månad — high yield tappar."),
    "curve": ("Räntekurvan vänder", "T10Y2Y positiv igen efter att ha varit inverterad de senaste två åren."),
    "rotation": ("Defensiv rotation", "XLU+XLP har gått minst 5 % bättre än XLK+XLY de senaste tre månaderna."),
    "stretch": ("Eufori", "Indexet mer än 15 % över SMA200."),
}


# ── Hjälpare ────────────────────────────────────────────────────────────────
def _close(df) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    s = df["Close"] if isinstance(df, pd.DataFrame) else df
    s = s.astype(float).dropna()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s if len(s) else None


def _on(s: Optional[pd.Series], idx) -> pd.Series:
    """Serien på indexets dagar med senast KÄNDA värde (ffill), NaN före start."""
    if s is None or len(s) == 0:
        return pd.Series(np.nan, index=idx)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


# ── Signalerna (kausala) ────────────────────────────────────────────────────
def compute_signals(index: pd.Series, data: dict, breadth: bool = True) -> tuple:
    """(aktiv: DataFrame bool, tillgänglig: DataFrame bool). data: ticker/FRED-id → serie."""
    idx = index.index
    c = index
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    act, avail = pd.DataFrame(index=idx), pd.DataFrame(index=idx)

    act["below_200"], avail["below_200"] = c < sma200, sma200.notna()
    act["death_cross"], avail["death_cross"] = sma50 < sma200, sma200.notna()
    act["stretch"], avail["stretch"] = c / sma200 - 1 > 0.15, sma200.notna()

    if breadth:
        etfs = {t: _close(data.get(t)) for t in BREADTH_ETFS}
        if all(s is not None for s in etfs.values()):
            df = pd.concat(etfs, axis=1, join="inner").dropna()
            pct = pd.concat({t: df[t] > _ema(df[t], 50) for t in df.columns}, axis=1).sum(axis=1) / len(df.columns) * 100
            pct = _on(pct.iloc[50:], idx)
            near_top = c >= 0.97 * c.rolling(252, min_periods=50).max()
            act["breadth_div"] = near_top & (pct <= pct.rolling(63, min_periods=20).max() - 30)
            avail["breadth_div"] = pct.notna()
        else:
            act["breadth_div"], avail["breadth_div"] = False, False

    vix, vix3m = _on(_close(data.get("^VIX")), idx), _on(_close(data.get("^VIX3M")), idx)
    act["vix_spike"] = (vix >= 1.5 * vix.rolling(20, min_periods=10).min()) & (vix > 20)
    avail["vix_spike"] = vix.notna()
    act["vix_inverted"], avail["vix_inverted"] = vix > vix3m, vix.notna() & vix3m.notna()

    hyg, ief = _close(data.get("HYG")), _close(data.get("IEF"))
    if hyg is not None and ief is not None:
        ratio = (hyg / ief).dropna()
        ok = (ratio < ratio.rolling(50).mean()) & (ratio / ratio.shift(21) - 1 < -0.02)
        act["credit"], avail["credit"] = _on(ok.astype(float), idx) > 0, _on(ratio.rolling(50).mean(), idx).notna()
    else:
        act["credit"], avail["credit"] = False, False

    curve = _close(data.get("T10Y2Y"))
    if curve is not None:
        was_inv = (curve <= 0).astype(float).rolling(504, min_periods=1).max() > 0
        ok = was_inv & (curve > 0)
        act["curve"], avail["curve"] = _on(ok.astype(float), idx) > 0, _on(curve, idx).notna()
    else:
        act["curve"], avail["curve"] = False, False

    legs = {t: _close(data.get(t)) for t in ("XLU", "XLP", "XLK", "XLY")}
    if all(s is not None for s in legs.values()):
        df = pd.concat(legs, axis=1, join="inner").dropna()
        rel = (df["XLU"] + df["XLP"]) / (df["XLK"] + df["XLY"])
        ok = rel / rel.shift(63) - 1 > 0.05
        act["rotation"], avail["rotation"] = _on(ok.astype(float), idx) > 0, _on(rel.shift(63), idx).notna()
    else:
        act["rotation"], avail["rotation"] = False, False

    act = act.fillna(False).astype(bool)
    avail = avail.fillna(False).astype(bool)
    return act & avail, avail


def score(active: pd.DataFrame) -> pd.Series:
    return active.sum(axis=1).astype(int)


def level_of(points: int) -> str:
    name = LEVELS[0][0]
    for n, lo in LEVELS:
        if points >= lo:
            name = n
    return name


# ── Målet och episoderna ────────────────────────────────────────────────────
def forward_event(close: pd.Series, horizon: int = HORIZON, drawdown: float = DRAWDOWN) -> pd.Series:
    """1.0 om lägsta stängning de kommande `horizon` dagarna ≤ dagens × (1 − drawdown),
    0.0 annars, NaN de sista dagarna där framtiden inte är känd än."""
    fut_min = close[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    ev = (fut_min <= close * (1 - drawdown)).astype(float)
    ev[fut_min.isna()] = np.nan
    return ev


def drawdown_episodes(close: pd.Series, drawdown: float = DRAWDOWN) -> list:
    """[(topp, datum då −drawdown nåddes, botten, djup %)] — en episod per topp."""
    out, peak, peak_d, in_dd, trough, trough_d, cross_d = [], None, None, False, None, None, None
    for d, v in close.items():
        if peak is None or v > peak:
            if in_dd:
                out.append((peak_d, cross_d, trough_d, round((trough / peak - 1) * 100, 1)))
                in_dd = False
            peak, peak_d = v, d
            continue
        if not in_dd and v <= peak * (1 - drawdown):
            in_dd, cross_d, trough, trough_d = True, d, v, d
        if in_dd and v < trough:
            trough, trough_d = v, d
    if in_dd:
        out.append((peak_d, cross_d, trough_d, round((trough / peak - 1) * 100, 1)))
    return out


# ── Kalibreringen ───────────────────────────────────────────────────────────
@dataclass
class Calibration:
    start: str
    end: str
    days: int
    base_rate: float
    by_level: dict = field(default_factory=dict)          # nivå → {days, hit_rate}
    episodes: list = field(default_factory=list)          # [{peak, cross, trough, depth, warned}]
    alarms: dict = field(default_factory=dict)            # {total, hits, false}


def calibrate(close: pd.Series, active: pd.DataFrame, avail: pd.DataFrame,
              horizon: int = HORIZON, drawdown: float = DRAWDOWN) -> Optional[Calibration]:
    full = avail.all(axis=1)
    if not full.any():
        return None
    start = full.idxmax()                                  # första dagen alla signaler har data
    c = close[close.index >= start]
    pts = score(active)[c.index]
    ev = forward_event(close, horizon, drawdown)[c.index]
    known = ev.notna()
    if known.sum() == 0:
        return None
    lv = pts.map(level_of)
    cal = Calibration(str(c.index[0].date()), str(c.index[-1].date()), int(known.sum()),
                      round(float(ev[known].mean()) * 100, 1))
    for name, _lo in LEVELS:
        m = known & (lv == name)
        cal.by_level[name] = {"days": int(m.sum()),
                              "hit_rate": round(float(ev[m].mean()) * 100, 1) if m.any() else None}
    high = lv == LEVELS[-1][0]
    warn = lv != LEVELS[0][0]
    for peak, cross, trough, depth in drawdown_episodes(c, drawdown):
        before = warn[(warn.index < cross) & (warn.index >= cross - pd.tseries.offsets.BDay(horizon))]
        cal.episodes.append({"peak": str(peak.date()), "cross": str(cross.date()), "trough": str(trough.date()),
                             "depth": depth, "warned": bool(before.any()),
                             "high": bool(high[before.index].any()) if len(before) else False})
    starts = high & ~high.shift(1, fill_value=False)
    total = hits = 0
    for d in starts[starts].index:
        if pd.isna(ev.get(d, np.nan)):
            continue
        total += 1
        hits += int(ev[d] == 1.0)
    cal.alarms = {"total": total, "hits": hits, "false": total - hits}
    return cal


# ── Resultat ────────────────────────────────────────────────────────────────
@dataclass
class MarketRisk:
    market: str
    label: str
    date: Optional[str] = None
    price: Optional[float] = None
    points: Optional[int] = None
    level: Optional[str] = None
    signals: list = field(default_factory=list)            # [{key, label, why, active, available}]
    calibration: Optional[Calibration] = None
    error: Optional[str] = None
    close: Optional[pd.Series] = None                      # indexet (för grafen)
    history: Optional[pd.Series] = None                    # poäng per dag

    @property
    def possible(self) -> int:
        return sum(1 for s in self.signals if s["available"])


def _fred_default(series_id: str) -> Optional[pd.Series]:
    """FRED-serie med datum (cachad 12 h i processen). None vid fel."""
    hit = _FRED_CACHE.get(series_id)
    if hit and time.time() - hit[0] < 43_200:
        return hit[1]
    try:
        import requests
        r = requests.get(FRED_URL.format(id=series_id), timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "value"]
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=pd.to_datetime(df["date"])).dropna()
    except Exception as exc:
        logger.warning("FRED %s: %s", series_id, exc)
        return None
    _FRED_CACHE[series_id] = (time.time(), s)
    return s


_FRED_CACHE: dict = {}


def needed_tickers(market: str) -> list:
    base = [MARKETS[market]["ticker"], "^VIX", "^VIX3M", "HYG", "IEF", "XLU", "XLP", "XLK", "XLY"]
    if MARKETS[market]["breadth"]:
        base += [t for t in BREADTH_ETFS if t not in base]
    return base


def evaluate(market: str, getter: Optional[Callable] = None, fred_getter: Optional[Callable] = None,
             horizon: int = HORIZON, drawdown: float = DRAWDOWN) -> MarketRisk:
    cfg = MARKETS[market]
    out = MarketRisk(market, cfg["label"])
    if getter is None:
        from market_prices import ohlcv as getter
    fred_getter = fred_getter or _fred_default
    data = {}
    for t in needed_tickers(market):
        try:
            data[t] = getter(t, "max")
        except Exception as exc:
            logger.warning("marknadsrisk %s: %s", t, exc)
            data[t] = None
    data["T10Y2Y"] = fred_getter("T10Y2Y")
    close = _close(data.get(cfg["ticker"]))
    if close is None or len(close) < 260:
        out.error = f"DATA UNAVAILABLE — ingen kurshistorik för {cfg['ticker']}"
        return out
    active, avail = compute_signals(close, data, breadth=cfg["breadth"])
    last = close.index[-1]
    out.date, out.price = str(last.date()), round(float(close.iloc[-1]), 2)
    out.points = int(active.loc[last].sum())
    out.level = level_of(out.points)
    out.signals = [{"key": k, "label": SIGNALS[k][0], "why": SIGNALS[k][1], "active": bool(active.loc[last, k]),
                    "available": bool(avail.loc[last, k])} for k in active.columns]
    out.calibration = calibrate(close, active, avail, horizon, drawdown)
    out.close, out.history = close, score(active)
    return out
