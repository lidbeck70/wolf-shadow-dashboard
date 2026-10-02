"""
ovtlyr_nine.py — OVTLYR Nine (Slingshot Setup): marknad, sektor och aktie.

Strukturen följer OVTLYR:s publika beskrivning:
  MARKET 40 %  Trend · Signal · Breadth
  SECTOR 30 %  Fear & Greed · Breadth
  STOCK  30 %  Trend · Signal · Fear & Greed · Blocks

Panelen har ingen OVTLYR-data. Varje faktor räknas därför ur riktiga priser
(SPY, sektor-ETF:er, aktien) och är märkt WOLF APPROXIMATION — aldrig
OVTLYR:s egen beräkning. Definitionerna:

  Trend         EMA10 > EMA20 och kurs > EMA50
  Signal        kurs ≥ EMA20 (SPY under EMA20 = OVTLYR:s säljsignal för marknaden)
  Market breadth  andel av de 11 SPDR-sektor-ETF:erna över EMA50 ≥ 50 %
                  och inte fallande (≥ sitt EMA5) — ersätter OVTLYR:s Bull List
  Sector F&G    syntetiskt F&G (screener_ovtlyr) för sektor-ETF:en: under 90
                och stigande mot för fem dagar sedan
  Sector breadth  sektor-ETF:en över EMA20 och EMA20 stigande — ETF-trend i
                  stället för andel bolag (bolagsdata per sektor saknas)
  Stock F&G     samma syntetiska F&G för aktien
  Blocks        inga restriktiva order blocks (ovtlyr.indicators.orderblocks)

Datakvalitet: varje faktor bär värde, tidsstämpel, källa och status.
DATA UNAVAILABLE och STALE DATA räknas aldrig som PASS.
Nine (setup) är inte en entry — det avgör Viking Execution (PR 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

APPROXIMATION = "WOLF APPROXIMATION"
PASS, FAIL, UNAVAILABLE, STALE = "PASS", "FAIL", "DATA UNAVAILABLE", "STALE DATA"
FULL, NOT_ALIGNED = "FULL ALIGNMENT", "NOT ALIGNED"

MARKET_TICKER = "SPY"
SECTOR_ETFS = {"Energy": "XLE", "Basic Materials": "XLB", "Industrials": "XLI",
               "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Healthcare": "XLV",
               "Financial Services": "XLF", "Technology": "XLK", "Communication Services": "XLC",
               "Utilities": "XLU", "Real Estate": "XLRE"}
WEIGHTS = {"market": 40, "sector": 30, "stock": 30}
LAYER_SIZE = {"market": 3, "sector": 2, "stock": 4}
NINE_TOTAL = 9

BREADTH_MIN_PCT = 50.0             # minst hälften av sektorerna över EMA50
BREADTH_EMA = 5                    # "inte fallande" = andelen ≥ sitt EMA5
BREADTH_MIN_ETFS = 8               # färre ETF:er med data → DATA UNAVAILABLE
FG_MAX = 90.0                      # extrem girighet räknas inte som stigande läge
FG_LOOKBACK = 5                    # F&G stigande mot för så många dagar sedan
SECTOR_EMA_RISE_DAYS = 5
MIN_BARS = 60                      # kortare historik → DATA UNAVAILABLE
STALE_BDAYS = 3                    # senaste stängning äldre än så → STALE DATA
PERIOD = "1y"


@dataclass
class Factor:
    key: str
    label: str
    layer: str
    status: str                    # PASS | FAIL | DATA UNAVAILABLE | STALE DATA
    value: Optional[float] = None
    detail: str = ""
    timestamp: Optional[str] = None
    source: str = ""
    approximation: bool = True

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def as_dict(self) -> dict:
        return {"value": self.value, "timestamp": self.timestamp, "source": self.source,
                "status": "valid" if self.status in (PASS, FAIL) else self.status,
                "passed": self.passed, "detail": self.detail,
                "label": APPROXIMATION if self.approximation else "OVTLYR"}


@dataclass
class NineResult:
    ticker: str
    market: list = field(default_factory=list)
    sector: list = field(default_factory=list)
    stock: list = field(default_factory=list)
    sector_etf: Optional[str] = None
    etf_states: dict = field(default_factory=dict)      # ETF → (kurs/EMA50 − 1) i %
    sector_source: str = ""                             # varifrån sektorn kom (Börsdata / Yahoo)

    @property
    def factors(self) -> list:
        return self.market + self.sector + self.stock

    def layer(self, name: str) -> list:
        return getattr(self, name)

    def layer_passed(self, name: str) -> int:
        return sum(f.passed for f in self.layer(name))

    @property
    def passed(self) -> int:
        return sum(f.passed for f in self.factors)

    def layer_points(self, name: str) -> float:
        return self.layer_passed(name) / LAYER_SIZE[name] * WEIGHTS[name]

    @property
    def weighted(self) -> float:
        return round(sum(self.layer_points(n) for n in WEIGHTS), 1)

    @property
    def failed(self) -> list:
        return [f for f in self.factors if f.status == FAIL]

    @property
    def unavailable(self) -> list:
        return [f for f in self.factors if f.status in (UNAVAILABLE, STALE)]

    @property
    def status(self) -> str:
        if self.passed == NINE_TOTAL:
            return FULL
        return NOT_ALIGNED if self.failed else UNAVAILABLE

    def get(self, key: str) -> Optional[Factor]:
        return next((f for f in self.factors if f.key == key), None)

    def as_dict(self) -> dict:
        """Specens struktur: market/sector/stock, nine_score, nine_passed, nine_total."""
        out = {name: {f.key.split(".", 1)[1]: f.as_dict() for f in self.layer(name)} for name in WEIGHTS}
        out.update(nine_score=self.weighted, nine_passed=self.passed, nine_total=NINE_TOTAL,
                   status=self.status, sector_etf=self.sector_etf, label=APPROXIMATION)
        return out


# ── Hjälpare ────────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _close(df) -> Optional[pd.Series]:
    if df is None or len(df) == 0 or "Close" not in df:
        return None
    c = df["Close"].astype(float).dropna()
    return c if len(c) else None


def _stamp(s: pd.Series) -> str:
    return str(s.index[-1])[:10]


def _is_stale(s: pd.Series, today) -> bool:
    last = pd.Timestamp(s.index[-1])
    if last.tzinfo is not None:
        last = last.tz_localize(None)
    t = pd.Timestamp(today).normalize()
    return int(np.busday_count(last.date(), t.date())) > STALE_BDAYS


def _gate(key: str, label: str, layer: str, series: Optional[pd.Series], source: str, today,
          check: Callable[[pd.Series], tuple]) -> Factor:
    """Gemensam ram: saknad/kort/gammal data → aldrig PASS; annars check(series) → (ok, värde, text)."""
    if series is None or len(series) < MIN_BARS:
        n = 0 if series is None else len(series)
        return Factor(key, label, layer, UNAVAILABLE, detail=f"för lite data ({n} dagar, kräver {MIN_BARS})",
                      source=source)
    ts = _stamp(series)
    if _is_stale(series, today):
        return Factor(key, label, layer, STALE, detail=f"senaste stängning {ts}", timestamp=ts, source=source)
    try:
        ok, value, text = check(series)
    except Exception as exc:                            # pragma: no cover — skyddsnät
        logger.warning("nine %s: %s", key, exc)
        return Factor(key, label, layer, UNAVAILABLE, detail=f"beräkningen misslyckades ({exc})",
                      timestamp=ts, source=source)
    if ok is None:
        return Factor(key, label, layer, UNAVAILABLE, value=value, detail=text, timestamp=ts, source=source)
    return Factor(key, label, layer, PASS if ok else FAIL, value=value, detail=text, timestamp=ts, source=source)


# ── Faktorerna ──────────────────────────────────────────────────────────────
def _trend(c: pd.Series) -> tuple:
    e10, e20, e50 = (float(_ema(c, n).iloc[-1]) for n in (10, 20, 50))
    p = float(c.iloc[-1])
    ok = e10 > e20 and p > e50
    return ok, round(p, 2), f"EMA10 {e10:,.2f} {'>' if e10 > e20 else '≤'} EMA20 {e20:,.2f} · kurs {p:,.2f} " \
                            f"{'>' if p > e50 else '≤'} EMA50 {e50:,.2f}"


def _signal(c: pd.Series) -> tuple:
    e20, p = float(_ema(c, 20).iloc[-1]), float(c.iloc[-1])
    ok = p >= e20
    return ok, round(p, 2), f"kurs {p:,.2f} {'≥' if ok else '<'} EMA20 {e20:,.2f} → {'BUY' if ok else 'SELL'}"


def fear_greed(df: pd.DataFrame) -> Optional[float]:
    """Syntetiskt F&G (screener_ovtlyr._score_fear_greed). None vid för lite data."""
    if df is None or len(df) < 40 or not {"Close", "High", "Low", "Volume"} <= set(df.columns):
        return None
    from screener_ovtlyr import _score_fear_greed
    return float(_score_fear_greed(df))


def fear_greed_series(df: pd.DataFrame) -> pd.Series:
    """Samma syntetiska F&G som screener_ovtlyr._score_fear_greed, men för varje
    dag (bara data fram till dagen — används av backtestet utan look-ahead)."""
    from screener_ovtlyr import _rsi
    c, v = df["Close"].astype(float), df["Volume"].astype(float)
    rng = df["High"].astype(float) - df["Low"].astype(float)
    z = ((v - v.rolling(20).mean()) / np.maximum(1.0, v.rolling(20).std())).clip(-3, 3)
    ma20 = c.rolling(20).mean()
    ratio = rng.rolling(5).mean() / np.maximum(0.001, rng.rolling(20).mean())
    score = ((z + 3) / 6 * 20 + _rsi(c, 14) / 100 * 20
             + ((c - ma20) / ma20 + 0.05).div(0.10).mul(20).clip(0, 20)
             + ((1.5 - ratio) / 1.0 * 20).clip(0, 20)
             + (c.diff() > 0).astype(float).rolling(20).sum() / 20 * 20)
    return score.clip(0, 100).round(1)


def _fg_check(df: pd.DataFrame) -> Callable:
    def check(_c):
        now, prev = fear_greed(df), fear_greed(df.iloc[:-FG_LOOKBACK])
        if now is None or prev is None:
            return None, now, "F&G kunde inte räknas (OHLCV saknas)"
        ok = now >= prev and now < FG_MAX
        why = ("extrem girighet" if now >= FG_MAX else "stigande" if now >= prev else "fallande")
        return ok, round(now, 1), f"F&G {now:.0f} (för {FG_LOOKBACK} dagar sedan {prev:.0f}) — {why}"
    return check


def _sector_breadth(c: pd.Series) -> tuple:
    e20 = _ema(c, 20)
    p, now, before = float(c.iloc[-1]), float(e20.iloc[-1]), float(e20.iloc[-1 - SECTOR_EMA_RISE_DAYS])
    ok = p > now and now > before
    return ok, round(p, 2), (f"kurs {p:,.2f} {'>' if p > now else '≤'} EMA20 {now:,.2f}, EMA20 "
                             f"{'stigande' if now > before else 'fallande'} — ETF-trend i stället för andel bolag")


def breadth_series(etf_closes: dict) -> pd.Series:
    """Andel (%) av ETF:erna över sitt EMA50 per dag — bara dagar där alla har pris."""
    cols = {t: c for t, c in (etf_closes or {}).items() if c is not None and len(c) >= MIN_BARS}
    if len(cols) < BREADTH_MIN_ETFS:
        return pd.Series(dtype=float)
    df = pd.concat(cols, axis=1, join="inner").dropna()
    above = pd.concat({t: df[t] > _ema(df[t], 50) for t in df.columns}, axis=1)
    return (above.sum(axis=1) / len(df.columns) * 100).rename("breadth")


def _market_breadth(etf_closes: dict, today) -> tuple:
    s = breadth_series(etf_closes)
    n = len([c for c in (etf_closes or {}).values() if c is not None and len(c) >= MIN_BARS])
    if len(s) < MIN_BARS:
        return Factor("market.breadth", "Breadth", "market", UNAVAILABLE,
                      detail=f"sektor-ETF:er med data: {n} av 11 (kräver {BREADTH_MIN_ETFS})",
                      source="Yahoo Finance (11 SPDR-sektor-ETF:er)"), {}

    def check(b):
        ema = float(_ema(b, BREADTH_EMA).iloc[-1])
        now = float(b.iloc[-1])
        ok = now >= BREADTH_MIN_PCT and now >= ema
        return ok, round(now, 1), (f"{now:.0f} % av sektorerna över EMA50 (EMA{BREADTH_EMA} {ema:.0f} %) — "
                                   f"{'stigande/stabil' if now >= ema else 'fallande'}; kräver ≥ "
                                   f"{BREADTH_MIN_PCT:.0f} % och inte fallande")
    f = _gate("market.breadth", "Breadth", "market", s, "Yahoo Finance (11 SPDR-sektor-ETF:er)", today, check)
    states = {}
    for t, c in (etf_closes or {}).items():
        if c is not None and len(c) >= MIN_BARS:
            states[t] = round((float(c.iloc[-1]) / float(_ema(c, 50).iloc[-1]) - 1) * 100, 1)
    return f, states


def _blocks(df: pd.DataFrame, ob_analysis: Optional[dict]) -> Callable:
    def check(c):
        oa = ob_analysis
        if oa is None:
            from ovtlyr.indicators.orderblocks import classify_price_vs_ob, detect_orderblocks
            obs = detect_orderblocks(df)
            oa = classify_price_vs_ob(float(c.iloc[-1]), obs) if obs else {"signal_bias": "HOLD"}
        bias = str(oa.get("signal_bias", "HOLD")).upper()
        near = bool(oa.get("approaching_bearish", False))
        ok = bias not in ("SELL", "REDUCE") and not near
        return ok, None, (f"order block-bias {bias}" + (", nära ett bearish block" if near else "")
                          + (" — fritt" if ok else " — restriktivt"))
    return check


# ── Huvudfunktionen ─────────────────────────────────────────────────────────
def compute(ticker: str, stock_df: Optional[pd.DataFrame], spy_df: Optional[pd.DataFrame],
            sector_etf: Optional[str], sector_df: Optional[pd.DataFrame], etf_closes: dict,
            ob_analysis: Optional[dict] = None, today=None) -> NineResult:
    """Ren funktion: alla priser in, NineResult ut. Inget nätverk."""
    today = today if today is not None else pd.Timestamp.today()
    spy_c, stock_c, sec_c = _close(spy_df), _close(stock_df), _close(sector_df)
    yahoo = "Yahoo Finance"
    r = NineResult(ticker=ticker, sector_etf=sector_etf)

    r.market = [_gate("market.trend", "Trend", "market", spy_c, f"{yahoo} {MARKET_TICKER}", today, _trend),
                _gate("market.signal", "Signal", "market", spy_c, f"{yahoo} {MARKET_TICKER}", today, _signal)]
    breadth, r.etf_states = _market_breadth(etf_closes, today)
    r.market.append(breadth)

    if not sector_etf:
        r.sector = [Factor("sector.fear_greed", "Fear & Greed", "sector", UNAVAILABLE,
                           detail="aktiens sektor okänd — ingen sektor-ETF", source=yahoo),
                    Factor("sector.breadth", "Breadth", "sector", UNAVAILABLE,
                           detail="aktiens sektor okänd — ingen sektor-ETF", source=yahoo)]
    else:
        src = f"{yahoo} {sector_etf}"
        r.sector = [_gate("sector.fear_greed", "Fear & Greed", "sector", sec_c, src, today, _fg_check(sector_df)),
                    _gate("sector.breadth", "Breadth", "sector", sec_c, src, today, _sector_breadth)]

    src = f"{yahoo} {ticker}"
    r.stock = [_gate("stock.trend", "Trend", "stock", stock_c, src, today, _trend),
               _gate("stock.signal", "Signal", "stock", stock_c, src, today, _signal),
               _gate("stock.fear_greed", "Fear & Greed", "stock", stock_c, src, today, _fg_check(stock_df)),
               _gate("stock.blocks", "Blocks", "stock", stock_c, f"{src} (order blocks)", today,
                     _blocks(stock_df, ob_analysis))]
    return r


# ── Data ────────────────────────────────────────────────────────────────────
def sector_etf_for(sector: Optional[str]) -> Optional[str]:
    return SECTOR_ETFS.get(str(sector or "").strip())


# Börsdatas sektorId (1–10, contrarian_alpha.necessity.BORSDATA_SECTOR_MAP) → SPDR-ETF
BORSDATA_SECTOR_ETFS = {1: ("Finans & Fastighet", "XLF"), 2: ("Dagligvaror", "XLP"), 3: ("Energi", "XLE"),
                        4: ("Hälsovård", "XLV"), 5: ("Industri", "XLI"), 6: ("Informationsteknik", "XLK"),
                        7: ("Material", "XLB"), 8: ("Sällanköpsvaror", "XLY"), 9: ("Telekommunikation", "XLC"),
                        10: ("Kraftförsörjning", "XLU")}
SECTOR_TTL_OK, SECTOR_TTL_FAIL = 86_400, 600       # ett misslyckat uppslag provas igen efter tio minuter
_SECTOR_CACHE: dict = {}
_BD_STATE: dict = {}


def _cached(key, fn):
    import time
    hit = _SECTOR_CACHE.get(key)
    if hit and time.time() - hit[0] < (SECTOR_TTL_OK if hit[1] is not None else SECTOR_TTL_FAIL):
        return hit[1]
    try:
        val = fn()
    except Exception as exc:
        logger.debug("sektor %s: %s", key, exc)
        val = None
    _SECTOR_CACHE[key] = (time.time(), val)
    return val


def _sector_default(ticker: str) -> Optional[str]:
    """yfinance-sektorn ('Technology' …). Bara lyckade svar cachas länge."""
    def fetch():
        import yfinance as yf
        return (yf.Ticker(ticker).info or {}).get("sector") or None
    return _cached(("yf", str(ticker).upper()), fetch)


def _bd_sector_id(ticker: str) -> Optional[int]:
    """Börsdatas sektorId för tickern — nordiska listan först, sedan den globala."""
    def fetch():
        api = _BD_STATE.get("api")
        if api is None:
            from borsdata_api import BorsdataAPI
            api = BorsdataAPI()
            _BD_STATE["api"] = api
        if not api.is_configured:
            return None
        iid = api.resolve_instrument_id(str(ticker))
        if iid is not None and api._id_map and iid in api._id_map:
            return api._id_map[iid].get("sectorId")
        glob = _BD_STATE.get("global")
        if glob is None:
            glob = {str(i.get("ticker", "")).upper(): i.get("sectorId") for i in api.get_global_instruments_list()}
            _BD_STATE["global"] = glob
        return glob.get(str(ticker).upper())
    return _cached(("bd", str(ticker).upper()), fetch)


def resolve_sector(ticker: str, sector_getter: Optional[Callable] = None,
                   bd_sector: Optional[Callable] = None) -> tuple:
    """(sektor-ETF | None, källtext). Börsdata först (inga Yahoo-anrop, finns för
    nordiska bolag), sedan Yahoos sektor."""
    tried = []
    sid = (bd_sector or _bd_sector_id)(ticker)
    if sid in BORSDATA_SECTOR_ETFS:
        name, etf = BORSDATA_SECTOR_ETFS[sid]
        return etf, f"Börsdata · {name} → {etf}"
    tried.append("Börsdata" + (f" (sektorId {sid})" if sid is not None else ""))
    sec = (sector_getter or _sector_default)(ticker)
    etf = sector_etf_for(sec)
    if etf:
        return etf, f"Yahoo · {sec} → {etf}"
    tried.append("Yahoo" + (f" ({sec})" if sec else ""))
    return None, "sektor okänd — provade " + " och ".join(tried)


def evaluate(ticker: str, stock_df: Optional[pd.DataFrame] = None, ob_analysis: Optional[dict] = None,
             getter: Optional[Callable] = None, sector_getter: Optional[Callable] = None,
             today=None, bd_sector: Optional[Callable] = None) -> NineResult:
    """Hämtar SPY, sektor-ETF:erna och (vid behov) aktien via den delade
    priscachen och räknar Nine. Saknad data blir DATA UNAVAILABLE."""
    if getter is None:
        from market_prices import ohlcv as getter

    def _get(t):
        try:
            df = getter(t, PERIOD)
        except Exception as exc:
            logger.warning("nine: %s: %s", t, exc)
            return None
        if df is None or len(df) == 0:
            return None
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df

    if stock_df is None:
        stock_df = _get(ticker)
    spy = _get(MARKET_TICKER)
    etfs = {t: _get(t) for t in SECTOR_ETFS.values()}
    etf_closes = {t: _close(d) for t, d in etfs.items()}
    sector_etf, sector_src = resolve_sector(ticker, sector_getter, bd_sector)
    r = compute(ticker, stock_df, spy, sector_etf, etfs.get(sector_etf) if sector_etf else None,
                etf_closes, ob_analysis=ob_analysis, today=today)
    r.sector_source = sector_src
    if sector_etf is None:
        for f in r.sector:
            f.detail = sector_src
    return r
