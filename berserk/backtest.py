"""
berserk/backtest.py — backtest av 🪓 BERSERK, rapporterat i R, utan look-ahead.

  * Signalen bedöms på signaldagens stängning; entry nästa dags öppning.
  * Stoppar ligger intradag (low ≤ stopp → exit på stoppen, eller öppningen vid
    gap under). Stängningsregler ger exit nästa dags öppning.
  * En position åt gången per ticker. Flera setups samma dag → S2, S1, S3.
  * Drivaren per tema: den första i preferensordningen som har data från
    periodens start (skog 2008 → WOOD, skog 2023 → trävaruterminen). Saknas en
    sådan används den med längst historik — teman utan drivare får bara S3.

Exit per setup (R = (exit − entry) / initial risk):
  S1  stopp 2 ATR · stängning ≥ +1 ATR → stopp entry − 0,25 ATR · stängning ≥ +2 ATR
      → trailing: stängning under EMA20 · tesen bruten: råvaran under SMA50 ·
      tidsstopp: 15 dagar utan stängning ≥ +0,5 ATR
  S2  stopp 2 ATR · stängning ≥ +2 ATR → trailing: stängning under EMA50 · tesen
      bruten: råvaran under EMA50
  S3  katastrofstopp 3 ATR · stängning över SMA5 · efter 5 dagar

Affärerna är viking_backtest.Trade, så portföljläget (viking_portfolio) och
robusthetsverktygen (viking_robustness) fungerar oförändrade. features bär
setup, tema, komplex, risk per setup och mått för kantanalysen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

import ovtlyr_nine as on
import viking_backtest as vb
import viking_portfolio as vp
from berserk import signals as sg
from berserk import themes as th
from berserk import universe as uv

WARMUP_BARS = 260
RISK_BY_SETUP = {sg.S1: 1.25, sg.S2: 1.25, sg.S3: 1.0}      # % av kapitalet
STOP_ATR = {sg.S1: 2.0, sg.S2: 2.0, sg.S3: 3.0}
S1_BE_ATR, S1_BE_GIVE, S1_TRAIL_ATR, S1_TIME_DAYS, S1_TIME_ATR = 1.0, 0.25, 2.0, 15, 0.5
S2_TRAIL_ATR = 2.0
S3_MAX_DAYS = 5
MIN_TURNOVER_M = 5.0             # miljoner i lokal valuta per dag, snitt 20 d (USA: USD)

NOTES = (
    "Entry på nästa dags öppning efter en stängd signaldag; stängningsregler ger exit på nästa öppning.",
    "Rapportspärren ingår inte — historiska rapportdatum saknas.",
    "Universum är dagens bolag: de som gick under på botten av en cykel saknas (överlevnadsbias) — slår "
    "hårdast mot S2.",
    "ETF:erna är amerikanska med lång historik; svenska ETC:er har samma exponering men inte identisk kurs. "
    "Terminsbaserade ETF:er (USO, UNG) tappar på rullningen — det ingår i kursen.",
    "London-aktier noteras i pence: likviditetsgränsen räknas i lokal enhet.",
    "Egen period: köpsignaler bara inom perioden, men öppna affärer löper klart på senare kurser — annars "
    "räknas förlorarna (stängs snabbt) men inte vinnarna som ännu ligger i trend vid periodens slut.",
)


@dataclass
class Config:
    setups: tuple = sg.SETUPS
    years: int = 5
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    min_turnover_m: float = MIN_TURNOVER_M
    market_gate: bool = True             # regionens index (universe.REGION_INDEX) över SMA200
    s3_regions: Optional[tuple] = None   # S3 bara i dessa regioner (None = alla; ETF:er räknas som USA)
    data_guard: bool = True              # inga signaler 20 dagar efter ett dagshopp > 60 % (signals.jump_block)
    commodity_gate: bool = False         # råvarugrinden: utanför commodity_free bara när DBC är över SMA200
    commodity_free: tuple = ("Norden",)  # regioner som handlas oavsett råvarugrind
    max_gap_atr: Optional[float] = None  # ingen entry när öppningen gappar mer än så många ATR över stängningen
    regions: Optional[tuple] = None      # bara dessa regioner ("Norden", "USA", "Kanada", "London", "ETF")
    exits_after_end: bool = True         # egen period: signaler inom perioden, affärerna löper klart på senare data
    risk_by_setup: dict = field(default_factory=lambda: dict(RISK_BY_SETUP))


MAX_HEAT_PCT = 6.0


def portfolio_config(max_heat: float = MAX_HEAT_PCT, max_s3: Optional[int] = None) -> vp.PortfolioConfig:
    """BERSERK:s portfölj: max 8 positioner, 20 % per position, 2 per tema, 4 per komplex, 6 % värme.
    max_s3 = högst så många samtidiga S3-positioner (None = ingen gräns) — så att S1/S2 får plats."""
    caps = (("complex", 4),) + ((("s3", int(max_s3)),) if max_s3 else ())
    return vp.PortfolioConfig(max_position_pct=20.0, max_positions=8, max_heat_pct=float(max_heat),
                              one_per_sector=True, sector_cap=2, group_caps=caps, max_daily_losses=2)


def pick_driver(theme: str, start, series: dict) -> tuple:
    """(symbol, serie) — första drivaren (preferensordning) med data från start; täcker ingen
    starten väljs den med längst historik."""
    cands = [(s, series.get(s)) for s in th.drivers(theme)]
    cands = [(s, x) for s, x in cands if x is not None and len(x.dropna()) > 0]
    for s, x in cands:
        if x.dropna().index[0] <= pd.Timestamp(start):
            return s, x
    return min(cands, key=lambda sx: sx[1].dropna().index[0]) if cands else (None, None)


def _rank(setup: str, f: pd.DataFrame, i: int) -> float:
    """Rangordning inom setupen (högre först): S1 mest efter råvaran, S2 mest hatad, S3 mest översåld."""
    if setup == sg.S1:
        return float(-f["divergence"].iloc[i])
    if setup == sg.S2:
        return float(-f["dd252"].iloc[i])
    return float(-f["rsi2"].iloc[i])


def region_key(ticker: str) -> str:
    """Region för filter och uppdelning — råvaru-ETF:erna som egen grupp."""
    return "ETF" if uv.kind_of(ticker) == "etf" else uv.region_of(ticker)


def backtest_ticker(ticker: str, stock: pd.DataFrame, driver: Optional[pd.Series], cfg: Config = Config(),
                    start=None, market: Optional[pd.Series] = None, driver_symbol: Optional[str] = None,
                    end=None, commodity: Optional[pd.Series] = None, frame: Optional[pd.DataFrame] = None) -> dict:
    """Affärerna för en ticker. end = sista signaldag (affärer får löpa vidare på senare data). frame =
    färdigräknad signals.frame (återanvänds mellan varianter — beror inte på cfg)."""
    stock = stock.dropna(subset=["Open", "High", "Low", "Close"])
    n = len(stock)
    theme = uv.theme_of(ticker)
    res = {"ticker": ticker, "trades": [], "signals": {s: 0 for s in sg.SETUPS}, "thin": 0, "market_blocked": 0,
           "data_blocked": 0, "commodity_blocked": 0, "gap_skipped": 0, "theme": theme, "driver": driver_symbol}
    if n < WARMUP_BARS + 2:
        return res
    is_etf = uv.kind_of(ticker) == "etf"
    f = frame if frame is not None else sg.frame(stock, driver, is_etf=is_etf, market=market, commodity=commodity)
    gated = cfg.commodity_gate and uv.region_of(ticker) not in cfg.commodity_free
    end = pd.Timestamp(end) if end is not None else None
    o, h, lo, c = (stock[k].astype(float).values for k in ("Open", "High", "Low", "Close"))
    idx = stock.index
    first = WARMUP_BARS
    if start is not None:
        first = max(first, int((idx < pd.Timestamp(start)).sum()))
    allowed = [s for s in cfg.setups if not (s == sg.S3 and cfg.s3_regions is not None
                                             and uv.region_of(ticker) not in cfg.s3_regions)]
    i = first
    while i < n - 1:
        if end is not None and idx[i] > end:
            break
        setups = [s for s in sorted(allowed, key=lambda s: -sg.PRIORITY[s]) if bool(f[s].iloc[i])]
        if not setups:
            i += 1
            continue
        setup = setups[0]
        res["signals"][setup] += 1
        if cfg.min_turnover_m > 0 and not (f["turnover20"].iloc[i] >= cfg.min_turnover_m * 1e6):
            res["thin"] += 1
            i += 1
            continue
        if cfg.data_guard and bool(f["data_jump"].iloc[i]):
            res["data_blocked"] += 1
            i += 1
            continue
        if cfg.market_gate and not bool(f["market_ok"].iloc[i]):
            res["market_blocked"] += 1
            i += 1
            continue
        if gated and not bool(f["commodity_ok"].iloc[i]):
            res["commodity_blocked"] += 1
            i += 1
            continue
        atr0 = float(f["atr"].iloc[i])
        entry = float(o[i + 1])
        stop = entry - STOP_ATR[setup] * atr0
        risk = entry - stop
        if not (risk > 0 and atr0 > 0):
            i += 1
            continue
        if cfg.max_gap_atr is not None and (entry - c[i]) / atr0 > cfg.max_gap_atr:
            res["gap_skipped"] += 1                            # öppningen sprang iväg — köp inte
            i += 1
            continue
        feats = {"setup": setup, "s3": "S3" if setup == sg.S3 else None, "theme": theme, "complex": th.complex_of(theme), "kind": "etf" if is_etf else
                 "producent", "risk_pct": cfg.risk_by_setup.get(setup, 1.0),
                 "atr_pct": round(atr0 / c[i] * 100, 2), "rvol": _num(f["rvol"].iloc[i]),
                 "rsi2": _num(f["rsi2"].iloc[i], 1), "divergence": _num(f["divergence"].iloc[i] * 100, 1),
                 "dd252": _num(f["dd252"].iloc[i] * 100, 1), "gap_atr": round((o[i + 1] - c[i]) / atr0, 2)}
        t = vb.Trade(ticker, str(idx[i].date()), str(idx[i + 1].date()), round(entry, 4), round(stop, 4),
                     round(risk, 4), sg.PRIORITY[setup], mom63=round(_rank(setup, f, i), 4), sector=theme,
                     features=feats)
        out = simulate_exit(o, h, lo, c, f, i, setup, entry, stop, atr0, close_at_end=True)
        j = out["exit_idx"] if out["exit_idx"] is not None else n
        if out["exit_idx"] is not None:
            t.exit, t.exit_reason, t.exit_date = out["exit"], out["reason"], str(idx[j].date())
        if t.exit is None:
            t.exit, t.exit_reason, t.exit_date, t.open = float(c[-1]), "öppen", str(idx[-1].date()), True
            j = n - 1
        last = min(j, n - 1)
        t.dates = tuple(str(d.date()) for d in idx[i + 1:last + 1])
        t.path = tuple(round(float(x), 4) for x in c[i + 1:last + 1])
        t.r = round((t.exit - entry) / risk, 3)
        t.days = int(np.busday_count(pd.Timestamp(t.entry_date).date(), pd.Timestamp(t.exit_date).date()))
        res["trades"].append(t)
        i = max(j, i + 1)
    return res


def simulate_exit(o, h, lo, c, f: pd.DataFrame, i: int, setup: str, entry: float, stop: float, atr0: float,
                  close_at_end: bool = True) -> dict:
    """Exitreglerna för en affär med signal dag i och entry dag i+1 — EN källa för backtest och papperskonto.

    Returnerar {exit, reason, exit_idx, pending, cur_stop, armed_be, trailing}. exit_idx = dagen exiten sker
    (None = fortfarande öppen). En stängningsregel som slår till sista dagen: close_at_end=True stänger på
    stängningen (backtest), False lämnar den som pending (live: sälj på nästa öppning)."""
    n = len(c)
    cur_stop, armed_be, trailing, best_close = stop, False, False, -np.inf
    j, pending = i + 1, None
    while j < n:
        if pending is not None:
            return {"exit": float(o[j]), "reason": pending, "exit_idx": j, "pending": None, "cur_stop": cur_stop,
                    "armed_be": armed_be, "trailing": trailing}
        if lo[j] <= cur_stop:
            px = float(o[j]) if o[j] < cur_stop else float(cur_stop)
            reason = "breakeven-stopp" if armed_be else ("katastrofstopp" if setup == sg.S3 else "stopp")
            return {"exit": px, "reason": reason, "exit_idx": j, "pending": None, "cur_stop": cur_stop,
                    "armed_be": armed_be, "trailing": trailing}
        held = j - i                                           # handelsdagar sedan entry (entrydagen = 1)
        best_close = max(best_close, float(c[j]))
        reason = None
        if setup == sg.S1:
            if trailing and c[j] < f["ema20"].iloc[j]:
                reason = "trailing EMA20"
            elif not bool(f["d_above_sma50"].iloc[j]):
                reason = "råvaran under SMA50"
            elif held >= S1_TIME_DAYS and best_close < entry + S1_TIME_ATR * atr0:
                reason = "tidsstopp"
            if c[j] >= entry + S1_BE_ATR * atr0 and not armed_be:
                armed_be, cur_stop = True, max(cur_stop, entry - S1_BE_GIVE * atr0)
            if c[j] >= entry + S1_TRAIL_ATR * atr0:
                trailing = True
        elif setup == sg.S2:
            if trailing and c[j] < f["ema50"].iloc[j]:
                reason = "trailing EMA50"
            elif not bool(f["d_above_ema50"].iloc[j]):
                reason = "råvaran under EMA50"
            if c[j] >= entry + S2_TRAIL_ATR * atr0:
                trailing = True
        else:
            if c[j] > f["sma5"].iloc[j]:
                reason = "över SMA5"
            elif held >= S3_MAX_DAYS:
                reason = "efter 5 dagar"
        if reason:
            pending = reason
            if j == n - 1 and close_at_end:                    # ingen nästa dag — stäng på stängningen
                return {"exit": float(c[j]), "reason": reason, "exit_idx": j, "pending": None,
                        "cur_stop": cur_stop, "armed_be": armed_be, "trailing": trailing}
        j += 1
    return {"exit": None, "reason": None, "exit_idx": None, "pending": pending, "cur_stop": cur_stop,
            "armed_be": armed_be, "trailing": trailing}


def _num(v, nd=2):
    try:
        return None if v is None or pd.isna(v) else round(float(v), nd)
    except (TypeError, ValueError):
        return None


def run(tickers: list, getter: Optional[Callable] = None, cfg: Config = Config(), progress: Optional[Callable] = None,
        today=None, nordic_provider: Optional[Callable] = None, frame_cache: Optional[dict] = None) -> dict:
    """Backtest över tickers (producenter och ETF:er ur berserk.universe; okända hoppas över). frame_cache
    (valfri dict) delar signals.frame mellan varianter med samma kursdata och period."""
    if getter is None:
        from market_prices import ohlcv as getter
    start, end, years = vb.period_of(cfg, today)
    fetch_years = int(np.ceil(((pd.Timestamp(today) if today is not None else pd.Timestamp.today()) - start).days
                              / 365.25)) + 2                       # uppvärmning: SMA200 och femårsintervallet
    period = f"{fetch_years}y" if cfg.start_year is None else "max"

    def _get(t):
        try:
            df = getter(t, period)
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        if not cfg.exits_after_end:
            df = df[df.index <= end]
        return df if len(df) else None

    if cfg.regions is not None:
        tickers = [t for t in tickers if not uv.theme_of(t) or region_key(t) in cfg.regions]

    themes = {uv.theme_of(t) for t in tickers if uv.theme_of(t)}
    series = {}
    for theme in themes:
        for s in th.drivers(theme):
            if s not in series:
                d = _get(s)
                series[s] = None if d is None else d["Close"].astype(float).dropna()
    chosen = {theme: pick_driver(theme, start, series) for theme in themes}
    # Marknadsgrinden per region: OMXS30 (Börsdata), SPY, TSX, FTSE — saknas ett index gäller SPY
    regions = {uv.region_of(t) for t in tickers}
    markets = {}
    for region in sorted(regions | {"USA"}):
        sym = uv.REGION_INDEX[region]
        if region == "Norden":
            nm = (nordic_provider or (lambda: on.nordic_market(fetch_years * 262)))()
            if nm and nm.get("close") is not None:
                markets[sym] = nm["close"].astype(float) if cfg.exits_after_end else vb._cut(
                    nm["close"].astype(float), end)
        else:
            d = _get(sym)
            if d is not None:
                markets[sym] = d["Close"].astype(float)
    dbc = _get("DBC")
    commodity = dbc["Close"].astype(float) if dbc is not None else None
    per, trades = [], []
    for k, t in enumerate(tickers):
        theme = uv.theme_of(t)
        df = _get(t) if theme else None
        if df is None:
            per.append({"ticker": t, "trades": [], "signals": {s: 0 for s in sg.SETUPS}, "thin": 0,
                        "market_blocked": 0, "theme": theme, "driver": None,
                        "error": "okänd ticker (inte i BERSERK-universumet)" if not theme else "DATA UNAVAILABLE"})
        else:
            sym, drv = chosen.get(theme, (None, None))
            idx_sym = uv.REGION_INDEX[uv.region_of(t)]
            mkt = markets.get(idx_sym) if markets.get(idx_sym) is not None else markets.get("SPY")
            key = (t, str(start.date()))
            fr = frame_cache.get(key) if frame_cache is not None else None
            if fr is None:
                fr = sg.frame(df.dropna(subset=["Open", "High", "Low", "Close"]), drv,
                              is_etf=uv.kind_of(t) == "etf", market=mkt, commodity=commodity)
                if frame_cache is not None:
                    frame_cache[key] = fr
            r = backtest_ticker(t, df, drv, cfg, start=start, market=mkt, driver_symbol=sym, end=end,
                                commodity=commodity, frame=fr)
            per.append(r)
            trades += r["trades"]
        if progress is not None:
            progress(k + 1, len(tickers), t)
    bench = {name: s[(s.index >= start) & (s.index <= end)] for name, s in markets.items() if s is not None}
    if commodity is not None:
        bench["DBC (råvarukorg)"] = commodity[(commodity.index >= start) & (commodity.index <= end)]
    return {"trades": trades, "per_ticker": per, "metrics": vb.metrics(trades), "notes": NOTES, "config": cfg,
            "period": {"start": str(start.date()), "end": str(end.date()), "years": round(years, 2)},
            "benchmarks": bench, "drivers": {theme: sym for theme, (sym, _x) in chosen.items()},
            "thin": sum(p.get("thin", 0) for p in per), "market_blocked": sum(p.get("market_blocked", 0) for p in per),
            "data_blocked": sum(p.get("data_blocked", 0) for p in per),
            "commodity_blocked": sum(p.get("commodity_blocked", 0) for p in per),
            "gap_skipped": sum(p.get("gap_skipped", 0) for p in per)}
