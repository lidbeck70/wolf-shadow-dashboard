"""
berserk/live.py — 🪓 BERSERK idag: regimen per råvarutema och skannern.

Samma regler som backtestet (signals.frame) på senaste STÄNGDA dag — en signal
idag betyder köp på nästa öppning. Rena funktioner med injicerbar hämtare, så
att sidorna (regime_ui, screen_ui) och den schemalagda skanningen (PR 3) delar
logiken.

  load_drivers / load_markets   drivare per tema och regionernas index
  theme_states                  per tema: trend, cykelläge (percentil av femårs-
                                intervallet), baisse/vändning, producenterna mot
                                råvaran och vilka setups som kan vara aktiva
  market_states                 regionernas index mot SMA200
  scan                          KÖP (setup idag + grindar), BEVAKA (setupen nästan
                                klar — triggern saknas) och plan: entry ≈ stängning,
                                stopp, positionsstorlek med BERSERK:s risk per setup
  portfolio_flags               tema-, komplex- och platsspärrar mot innehaven
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

import ovtlyr_nine as on
from berserk import backtest as bt
from berserk import signals as sg
from berserk import themes as th
from berserk import universe as uv

DRIVER_PERIOD, STOCK_PERIOD = "7y", "2y"      # femårsintervallet + uppvärmning · SMA200 och 52 veckor
KOP, BEVAKA, INGET = "KÖP", "BEVAKA", ""
WATCH_RSI2 = 25.0                              # S3 bevaka: RSI(2) under 25 i upptrend
PHASES = ("BAISSE · VÄNDER", "BAISSE", "UPPTREND", "STARK", "SVAG", "INGEN DRIVARE")


def _get_fn(getter: Optional[Callable]):
    if getter is None:
        from market_prices import ohlcv as getter
    return getter


def _close(df) -> Optional[pd.Series]:
    if df is None or len(df) == 0 or "Close" not in df:
        return None
    s = df["Close"].astype(float).dropna()
    if getattr(s.index, "tz", None) is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s if len(s) else None


def load_drivers(themes, getter: Optional[Callable] = None, today=None) -> dict:
    """{tema: (symbol, serie)} — första drivaren med fem års historik, annars den längsta."""
    getter = _get_fn(getter)
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    series = {}
    for theme in themes:
        for s in th.drivers(theme):
            if s not in series:
                try:
                    series[s] = _close(getter(s, DRIVER_PERIOD))
                except Exception:
                    series[s] = None
    return {theme: bt.pick_driver(theme, today - pd.DateOffset(years=5), series) for theme in themes}


def load_markets(regions, getter: Optional[Callable] = None, nordic_provider: Optional[Callable] = None) -> dict:
    """{indexsymbol: serie} för regionerna (OMXS30 ur Börsdata, övriga Yahoo)."""
    getter = _get_fn(getter)
    out = {}
    for region in sorted(set(regions) | {"USA"}):
        sym = uv.REGION_INDEX[region]
        try:
            if region == "Norden":
                nm = (nordic_provider or on.nordic_market)()
                s = nm.get("close") if nm else None
            else:
                s = _close(getter(sym, STOCK_PERIOD))
        except Exception:
            s = None
        if s is not None and len(s):
            out[sym] = s.astype(float)
    return out


# ── Regimen ─────────────────────────────────────────────────────────────────
def phase_of(row: dict) -> str:
    if not row.get("driver"):
        return "INGEN DRIVARE"
    if row["bear"] and row["turn"]:
        return "BAISSE · VÄNDER"
    if row["bear"]:
        return "BAISSE"
    if row["strong"] and row["uptrend"]:
        return "STARK"
    if row["uptrend"]:
        return "UPPTREND"
    return "SVAG"


def theme_states(drivers: dict, producer_closes: Optional[dict] = None) -> list:
    """En rad per tema: drivarens läge och producenternas divergens mot råvaran (median, 63 d)."""
    producer_closes = producer_closes or {}
    by = uv.by_theme(list(uv.PRODUCERS))
    rows = []
    for theme in th.THEMES:
        sym, d = drivers.get(theme, (None, None))
        row = {"theme": theme, "label": th.label(theme), "complex": th.complex_of(theme), "driver": sym,
               "close": None, "ret63": None, "dd252": None, "range5y": None, "bear": False, "turn": False,
               "strong": False, "uptrend": False, "producers": len(by.get(theme, [])), "divergence": None,
               "lagging": 0, "date": None}
        if d is not None and len(d) > 60:
            f = sg.driver_frame(d)
            last = f.iloc[-1]
            lo5 = d.rolling(1260, min_periods=252).min().iloc[-1]
            hi5 = d.rolling(1260, min_periods=252).max().iloc[-1]
            row.update(close=round(float(last["close"]), 4), date=str(d.index[-1].date()),
                       ret63=_pct(last["ret63"]), dd252=_pct(d.iloc[-1] / d.tail(252).max() - 1),
                       range5y=None if pd.isna(lo5) or hi5 == lo5 else round(float((d.iloc[-1] - lo5) / (hi5 - lo5)
                                                                                * 100), 0),
                       bear=bool(last["bear"]), turn=bool(last["turn"]), strong=bool(last["strong"]),
                       uptrend=bool(last["uptrend"]))
            divs = []
            for t in by.get(theme, []):
                c = producer_closes.get(t)
                if c is not None and len(c) > sg.RET_N and row["ret63"] is not None:
                    divs.append((float(c.iloc[-1] / c.iloc[-1 - sg.RET_N] - 1) * 100) - row["ret63"])
            if divs:
                row["divergence"] = round(float(np.median(divs)), 1)
                row["lagging"] = int(sum(1 for x in divs if x <= sg.DIV_LAG * 100))
        row["phase"] = phase_of(row)
        row["setups"] = active_setups(row)
        rows.append(row)
    return rows


def active_setups(row: dict) -> list:
    """Vilka setups temats läge tillåter just nu (aktierna avgör sedan om de triggar)."""
    out = []
    if not row.get("driver"):
        return [sg.S3]
    if row["strong"] and row["lagging"] > 0:
        out.append(sg.S1)
    if row["bear"] and row["turn"]:
        out.append(sg.S2)
    if row["uptrend"]:
        out.append(sg.S3)
    return out


def market_states(markets: dict) -> list:
    rows = []
    for region, sym in uv.REGION_INDEX.items():
        s = markets.get(sym)
        if s is None or len(s) < 200:
            rows.append({"region": region, "index": sym, "ok": None, "vs_sma200": None, "date": None})
            continue
        sma = s.rolling(200).mean().iloc[-1]
        rows.append({"region": region, "index": sym, "ok": bool(s.iloc[-1] > sma),
                     "vs_sma200": round(float(s.iloc[-1] / sma - 1) * 100, 1), "date": str(s.index[-1].date())})
    return rows


def _pct(v) -> Optional[float]:
    try:
        return None if v is None or pd.isna(v) else round(float(v) * 100, 1)
    except (TypeError, ValueError):
        return None


# ── Skannern ────────────────────────────────────────────────────────────────
def evaluate(ticker: str, stock: pd.DataFrame, driver: Optional[pd.Series], market: Optional[pd.Series],
             capital: float = 100_000.0, cfg: bt.Config = bt.Config(), driver_symbol: Optional[str] = None,
             commodity: Optional[pd.Series] = None) -> dict:
    """En rad: status (KÖP/BEVAKA/—), setup, varför, och planen för nästa öppning."""
    theme = uv.theme_of(ticker)
    row = {"ticker": ticker, "theme": theme, "label": th.label(theme), "complex": th.complex_of(theme),
           "region": uv.region_of(ticker), "kind": uv.kind_of(ticker), "driver": driver_symbol,
           "status": INGET, "setup": None, "why": [], "close": None, "stop": None, "position_pct": None,
           "shares": None, "risk_pct": None, "date": None, "error": None}
    stock = stock.dropna(subset=["Open", "High", "Low", "Close"]) if stock is not None else None
    if stock is None or len(stock) < 220:
        row["error"] = "DATA UNAVAILABLE — för lite kurshistorik"
        return row
    is_etf = row["kind"] == "etf"
    f = sg.frame(stock, driver, is_etf=is_etf, market=market if cfg.market_gate else None, commodity=commodity)
    last = f.iloc[-1]
    c = float(stock["Close"].iloc[-1])
    row.update(close=round(c, 4), date=str(stock.index[-1].date()), rsi2=_round(last["rsi2"], 1),
               rvol=_round(last["rvol"], 2), divergence=_round(last["divergence"] * 100, 1),
               dd252=_round(last["dd252"] * 100, 1))
    gates_ok = True
    if cfg.min_turnover_m > 0 and not (last["turnover20"] >= cfg.min_turnover_m * 1e6):
        row["why"].append(f"omsättning under {cfg.min_turnover_m:g} milj/dag")
        gates_ok = False
    if not bool(last["market_ok"]):
        row["why"].append(f"{uv.REGION_INDEX[row['region']]} under SMA200")
        gates_ok = False
    if cfg.regions is not None and bt.region_key(ticker) not in cfg.regions:
        row["why"].append("regionen ingår inte i reglerna")
        gates_ok = False
    if cfg.commodity_gate and row["region"] not in cfg.commodity_free and not bool(last["commodity_ok"]):
        row["why"].append(f"råvarugrind: {'koppar/guld-kvoten' if cfg.gate_kind == bt.GATE_CU_AU else 'råvarukorgen (DBC)'}"
                          " under SMA200")
        gates_ok = False
    if cfg.data_guard and bool(last["data_jump"]):
        row["why"].append(f"datavakt: kurshopp > {sg.JUMP_MAX * 100:.0f} % på en dag senaste "
                          f"{sg.JUMP_BLOCK_DAYS} dagarna — kontrollera split/utdelning")
        gates_ok = False
    setups = [s for s in cfg.setups if not (s == sg.S3 and cfg.s3_regions is not None
                                            and row["region"] not in cfg.s3_regions)]
    fired = [s for s in sorted(setups, key=lambda s: -sg.PRIORITY[s]) if bool(last[s])]
    if cfg.s1_top_block and sg.S1 in fired and bool(last["d_top"]):
        fired.remove(sg.S1)
        row["why"].append("S1 spärrad: råvaran i TOPP (tioårspercentil ≥ 90 — Blindspot)")
    row["satellite"] = "SAT" if bt.is_satellite(ticker, cfg) else None
    if fired:
        row["setup"] = fired[0]
        row["status"] = KOP if gates_ok else BEVAKA
        row["why"].insert(0, f"{fired[0]} idag")
    else:
        watch = watch_reason(f, is_etf, driver is not None)
        if watch:
            row["setup"], reason = watch
            row["status"] = BEVAKA
            row["why"].insert(0, reason)
    if row["setup"]:
        atr0 = float(last["atr"])
        stop = c - bt.STOP_ATR[row["setup"]] * atr0
        risk_pct = cfg.risk_by_setup.get(row["setup"], 1.0) * (cfg.satellite_risk if row["satellite"] else 1.0)
        stop_pct = (c - stop) / c * 100 if c > stop else None
        pos = min(risk_pct / stop_pct * 100, bt.portfolio_config().max_position_pct) if stop_pct else None
        row.update(stop=round(stop, 4), risk_pct=risk_pct, position_pct=None if pos is None else round(pos, 1),
                   shares=None if pos is None or c <= 0 else int(capital * pos / 100 // c),
                   atr=round(atr0, 4))
    return row


def watch_reason(f: pd.DataFrame, is_etf: bool, has_driver: bool) -> Optional[tuple]:
    """(setup, skäl) när setupen nästan är klar — allt utom triggern. S2 före S1 före S3."""
    last = f.iloc[-1]
    if bool(last["s2_setup"]):
        return sg.S2, "S2: råvaran vänder och aktien är hatad — väntar på 20-dagarshögsta med volym"
    if bool(last["s1_setup"]):
        return sg.S1, (f"S1: aktien {last['divergence'] * 100:+.0f} pe mot råvaran — väntar på vändning "
                       f"med volym")
    if bool(last["s3_trend"]) and last["rsi2"] < WATCH_RSI2:
        return sg.S3, (f"S3: RSI(2) {last['rsi2']:.0f} i upptrend — snapback vid RSI(2) < {sg.RSI2_MAX:g} och "
                       f"under nedre Bollinger")
    return None


def _round(v, nd):
    try:
        return None if v is None or pd.isna(v) else round(float(v), nd)
    except (TypeError, ValueError):
        return None


def scan(tickers: list, getter: Optional[Callable] = None, capital: float = 100_000.0, cfg: bt.Config = bt.Config(),
         today=None, nordic_provider: Optional[Callable] = None, progress: Optional[Callable] = None,
         drivers: Optional[dict] = None, markets: Optional[dict] = None, keep: Optional[dict] = None) -> dict:
    """Skanna listan: {rows, drivers, markets, when}. Okända tickers markeras. keep (valfri dict) fylls med
    {ticker: {stock, driver, market}} — papperskontot förvaltar positionerna på samma data."""
    getter = _get_fn(getter)
    known = [t for t in tickers if uv.theme_of(t)]
    themes = {uv.theme_of(t) for t in known}
    drivers = drivers if drivers is not None else load_drivers(themes, getter, today)
    markets = markets if markets is not None else load_markets({uv.region_of(t) for t in known}, getter,
                                                                nordic_provider)
    commodity = None
    if cfg.commodity_gate:
        try:
            commodity = bt.gate_series(cfg.gate_kind, lambda s: getter(s, STOCK_PERIOD))
        except Exception:
            commodity = None
    rows = []
    for k, t in enumerate(tickers):
        theme = uv.theme_of(t)
        if not theme:
            rows.append({"ticker": t, "status": INGET, "error": "okänd ticker (inte i BERSERK-universumet)",
                         "why": [], "setup": None})
        else:
            try:
                df = getter(t, STOCK_PERIOD)
                if df is not None and getattr(df.index, "tz", None) is not None:
                    df = df.copy()
                    df.index = df.index.tz_localize(None)
            except Exception:
                df = None
            sym, drv = drivers.get(theme, (None, None))
            mkt = markets.get(uv.REGION_INDEX[uv.region_of(t)])
            if keep is not None and df is not None:
                keep[t] = {"stock": df, "driver": drv, "market": mkt}
            rows.append(evaluate(t, df, drv, mkt if mkt is not None else markets.get("SPY"), capital, cfg, sym,
                                 commodity=commodity))
        if progress is not None:
            progress(k + 1, len(tickers), t)
    return {"rows": rows, "drivers": {th_: s for th_, (s, _x) in drivers.items()},
            "when": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}


ORDER = {KOP: 0, BEVAKA: 1, INGET: 2}


def sort_rows(rows: list) -> list:
    """KÖP först (S2, S1, S3), sedan BEVAKA, sedan resten."""
    return sorted(rows, key=lambda r: (ORDER.get(r.get("status"), 3), -sg.PRIORITY.get(r.get("setup"), 0),
                                       r.get("ticker", "")))


# ── Portföljspärrar mot innehaven ───────────────────────────────────────────
def portfolio_flags(rows: list, held: list) -> list:
    """Markera KÖP-rader som skulle bryta BERSERK:s gränser mot innehaven (tickers i universumet)."""
    pc = bt.portfolio_config()
    held = [t for t in dict.fromkeys(str(h).upper() for h in held) if uv.theme_of(t)]
    themes = [uv.theme_of(t) for t in held]
    cxs = [th.complex_of(x) for x in themes]
    for r in rows:
        if r.get("status") != KOP:
            continue
        flags = []
        if r["ticker"] in held:
            flags.append("ÄGS REDAN")
        if themes.count(r["theme"]) >= pc.sector_cap:
            flags.append("TEMA FULLT")
        if dict(pc.group_caps).get("complex") and cxs.count(r["complex"]) >= dict(pc.group_caps)["complex"]:
            flags.append("KOMPLEX FULLT")
        if pc.max_positions and len(held) >= pc.max_positions:
            flags.append("MAX 8 POSITIONER")
        r["flags"] = flags
    return rows


def held_tickers() -> list:
    """Öppna positioner i Holdings (alla hinkar) — tom lista om registret inte går att läsa."""
    try:
        import positions
        return [p["ticker"] for p in positions.all_positions()]
    except Exception:
        return []
