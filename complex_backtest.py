"""
complex_backtest.py — fas 3b: Viking Nine per komplex mot standard-Viking Nine på samma bolag (K9).

Motorn är viking_backtest.backtest_ticker (oförändrad). Per råvarukomplex körs komplexets nordiska producenter
tre gånger, så att varje premiss mäts för sig (analysen 2026-10-24 §5.4, beslut 4–6):

  standard     SPDR-sektorserien som i dag (Börsdata-sektor → XLB/XLE/XLP …)
  komplex      komplexprofilens sektorserie (COPX, GDX, XLE, DBA, WOOD, BDRY)
  + drivare    samma affärer, men bara de där komplexets drivare hade vänt på signaldagen (EMA20 stigande och
               över EMA50 — complex_nine:s grind), filtrerat på färdiga affärer som i fas 3a

Marknad och bredd är desamma i alla tre (OMXS30 + Large Cap via nordic_market), så skillnaden är premissen.
Allt utan facit: drivarens läge räknas på data t.o.m. signaldagen.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

import complex_profiles as cp
import ovtlyr_nine as on
import viking_backtest as vbt
from berserk import signals as bsg


def driver_turn_series(series: Optional[pd.Series]) -> Optional[pd.Series]:
    """Drivarens grind per dag (bool): EMA20 stigande och kurs över EMA50 — BERSERK:s turn."""
    if series is None or len(series) < 60:
        return None
    return bsg.driver_frame(pd.Series(series).astype(float))["turn"].astype(bool)


def gate_ok(turn: Optional[pd.Series], date) -> Optional[bool]:
    """Grindens läge på signaldagen (senast kända värde) — None utan serie."""
    if turn is None or turn.empty:
        return None
    k = int(turn.index.searchsorted(pd.Timestamp(date), side="right")) - 1
    return bool(turn.iloc[k]) if k >= 0 else None


def run_complex(profile: cp.ComplexProfile, tickers: list, getter: Callable, nordic: Optional[dict],
                cfg: vbt.Config = vbt.Config(), start=None, driver: Optional[pd.Series] = None) -> dict:
    """{standard, komplex, drivare: [Trade]} för komplexets bolag. getter(ticker, period) → OHLCV."""
    def _get(t):
        try:
            df = getter(t, "max")
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df
    m_df = pd.DataFrame({"Close": nordic["close"]}) if nordic and nordic.get("close") is not None else _get(on.MARKET_TICKER)
    breadth = (nordic or {}).get("breadth")
    label = on.NORDIC_LABEL if nordic else "SPY"
    sec_c = _get(profile.sector_symbol)
    turn = driver_turn_series(driver)
    out = {"standard": [], "komplex": [], "drivare": []}
    for t in tickers:
        df = _get(t)
        if df is None:
            continue
        etf, _src = on.resolve_sector(t)
        std = vbt.backtest_ticker(t, df, m_df, _get(etf) if etf else None, breadth, cfg, start=start, market_label=label)
        cx = vbt.backtest_ticker(t, df, m_df, sec_c, breadth, cfg, start=start, market_label=label)
        out["standard"] += std["trades"]
        out["komplex"] += cx["trades"]
        out["drivare"] += [tr for tr in cx["trades"] if gate_ok(turn, tr.signal_date) is not False]
    return out


def summary(trades: list) -> dict:
    rs = [t.r for t in trades if not t.open and t.r is not None]
    if not rs:
        return {"n": 0, "exp": None, "win": None, "sum_r": 0.0}
    return {"n": len(rs), "exp": round(sum(rs) / len(rs), 3), "win": round(sum(1 for x in rs if x > 0) / len(rs) * 100, 1),
            "sum_r": round(sum(rs), 1)}


def run(getter: Optional[Callable] = None, members: Optional[dict] = None, nordic: Optional[dict] = None,
        drivers: Optional[dict] = None, cfg: vbt.Config = vbt.Config(), start=None) -> list:
    """[{complex, variant, n, exp, win, sum_r}] för alla råvarukomplex. members = {komplex: [tickers]} (förval:
    BERSERK:s nordiska producenter per komplex), drivers = {komplex: serie}."""
    from berserk import universe as buv
    from valhall import complexes as vcx
    if getter is None:
        from market_prices import ohlcv as getter
    if members is None:
        members = {k: [t for th in vcx.themes_of(k) for t in buv.by_theme().get(th, [])
                       if buv.region_of(t) == "Norden" and buv.kind_of(t) == "producent"] for k in vcx.KEYS}
    rows = []
    for p in cp.complexes():
        tickers = members.get(p.key) or []
        if not tickers:
            continue
        res = run_complex(p, tickers, getter, nordic, cfg, start, (drivers or {}).get(p.key))
        for variant in ("standard", "komplex", "drivare"):
            rows.append({"complex": p.key, "variant": variant, "tickers": len(tickers), **summary(res[variant])})
    return rows
