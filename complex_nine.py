"""
complex_nine.py — Viking Nine per komplex: ett tunt omslag utanpå motorn (K3).

Samma motor, samma regler: ovtlyr_nine.compute (ren funktion) får komplexets sektorserie, marknadsserie och bredd
injicerade; viking_execution.evaluate_entry får profilens risk-% och positionstak; kategorin är densamma
(GOLDEN TICKET / READY / DEVELOPING / REJECTED). Det enda som läggs till ligger utanför motorn:

  drivargrinden   komplexets råvarudrivare (berserk.signals.driver_frame) måste ha vänt ("turn": EMA20 stigande
                  och över EMA50) — annars blir GOLDEN TICKET → READY, precis som "SEKTOR UPPTAGEN" i dag
  drivarexit      drivaren stänger under EMA50 (BERSERK S2:s regel) — en extra utgång som papperet kan läsa
  producentbredd  andel av komplexets nordiska producenter över EMA50 som MARKET-lagrets bredd

Raderna har samma form som viking_screen.evaluate_ticker (nine, decision, category, sector_busy) plus
"profile" och "driver", så log_entry, Valhalls compact() och papperet fungerar oförändrade.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import pandas as pd

import complex_profiles as cp
import ovtlyr_nine as on
import viking_execution as vx
import viking_screen as vs
from berserk import signals as bsg
from valhall import complexes as vcx

DRIVER_PERIOD = "7y"
BREADTH_EMA = 50
BREADTH_MIN_MEMBERS = 3
DRIVER_BLOCK = "DRIVAREN HAR INTE VÄNT"


# ── Drivaren ────────────────────────────────────────────────────────────────
def driver_state(profile: cp.ComplexProfile, getter: Optional[Callable] = None, today=None) -> Optional[dict]:
    """Komplexets drivare i dag: {theme, symbol, phase, turn, strong, below_ema50, detail} — None när
    profilen saknar drivare; passed=None när ingen prisserie gick att hämta (grinden kan då inte spärra)."""
    if not profile.driver_theme:
        return None
    from berserk import live
    try:
        sym, series = live.load_drivers([profile.driver_theme], getter, today).get(profile.driver_theme, (None, None))
    except Exception:
        sym, series = None, None
    if series is None or len(series) < 220:
        return {"theme": profile.driver_theme, "symbol": sym, "phase": None, "passed": None, "turn": None,
                "strong": None, "below_ema50": None, "detail": f"{profile.driver_theme}: ingen prisserie — grinden kan inte avgöra"}
    row = bsg.driver_frame(pd.Series(series).astype(float)).iloc[-1].to_dict()
    phase = live.phase_of({"driver": sym, **row})
    turn = bool(row.get("turn"))
    passed = turn if profile.driver_gate == cp.GATE_TURN else True
    return {"theme": profile.driver_theme, "symbol": sym, "phase": phase, "passed": passed, "turn": turn,
            "strong": bool(row.get("strong")), "below_ema50": bool(float(row["close"]) < float(row["ema50"])),
            "detail": f"{sym}: {phase}" + ("" if passed else f" — {DRIVER_BLOCK}")}


def driver_exit(profile: cp.ComplexProfile, getter: Optional[Callable] = None, today=None) -> Optional[dict]:
    """Extra utgång för öppna positioner: drivaren stänger under EMA50 (BERSERK S2). None utan drivare."""
    d = driver_state(profile, getter, today)
    if not d or d.get("below_ema50") is None:
        return None
    return {"triggered": bool(d["below_ema50"]), "detail": f"{d['symbol']}: {'under' if d['below_ema50'] else 'över'} EMA50"}


# ── Marknad och bredd ───────────────────────────────────────────────────────
def producer_breadth(profile: cp.ComplexProfile, getter: Optional[Callable] = None, now=None) -> Optional[pd.Series]:
    """Andel (0–100) av komplexets nordiska producenter över EMA50, dag för dag. None om färre än tre har data."""
    from berserk import universe as buv
    if getter is None:
        from market_prices import ohlcv as getter
    members = [t for theme in vcx.themes_of(profile.key) for t in buv.by_theme().get(theme, [])
               if buv.kind_of(t) == "producent" and buv.region_of(t) == "Norden"]
    closes = {}
    for t in members:
        try:
            df = getter(t, on.PERIOD)
        except Exception:
            df = None
        c = on._close(on.closed_only(df, t, now)) if df is not None and len(df) else None
        if c is not None and len(c) >= BREADTH_EMA:
            closes[t] = c
    if len(closes) < BREADTH_MIN_MEMBERS:
        return None
    df = pd.DataFrame(closes).sort_index().ffill()
    above = df.gt(df.ewm(span=BREADTH_EMA, adjust=False).mean())
    return (above.sum(axis=1) / df.notna().sum(axis=1).clip(lower=1) * 100).dropna()


def market_inputs(profile: cp.ComplexProfile, getter: Optional[Callable] = None, now=None) -> dict:
    """compute()-argumenten för MARKET-lagret: OMXS30 ur nordic_market, bredden enligt profilen."""
    nm = on.nordic_market(getter=getter) or {}
    kw: dict = {}
    if nm.get("close") is not None:
        kw.update(market_label=f"{cp.NORDIC_MARKET} · {profile.label}", market_source=nm.get("source"),
                  market_close=on.closed_only(nm["close"], on.NORDIC_CLOCK, now))
    else:
        kw.update(market_label=f"{on.MARKET_TICKER} ({cp.NORDIC_MARKET} saknas)")
    breadth, src = None, ""
    if profile.breadth == cp.PRODUCERS:
        breadth = producer_breadth(profile, getter, now)
        src = f"producentbredd {profile.label} (andel över EMA{BREADTH_EMA})" if breadth is not None else ""
    if breadth is None and nm.get("breadth") is not None:
        breadth, src = on.closed_only(nm["breadth"], on.NORDIC_CLOCK, now), nm.get("breadth_source") or "Large Cap"
    kw.update(breadth_pct=breadth if breadth is not None else pd.Series(dtype=float), breadth_source=src or "bredd saknas")
    return kw


# ── En aktie ────────────────────────────────────────────────────────────────
def evaluate_complex(ticker: str, profile: cp.ComplexProfile, getter: Optional[Callable] = None,
                     earnings_getter: Optional[Callable] = None, capital: float = 100_000.0,
                     risk_getter: Optional[Callable] = None, trades: Optional[list] = None,
                     now: Optional[datetime] = None, today=None, held: Optional[dict] = None,
                     market_kw: Optional[dict] = None, driver: Optional[dict] = None) -> dict:
    """En rad i viking_screen.evaluate_ticker-form + profile/driver. market_kw/driver kan förberäknas per
    skanning (samma för alla aktier i komplexet)."""
    if getter is None:
        from market_prices import ohlcv as getter
    row = {"ticker": ticker, "error": None, "nine": None, "decision": None, "category": "REJECTED",
           "sector_busy": None, "profile": profile.key, "driver": driver}
    try:
        df = getter(ticker, on.PERIOD)
    except Exception as exc:
        df, row["error"] = None, str(exc)
    if df is None or len(df) < on.MIN_BARS:
        row["error"] = row["error"] or "DATA UNAVAILABLE — för lite kurshistorik"
        return row
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df = on.closed_only(df, ticker, now)
    try:
        sec = getter(profile.sector_symbol, on.PERIOD)
        sec = on.closed_only(sec, profile.sector_symbol, now) if sec is not None and len(sec) else None
    except Exception:
        sec = None
    ob = vs._ob_analysis(df)
    mk = market_kw if market_kw is not None else market_inputs(profile, getter, now)
    nine = on.compute(ticker, df, None, profile.sector_symbol, sec, {}, ob_analysis=ob, today=today, **mk)
    nine.sector_source = f"{profile.label}: {profile.sector_symbol} (komplexprofil)"
    ed = None
    if earnings_getter is not None and nine.passed == on.NINE_TOTAL:
        try:
            ed = earnings_getter(ticker)
        except Exception:
            ed = None
    risk = None
    if risk_getter is not None:
        try:
            risk = risk_getter(ticker)
        except Exception:
            risk = None
    d = vx.evaluate_entry(ticker, df, nine=nine, capital=capital, ob_analysis=ob, earnings_date=ed,
                          earnings_known=ed is not None, trades=trades, now=now, market_risk=risk,
                          risk_pct=profile.risk_pct, max_position_pct=profile.max_position_pct)
    category = vx.watchlist_category(nine.passed, d.status)
    busy = vs.sector_busy(ticker, profile.key, held)                 # en position per komplex
    if busy and category == "GOLDEN TICKET":
        category = "READY"
    drv = driver if driver is not None else driver_state(profile, getter, today)
    if drv and drv.get("passed") is False and category == "GOLDEN TICKET":   # drivargrinden: entryn väntar
        category = "READY"
        d.reasons = list(d.reasons or []) + [f"{DRIVER_BLOCK} ({drv.get('detail')})"]
    row.update(nine=nine, decision=d, category=category, sector_busy=busy, driver=drv)
    return row


def scan_complex(profile: cp.ComplexProfile, tickers: list, closes_getter: Optional[Callable] = None,
                 max_candidates: int = 8, today=None, getter: Optional[Callable] = None, now=None, **kw) -> dict:
    """Som viking_screen.scan men per profil: steg 1 (trend + signal, momentum 63 d) över komplexets bolag,
    steg 2 evaluate_complex på de bästa. Marknad, bredd och drivare räknas en gång per skanning."""
    tickers = list(dict.fromkeys(t for t in tickers if t))
    closes = (closes_getter or vs._closes_default)(tickers)
    closes = {t: closes.get(t) for t in tickers}
    ranked, funnel = vs.stage1(closes, today=today)
    n = max(1, min(int(max_candidates), vs.CANDIDATES_CAP))
    cands = [t for t, _r in ranked[:n]]
    funnel.update(candidates=len(cands), momentum={t: r for t, r in ranked}, profile=profile.key)
    mk = market_inputs(profile, getter, now)
    drv = driver_state(profile, getter, today)
    rows = [evaluate_complex(t, profile, getter=getter, today=today, now=now, market_kw=mk, driver=drv, **kw)
            for t in cands]
    return {"rows": rows, "funnel": funnel, "profile": profile.key, "driver": drv}
