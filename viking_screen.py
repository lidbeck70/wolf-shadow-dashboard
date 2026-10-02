"""
viking_screen.py — Vikings två screeners, bevakningslistan och signalloggen.

  OVTLYR SCREEN          alla tickers sorterade på OVTLYR Nine (9/9, 8/9, 7/9 …)
  VIKING MOMENTUM SCREEN Nine ≥ 7/9 · kurs > EMA10 > EMA20 > EMA50 · RSI > 50 och
                         stigande momentum · inget bearish block nära · R/R ≥ 2 ·
                         relativ volym (valfritt krav, av som standard)
  BEVAKNINGSLISTA        GOLDEN TICKET (9/9 + PASS) · READY (9/9 + WAIT) ·
                         DEVELOPING (7–8/9) · REJECTED (≤ 6/9)
  SIGNALLOGG             varje GOLDEN TICKET / READY med tidsstämpel, Nine per
                         lager, Viking-poäng, entry, stopp, ATR, position, R/R,
                         skäl och åtgärd — en rad per ticker och dag

Screening ≠ automatisk entry. Kategorierna beskriver hur många av systemets
villkor som är uppfyllda — inte aktiens kvalitet.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import pandas as pd

import ovtlyr_nine as on
import viking_execution as vx

MAX_TICKERS = 60
MOMENTUM_MIN_NINE = 7
LOG_STORE = "viking_signals"
LOG_MAX = 1000
LOGGED_CATEGORIES = ("GOLDEN TICKET", "READY")
CATEGORY_ORDER = ("GOLDEN TICKET", "READY", "DEVELOPING", "REJECTED")


def parse_tickers(raw: str) -> list:
    out = []
    for part in str(raw or "").replace(";", ",").replace("\n", ",").replace(" ", ",").split(","):
        t = part.strip().upper()
        if t and t not in out:
            out.append(t)
    return out[:MAX_TICKERS]


def _ob_analysis(df: pd.DataFrame) -> dict:
    try:
        from ovtlyr.indicators.orderblocks import classify_price_vs_ob, detect_orderblocks
        obs = detect_orderblocks(df)
        return classify_price_vs_ob(float(df["Close"].iloc[-1]), obs) if obs else {"signal_bias": "HOLD"}
    except Exception:
        return {"signal_bias": "HOLD"}


def evaluate_ticker(ticker: str, getter: Optional[Callable] = None, sector_getter: Optional[Callable] = None,
                    earnings_getter: Optional[Callable] = None, capital: float = 100_000.0,
                    trades: Optional[list] = None, now: Optional[datetime] = None, today=None) -> dict:
    """En rad: Nine + beslut + kategori. Fel → raden markeras DATA UNAVAILABLE."""
    if getter is None:
        from market_prices import ohlcv as getter
    row = {"ticker": ticker, "error": None, "nine": None, "decision": None, "category": "REJECTED"}
    try:
        df = getter(ticker, on.PERIOD)
    except Exception as exc:
        df = None
        row["error"] = str(exc)
    if df is None or len(df) < on.MIN_BARS:
        row["error"] = row["error"] or "DATA UNAVAILABLE — för lite kurshistorik"
        return row
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    ob = _ob_analysis(df)
    nine = on.evaluate(ticker, stock_df=df, ob_analysis=ob, getter=getter, sector_getter=sector_getter, today=today)
    ed = None
    if earnings_getter is not None:
        try:
            ed = earnings_getter(ticker)
        except Exception:
            ed = None
    d = vx.evaluate_entry(ticker, df, nine=nine, capital=capital, ob_analysis=ob, earnings_date=ed,
                          earnings_known=ed is not None, trades=trades, now=now)
    row.update(nine=nine, decision=d, category=vx.watchlist_category(nine.passed, d.status))
    return row


def run(tickers: list, progress: Optional[Callable] = None, **kw) -> list:
    rows = []
    for i, t in enumerate(tickers[:MAX_TICKERS]):
        rows.append(evaluate_ticker(t, **kw))
        if progress is not None:
            progress(i + 1, len(tickers[:MAX_TICKERS]), t)
    return rows


def _check(d, key) -> Optional[vx.Check]:
    return next((c for c in (d.checks if d else []) if c.key == key), None)


def ovtlyr_screen(rows: list) -> list:
    """Alla rader, sorterade på Nine (antal godkända, sedan viktad poäng)."""
    def key(r):
        n = r["nine"]
        return (-(n.passed if n else -1), -(n.weighted if n else 0), r["ticker"])
    return sorted(rows, key=key)


def momentum_pass(row: dict, require_volume: bool = False) -> tuple:
    """(klar, [skäl]) för VIKING MOMENTUM SCREEN."""
    n, d = row["nine"], row["decision"]
    if n is None or d is None:
        return False, ["DATA UNAVAILABLE"]
    why = []
    if n.passed < MOMENTUM_MIN_NINE:
        why.append(f"Nine {n.passed}/9 < {MOMENTUM_MIN_NINE}")
    for key, label in (("trend", "trendstruktur"), ("momentum", "momentum"), ("rr", "R/R")):
        c = _check(d, key)
        if c is None or not c.passed:
            why.append(label)
    blocks = n.get("stock.blocks")
    if blocks is None or not blocks.passed:
        why.append("bearish block nära")
    if require_volume:
        c = _check(d, "volume")
        if c is None or not c.passed:
            why.append("volym")
    return not why, why


def momentum_screen(rows: list, require_volume: bool = False) -> list:
    return [r for r in ovtlyr_screen(rows) if momentum_pass(r, require_volume)[0]]


def watchlist(rows: list) -> dict:
    out = {c: [] for c in CATEGORY_ORDER}
    for r in ovtlyr_screen(rows):
        out.setdefault(r["category"], []).append(r)
    return out


# ── Signalloggen ────────────────────────────────────────────────────────────
def log_entry(row: dict, now: Optional[datetime] = None, source: str = "screen") -> Optional[dict]:
    n, d = row.get("nine"), row.get("decision")
    if n is None or d is None:
        return None
    p = d.position
    now = now or datetime.now()
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M"), "date": now.strftime("%Y-%m-%d"), "ticker": row["ticker"],
        "ovtlyr_nine": f"{n.passed}/9", "nine_weighted": n.weighted,
        "market": f"{n.layer_passed('market')}/3", "sector": f"{n.layer_passed('sector')}/2",
        "stock": f"{n.layer_passed('stock')}/4", "viking": f"{d.execution_passed}/{d.execution_total}",
        "entry": d.entry, "stop": p.stop if p else None, "atr": p.atr if p else None,
        "shares": p.shares if p else None, "risk_sek": p.risk_amount if p else None, "rr": d.rr,
        "status": d.status, "action": row.get("category"), "trigger_date": d.trigger_date,
        "reason": "; ".join(d.reasons)[:400], "source": source, "label": on.APPROXIMATION,
    }


def append_log(log: list, rows: list, now: Optional[datetime] = None, source: str = "screen",
               categories=LOGGED_CATEGORIES) -> tuple:
    """(ny logg, antal nya/uppdaterade). En rad per ticker och dag — senaste vinner."""
    log = list(log or [])
    n = 0
    for r in rows:
        if r.get("category") not in categories:
            continue
        e = log_entry(r, now, source)
        if e is None:
            continue
        idx = next((i for i, x in enumerate(log) if x.get("date") == e["date"] and x.get("ticker") == e["ticker"]),
                   None)
        if idx is None:
            log.append(e)
            n += 1
        elif {k: v for k, v in log[idx].items() if k != "timestamp"} != {k: v for k, v in e.items()
                                                                          if k != "timestamp"}:
            log[idx] = e
            n += 1
    return log[-LOG_MAX:], n
