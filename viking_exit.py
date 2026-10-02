"""
viking_exit.py — Vikings EXIT ENGINE: förvaltning av en öppen position.

Separat från entryn (viking_execution.py) och från setupen (ovtlyr_nine.py).
Varje exitregel ur strategy_rules.py (Viking) är en egen kontroll med
uträkningen utskriven:

  HARD MARKET EXIT  SPY stänger under EMA20 → stäng ALLT
  INITIAL STOP      entry − 1,5 × ATR14 (ATR vid entrydagen)
  TRAILING STOP     stängning under EMA10
  BE EXIT           när stoppen flyttats till breakeven (ny högre topp efter
                    entry): stängning under gårdagens low
  BEARISH BLOCK     kursen går in i ett bearish order block
  GAP & CRAP        gap upp över gårdagens high, stänger under gårdagens close
  SECTOR/BREADTH    sektorn bearish OCH marknadsbredden faller (OVTLYR Nine)
  STOCK SIGNAL      aktiens signal SELL (stängning under EMA20, OVTLYR Nine)
  F&G EXIT          entry-F&G 0–50 → exit vid 63; 50–75 → +10; 75+ → +5
  EARNINGS          rapport inom 5 handelsdagar → EXIT / REDUCE BEFORE EARNINGS

Status: CLOSE ALL (marknaden), EXIT (någon regel utlöst), HOLD.
Saknad data utlöser ingen exit men visas som DATA UNAVAILABLE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import viking_execution as vx

CLOSE_ALL, EXIT, HOLD = "CLOSE ALL", "EXIT", "HOLD"
ACTIVE, CLEAR, UNAVAILABLE = "EXIT", "OK", "DATA UNAVAILABLE"
BE_LOOKBACK = 10                   # "ny högre topp" = high över högsta high de 10 dagarna före entry
FG_TARGETS = ((50.0, None, 63.0), (75.0, 10.0, None), (101.0, 5.0, None))   # (entry-F&G under, spread, fast nivå)


@dataclass
class Trigger:
    key: str
    label: str
    status: str                    # EXIT | OK | DATA UNAVAILABLE
    detail: str

    @property
    def active(self) -> bool:
        return self.status == ACTIVE


@dataclass
class ExitDecision:
    ticker: str
    status: str
    entry: float
    entry_date: str
    price: Optional[float] = None
    initial_stop: Optional[float] = None
    current_stop: Optional[float] = None
    trailing_stop: Optional[float] = None       # EMA10
    breakeven_armed: bool = False
    r_now: Optional[float] = None
    fg_entry: Optional[float] = None
    fg_target: Optional[float] = None
    triggers: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    @property
    def active(self) -> list:
        return [t for t in self.triggers if t.active]


def fg_target(fg_at_entry: Optional[float]) -> Optional[float]:
    """Vikings F&G-target: 0–50 → 63, 50–75 → entry + 10, 75+ → entry + 5."""
    if fg_at_entry is None:
        return None
    for below, spread, fixed in FG_TARGETS:
        if fg_at_entry < below:
            return fixed if fixed is not None else round(min(100.0, fg_at_entry + spread), 1)
    return None


def breakeven_armed(df: pd.DataFrame, entry_date) -> bool:
    """Ny högre topp efter entry: något high efter entrydagen över högsta high
    de BE_LOOKBACK dagarna före (och med) entrydagen."""
    d = pd.Timestamp(entry_date)
    before = df[df.index <= d]["High"].tail(BE_LOOKBACK)
    after = df[df.index > d]["High"]
    return bool(len(before) and len(after) and float(after.max()) > float(before.max()))


def evaluate_exit(ticker: str, df: pd.DataFrame, entry: float, entry_date, nine=None,
                  ob_analysis: Optional[dict] = None, spy_df: Optional[pd.DataFrame] = None,
                  earnings_date=None, be_moved: Optional[bool] = None, today=None) -> ExitDecision:
    """df = aktiens dagsdata (DatetimeIndex). be_moved None = räkna själv ur ny högre topp."""
    entry = float(entry)
    ed = pd.Timestamp(entry_date)
    out = ExitDecision(ticker, HOLD, round(entry, 2), str(ed.date()))
    if df is None or len(df) < 25 or not (entry > 0):
        out.status = HOLD
        out.reasons = ["DATA UNAVAILABLE — för lite kurshistorik eller inget entrypris"]
        return out
    c, hi, lo, op = (df[k].astype(float) for k in ("Close", "High", "Low", "Open"))
    price, prev_close, prev_low, prev_high = float(c.iloc[-1]), float(c.iloc[-2]), float(lo.iloc[-2]), float(hi.iloc[-2])
    out.price = round(price, 2)
    t = []

    # HARD MARKET EXIT — SPY (riktig) under EMA20
    spy_sig = nine.get("market.signal") if nine is not None and hasattr(nine, "get") else None
    if spy_sig is not None and spy_sig.status in ("PASS", "FAIL"):
        t.append(Trigger("market", "Hard market exit", ACTIVE if spy_sig.status == "FAIL" else CLEAR,
                         f"SPY: {spy_sig.detail}"))
    elif spy_df is not None and len(spy_df) >= 25:
        s = spy_df["Close"].astype(float)
        e20 = float(vx._ema(s, 20).iloc[-1])
        below = float(s.iloc[-1]) < e20
        t.append(Trigger("market", "Hard market exit", ACTIVE if below else CLEAR,
                         f"SPY {float(s.iloc[-1]):,.2f} {'<' if below else '≥'} EMA20 {e20:,.2f}"))
    else:
        t.append(Trigger("market", "Hard market exit", UNAVAILABLE, "SPY-data saknas"))

    # INITIAL STOP — ATR vid entrydagen
    a = vx.atr(df)
    a_entry = a[a.index <= ed]
    atr_e = float(a_entry.iloc[-1]) if len(a_entry) else float(a.iloc[-1])
    init = entry - vx.ATR_STOP_MULT * atr_e
    out.initial_stop = round(init + 1e-9, 2)
    risk = entry - init
    out.r_now = round((price - entry) / risk, 2) if risk > 0 else None
    armed = breakeven_armed(df, ed) if be_moved is None else bool(be_moved)
    out.breakeven_armed = armed
    stop = max(init, entry) if armed else init
    out.current_stop = round(stop + 1e-9, 2)
    hit = price <= stop
    t.append(Trigger("stop", "Breakeven-stopp" if armed else "Initial stop", ACTIVE if hit else CLEAR,
                     f"stängning {price:,.2f} {'≤' if hit else '>'} stopp {stop:,.2f} "
                     + (f"(flyttad till breakeven {entry:,.2f})" if armed else
                        f"(entry − {vx.ATR_STOP_MULT:g} × ATR14 {atr_e:,.2f})")))

    # TRAILING STOP — EMA10
    e10 = float(vx._ema(c, 10).iloc[-1])
    out.trailing_stop = round(e10, 2)
    t.append(Trigger("trail", "Trailing stop (EMA10)", ACTIVE if price < e10 else CLEAR,
                     f"stängning {price:,.2f} {'<' if price < e10 else '≥'} EMA10 {e10:,.2f}"))

    # BE EXIT — efter flyttad stopp: stängning under gårdagens low
    if armed:
        t.append(Trigger("be", "BE exit", ACTIVE if price < prev_low else CLEAR,
                         f"stoppen flyttad: stängning {price:,.2f} {'<' if price < prev_low else '≥'} "
                         f"gårdagens low {prev_low:,.2f}"))
    else:
        t.append(Trigger("be", "BE exit", CLEAR, "stoppen inte flyttad till breakeven ännu (ingen ny högre topp)"))

    # BEARISH BLOCK
    ob = (ob_analysis or {}).get("nearest_bearish_ob")
    ob_low = getattr(ob, "low", None) if ob is not None and not isinstance(ob, dict) else (ob or {}).get("low")
    ob_high = getattr(ob, "high", None) if ob is not None and not isinstance(ob, dict) else (ob or {}).get("high")
    if ob_low is None:
        t.append(Trigger("block", "Bearish block", CLEAR, "inget aktivt bearish order block ovanför"))
    else:
        inside = float(ob_low) <= price
        t.append(Trigger("block", "Bearish block", ACTIVE if inside else CLEAR,
                         f"bearish block {float(ob_low):,.2f}–{float(ob_high or ob_low):,.2f} · kurs {price:,.2f} "
                         f"{'inne i blocket' if inside else 'under blocket'}"))

    # GAP & CRAP
    o = float(op.iloc[-1])
    gap = o > prev_high
    crap = gap and price < prev_close
    t.append(Trigger("gap", "Gap & crap", ACTIVE if crap else CLEAR,
                     (f"öppnade {o:,.2f} över gårdagens high {prev_high:,.2f} och stängde {price:,.2f} "
                      f"{'under' if price < prev_close else 'över'} gårdagens close {prev_close:,.2f}")
                     if gap else "ingen gap upp i dag"))

    # SECTOR/BREADTH och STOCK SIGNAL — ur OVTLYR Nine
    if nine is not None and hasattr(nine, "get"):
        sb, mb, ss = nine.get("sector.breadth"), nine.get("market.breadth"), nine.get("stock.signal")
        known = all(f is not None and f.status in ("PASS", "FAIL") for f in (sb, mb))
        both = known and sb.status == "FAIL" and mb.status == "FAIL"
        t.append(Trigger("breadth", "Sector/breadth", ACTIVE if both else CLEAR if known else UNAVAILABLE,
                         f"sektor {sb.status if sb else '—'} · marknadsbredd {mb.status if mb else '—'}"
                         + (" — båda bearish" if both else "")))
        if ss is not None and ss.status in ("PASS", "FAIL"):
            t.append(Trigger("signal", "Stock signal", ACTIVE if ss.status == "FAIL" else CLEAR, ss.detail))
        else:
            t.append(Trigger("signal", "Stock signal", UNAVAILABLE, "aktiens signal saknas"))
    else:
        t.append(Trigger("breadth", "Sector/breadth", UNAVAILABLE, "OVTLYR Nine saknas"))
        t.append(Trigger("signal", "Stock signal", UNAVAILABLE, "OVTLYR Nine saknas"))

    # F&G EXIT — target ur F&G vid entry
    from ovtlyr_nine import fear_greed
    fg_e = fear_greed(df[df.index <= ed])
    fg_now = fear_greed(df)
    out.fg_entry = None if fg_e is None else round(fg_e, 1)
    out.fg_target = fg_target(fg_e)
    if fg_e is None or fg_now is None:
        t.append(Trigger("fg", "F&G exit", UNAVAILABLE, "F&G vid entry eller i dag kunde inte räknas"))
    else:
        hit = fg_now >= out.fg_target
        t.append(Trigger("fg", "F&G exit", ACTIVE if hit else CLEAR,
                         f"F&G vid entry {fg_e:.0f} → target {out.fg_target:g} · nu {fg_now:.0f} "
                         f"[{'syntetiskt F&G — WOLF APPROXIMATION'}]"))

    # EARNINGS
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    if earnings_date is None:
        t.append(Trigger("earnings", "Earnings", UNAVAILABLE, "rapportdatum okänt — kontrollera själv"))
    else:
        days = vx.trading_days_until(earnings_date, today)
        near = 0 <= days <= vx.EARNINGS_BUFFER_DAYS
        t.append(Trigger("earnings", "Earnings", ACTIVE if near else CLEAR,
                         f"rapport {pd.Timestamp(earnings_date).date()} — "
                         + ("EXIT / REDUCE BEFORE EARNINGS" if near else
                            f"{days} handelsdagar bort" if days > 0 else "passerad")))

    out.triggers = t
    if t[0].active:
        out.status = CLOSE_ALL
        out.reasons = ["HARD MARKET EXIT — SPY under EMA20: stäng alla positioner"]
    elif out.active:
        out.status = EXIT
        out.reasons = [f"{x.label}: {x.detail}" for x in out.active]
    else:
        out.status = HOLD
        out.reasons = ["Ingen exitregel utlöst — låt trailing-stoppen arbeta"]
    return out
