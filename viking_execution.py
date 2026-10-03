"""
viking_execution.py — VIKING EXECUTION + RISK ENGINE + evaluate_entry().

OVTLYR Nine (ovtlyr_nine.py) svarar på om setupen finns. Den här modulen
svarar på om entryn finns, och hur stor den får vara:

  VIKING EXECUTION (7 filter)
    Momentum      RSI14 > 50 · kurs > föregående dags low · RSI stigande
    Trend         kurs > EMA10 > EMA20 > EMA50
    Volym         relativ volym ≥ 1,20 — visas men krävs inte förrän backtestat
    Entry candle  triggercandlen har stängt och är grön (close > open, ≥ föregående close)
    No chase      kursen högst 2 % över triggerns stängning
    R/R           (närmaste motstånd − entry) / (entry − stopp) ≥ 2
    Earnings      ingen rapport inom 5 handelsdagar

  RISK ENGINE
    riskbudget = kapital × 1,5 %          stopp = entry − 1,5 × ATR14
    aktier = floor(budget / stoppavstånd), men positionen högst 25 % av kapitalet
    max två förlustaffärer per dag — sedan ingen ny entry

  BESLUT
    GOLDEN TICKET  Nine 9/9 och alla krav i exekveringen och risken klara
    WAIT           bra eller nästan bra setup, men entryn är inte klar
    NO TRADE       Nine ≤ 6/9, marknadens signal SELL (SPY / OMXS30), rapport nära, eller dagsgränsen nådd

Appen förutsäger ingenting: den svarar på hur många villkor som är uppfyllda nu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional

import numpy as np
import pandas as pd

# ── Konfiguration ───────────────────────────────────────────────────────────
RSI_MIN = 50.0
RELATIVE_VOLUME_MIN = 1.20
VOLUME_REQUIRED = False            # krävs först när backtestet visat att det hjälper
MAX_CHASE_PCT = 2.0
MINIMUM_RR = 2.0
RESISTANCE_LOOKBACK = 250          # swing-högsta inom ett år om inget bearish order block finns
EARNINGS_BUFFER_DAYS = 5
EARNINGS_UNKNOWN_BLOCKS = True     # okänt rapportdatum → WAIT (saknad data är aldrig grönt)
MAX_DAILY_LOSSES = 2
MAX_POSITION_PCT = 25.0            # exponeringstak per position (separat från risken)
NINE_DEVELOPING = 7                # 7–8/9 = DEVELOPING → WAIT; ≤ 6 → NO TRADE


def _viking_params() -> dict:
    try:
        from strategies.viking import DEFAULT_PARAMS
        return dict(DEFAULT_PARAMS)
    except Exception:                                   # pragma: no cover
        return {"risk_pct": 0.015, "atr_stop_mult": 1.5, "atr_period": 14}


_P = _viking_params()
MAX_RISK_PCT = float(_P.get("risk_pct", 0.015)) * 100      # 1,5 — samma tal som strategies/viking.py
ATR_STOP_MULT = float(_P.get("atr_stop_mult", 1.5))
ATR_PERIOD = int(_P.get("atr_period", 14))

GO, WAIT, NO_TRADE = "GOLDEN TICKET", "WAIT", "NO TRADE"
PASS, FAIL, INFO, UNAVAILABLE = "PASS", "FAIL", "INFO", "DATA UNAVAILABLE"
CONFIRMED, WAIT_CLOSE, FAILED = "CONFIRMED", "WAIT FOR CLOSE", "FAILED"
NO_CHASE, INSUFFICIENT_RR = "NO CHASE", "INSUFFICIENT R/R"
EARNINGS_RISK, DAILY_LIMIT = "EARNINGS RISK", "DAILY LOSS LIMIT REACHED"
MARKET_RISK_HIGH = "MARKNADSRISK HÖG"
MARKET_RISK_ELEVATED = "MARKNADSRISK FÖRHÖJD"

_NORDIC = {".ST": ("Europe/Stockholm", time(17, 30)), ".OL": ("Europe/Oslo", time(16, 20)),
           ".CO": ("Europe/Copenhagen", time(17, 0)), ".HE": ("Europe/Helsinki", time(18, 30))}
_US_CLOSE = ("America/New_York", time(16, 0))


# ── Indikatorer ─────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilders RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilders ATR — samma som strategies/viking.py."""
    h, lo, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ── Byggstenar ──────────────────────────────────────────────────────────────
@dataclass
class Check:
    key: str
    label: str
    status: str                    # PASS | FAIL | INFO | DATA UNAVAILABLE
    detail: str
    required: bool = True
    flag: Optional[str] = None     # WAIT FOR CLOSE | NO CHASE | INSUFFICIENT R/R | EARNINGS RISK

    @property
    def passed(self) -> bool:
        return self.status == PASS


def candle_closed(ticker: str, last_bar, now: Optional[datetime] = None) -> bool:
    """False när sista dagsstapeln är dagens och börsen inte har stängt än."""
    import zoneinfo
    suffix = next((s for s in _NORDIC if str(ticker).upper().endswith(s)), None)
    tz_name, close_t = _NORDIC[suffix] if suffix else _US_CLOSE
    tz = zoneinfo.ZoneInfo(tz_name)
    now = now.astimezone(tz) if now is not None and now.tzinfo else (
        now.replace(tzinfo=tz) if now is not None else datetime.now(tz))
    bar_day = pd.Timestamp(last_bar).date()
    return not (bar_day == now.date() and now.time() < close_t)


def nearest_resistance(df: pd.DataFrame, entry: float, ob_analysis: Optional[dict] = None) -> tuple:
    """(nivå | None, källa). Närmaste bearish order block ovanför entry, annars
    högsta high inom ett år ovanför entry. None = inget motstånd (fri väg)."""
    ob = (ob_analysis or {}).get("nearest_bearish_ob")
    lvl = getattr(ob, "low", None) if ob is not None else None
    if isinstance(ob, dict):
        lvl = ob.get("low")
    if lvl is not None and float(lvl) > entry:
        return float(lvl), "bearish order block"
    # Triggercandlens eget high är inget motstånd — bara staplarna före den
    hi = df["High"].astype(float).iloc[:-1].tail(RESISTANCE_LOOKBACK)
    above = hi[hi > entry * 1.001]
    if len(above):
        return float(above.max()), "högsta high inom ett år"
    return None, "inget motstånd inom ett år"


def trading_days_until(day, today) -> int:
    return int(np.busday_count(pd.Timestamp(today).date(), pd.Timestamp(day).date()))


def daily_losses(trades: list, today) -> int:
    """Antal stängda förlustaffärer i journalen med exitdatum i dag."""
    d = str(pd.Timestamp(today).date())
    n = 0
    for t in trades or []:
        if str(t.get("exit_date") or "")[:10] != d:
            continue
        pnl = t.get("pnl_pct")
        r = t.get("r_multiple")
        if (pnl is not None and float(pnl) < 0) or (r is not None and float(r) < 0):
            n += 1
    return n


# ── Riskmotorn ──────────────────────────────────────────────────────────────
@dataclass
class Position:
    capital: float
    entry: float
    atr: float
    stop: float
    stop_distance: float
    risk_budget: float
    shares: int
    shares_by_risk: int
    position_value: float
    exposure_pct: float
    risk_amount: float
    risk_pct: float
    capped_by_exposure: bool
    max_risk_pct: float = MAX_RISK_PCT
    max_position_pct: float = MAX_POSITION_PCT


def size_position(capital: float, entry: float, atr_value: float, risk_pct: float = MAX_RISK_PCT,
                  atr_mult: float = ATR_STOP_MULT, max_position_pct: float = MAX_POSITION_PCT) -> Optional[Position]:
    """Risk = avståndet till stoppen × antal aktier — aldrig över risk_pct av kapitalet.
    Exponeringen (positionens värde) är en separat kontroll."""
    if not (capital > 0 and entry > 0 and atr_value > 0):
        return None
    stop_distance = atr_value * atr_mult
    stop = entry - stop_distance
    budget = capital * risk_pct / 100
    by_risk = math.floor(budget / stop_distance + 1e-9)
    by_exposure = math.floor(capital * max_position_pct / 100 / entry + 1e-9)
    shares = max(0, min(by_risk, by_exposure))
    value = shares * entry
    risk_amount = shares * stop_distance
    return Position(capital, round(entry, 2), round(atr_value, 2), round(stop + 1e-9, 2), round(stop_distance, 3),
                    round(budget, 2), shares, by_risk, round(value, 2), round(value / capital * 100, 1),
                    round(risk_amount, 2), round(risk_amount / capital * 100, 2), by_exposure < by_risk,
                    risk_pct, max_position_pct)


# ── Exekveringen ────────────────────────────────────────────────────────────
@dataclass
class EntryDecision:
    ticker: str
    status: str                    # GOLDEN TICKET | WAIT | NO TRADE
    nine_passed: Optional[int]
    nine_total: int
    checks: list = field(default_factory=list)
    position: Optional[Position] = None
    entry: Optional[float] = None
    trigger_date: Optional[str] = None
    resistance: Optional[float] = None
    resistance_source: str = ""
    rr: Optional[float] = None
    flags: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    @property
    def execution_passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def execution_total(self) -> int:
        return len(self.checks)

    @property
    def missing(self) -> list:
        return [c for c in self.checks if c.required and not c.passed]

    def as_dict(self) -> dict:
        p = self.position
        return {"status": self.status, "ovtlyr_nine": f"{self.nine_passed}/{self.nine_total}",
                "viking_score": f"{self.execution_passed}/{self.execution_total}", "entry": self.entry,
                "stop": p.stop if p else None, "shares": p.shares if p else None,
                "risk_sek": p.risk_amount if p else None, "rr": self.rr, "flags": list(self.flags),
                "reasons": list(self.reasons)}


def execution_checks(ticker: str, df: pd.DataFrame, ob_analysis: Optional[dict] = None,
                     earnings_date=None, earnings_known: bool = True, now: Optional[datetime] = None,
                     current_price: Optional[float] = None) -> tuple:
    """([Check], entry, trigger_date, stop_distance, resistance, källa, rr). df = dagsdata med DatetimeIndex."""
    df = df.dropna(subset=["Close"])
    closed = candle_closed(ticker, df.index[-1], now)
    trig = df if closed else df.iloc[:-1]               # triggercandlen = senaste STÄNGDA stapel
    c, o, lo = trig["Close"].astype(float), trig["Open"].astype(float), trig["Low"].astype(float)
    entry = float(c.iloc[-1])
    price = float(current_price if current_price is not None else df["Close"].iloc[-1])
    r = rsi(c)
    checks = []

    # A. Momentum
    rsi_now, rsi_prev = float(r.iloc[-1]), float(r.iloc[-2])
    prev_low = float(lo.iloc[-2])
    ok = rsi_now > RSI_MIN and entry > prev_low and rsi_now > rsi_prev
    checks.append(Check("momentum", "Momentum", PASS if ok else FAIL,
                        f"RSI {rsi_now:.1f} ({'>' if rsi_now > RSI_MIN else '≤'} {RSI_MIN:g}, "
                        f"{'stigande' if rsi_now > rsi_prev else 'fallande'} från {rsi_prev:.1f}) · stängning "
                        f"{entry:,.2f} {'>' if entry > prev_low else '≤'} föregående low {prev_low:,.2f}"))

    # B. Trendstruktur
    e10, e20, e50 = (float(_ema(c, n).iloc[-1]) for n in (10, 20, 50))
    ok = entry > e10 > e20 > e50
    checks.append(Check("trend", "Trendstruktur", PASS if ok else FAIL,
                        f"kurs {entry:,.2f} · EMA10 {e10:,.2f} · EMA20 {e20:,.2f} · EMA50 {e50:,.2f} — "
                        f"{'kurs > EMA10 > EMA20 > EMA50' if ok else 'stacken är inte i ordning'}"))

    # C. Volym (visas, krävs inte förrän backtestat)
    v = trig["Volume"].astype(float) if "Volume" in trig else None
    if v is None or len(v) < 21 or float(v.iloc[-21:-1].mean()) <= 0:
        checks.append(Check("volume", "Volym", UNAVAILABLE, "volymdata saknas", required=VOLUME_REQUIRED))
    else:
        rv = float(v.iloc[-1]) / float(v.iloc[-21:-1].mean())
        checks.append(Check("volume", "Volym", PASS if rv >= RELATIVE_VOLUME_MIN else FAIL,
                            f"relativ volym {rv:.2f}× (gräns {RELATIVE_VOLUME_MIN:.2f}×)"
                            + ("" if VOLUME_REQUIRED else " — visas, krävs inte förrän backtestat"),
                            required=VOLUME_REQUIRED))

    # Entry candle
    green = float(c.iloc[-1]) > float(o.iloc[-1]) and float(c.iloc[-1]) >= float(c.iloc[-2])
    trig_date = str(trig.index[-1])[:10]
    if not closed and not green:
        checks.append(Check("candle", "Entry candle", FAIL, f"dagens candle är inte stängd och föregående "
                            f"({trig_date}) var inte grön", flag=WAIT_CLOSE))
    elif not closed:
        checks.append(Check("candle", "Entry candle", FAIL, f"dagens candle är inte stängd — trigger = "
                            f"{trig_date}; vänta på stängning innan entry", flag=WAIT_CLOSE))
    else:
        checks.append(Check("candle", "Entry candle", PASS if green else FAIL,
                            f"{CONFIRMED if green else FAILED}: {trig_date} stängde "
                            f"{'grön' if green else 'inte grön'} (close {entry:,.2f} mot open "
                            f"{float(o.iloc[-1]):,.2f}, föregående close {float(c.iloc[-2]):,.2f})"))

    # No chase
    dist = (price / entry - 1) * 100
    ok = dist <= MAX_CHASE_PCT
    checks.append(Check("chase", "No chase", PASS if ok else FAIL,
                        f"triggerns stängning {entry:,.2f} · kurs nu {price:,.2f} · avstånd {dist:+.2f} % "
                        f"(max {MAX_CHASE_PCT:g} %)", flag=None if ok else NO_CHASE))

    # Motstånd och R/R (stoppen = 1,5 × ATR14)
    a = float(atr(trig).iloc[-1])
    stop_distance = a * ATR_STOP_MULT
    res, src = nearest_resistance(trig, entry, ob_analysis)
    if res is None:
        rr = None
        checks.append(Check("rr", "Motstånd / R/R", PASS, f"{src} — fri väg uppåt, R/R mäts inte"))
    else:
        rr = round((res - entry) / stop_distance, 2) if stop_distance > 0 else None
        ok = rr is not None and rr >= MINIMUM_RR
        checks.append(Check("rr", "Motstånd / R/R", PASS if ok else FAIL,
                            f"motstånd {res:,.2f} ({src}) · uppsida {res - entry:,.2f} · risk {stop_distance:,.2f} "
                            f"→ {rr:.2f}R (minst {MINIMUM_RR:g}R)", flag=None if ok else INSUFFICIENT_RR))

    # Earnings
    today = pd.Timestamp(now.date() if now is not None else pd.Timestamp.today().date())
    if earnings_date is None or not earnings_known:
        checks.append(Check("earnings", "Earnings", UNAVAILABLE if EARNINGS_UNKNOWN_BLOCKS else INFO,
                            "rapportdatum okänt — kontrollera själv före entry", required=EARNINGS_UNKNOWN_BLOCKS))
    else:
        days = trading_days_until(earnings_date, today)
        near = 0 <= days <= EARNINGS_BUFFER_DAYS
        checks.append(Check("earnings", "Earnings", FAIL if near else PASS,
                            f"rapport {str(pd.Timestamp(earnings_date).date())} — "
                            + (f"om {days} handelsdagar (≤ {EARNINGS_BUFFER_DAYS}): ingen ny position"
                               if near else f"{days} handelsdagar bort" if days > 0 else "passerad"),
                            flag=EARNINGS_RISK if near else None))
    return checks, entry, trig_date, stop_distance, res, src, rr, a


def evaluate_entry(ticker: str, df: pd.DataFrame, nine=None, capital: float = 100_000.0,
                   ob_analysis: Optional[dict] = None, earnings_date=None, earnings_known: bool = True,
                   trades: Optional[list] = None, now: Optional[datetime] = None,
                   current_price: Optional[float] = None, max_position_pct: float = MAX_POSITION_PCT,
                   risk_pct: float = MAX_RISK_PCT, market_risk: Optional[dict] = None) -> EntryDecision:
    """Setup (Nine) + exekvering + risk → GOLDEN TICKET / WAIT / NO TRADE med skäl.
    market_risk: market_risk_gate-nivån för aktiens marknad — FÖRHÖJD eller HÖG spärrar nya entries."""
    nine_passed = getattr(nine, "passed", None)
    nine_total = 9
    if df is None or len(df) < 60:
        return EntryDecision(ticker, NO_TRADE, nine_passed, nine_total,
                             reasons=["DATA UNAVAILABLE — för lite kurshistorik (kräver 60 dagar)"])
    checks, entry, trig_date, stop_dist, res, src, rr, a = execution_checks(
        ticker, df, ob_analysis, earnings_date, earnings_known, now, current_price)
    pos = size_position(capital, entry, a, risk_pct=risk_pct, max_position_pct=max_position_pct)
    d = EntryDecision(ticker, WAIT, nine_passed, nine_total, checks, pos, round(entry, 2), trig_date,
                      None if res is None else round(res, 2), src, rr)
    d.flags = [c.flag for c in checks if c.flag]

    today = now if now is not None else datetime.now()
    losses = daily_losses(trades or [], today)
    hard, soft = [], []
    if losses >= MAX_DAILY_LOSSES:
        d.flags.append(DAILY_LIMIT)
        hard.append(f"{DAILY_LIMIT} — {losses} förlustaffärer i dag. Trading disabled for today.")
    if nine_passed is None:
        soft.append("OVTLYR Nine saknas — setupen kan inte bedömas")
    elif nine_passed < NINE_DEVELOPING:
        hard.append(f"OVTLYR Nine {nine_passed}/9 — setupen finns inte (≤ {NINE_DEVELOPING - 1}/9)")
    elif nine_passed < 9:
        soft.append(f"OVTLYR Nine {nine_passed}/9 — DEVELOPING, kräver 9/9")
    spy = nine.get("market.signal") if nine is not None and hasattr(nine, "get") else None
    if spy is not None and spy.status == "FAIL":
        hard.append(f"{getattr(nine, 'market_label', 'SPY')} under EMA20 — marknadens säljsignal, "
                    f"inga nya affärer")
    import market_risk_gate as mrg
    if mrg.blocks_viking_entry(market_risk):
        flag = MARKET_RISK_HIGH if market_risk.get("level") == mrg.HIGH else MARKET_RISK_ELEVATED
        d.flags.append(flag)
        hard.append(f"{flag} ({market_risk.get('label', '')}: {market_risk.get('points')} av "
                    f"{market_risk.get('possible')} varningar) — inga nya entries")
    if any(c.flag == EARNINGS_RISK for c in checks):
        hard.append("EARNINGS RISK — rapport inom fem handelsdagar")
    for c in checks:
        if c.required and not c.passed and c.flag != EARNINGS_RISK:
            soft.append(f"{c.label}: {c.flag or c.status} — {c.detail}")
    if pos is None or pos.shares <= 0:
        hard.append("Positionen blir 0 aktier — stoppavståndet är större än riskbudgeten")

    if hard:
        d.status, d.reasons = NO_TRADE, hard + soft
    elif soft:
        d.status, d.reasons = WAIT, soft
    else:
        d.status = GO
        d.reasons = [f"OVTLYR Nine 9/9 och Viking Execution klar — risk {pos.risk_pct:g} % "
                     f"({pos.risk_amount:,.0f}), {pos.shares} aktier"]
    return d


def watchlist_category(nine_passed: Optional[int], decision_status: Optional[str] = None) -> str:
    """GOLDEN TICKET (9/9 + PASS) · READY (9/9 + WAIT) · DEVELOPING (7–8/9) · REJECTED (≤ 6/9).
    Beskriver hur många villkor som är uppfyllda — inte aktiens kvalitet."""
    if nine_passed is None:
        return "REJECTED"
    if nine_passed >= 9:
        return "GOLDEN TICKET" if decision_status == GO else "READY"
    return "DEVELOPING" if nine_passed >= NINE_DEVELOPING else "REJECTED"
