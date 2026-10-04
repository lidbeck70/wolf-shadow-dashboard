"""
viking_screen.py — Viking Nine-screenern: en resultatlista och signalloggen.

Skanningen går i två steg så att ett helt universum (Norden, USA …) hinner:
  Steg 1  alla tickers i en batch (bara stängningskurser): aktiens Trend och
          Signal ur OVTLYR Nine — utan dem kan Nine aldrig bli 9/9. Klara
          tickers rangordnas på momentum (avkastning 3 mån).
  Steg 2  de bästa kandidaterna (förval 40) får full OVTLYR Nine (sektor, F&G,
          order blocks) och Viking Execution. Rapportdatum hämtas bara för 9/9.

  RESULTAT (results)     en lista sorterad på kategori och sedan Nine; filtret
                         "bara momentumkandidater" kräver:
                         Nine ≥ 7/9 · kurs > EMA10 > EMA20 > EMA50 · RSI > 50 och
                         stigande momentum · inget bearish block nära · R/R ≥ 2 ·
                         relativ volym (valfritt krav, av som standard)
  KATEGORI               GOLDEN TICKET (9/9 + PASS) · READY (9/9 + WAIT) ·
                         DEVELOPING (7–8/9) · REJECTED (≤ 6/9)
  SIGNALLOGG             varje GOLDEN TICKET / READY med tidsstämpel, Nine per
                         lager, Viking-poäng, entry, stopp, ATR, position, R/R,
                         skäl och åtgärd — en rad per ticker och dag

  SEKTOR UPPTAGEN        en aktie per sektor: har du redan en öppen position i
                         Viking Portfolio (positions.py) i samma sektor (sektor-ETF)
                         flaggas kandidaten, och GOLDEN TICKET blir READY — entryn
                         väntar tills sektorn är ledig (backtest: lägre drawdown)

Screening ≠ automatisk entry. Kategorierna beskriver hur många av systemets
villkor som är uppfyllda — inte aktiens kvalitet.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import pandas as pd

import ovtlyr_nine as on
import viking_execution as vx

MAX_TICKERS = 60                   # egen lista
MAX_CANDIDATES = 40                # steg 2, förval
CANDIDATES_CAP = 100
BATCH_SIZE = 100
MOMENTUM_DAYS = 63                 # rangordning i steg 1: avkastning tre månader
MOMENTUM_MIN_NINE = 7
LOG_STORE = "viking_signals"
LOG_MAX = 1000
LOGGED_CATEGORIES = ("GOLDEN TICKET", "READY")
CATEGORY_ORDER = ("GOLDEN TICKET", "READY", "DEVELOPING", "REJECTED")
SECTOR_BUSY = "SEKTOR UPPTAGEN"
HELD_BUCKET = "ovtlyr"             # positions.py: Viking Portfolio
US_NOTE = ("USA-signaler: i backtestet (USA 25, 5 år) gav Viking Nine bara +0,07R per affär — låg kant. "
           "Bäst som komplement till Norden, inte som eget system.")


def held_sectors(sector_getter: Optional[Callable] = None, positions_getter: Optional[Callable] = None,
                 bd_sector: Optional[Callable] = None) -> dict:
    """{sektor-ETF: ticker} för öppna positioner i Viking Portfolio. Okänd sektor räknas inte."""
    if positions_getter is None:
        import positions
        positions_getter = lambda: positions.open_positions(bucket=HELD_BUCKET)  # noqa: E731
    out = {}
    for p in positions_getter() or []:
        t = str(p.get("ticker") or "").upper()
        if not t:
            continue
        try:
            etf, _src = on.resolve_sector(t, sector_getter, bd_sector)
        except Exception:
            etf = None
        if etf and etf not in out:
            out[etf] = t
    return out


def sector_busy(ticker: str, sector_etf: Optional[str], held: Optional[dict]) -> Optional[str]:
    """Tickern som redan håller sektorn — None om sektorn är ledig, okänd eller aktien själv ägs."""
    owner = (held or {}).get(sector_etf) if sector_etf else None
    return owner if owner and owner != str(ticker).upper() else None


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
                    risk_getter: Optional[Callable] = None,
                    trades: Optional[list] = None, now: Optional[datetime] = None, today=None,
                    held: Optional[dict] = None) -> dict:
    """En rad: Nine + beslut + kategori. Fel → raden markeras DATA UNAVAILABLE.
    held = {sektor-ETF: ticker} för öppna positioner (held_sectors) → SEKTOR UPPTAGEN."""
    if getter is None:
        from market_prices import ohlcv as getter
    row = {"ticker": ticker, "error": None, "nine": None, "decision": None, "category": "REJECTED",
           "sector_busy": None}
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
    if earnings_getter is not None and nine.passed == on.NINE_TOTAL:      # bara där det kan avgöra GO
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
                          earnings_known=ed is not None, trades=trades, now=now, market_risk=risk)
    category = vx.watchlist_category(nine.passed, d.status)
    busy = sector_busy(ticker, nine.sector_etf, held)
    if busy and category == "GOLDEN TICKET":                     # en aktie per sektor — entryn väntar
        category = "READY"
    row.update(nine=nine, decision=d, category=category, sector_busy=busy)
    return row


# ── Steg 1: hela universumet ────────────────────────────────────────────────
def _closes_default(tickers: list) -> dict:
    from market_prices import closes
    out = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        out.update(closes(tickers[i:i + BATCH_SIZE], on.PERIOD))
    return out


def stage1(closes: dict, today=None) -> tuple:
    """([(ticker, avkastning 3 mån %)] sorterade fallande, {universe, data, passed})."""
    today = today if today is not None else pd.Timestamp.today()
    passed, with_data = [], 0
    for t, c in (closes or {}).items():
        if c is None or len(c) < on.MIN_BARS:
            continue
        c = c.astype(float).dropna()
        if len(c) < on.MIN_BARS or on._is_stale(c, today):
            continue
        with_data += 1
        if not (on._trend(c)[0] and on._signal(c)[0]):
            continue
        ret = (float(c.iloc[-1]) / float(c.iloc[-min(MOMENTUM_DAYS, len(c) - 1) - 1]) - 1) * 100
        passed.append((t, round(ret, 1)))
    passed.sort(key=lambda x: -x[1])
    return passed, {"universe": len(closes or {}), "data": with_data, "passed": len(passed)}


def scan(tickers: list, closes_getter: Optional[Callable] = None, max_candidates: int = MAX_CANDIDATES,
         progress: Optional[Callable] = None, today=None, **kw) -> dict:
    """Hela skanningen: steg 1 över universumet, steg 2 på de bästa kandidaterna."""
    tickers = list(dict.fromkeys(t for t in tickers if t))
    closes = (closes_getter or _closes_default)(tickers)
    closes = {t: closes.get(t) for t in tickers}
    ranked, funnel = stage1(closes, today=today)
    n = max(1, min(int(max_candidates), CANDIDATES_CAP))
    cands = [t for t, _r in ranked[:n]]
    funnel.update(candidates=len(cands), momentum={t: r for t, r in ranked})
    rows = run(cands, progress=progress, today=today, **kw)
    return {"rows": rows, "funnel": funnel}


def run(tickers: list, progress: Optional[Callable] = None, **kw) -> list:
    rows = []
    tickers = list(tickers[:CANDIDATES_CAP])
    for i, t in enumerate(tickers):
        rows.append(evaluate_ticker(t, **kw))
        if progress is not None:
            progress(i + 1, len(tickers), t)
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


def results(rows: list, momentum_only: bool = False, require_volume: bool = False) -> list:
    """EN lista: sorterad på kategori (GOLDEN TICKET → REJECTED), sedan Nine.
    momentum_only = samma urval som VIKING MOMENTUM SCREEN."""
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    ranked = ovtlyr_screen(rows)
    out = sorted(ranked, key=lambda r: order.get(r["category"], len(order)) if r["nine"] is not None else len(order))
    if momentum_only:
        out = [r for r in out if momentum_pass(r, require_volume)[0]]
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
        "reason": "; ".join(([f"{SECTOR_BUSY} ({row['sector_busy']})"] if row.get("sector_busy") else [])
                            + list(d.reasons))[:400],
        "source": source, "label": on.APPROXIMATION,
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
