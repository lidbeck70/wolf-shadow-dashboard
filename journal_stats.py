"""
journal_stats.py — statistikbladet ur tradingjournal_swing.xlsx.

Rena funktioner, ingen Streamlit. Trade Journal-fliken renderar dem; testerna
kör dem mot arkets formler.

Arkets beräknade kolumner:
  L  resultat kr  = (sälj − köp) × antal − courtage
  M  resultat %   = (sälj − köp) / köp
  N  innehav dgr  = säljdatum − köpdatum
  O  R-multipel   = resultat % / ((köp − stop) / köp)

Statistikbladet:
  vinstandel      = COUNTIF(L>0) / COUNT(L)
  payoff-kvot     = AVERAGEIF(L>0) / ABS(AVERAGEIF(L<0))     mål > 2,0
  snitt-R         = AVERAGE(O)                               mål > 0,3R
  snitt innehav   = AVERAGE(N)                               förväntat 14–90 d
  setup A vs B    = AVERAGEIFS(M, setup, "A"/"B")            endast avslutade
  exits per regel = COUNTIF(J, "1"/"2"/"3"/"Delvinst")

R-multipeln är den enda siffran som inte går att fejka i efterhand: den mäter
mot stoppen du faktiskt la, inte mot den du önskar att du lagt. Därför räknas
den ur stoppen här i stället för att skrivas in.
"""

from __future__ import annotations

from typing import Optional

# Arket varnar under 20 avslutade affärer. Masterguiden skriver "15–20";
# routines.MIN_TRADES_FOR_STATS håller den nedre gränsen, det här den övre —
# under den här är vinstandel och payoff-kvot brus.
MIN_TRADES = 20

PAYOFF_TARGET = 2.0        # vinnarna ska vara 2–3x förlusterna
AVG_R_TARGET = 0.3
HOLD_MIN, HOLD_MAX = 14, 90    # förväntad innehavstid i dagar
WIN_RATE_LOW, WIN_RATE_HIGH = 40.0, 55.0   # normalt för momentum

# Säljreglerna (arkets dropdown i kolumn J)
SELL_MA50, SELL_STOP, SELL_RANK = "1", "2", "3"
SELL_PARTIAL = "Delvinst"
SELL_RULES = (SELL_MA50, SELL_STOP, SELL_RANK, SELL_PARTIAL)
SELL_RULE_LABEL = {
    SELL_MA50: "1 — stängning under MA50",
    SELL_STOP: "2 — stop −10 %",
    SELL_RANK: "3 — ur topp 40",
    SELL_PARTIAL: "Delvinst (+20 %, halva)",
}

SETUPS = ("A", "B")
SETUP_LABEL = {"A": "A — pullback", "B": "B — utbrott"}


def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


# ── Per affär ────────────────────────────────────────────────────────────────
def pnl_amount(entry, exit_price, shares, fees=0) -> Optional[float]:
    """(sälj − köp) × antal − courtage."""
    e, x, n = _num(entry), _num(exit_price), _num(shares)
    if e is None or x is None or n is None:
        return None
    return (x - e) * n - (_num(fees, 0.0) or 0.0)


def pnl_pct(entry, exit_price) -> Optional[float]:
    e, x = _num(entry), _num(exit_price)
    if e is None or x is None or e == 0:
        return None
    return (x - e) / e * 100


def risk_pct(entry, stop) -> Optional[float]:
    """Avståndet till stoppen i procent — nämnaren i R-multipeln."""
    e, s = _num(entry), _num(stop)
    if e is None or s is None or e == 0 or e == s:
        return None
    return (e - s) / e * 100


def r_multiple(entry, stop, exit_price=None, result_pct=None) -> Optional[float]:
    """Resultatet i risk-enheter: resultat % ÷ avståndet till stoppen.

    Skicka antingen säljkursen eller ett färdigt resultat i procent.
    """
    risk = risk_pct(entry, stop)
    if risk is None or risk == 0:
        return None
    res = _num(result_pct)
    if res is None:
        res = pnl_pct(entry, exit_price)
    if res is None:
        return None
    return res / risk


def holding_days(entry_date, exit_date) -> Optional[int]:
    """Säljdatum − köpdatum, i dagar."""
    from datetime import date, datetime

    def _d(v):
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        if not v:
            return None
        try:
            return datetime.fromisoformat(str(v)[:10]).date()
        except (ValueError, TypeError):
            return None

    a, b = _d(entry_date), _d(exit_date)
    if a is None or b is None:
        return None
    return (b - a).days


# ── Över alla affärer ────────────────────────────────────────────────────────
def _closed(trades: list) -> list:
    """Endast avslutade affärer — arket räknar på ifylld säljkurs."""
    out = []
    for t in trades or []:
        if not isinstance(t, dict):
            continue
        if _num(t.get("pnl_amount")) is not None or _num(t.get("exit_price")):
            out.append(t)
    return out


def win_rate(trades: list) -> Optional[float]:
    closed = _closed(trades)
    vals = [_num(t.get("pnl_amount")) for t in closed]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return len([v for v in vals if v > 0]) / len(vals) * 100


def payoff_ratio(trades: list) -> Optional[float]:
    """Snittvinst ÷ |snittförlust|. None om någon sida saknas.

    Arket visar "-" tills båda finns — en payoff-kvot utan förluster är inte
    oändligt bra, den är oberäknad.
    """
    vals = [_num(t.get("pnl_amount")) for t in _closed(trades)]
    wins = [v for v in vals if v is not None and v > 0]
    losses = [v for v in vals if v is not None and v < 0]
    if not wins or not losses:
        return None
    return (sum(wins) / len(wins)) / abs(sum(losses) / len(losses))


def average(trades: list, field: str) -> Optional[float]:
    vals = [_num(t.get(field)) for t in _closed(trades)]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def setup_breakdown(trades: list) -> dict:
    """{setup: {"count": n, "avg_pct": x}} för avslutade affärer."""
    out = {}
    for s in SETUPS:
        rows = [t for t in _closed(trades)
                if str(t.get("setup", "")).upper() == s]
        pcts = [_num(t.get("pnl_pct")) for t in rows]
        pcts = [p for p in pcts if p is not None]
        out[s] = {"count": len(rows),
                  "avg_pct": (sum(pcts) / len(pcts)) if pcts else None}
    return out


def exit_breakdown(trades: list) -> dict:
    """{säljregel: antal} — hur du faktiskt kommer ur affärerna."""
    out = {r: 0 for r in SELL_RULES}
    for t in _closed(trades):
        rule = str(t.get("sell_rule", "")).strip()
        if rule in out:
            out[rule] += 1
    return out


def enough_trades(trades: list) -> bool:
    """Nog med avslutade affärer för att statistiken ska betyda något."""
    return len(_closed(trades)) >= MIN_TRADES


def summary(trades: list) -> dict:
    """Hela statistikbladet i ett anrop."""
    closed = _closed(trades)
    return {
        "closed": len(closed),
        "enough": enough_trades(trades),
        "win_rate": win_rate(trades),
        "payoff": payoff_ratio(trades),
        "avg_r": average(trades, "r_multiple"),
        "avg_pct": average(trades, "pnl_pct"),
        "avg_days": average(trades, "holding_days"),
        "total": sum(v for v in (_num(t.get("pnl_amount")) for t in closed)
                     if v is not None),
        "setups": setup_breakdown(trades),
        "exits": exit_breakdown(trades),
    }
