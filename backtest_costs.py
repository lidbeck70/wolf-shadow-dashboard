"""
backtest_costs.py — enkel kostnadsmodell per affär för backtesterna: courtage, spread och växling efter var
papperet handlas, dras från R i efterhand. Motorerna (viking_backtest, berserk.backtest) är oförändrade.

Räknemetoden är viking_robustness.with_costs (kostnad i R = kostnad i % / stoppavstånd i %, plus extra
glidning på stoppar). Här väljs bara nivån per affär i stället för en fast nivå:

  Norden   courtage 2 × 15 bp + halv spread 2 × 5 bp                 = 40 bp tur och retur
  Utland   samma + valutaväxling 2 × 25 bp (svenskt konto)           = 90 bp tur och retur
  Stopp    + 25 bp glidning (viking_robustness.GAP_SLIPPAGE_BPS) när affären gick ut på ett stopp

Nivåerna är antaganden för ett litet konto hos en svensk nätmäklare, inte uppmätta. Ändra dem här.
"""
from __future__ import annotations

from typing import Optional

NORDIC_SUFFIXES = (".ST", ".HE", ".CO", ".OL", ".IC")
ROUND_TRIP_BPS = {"norden": 40.0, "utland": 90.0}


def market_of(ticker: str) -> str:
    return "norden" if str(ticker or "").upper().endswith(NORDIC_SUFFIXES) else "utland"


def bps_for(ticker: str) -> float:
    """Tur och retur i baspunkter för papperet (utan stoppglidning)."""
    return ROUND_TRIP_BPS[market_of(ticker)]


def net_trades(trades: list, gap_bps: Optional[float] = None) -> list:
    """Kopior av affärerna med R efter kostnad (öppna och ofullständiga affärer orörda)."""
    import viking_robustness as rb
    gap = rb.GAP_SLIPPAGE_BPS if gap_bps is None else gap_bps
    return [rb.with_costs([t], bps_for(t.ticker), gap)[0] for t in trades]


def compare(trades: list) -> dict:
    """Brutto mot netto för stängda affärer: n, expectancy, summa R och snittkostnad i R."""
    net = net_trades(trades)
    pairs = [(a.r, b.r) for a, b in zip(trades, net) if not a.open and a.r is not None and b.r is not None]
    if not pairs:
        return {"n": 0, "exp": None, "exp_net": None, "sum_r": 0.0, "sum_r_net": 0.0, "cost_r": None}
    g, n = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"n": len(pairs), "exp": round(sum(g) / len(g), 3), "exp_net": round(sum(n) / len(n), 3),
            "sum_r": round(sum(g), 1), "sum_r_net": round(sum(n), 1),
            "cost_r": round((sum(g) - sum(n)) / len(pairs), 3)}
