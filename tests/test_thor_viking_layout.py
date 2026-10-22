"""
Thor, Viking Nine: en beslutsrad (verdikt, siffror, första grinden som stoppar, stängningarna), en kompakt
9-ruta och en Execution-rad; orsakerna i en expander. Viking visar sin screenerrad och en knapp till Viking Nine.
Inga regler ändrade — samma motorer. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ovtlyr_nine as on  # noqa: E402
import viking_execution as vx  # noqa: E402
from thor import cards  # noqa: E402
from thor import regim  # noqa: E402


def _nine(fails=()):
    r = on.NineResult(ticker="NEM", sector_etf="GDX", market_label="SPY")
    spec = {"market": ("Trend", "Signal", "Breadth"), "sector": ("Fear & Greed", "Breadth"),
            "stock": ("Trend", "Signal", "Fear & Greed", "Blocks")}
    for layer, labels in spec.items():
        for lbl in labels:
            key = f"{layer}.{lbl.lower().replace(' & ', '_')}"
            status = on.FAIL if key in fails else on.PASS
            getattr(r, layer).append(on.Factor(key, lbl, layer, status, detail=f"{lbl} säger {status}", timestamp="2026-10-21"))
    return r


def _decision(status=vx.GO, failed=(), reasons=()):
    checks = [vx.Check(k, lbl, vx.FAIL if k in failed else vx.PASS, f"{lbl} …", flag=vx.WAIT_CLOSE if k == "candle" and k in failed else None)
              for k, lbl in (("momentum", "Momentum"), ("structure", "Trendstruktur"), ("candle", "Entry candle"),
                             ("volume", "Volym"), ("chase", "No chase"), ("rr", "Motstånd / R/R"), ("earnings", "Earnings"))]
    return vx.EntryDecision(ticker="NEM", status=status, nine_passed=9, nine_total=9, checks=checks, entry=57.67,
                            rr=2.4, resistance=70.0, reasons=list(reasons))


def test_first_block_follows_the_order_of_play():
    text, col = regim.first_block(_nine(fails=("market.trend", "stock.signal")), _decision(vx.WAIT, failed=("candle",)))
    assert text.startswith("Nine: Market Trend (FAIL)") and col == regim.RED           # marknaden före aktien
    text, col = regim.first_block(_nine(), _decision(vx.WAIT, failed=("candle",)))
    assert text.startswith("Execution: Entry candle (WAIT FOR CLOSE)") and col == regim.AMBER
    text, col = regim.first_block(_nine(), _decision(vx.NO_TRADE, reasons=["DAILY LOSS LIMIT REACHED — 2 förlustaffärer"]))
    assert text.startswith("DAILY LOSS LIMIT") and col == regim.RED
    text, col = regim.first_block(_nine(), _decision())
    assert text.startswith("inget") and col == regim.GOLD


def test_the_three_rows_say_what_the_cards_said():
    nine = _nine(fails=("market.trend", "market.signal"))
    html = cards.vn_nine_compact(nine)
    for part in ("MARKET SPY", "1/3 ✗✗✓", "SECTOR GDX", "2/2 ✓✓", "STOCK", "4/4", "TOTAL", "7/9", "Trend · Signal"):
        assert part in html, part
    d = _decision(vx.WAIT, failed=("candle", "volume"))
    row = cards.vn_execution_row(d)
    assert "VIKING EXECUTION 5/7" in row and "✗ Entry candle · WAIT FOR CLOSE" in row and "✓ Momentum" in row
    top = cards.vn_decision_row("WAIT", "7/9", "5/7", "Nine: Market Trend (FAIL)", regim.RED,
                                regim._levels(d), regim.bars_caption(nine))
    for part in ("WAIT", "Nine <b>7/9</b>", "Execution <b>5/7</b>", "Stoppar: Nine: Market Trend", "entry 57.67",
                 "R 2.4", "Stängda candles: aktien t.o.m. 2026-10-21"):
        assert part in top, part
    free = vx.EntryDecision(ticker="X", status=vx.GO, nine_passed=9, nine_total=9, entry=10.0, rr=None, resistance=None)
    assert regim._levels(free) == "entry 10.00 · R fri väg"
