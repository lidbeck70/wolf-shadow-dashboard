"""
Viking PR 1 — felrättningar före ombyggnaden (OVTLYR Nine / Viking Execution).

  1. SL/TP-kalkylatorn på Viking Regime hade 5 % risk som standard → 530
     aktier i stället för 159 (kapital 100 000, ATR 6,29, 1,5 × ATR). Nu
     motorns 1,5 %.
  2. DATA UNAVAILABLE räknas aldrig som PASS: F&G var en konstant (48) som
     klarade två grindar, sektorbredden en påhittad tabell. Nu fäller okänd
     data sina grindar och visas som DATA UNAVAILABLE. Holdings, som skickar
     riktiga värden, påverkas inte.
  3. Bull List-mätaren säger inte längre BEST ENTRY på rädsla ensam.
  4. Viking-backtestet stoppar på motorns 1,5 × ATR, inte 0,5 ×.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _code(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


# ── 1. Positionsstorlek ──────────────────────────────────────────────────────
def test_viking_sizing_example_from_the_spec():
    from strategies.viking import DEFAULT_PARAMS as P
    assert P["risk_pct"] == 0.015 and P["atr_stop_mult"] == 1.5
    budget = 100_000 * P["risk_pct"]
    stop = 6.29 * P["atr_stop_mult"]
    assert budget == 1500 and math.isclose(stop, 9.435)
    assert math.floor(budget / stop) == 158          # 158,98 — avrundas nedåt, aldrig mer än 1,5 %
    assert int(100_000 * 0.05 / stop) == 529          # det gamla 5 %-felet


def test_regime_calculator_reads_the_engine_risk():
    src = _code(_src("ovtlyr", "ui", "layout.py"))
    assert 'value=5.0, min_value=0.5' not in src
    assert '_VK_RISK.get("risk_pct"' in src and "value=_vk_risk_pct" in src


# ── 2. DATA UNAVAILABLE räknas aldrig som PASS ───────────────────────────────
def _trend(up=True):
    p = 110.0 if up else 90.0
    return {"price": p, "ema10": 105.0, "ema20": 100.0, "ema50": 95.0, "ema200": 80.0,
            "trend_state": "bullish", "regime_color": "green"}


def _sig(sentiment, sector_green):
    from ovtlyr.signals.longterm_signals import compute_longterm_signal
    return compute_longterm_signal(_trend(), sentiment, {"risk_score": 40, "atr14": 2.0},
                                   {"signal_bias": "BUY"}, sector_green)


def test_unknown_fear_greed_and_breadth_never_pass():
    s = _sig({"score": None, "label": "DATA UNAVAILABLE", "available": False}, None)
    by = {g["rule"][:2]: g for g in s["gates"]}
    for k in ("3.", "4.", "5.", "8."):
        assert by[k]["passed"] is False and by[k]["status"] == "DATA UNAVAILABLE", k
        assert "räknas inte som PASS" in by[k]["detail"]
    for k in ("1.", "2.", "6.", "7.", "9."):
        assert by[k]["status"] == "PASS", k
    # 2/3 × 40 + 0 + 3/4 × 30 = 49 → aldrig BUY på okänd data
    assert s["ovtlyr_nine"] == 49 and s["signal"] == "HOLD"
    assert not any(t["active"] for t in s["exit_triggers"] if "Fear & Greed" in t["trigger"])


def test_known_data_behaves_as_before():
    """Holdings skickar score 50 och sector_green=True — samma resultat som förut."""
    s = _sig({"score": 50, "label": "Neutral"}, True)
    assert s["ovtlyr_nine"] == 100 and s["signal"] == "BUY"
    assert all(g["status"] == "PASS" for g in s["gates"])
    red = _sig({"score": 50}, False)
    by = {g["rule"][:2]: g for g in red["gates"]}
    assert by["3."]["status"] == "FAIL" and by["5."]["status"] == "FAIL"
    hot = _sig({"score": 85}, True)
    assert hot["signal"] == "REDUCE"


def test_regime_page_has_no_fake_data_left():
    src = _src("ovtlyr", "ui", "layout.py")
    assert '"Technology":   {"state": "bullish"' not in src            # påhittad sektortabell borta
    assert "return compute_sentiment({}, {}, {}, {})" not in src        # konstant F&G borta
    assert "if breadth_data else None" in src
    from ovtlyr.ui import layout
    assert layout.SENTIMENT_UNAVAILABLE["score"] is None and layout.SENTIMENT_UNAVAILABLE["available"] is False


# ── 3. Ingen BEST ENTRY på rädsla ensam ──────────────────────────────────────
def test_bull_list_gauge_label():
    src = _src("ovtlyr", "ui", "charts.py")
    assert "BEST ENTRY" not in src and "EXTREME FEAR — POTENTIAL OPPORTUNITY ZONE" in src


# ── 4. Backtestet ────────────────────────────────────────────────────────────
def test_viking_backtest_stop_follows_the_engine():
    import backtest_engine as be
    from strategies.viking import DEFAULT_PARAMS as P
    assert be.atr_stop_mult("ovtlyr") == P["atr_stop_mult"] == 1.5
    assert be.atr_stop_mult("swing") == 0.5 and be.atr_stop_mult("long") == 0.5      # övriga orörda
    assert "atr.iloc[i] * 0.5" not in _src("backtest_engine.py")
