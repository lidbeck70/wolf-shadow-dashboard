"""
Thor-granskningen 2026-10-23, PR A — Regim-fliken: BERSERK-layouten får inte krascha när marknadsserien är en
pandas-Series (`or` på en Series), Long Screener-layouten läser samma Valhall-vy som Cykel-fliken (temat för
utländska råvarubolag), Arkens dom nämner rätt strategi vid cykeltopp, och 'ERIC B' (Börsdata-form) identifieras om
till ERIC-B.ST när listan har raden. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
import test_thor_ui as tu  # noqa: E402
from test_thor_dom import CTX, _judge, _sheets  # noqa: E402
from test_thor_regim import _prices, _run, engines  # noqa: E402,F401,F811
from thor import dom  # noqa: E402
from thor import dossier as td  # noqa: E402


def test_berserk_layout_survives_series_markets(engines, monkeypatch):  # noqa: F811
    """`markets.get(idx) or markets.get('SPY')` kastade ValueError så fort första träffen var en Series —
    varje amerikansk aktie (och nordiska när OMXS30 fanns) fick 'kunde inte ritas'."""
    from berserk import live
    monkeypatch.setattr(live, "load_markets", lambda regions, getter=None, nordic_provider=None:
                        {"SPY": _prices("SPY")["Close"], "OMXS30": _prices("OMXS30")["Close"]})
    for ticker in ("NEM", "BOL.ST"):
        _at, text = _run("berserk", ticker)
        assert "BERSERK — AKTIEN" in text and "kunde inte ritas" not in text and "ambiguous" not in text, ticker


def test_long_screener_layout_uses_the_same_valhall_view_as_the_cycle_tab(engines):  # noqa: F811
    """NEM (USA, tema guld) ska bedömas på temats läge — inte på Nordens Material-status."""
    ident = td.identify("NEM")
    seg = td.cycle(ident, "long", fx.BOARD, phase_fn=lambda t: (None, "")).get("segment") or {}
    assert seg.get("view") == "temat"
    _at, text = _run("long", "NEM")
    assert f"Valhall gron: temat {seg.get('status')}" in text


def test_dom_names_the_judged_strategy_at_the_cycle_top():
    top = {"ctx": {**CTX, "valhall": {**CTX["valhall"], "cykel": "TOPP"}}}
    for strategy, who in (("rule", "Rick Rule köper inte här"), ("durrett", "Durrett köper inte här"),
                          ("tiggre", "Tiggre köper inte här")):
        d = _judge(_sheets(), strategy=strategy, **top)
        [c] = [c for c in d.checks if c.key == "valhall"]
        assert c.status == dom.FAIL and who in c.note, (strategy, c.note)
    checks = dom._cycle_checks("ember", {**CTX, "valhall": {**CTX["valhall"], "cykel": "TOPP"}})
    [c] = [c for c in checks[0] if c.key == "valhall"]
    assert "ingen ny position" in c.note and "Rick Rule" not in c.note


def test_borsdata_form_is_reidentified_from_the_list(monkeypatch):
    """'ERIC B' utan Börsdata blev 'ERIC-B' mot SPY medan kortet visade ERIC-B.ST:s rader."""
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    at = AppTest.from_function(tu._script)
    at.session_state["thor_ticker"] = "ERIC B"
    at.run(timeout=120)
    assert not at.exception, at.exception
    cache = at.session_state["thor_cache"]
    assert "ERIC-B.ST" in cache and cache["ERIC-B.ST"][1]["ident"].yf == "ERIC-B.ST"
    assert cache["ERIC-B.ST"][1]["ident"].nordic
