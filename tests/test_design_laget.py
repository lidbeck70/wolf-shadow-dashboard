"""Designförslaget PR 5: ODIN → Läget — fyra ljus ur sparade källor, ingen egen slutsats, inget live."""
import pathlib

from odin import laget as lg
from ui import nav

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOARD = {"segments": [{"key": "a", "nordic": {"status": "GRÖN"}}, {"key": "b", "nordic": {"status": "GRÖN"}},
                      {"key": "c", "nordic": {"status": "RÖD"}}, {"key": "d", "nordic": None}]}
RISK = {"generated": "2026-10-24 06:00 UTC",
        "levels": {"SPY": {"label": "SPY", "level": "LÅG"}, "OMXS30": {"label": "OMXS30", "level": "FÖRHÖJD"}}}


def test_allocation_uses_the_breaker():
    assert lg.allocation_light({"peak": 100, "current": 95})[0] == "NORMAL"
    assert lg.allocation_light({"peak": 100, "current": 85})[0] == "SKÄRPT"
    assert lg.allocation_light({"peak": 100, "current": 80})[0] == "HALVERAD RISK"
    assert lg.allocation_light(None)[0] == lg.SAKNAS


def test_swing_light():
    assert lg.swing_light({"regime": "gul", "generated": "2026-10-24T06:00"}) == ("GUL", "2026-10-24")
    assert lg.swing_light({})[0] == lg.SAKNAS


def test_risk_light_per_market_and_worst():
    assert lg.risk_light(RISK)[0] == "SPY LÅG · OMXS30 FÖRHÖJD"
    assert lg.risk_worst(RISK) == "FÖRHÖJD"
    assert lg.risk_light(None)[0] == lg.SAKNAS


def test_nordic_counts_green_segments():
    assert lg.nordic_light(BOARD) == ("2/3 GRÖN", "GRÖN")
    assert lg.nordic_light({})[0] == lg.SAKNAS


def test_row_has_four_lights_in_order_and_no_verdict():
    items = lg.capital_row({"peak": 100, "current": 99}, {"regime": "GRÖN"}, RISK, BOARD)
    assert [t for t, *_ in items] == ["Allokering", "Swing", "Marknadsrisk", "Norden"]
    text = lg.row_text(items)
    assert text == "Allokering NORMAL · Swing GRÖN · Marknadsrisk SPY LÅG · OMXS30 FÖRHÖJD · Norden 2/3 GRÖN"
    assert "köp" not in text.lower()


def test_laget_is_first_under_marknad_and_swing_alert_lands_there():
    assert nav.options("odin/Marknad")[0] == "Läget"
    assert nav.target_slug("swing_regime") == "odin/marknad/laget"


def test_alert_scan_saves_the_risk_level():
    src = (ROOT / "alert_scan.py").read_text(encoding="utf-8")
    assert 'save_blob("market_risk.json"' in src


def test_laget_does_not_compute_market_risk_on_load():
    src = (ROOT / "odin" / "laget.py").read_text(encoding="utf-8")
    load_src = src.split("def load()")[1].split("def render_laget_page")[0]
    assert "evaluate" not in load_src and "market_risk_gate" not in load_src
