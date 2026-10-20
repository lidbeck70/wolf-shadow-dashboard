"""
Thor fas 8, steg 1b–1c — ⚡ Skanna nu (thor.live) och körningarnas färskhet per strategi. Inget nätverk:
skanningsfunktionerna byts mot fejkade.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_ui as tu  # noqa: E402
from thor import live  # noqa: E402
from thor import registry as reg  # noqa: E402
from thor import ui as tui  # noqa: E402

NEW_WOLF = {"generated": "2026-10-20T10:00:00+00:00", "rows": [{"ticker": "ABB.ST", "name": "ABB", "score": 91.0}]}


def _fake_arc(data):
    blob = dict(data.get("arc") or {})
    blob["wolf"] = NEW_WOLF
    return blob


def test_every_strategy_is_live_or_scheduled():
    assert set(live.RUNNERS) | set(live.HEAVY) == set(reg.STRATEGIES)
    assert not set(live.RUNNERS) & set(live.HEAVY)
    assert live.can_run("wolf") and not live.can_run("quality") and not live.can_run(None)


def test_run_swaps_the_blob_and_stamps_live():
    data = dict(tu.DATA)
    out = live.run("wolf", data, runners={"wolf": ("arc", _fake_arc)})
    assert out["arc"]["wolf"] is NEW_WOLF and out["arc"]["viking"] is data["arc"]["viking"]   # Viking behålls
    assert "wolf" in out["live"] and "live" not in data and data["arc"]["wolf"] is not NEW_WOLF  # gamla orörd


def test_viking_nine_live_keeps_usa_from_the_last_run(monkeypatch):
    import viking_nine_scan as vns
    seen = {}

    def universes(usa=True, **kw):
        seen["usa"] = usa
        return {"norden": {"BOL.ST": ("Boliden", "materials")}, "usa": {}}

    def run(members=None, dry_run=False, **kw):
        assert dry_run and members["norden"]
        return {"payload": {"generated": "nu", "rows": [{"ticker": "BOL.ST"}],
                            "markets": {"norden": {"funnel": {}, "rows": [{"ticker": "BOL.ST"}]},
                                        "usa": {"funnel": {"universe": 0}, "rows": []}}}}

    monkeypatch.setattr(vns, "universes", universes)
    monkeypatch.setattr(vns, "run", run)
    old = {"viking_nine": {"markets": {"usa": {"funnel": {"universe": 900}, "rows": [{"ticker": "NEM"}]}}}}
    out = live._viking_nine(old)
    assert seen["usa"] is False
    assert [r["ticker"] for r in out["rows"]] == ["BOL.ST", "NEM"] and out["markets"]["usa"]["funnel"]["universe"] == 900


def test_freshness_states():
    rows = tu._rows()
    fresh = {k: (n, state) for k, _, n, _, state in live.freshness(rows, {"live": {"berserk": "2026-10-20 10:00"}})}
    assert fresh["berserk"][1] == "LIVE"
    assert fresh["wolf"][0] == 3 and fresh["wolf"][1] in ("OK", "GAMMAL")
    missing = [k for k, (n, s) in fresh.items() if not n and k != "berserk"]
    assert missing and all(fresh[k][1] == "SAKNAS" for k in missing)
    table = tui.freshness_rows(rows, {})
    how = {r["Strategi"]: r["Körs"] for r in table}
    assert how["Wolf"] == "⚡ Skanna nu" and how["Quality Contrarian"].startswith("schemat")


def test_scan_now_button_rebuilds_the_list(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    monkeypatch.setitem(live.RUNNERS, "wolf", ("arc", _fake_arc))
    at = AppTest.from_function(tu._script)
    at.run(timeout=60)
    assert not [b for b in at.button if b.label == "⚡ Skanna nu"]                 # alla strategier: ingen knapp
    at.selectbox(key="thor_pick").select("Wolf").run(timeout=60)
    assert not at.exception
    assert "ERIC-B.ST" in " ".join(m.value for m in at.markdown)
    [btn] = [b for b in at.button if b.label == "⚡ Skanna nu"]
    btn.click().run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "ABB.ST" in text and "ERIC-B.ST" not in text and "1 KANDIDATER" in text
    assert any("skannad live" in c.value for c in at.caption)


def test_heavy_strategy_says_scheduled(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    at = AppTest.from_function(tu._script)
    at.run(timeout=60)
    at.selectbox(key="thor_pick").select("Quality Contrarian").run(timeout=60)
    assert not at.exception
    assert not [b for b in at.button if b.label == "⚡ Skanna nu"]
    assert any("körs i schemat" in c.value for c in at.caption)
