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
    assert live.can_run("insider") and not live.can_run("quality") and not live.can_run(None)
    assert {"wolf", "viking", "berserk"} <= set(live.HEAVY)          # Arc och BERSERK: 6–40 min, bara i schemat


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
    assert how["Insider Norden"] == "⚡ Skanna nu" and how["Quality Contrarian"].startswith("schemat")
    assert how["Wolf"].startswith("schemat — Arc") and how["BERSERK"].startswith("schemat — BERSERK")


def test_scan_now_button_rebuilds_the_list(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    monkeypatch.delitem(live.HEAVY, "wolf")
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


# ── Granskningen 2026-10-23 (L1, L7, L8, L9, L11) ────────────────────────────
def test_failed_live_scan_keeps_the_gist_rows(monkeypatch):
    """Skannern returnerar {rows: [], error: …} utan undantag — förut tömdes listan och stämplades LIVE."""
    import pytest
    data = dict(tu.DATA)
    failing = lambda d: {**(d.get("arc") or {}), "wolf": {"generated": "2026-10-20T10:00", "rows": [], "error": "yfinance down"}}  # noqa: E731
    with pytest.raises(live.LiveScanError, match="yfinance down"):
        live.run("wolf", data, runners={"wolf": ("arc", failing)})
    empty = lambda d: {**(d.get("arc") or {}), "wolf": {"generated": "2026-10-20T10:00", "rows": [], "error": None}}  # noqa: E731
    with pytest.raises(live.LiveScanError, match="0 rader"):
        live.run("wolf", data, runners={"wolf": ("arc", empty)})
    assert "live" not in data and data["arc"]["wolf"]["rows"]                      # ingenting byttes


def test_failed_live_scan_shows_an_error_and_keeps_the_list(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    monkeypatch.delitem(live.HEAVY, "wolf")
    monkeypatch.setitem(live.RUNNERS, "wolf", ("arc", lambda d: {**(d.get("arc") or {}), "wolf": {"rows": [], "error": "yfinance down"}}))
    at = AppTest.from_function(tu._script)
    at.run(timeout=60)
    at.selectbox(key="thor_pick").select("Wolf").run(timeout=60)
    [btn] = [b for b in at.button if b.label == "⚡ Skanna nu"]
    btn.click().run(timeout=60)
    assert not at.exception
    assert any("yfinance down" in e.value for e in at.error)
    assert "ERIC-B.ST" in " ".join(m.value for m in at.markdown)                      # Gistens rader kvar
    assert not any("skannad live" in c.value for c in at.caption)


def test_scanner_error_fields_show_as_fel_not_saknas():
    import test_thor_candidates as fx
    from thor import candidates as tc
    data = {**fx.BLOBS, "arc": {"wolf": {"generated": fx.GEN, "rows": [], "error": "yfinance: connection refused"},
                                "viking": {"generated": fx.GEN, "rows": [], "error": "boom"}},
            "insider": {"generated": fx.GEN, "clusters": [], "error": "BORSDATA_API_KEY saknas"},
            "screens": {"generated": fx.GEN, "screens": {"rule": {"label": "Rule", "rows": [], "error": "Kräver Pro+"}}}}
    rows = tc.build(data, today="2026-10-15", errors={})
    errs = live.blob_errors(data)
    assert errs["wolf"][0].startswith("yfinance") and errs["insider"][0].startswith("BORSDATA") and "rule: Kräver Pro+" in errs["hav"][0]
    fresh = {k: (asof, s) for k, _l, _n, asof, s in live.freshness(rows, data)}
    assert fresh["wolf"][1] == "FEL" and fresh["insider"][1] == "FEL" and fresh["hav"][1] == "FEL"
    assert fresh["wolf"][0].startswith(fx.GEN[:10])
    assert live.blob_errors(dict(tu.DATA)) == {}                                      # friska blobbar: inget fel


def test_partial_live_scan_is_labelled_as_such():
    import test_thor_candidates as fx
    data = dict(fx.BLOBS)
    out = tui.rebuild(live.run("rule", data, runners={"rule": ("screens", lambda d: {
        "generated": "2026-10-20T10:00:00+00:00", "screens": {"rule": {"label": "Rule", "rows": [{"ticker": "X"}], "error": None}}})}))
    fresh = {k: (asof, s) for k, _l, _n, asof, s in live.freshness(out["rows"], out)}
    assert fresh["rule"][1] == "LIVE (håven)" and "övriga rader 2026-10-14" in fresh["rule"][0]
    assert "2026-10-14" in {c.asof for c in out["rows"] if c.strategy == "rule"}      # kartans rader behåller sitt datum


def test_viking_nine_live_rescans_only_todays_nordic_rows(monkeypatch):
    import viking_nine_scan as vns
    seen = {}
    norden = {"BOL.ST": ("Boliden", "materials"), "ERIC-B.ST": ("Ericsson", "technology")}
    monkeypatch.setattr(vns, "universes", lambda usa=True, **kw: {"norden": dict(norden), "usa": {}})

    def run(members=None, dry_run=False, **kw):
        seen["norden"] = sorted(members["norden"])
        empty = {"funnel": {}, "rows": []}
        return {"payload": {"generated": "nu", "rows": [], "markets": {"norden": dict(empty), "usa": dict(empty)}}}
    monkeypatch.setattr(vns, "run", run)
    old = {"viking_nine": {"markets": {"norden": {"rows": [{"ticker": "BOL.ST"}]}, "usa": {"funnel": {}, "rows": []}}}}
    live._viking_nine(old)
    assert seen["norden"] == ["BOL.ST"]                                                 # bara dagens rad
    live._viking_nine({})
    assert seen["norden"] == ["BOL.ST", "ERIC-B.ST"]                                    # inga rader: hela Norden


def test_live_screens_get_their_new_marks(monkeypatch):
    import screens_scan
    monkeypatch.setattr(live, "_api", lambda: None)
    monkeypatch.setattr(screens_scan, "scan", lambda api: {"generated": "2026-10-20T10:00:00+00:00", "screens": {
        "rule": {"label": "Rule", "rows": [{"ticker": "NEM"}, {"ticker": "AEM"}], "error": None}}})
    prev = {"generated": "2026-10-14T10:00:00+00:00", "screens": {"rule": {"rows": [{"ticker": "NEM", "first_seen": "2026-10-01"}]}}}
    out = live._screens({"screens": prev})
    assert out["screens"]["rule"]["new"] == ["AEM"]
    assert {r["ticker"]: r["first_seen"] for r in out["screens"]["rule"]["rows"]} == {"NEM": "2026-10-01", "AEM": "2026-10-20"}
