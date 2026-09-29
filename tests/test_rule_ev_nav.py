"""
Rick Rule — EV/NAV som egen värderingsrad, utanför 0–5-poängen.

EV ur Börsdata (sifferuppdateringen eller "Hämta EV nu"), NAV efter skatt och
metallpriset den är räknad på skrivs in (eller föreslås av extraktorn).
"""
import copy
import os
import sys

import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import producers as prod  # noqa: E402


def test_ev_nav_and_bands():
    assert prod.ev_nav(600, 1000) == 0.6
    assert prod.ev_nav(None, 1000) is None and prod.ev_nav(600, 0) is None and prod.ev_nav(600, -5) is None
    lab = lambda ev: prod.ev_nav_verdict({"ev_musd": ev, "nav_musd": 1000, "nav_price": 2500,
                                          "price": 2600}).label
    assert lab(600) == prod.NAV_CHEAP and lab(700) == prod.NAV_FAIR
    assert lab(1000) == prod.NAV_FAIR and lab(1200) == prod.NAV_RICH
    assert prod.ev_nav_verdict({"ev_musd": 600}) is None


def test_nav_price_deck_warning():
    assert prod.nav_price_gap(1800, 3000) == 66.7 and prod.nav_price_gap(None, 3000) is None
    v = prod.ev_nav_verdict({"ev_musd": 600, "nav_musd": 1000, "nav_price": 1800, "price": 3000})
    assert "underskattar" in v.why and "+67 %" in v.why
    v = prod.ev_nav_verdict({"ev_musd": 600, "nav_musd": 1000, "nav_price": 3000, "price": 2000})
    assert "överskattar" in v.why
    v = prod.ev_nav_verdict({"ev_musd": 600, "nav_musd": 1000, "nav_price": 2500, "price": 2600})
    assert "OBS" not in v.why                                   # inom ±20 %
    v = prod.ev_nav_verdict({"ev_musd": 600, "nav_musd": 1000})
    assert "Ange metallpriset" in v.why


def test_ev_nav_does_not_change_the_score():
    row = {"price": 2600, "unit_cost": 1400, "jurisdiktion": True, "kapitaldisciplin": True, "insyn": True}
    before = prod.producer_score(row)
    row.update(ev_musd=5000, nav_musd=1000)                    # 5× NAV — dyrt
    assert prod.producer_score(row) == before == 5
    r = prod.ranked_producers([row])[0]
    assert r["ev_nav"] == 5.0
    csv = prod._csv_row(r)
    assert csv["_ev_nav"] == 5.0 and csv["_ev_nav_band"] == prod.NAV_RICH


def test_refresh_puts_ev_on_rick_rule_rows_only():
    import sheets_refresh as sr
    from test_sheets_refresh import _API, _sheets
    api = _API()
    api.snaps[40]["ev"] = 100000.0                             # BOL, SEK
    rows = sr.refresh(api, _sheets())["rows"]
    assert rows["producers:r1"]["ev_musd"] == round(100000.0 * sr.FX_TO_USD["SEK"], 1)
    assert "ev_musd" not in rows["producers:y1"]                # inte Royalty C


def test_extractor_offers_nav_on_the_rule_sheet():
    from ai import extract_prompt as xp
    keys = {f.key for f in xp.FIELDS["rule"]}
    assert {"nav_musd", "nav_price"} <= keys
    assert ("nav", 1.0) in xp.ALIASES["nav_musd"] and ("nav_musd", 1.0) in xp.ALIASES["npv_musd"]


def _stores(row):
    return {"producers": {"producers": [dict({"id": "p1", "ticker": "NEM", "name": "Newmont",
                                               "commodity": "Guld", "date": "2026-09-01"}, **row)],
                          "royalty": []}}


def _patch(monkeypatch, stores):
    import storage
    import refresh_ui
    import screens_ui
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setenv("EVNAV_TEST_ROOT", ROOT)


def _app():
    import os as _o, sys as _s
    _s.path.insert(0, _o.environ["EVNAV_TEST_ROOT"])
    import producers
    producers.render_producers_page(sheet="Rick Rule")


def test_rick_rule_page_shows_the_ev_nav_row(monkeypatch):
    from streamlit.testing.v1 import AppTest
    _patch(monkeypatch, _stores({"price": 3000, "unit_cost": 1400, "ev_musd": 600.0, "nav_musd": 1000.0,
                                 "nav_price": 1800.0}))
    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("0.60× NAV" in e.label for e in at.expander)          # rubriken
    html = " ".join(m.value for m in at.markdown)
    assert "Värdering — EV/NAV" in html and "0.60× NAV" in html and prod.NAV_CHEAP in html
    assert "underskattar" in html
    assert at.number_input(key="pr_nav_p1").value == 1000.0


def test_fetch_ev_now_writes_ev_from_borsdata(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import borsdata_api
    import sheets_refresh as sr
    _patch(monkeypatch, _stores({"nav_musd": 1000.0}))
    monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda: object())
    monkeypatch.setattr(sr, "fetch_row", lambda api, t, iid=None: {"ev_musd": 850.0, "asof": "2026-09-29"})
    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert any("Fyll i EV och NAV" in c.value for c in at.caption)
    at.button(key="pr_evget_p1").click().run()
    assert not at.exception, at.exception
    assert at.number_input(key="pr_evm_p1").value == 850.0
    assert any("EV 850 MUSD ur Börsdata" in c.value for c in at.caption)
    assert at.session_state["producers"]["producers"][0]["ev_musd"] == 850.0
    html = " ".join(m.value for m in at.markdown)
    assert "0.85× NAV" in html and prod.NAV_FAIR in html
