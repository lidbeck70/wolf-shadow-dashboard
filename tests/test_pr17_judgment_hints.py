"""
PR 17 — förslag på bedömningarna, aldrig ifyllnad.

Jurisdiktion ur Fraser-listan (utdraget eller Durrett-arkets bolag), insyn
och Sprotts ägarfaktor ur insynsägandet, DS "Historik" ur fem års
aktiehistorik. Arken visar förslaget som text med källa; kryssa gör du.
"""
import copy
import os
import sys

import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import judgment_hints as jh   # noqa: E402


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── Reglerna ─────────────────────────────────────────────────────────────────
def test_jurisdiction_follows_the_fraser_lists_and_stays_silent_when_unknown():
    assert jh.jurisdiction_verdict("Nevada, USA") == (True, "nevada")
    assert jh.jurisdiction_verdict("Val-d'Or, Québec") == (True, "québec")
    assert jh.jurisdiction_verdict("Chubut, Argentina") == (False, "chubut")
    assert jh.jurisdiction_verdict("Mali") is None and jh.jurisdiction_verdict("") is None
    assert jh.jurisdiction_points("Western Australia") == 2 and jh.jurisdiction_points("Zimbabwe") == 0
    assert jh.jurisdiction_points("Peru") is None


def test_insider_and_dilution_thresholds():
    assert jh.insider_ok(4.9) is False and jh.insider_ok(5.0) is True and jh.insider_ok(None) is None
    assert jh.owner_points(12) == 2 and jh.owner_points(10) == 1 and jh.owner_points(7) == 1
    assert jh.owner_points(2) == 0 and jh.owner_points("x") is None
    assert jh.ds_history_points(60) == 2 and jh.ds_history_points(30) == 1
    assert jh.ds_history_points(5) == 0 and jh.ds_history_points(None) is None


# ── Underlaget ───────────────────────────────────────────────────────────────
def _extraction(**fields):
    return {"parsed": {"fields": {k: {"value": v} for k, v in fields.items()}}, "doc": "DFS 2025"}


def test_gather_prefers_the_extraction_and_falls_back_to_the_durrett_sheet():
    from confidence import store as cs
    from confidence.data.provenance import dp
    import durrett_cases as dcs
    data = cs.default()
    c = dcs.gold_producer()
    c.jurisdiction, c.country = "Quebec", "Kanada"
    c.set("insider_ownership_pct", dp(8.0, kind="ACTUAL", source="MD&A"))
    cs.put(data, c)
    g = jh.gather("GPR", data, _extraction(jurisdiction="Nevada, USA", insider_ownership=12))
    assert g["jurisdiction"] == ("Nevada, USA", "utdrag ur DFS 2025")
    assert g["insider_pct"] == (12.0, "utdrag ur DFS 2025")
    g = jh.gather("GPR", data, None)                                 # inget utdrag → arket
    assert g["jurisdiction"] == ("Quebec, Kanada", "Durrett-arket")
    assert g["insider_pct"] == (8.0, "Durrett-arket")
    assert jh.gather("FINNS-INTE", data, None) == {}


def test_hints_are_text_with_source_and_go_quiet_when_the_sheet_already_agrees(monkeypatch):
    monkeypatch.setattr(jh, "gather_for", lambda t: {
        "jurisdiction": ("Nevada, USA", "utdrag ur DFS 2025"), "insider_pct": (12.0, "Durrett-arket")})
    h = jh.jurisdiction_hint("X", False)
    assert h.startswith("Förslag: Jurisdiktion OK") and "Nevada" in h and "övre halvan" in h and "Kryssa själv" in h
    assert jh.jurisdiction_hint("X", True) is None                    # stämmer redan
    assert jh.jurisdiction_hint("X", 0, as_points=True).startswith("Förslag: Jurisdiktion 2 p")
    assert jh.jurisdiction_hint("X", 2, as_points=True) is None
    assert jh.insider_hint("X", False).startswith("Förslag: Insynsägande ✓") and "12 %" in jh.insider_hint("X", False)
    assert jh.insider_hint("X", 1, as_points=True).startswith("Förslag: Ägare & management 2 p")
    assert jh.insider_hint("X", 2, as_points=True) is None
    monkeypatch.setattr(jh, "gather_for", lambda t: {"jurisdiction": ("Mali", "x")})
    assert jh.jurisdiction_hint("X", False) is None                   # okänd jurisdiktion: tyst
    assert jh.insider_hint("X", False) is None


# ── Kopplingarna ─────────────────────────────────────────────────────────────
def test_sheets_show_the_hints_and_the_job_reports_five_year_growth():
    assert "jh.jurisdiction_hint(row.get(\"ticker\", \"\"), bool(row.get(\"jurisdiktion\"))" in _src("producers.py")
    assert "jh.insider_hint(row.get(\"ticker\", \"\"), bool(row.get(\"insyn\"))" in _src("producers.py")
    assert "as_points=True" in _src("scoring.py") and "jh.insider_hint" in _src("scoring.py")
    assert 'jh.jurisdiction_hint(cand.get("ticker", ""), fac.get("jurisdiktion"), as_points=True)' in _src("tiggre.py")
    assert '"shares_growth_5y_pct", row_id=key' in _src("controls_ui.py")
    import sheets_refresh as sr
    from test_pr15_report_fields import _Reports
    out = sr.report_fields(_Reports(years=tuple(range(2019, 2026))), 7)
    assert out["shares_5y_ago_m"] == 90.0 and out["shares_growth_5y_pct"] == 55.6      # 140/90


def test_rick_rule_page_shows_the_proposals_from_an_extraction(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import storage
    stores = {"producers": {"producers": [{"id": "p1", "ticker": "NEM", "name": "Newmont",
                                           "commodity": "Guld", "date": "2026-09-01", "factors": {}}],
                            "royalty": []}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import refresh_ui, screens_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setenv("PR17_TEST_ROOT", ROOT)

    def app():
        import os as _o, sys as _s
        _s.path.insert(0, _o.environ["PR17_TEST_ROOT"])
        import producers
        producers.render_producers_page(sheet="Rick Rule")

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["xt_doc:NEM"] = {
        "parsed": {"fields": {"jurisdiction": {"value": "Nevada, USA"}, "insider_ownership": {"value": 12}}},
        "doc": "Presentation Q2", "sheet": "Tiggre", "model": "m", "pages": 3, "chars": 100, "when": "2026-09-23 10:00"}
    at.run()
    assert not at.exception, at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "Förslag: Jurisdiktion OK" in caps and "Nevada" in caps
    assert "Förslag: Insynsägande ✓" in caps and "Presentation Q2" in caps
