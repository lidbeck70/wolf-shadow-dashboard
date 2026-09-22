"""
Rapporter, Thesis Killer, rekommendation, Investment Card, förslag ur
granskningsarken, lagringsformen, extraktionsarket — och att fliken ritar.
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence_cases as cc
from confidence import config as cfg
from confidence import prefill
from confidence import reports
from confidence import store as cs
from confidence.data.models import CompanyInput
from confidence.data.provenance import dp
from confidence.why_now import Signals

TODAY = date(2026, 9, 22)


def _an(mk, signals=None):
    return reports.analyze(mk(), cc.COPPER_OVERRIDES, signals, TODAY)


# ── Thesis Killer + rekommendation ───────────────────────────────────────────
def test_thesis_killer_always_has_five_ranked_risks():
    for name, mk in cc.ALL.items():
        a = _an(mk)
        assert len(a.risks) >= reports.MIN_RISKS, name
        sev = [reports._SEV[r.level] for r in a.risks]
        assert sev == sorted(sev, reverse=True), name
        assert all(r.why for r in a.risks), name


def test_recommendation_matrix():
    assert _an(cc.producer_high_quality).recommendation == reports.BUY
    assert _an(cc.developer_high_quality).recommendation == reports.BUY
    ov = _an(cc.developer_overvalued)
    assert ov.recommendation == reports.WATCH and any(r.name == "Värdering" and r.level == "HIGH" for r in ov.risks)
    weak = _an(cc.developer_excellent_low_confidence)
    assert weak.recommendation == reports.REJECT and "kill switch" in weak.recommendation_why
    assert {r.name for r in weak.risks if r.level == "CRITICAL"} == {"Finansiering", "Resurs"}
    assert _an(cc.developer_cheap_but_poor).recommendation == reports.PASS
    assert _an(cc.missing_everything).recommendation == reports.PASS
    exp = _an(cc.explorer_early)
    assert exp.recommendation == reports.PASS
    assert any(r.name == "Resurs" and r.level == "HIGH" for r in exp.risks)      # explorer: HIGH, inte CRITICAL


def test_stress_and_leverage_show_up_as_risks():
    lev = _an(cc.producer_leveraged)
    names = {r.name: r.level for r in lev.risks}
    assert names["Skuldsättning"] == "HIGH" and names["Råvarupris"] == "HIGH"
    st_ = _an(cc.developer_price_stressed)
    assert any(r.name == "Råvarupris" and r.level == "HIGH" and "TAK" in r.why for r in st_.risks)
    top = _an(cc.developer_high_quality, Signals(cycle_label="TOPP"))
    assert any(r.name == "Cykelläge" and r.level == "HIGH" for r in top.risks)


def test_investment_card_has_the_claude_md_fields():
    card = reports.investment_card(_an(cc.developer_high_quality))
    for k in ("ticker", "strategy", "regime", "sector_score", "setup_score", "passed_rules", "failed_rules",
              "ai_comment", "entry_zone", "stop", "target", "recommendation", "confidence", "asymmetry"):
        assert k in card, k
    assert card["rating"] == "PICK" and card["sleeve"] == "optionalitet" and card["failed_rules"] == []
    assert card["asymmetry"].endswith("(STARK ASYMMETRI)") and card["missing_fields"] == 0
    poor = reports.investment_card(_an(cc.developer_cheap_but_poor))
    assert "Economics" in poor["failed_rules"] and "Valuation" in poor["passed_rules"]


def test_report_markdown_and_json_are_complete():
    a = _an(cc.developer_high_quality)
    md = reports.report_markdown(a)
    for section in ("## Case Score", "## Confidence Score", "## Why Now", "## Regional knapphet",
                    "## Time-to-money", "## Scenarier", "## Thesis Killer", "**Antaganden**", "5× — vad måste hända"):
        assert section in md, section
    assert "DATA_MISSING" not in md.split("## Confidence Score")[0]   # DEV har alla Case-fält
    d = reports.analysis_json(a)
    json.dumps(d)
    assert d["card"]["ticker"] == "DEV" and d["case"]["total"] == a.case.total
    md2 = reports.report_markdown(_an(cc.missing_everything))
    assert "## DATA_MISSING" in md2


def test_signals_feed_why_now_and_ttm_fills_in():
    a = _an(cc.developer_high_quality, Signals(cycle_label="TIDIG"))
    assert a.why_now.components["Cykelläge"] == 30 and a.why_now.components["Time-to-money"] == 4   # 3 år
    assert "Why Now" in a.why_now.notes[0] or a.why_now.notes


# ── förslag ur granskningsarken ──────────────────────────────────────────────
PRODUCERS = {"producers": [{"id": "a1", "ticker": "dev", "name": "x", "date": "2026-08-01", "unit_cost": 2.4,
                            "mine_life": 17, "price": 4.4, "aqs_kostnad": 2, "aqs_livslangd": 2,
                            "aqs_metallurgi": 1, "aqs_infrastruktur": 1, "aqs_expansion": 2, "aqs_management": 2,
                            "ds_runway": 0, "ds_capex": 1, "ds_warrants": 0, "ds_aktier_3ar": 1, "ds_historik": 0}],
             "royalty": []}
TIGGRE = {"candidates": [{"id": "t1", "ticker": "DEV", "nav": 1900, "mcap": 720,
                          "screen": {"fs": True, "permits": True, "funded": False}}]}


def test_prefill_maps_sheets_to_datapoints_with_source():
    c = CompanyInput(ticker="DEV", commodity="copper", stage="developer", maturity="pea")
    props = dict((k, v) for k, v, _a in prefill.proposals(c, PRODUCERS, TIGGRE))
    assert props["res_grade"].value == 2 and props["res_mine_life"].value == 1 and props["res_metallurgy"].value == 0
    assert props["res_infrastructure"].value == 2 and props["res_expansion"].value == 3
    assert props["mgmt_track_record"].value == 1 and props["dilution_score"].value == 2
    assert props["aisc"].value == 2.4 and props["mine_life_years"].value == 17 and props["commodity_price"].value == 4.4
    assert props["npv_musd"].value == 1900 and props["nav_musd"].value == 1900 and props["market_cap_musd"].value == 720
    assert props["permits_granted"].value is True and props["financing_committed"].value is False
    assert props["maturity"] == "dfs"
    assert all(p.kind == "ESTIMATE" and "Granskningsarket" in p.source for k, p in props.items() if k != "maturity")
    # redan inläst = inget förslag; annat värde = "ersätter"
    c.set("aisc", props["aisc"])
    c.set("npv_musd", dp(1800, kind="ESTIMATE", source="DFS"))
    again = {k: a for k, _v, a in prefill.proposals(c, PRODUCERS, TIGGRE)}
    assert "aisc" not in again and again["npv_musd"] is True
    assert prefill.proposals(CompanyInput(ticker="ZZZ"), PRODUCERS, TIGGRE) == []
    assert prefill.proposals(c, None, None) == []


# ── lagring ──────────────────────────────────────────────────────────────────
def test_store_roundtrip_and_overrides():
    data = cs.normalize({"companies": "trasigt"})
    assert data == cs.default()
    c = cc.developer_high_quality()
    c.ticker = " dev "
    cs.put(data, c)
    assert list(cs.companies(data)) == ["DEV"] and cs.get(data, "dev").num("npv_musd") == 1800
    cs.set_override(data, "copper", "supply_balance_pct", {"value": 8, "kind": "ESTIMATE", "source": "ICSG"})
    cs.set_override(data, "copper", "adjustments", ["adj_recycling"])
    assert cs.overrides(data)["copper"]["supply_balance_pct"]["value"] == 8
    cs.set_override(data, "copper", "supply_balance_pct", None)
    assert "supply_balance_pct" not in cs.overrides(data)["copper"]
    json.dumps(data)
    cs.remove(data, "DEV")
    assert cs.companies(data) == {}


# ── extraktionsarket ─────────────────────────────────────────────────────────
def test_extraction_sheet_keys_match_the_field_registry():
    from ai import extract_prompt as xp
    for f in xp.FIELDS["confidence"]:
        if f.key == "jurisdiction":
            continue
        spec = cfg.FIELD_BY_KEY[f.key]
        assert (spec.kind in ("number", "int")) == (f.kind == "number") or not f.apply, f.key
        assert (spec.kind == "bool") == (f.kind == "bool"), f.key
    prompt = xp.build_extract_prompt("confidence", "DEV", "Test", "[Sida 1] NPV after tax USD 1,800M")
    assert "npv_stress_price_musd" in prompt and "Confidence score" in prompt
    props = xp.proposals("confidence", {"fields": {"npv_musd": {"value": "1 800", "unit": "MUSD", "page": 1,
                                                                 "quote": "NPV", "confidence": "high"},
                                                    "permits_granted": {"value": "false"}}})
    assert [(p["key"], p["value"]) for p in props] == [("npv_musd", 1800.0), ("permits_granted", False)]


# ── fliken ───────────────────────────────────────────────────────────────────
def test_tab_renders_with_a_company_and_without_network(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import storage

    data = cs.default()
    cs.put(data, cc.developer_high_quality())
    data["commodity_overrides"] = dict(cc.COPPER_OVERRIDES)
    stores = {"confidence": data, "producers": PRODUCERS, "tiggre": TIGGRE, "rotation": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name, default)))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def app():                                        # körs som eget skript: egna importer
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["CONF_TEST_ROOT"])
        from confidence.ui import render_confidence_page
        render_confidence_page()

    monkeypatch.setenv("CONF_TEST_ROOT", root)

    for sub in ("Analys", "Indata", "Råvaror", "Signaler"):
        at = AppTest.from_function(app, default_timeout=30)
        at.session_state["conf_sub"] = sub
        at.session_state["conf_last"] = "DEV"
        at.run()
        assert not at.exception, (sub, at.exception)
        text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        if sub == "Analys":
            assert "BUY CANDIDATE" in text and "Thesis Killer" in text
            assert at.metric[0].value == "86.2"
        elif sub == "Indata":
            assert any("Economics" in e.label for e in at.expander)
