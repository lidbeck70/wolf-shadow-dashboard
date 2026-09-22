"""
Durrett-motorn (SPEC §42): klassificering, FD-aktier, utspädning, börs-
värde/EV, NPV/CAPEX, tillväxt, framtida FCF, upside-multipel, runway,
resurs-/reservvärdering, delpoäng, saknad data, scenarier, red flags,
confidence — och kantfallen (noll skuld, negativ kassa, negativt FCF,
saknad AISC, saknade resurser/reserver, negativt NPV, noll CapEx, noll
produktion, extrem utspädning, valutakonflikt).
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import durrett_cases as dcs
from confidence.data.models import CompanyInput
from confidence.data.provenance import dp
from engines.contract import validate_result
from engines.durrett import config as dc
from engines.durrett import balance_sheet, dilution, growth, upside, valuation
from engines.durrett._base import Ctx, score_from, steps_ge, steps_le
from engines.durrett.engine import analyze, to_engine_result
from engines.durrett.models import to_jsonable

T = date(2026, 9, 22)


def _a(mk, **kw):
    return analyze(mk(), today=T, **kw)


# ── rena beräkningar ─────────────────────────────────────────────────────────
def test_fd_shares_and_dilution_formulas():
    assert dilution.fd_shares(100, 5, 3, 2, 1) == 111 and dilution.fd_shares(100) == 100
    assert dilution.fd_shares(None, 5) is None
    assert dilution.dilution_pct(120, 100) == pytest.approx(20.0) and dilution.dilution_pct(100, 0) is None
    assert growth.growth_multiple(100, 300) == 3.0 and growth.growth_multiple(0, 300) is None
    assert growth.cagr_pct(100, 300, 3) == pytest.approx(44.22, abs=0.01) and growth.cagr_pct(100, 300, 0) is None
    assert valuation.future_operating_cf(200_000, 3000, 1450) == pytest.approx(310.0)
    assert valuation.future_ev(100, 5) == 500 and valuation.future_ev(None, 5) is None
    assert upside.upside_multiple(1000, 100) == 10.0 and upside.upside_multiple(1000, 0) is None
    assert balance_sheet.runway_years(30, 12) == pytest.approx(2.5) and balance_sheet.runway_years(30, 0) is None
    assert balance_sheet.runway_years(-5, 12) == 0.0                                 # negativ kassa → 0 år
    assert steps_ge(None, ((1, 100),)) is None and steps_le(0.5, ((1.0, 100), (2.0, 50))) == 100


def test_score_from_redistributes_weights_and_returns_na_when_too_little_is_known():
    s = score_from("x", "X", {"a": (80, "bra"), "b": (None, "saknas"), "c": (40, "svagt")}, {"a": 50, "b": 30, "c": 20})
    assert s.value == pytest.approx((80 * 50 + 40 * 20) / 70, abs=0.1)
    assert len(s.positive) == 1 and len(s.negative) == 1 and any("saknas" in u for u in s.unknown)
    s2 = score_from("x", "X", {"a": (80, "bra"), "b": (None, "?")}, {"a": 30, "b": 70})
    assert s2.value is None and "för lite känt" in s2.reason
    assert "N/A" in s2.explain()


# ── klassificering ───────────────────────────────────────────────────────────
def test_classification_covers_all_types_with_reasons():
    assert _a(dcs.gold_producer).classification.company_type == dc.PRODUCER
    assert _a(dcs.copper_developer).classification.company_type == dc.DEVELOPER
    assert _a(dcs.lithium_explorer).classification.company_type == dc.EXPLORER
    r = _a(dcs.royalty_company).classification
    assert r.company_type == dc.ROYALTY and any("royalty" in x for x in r.reasons)
    hyb = dcs.gold_producer()
    hyb.set("royalty_revenue_share_pct", dp(45, kind="ACTUAL", source="FS"))
    assert analyze(hyb, today=T).classification.company_type == dc.HYBRID
    unk = CompanyInput(ticker="U", commodity="gold", stage="producer", maturity="pea")
    c = analyze(unk, today=T).classification
    assert c.company_type == dc.UNKNOWN and c.confidence == "low" and c.reasons


# ── aktiestruktur, börsvärde, EV ─────────────────────────────────────────────
def test_share_structure_market_cap_and_ev():
    ctx = Ctx(dcs.gold_producer())
    ss = dilution.share_structure(ctx)
    assert ss["fd_m"] == 260 and ss["basic_m"] == 250 and ss["dilution_3y"] == pytest.approx(250 / 230 * 100 - 100)
    assert ss["mcap_usd"] == 1500 and ss["fd_mcap_usd"] == pytest.approx(6.0 * 260)
    assert ss["net_debt_usd"] == -80 and ss["ev_usd"] == 1420 and ss["fd_ev_usd"] == pytest.approx(1560 - 80)
    assert ss["serial_diluter"] is False
    ctx = Ctx(dcs.lithium_explorer())                          # CAD → USD via fx_to_usd
    ss = dilution.share_structure(ctx)
    assert ss["mcap_usd"] == pytest.approx(60 * 0.73) and ss["price_usd"] == pytest.approx(0.6 * 0.73)


def test_serial_diluter_is_flagged_and_caps_dilution_score():
    a = _a(dcs.serial_diluter)
    assert a.dilution_score.value <= 25 and any("SERIAL DILUTER" in n for n in a.dilution_score.negative)
    f = [x for x in a.red_flags if x.flag == "Serial diluter"]
    assert f and f[0].severity == "CRITICAL" and f[0].source and f[0].date
    assert a.metrics["dilution_3y"]["value"] == pytest.approx(150.0)


# ── properties / valuation / upside ──────────────────────────────────────────
def test_resources_and_reserves_are_kept_apart_with_categories():
    a = _a(dcs.gold_producer)
    assert a.metrics["reserve_total"]["value"] == 3_000_000 and "proven+probable" in a.metrics["reserve_total"]["note"]
    assert a.metrics["resource_total"]["value"] == 3_200_000 and "inferred" in a.metrics["resource_total"]["note"]
    assert a.metrics["mcap_per_reserve_unit"]["value"] == pytest.approx(500.0)
    assert a.metrics["ev_per_resource_unit"]["value"] == pytest.approx(1420e6 / 3_200_000)
    assert a.metrics["ev_per_resource_unit"]["unit"] == "USD/oz"
    d = _a(dcs.copper_developer)
    assert d.metrics["npv_capex"]["value"] == pytest.approx(2.25)
    assert d.metrics["ev_npv"]["value"] == pytest.approx((700 - 250) / 1800)


def test_future_fcf_multiple_and_upside_are_traceable():
    a = _a(dcs.gold_producer)
    fe = a.metrics
    assert fe["future_operating_cf"]["value"] == pytest.approx(300_000 * (3000 - 1450) / 1e6)
    assert fe["future_fcf"]["value"] == pytest.approx(465 * 0.73)
    assert fe["future_ev"]["value"] == pytest.approx(465 * 0.73 * 5) and "config default" in fe["future_ev"]["note"]
    assert a.upside_multiple == pytest.approx((465 * 0.73 * 5 + 80) / 1560)
    assert a.metrics["mcap_future_earnings"]["value"] == pytest.approx(1500 / (465 * 0.73))
    assert a.base_case.upside_pct == pytest.approx((a.upside_multiple - 1) * 100, abs=0.5)  # Base = samma kedja
    assert a.base_case.fair_value_per_share == pytest.approx(a.base_case.future_mcap_musd / 260)


def test_valuation_multiple_is_an_input_not_hardcoded():
    c = dcs.gold_producer()
    c.set("target_ev_ebitda", dp(10, kind="ASSUMPTION", source="stort lågriskbolag"))
    a = analyze(c, today=T)
    assert "fält" in a.metrics["future_ev"]["note"] and a.metrics["future_ev"]["value"] == pytest.approx(465 * 0.73 * 10)
    cfg = json.loads(json.dumps(dc.DURRETT_CONFIG))
    cfg["valuation_multiples"]["default"] = 8.0
    b = analyze(dcs.gold_producer(), config=cfg, today=T)
    assert b.metrics["future_ev"]["value"] == pytest.approx(465 * 0.73 * 8)


# ── kassa / runway / finansiering ────────────────────────────────────────────
def test_runway_and_funding_cliff():
    d = _a(dcs.copper_developer)
    assert d.metrics["cash_runway_years"]["value"] == pytest.approx(250 / 48)
    assert "funding_cliff_gap" not in d.metrics                          # 60 < 250 + 400
    h = _a(dcs.developer_huge_capex_small_company)
    assert h.metrics["funding_cliff_gap"]["value"] == pytest.approx(50.0)
    assert h.financing_score.value < 35 and h.risk_level == "HIGH"
    flags = {f.flag: f.severity for f in h.red_flags}
    assert flags["Huge CAPEX"] == "HIGH" and flags["Financing cliff"] == "HIGH"
    assert flags["Financing required before construction"] == "HIGH" and flags["Low NPV/CAPEX"] == "MEDIUM"


# ── typ-specifikt ────────────────────────────────────────────────────────────
def test_developer_fit_checklist_is_0_to_6_and_never_buy():
    d = _a(dcs.copper_developer)
    ck = d.developer_checklist
    assert ck["total"] == 6 and ck["label"].startswith("DURRETT DEVELOPER FIT") and "BUY" not in ck["label"]
    names = {n: ok for n, ok, _w in ck["items"]}
    assert names["Strong Project"] is True and names["Good Location"] is True and names["Strong Insiders"] is True
    assert names["Path to Production"] is True and names["High Upside Potential"] is False       # 1,97× < 3
    assert ck["passed"] == 5
    assert _a(dcs.gold_producer).developer_checklist is None


def test_explorer_profile_positions_on_lassonde_and_picks_play():
    e = _a(dcs.lithium_explorer)
    p = e.explorer_profile
    assert p["play"] == "DISCOVERY PLAY" and p["lassonde"] == "discovery" and p["discovery_score"] > 60
    assert p["optionality_score"] is None and "TODO" in p["optionality"]["reason"]   # ingen trappa för USD/t LCE
    assert e.upside_score.value == 80                                                # discovery option 4/5
    assert e.cost_score.value is None and e.properties_score.value is None
    g = dcs.lithium_explorer()
    g.commodity = "gold"
    g.set("implied_value_per_unit", dp(30, kind="ASSUMPTION", source="peer-affärer 2026"))
    g.set("resource_unit", dp("oz"))
    p2 = analyze(g, today=T).explorer_profile
    assert p2["optionality_score"] is not None and "implicit värde" in p2["optionality"]["reason"]


def test_royalty_is_not_forced_through_mining_economics():
    r = _a(dcs.royalty_company)
    assert r.cost_score.value is None and "royalty" in r.cost_score.reason
    assert r.quality_score.value is not None and "costs" not in r.quality_score.components
    assert r.classification.profile == "royalty"


# ── scenarier ────────────────────────────────────────────────────────────────
def test_scenarios_use_visible_price_table_and_overrides():
    a = _a(dcs.gold_producer)
    bear, base, bull = a.bear_case, a.base_case, a.bull_case
    assert (bear.commodity_price, base.commodity_price, bull.commodity_price) == (2500, 3000, 3500)
    assert any("pristabell" in x.source for x in base.assumptions)
    assert bear.upside_pct < base.upside_pct < bull.upside_pct
    assert bear.multiple == 4 and bull.multiple == 7 and bear.aisc == pytest.approx(1450 * 1.10)
    o = analyze(dcs.gold_producer(), today=T, scenario_overrides={"base": {"commodity_price": 2800, "multiple": 6}})
    assert o.base_case.commodity_price == 2800 and o.base_case.multiple == 6
    assert any(x.source == "override" for x in o.base_case.assumptions)
    # råvara utan pristabell: spot ± % och det sägs
    e = _a(dcs.lithium_explorer)
    assert e.base_case.reason                                                           # ingen produktion → N/A
    n = dcs.copper_developer()
    n.commodity = "nickel"
    n.set("commodity_price", dp(8.0, kind="ACTUAL", source="LME"))
    nb = analyze(n, today=T).bear_case
    assert "ingen pristabell" in [x for x in nb.assumptions if x.name == "Råvarupris"][0].source


def test_developer_scenarios_subtract_unfunded_capex():
    d = _a(dcs.copper_developer)
    base = d.base_case
    assert base.available and any("ofinansierad CapEx" in s for s in base.steps)
    unfunded = max(0.0, 800 - 250 - 400)
    assert base.future_mcap_musd == pytest.approx(base.future_ev_musd - (0 - 250) - unfunded)
    assert d.bear_case.capex_musd == pytest.approx(960)


# ── saknad data, N/A, kantfall ───────────────────────────────────────────────
def test_missing_data_gives_na_not_zero():
    m = _a(dcs.missing_everything)
    assert m.quality_score.value is None and m.risk_score.value is None and m.risk_level == "UNKNOWN"
    assert all(s.value is None for s in m.scores() if s.key != "red_flags")
    assert m.red_flag_score.value == 100 and m.red_flags == []
    assert m.upside_multiple is None and m.base_case.reason and m.missing_data
    assert any("N/A" in line for line in m.log)
    r = to_engine_result(m)
    assert validate_result(r.as_dict()) == [] and not r.is_complete()


def test_edge_cases_do_not_fail_silently():
    c = dcs.gold_producer()
    c.set("debt_musd", dp(0, kind="ACTUAL", source="FS"))                       # noll skuld
    c.set("cash_musd", dp(-5, kind="ACTUAL", source="FS", note="övertrasserad"))  # negativ kassa (tillåts ej → fel → N/A)
    c.set("free_cash_flow_musd", dp(-30, kind="ACTUAL", source="FS"))
    c.fields.pop("aisc"); c.fields.pop("cash_cost")
    a = analyze(c, today=T)
    assert a.cost_score.components["margin"][0] is None and "aisc" in a.missing_data
    assert a.base_case.reason == "AISC/break-even saknas" and a.upside_multiple is None
    assert a.metrics.get("net_debt_usd", {}).get("value") == 0.0                    # −5 avvisas, 0 skuld
    d = dcs.copper_developer()
    d.set("npv_musd", dp(-100, kind="ESTIMATE", source="PEA"))
    d.set("capex_musd", dp(0, kind="ESTIMATE", source="PEA"))
    b = analyze(d, today=T)
    assert b.metrics.get("npv_capex") is None and b.valuation_score.components["ev_npv"][0] == 5.0
    assert b.properties_score.components["npv_capex"][0] is None
    p = dcs.gold_producer()
    p.set("production_current", dp(0, kind="ACTUAL", source="FS"))
    z = analyze(p, today=T)
    assert z.growth_score.components["production"][0] is None                     # 0 → multipel odefinierad
    for k in ("reserve_proven", "reserve_probable", "resource_measured", "resource_indicated", "resource_inferred"):
        p.fields.pop(k)
    z = analyze(p, today=T)
    assert "reserve_total" not in z.metrics and z.valuation_score.components["resource_value"][0] is None
    assert "reserve_proven" in z.missing_data


def test_currency_mismatch_gives_na_with_reason():
    a = _a(dcs.currency_mismatch)
    assert a.market_cap_musd is None and "fx_to_usd" in a.missing_data
    assert a.upside_multiple is None and a.base_case.reason
    assert any("växelkurs" in line for line in a.log)


# ── red flags, confidence, kontrakt ──────────────────────────────────────────
def test_red_flags_have_all_fields_and_lower_the_flag_score():
    a = _a(dcs.leveraged_producer)
    flags = {f.flag: f for f in a.red_flags}
    assert flags["High AISC"].severity == "HIGH" and flags["Weak balance sheet"].severity == "HIGH"
    assert flags["Serial diluter"].severity == "CRITICAL"
    for f in a.red_flags:
        assert f.reason and f.severity in dc.SEVERITY_RANK
        assert set(f.as_dict()) == {"flag", "severity", "reason", "source", "data", "date"}
    assert a.red_flag_score.value == pytest.approx(100 - 30 - 15 - 15 - 3)
    assert a.risk_score.value < _a(dcs.gold_producer).risk_score.value


def test_confidence_is_integrated_and_separate_from_attractiveness():
    good, weak = _a(dcs.copper_developer), _a(dcs.lithium_explorer)
    assert good.data_confidence > weak.data_confidence
    assert good.confidence_score.value < good.quality_score.value                  # hög kvalitet, måttlig säkerhet
    assert any("Data Confidence" in p for p in good.confidence_score.positive)
    assert any("robusthet" in p for p in good.confidence_score.positive + good.confidence_score.negative)
    assert weak.confidence_score.value < 20 and any("N/A" in n for n in weak.confidence_score.negative)
    assert good.quality_score.components["momentum"][1] == 0 if "momentum" in good.quality_score.components else True


def test_engine_result_matches_the_contract_and_is_json():
    for name, mk in dcs.ALL.items():
        r = to_engine_result(_a(mk))
        assert validate_result(r.as_dict()) == [], name
        json.dumps(r.as_dict())
        assert r.engine == "durrett" and "classification" in r.extras and "thesis" in r.extras
    a = _a(dcs.gold_producer)
    th = a.thesis
    assert set(th) >= {"why_it_could_work", "why_it_could_fail", "key_catalysts", "key_risks", "what_must_go_right",
                       "what_would_invalidate", "next_milestones"}
    assert "BUY" not in json.dumps(th) and a.catalysts[0].expected == "2026-Q4"      # sorterad på datum
    assert isinstance(to_jsonable(a), dict)


def test_explain_score_lists_drivers():
    a = _a(dcs.gold_producer)
    txt = a.explain_score("management")
    assert txt.startswith("Management: ") and "+ track_record" in txt and "Positive:" in txt
    assert "okänd poäng" in a.explain_score("nope")


def test_config_weights_are_used_and_momentum_is_outside_quality():
    assert sum(dc.DURRETT_CONFIG["weights"].values()) == 100
    a = _a(dcs.gold_producer)
    assert "momentum" not in a.quality_score.components and a.momentum_score.value is not None
    cfg = json.loads(json.dumps(dc.DURRETT_CONFIG))
    cfg["weights"] = {"valuation": 100, **{k: 0 for k in cfg["weights"] if k != "valuation"}}
    b = analyze(dcs.gold_producer(), config=cfg, today=T)
    assert b.quality_score.value == b.valuation_score.value


# ── momentum ur kurshistorik ─────────────────────────────────────────────────
def test_momentum_from_history_is_pure_and_honest_about_short_history():
    from engines.durrett import momentum_fetch as mf
    closes = [100 + i * 0.5 for i in range(300)]                  # stigande trend
    vols = [1000] * 189 + [1500] * 63 + [1500] * 48
    m = mf.momentum_from_history(closes, vols)
    last, ma = closes[-1], sum(closes[-200:]) / 200
    assert m["price_vs_ma200_pct"] == pytest.approx((last / ma - 1) * 100)
    assert m["momentum_6m_pct"] == pytest.approx((last / closes[-127] - 1) * 100)
    assert m["volume_trend"] == 2 and m["days"] == 300
    short = mf.momentum_from_history(closes[:100])
    assert short["price_vs_ma200_pct"] is None and short["momentum_6m_pct"] is None and short["volume_trend"] is None
    assert mf.momentum_from_history([]) == {"price_vs_ma200_pct": None, "momentum_6m_pct": None,
                                            "volume_trend": None, "days": 0}
    pts = mf.as_datapoints(m, "GPR", date(2026, 9, 22))
    assert set(pts) == {"price_vs_ma200_pct", "momentum_6m_pct", "volume_trend"}
    assert pts["price_vs_ma200_pct"].kind == "ACTUAL" and pts["volume_trend"].kind == "MODELLED"
    assert pts["momentum_6m_pct"].pub_date == "2026-09-22" and "yfinance GPR" in pts["momentum_6m_pct"].source
    c = dcs.gold_producer()
    for k in ("price_vs_ma200_pct", "momentum_6m_pct", "volume_trend", "rs_rank", "sector_momentum",
              "commodity_momentum", "news_flow"):
        c.fields.pop(k, None)
    for k, p in pts.items():
        c.set(k, p)
    a = analyze(c, today=T)
    assert a.momentum_score.value is not None and "price_trend" in a.momentum_score.components
    assert "momentum" not in a.quality_score.components                # fortfarande utanför Quality


# ── fliken ───────────────────────────────────────────────────────────────────
def test_durrett_tab_renders_all_subtabs_without_network(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import storage
    from confidence import store as cs

    data = cs.default()
    for mk in (dcs.gold_producer, dcs.copper_developer, dcs.lithium_explorer):
        cs.put(data, mk())
    stores = {"confidence": data, "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name, default)))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv("DURRETT_TEST_ROOT", root)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["DURRETT_TEST_ROOT"])
        from engines.durrett.ui import render_durrett_page
        render_durrett_page()

    for sub in ("Analys", "Scenarier", "Indata", "Katalysatorer", "Peers"):
        at = AppTest.from_function(app, default_timeout=60)
        at.session_state["durrett_sub"] = sub
        at.session_state["durrett_last"] = "GPR"
        at.run()
        assert not at.exception, (sub, at.exception)
        if sub == "Analys":
            text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption) + \
                " ".join(c.value for c in at.code)
            assert "DURRETT ANALYSIS" in text and "Properties" in text and "Red flags" in text
            assert at.metric[3].label == "DURRETT SCORE" and at.metric[3].value != "N/A"
            assert "BUY" not in text.replace("BUY CANDIDATE", "")
