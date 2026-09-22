"""
Scenarier — kedjan pris → produktion → EBITDA → FCF → EV/NAV → equity →
aktie, alla antaganden synliga; asymmetri; 5×/10×-vägar; time-to-money.
"""
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence_cases as cc
from confidence import config as cfg
from confidence.data.models import to_jsonable
from confidence.data.provenance import dp
from confidence.scenarios import remaining_stages, scenario_set, time_to_money

TODAY = date(2026, 9, 22)


def test_scenario_config_is_consistent():
    assert [k for k, _l, _p, _c in cfg.SCENARIOS] == ["bear", "base", "bull", "super_bull"]
    assert sum(cfg.SCENARIO_PROBS.values()) == 100
    assert [s.key for s in remaining_stages("pea")] == ["pea", "pfs", "dfs", "fid", "construction"]
    assert remaining_stages("production") == []


def test_producer_chain_is_arithmetic_and_visible():
    ss = scenario_set(cc.producer_high_quality())
    base = ss.scenario("base")
    # 400 M lb × (4,5 − 2,3) = 880 MUSD EBITDA; EV 5× = 4 400; equity = 4 400 − 200 + 600 = 4 800
    assert base.ebitda_musd == 880 and base.value_musd == 4400 and base.equity_musd == 4800
    assert base.fcf_musd == 660 and base.share_price == 9.6
    assert abs(base.upside_pct - (-4.0)) < 0.01
    bear, bull = ss.scenario("bear"), ss.scenario("bull")
    assert abs(bear.equity_musd - 2100) < 1e-6 and abs(bull.equity_musd - 7500) < 1e-6
    assert any("AISC" in s for s in base.steps) and any("skatt 25 %" in s for s in base.steps)
    # antaganden ur config syns som ASSUMPTION
    names = {a.name: a for a in ss.assumptions}
    assert names["EV/EBITDA i scenariot"].kind == "ASSUMPTION" and names["Skattesats"].kind == "ASSUMPTION"
    assert names["AISC / opex per enhet"].kind == "ACTUAL"
    assert ss.missing == [] and all(s.complete for s in ss.scenarios)


def test_producer_multiplier_paths():
    ss = scenario_set(cc.producer_high_quality())
    p5 = ss.paths[0]
    # EV 5×: 25 000 + 200 − 600 = 24 600 → EBITDA 4 920 → pris 2,3 + 12,3 = 14,6
    assert p5.target == 5 and abs(p5.required_price - 14.6) < 1e-6
    assert abs(p5.price_change_pct - (14.6 / 4.5 - 1) * 100) < 1e-6
    assert abs(p5.production_factor - 4920 / 2.2 / 400) < 1e-6
    assert ss.paths[1].required_price > p5.required_price
    assert all("EV/EBITDA 5" in p.steps[0] for p in ss.paths)


def test_developer_nav_moves_with_fs_sensitivity_and_capex_stress():
    ss = scenario_set(cc.developer_high_quality())
    base, bear, bull = ss.scenario("base"), ss.scenario("bear"), ss.scenario("bull")
    # lutning (1800 − 900)/20 = 45 MUSD per %; Bear −30 % → 450, −20 % CapEx 160 → 290
    assert base.value_musd == 1800 and abs(bear.value_musd - 290) < 1e-6 and abs(bull.value_musd - 3150) < 1e-6
    assert abs(base.equity_musd - (1800 * 0.7 + 250)) < 1e-6
    assert any("FS-känsligheten" in s for s in base.steps)
    assert any("linjär extrapolation" in s for s in bull.steps) and not any("extrapolation" in s for s in base.steps)
    assert any("CapEx +20 %" in s for s in bear.steps) and not any("CapEx" in s for s in bull.steps)
    a = ss.asymmetry
    assert a.band == "STARK ASYMMETRI" and a.ratio > 3 and a.downside_pct < 0 < a.upside_pct
    assert abs(a.expected_pct - sum(cfg.SCENARIO_PROBS[s.key] / 100 * s.upside_pct for s in ss.scenarios)) < 1e-6


def test_developer_paths_show_rerating_and_price():
    ss = scenario_set(cc.developer_high_quality())
    p5, p10 = ss.paths
    assert abs(p5.required_p_nav - 3500 / 1800) < 1e-6 and abs(p10.required_p_nav - 7000 / 1800) < 1e-6
    # NAV_req = (3500 − 250)/0,7 = 4 643 → Δ % = (4643 − 1800)/45
    assert abs(p5.price_change_pct - (3250 / 0.7 - 1800) / 45) < 1e-6
    assert p5.required_price > 4.5 and any("omvärdering" in s for s in p5.steps)


def test_annuity_fallback_when_fs_sensitivity_is_missing():
    ss = scenario_set(cc.developer_capex_stressed_modelled())
    base = ss.scenario("base")
    assert any("annuitet" in s for s in base.steps)
    assert any("Diskonteringsränta" in a.name and a.kind == "ESTIMATE" for a in ss.assumptions)   # ur DFS
    ss2 = scenario_set(cc.missing_everything())
    assert any(a.kind == "ASSUMPTION" and a.source == "confidence.config" for a in ss2.assumptions)
    r, life, tax = 0.08, 18, 0.25
    slope = 4.5 / 100 * 180e6 / 1e6 * (1 - tax) * (1 - (1 + r) ** -life) / r
    assert abs(ss.scenario("bull").value_musd - (150 + slope * 30)) < 1e-6


def test_explorer_and_missing_data_get_no_numbers():
    ss = scenario_set(cc.explorer_early())
    assert ss.scenarios == [] and ss.asymmetry is None and "npv_musd" in ss.missing
    ss = scenario_set(cc.missing_everything())
    assert not any(s.complete for s in ss.scenarios) and ss.asymmetry is None
    assert all(p.required_price is None for p in ss.paths)
    assert any("DATA_MISSING" in s for sc in ss.scenarios for s in sc.steps)
    assert "ofullständiga scenarier — se DATA_MISSING i stegen" in ss.flags


def test_royalty_uses_fixed_costs_from_margin():
    ss = scenario_set(cc.royalty())
    base, bull = ss.scenario("base"), ss.scenario("bull")
    # intäkt 200 000 × 2 500 = 500 MUSD; fasta 110; EBITDA 390
    assert abs(base.revenue_musd - 500) < 1e-6 and abs(base.ebitda_musd - 390) < 1e-6
    assert abs(bull.ebitda_musd - (650 - 110)) < 1e-6
    assert any("fasta kostnader" in s for s in base.steps)


def test_negative_bear_equity_is_floored_and_jsonable():
    c = cc.developer_cheap_but_poor()
    ss = scenario_set(c)
    bear = ss.scenario("bear")
    assert bear.value_musd < 0 and bear.equity_musd == 5.0                  # max(NAV, 0) + kassa
    assert isinstance(to_jsonable(ss), dict)


def test_time_to_money_stage_table_vs_company_plan():
    ttm = time_to_money(cc.developer_high_quality(), TODAY)
    assert [s.key for s in ttm.stages] == ["dfs", "fid", "construction"] and ttm.typical_years == 4.5
    assert ttm.years == 3 and ttm.basis == "bolagets plan" and ttm.confidence == "medel" and ttm.flags == []
    c = cc.developer_high_quality()
    c.set("first_cashflow_year", dp(2027, kind="GUIDANCE", source="Presentation"))
    ttm = time_to_money(c, TODAY)
    assert ttm.years == 1 and ttm.confidence == "låg" and any("aggressiv" in f for f in ttm.flags)
    ttm = time_to_money(cc.explorer_early(), TODAY)
    assert ttm.basis.startswith("MODELLED") and ttm.years == 9.5 and ttm.confidence == "låg"
    ttm = time_to_money(cc.producer_high_quality(), TODAY)
    assert ttm.years == 0 and ttm.confidence == "hög" and ttm.stages == []
    ttm = time_to_money(cc.developer_cheap_but_poor(), TODAY)
    assert ttm.confidence == "låg" and len(ttm.flags) == 2


def test_no_scenario_number_is_nan_or_inf():
    for name, mk in cc.ALL.items():
        ss = scenario_set(mk())
        for s in ss.scenarios:
            for v in (s.price, s.revenue_musd, s.ebitda_musd, s.fcf_musd, s.value_musd, s.equity_musd, s.upside_pct):
                assert v is None or math.isfinite(v), (name, s.key)
