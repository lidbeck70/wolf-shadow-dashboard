"""
Case Score — specens fall: högkvalitativ producent/developer, tidig explorer,
belånad producent, billigt men dåligt projekt, utmärkt men övervärderat,
saknad data, pris- och capex-stress, royalty. Plus spec-tabellernas gränser
och att varje poäng går att förklara.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence_cases as cc
from confidence import config as cfg
from confidence.data.models import CompanyInput, to_jsonable
from confidence.data.provenance import dp
from confidence.scoring import case_score, rating_for
from confidence.scoring._steps import step_ge, step_le, table_lt

TODAY = date(2026, 9, 22)


def _cs(mk):
    return case_score(mk(), today=TODAY, commodity_overrides=cc.COPPER_OVERRIDES)


def _pts(cs, key):
    return cs.pillar(key).points


# ── tabellgränser (SPEC) ─────────────────────────────────────────────────────
def test_spec_tables_at_their_boundaries():
    t = cfg.P_NAV_TABLE
    assert [table_lt(v, t, cfg.P_NAV_BEYOND) for v in (0.29, 0.30, 0.49, 0.69, 0.89, 1.09, 1.29, 1.49, 1.50)] \
        == [10, 9, 9, 8, 7, 5, 3, 1, 0]
    t = cfg.ND_EBITDA_TABLE
    assert [table_lt(v, t, cfg.ND_EBITDA_BEYOND) for v in (-0.1, 0.0, 0.49, 0.99, 1.49, 1.99, 2.99, 3.99, 4.0)] \
        == [10, 9, 9, 8, 7, 5, 3, 1, 0]
    t = cfg.TIME_TO_MONEY_TABLE
    assert [table_lt(v, t, cfg.TIME_TO_MONEY_BEYOND) for v in (0, 1.9, 2, 2.9, 4.9, 6.9, 9.9, 14.9, 15)] \
        == [10, 10, 9, 9, 8, 6, 4, 2, 1]
    t = cfg.SUPPLY_BALANCE_TABLE
    assert [table_lt(v, t, cfg.SUPPLY_BALANCE_MAX_POINTS) for v in (-12, -10, -7, -5, -2, 0, 3, 5, 12, 20, 25, 30, 40)] \
        == [0, 2, 2, 4, 4, 4, 4, 6, 9, 12, 12, 15, 15]
    assert step_ge(40, cfg.AISC_MARGIN_STEPS) == 4 and step_ge(39.9, cfg.AISC_MARGIN_STEPS) == 3
    assert step_le(2.0, cfg.PAYBACK_STEPS) == 2 and step_le(3.5, cfg.PAYBACK_STEPS) == 0
    assert step_ge(None, cfg.IRR_STEPS) == 0 and table_lt(None, t, 15) == 0
    assert [rating_for(v) for v in (100, 90, 89.9, 80, 79.9, 70, 60, 50, 49.9, 0)] == \
        ["ELITE", "ELITE", "PICK", "PICK", "STRONG CANDIDATE", "STRONG CANDIDATE", "WATCHLIST",
         "SPECULATIVE", "PASS", "PASS"]


# ── specens fall ─────────────────────────────────────────────────────────────
def test_high_quality_producer_scores_pick_with_full_economics():
    cs = _cs(cc.producer_high_quality)
    assert cs.rating == "PICK" and 80 <= cs.total < 90
    assert _pts(cs, "economics") == 15 and _pts(cs, "balance_sheet") == 10   # ND/EBITDA < 0 → 10
    assert _pts(cs, "production_growth") == 10                               # kassaflöde nu
    assert not any(p.caps for p in cs.pillars) and cs.missing == []
    assert cs.discovery_option is None


def test_leveraged_producer_loses_balance_sheet_and_gets_stress_cap():
    hq, lev = _cs(cc.producer_high_quality), _cs(cc.producer_leveraged)
    assert _pts(lev, "balance_sheet") == 0                                   # ND/EBITDA 4,5 → 0
    econ = lev.pillar("economics")
    assert econ.points == cfg.ECON_CAP_STRESS_NEGATIVE and ("pris −20 %", 5) in econ.caps
    assert any("MODELLED" in n for n in econ.notes)
    assert lev.total < hq.total - 20 and lev.rating == "WATCHLIST"


def test_high_quality_developer_routes_developer_metrics():
    cs = _cs(cc.developer_high_quality)
    assert cs.rating == "PICK"
    econ = cs.pillar("economics")
    assert econ.points == 15 and set(econ.components) == {"NPV/CapEx", "IRR", "Payback", "Break-even-täckning",
                                                          "Capex-intensitet"}
    assert _pts(cs, "production_growth") == 8                                # 2029 → 3 år → 3–5
    assert _pts(cs, "valuation") == 9                                        # P/NAV 0,39
    bal = cs.pillar("balance_sheet")
    assert bal.points == 9 and bal.components["Finansieringsgap"] == 3 and bal.components["Runway"] == 3
    assert any("stress pris −20 %: NPV 900" in n for n in econ.notes)


def test_excellent_but_overvalued_loses_exactly_valuation():
    hq, ov = _cs(cc.developer_high_quality), _cs(cc.developer_overvalued)
    assert _pts(ov, "valuation") == 0                                        # P/NAV 1,67 > 1,50
    assert hq.total - ov.total == _pts(hq, "valuation")
    assert ov.rating == "STRONG CANDIDATE"


def test_cheap_but_poor_project_is_pass_despite_full_valuation():
    cs = _cs(cc.developer_cheap_but_poor)
    assert _pts(cs, "valuation") == 10 and cs.rating == "PASS" and cs.total < 50
    assert _pts(cs, "economics") == 0 and _pts(cs, "management") == 0
    bal = cs.pillar("balance_sheet")
    assert bal.points == 0 and bal.components["Utspädning (DS)"] == -2      # klampas vid 0


def test_price_stress_caps_economics_only():
    hq, st = _cs(cc.developer_high_quality), _cs(cc.developer_price_stressed)
    econ = st.pillar("economics")
    assert econ.points == cfg.ECON_CAP_STRESS_NEGATIVE
    assert [c[0] for c in econ.caps] == ["pris −20 %", "IRR under stress"]
    for p in hq.pillars:
        if p.key != "economics":
            assert p.points == st.pillar(p.key).points
    assert hq.total - st.total == 15 - cfg.ECON_CAP_STRESS_NEGATIVE


def test_capex_stress_is_modelled_when_fs_sensitivity_is_missing():
    cs = _cs(cc.developer_capex_stressed_modelled)
    econ = cs.pillar("economics")
    assert ("capex +20 %", cfg.ECON_CAP_STRESS_NEGATIVE) in econ.caps
    assert any("MODELLED" in n and "capex" in n for n in econ.notes)
    assert "npv_stress_capex_musd" in cs.missing and "npv_stress_price_musd" in cs.missing


def test_early_explorer_keeps_discovery_option_outside_the_score():
    cs = _cs(cc.explorer_early)
    assert cs.discovery_option == 4.0 and cs.total < 50 and cs.rating == "PASS"
    assert sum(p.points for p in cs.pillars) == cs.total                    # DO ej inräknad
    bal = cs.pillar("balance_sheet")
    assert any("skalad" in n for n in bal.notes) and bal.points == 4        # 2/5 → 4/10
    assert "Finansieringsgap" not in bal.components
    assert _pts(cs, "demand_scarcity") == 0 and "commodity.supply_balance_pct" in cs.missing


def test_missing_data_gives_zero_not_a_guess():
    cs = _cs(cc.missing_everything)
    for p in cs.pillars:
        if p.key not in ("strategic_commodity", "demand_scarcity"):
            assert p.points == 0, p.key
    assert cs.rating == "PASS" and len(cs.missing) >= 25
    assert any("DATA_MISSING" in f for f in cs.flags)
    assert all("DATA_MISSING" in n for p in cs.pillars if p.key == "economics"
               for n in p.notes if "/" in n)


def test_supply_adjustments_subtract_and_never_go_below_zero():
    base = {"copper": dict(cc.COPPER_OVERRIDES["copper"])}
    base["copper"]["adjustments"] = ["adj_substitution", "adj_recycling", "adj_project_delays"]
    cs = case_score(cc.producer_high_quality(), today=TODAY, commodity_overrides=base)
    sc = cs.pillar("demand_scarcity")
    assert sc.components["Utbudsbalans"] == 9 and sc.components["Justeringar"] == -1 and sc.points == 8
    base["copper"]["supply_balance_pct"] = {"value": -15, "kind": "ESTIMATE", "source": "ICSG"}
    cs = case_score(cc.producer_high_quality(), today=TODAY, commodity_overrides=base)
    assert cs.pillar("demand_scarcity").points == 0


def test_unknown_commodity_zeros_the_commodity_pillars():
    c = CompanyInput(ticker="X", commodity="betting", stage="producer")
    cs = case_score(c, today=TODAY)
    assert _pts(cs, "strategic_commodity") == 0 and _pts(cs, "demand_scarcity") == 0
    assert "commodity" in cs.missing and any("okänd" in f for f in cs.flags)


def test_royalty_skips_aisc_and_scales():
    cs = _cs(cc.royalty)
    econ = cs.pillar("economics")
    assert "AISC-marginal" not in econ.components and "Break-even-täckning" not in econ.components
    assert econ.points == 10 and any("skalad 9 → 15" in n for n in econ.notes)   # 6/9 → 10/15
    assert _pts(cs, "balance_sheet") == 8                                   # ND/EBITDA 0,8
    assert _pts(cs, "valuation") == 0                                       # EV/EBITDA 15, P/NAV 1,29


# ── förklarbarhet + determinism ──────────────────────────────────────────────
def test_every_point_is_explained_and_bounded():
    for name, mk in cc.ALL.items():
        cs = _cs(mk)
        assert abs(sum(p.points for p in cs.pillars) - cs.total) < 0.06, name
        for p in cs.pillars:
            assert 0 <= p.points <= p.max, (name, p.key)
            for comp in p.components:
                assert any(comp in n for n in p.notes), (name, p.key, comp)
            assert p.notes, (name, p.key)


def test_same_input_same_output_and_jsonable():
    a, b = _cs(cc.developer_high_quality), _cs(cc.developer_high_quality)
    assert to_jsonable(a) == to_jsonable(b)
    assert isinstance(to_jsonable(a)["pillars"][0]["components"], dict)


def test_invalid_field_is_ignored_not_scored():
    c = cc.producer_high_quality()
    c.set("aisc", dp("billig", kind="ACTUAL", source="FS"))
    cs = case_score(c, today=TODAY, commodity_overrides=cc.COPPER_OVERRIDES)
    econ = cs.pillar("economics")
    assert econ.components["AISC-marginal"] == 0 and "aisc" in econ.missing
