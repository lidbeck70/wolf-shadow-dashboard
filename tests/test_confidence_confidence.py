"""
Confidence Score — sju delar, kill-caps efter summan, band. Kärnkravet:
hög Case Score + låg Confidence är INTE ett high-confidence-case, och
Confidence ändrar aldrig Case Score.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence_cases as cc
from confidence import config as cfg
from confidence.data.models import to_jsonable
from confidence.data.provenance import dp
from confidence.scoring import band_for, case_score, confidence_score, kill_caps

TODAY = date(2026, 9, 22)


def _cf(mk_or_company):
    c = mk_or_company() if callable(mk_or_company) else mk_or_company
    return confidence_score(c, today=TODAY)


def _pts(cf, key):
    return cf.part(key).points


def test_bands_at_their_boundaries():
    assert [band_for(v) for v in (100, 90, 89.9, 80, 70, 60, 50, 49.9)] == \
        ["VERIFIED", "VERIFIED", "HIGH CONFIDENCE", "HIGH CONFIDENCE", "GOOD CONFIDENCE", "MODERATE",
         "SPECULATIVE", "LOW CONFIDENCE"]


def test_high_quality_producer_is_high_confidence_but_unassessed_parts_stay_zero():
    cf = _cf(cc.producer_high_quality)
    assert cf.band == "HIGH CONFIDENCE" and cf.caps_applied == []
    dq = cf.part("data_quality")
    assert dq.components["Oberoende verifiering"] == 0 and dq.components["Källkonsistens"] == 0
    assert "independent_verification" in cf.missing and "cross_source_consistency" in cf.missing
    assert dq.components["Källkvalitet"] > 4 and 3 <= dq.components["Färskhet"] <= 4
    assert _pts(cf, "project_maturity") == 15 and _pts(cf, "financing_certainty") == 10
    assert _pts(cf, "timeline_certainty") == 10 and _pts(cf, "management_track_record") == 10
    assert any("TODO" in f and "record_price_dependent" in f for f in cf.flags)


def test_developer_maturity_interpolates_milestones():
    cf = _cf(cc.developer_high_quality)
    pm = cf.part("project_maturity")
    assert abs(pm.points - (9 + 6 / 7 * 3)) < 0.01 and "6/7 milstolpar" in pm.notes[0]
    assert _pts(cf, "financing_certainty") == 8                       # 50 % åtagen (5) + runway (3)
    ec = cf.part("economic_certainty")
    assert ec.components["Prisstress"] == 3 and ec.components["Studiekvalitet"] == 4    # NPV 900/1800 × 6
    assert cf.band == "GOOD CONFIDENCE"


def test_high_case_score_with_low_confidence_is_not_a_high_confidence_case():
    good, weak = cc.developer_high_quality(), cc.developer_excellent_low_confidence()
    cs_good = case_score(good, today=TODAY, commodity_overrides=cc.COPPER_OVERRIDES)
    cs_weak = case_score(weak, today=TODAY, commodity_overrides=cc.COPPER_OVERRIDES)
    cf_good, cf_weak = _cf(good), _cf(weak)
    # Case: samma tal → samma pelare (utom finansieringen som faktiskt skiljer)
    for p in cs_good.pillars:
        if p.key != "balance_sheet":
            assert p.points == cs_weak.pillar(p.key).points, p.key
    assert cs_weak.rating == "PICK" and cs_weak.total > 80
    # Confidence: ASSUMPTION utan datum, ingen oberoende resurs, ingen finansiering
    assert cf_weak.band == "LOW CONFIDENCE" and cf_weak.total < 50 < cf_good.total
    assert cf_weak.part("data_quality").components["Färskhet"] == 0
    assert [k for k, _l, _t in cf_weak.caps_applied] == ["no_independent_resource", "no_financing_plan"]
    assert any(f.startswith("KILL:") for f in cf_weak.flags)


def test_kill_caps_apply_after_the_sum():
    c = cc.developer_high_quality()
    c.set("independent_resource_estimate", dp(False, kind="ACTUAL", source="43-101 saknas", source_type="primary"))
    cf = _cf(c)
    assert cf.raw_total > 50 and cf.total == 50 and cf.band == "SPECULATIVE"
    assert kill_caps(c) == [("no_independent_resource", 50, "Ingen oberoende resursuppskattning")]
    c.set("capital_destruction_history", dp(True, kind="ACTUAL", source="Historik", source_type="primary"))
    cf = _cf(c)
    assert cf.total == 50 and [k for k, _l, _t in cf.caps_applied] == ["no_independent_resource",
                                                                       "capital_destruction_history"]
    # rekordprisberoende: tak 65 OCH Economic Certainty × 0,3
    c2 = cc.developer_high_quality()
    c2.set("record_price_dependent", dp(True, kind="ACTUAL", source="FS-pris 6 USD", source_type="independent"))
    cf2 = _cf(c2)
    base = _cf(cc.developer_high_quality)
    assert cf2.total <= 65 and abs(_pts(cf2, "economic_certainty") - _pts(base, "economic_certainty") * 0.3) < 0.05
    assert not any("record_price_dependent" in f and "TODO" in f for f in cf2.flags)


def test_confidence_never_changes_case_score():
    c = cc.developer_high_quality()
    before = to_jsonable(case_score(c, today=TODAY, commodity_overrides=cc.COPPER_OVERRIDES))
    confidence_score(c, today=TODAY)
    c.set("independent_resource_estimate", dp(False, kind="ACTUAL", source="x", source_type="primary"))
    confidence_score(c, today=TODAY)
    after = to_jsonable(case_score(c, today=TODAY, commodity_overrides=cc.COPPER_OVERRIDES))
    assert before == after


def test_missing_data_lowers_confidence_instead_of_guessing():
    cf = _cf(cc.missing_everything)
    assert cf.band == "LOW CONFIDENCE" and cf.total < 10 and len(cf.missing) >= 15
    assert _pts(cf, "data_quality") == 0
    assert _pts(cf, "project_maturity") == cfg.MATURITY_BASE["pea"][0]      # bara grundnivån
    assert any("saknas" in f for f in cf.flags)


def test_early_explorer_is_capped_and_scaled():
    cf = _cf(cc.explorer_early)
    assert cf.band == "LOW CONFIDENCE"
    assert [k for k, _l, _t in cf.caps_applied] == ["no_independent_resource"]
    fin = cf.part("financing_certainty")
    assert "Åtagen andel av CapEx" not in fin.components and any("skalad" in n for n in fin.notes)
    assert abs(fin.points - 2 / 3 * 10) < 0.05
    assert _pts(cf, "resource_certainty") < 1                            # exploration_target × 0,1


def test_modelled_stress_is_capped_below_fs_sensitivity():
    cf = _cf(cc.developer_capex_stressed_modelled)
    ec = cf.part("economic_certainty")
    assert ec.components["Prisstress"] <= cfg.ECON_CERTAINTY_MODELLED_CAP["price_stress"]
    assert ec.components["Capexstress"] == 0                              # NPV 150 − 180 < 0
    assert sum("MODELLED" in n for n in ec.notes) == 2


def test_npv_price_above_spot_is_penalised():
    c = cc.developer_high_quality()
    c.set("npv_price_assumption", dp(5.5, kind="ESTIMATE", source="DFS", source_type="independent"))
    cf = _cf(c)
    ec = cf.part("economic_certainty")
    assert ec.components["NPV-pris över spot"] == -cfg.NPV_PRICE_ABOVE_SPOT_PENALTY
    assert _pts(cf, "economic_certainty") == _pts(_cf(cc.developer_high_quality), "economic_certainty") - 3


def test_freshness_decays_with_today():
    now, later = _cf(cc.producer_high_quality), confidence_score(cc.producer_high_quality(), today=date(2029, 1, 1))
    assert later.part("data_quality").components["Färskhet"] < now.part("data_quality").components["Färskhet"]
    assert later.total < now.total


def test_every_part_is_explained_bounded_and_sums():
    for name, mk in cc.ALL.items():
        cf = _cf(mk)
        assert abs(sum(p.points for p in cf.parts) - cf.raw_total) < 0.06, name
        assert cf.total <= cf.raw_total
        for p in cf.parts:
            assert 0 <= p.points <= p.max, (name, p.key)
            for comp in p.components:
                assert any(comp in n for n in p.notes), (name, p.key, comp)
