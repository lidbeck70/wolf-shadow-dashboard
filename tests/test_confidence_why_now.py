"""
Why Now — sex signaler ur repots moduler som DATA in; täckning i stället
för gissning. Regional knapphet — jurisdiktion ur repots tabell,
koncentration/västlig andel ur registret, aldrig skalad upp på tunt underlag.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence_cases as cc
from confidence import commodities as com
from confidence import config as cfg
from confidence.regional import regional_scarcity
from confidence.why_now import Signals, cycle_label_from_percentile, signals_from_sources, why_now

_USGS = {"kind": "ESTIMATE", "source": "USGS MCS 2026", "source_type": "primary", "pub_date": "2026-01-31"}
OVERRIDES = {"copper": {**cc.COPPER_OVERRIDES["copper"],
                        "supply_concentration_pct": {"value": 28, **_USGS}, "top_supplier": "Chile",
                        "western_share_pct": {"value": 45, **_USGS}}}
ROTATION = {"month": "2026-09", "grades": {"koppar": {"hatred": 4, "fundamentals": 5, "catalyst": 3,
                                                       "case_intact": True}}}
THEMES = [{"key": "koppar", "cykel_label": "TIDIG", "percentile_10y": 22.0},
          {"key": "guld", "cykel_label": "TOPP", "percentile_10y": 96.0}]
RATIOS = {"copper_gold": {"status": "RUBBER_BAND_STRETCHED"}, "gold_silver": {"status": "NEUTRAL"}}
COMPLEXES = {"basmetaller": {"verdict": "SELEKTIV"}, "adelmetaller": {"verdict": "AV"}}
E2R = {"copper": ["copper_gold"], "silver": ["gold_silver"], "gold_miner": ["metal_miners"]}


def _cu():
    return com.get("copper", OVERRIDES)


def test_adapter_crosswalks_repo_structures():
    s = signals_from_sources(_cu(), ROTATION, THEMES, RATIOS, COMPLEXES, exposure_to_ratio=E2R,
                             time_to_money_years=3)
    assert s.cycle_label == "TIDIG" and s.cycle_percentile == 22.0
    assert s.rotation_grade["fundamentals"] == 5 and s.rotation_month == "2026-09"
    assert s.ratio_status == "RUBBER_BAND_STRETCHED" and s.ratio_key == "copper_gold"
    assert s.complex_verdict == "SELEKTIV" and s.complex_key == "basmetaller"
    # guld: tema TOPP, komplex AV, ingen ratio för gold_miner i RATIOS, ej graderad i rotationen
    g = signals_from_sources(com.get("gold"), ROTATION, THEMES, RATIOS, COMPLEXES, exposure_to_ratio=E2R)
    assert g.cycle_label == "TOPP" and g.complex_verdict == "AV" and g.ratio_status is None and g.rotation_grade is None
    assert signals_from_sources(None).notes


def test_why_now_full_coverage_is_explained():
    s = signals_from_sources(_cu(), ROTATION, THEMES, RATIOS, COMPLEXES, exposure_to_ratio=E2R, time_to_money_years=3)
    w = why_now(_cu(), s)
    assert w.components == {"Cykelläge": 30, "Triple Signal": 18.75, "Gummiband": 15, "Komplex": 5,
                            "Utbud": 6.0, "Time-to-money": 4}
    assert w.score == 78.8 and w.band == "NU" and w.coverage == 1.0 and w.flags == [] and w.missing == []
    for comp in w.components:
        assert any(n.startswith(comp) for n in w.notes), comp
    assert any("12/15" in n for n in w.notes)


def test_why_now_thin_coverage_is_not_scaled_up():
    w = why_now(_cu(), Signals())                                    # bara utbud (10 av 100)
    assert w.coverage == 0.1 and w.score == 6.0 and w.band == "INTE NU"
    assert any("råpoäng utan uppskalning" in f for f in w.flags)
    w = why_now(None, Signals())
    assert w.score == 0 and w.coverage == 0 and any("saknar alla signaler" in f for f in w.flags)
    # 60 % täckning skalas, men flaggas med vad som saknas
    s = Signals(cycle_label="TIDIG", rotation_grade={"hatred": 5, "fundamentals": 5, "catalyst": 5})
    w = why_now(_cu(), s)
    assert w.coverage == 0.65 and abs(w.score - 61 / 65 * 100) < 0.1
    assert any("täckning 65 %" in f and "ratio_status" in f for f in w.flags)


def test_broken_case_and_top_of_cycle_score_zero():
    s = Signals(cycle_label="TOPP", rotation_grade={"hatred": 5, "fundamentals": 5, "catalyst": 5, "case_intact": False},
                ratio_status="NEUTRAL", complex_verdict="AV", time_to_money_years=8)
    w = why_now(_cu(), s)
    assert w.components["Cykelläge"] == 0 and w.components["Triple Signal"] == 0
    assert w.components["Time-to-money"] == cfg.WHY_NOW_TTM_BEYOND and w.band == "INTE NU"
    assert any("brutet" in n for n in w.notes)


def test_cycle_label_rule_matches_theme_board():
    assert cycle_label_from_percentile(95) == "TOPP" and cycle_label_from_percentile(70) == "SEN"
    assert cycle_label_from_percentile(30) == "TIDIG" and cycle_label_from_percentile(None) is None
    assert cycle_label_from_percentile(50, 1.0) == "MITTEN" and cycle_label_from_percentile(50, -1.0) == "TIDIG"


def test_regional_uses_repo_jurisdiction_table_and_registry():
    r = regional_scarcity(cc.developer_high_quality(), _cu())              # British Columbia 85
    assert r.jurisdiction_score == 85 and r.components["Jurisdiktion"] == 42.5
    assert r.components["Koncentration"] == 5 and r.components["Västligt gap"] == 7
    assert r.score == 54.5 and r.band == "NORMAL" and r.coverage == 1.0 and r.flags == []
    assert any("Chile" in n and "USGS" in n for n in r.notes)


def test_regional_halves_concentration_outside_safe_jurisdictions():
    r = regional_scarcity(cc.developer_cheap_but_poor(), _cu())            # AR/Salta saknas i tabellen
    assert "jurisdiction" in r.missing and r.components["Koncentration"] == 2.5
    assert any("Salta" in n and "finns inte i repots" in n for n in r.notes)
    assert r.coverage == 0.5 and r.score == 9.5 and r.band == "RIKLIG"       # ej uppskalad
    assert any("råpoäng" in f for f in r.flags)


def test_regional_jurisdiction_alone_is_not_scarcity():
    r = regional_scarcity(cc.explorer_early(), com.get("lithium"))           # Ontario 88, inget om litium
    assert r.components == {"Jurisdiktion": 44.0} and r.score == 44.0 and r.band == "NORMAL"
    assert "commodity.supply_concentration_pct" in r.missing and any("råpoäng" in f for f in r.flags)
    r = regional_scarcity(cc.royalty(), None)
    assert r.score == 0 and r.coverage == 0


def test_registry_override_keeps_base_untouched():
    assert com.get("copper").supply_concentration_pct.missing and _cu().top_supplier == "Chile"
    assert com.get("copper").top_supplier == ""
