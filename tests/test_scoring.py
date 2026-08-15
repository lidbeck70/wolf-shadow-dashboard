"""
Tests for scoring.py — Poängmodellen (Sprott + Durrett).

Asserted against poangmodell_sprott_durrett.xlsx:

  Sprott  F: =IF(OR(D="",E="",E=0),"",D/E)          runway
          L: =IF(COUNT(G:K)=0,"",SUM(G:K))          totalpoäng
          M: >=8 Kärninnehav, >=6 Bevakningslista, annars Passa
  Durrett F: =IF(OR(D="",E="",E=0),"",D/E)          MCap/uns
          H: =IF(OR(D="",G="",G=0),"",D/G)          MCap/framtida vinst
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scoring as sc


# ── Faktorerna ───────────────────────────────────────────────────────────────
def test_five_factors_with_all_three_levels_described():
    assert len(sc.FACTORS) == 5
    assert [f.key for f in sc.FACTORS] == ["balans", "vardering", "tillvaxt",
                                           "agare", "havstang"]
    for f in sc.FACTORS:
        assert f.two and f.one and f.zero, f.key


def test_total_score_sums_and_clamps():
    assert sc.total_score({"balans": 2, "vardering": 2, "tillvaxt": 2,
                           "agare": 2, "havstang": 2}) == sc.MAX_SCORE
    assert sc.total_score({"balans": 2, "vardering": 1}) == 3
    # utanför 0–2 klipps
    assert sc.total_score({"balans": 9, "vardering": -4}) == 2


def test_an_unscored_row_has_no_verdict_rather_than_zero():
    """The sheet leaves it blank: no score is not the same as a bad score."""
    assert sc.total_score({}) is None
    assert sc.total_score({"balans": ""}) is None
    assert sc.verdict(None) is None
    # men en rad som faktiskt fått nollor har ett betyg
    assert sc.total_score({"balans": 0}) == 0
    assert sc.verdict(0) == sc.PASS


def test_verdict_thresholds_match_the_sheet():
    assert sc.verdict(10) == sc.CORE
    assert sc.verdict(8) == sc.CORE
    assert sc.verdict(7) == sc.WATCH
    assert sc.verdict(6) == sc.WATCH
    assert sc.verdict(5) == sc.PASS
    assert sc.verdict(0) == sc.PASS
    assert (sc.CORE_MIN, sc.WATCH_MIN) == (8, 6)


def test_thresholds_are_the_same_ones_the_migration_spec_reuses_everywhere():
    """Spec: "grön >= 8, gul 6-7, röd < 6 — samma trösklar överallt"."""
    import tiggre
    assert sc.CORE_MIN == tiggre.SCORE_MIN == 8


# ── Sprott: runwayen ─────────────────────────────────────────────────────────
def test_runway_years():
    assert sc.runway_years(30, 10) == 3.0
    assert sc.runway_years(5, 10) == 0.5


def test_runway_is_blank_when_burn_is_missing_or_zero():
    """A zero burn must not divide — the sheet guards E=0 explicitly."""
    assert sc.runway_years(30, 0) is None
    assert sc.runway_years(30, None) is None
    assert sc.runway_years(None, 10) is None
    assert sc.runway_years(30, "") is None


def test_runway_points_are_a_suggestion_at_the_sheets_thresholds():
    assert sc.runway_points(2.5) == 2      # > 2 år
    assert sc.runway_points(2.0) == 1      # exakt 2 år är inte "över 2"
    assert sc.runway_points(1.0) == 1
    assert sc.runway_points(0.9) == 0      # under 12 mån = emission väntar
    assert sc.runway_points(None) is None


# ── Durrett: värderingen ─────────────────────────────────────────────────────
def test_mcap_per_oz():
    assert sc.mcap_per_oz(500, 5) == 100.0        # MUSD / Moz = $/oz
    assert sc.mcap_per_oz(500, 0) is None
    assert sc.mcap_per_oz(None, 5) is None


def test_mcap_per_earnings_and_the_buy_rule():
    assert sc.mcap_per_earnings(500, 100) == 5.0
    assert sc.durrett_buy_ok(5.0)
    assert not sc.durrett_buy_ok(10.0)            # regeln är "under 10x"
    assert not sc.durrett_buy_ok(12.0)
    assert not sc.durrett_buy_ok(None)
    assert sc.DURRETT_BUY_MAX == 10.0


def test_future_profit_helper():
    """Guidens formel: produktion × (målpris − AISC)."""
    assert sc.future_profit(200, 3000, 1500) == 300_000    # koz × $/oz
    assert sc.future_profit(200, 1500, 1500) == 0          # ingen marginal
    assert sc.future_profit(None, 3000, 1500) is None


def test_the_helper_feeds_the_buy_rule_in_the_units_the_sheet_uses():
    """200 koz vid $3000 med AISC $1500 -> 300 MUSD/år."""
    musd = sc.future_profit(200, 3000, 1500) / 1000.0
    assert musd == 300.0
    assert sc.mcap_per_earnings(2000, musd) is not None
    assert round(sc.mcap_per_earnings(2000, musd), 2) == 6.67
    assert sc.durrett_buy_ok(sc.mcap_per_earnings(2000, musd))


# ── Listan ───────────────────────────────────────────────────────────────────
def test_ranked_sorts_best_first_and_keeps_unscored_rows_last():
    rows = [{"id": "a", "ticker": "MID", "factors": {"balans": 2, "vardering": 2,
                                                     "tillvaxt": 2}},
            {"id": "b", "ticker": "TOP", "factors": {f.key: 2 for f in sc.FACTORS}},
            {"id": "c", "ticker": "NEW", "factors": {}}]
    out = sc.ranked(rows)
    assert [r["row"]["ticker"] for r in out] == ["TOP", "MID", "NEW"]
    assert out[0]["verdict"] == sc.CORE
    assert out[-1]["score"] is None and out[-1]["verdict"] is None
    assert sc.ranked([]) == []
    assert sc.ranked(None) == []


# ── Mot resten av panelen ────────────────────────────────────────────────────
def test_the_model_is_metal_agnostic_and_reads_the_rotation():
    """Spec: "Extra kolumn råvara så modellen funkar för alla metaller"."""
    import rotation
    assert len(sc.COMMODITIES) == len(rotation.COMMODITIES)
    assert "Uran" in sc.COMMODITIES and "Koppar" in sc.COMMODITIES


def test_sprott_position_rule_matches_its_playbook():
    import strategy_rules as sr
    pb = sr.PLAYBOOKS["sprott"]
    note = sc.POSITION_NOTE[sc.SPROTT]
    assert "1–2 %" in note and "1–2 %" in pb.risk.position_size
    assert "10–15" in note and "10–15" in pb.risk.max_positions
