"""
Tests for producers.py — Rick Rule + Royalty C.

Asserted against migrationsspecen §6:

  Rick Rule (guidens "Producenter A"): marginal = (pris−kostnad)/pris ·
  poäng = (marginal >= 40 % -> 2p, >= 25 % -> 1p) + tre checkfält -> 0-5 ·
  badge: >= 4 Köpkandidat · 3 Bevaka · < 3 Passa.

  Royalty C: rabatt vs botten · vs median · GEO-tillväxt · signal:
  rabatt <= 15 % & vs median <= 0 & GEO-tillväxt > 0 -> "KÖPLÄGE" ·
  rabatt <= 15 % -> "Nära hist. botten — kolla GEO" · GEO < 0 ->
  "Varning: GEO/aktie krymper" · annars "Neutral".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import producers as p


# ── Rick Rule ────────────────────────────────────────────────────────────────
def test_margin_and_its_points_at_the_specs_boundaries():
    assert p.margin_pct(100, 60) == 40.0
    assert p.margin_points(40.0) == 2       # exakt 40 %
    assert p.margin_points(39.9) == 1
    assert p.margin_points(25.0) == 1       # exakt 25 %
    assert p.margin_points(24.9) == 0
    assert p.margin_points(None) == 0


def test_margin_is_rounded_so_the_boundaries_hold():
    """A cost that is exactly 60 % of price must score 2, not 1."""
    assert p.margin_pct(3000, 1800) == 40.0
    assert p.margin_points(p.margin_pct(3000, 1800)) == 2
    assert p.margin_pct(80, 60) == 25.0
    assert p.margin_points(p.margin_pct(80, 60)) == 1


def test_a_negative_margin_is_a_loss_making_producer():
    assert p.margin_pct(100, 130) == -30.0
    assert p.margin_points(-30.0) == 0


def test_margin_needs_a_price():
    assert p.margin_pct(0, 50) is None
    assert p.margin_pct(None, 50) is None
    assert p.margin_pct(100, None) is None


def test_producer_score_combines_margin_and_the_three_checks():
    row = {"price": 100, "unit_cost": 55, "jurisdiktion": True,
           "kapitaldisciplin": True, "insyn": True}
    assert p.margin_pct(100, 55) == 45.0
    assert p.producer_score(row) == 2 + 3 == p.PROD_MAX_SCORE
    assert p.producer_verdict(p.producer_score(row)).label == p.P_BUY


def test_producer_score_is_blank_without_a_margin():
    """Three ticked boxes and no margin is not a 3-point company."""
    assert p.producer_score({"jurisdiktion": True, "kapitaldisciplin": True,
                             "insyn": True}) is None
    assert p.producer_verdict(None) is None


def test_producer_verdict_thresholds():
    assert p.producer_verdict(5).label == p.P_BUY
    assert p.producer_verdict(4).label == p.P_BUY
    assert p.producer_verdict(3).label == p.P_WATCH
    assert p.producer_verdict(2).label == p.P_PASS
    assert p.producer_verdict(0).label == p.P_PASS


def test_jurisdiction_is_additive_here_not_a_veto():
    """Worth pinning, because the neighbouring sheet treats it differently.

    Lobo's criteria say a bottom-half jurisdiction is a pass regardless of the
    project. The Rick Rule sheet does not: the spec's model is a plain sum, so a
    low-cost producer can reach Köpkandidat on margin plus two other checks
    with the jurisdiction box unticked. That is the sheet's rule, not an
    oversight here — change it in one place if you want the veto.
    """
    row = {"price": 100, "unit_cost": 40}                  # 60 % marginal = 2p
    assert p.producer_score(row) == 2
    assert p.producer_verdict(2).label == p.P_PASS
    row["kapitaldisciplin"] = True                         # 3p
    assert p.producer_verdict(p.producer_score(row)).label == p.P_WATCH
    row["insyn"] = True                                    # 4p, utan jurisdiktion
    assert p.producer_score(row) == 4
    assert p.producer_verdict(p.producer_score(row)).label == p.P_BUY


# ── Royalty C ────────────────────────────────────────────────────────────────
def _roy(**kw) -> dict:
    base = {"pnav_now": 1.10, "pnav_bottom": 1.00, "ev_now": 18.0,
            "ev_median": 20.0, "geo_now": 0.05, "geo_3y": 0.04}
    base.update(kw)
    return base


def test_the_three_royalty_measures():
    assert p.discount_vs_bottom(1.10, 1.00) == 10.0
    assert p.vs_median(18.0, 20.0) == -10.0
    assert p.geo_growth(0.05, 0.04) == 25.0


def test_royalty_measures_guard_zero_denominators():
    assert p.discount_vs_bottom(1.1, 0) is None
    assert p.vs_median(18, 0) is None
    assert p.geo_growth(0.05, 0) is None
    assert p.geo_growth(None, 0.04) is None


def test_buy_signal_needs_all_three_conditions():
    assert p.royalty_signal(_roy()).label == p.R_BUY
    # för dyr mot botten
    assert p.royalty_signal(_roy(pnav_now=1.30)).label == p.R_NEUTRAL
    # över egen median
    assert p.royalty_signal(_roy(ev_now=25.0)).label == p.R_NEAR
    # GEO växer inte
    assert p.royalty_signal(_roy(geo_now=0.04)).label == p.R_NEAR


def test_the_discount_boundary_is_fifteen_percent():
    assert p.royalty_signal(_roy(pnav_now=1.15)).label == p.R_BUY   # exakt 15 %
    assert p.royalty_signal(_roy(pnav_now=1.16)).label == p.R_NEUTRAL
    assert p.DISCOUNT_MAX == 15.0


def test_a_shrinking_geo_overrides_a_cheap_price():
    """Cheap because it diluted is not cheap. The warning must win."""
    row = _roy(geo_now=0.03)        # −25 % GEO/aktie, priset nära botten
    assert p.geo_growth(0.03, 0.04) < 0
    v = p.royalty_signal(row)
    assert v.label == p.R_GEO_WARN
    assert "utspädningen" in v.why


def test_royalty_signal_never_crashes_on_an_empty_row():
    v = p.royalty_signal({})
    assert v.label == p.R_NEUTRAL


# ── Listorna ─────────────────────────────────────────────────────────────────
def test_producers_rank_best_first_with_unscored_last():
    rows = [{"id": "a", "ticker": "MID", "price": 100, "unit_cost": 70},
            {"id": "b", "ticker": "TOP", "price": 100, "unit_cost": 50,
             "jurisdiktion": True, "kapitaldisciplin": True, "insyn": True},
            {"id": "c", "ticker": "NEW"}]
    out = p.ranked_producers(rows)
    assert [r["row"]["ticker"] for r in out] == ["TOP", "MID", "NEW"]
    assert out[-1]["score"] is None
    assert p.ranked_producers([]) == []
    assert p.ranked_producers(None) == []


def test_royalty_ranks_buys_first_and_warnings_last():
    rows = [{"id": "a", "ticker": "NEUTRAL", **_roy(pnav_now=1.30)},
            {"id": "b", "ticker": "WARN", **_roy(geo_now=0.03)},
            {"id": "c", "ticker": "BUY", **_roy()},
            {"id": "d", "ticker": "NEAR", **_roy(ev_now=25.0)}]
    out = p.ranked_royalty(rows)
    assert [r["row"]["ticker"] for r in out] == ["BUY", "NEAR", "NEUTRAL", "WARN"]
    assert p.ranked_royalty([]) == []


# ── Mot resten av panelen ────────────────────────────────────────────────────
def test_the_commodity_list_comes_from_the_rotation():
    import rotation
    assert len(p.COMMODITIES) == len(rotation.COMMODITIES)


def test_royalty_levels_match_the_playbooks_sell_rule():
    """Nivå 1 säljs aldrig, 2 vid stagnerande GEO i 2 år, 3 vid uppköp."""
    import reference
    row = reference.sell_rule("royalty").rule
    assert "Nivå 1" in row and "Nivå 2" in row and "Nivå 3" in row
    assert set(p.ROYALTY_LEVELS) == {1, 2, 3}
    assert "aldrig" in p.LEVEL_NOTE[1]
    assert "GEO" in p.LEVEL_NOTE[2]


# ── Guidens egna exempel (Masterguiden 4.0, Del 4, Strategi 1: Rule) ─────────
# Rick Rule-arket ÄR guidens "Producenter A" — den säger det rakt ut, och båda
# exemplen nedan är dess egna siffror.
def test_the_guides_aisc_example_scores_two_points():
    """"AISC $1 400 vid guld $2 600 = 46 % marginal = lågkostnad (2 p)"."""
    m = p.margin_pct(2600, 1400)
    assert round(m, 1) == 46.2
    assert p.margin_points(m) == 2


def test_the_guides_mini_example_scores_five_of_five():
    """EV/EBITDA 3,8 · nettokassa · soliditet 64 % · AISC i 25:e percentilen ·
    Nevada · VD äger 8 årslöner · 6 % direktavkastning -> 5/5 -> full position."""
    row = {"ticker": "MINI", "ev_ebitda": 3.8, "nd_ebitda": -0.5,
           "price": 2600, "unit_cost": 1400,      # lågkostnad -> 2 p
           "jurisdiktion": True,                   # Nevada
           "insyn": True,                          # VD äger 8 årslöner
           "kapitaldisciplin": True,               # 6 % direktavkastning
           "mine_life": 12}
    assert p.producer_score(row) == p.PROD_MAX_SCORE == 5
    assert p.producer_verdict(p.producer_score(row),
                              p.asset_dying(row)).label == p.P_BUY


def test_the_same_company_with_a_four_year_mine_is_a_pass():
    """The guide's own follow-up: "Samma bolag men med gruvlivslängd 4 år →
    priset är lågt av ett skäl → passa"."""
    row = {"price": 2600, "unit_cost": 1400, "jurisdiktion": True,
           "insyn": True, "kapitaldisciplin": True, "mine_life": 4}
    assert p.producer_score(row) == 5          # poängen är oförändrad...
    assert p.asset_dying(row) is True
    v = p.producer_verdict(p.producer_score(row), p.asset_dying(row))
    assert v.label == p.P_PASS                 # ...men bedömningen är inte det
    assert p.DYING_ASSET in v.why


def test_the_rp_ratio_strikes_oil_and_gas_the_same_way():
    row = {"price": 80, "unit_cost": 40, "jurisdiktion": True, "insyn": True,
           "kapitaldisciplin": True, "rp_ratio": 6}
    assert p.asset_dying(row) is True
    assert p.producer_verdict(5, True).label == p.P_PASS
    row["rp_ratio"] = 10
    assert p.asset_dying(row) is False


def test_the_strike_boundaries_are_the_guides():
    assert p.LIFE_MIN_YEARS == 5.0 and p.RP_MIN_YEARS == 8.0
    assert p.asset_dying({"mine_life": 5}) is False
    assert p.asset_dying({"mine_life": 4.9}) is True
    assert p.asset_dying({"rp_ratio": 8}) is False
    assert p.asset_dying({"rp_ratio": 7.9}) is True


def test_an_unanswered_life_question_is_not_a_clean_bill_of_health():
    """None, not False — the scorecard's gap rule handles it from there."""
    assert p.asset_dying({}) is None
    assert p.asset_dying({"price": 100, "unit_cost": 50}) is None
    # ...och den dömer inte ut bolaget här
    assert p.producer_verdict(5, None).label == p.P_BUY


def test_an_empty_year_field_is_not_zero_years():
    """Buggen: EQX, 25 års gruvlivslängd, dömdes ut som döende tillgång.

    Fälten ritas med number_input(min_value=0.0) och kan inte lämnas tomma, så
    ett orört R/P sparas som 0,0. Läst bokstavligt är 0 < 8 och varje
    guldproducent blev en passa — R/P gäller bara olja och gas och står därför
    alltid kvar på noll.
    """
    eqx = {"ticker": "EQX", "price": 4250, "unit_cost": 2175,
           "jurisdiktion": True, "insyn": True, "kapitaldisciplin": True,
           "mine_life": 25.0, "rp_ratio": 0.0}
    assert p.asset_dying(eqx) is False
    assert p.producer_verdict(p.producer_score(eqx),
                              p.asset_dying(eqx)).label == p.P_BUY
    # ...och spegelfallet: ett oljebolag utan gruvlivslängd
    assert p.asset_dying({"rp_ratio": 12.0, "mine_life": 0.0}) is False


def test_both_year_fields_at_zero_means_unanswered_not_dying():
    """0/tomt är EJ IFYLLT — hint i UI:t, inte röd varning."""
    assert p.asset_dying({"mine_life": 0.0, "rp_ratio": 0.0}) is None
    assert p.asset_dying({"mine_life": 0}) is None
    assert p.asset_dying({"mine_life": ""}) is None
    # obesvarad fråga sänker inte bolaget här
    assert p.producer_verdict(5, None).label == p.P_BUY


def test_the_strike_still_fires_on_a_real_short_life():
    """Nollan får inte tystas genom att strykregeln slutar fungera."""
    assert p.asset_dying({"mine_life": 4.0, "rp_ratio": 0.0}) is True
    assert p.asset_dying({"mine_life": 0.0, "rp_ratio": 6.0}) is True
    assert p.asset_dying({"mine_life": 25.0, "rp_ratio": 6.0}) is True
    v = p.producer_verdict(5, p.asset_dying({"mine_life": 4.0}))
    assert v.label == p.P_PASS


def test_the_header_status_follows_the_recomputed_verdict():
    """Headern läser ranked_producers, som räknar om vid varje rerun."""
    rows = [{"id": "a", "ticker": "EQX", "price": 4250, "unit_cost": 2175,
             "jurisdiktion": True, "insyn": True, "kapitaldisciplin": True,
             "mine_life": 25.0, "rp_ratio": 0.0}]
    out = p.ranked_producers(rows)
    assert out[0]["score"] == 5
    assert out[0]["dying"] is False
    assert out[0]["verdict"].label == p.P_BUY
    # samma rad, kort livslängd -> headern vänder i samma rerun
    rows[0]["mine_life"] = 4.0
    assert p.ranked_producers(rows)[0]["verdict"].label == p.P_PASS


def test_ranked_exposes_the_dying_flag():
    rows = [{"id": "a", "ticker": "DYING", "price": 100, "unit_cost": 40,
             "jurisdiktion": True, "insyn": True, "kapitaldisciplin": True,
             "mine_life": 3}]
    out = p.ranked_producers(rows)
    assert out[0]["dying"] is True
    assert out[0]["score"] == 5
    assert out[0]["verdict"].label == p.P_PASS


# ── Namngivningen ────────────────────────────────────────────────────────────
def test_the_sheet_is_named_after_its_author_like_the_others():
    """Sprott, Durrett and Lobo Tiggre carry their names; this one now does too."""
    assert p.SHEET_RULE == "Rick Rule"
    assert p.SHEET_ROYALTY == "Royalty C"


def test_the_guides_own_name_is_kept_for_traceability():
    """Renaming the tab must not orphan the guide's wording — it is what you
    search for when checking the panel against the PDF."""
    assert p.GUIDE_NAME_RULE == "Producenter A"
    import strategy_rules as sr
    pb = sr.PLAYBOOKS["rule"]
    assert p.SHEET_RULE in pb.where
    assert p.GUIDE_NAME_RULE in pb.where
    assert p.GUIDE_NAME_RULE in pb.support_note
    # och guidens formulering står kvar i regeln som citerar den
    cost_rule = [r for r in pb.entry if "AISC" in r.explanation][0]
    assert p.GUIDE_NAME_RULE in cost_rule.panel_guide


def test_the_storage_keys_did_not_change_with_the_label():
    """Renaming a tab must not orphan saved rows in the Gist."""
    assert p.PRODUCERS == "producers"
    assert p.ROYALTY == "royalty"
