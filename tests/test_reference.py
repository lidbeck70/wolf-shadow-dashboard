"""
Tests for reference.py — Snabbreferensen (Masterguiden Del 7).

The point of these is not that the tables have rows. It is that the one-page
summary can never say something the engines do not actually do: a sell rule
here that drifts from the playbook it summarises would be a trap, since this
is the page you read mid-trade instead of the playbook.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reference as ref
import strategy_rules as sr
import allocator
import tiggre


# ── Bindningen mot strategierna ──────────────────────────────────────────────
def test_every_keyed_row_names_a_real_playbook():
    for s in ref.SELL_RULES:
        if s.key:
            assert s.key in sr.PLAYBOOKS, f"säljregel utan playbook: {s.key}"
    for s in ref.SCREENERS:
        if s.key:
            assert s.key in sr.PLAYBOOKS, f"screener utan playbook: {s.key}"


def test_the_unkeyed_rows_are_the_cross_strategy_ones():
    """De hör till ingen enskild playbook.

    Palladium/Portföljnivå är råvaru- och portföljnivå. Lukacs FV är en
    värderingsmodul som läggs ovanpå producentstrategierna — den ägs av Rule
    lika lite som av Durrett, och därför av ingen av dem.
    """
    unkeyed = [s.strategy for s in ref.SELL_RULES if not s.key]
    assert unkeyed == ["Palladium / Litium / Silver", "Lukacs FV",
                       "Portföljnivå"]


def test_every_masterguide_strategy_has_a_sell_row():
    """The guide's seven strategies plus the two cross-cutting rules."""
    for key in ("rule", "sprott", "durrett", "tiggre", "royalty", "momentum",
                "insider"):
        assert ref.sell_rule(key) is not None, key
    assert len(ref.SELL_RULES) == 10


def test_lookup_helpers():
    assert ref.sell_rule("tiggre").strategy == "Tiggre"
    assert ref.sell_rule("nonsense") is None
    assert ref.sell_rule("") is None
    assert ref.screener("rule").name == "Överlevarna (Rule)"
    assert ref.screener("nonsense") is None


# ── Anti-drift: sammanfattningen mot motorerna ───────────────────────────────
def test_swing_sell_row_matches_the_momentum_playbook():
    row = ref.sell_rule("momentum").rule
    risk = sr.PLAYBOOKS["momentum"].risk
    assert "−10 %" in row and "−10 %" in risk.stop
    assert "+20 %" in row and "+20 %" in risk.targets
    assert "MA50" in row
    assert "topp 40" in row


def test_tiggre_sell_row_matches_the_tiggre_engine():
    row = ref.sell_rule("tiggre").rule
    assert f"+{int(tiggre.FREE_RIDE_PCT)} %" in row          # free ride
    assert f"{tiggre.NAV_TARGET}".replace(".", ",") in row   # 0,8x NAV
    assert "Tes bruten" in row


def test_portfolio_row_matches_the_circuit_breaker():
    row = [s for s in ref.SELL_RULES if s.strategy == "Portföljnivå"][0].rule
    levels = {lvl[2]: (lvl[0], lvl[1]) for lvl in allocator.BREAKER_LEVELS}
    assert "SKÄRPT" in levels and "HALVERAD RISK" in levels
    assert f"−{int(levels['SKÄRPT'][0])} %" in row           # −10 % skärpt
    assert f"−{int(levels['HALVERAD RISK'][0])} %" in row    # −20 % halverad
    assert "skärpt" in row and "halverad risk" in row


def test_never_hold_metals_row_matches_the_kartbok():
    import commodity_book as book
    row = [s for s in ref.SELL_RULES if not s.key][0]
    for name in ("Palladium", "Litium", "Silver"):
        assert name in row.strategy
        assert book.chapter(name.lower()).pitfall, name


def _playbook_text(key: str) -> str:
    pb = sr.PLAYBOOKS[key]
    return " ".join(
        [pb.idea, pb.where, " ".join(pb.workflow)]
        + [r.text + r.explanation for r in pb.entry + pb.exit])


def test_screener_filters_agree_where_the_playbook_states_a_number():
    """Where both sides quote a threshold they must be the same threshold."""
    assert "0,5" in ref.screener("rule").filters
    assert "0,5" in _playbook_text("rule")
    assert "18 mån" in _playbook_text("sprott")        # runway-grinden
    assert "MA200" in ref.screener("momentum").filters
    assert "MA200" in _playbook_text("momentum")


def test_the_screener_table_supplies_what_the_playbooks_only_refer_to():
    """The gap this table closes.

    Several playbooks say "kör Sprott-screenern" or "kör Insider-grinden"
    without ever defining the filter — the definition lived only in the guide.
    If a playbook ever starts spelling its own filter out, that is fine; this
    test only guards that the definition exists in at least one place.
    """
    for key in ("sprott", "durrett", "royalty", "insider"):
        assert ref.screener(key) is not None, key
        assert len(ref.screener(key).filters) > 40, key


# ── Tabellerna ───────────────────────────────────────────────────────────────
def test_all_eight_screeners_are_here():
    assert len(ref.SCREENERS) == 8
    assert [s.name for s in ref.SCREENERS] == [
        "Överlevarna (Rule)", "Optionalitet (Sprott)", "Durrett",
        "Tiggre (sweet spot)", "Royalty", "Swing – universum",
        "Insider – grind", "Lukacs Discovery"]
    for s in ref.SCREENERS:
        assert s.filters and s.where


def test_all_thirteen_sources_are_here_with_a_lookup_hint():
    assert len(ref.SOURCES) == 13
    for s in ref.SOURCES:
        assert s.number and s.source and s.where, s.number


def test_glossary_covers_the_twenty_terms():
    assert len(ref.GLOSSARY) == 20
    terms = [t.term for t in ref.GLOSSARY]
    for expected in ("EV/EBITDA", "F-score", "Runway", "NAV / P/NAV", "GEO",
                     "Free ride", "R-multipel", "TC-avgifter"):
        assert expected in terms
    for t in ref.GLOSSARY:
        assert len(t.meaning) > 15, t.term


def test_glossary_search():
    assert len(ref.find_terms("")) == len(ref.GLOSSARY)
    assert [t.term for t in ref.find_terms("nav")] == ["NAV / P/NAV"]
    assert ref.find_terms("KOSTNADSKURVA")[0].term == "Kostnadskurva"
    # searches the meaning too, not only the term
    assert any(t.term == "Realränta" for t in ref.find_terms("inflation"))
    assert ref.find_terms("zzz") == []


# ── Riskdoktrinen (Masterguiden 4.0, Del 2) ──────────────────────────────────
def test_the_three_loss_types_and_which_one_is_fatal():
    assert [lt.name for lt in ref.LOSS_TYPES] == [
        "Marknadsförlust", "Modellförlust", "Permanent kapitalförlust"]
    fatal = ref.LOSS_TYPES[2]
    assert "aldrig kommer tillbaka" in fatal.what
    assert "enda förlust systemet inte tål" in fatal.response
    for lt in ref.LOSS_TYPES:
        assert lt.what and lt.response, lt.name


def test_the_two_cross_strategy_rules():
    assert "snitta aldrig ner" in ref.AVERAGING_RULE.lower()
    assert "ursprungstesen" in ref.AVERAGING_RULE
    assert "inget enskilt verktyg" in ref.TOOL_RULE.lower()


def test_control_signals_cover_all_three_controls_plus_averaging():
    labels = [a for a, _b in ref.CONTROL_SIGNALS]
    assert labels == ["AQS svag", "DS hög", "CSM Bear-katastrof",
                      "Kontrollerna försämrade"]
    for _label, action in ref.CONTROL_SIGNALS:
        assert action


def test_the_nine_questions_are_nine():
    assert len(ref.NINE_QUESTIONS) == 9
    joined = " ".join(ref.NINE_QUESTIONS).lower()
    for topic in ("sektorn", "tillgången", "råvarupriset", "utspädd",
                  "katalysator", "säljer jag om jag har rätt",
                  "säljer jag om jag har fel", "positionen"):
        assert topic in joined


def test_the_control_signals_agree_with_the_control_engine():
    """DS high means smaller or wait — the same rule controls.py enforces."""
    import controls as ctl
    ds_action = dict(ref.CONTROL_SIGNALS)["DS hög"]
    assert "finansieringsbesked" in ds_action
    assert ctl.DS_BLOCK_TEXT.endswith("finansieringsbeskedet")


def test_the_lukacs_discovery_screener_is_flagged_as_a_discovery_tool():
    """Skuldtaket 3,5 är lösare än Rules 0,5 — medvetet, för deleveraging-case.

    Utan den markeringen läses raden som ett köpfilter, och då har man just
    släppt igenom bolag Rule-screenern finns för att stoppa.
    """
    s = ref.screener_by_name("Lukacs Discovery")
    assert s is not None
    assert "3,5" in s.filters
    assert "ALDRIG köpfilter" in s.where


def test_the_lukacs_sell_row_carries_both_thresholds():
    row = [r for r in ref.SELL_RULES if r.strategy == "Lukacs FV"][0]
    assert "20 %" in row.rule and "25 %" in row.rule
    assert "modellförlust" in row.rule
