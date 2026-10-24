"""Designförslaget PR 6: Storlek-raden — riskstorlek och tak sida vid sida, formeln synlig, inget 'köp högst'."""
import pathlib

import positions
from thor import registry as reg
from thor import storlek as sk

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_size_matches_the_design_example():
    sz = sk.size(100.0, 92.0, 100000, risk=1.5, cap=6.0)
    assert sz["target_3r"] == 124.0
    assert sz["risk_amt"] == 1500 and sz["risk_shares"] == 187
    assert sz["cap_amt"] == 6000 and sz["cap_shares"] == 60
    assert "riskstorlek 187 st" in sz["formula"] and "tak ur Allokering 60 st" in sz["formula"]


def test_both_numbers_shown_never_merged():
    sz = sk.size(100.0, 92.0, 100000, risk=1.5, cap=6.0)
    assert "köp högst" not in sz["formula"] and "min(" not in sz["formula"]


def test_size_needs_a_valid_stop():
    assert sk.size(100, 101, 1000) is None
    assert sk.size(100, 0, 1000) is None
    assert sk.size(None, 90, 1000) is None
    assert sk.size(100, 90, 0) is None


def test_risk_pct_is_read_from_the_strategy_modules():
    from strategies.wolf import DEFAULT_PARAMS as w
    from strategies.viking import DEFAULT_PARAMS as v
    assert sk.risk_pct("wolf") == w["risk_pct"] * 100
    assert sk.risk_pct("viking") == v["risk_pct"] * 100
    assert sk.risk_pct("momentum") is None and sk.risk_pct("rule") is None


def test_cap_comes_from_the_allocator():
    import allocator as al
    assert sk.cap_pct("viking_nine") == al.RULE_BY_KEY["viking_nine"].hard_cap
    assert sk.cap_pct("momentum") == al.RULE_BY_KEY["swing"].hard_cap
    assert sk.cap_pct("hav") is None


def test_every_thor_strategy_has_a_position_tag():
    for key in reg.STRATEGIES:
        assert sk.POSITION_TAG[key] in positions.STRATEGIES, key


def test_position_args_for_log_buy():
    sz = sk.size(100.0, 92.0, 100000, risk=1.5)
    a = sk.position_args("ABC.ST", "ABC", "viking_nine", sz, 50)
    assert a == {"ticker": "ABC.ST", "strategy": "Viking Nine", "entry_price": 100.0, "shares": 50, "name": "ABC",
                 "stop": 92.0, "target": 124.0, "source": "thor"}


def test_sheet_strategies_go_through_the_scorecard():
    assert {"rule", "durrett", "tiggre", "royalty"} <= sk.SHEET_STRATEGIES


def test_engines_untouched():
    src = (ROOT / "thor" / "storlek.py").read_text(encoding="utf-8")
    assert "import thor.regim" not in src and "journal_stats" not in src.split('"""', 2)[2]
