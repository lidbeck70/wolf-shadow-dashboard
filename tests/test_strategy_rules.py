"""
Tests for strategy_rules.py — the single source of truth for strategy playbooks.

The reason this module (and this test) exists: the Wolf risk-per-trade used to be
stated as 1 % (CLAUDE.md / REGIME gate), 2 % (code + STRATEGIES tab) and 5 %
(RULES tab rule 8) simultaneously. These tests fail if a playbook's stated risk
ever drifts from the risk_pct the code actually trades with, or if the legacy
rule lists lose content.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import strategy_rules as sr


# ── Structure ────────────────────────────────────────────────────────────────
def test_all_playbooks_complete():
    for key, pb in sr.PLAYBOOKS.items():
        assert pb.key == key
        assert pb.name and pb.tagline and pb.idea
        assert pb.level in (sr.LEVEL_BEGINNER, sr.LEVEL_MEDIUM, sr.LEVEL_ADVANCED)
        assert pb.entry and pb.exit, f"{key} missing entry/exit rules"
        assert pb.workflow, f"{key} missing workflow"
        assert pb.cheatsheet, f"{key} missing cheatsheet"
        assert pb.pitfalls, f"{key} missing pitfalls"
        assert pb.where, f"{key} must say where in the panel it lives"


def test_every_rule_has_a_panel_guide():
    """A rule without 'where do I look?' is not learnable."""
    for key, pb in sr.PLAYBOOKS.items():
        for rule in pb.entry + pb.exit + pb.mindset:
            assert rule.text, f"{key} rule {rule.number} has no text"
            assert rule.explanation, f"{key} rule {rule.number} has no explanation"
            assert rule.panel_guide, f"{key} rule {rule.number} has no panel_guide"


def test_rules_are_numbered_from_one():
    for pb in sr.PLAYBOOKS.values():
        for group in (pb.entry, pb.exit, pb.mindset):
            if group:
                assert [r.number for r in group] == list(range(1, len(group) + 1))


def test_every_playbook_has_hard_rules():
    """Each strategy must mark at least one non-negotiable rule."""
    for key, pb in sr.PLAYBOOKS.items():
        assert any(r.hard for r in pb.entry + pb.exit), f"{key} has no hard rule"


def test_learning_order_covers_all_and_starts_with_beginner():
    assert set(sr.LEARNING_ORDER) == set(sr.PLAYBOOKS)
    assert sr.PLAYBOOKS[sr.LEARNING_ORDER[0]].level == sr.LEVEL_BEGINNER


# ── The anti-drift guard ─────────────────────────────────────────────────────
def _pct_in(text: str) -> set:
    """All percentages mentioned in a risk string, as floats.

    Handles ranges written with an en/em dash where only the last number carries
    the sign ("12–20 %" -> {12.0, 20.0}).
    """
    num = r"\d+(?:[.,]\d+)?"
    out = set()
    for lo, hi in re.findall(rf"({num})\s*[–—-]\s*({num})\s*%", text):
        out.add(float(lo.replace(",", ".")))
        out.add(float(hi.replace(",", ".")))
    out.update(float(m.replace(",", ".")) for m in re.findall(rf"({num})\s*%", text))
    return out


def test_risk_matches_the_code_that_actually_trades():
    """A playbook's stated risk must contain the strategy's real risk_pct.

    This is the check that would have caught the 1 % / 2 % / 5 % drift.
    """
    from strategies.wolf import DEFAULT_PARAMS as WOLF_P
    from strategies.alpha import DEFAULT_PARAMS as ALPHA_P
    from strategies.viking import DEFAULT_PARAMS as VIKING_P

    for key, params in (("wolf", WOLF_P), ("alpha", ALPHA_P), ("viking", VIKING_P)):
        code_pct = float(params["risk_pct"]) * 100
        stated = _pct_in(sr.PLAYBOOKS[key].risk.risk_per_trade)
        assert code_pct in stated, (
            f"{key}: playbook says {stated}, code trades {code_pct} % "
            f"(risk_pct={params['risk_pct']})")


def test_momentum_risk_is_consistent_with_its_own_sizing():
    """Momentum states ≈1,2–2 %; that must equal position size × stop."""
    pb = sr.PLAYBOOKS["momentum"]
    size = _pct_in(pb.risk.position_size)          # 12–20 %
    stop = _pct_in(pb.risk.stop)                   # 10 %, 20 %
    assert {12.0, 20.0} <= size
    assert 10.0 in stop
    stated = _pct_in(pb.risk.risk_per_trade)
    assert {1.2, 2.0} <= stated, stated            # 12%*10% .. 20%*10%


# ── Legacy compatibility ─────────────────────────────────────────────────────
def test_legacy_aliases_preserve_original_rule_counts():
    """rules_page.py historically owned these lists; nothing may be lost."""
    assert len(sr.SWING_RULES) == 11        # Wolf: 7 entry + 4 exit
    assert len(sr.LONGTERM_RULES) == 10     # Alpha: 7 entry + 3 exit
    assert len(sr.OVTLYR_ENTRY_RULES) == 10
    assert len(sr.OVTLYR_EXIT_RULES) == 10
    assert len(sr.OVTLYR_MINDSET) == 3
    for rule in sr.SWING_RULES + sr.LONGTERM_RULES:
        assert {"number", "text", "explanation", "panel_guide"} <= set(rule)


def test_rules_page_and_overview_read_the_same_source():
    """Both tabs must render from strategy_rules, not their own copies."""
    from ovtlyr.ui import rules_page
    from tabs import strategy_overview

    assert rules_page.PLAYBOOKS is sr.PLAYBOOKS
    for key, pb in sr.PLAYBOOKS.items():
        meta_risk = strategy_overview._META[key]["risk"][0]
        assert pb.risk.risk_per_trade in meta_risk
