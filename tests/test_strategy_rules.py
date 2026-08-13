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


def test_stop_multipliers_match_the_code():
    """Stated ATR stop distance must equal the multiplier the code trades.

    Caught a real one: viking.py was raised to 1.5×ATR ("to give price room")
    but every rule, the STRATEGIES tab and the panel hints still said ½ ATR —
    a 3× difference in where the stop sits.
    """
    from strategies.wolf import DEFAULT_PARAMS as WOLF_P
    from strategies.viking import DEFAULT_PARAMS as VIKING_P

    def _mults(text: str) -> set:
        """Multipliers written as '2,5 × ATR' / '1.5× ATR14'."""
        return {float(m.replace(",", "."))
                for m in re.findall(r"(\d+(?:[.,]\d+)?)\s*×\s*ATR", text)}

    checks = [("wolf", float(WOLF_P["atr_mult"])),
              ("viking", float(VIKING_P["atr_stop_mult"]))]
    for key, code_mult in checks:
        stated = _mults(sr.PLAYBOOKS[key].risk.stop)
        assert code_mult in stated, (
            f"{key}: playbook stop says {stated}× ATR, code uses {code_mult}× "
            f"— the rules and the engine disagree")


def test_viking_sltp_calculator_reads_the_engine_multiplier():
    """The live OVTLYR/Viking SL/TP calculator must not hardcode a multiplier.

    It computed a ½ ATR stop while viking.py traded 1.5× ATR, so the panel
    suggested a stop three times tighter than the strategy's own.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "ovtlyr", "ui", "layout.py"), encoding="utf-8").read()

    # The calculator block must source the multiplier from the strategy.
    assert "atr_stop_mult" in src, (
        "OVTLYR SL/TP calculator no longer reads atr_stop_mult from the engine")
    # And must not go back to deriving the stop from a hardcoded half-ATR.
    assert "_sl = _price - _half_atr" not in src, (
        "OVTLYR SL/TP calculator is hardcoding a ½ ATR stop again")


def test_ember_thresholds_track_ember_config():
    """Ember's playbook interpolates live values from ember/config.py."""
    from ember.config import RISK_PCT, PULLBACK_EMA_PCT, RSI_ENTRY_MAX

    pb = sr.PLAYBOOKS["ember"]
    assert RISK_PCT * 100 in _pct_in(pb.risk.risk_per_trade)
    entry_text = " ".join(r.text for r in pb.entry)
    assert f"{PULLBACK_EMA_PCT:.0f} %" in entry_text, entry_text
    assert str(RSI_ENTRY_MAX) in entry_text, entry_text


_PANEL_NATIVE = ("momentum", "quality", "alpha", "viking", "wolf", "contrarian", "ember")
_COMMODITY = ("royalty", "rule", "durrett", "tiggre", "sprott")


def test_new_playbooks_are_registered_and_ordered():
    """Every strategy must be reachable, and each ladder group must not get easier."""
    for key in _PANEL_NATIVE + _COMMODITY:
        assert key in sr.PLAYBOOKS, f"{key} missing from registry"
        assert key in sr.LEARNING_ORDER, f"{key} missing from learning order"
    assert set(sr.LEARNING_ORDER) == set(sr.PLAYBOOKS)

    # The ladder has two groups — panel-native strategies first, then the
    # Masterguide commodity family. Difficulty must not decrease inside a group.
    rank = {sr.LEVEL_BEGINNER: 0, sr.LEVEL_MEDIUM: 1, sr.LEVEL_ADVANCED: 2}
    order = list(sr.LEARNING_ORDER)
    split = order.index(_COMMODITY[0])
    for group in (order[:split], order[split:]):
        levels = [rank[sr.PLAYBOOKS[k].level] for k in group]
        assert levels == sorted(levels), (group, levels)


def test_support_status_is_stated_honestly():
    """Each playbook must say what the panel actually implements.

    The point of this field: Masterguiden's five commodity strategies are not in
    the panel, and pretending otherwise is how a rule gets skipped mid-trade.
    """
    valid = {sr.SUPPORT_FULL, sr.SUPPORT_PARTIAL, sr.SUPPORT_MANUAL}
    for key, pb in sr.PLAYBOOKS.items():
        assert pb.support in valid, f"{key}: bad support value {pb.support!r}"
        assert pb.source, f"{key}: missing source"
        if pb.support != sr.SUPPORT_FULL:
            assert pb.support_note, (
                f"{key} is not fully in the panel and must say what is missing")

    # The commodity family is documentation, not implementation.
    for key in _COMMODITY:
        assert sr.PLAYBOOKS[key].support in (sr.SUPPORT_MANUAL, sr.SUPPORT_PARTIAL), key
        assert "Masterguiden" in sr.PLAYBOOKS[key].source, key


def test_commodity_playbooks_carry_their_hard_numbers():
    """Spot-check the figures that decide position size and go/no-go."""
    tiggre = sr.PLAYBOOKS["tiggre"]
    joined = " ".join(r.text + r.explanation for r in tiggre.entry + tiggre.exit)
    assert "0,8× NAV" in joined or "0,8x NAV" in joined
    assert "U/N" in joined and "≥ 3" in joined
    assert "after tax" in joined.lower()
    assert any("+100 %" in r.text for r in tiggre.exit)      # free ride

    sprott = sr.PLAYBOOKS["sprott"]
    assert "18 mån" in " ".join(r.explanation for r in sprott.entry)   # runway
    assert "10–15" in sprott.risk.max_positions

    rule = sr.PLAYBOOKS["rule"]
    assert "0,5" in " ".join(r.explanation for r in rule.entry)        # ND/EBITDA

    royalty = sr.PLAYBOOKS["royalty"]
    assert "70 %" in " ".join(r.explanation for r in royalty.entry)    # gross margin


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


def _real_tab_names() -> set:
    """Parse the actual tab + sub-tab labels straight out of wolf_panel.py.

    Ground truth, so a renamed or moved tab makes the rules fail rather than
    silently sending the user to a path that no longer exists.
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "wolf_panel.py"), encoding="utf-8").read()
    names = set()
    block = re.search(r"tab_labels\s*=\s*\[(.*?)\]", src, re.S)
    if block:
        names |= {n.strip() for n in re.findall(r'"([^"]+)"', block.group(1))}
    for m in re.finditer(r'st\.radio\(\s*""\s*,\s*(\[[^\]]+\])', src, re.S):
        names |= {n.strip() for n in re.findall(r'"([^"]+)"', m.group(1))}
    return {n for n in names if n}


def test_panel_guides_point_at_tabs_that_exist():
    """Every 'X → Y' navigation hint must name real tabs.

    This is the check that caught the stale SECTOR & REGIME / SENTIMENT /
    SCREENER / WOLF REGIME labels left over from an older tab layout.
    """
    real = _real_tab_names()
    # Tokens that are buttons, fields or badges rather than tabs.
    allowed_extra = {
        "ANALYSERA", "Regime Score", "Entry Checklist", "Gate", "Holdings",
        "Backtest", "Sentiment", "Heatmap", "Retail Pulse", "Trade Journal",
        "Swing", "Swing Screener", "Swing Regime", "Long Screener",
        "Contrarian Alpha", "Market Cycle", "Arc Screener", "Arc Regime",
        "Alpha Regime", "Flow Divergence", "Wolf Regime", "Viking Regime",
        "Quality & Contrarian", "Long Trend", "Odin's Blindspot",
    }
    # Tab labels carry emoji ("🔥 EMBER"); the extractor below sees only the
    # letters, so index both forms.
    def _plain(name: str) -> str:
        return "".join(ch for ch in name if ch.isalnum() or ch in " &'-").strip()

    known = {n.strip() for n in real} | allowed_extra
    known |= {_plain(n) for n in known}
    # Uppercase top-level tokens that must resolve.
    tops = {"HOME", "SCREENING", "REGIME", "INTELLIGENCE", "PORTFOLIO",
            "ALERTS", "RULES", "STRATEGIES"}
    for _t in real:
        stripped = _t.strip().strip("🏠🔱📡👁💼🔔📋🧬 ")
        if stripped.isupper():
            tops.add(stripped)

    offenders = []
    for key, pb in sr.PLAYBOOKS.items():
        texts = ([pb.where] + list(pb.workflow)
                 + [r.panel_guide for r in pb.entry + pb.exit + pb.mindset])
        for text in texts:
            for m in re.finditer(r"\b([A-ZÅÄÖ][A-ZÅÄÖ'&\s]{3,25}?)\s*(?:→|:)", text):
                name = m.group(1).strip()
                if name in {"OBS", "T1", "T2", "E1", "E2", "F1"}:
                    continue
                head = name.split()[0]
                if head not in tops and name not in known:
                    offenders.append(f"{key}: '{name}' in {text[:60]}...")
    assert not offenders, "Stale panel references:\n" + "\n".join(offenders)


def test_panel_guide_table_uses_real_paths():
    """The 🗺 FLIKGUIDE map must not send the user to a tab that moved."""
    from ovtlyr.ui.rules_page import _PANEL_GUIDE

    real = _real_tab_names()
    plain = {"".join(c for c in n if c.isalnum() or c in " &'-").strip() for n in real}
    for tab, _rules, _usage in _PANEL_GUIDE:
        parts = [p.strip() for p in tab.split("→")]
        head = parts[0]
        assert head in {"HOME", "SCREENING", "REGIME", "INTELLIGENCE",
                        "PORTFOLIO", "ALERTS", "RULES", "STRATEGIES"}, head
        for part in parts[1:]:
            clean = "".join(c for c in part if c.isalnum() or c in " &'-").strip()
            assert part in real or clean in plain, (
                f"FLIKGUIDE row '{tab}' names '{part}', which is not a real tab")


def test_ember_paths_include_their_intermediate_tab():
    """Ember lives one level deeper than SCREENING/REGIME."""
    pb = sr.PLAYBOOKS["ember"]
    texts = [pb.where] + list(pb.workflow) + [r.panel_guide for r in pb.entry + pb.exit]
    for t in texts:
        if "🔥 EMBER" in t:
            assert "Arc Screener" in t, f"missing 'Arc Screener' level: {t[:70]}"
        if "🌍 EMBER Regime" in t:
            assert "Arc Regime" in t, f"missing 'Arc Regime' level: {t[:70]}"


def test_no_rules_section_is_orphaned():
    """Every rendering section must actually be reachable from the page.

    Regression guard: rewriting render_rules_page once dropped the call to
    _render_ember_full_ruleset (the 13-section Ember ruleset) and the panel
    guide table, leaving them defined but unreachable.
    """
    import inspect
    from ovtlyr.ui import rules_page

    src = inspect.getsource(rules_page)
    for fn in ("_render_ember_full_ruleset", "_page_panel_guide", "_page_start",
               "_page_rules", "_page_cheatsheet", "render_strategy_guides"):
        assert hasattr(rules_page, fn), f"{fn} missing"
        # defined once, called at least once
        assert src.count(fn) >= 2, f"{fn} is defined but never called"


def test_rules_page_and_overview_read_the_same_source():
    """Both tabs must render from strategy_rules, not their own copies."""
    from ovtlyr.ui import rules_page
    from tabs import strategy_overview

    assert rules_page.PLAYBOOKS is sr.PLAYBOOKS
    for key, pb in sr.PLAYBOOKS.items():
        meta_risk = strategy_overview._META[key]["risk"][0]
        assert pb.risk.risk_per_trade in meta_risk
