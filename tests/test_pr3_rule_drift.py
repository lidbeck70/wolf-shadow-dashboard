"""
PR 3 av panelgenomgången — regeldrift.

Regeltexten och koden hade glidit isär på flera ställen: ½ ATR / 1 % låg kvar i
Wolf-guiden, Viking-exiten, Alpha-grinden och REGIME-flikens grindar och
SL/TP-kalkylator; Deep Contrarian sa +15 % där koden släpper +5 %; Sprotts
18-månadersregel och Durretts "≥ 8 OCH under 10x" var text utan grind;
journalen kunde bara bokföra tre av tretton strategier; masterguidens
positionsstorlekar gick över allokerarens hårda tak.

Testerna här låser varje punkt mot den kod som faktiskt handlar.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _code_lines(text: str) -> str:
    """Källkoden utan kommentarer — historiken får nämna det gamla talet."""
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


# ── ½ ATR / 1 % är borta ur regeltext och grindar ────────────────────────────
def test_regime_gates_and_calculator_use_the_wolf_engine_multiplier():
    src = _code_lines(_src("tabs", "regime.py"))
    assert "_WOLF_ATR_MULT" in src and "_WOLF_RISK_PCT" in src
    assert "_half_atr" not in src and "½" not in src
    # Stoppen får inte räknas som halva ATR:en igen — varken * 0.5 eller / 2.
    assert not re.search(r"atr\w*\s*\*\s*0\.5", src, flags=re.I), "ATR * 0.5 is back"
    assert not re.search(r"atr\w*\s*/\s*2\b", src, flags=re.I), "ATR / 2 is back"
    # Risk-% i kalkylatorn får inte falla tillbaka på 5 %.
    assert 'value=5.0' not in src


def test_alpha_gate_9_sizes_on_ema200_not_half_atr():
    src = _code_lines(_src("long_regime_monitor.py"))
    assert "½ ATR" not in src and "atr/2" not in src
    assert "Kapital × 1,5 %" in src and "EMA200" in src


def test_viking_exit_trigger_quotes_the_engine_stop():
    from strategies.viking import DEFAULT_PARAMS as VIKING_P
    import ovtlyr.signals.longterm_signals as ls
    assert ls._VIKING_ATR_MULT == float(VIKING_P["atr_stop_mult"])
    src = _code_lines(_src("ovtlyr", "signals", "longterm_signals.py"))
    assert "½ ATR" not in src


def test_no_half_atr_left_in_any_rendered_rule_text():
    """Guiderna i rules_page, strategy_overview och backtest-docstringen."""
    for path in (("ovtlyr", "ui", "rules_page.py"),
                 ("tabs", "strategy_overview.py"),
                 ("backtest_engine.py",)):
        src = _code_lines(_src(*path))
        assert "½" not in src, f"{'/'.join(path)} still says ½ ATR"


def test_wolf_backtest_defaults_are_the_wolf_engine_defaults():
    """Den fristående backtesten utan preset ska testa samma Wolf som RULES beskriver."""
    from strategies.wolf import DEFAULT_PARAMS as WOLF_P
    import wolf_shadow_backtest as bt
    assert bt.CONFIG["atr_mult"] == WOLF_P["atr_mult"]
    assert bt.CONFIG["risk_pct"] == WOLF_P["risk_pct"]
    assert (bt.CONFIG["tp1_rr"], bt.CONFIG["tp2_rr"]) == (WOLF_P["tp1_r"], WOLF_P["tp2_r"])
    assert (bt.CONFIG["tp1_pct"], bt.CONFIG["tp2_pct"]) == (WOLF_P["tp1_pct"], WOLF_P["tp2_pct"])
    assert bt.CONFIG["core_pct"] == WOLF_P["core_pct"]


# ── Deep Contrarian: +5 %, inte +15 % ────────────────────────────────────────
def test_deep_contrarian_hope_window_matches_the_screener_and_regime_code():
    import strategy_rules as sr
    from alpha_regime.contrarian_signals import MAX_ABOVE_MA200_PCT, MIN_NEAR_MA200_PCT
    from contrarian_alpha.engine import PipelineConfig
    assert MAX_ABOVE_MA200_PCT == PipelineConfig().deep_max_above_sma200_pct == 5.0

    pb = sr.PLAYBOOKS["contrarian"]
    text = " ".join([r.text for r in pb.entry] + [f"{k} {v}" for k, v in pb.cheatsheet])
    assert "+15 %" not in text
    assert f"−{abs(MIN_NEAR_MA200_PCT):g} % till +{MAX_ABOVE_MA200_PCT:g} %" in text

    guide = _code_lines(_src("ovtlyr", "ui", "rules_page.py"))
    assert "+15%" not in guide and "till +5%" in guide


# ── Sprott: 18 månaders runway är en grind, inte bara text ──────────────────
def test_sprott_runway_gate_is_18_months_everywhere():
    import scoring as sc
    import reference
    import strategy_rules as sr
    assert sc.SPROTT_RUNWAY_MIN_MONTHS == 18
    assert sc.sprott_runway_ok(1.5) is True          # exakt 18 mån räcker
    assert sc.sprott_runway_ok(1.49) is False
    assert sc.sprott_runway_ok(None) is None         # okänd är inte ett nej
    sell = next(r for r in reference.SELL_RULES if r.key == "sprott")
    assert "18 mån" in sell.rule and "12 mån" not in sell.rule
    assert "18 mån" in " ".join(r.explanation for r in sr.PLAYBOOKS["sprott"].entry)


def test_sprott_short_runway_is_a_pass_regardless_of_score():
    import scoring as sc
    full = {f.key: 2 for f in sc.FACTORS}
    short = {"ticker": "S", "factors": full, "cash": 10, "burn": 10}     # 12 mån
    vd, why = sc.gated_verdict(sc.SPROTT, short)
    assert vd == sc.PASS and "18 mån" in why
    fine = {"ticker": "S", "factors": full, "cash": 30, "burn": 10}      # 36 mån
    assert sc.gated_verdict(sc.SPROTT, fine) == (sc.CORE, "")
    unknown = {"ticker": "S", "factors": full}
    assert sc.gated_verdict(sc.SPROTT, unknown) == (sc.CORE, "")
    assert sc.gated_verdict(sc.SPROTT, {"ticker": "S", "factors": {}}) == (None, "")


# ── Durrett: Kärninnehav kräver ≥ 8 OCH under 10× ────────────────────────────
def test_durrett_core_requires_the_buy_rule():
    import scoring as sc
    full = {f.key: 2 for f in sc.FACTORS}
    no_ratio = {"ticker": "D", "factors": full}
    vd, why = sc.gated_verdict(sc.DURRETT, no_ratio)
    assert vd == sc.WATCH and "10×" in why
    expensive = {"ticker": "D", "factors": full, "mcap": 1200, "profit": 100}   # 12×
    vd, why = sc.gated_verdict(sc.DURRETT, expensive)
    assert vd == sc.WATCH and "12.0×" in why
    cheap = {"ticker": "D", "factors": full, "mcap": 500, "profit": 100}        # 5×
    assert sc.gated_verdict(sc.DURRETT, cheap) == (sc.CORE, "")
    # Grinden sänker bara Kärninnehav — en 6-poängare är Bevakningslista ändå.
    watch = {"ticker": "D", "factors": {"balans": 2, "vardering": 2, "tillvaxt": 2}}
    assert sc.gated_verdict(sc.DURRETT, watch) == (sc.WATCH, "")


def test_ranked_carries_the_gate_and_stays_pure_without_a_key():
    import scoring as sc
    full = {f.key: 2 for f in sc.FACTORS}
    rows = [{"id": "a", "ticker": "D", "factors": full, "mcap": 1200, "profit": 100}]
    gated = sc.ranked(rows, sc.DURRETT)[0]
    assert gated["verdict"] == sc.WATCH and gated["gate"]
    pure = sc.ranked(rows)[0]
    assert pure["verdict"] == sc.CORE and pure["gate"] == ""
    assert pure["score"] == gated["score"] == sc.MAX_SCORE


def test_review_link_reads_the_gated_verdict():
    import scoring as sc
    import review_link as rl
    full = {f.key: 2 for f in sc.FACTORS}
    stores = {"scoring": {"sprott": [{"ticker": "S", "factors": full, "cash": 5, "burn": 10}],
                          "durrett": [{"ticker": "D", "factors": full, "mcap": 1200, "profit": 100}]}}
    s = rl.review("sprott", "S", stores)
    assert s["status"] == rl.FAIL and "18 mån" in s["note"]
    d = rl.review("durrett", "D", stores)
    assert d["status"] == rl.MANUAL and "10×" in d["note"]


# ── Positionsstorlek: masterguiden får inte gå över allokerarens tak ─────────
def test_masterguide_position_sizes_fit_under_the_allocator_hard_caps():
    import allocator
    import scoring as sc
    import strategy_rules as sr
    num = r"\d+(?:[.,]\d+)?"
    for key in ("sprott", "durrett"):
        cap = allocator.RULE_BY_KEY[key].hard_cap
        size = sr.PLAYBOOKS[key].risk.position_size
        stated = [float(m.replace(",", ".")) for m in re.findall(rf"({num})\s*%", size)]
        assert stated, key
        assert max(stated) <= cap, f"{key}: '{size}' exceeds hard cap {cap} %"
    assert "1,5 %" in sc.POSITION_NOTE[sc.SPROTT]


# ── Journalen: alla strategier ───────────────────────────────────────────────
def test_journal_can_book_every_playbook():
    import strategy_rules as sr
    import trade_journal as tj
    keys = tj.journal_strategies()
    assert keys[:3] == ["wolf", "viking", "alpha"]          # som förut, först
    assert set(keys) == set(sr.PLAYBOOKS)
    assert len(keys) == len(set(keys))
    assert tj._strategy_label("contrarian") == "Deep Contrarian"
    assert tj._strategy_label("wolf") == "Wolf"
