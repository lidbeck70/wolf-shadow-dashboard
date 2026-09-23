"""
PR 4 av panelgenomgången — allokering och exits.

- Allokeraren kände bara till masterguidens sju strategier. Wolf, Viking,
  Ember, Alpha, Quality och Deep Contrarian hade varken sleeve eller
  positionstak, så deras positioner kunde inte kontrolleras alls.
- Momentum-playbooken sa "12–20 %" utan bas, medan allokerarens swing-tak är
  6 % av totalen.
- Alpha-motorn satte stoppen på 2 × ATR och sålde allt vid EMA200-brott,
  medan playbooken säger stop = EMA200, brott = minska 50 %, röd regim =
  sälj resten. cycle_min fanns som parameter men lästes aldrig.
- Tiggres "0,8–1,0× NAV = resten" och "−40 % = omvärdera" var bara text.
- Scorecard läste inte Durrett/Confidence-arket, och tog positionstaket per
  sleeve i stället för per strategi (tiggre/sprott delar sleeve, inte tak).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allocator as a
import strategy_rules as sr
import tiggre as t
from strategies import alpha


# ── Allokeraren ──────────────────────────────────────────────────────────────
def test_every_playbook_has_a_sleeve_and_a_position_rule():
    for key in sr.PLAYBOOKS:
        assert key in a.STRATEGY_SLEEVE, f"{key} saknar sleeve"
        assert a.STRATEGY_SLEEVE[key] in a.SLEEVE_BY_KEY, key
        rule = a.rule_for_strategy(key)
        assert rule is not None, f"{key} saknar positionsregel"
        assert rule.sleeve == a.STRATEGY_SLEEVE[key], key


def test_the_long_term_sleeve_is_a_cap_not_a_target():
    """Alpha/Quality/Deep Contrarian ligger utanför masterguidens fördelning:
    mål 0 (summan är fortfarande 100), ingen undre ram, tak 10 % per aktie."""
    s = a.SLEEVE_BY_KEY["langsiktigt"]
    assert (s.target, s.lo, s.position_cap, s.commodity) == (0, 0, 10.0, False)
    assert sum(x.target for x in a.SLEEVES) == 100
    assert a.sleeve_status("langsiktigt", 0)[0] == "inom ram"
    assert a.sleeve_status("langsiktigt", 35)[0] == "över ram"
    # nytt kapital styrs aldrig hit av modellen
    v = {x.key: 0.0 for x in a.SLEEVES}
    v["royalty"] = 100.0
    assert a.next_capital_target(v).key != "langsiktigt"


def test_panel_swing_strategies_share_the_momentum_cap():
    for key in ("wolf", "viking", "ember", "momentum"):
        r = a.rule_for_strategy(key)
        assert r.sleeve == "swing" and r.hard_cap == 6.0, key
    assert a.rule_for_strategy("momentum").key == "swing"


def test_an_old_swing_position_is_still_momentum_swing():
    """Swing-delen fick fler regler men en gammal rad utan typ gissas inte
    fel: masterguidens swing ÄR Momentum-swing."""
    assert a.position_rule({"sleeve": "swing"}).key == "swing"
    assert a.unresolved_positions([{"sleeve": "swing"}]) == []
    assert a.position_rule({"rule": "wolf"}).key == "wolf"
    assert a.position_rule({"rule": "momentum"}).key == "swing"


def test_a_wolf_position_over_six_percent_breaches():
    pos = [{"ticker": "W", "rule": "wolf", "sleeve": "swing", "value": 7.0}]
    b = a.position_breaches(pos, total=100.0)
    assert [x["ticker"] for x in b] == ["W"] and b[0]["cap"] == 6.0
    pos[0]["value"] = 5.0
    assert a.position_breaches(pos, total=100.0) == []


def test_an_alpha_position_over_ten_percent_breaches():
    pos = [{"ticker": "A", "rule": "alpha", "sleeve": "langsiktigt", "value": 11.0}]
    assert a.position_breaches(pos, total=100.0)[0]["cap"] == 10.0
    old = {"ticker": "L", "sleeve": "langsiktigt", "value": 5.0}
    assert a.position_rule(old) is None                      # typ ska väljas
    assert a.unresolved_positions([old]) == [old]


def test_momentum_sizing_fits_under_the_allocator_cap():
    """12–20 % av swing-delen (mål 20 %) är 2,4–4 % av totalen — under 6 %."""
    pb = sr.PLAYBOOKS["momentum"]
    assert "av swing-kapitalet" in pb.risk.position_size
    assert "6 %" in pb.risk.position_size
    sleeve = a.SLEEVE_BY_KEY["swing"]
    cap = a.rule_for_strategy("momentum").hard_cap
    assert 0.20 * sleeve.target <= cap


def test_scorecard_cap_is_per_strategy_not_per_sleeve():
    import scorecard as sc
    assert sc._strategy_cap("tiggre") == 4.0
    assert sc._strategy_cap("sprott") == 1.5
    assert sc._strategy_cap("wolf") == 6.0
    assert sc._strategy_cap("alpha") == 10.0
    assert sc._strategy_cap("producenter") == 4.0        # sleeve-namn fungerar som förut
    assert sc._strategy_cap("okänd") is None


# ── Alpha-motorn mot playbooken ──────────────────────────────────────────────
def _trend(seed: int = 4, n: int = 260, drift: float = 0.25, vol: float = 0.8) -> pd.DataFrame:
    """Seedad slumpvandring med positiv drift: alla fyra tekniska grindar
    gröna (en rent monoton serie ger RSI = NaN i motorns RSI-formel)."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(drift + vol * rng.standard_normal(n))
    return pd.DataFrame({"Open": close, "High": close + 1.0, "Low": close - 1.0,
                         "Close": close, "Volume": 1_000_000})


def test_alpha_stop_is_the_ema200_and_the_position_is_capped_at_ten_percent():
    df = _trend()
    sig = alpha.entry_fn(df)
    assert sig["signal"] in ("BUY", "STRONG BUY")
    e200 = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
    assert sig["stop_loss"] == e200
    assert sig["stop_loss"] < sig["entry_price"]
    assert "atr_mult" not in alpha.DEFAULT_PARAMS

    size = alpha.risk_fn(df, capital=1_000_000)
    assert size["stop_loss"] == e200
    assert abs(size["stop_distance"] - (size["entry_price"] - e200)) < 1e-9
    assert size["position_pct"] <= alpha.DEFAULT_PARAMS["max_position_pct"] + 1e-9
    # seed 4: stoppavståndet är stort nog att risken, inte taket, styr storleken
    assert size["position_pct"] < 0.10 - 1e-6
    assert abs(size["shares"] - int(0.015 * 1_000_000 / size["stop_distance"])) == 0


def test_alpha_position_cap_binds_when_price_hugs_the_ema200():
    """Kurs strax över EMA200 → litet stoppavstånd → 1,5 % risk skulle ge en
    jättestor position. Playbookens 10 %-tak tar över."""
    df = _trend(seed=3, drift=0.02, vol=0.3)       # nästan platt: EMA200 ≈ kurs
    size = alpha.risk_fn(df, capital=1_000_000)
    assert 0 < size["stop_distance"] < 5
    # 1,5 % risk / ett stoppavstånd på några kronor vore långt över 10 %
    assert 0.015 * 1_000_000 / size["stop_distance"] * size["entry_price"] > 0.10 * 1_000_000
    assert 0 < size["position_pct"] <= 0.10 + 1e-9


def test_alpha_entry_respects_cycle_min_when_a_regime_is_given():
    df = _trend()
    assert alpha.entry_fn(df, {"cycle_score": 3})["signal"] in ("BUY", "STRONG BUY")
    red = alpha.entry_fn(df, {"cycle_score": 0})
    assert red["signal"] not in ("BUY", "STRONG BUY")
    assert red["gates"][0]["rule"].startswith("Green regime") and not red["gates"][0]["passed"]


def test_alpha_ema200_breach_reduces_half_and_red_regime_sells_the_rest():
    df = _trend()
    e200 = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
    below = df.copy()
    below.loc[below.index[-1], "Close"] = e200 * 0.98
    pos = {"entry_price": 120.0}
    first = alpha.exit_fn(pos, below)
    assert first["exit"] is False and first["reduce"] == 0.5
    assert first["reason"] == "EMA200_BREACH"
    held = alpha.exit_fn({**pos, "reduced": True}, below)
    assert held["exit"] is False and held["reduce"] == 0.0
    assert held["reason"] == "EMA200_BREACH_HELD"
    red = alpha.exit_fn({**pos, "reduced": True, "cycle_score": 0}, below)
    assert red["exit"] is True and red["reason"] == "REGIME_RED"
    # över EMA200 i grön regim: ingenting
    calm = alpha.exit_fn({**pos, "cycle_score": 2}, df)
    assert calm["exit"] is False and calm["reduce"] == 0.0 and calm["reason"] is None


def test_alpha_explicit_hard_stop_is_still_a_full_exit():
    df = _trend()
    price = float(df["Close"].iloc[-1])
    out = alpha.exit_fn({"entry_price": price * 0.9, "stop_loss": price + 1}, df)
    assert out["exit"] is True and out["reason"] == "STOP_LOSS"


def test_alpha_playbook_and_overview_say_the_same_exit():
    pb = sr.PLAYBOOKS["alpha"]
    assert "EMA200" in pb.risk.stop and "50 %" in pb.risk.stop
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tabs", "strategy_overview.py"), encoding="utf-8").read()
    assert "reduce 50%" in src and "sell the rest" in src


# ── Tiggre: säljreglerna som signaler ────────────────────────────────────────
def test_nav_exit_stage_thresholds():
    assert t.nav_exit_stage(0.79) is None
    assert t.nav_exit_stage(0.8) == "etapp"
    assert t.nav_exit_stage(0.99) == "etapp"
    assert t.nav_exit_stage(1.0) == "klar"
    assert t.nav_exit_stage(None) is None
    assert (t.NAV_EXIT_LO, t.NAV_EXIT_HI) == (0.8, 1.0)


def test_drawdown_review_at_minus_forty():
    assert t.REVIEW_DRAWDOWN_PCT == 40.0
    assert t.drawdown_review(10.0, 6.0)
    assert t.drawdown_review(10.0, 5.0)
    assert not t.drawdown_review(10.0, 6.01)
    assert not t.drawdown_review(0, 5.0) and not t.drawdown_review(10.0, None)


def test_position_signals_prefer_fresh_numbers_and_rank_sell_all_first():
    pos = {"entry": 10.0, "current": 12.0, "mcap": 200.0, "nav": 500.0,
           "half_sold": False, "triggers": {}, "catalysts": []}
    assert t.position_signals(pos) == []
    # färskt börsvärde lyfter P/NAV över 0,8 → etappförsäljning
    assert [k for k, _ in t.position_signals(pos, mcap_now=420.0)] == ["nav_exit"]
    # färsk kurs −40 % → omvärdera
    assert [k for k, _ in t.position_signals(pos, price_now=6.0)] == ["drawdown"]
    # free ride på färsk kurs
    assert [k for k, _ in t.position_signals(pos, price_now=21.0)] == ["free_ride"]
    # sälj allt går alltid först
    hot = dict(pos, triggers={"key_person": True})
    kinds = [k for k, _ in t.position_signals(hot, mcap_now=600.0, price_now=21.0)]
    assert kinds[0] == "sell_all" and set(kinds) == {"sell_all", "free_ride", "nav_exit"}


def test_refresh_job_emits_the_drawdown_event():
    import sheets_refresh as srf
    sheets = {"tiggre": {"positions": [{"id": "p9", "ticker": "DD", "entry": 10.0,
                                        "current": 9.0, "nav": 300.0}],
                         "candidates": []}}
    rows = {"tiggre:p9": {"price": 5.5, "mcap_musd": 50.0}}
    kinds = {e["kind"] for e in srf.build_events(sheets, rows)}
    assert "tiggre_drawdown" in kinds and "tiggre_free_ride" not in kinds


# ── Scorecard läser Durrett/Confidence-arket ─────────────────────────────────
def test_confidence_rows_flatten_the_store_and_collect_merges_them():
    import scorecard as sc
    store = {"companies": {"GPR": {"ticker": "GPR", "name": "Gold Prod", "stage": "producer",
                                   "commodity": "gold", "fields": {"market_cap_musd": {"value": 1}}},
                           "": {"ticker": "", "name": "no ticker"}}}
    rows = sc.confidence_rows(store)
    assert [r["ticker"] for r in rows] == ["GPR"]
    assert rows[0]["stage"] == "producer" and "fields" not in rows[0]
    assert sc.confidence_rows({}) == [] and sc.confidence_rows({"companies": []}) == []
    entries = sc.collect({"durrett": [{"ticker": "GPR", "name": "Gold Prod"}],
                          "confidence": rows})
    assert len(entries) == 1 and entries[0]["strategies"] == ["durrett", "confidence"]


def test_engine_summary_is_none_for_an_unknown_ticker():
    import scorecard as sc
    assert sc.engine_summary("XXXX-NOT-THERE") is None
