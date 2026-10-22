"""
Thor fas 7 — playbooks för ⚔️ Viking Nine och 🪓 BERSERK i strategy_rules (bara tillägg).

Talen i texten hämtas ur motorerna, så det här testet låser dem mot viking_execution, ovtlyr_nine och
berserk.backtest/signals — samma anti-driftvakt som Wolf och Ember har. Dessutom: registret, allokeraren,
Thor och FLIKGUIDEN pekar på de nya playbookerna, och Cykel-grinden blir MANUAL för dem (inga Market Cycle-
faser nämns, så inget nytt regelutfall uppstår).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import strategy_rules as sr  # noqa: E402
from test_strategy_rules import _pct_in  # noqa: E402


def _text(pb, groups=("entry", "exit")) -> str:
    return " ".join(r.text + " " + r.explanation for g in groups for r in getattr(pb, g))


def test_viking_nine_playbook_tracks_the_engines():
    import ovtlyr_nine as on
    import viking_execution as vx
    pb = sr.PLAYBOOKS["viking_nine"]
    assert vx.MAX_RISK_PCT in _pct_in(pb.risk.risk_per_trade)
    assert f"{vx.ATR_STOP_MULT:g} × ATR".replace(".", ",") in pb.risk.stop
    assert vx.MAX_POSITION_PCT in _pct_in(pb.risk.position_size)
    entry = _text(pb, ("entry",))
    assert f"relativ volym ≥ {vx.RELATIVE_VOLUME_MIN:.2f}×" in entry
    assert f"högst {vx.MAX_CHASE_PCT:g} %".replace(".", ",") in entry and f"R/R ≥ {vx.MINIMUM_RR:g}" in entry
    assert f"Ingen rapport inom {vx.EARNINGS_BUFFER_DAYS} handelsdagar" in entry
    assert f"max {vx.MAX_DAILY_LOSSES} förlustaffärer per dag" in entry
    assert "Nine 9/9 är setupen" in entry and f"{vx.NINE_DEVELOPING}–8/9 är DEVELOPING" in entry
    assert f"under {on.BREADTH_LOW:.0f} bara efter" in entry and f"över {on.BREADTH_HIGH:.0f} och nedvänd" in entry
    assert f"F&G under {on.FG_MAX:.0f}" in entry and f"äldre än {on.STALE_BDAYS} handelsdagar" in entry
    exits = _text(pb, ("exit",))
    assert "stäng allt" in exits and "EMA10 som trailing stop" in exits and "0–50 → exit vid 63" in exits
    assert pb.support == sr.SUPPORT_FULL and "⚔️ Viking Nine" in pb.where and "⚔️ Viking Nine Regime" in pb.where
    assert sum(r.hard for r in pb.entry + pb.exit) >= 5 and len(pb.entry) == 10 and len(pb.exit) == 10


def test_berserk_playbook_tracks_the_engines():
    from berserk import backtest as bb
    from berserk import signals as sg
    pb = sr.PLAYBOOKS["berserk"]
    risk = pb.risk.risk_per_trade
    assert bb.RISK_BY_SETUP[sg.S1] in _pct_in(risk) and bb.RISK_BY_SETUP[sg.S3] in _pct_in(risk)
    for s in sg.SETUPS:
        assert f"{bb.STOP_ATR[s]:g} × ATR14".replace(".", ",") in pb.risk.stop
    cfg, port = bb.live_config(), bb.live_portfolio()
    entry = _text(pb, ("entry",))
    assert (f"{sg.S3} — av i live" in entry) == (sg.S3 not in cfg.setups)
    assert f"gappar mer än {cfg.max_gap_atr:g} × ATR".replace(".", ",") in entry
    assert f"Likviditet ≥ {bb.MIN_TURNOVER_M:g} miljoner" in entry
    assert f"{', '.join(cfg.commodity_free)} är fri" in entry
    assert f"minst {abs(sg.DIV_LAG) * 100:.0f} procentenheter" in entry and f"relativ volym ≥ {sg.RVOL_S1:g}×".replace(".", ",") in entry
    assert f"minst {abs(sg.BEAR_DD) * 100:.0f} % under 52-veckorshögsta" in entry
    assert f"minst {abs(sg.HATED_DD) * 100:.0f} % under sin topp" in entry and f"(ETF {abs(sg.HATED_DD_ETF) * 100:.0f} %)" in entry
    assert f"relativ volym ≥ {sg.RVOL_S2:g}×".replace(".", ",") in entry
    assert f"dagshopp över {sg.JUMP_MAX * 100:.0f} %" in entry and f"spärrar {sg.JUMP_BLOCK_DAYS} dagar" in entry
    assert f"{port.max_positions} positioner" in pb.risk.max_positions and port.max_heat_pct in _pct_in(pb.risk.max_positions)
    assert port.max_position_pct in _pct_in(pb.risk.position_size)
    exits = _text(pb, ("exit",))
    assert "råvaran stänger under SMA50" in exits and "råvaran stänger under EMA50" in exits and "tidsstopp 15 dagar" in exits
    assert f"Max {port.max_daily_losses} förlustaffärer per dag" in exits
    assert pb.support == sr.SUPPORT_FULL and "🪓 BERSERK" in pb.where and "🪓 BERSERK Regime" in pb.where


def test_registry_allocator_thor_and_guide_point_at_the_new_playbooks():
    import allocator as a
    import cycle as cy
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    from thor import registry as reg
    assert reg.get("viking_nine").playbook == "viking_nine" and reg.get("berserk").playbook == "berserk"
    for key in ("viking_nine", "berserk"):
        assert a.STRATEGY_SLEEVE[key] == "swing" and a.rule_for_strategy(key).hard_cap == 6.0
        assert cy.playbook_phases(key) == {"buy": set(), "hold": set(), "sell": set()}        # inga cykelfaser → grinden MANUAL
        assert cy.gate_from_market_phase({"phase": "HOPE", "confidence": 60.0}, key)[0] == "MANUAL"
    order = list(sr.LEARNING_ORDER)
    assert order.index("viking_nine") == order.index("viking") + 1 and order.index("berserk") == order.index("ember") + 1
    tabs = [t for t, _r, _u in _PANEL_GUIDE]
    assert "THOR" in tabs and "ODINS VISDOM → Råvaror → 🪓 BERSERK Regime" in tabs
    # de gamla playbookerna är orörda
    assert len(sr.PLAYBOOKS["viking"].entry) == 10 and len(sr.PLAYBOOKS["ember"].entry) == 4
