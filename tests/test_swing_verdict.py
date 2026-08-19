"""
Tester för swing_verdict.py — momentum-domen och Copilotens sex entry-regler.

Domen läser tre källor: regimljuset, screenerrankingen och swing-positionerna.
Testerna bygger kända lägen i alla tre och kontrollerar att playbookens regler
— inte några nya — avgör.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import swing_verdict as sv

TODAY = date(2026, 8, 19)          # onsdag, vecka 2026-W34


def _screener(*rows, generated="2026-08-18T08:00"):
    return {"generated": generated, "top": list(rows)}


def _row(ticker="ANOT", rank=5, setupA=True, nearHigh=False,
         price=None, dist_ma50=0.05):
    return {"ticker": ticker, "rank": rank, "setupA": setupA,
            "nearHigh": nearHigh, "price": price, "dist_ma50": dist_ma50}


def _regime(light="GRÖN"):
    return {"regime": light, "generated": "2026-08-18T08:00"}


def _swing(positions=None, aboveMA200=None):
    data = {"positions": positions or [], "market": {}}
    if aboveMA200 is not None:
        data["market"]["aboveMA200"] = aboveMA200
    return data


# ── Tickerformerna ───────────────────────────────────────────────────────────
def test_borsdata_and_yfinance_forms_match_each_other():
    """Screenern säger "ERIC B", användaren skriver "ERIC-B.ST" — samma bolag."""
    assert sv.normalize_ticker("ERIC-B.ST") == sv.normalize_ticker("ERIC B")
    assert sv.normalize_ticker("ANOT.ST") == sv.normalize_ticker("anot")
    assert sv.normalize_ticker("VOLV-B.ST") == sv.normalize_ticker("VOLV B")
    data = _screener(_row(ticker="ERIC B"))
    assert sv.screener_row("ERIC-B.ST", data) is not None
    assert sv.screener_row("NOKIA.HE", data) is None


# ── Kandidat: köpgrindarna ───────────────────────────────────────────────────
def test_green_regime_top20_and_setup_is_a_buy_candidate():
    v = sv.verdict("ANOT.ST", _screener(_row()), _regime("GRÖN"), _swing(),
                   TODAY)
    assert v["verdict"] == sv.BUY
    assert "full storlek" in v["reasons"][0]


def test_yellow_regime_halves_the_size_and_rejects_breakouts():
    """GUL: endast setup A och halv storlek — B-utbrott jagas inte."""
    pullback = sv.verdict("ANOT", _screener(_row(setupA=True)),
                          _regime("GUL"), _swing(), TODAY)
    assert pullback["verdict"] == sv.BUY and "HALV" in pullback["reasons"][0]
    breakout = sv.verdict("ANOT",
                          _screener(_row(setupA=False, nearHigh=True)),
                          _regime("GUL"), _swing(), TODAY)
    assert breakout["verdict"] == sv.WATCH
    assert "setup A" in breakout["reasons"][0]


def test_red_regime_abstains_no_matter_the_rank():
    v = sv.verdict("ANOT", _screener(_row(rank=1)), _regime("RÖD"), _swing(),
                   TODAY)
    assert v["verdict"] == sv.ABSTAIN and "RÖD" in v["reasons"][0]


def test_rank_21_to_40_is_watch_not_buyable():
    v = sv.verdict("ANOT", _screener(_row(rank=27)), _regime("GRÖN"),
                   _swing(), TODAY)
    assert v["verdict"] == sv.WATCH and "topp 20" in v["reasons"][0]


def test_top20_without_setup_is_watch():
    v = sv.verdict("ANOT", _screener(_row(setupA=False, nearHigh=False)),
                   _regime("GRÖN"), _swing(), TODAY)
    assert v["verdict"] == sv.WATCH and "ingen setup" in v["reasons"][0]


def test_absent_from_the_list_is_abstain_with_the_generated_date():
    v = sv.verdict("OKÄND", _screener(_row()), _regime("GRÖN"), _swing(),
                   TODAY)
    assert v["verdict"] == sv.ABSTAIN
    assert "2026-08-18" in v["reasons"][0]


def test_no_regime_data_is_unknown_never_a_yes():
    v = sv.verdict("ANOT", _screener(_row()), {}, _swing(), TODAY)
    assert v["verdict"] == sv.UNKNOWN


def test_the_manual_market_filter_is_the_fallback():
    """Utan regimmotorn gäller Swing-flikens MA200-knapp — sämre upplösning,
    aldrig tyst."""
    v = sv.verdict("ANOT", _screener(_row()), {}, _swing(aboveMA200=True),
                   TODAY)
    assert v["verdict"] == sv.BUY
    v = sv.verdict("ANOT", _screener(_row()), {}, _swing(aboveMA200=False),
                   TODAY)
    assert v["verdict"] == sv.ABSTAIN


# ── Innehav: säljreglerna, först inträffad gäller ────────────────────────────
def _held(entry=10.0, half=False):
    return [{"ticker": "ANOT", "entry": entry, "date": "2026-08-03",
             "halfTaken": half}]


def test_rank_exit_sells_a_holding_that_left_the_list():
    v = sv.verdict("ANOT", _screener(), _regime("GRÖN"),
                   _swing(positions=_held()), TODAY)
    assert v["held"] is True and v["verdict"] == sv.SELL
    assert "rank-exit" in v["reasons"][0]


def test_the_stop_fires_before_everything_else():
    row = _row(price=8.9, dist_ma50=-0.05)          # −11 % OCH under MA50
    v = sv.verdict("ANOT", _screener(row), _regime("GRÖN"),
                   _swing(positions=_held()), TODAY)
    assert v["verdict"] == sv.SELL and "stoppen" in v["reasons"][0]


def test_a_close_under_ma50_sells():
    row = _row(price=9.8, dist_ma50=-0.02)
    v = sv.verdict("ANOT", _screener(row), _regime("GRÖN"),
                   _swing(positions=_held()), TODAY)
    assert v["verdict"] == sv.SELL and "MA50" in v["reasons"][0]


def test_plus_twenty_percent_takes_half_once():
    row = _row(price=12.5, dist_ma50=0.08)
    v = sv.verdict("ANOT", _screener(row), _regime("GRÖN"),
                   _swing(positions=_held()), TODAY)
    assert v["verdict"] == sv.PARTIAL
    # redan tagen delvinst upprepas inte
    v = sv.verdict("ANOT", _screener(row), _regime("GRÖN"),
                   _swing(positions=_held(half=True)), TODAY)
    assert v["verdict"] == sv.HOLD


def test_a_healthy_holding_holds():
    row = _row(price=10.8, dist_ma50=0.04)
    v = sv.verdict("ANOT", _screener(row), _regime("GRÖN"),
                   _swing(positions=_held()), TODAY)
    assert v["verdict"] == sv.HOLD


def test_stale_screener_data_warns_but_does_not_invent_a_verdict():
    old = _screener(_row(), generated="2026-08-01T08:00")
    v = sv.verdict("ANOT", old, _regime("GRÖN"), _swing(), TODAY)
    assert v["verdict"] == sv.BUY
    assert any("dagar gammal" in r for r in v["reasons"])


# ── Veckotak och positionstak ────────────────────────────────────────────────
def test_weekly_buys_counts_only_this_iso_week():
    positions = [{"ticker": "A", "date": "2026-08-17"},    # måndag v34
                 {"ticker": "B", "date": "2026-08-18"},
                 {"ticker": "C", "date": "2026-08-10"},    # v33
                 {"ticker": "D", "date": "trasigt datum"}]
    assert sv.weekly_buys({"positions": positions}, TODAY) == 2


def test_rule_checks_cover_all_six_momentum_rules():
    checks = sv.rule_checks("ANOT", _screener(_row()), _regime("GRÖN"),
                            _swing(), TODAY)
    assert set(checks) == {"marknadsfilter", "ranking", "setup",
                           "köp per vecka", "positioner", "positionsstorlek"}
    assert all(status in ("PASS", "MANUAL", "FAIL")
               for status, _n in checks.values())
    assert checks["marknadsfilter"][0] == "PASS"
    assert checks["ranking"][0] == "PASS"
    assert checks["setup"][0] == "PASS"


def test_the_weekly_cap_goes_pass_manual_fail():
    def with_buys(n):
        positions = [{"ticker": f"T{i}", "date": "2026-08-18"}
                     for i in range(n)]
        return sv.rule_checks("ANOT", _screener(_row()), _regime("GRÖN"),
                              {"positions": positions, "market": {}},
                              TODAY)["köp per vecka"][0]
    assert with_buys(0) == "PASS"
    assert with_buys(1) == "MANUAL"      # nästa köp är veckans sista
    assert with_buys(2) == "FAIL"


def test_the_position_cap_fails_at_eight():
    positions = [{"ticker": f"T{i}", "date": "2026-07-01"} for i in range(8)]
    checks = sv.rule_checks("ANOT", _screener(_row()), _regime("GRÖN"),
                            {"positions": positions, "market": {}}, TODAY)
    assert checks["positioner"][0] == "FAIL"


def test_missing_data_is_manual_with_instructions_everywhere():
    checks = sv.rule_checks("ANOT", {}, {}, {"positions": [], "market": {}},
                            TODAY)
    assert checks["marknadsfilter"][0] == "MANUAL"
    assert checks["ranking"][0] == "MANUAL"
    assert checks["setup"][0] == "MANUAL"
    assert "wolf_data.py" in checks["marknadsfilter"][1]
