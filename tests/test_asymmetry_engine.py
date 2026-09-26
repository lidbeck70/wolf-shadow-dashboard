"""
Wolf Asymmetry, steg A: motorerna mot de syntetiska arketyperna.

Nio fall: producent, belånad producent, utvecklare, explorer, övervärderad
producent, tomt bolag, prisstress −20 %, capex-smäll +40 % och
confidence-justering. Alla tal är påhittade (tests/durrett_cases.py).
"""
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from asymmetry import ASYMMETRY_CONFIG as CFG, Inputs, analyze  # noqa: E402
from asymmetry import config as acfg  # noqa: E402
from asymmetry.model import pct_change, step_table  # noqa: E402
from asymmetry.stress import adjusted_upside  # noqa: E402
from confidence.reports import confidence_score  # noqa: E402

TODAY = date(2026, 9, 26)


def _explains(steps, needle):
    return any(needle in s for s in steps)


# ── 1. producent ─────────────────────────────────────────────────────────────
def test_gold_producer():
    a = analyze(dcs.gold_producer(), 75)
    assert a.leverage.score == 6 and a.leverage.metric == "FCF"
    assert a.leverage.response_pct == pytest.approx(38.71, abs=0.01)
    assert a.leverage.flags == []
    assert _explains(a.leverage.steps, "+20 % råvarupris")
    assert _explains(a.leverage.steps, "tabell:")
    assert a.safety.label == "7/8"
    assert a.safety.component("capex").not_applicable
    assert a.safety.component("price").points == 2.0
    assert a.safety.component("valuation").points == 1.0
    assert a.break_even.band == acfg.BREAK_EVEN_STRONG
    assert a.break_even.margin_pct == pytest.approx(51.67, abs=0.01)
    assert a.base_upside_pct == pytest.approx(8.67, abs=0.01)
    assert a.adjusted_upside_pct == pytest.approx(6.5, abs=0.01)
    # produktion nu räcker — annual_production ska inte bokföras som saknad
    assert "annual_production" not in a.missing
    assert a.missing == []


def test_grid_is_monotone_in_price():
    a = analyze(dcs.gold_producer())
    fcfs = [p.fcf_musd for p in a.leverage.grid]
    assert [p.price_pct for p in a.leverage.grid] == list(CFG["price_steps_pct"])
    assert fcfs == sorted(fcfs)
    assert all(p.margin_pct is not None and p.revenue_musd > 0 for p in a.leverage.grid)


# ── 2. belånad producent ─────────────────────────────────────────────────────
def test_leveraged_producer_flags_both_sides():
    a = analyze(dcs.leveraged_producer(), 75)
    assert a.leverage.score == 10
    assert a.leverage.response_pct == pytest.approx(171.43, abs=0.01)
    assert a.leverage.flags[0] == "Hög hävstång OCH hög nedsideskänslighet"
    assert any(f.startswith("Negativ EBITDA vid -20 %") for f in a.leverage.flags)
    assert any(f.startswith("Akut finansieringsbehov: kassa 30 < 50") for f in a.leverage.flags)
    assert a.safety.label == "2/8"
    assert a.safety.component("balance").points == 0.0
    assert _explains(a.safety.component("balance").steps, "ÖVERLEVER INTE")
    assert a.break_even.band == acfg.BREAK_EVEN_WEAK
    assert a.base_upside_pct < 0


# ── 3. utvecklare ────────────────────────────────────────────────────────────
def test_copper_developer():
    a = analyze(dcs.copper_developer(), 75)
    assert a.leverage.score == 6 and a.leverage.metric == "FCF"
    assert a.safety.label == "10/10"
    assert not a.safety.component("capex").not_applicable
    assert _explains(a.safety.component("capex").steps, "CapEx +50 %")
    assert a.break_even.band == acfg.BREAK_EVEN_STRONG
    assert a.break_even.margin_pct == pytest.approx(42.22, abs=0.01)
    assert a.base_upside_pct == pytest.approx(115.71, abs=0.01)
    assert a.adjusted_upside_pct == pytest.approx(86.79, abs=0.01)
    base = Inputs(dcs.copper_developer()).point(0.0)
    assert base.irr_pct == pytest.approx(32.0)
    assert _explains(base.steps, "MODELLED")
    assert a.matrix.note == ""


# ── 4. explorer: allt DATA_MISSING, inget påhittat ───────────────────────────
def test_lithium_explorer_is_data_missing():
    a = analyze(dcs.lithium_explorer(), 40)
    assert a.leverage.label == "DATA_MISSING" and a.leverage.score is None
    assert a.safety.label == "DATA_MISSING" and a.safety.total is None
    assert all(c.points is None for c in a.safety.components)
    assert a.break_even.band == "DATA_MISSING"
    assert a.base_upside_pct is None and a.adjusted_upside_pct is None
    assert all(not p.complete for p in a.matrix.cells.values())
    assert "commodity_price" in a.missing and "npv_musd" in a.missing
    assert _explains(a.leverage.steps, "DATA_MISSING")


# ── 5. övervärderad producent ────────────────────────────────────────────────
def test_overvalued_producer_keeps_quality_loses_valuation():
    good, dear = analyze(dcs.gold_producer(), 75), analyze(dcs.overvalued_producer(), 75)
    assert dear.leverage.score == good.leverage.score
    assert dear.break_even.band == good.break_even.band
    assert dear.safety.component("price").points == good.safety.component("price").points
    assert dear.safety.component("valuation").points == 0.0
    assert dear.safety.label == "6/8"
    assert dear.base_upside_pct == pytest.approx(-72.83, abs=0.01)
    assert dear.adjusted_upside_pct == pytest.approx(-54.63, abs=0.01)


# ── 6. tomt bolag ────────────────────────────────────────────────────────────
def test_missing_everything_no_fabricated_numbers():
    a = analyze(dcs.missing_everything(), None)
    assert a.leverage.score is None and a.safety.total is None
    assert a.break_even.price is None and a.break_even.margin_pct is None
    assert a.base_upside_pct is None and a.adjusted_upside_pct is None
    assert set(a.missing) >= {"commodity_price", "market_cap_musd", "npv_musd", "capex_musd"}
    names = [x.name for x in a.assumptions]
    assert "Skattesats" in names and len(names) == len(set(names))


# ── 7. prisstress −20 % ──────────────────────────────────────────────────────
def test_price_stress_minus_20():
    gpr = analyze(dcs.gold_producer())
    cell = gpr.matrix.cell(-20.0, 0.0)
    assert cell.value_musd == pytest.approx(950.0)
    assert cell.upside_pct == pytest.approx(-31.33, abs=0.01)
    # producent: capex-axeln ändrar ingenting
    assert all(gpr.matrix.cell(-20.0, cp).value_musd == pytest.approx(950.0) for cp in gpr.matrix.capex_pct)
    assert gpr.matrix.note.startswith("Producent")
    lvp = analyze(dcs.leveraged_producer())
    down = lvp.matrix.cell(-20.0, 0.0)
    assert down.ebitda_musd < 0 and down.fcf_musd == down.ebitda_musd   # ingen skatt på förlust
    assert gpr.matrix.cell(-20.0, 0.0).equity_musd > gpr.matrix.cell(-20.0, 0.0).value_musd  # nettokassa


# ── 8. capex-smäll +40 % ─────────────────────────────────────────────────────
def test_capex_blowout_plus_40():
    cdv, hcx = analyze(dcs.copper_developer()), analyze(dcs.developer_huge_capex_small_company())
    c = cdv.matrix.cell(-20.0, 40.0)
    assert c.value_musd == pytest.approx(880.0)         # 1200 − 800 × 0.4
    assert c.upside_pct == pytest.approx(23.71, abs=0.01)
    assert _explains(c.steps, "CapEx +40 %")
    h = hcx.matrix.cell(-20.0, 40.0)
    assert h.value_musd == pytest.approx(240.0)         # 1200 − 2400 × 0.4
    assert h.upside_pct < 0                              # samma projekt, större capex → nedsida
    assert hcx.matrix.cell(0.0, 0.0).upside_pct == pytest.approx(520.0)
    assert h.irr_pct is not None and h.irr_pct < hcx.matrix.cell(0.0, 0.0).irr_pct
    # utvecklarens capex-marginal: NPV 2 600 tål +50 % av 2 400
    assert hcx.safety.component("capex").points == 2.0


# ── 9. confidence-justering och tak ──────────────────────────────────────────
def test_adjusted_upside_and_confidence_caps():
    assert adjusted_upside(100.0, 75) == pytest.approx(75.0)
    assert adjusted_upside(100.0, 120) == pytest.approx(100.0)   # klipps till 0–100
    assert adjusted_upside(100.0, -5) == pytest.approx(0.0)
    assert adjusted_upside(None, 75) is None and adjusted_upside(50.0, None) is None
    assert adjusted_upside(-40.0, 50) == pytest.approx(-20.0)   # nedsida skalas också
    assert CFG["adjusted_upside"]["formula"] in __import__("asymmetry.stress", fromlist=["FORMULAS"]).FORMULAS
    # riktig confidence in: GPR ≈ 75, NUL ≈ 4 → justeringen följer
    gpr = confidence_score(dcs.gold_producer(), TODAY)
    a = analyze(dcs.gold_producer(), gpr.total)
    assert a.confidence == gpr.total
    assert a.adjusted_upside_pct == pytest.approx(a.base_upside_pct * gpr.total / 100, abs=0.01)
    nul = confidence_score(dcs.missing_everything(), TODAY)
    assert nul.total < 10 and analyze(dcs.missing_everything(), nul.total).adjusted_upside_pct is None


# ── royalty: delvis mätbart ──────────────────────────────────────────────────
def test_royalty_partial_measurability():
    a = analyze(dcs.royalty_company(), 75)
    assert a.leverage.score == 4 and a.leverage.response_pct == pytest.approx(25.0)
    assert a.safety.label == "4/6 mätbara (av 8)"
    assert a.safety.component("price").points is None
    assert a.safety.component("capex").not_applicable
    assert a.break_even.band == "DATA_MISSING"


# ── hjälpfunktioner och config ───────────────────────────────────────────────
def test_helpers():
    assert pct_change(150, 100) == pytest.approx(50.0)
    assert pct_change(50, 0) is None and pct_change(None, 10) is None and pct_change(10, -5) is None
    assert step_table(80, CFG["commodity_leverage"]["table"], default=0) == 10
    assert step_table(9, CFG["commodity_leverage"]["table"], default=0) == 0
    assert step_table(None, CFG["commodity_leverage"]["table"]) is None
    assert step_table(2, ((1, "a"), (3, "b")), default="c", reverse=True) == "b"


def test_config_tables_are_ordered_and_capped():
    m = CFG["margin_of_safety"]
    for key in ("price", "capex", "opex", "balance", "valuation"):
        floors = [f for f, _ in m[key]]
        assert floors == sorted(floors, reverse=True), key
        assert max(p for _, p in m[key]) <= m["max_per_component"], key
    lev = CFG["commodity_leverage"]
    assert [f for f, _ in lev["table"]] == sorted((f for f, _ in lev["table"]), reverse=True)
    assert max(p for _, p in lev["table"]) == lev["max"]
    assert lev["probe_pct"] in CFG["price_steps_pct"] and lev["downside_probe_pct"] in CFG["price_steps_pct"]
    assert CFG["break_even"]["strong"] > CFG["break_even"]["moderate"] > 0


def test_every_archetype_runs_and_explains():
    for name, mk in dcs.ALL.items():
        a = analyze(mk(), 60)
        assert a.leverage.steps, name
        assert all(c.steps for c in a.safety.components), name
        assert a.break_even.steps, name
        assert len(a.matrix.cells) == len(CFG["stress_matrix"]["price_pct"]) * len(CFG["stress_matrix"]["capex_pct"])
        if a.base_upside_pct is None:
            assert a.adjusted_upside_pct is None and a.missing, name
