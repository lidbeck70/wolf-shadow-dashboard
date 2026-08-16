"""Lukacs FV-modulen mot Masterguiden 4.1, kapitlet Kontrollsystemen."""

import controls as ctl
import lukacs as fv


# ── Guidens eget räkneexempel ────────────────────────────────────────────────
def test_the_eqx_base_case_from_the_spec():
    """Base: forward FCF $900M · 1 200M aktier · klass B · yield 9 %
    → FV = (900/0,09)/1200 = $8,33/aktie."""
    assert fv.fair_value_per_share(900, 9, 1200) == round(8.333333, 6)
    # samma tal via guidens formulering — FCF per aktie ÷ yield
    per_share = 900 / 1200
    assert round(per_share / 0.09, 4) == round(8.3333, 4)


def test_class_b_allows_nine_percent():
    assert fv.yield_band("B") == (8.0, 10.0)
    assert fv.yield_error("B", 9) is None


def test_class_a_stops_a_twelve_percent_yield():
    """Låsningen ÄR modulen: utan den räknar man fram önskad uppsida."""
    err = fv.yield_error("A", 12)
    assert err is not None
    assert "6–8 %" in err and "klass A" in err
    # och den godkänner bandets kanter
    assert fv.yield_error("A", 6) is None
    assert fv.yield_error("A", 8) is None
    assert fv.yield_error("A", 8.1) is not None


def test_every_band_is_the_guides():
    assert fv.yield_band("A") == (6.0, 8.0)
    assert fv.yield_band("B") == (8.0, 10.0)
    assert fv.yield_band("C") == (10.0, 14.0)
    assert fv.yield_band("D") is None


def test_class_d_is_not_valued_on_fcf():
    assert fv.yield_error("D", 9) == fv.NOT_FCF_VALUED
    ev = fv.evaluate({"fcf_kvalitet": "D", "framtida_antal_aktier": 1200,
                      "aktuell_kurs": 5,
                      "fv": {ctl.BASE: {"forward_fcf_musd": 900,
                                        "target_yield": 9}}})
    assert ev["not_fcf_valued"] is True
    assert ev["values"] == {}          # ingen fair value räknas fram
    assert ev["mos"] is None


def test_an_unfilled_yield_is_not_an_error():
    """Noll = ej ifyllt. Ett tomt fält ska inte skrika innan man börjat."""
    assert fv.yield_error("B", 0) is None
    assert fv.yield_error("B", None) is None
    assert fv.fair_value_per_share(900, 0, 1200) is None


# ── Uppsida, säkerhetsmarginal, band ─────────────────────────────────────────
def test_upside_and_margin_of_safety():
    assert fv.upside_pct(8.3333, 5) == round((8.3333 / 5 - 1) * 100, 6)
    # MoS mäts mot FV, inte mot kursen
    assert fv.margin_of_safety(10, 6) == 40.0
    assert fv.margin_of_safety(10, 12) == -20.0


def test_the_mos_bands_are_the_guides():
    assert fv.mos_band(41) == fv.MOS_VERY
    assert fv.mos_band(40) == fv.MOS_ATTRACTIVE      # "> 40" mycket attraktiv
    assert fv.mos_band(25) == fv.MOS_ATTRACTIVE
    assert fv.mos_band(24.9) == fv.MOS_WATCH
    assert fv.mos_band(10) == fv.MOS_WATCH
    assert fv.mos_band(9.9) == fv.MOS_NONE
    assert fv.mos_band(None) is None


def test_exactly_twentyfive_percent_passes_the_gate():
    """Snabbreferensen: "MoS < 25 % = inget nytt köp" — 25,0 passerar alltså.

    Samma flyttalsfälla som strömbrytarens 20 % och råvarutakets 55 %: kvoten
    får inte bli 24,999999 på vägen.
    """
    assert fv.mos_passes_gate(25.0) is True
    assert fv.mos_passes_gate(24.999) is False
    mos = fv.margin_of_safety(100, 75)               # exakt 25 % via kvoten
    assert fv.mos_passes_gate(mos) is True
    assert fv.mos_band(mos) == fv.MOS_ATTRACTIVE


# ── Sannolikheter ────────────────────────────────────────────────────────────
def test_probabilities_default_to_twenty_sixty_twenty():
    assert fv.probabilities({}) == {ctl.BEAR: 20.0, ctl.BASE: 60.0,
                                    ctl.BULL: 20.0}
    # egna vikter ignoreras tills avvikelsen är ikryssad
    assert fv.probabilities({"probs": {ctl.BASE: 90}})[ctl.BASE] == 60.0


def test_a_deviation_without_motivation_is_an_error():
    row = {"prob_deviation": True,
           "probs": {ctl.BEAR: 10, ctl.BASE: 70, ctl.BULL: 20}}
    errs = fv.probability_errors(row)
    assert any("motivering" in e.lower() for e in errs)
    row["prob_motivation"] = "Basfallet är mer sannolikt efter FID."
    assert fv.probability_errors(row) == []


def test_probabilities_must_sum_to_a_hundred():
    row = {"prob_deviation": True, "prob_motivation": "x",
           "probs": {ctl.BEAR: 10, ctl.BASE: 70, ctl.BULL: 30}}
    assert any("110" in e for e in fv.probability_errors(row))


def test_expected_value_weighs_the_scenarios():
    vals = {ctl.BEAR: 4.0, ctl.BASE: 8.0, ctl.BULL: 16.0}
    ev = fv.expected_value(vals, fv.DEFAULT_PROBS)
    assert ev == round(4 * 0.2 + 8 * 0.6 + 16 * 0.2, 6) == 8.8
    # ett saknat scenario ger inget halvfärdigt EV
    assert fv.expected_value({ctl.BASE: 8.0}, fv.DEFAULT_PROBS) is None


# ── Säljregeln ───────────────────────────────────────────────────────────────
def test_the_trim_rule_only_fires_on_holdings():
    assert fv.trim_warning(15, is_holding=True) is True
    assert fv.trim_warning(15, is_holding=False) is False
    assert fv.trim_warning(20, is_holding=True) is False      # ~20 % är gränsen
    assert fv.trim_warning(None, is_holding=True) is False


# ── Deleveraging ─────────────────────────────────────────────────────────────
def test_debt_over_one_halves_the_position_on_its_own():
    """Guiden: "skuld > 1,0 vid köp → max halv position OCH krav
    år-till-låg-skuld < 3". Halveringen hänger på skulden ensam."""
    s = fv.deleveraging_state(1.5, years_to_low_debt=2.0)
    assert s["applies"] is True
    assert s["half_position"] is True      # trots att åren klarar kravet
    assert s["years_ok"] is True
    assert s["gaps"] == []
    assert fv.max_position_pct(1.5, 4.0, 2.0) == 2.0


def test_debt_at_or_under_one_does_not_trigger():
    assert fv.deleveraging_state(1.0)["applies"] is False
    assert fv.deleveraging_state(0.4)["applies"] is False
    assert fv.deleveraging_state(None)["applies"] is False
    assert fv.max_position_pct(0.4, 4.0) == 4.0


def test_years_to_low_debt_is_a_separate_requirement():
    assert fv.deleveraging_state(2.0, 4.0)["years_ok"] is False
    assert any("4" in g for g in fv.deleveraging_state(2.0, 4.0)["gaps"])
    assert fv.deleveraging_state(2.0, 2.9)["years_ok"] is True
    # ej ifyllt är en lucka, inte ett godkännande
    unfilled = fv.deleveraging_state(2.0, 0.0)
    assert unfilled["years_ok"] is None
    assert unfilled["gaps"]


# ── Hela modulen ─────────────────────────────────────────────────────────────
def _eqx(**over):
    row = {"fcf_kvalitet": "B", "framtida_antal_aktier": 1200.0,
           "aktuell_kurs": 5.0, "nd_ebitda": 0.15,
           "what_must_go_right": "Guld $2 400 · 500 koz · capex $180M",
           "fv": {ctl.BEAR: {"forward_fcf_musd": 400, "target_yield": 9},
                  ctl.BASE: {"forward_fcf_musd": 900, "target_yield": 9},
                  ctl.BULL: {"forward_fcf_musd": 1600, "target_yield": 9}}}
    row.update(over)
    return row


def test_evaluate_runs_the_whole_chain():
    ev = fv.evaluate(_eqx())
    assert round(ev["fv_base"], 2) == 8.33
    assert round(ev["upside_base"], 1) == 66.7
    assert round(ev["mos"], 1) == 40.0            # (8,33 − 5)/8,33
    assert ev["mos_band"] == fv.MOS_ATTRACTIVE
    assert ev["probs"] == fv.DEFAULT_PROBS
    assert round(ev["expected_value"], 2) == round(
        (400 / 0.09 / 1200) * 0.2 + (900 / 0.09 / 1200) * 0.6
        + (1600 / 0.09 / 1200) * 0.2, 2)
    assert ev["yield_errors"] == [] and ev["prob_errors"] == []
    assert ev["delev"]["applies"] is False


def test_nothing_computed_is_written_back_to_the_row():
    """Beräknat lagras aldrig — en inaktuell fair value är farligare än ingen."""
    row = _eqx()
    before = {k: v for k, v in row.items()}
    fv.evaluate(row)
    assert row == before
    for key in ("fair_value", "mos", "margin_of_safety", "expected_value",
                "upside"):
        assert key not in row


# ── Köpgrindens steg 5 ───────────────────────────────────────────────────────
def test_the_gate_is_mechanical_only_where_the_module_is_required():
    assert fv.fv_required(5.0, "producenter") is True
    assert fv.fv_required(1.5, "producenter") is False      # under 2 %
    assert fv.fv_required(5.0, "insider") is False
    assert fv.fv_required(5.0, "tiggre") is False
    # där den inte krävs rapporteras inga luckor — grinden kryssas manuellt
    assert fv.gate_gaps({}, 1.5, "producenter") == []


def test_the_gate_needs_both_margin_and_what_must_go_right():
    assert fv.gate_gaps(_eqx(), 5.0, "producenter") == []
    without = fv.gate_gaps(_eqx(what_must_go_right="  "), 5.0, "producenter")
    assert without == [fv.WMGR_MISSING]
    # dyr kurs -> för tunn marginal
    thin = fv.gate_gaps(_eqx(aktuell_kurs=7.0), 5.0, "producenter")
    assert any("Säkerhetsmarginal" in g for g in thin)


def test_a_yield_outside_the_band_blocks_the_gate():
    row = _eqx(fcf_kvalitet="A")          # bandet blir 6–8, yielden är 9
    gaps = fv.gate_gaps(row, 5.0, "producenter")
    assert any("utanför klass A:s band" in g for g in gaps)


def test_an_unfillable_base_is_a_gap_not_a_pass():
    row = _eqx()
    row["fv"][ctl.BASE] = {}
    gaps = fv.gate_gaps(row, 5.0, "producenter")
    assert any("Fair value för Base" in g for g in gaps)


def test_class_d_cannot_produce_a_margin_of_safety():
    gaps = fv.gate_gaps(_eqx(fcf_kvalitet="D"), 5.0, "producenter")
    assert any(fv.NOT_FCF_VALUED in g for g in gaps)


def test_the_deleveraging_gap_reaches_the_gate():
    row = _eqx(nd_ebitda=1.8)             # år till låg skuld saknas
    gaps = fv.gate_gaps(row, 5.0, "producenter")
    assert any("År till låg skuld" in g for g in gaps)
    row["ar_till_lag_skuld"] = 2.0
    assert fv.gate_gaps(row, 5.0, "producenter") == []
