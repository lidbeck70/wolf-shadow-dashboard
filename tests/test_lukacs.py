"""Lukacs FV-modulen mot Masterguiden 4.1, kapitlet Kontrollsystemen."""

import pytest
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


def test_a_missing_price_is_a_gap_not_a_crash():
    """Fair value för Base finns men kursen saknas (t.ex. efter att ChatGPT-svaret fyllt FV men inte kursen):
    grinden säger vad som fattas i stället för att formatera None — Scorecard-sidan föll på det."""
    row = _eqx()
    row.pop("aktuell_kurs", None)
    assert fv.evaluate(row)["fv_base"] is not None and fv.evaluate(row)["mos"] is None
    assert fv.gate_gaps(row, 5.0, "producenter") == [fv.MOS_NO_PRICE]
    assert not fv.gate_ok(row, 5.0, "producenter")                      # grinden är fortfarande inte grön


def test_a_non_positive_base_value_is_a_gap_not_a_crash():
    row = _eqx()
    row["fv"][ctl.BASE]["forward_fcf_musd"] = -50.0                    # negativt FCF → fair value under noll
    gaps = fv.gate_gaps(row, 5.0, "producenter")
    assert fv.MOS_NO_VALUE in gaps and not fv.gate_ok(row, 5.0, "producenter")


def test_the_scorecard_gate_survives_fv_without_a_price():
    """Samma väg som felet: scorecard.gate_state → valuation_gaps → lukacs.gate_gaps."""
    import scorecard as sc
    row = _eqx()
    row.pop("aktuell_kurs", None)
    entry = {"key": "x|X", "ticker": "X", "name": "X", "strategies": ["producenter"], "rows": {"producenter": row}}
    state = sc.gate_state({"strategy": "producenter", "position_pct_total": 5.0}, entry)
    assert not state["ready"] and any("Aktuell kurs saknas" in g for g in sc.valuation_gaps(entry, {
        "strategy": "producenter", "position_pct_total": 5.0}))


# ── Rimlighetskontrollerna (Vår Energi: FCF 12 MUSD, kurs i NOK) ─────────────
def _var_energi(**kw):
    row = {"fcf_kvalitet": "C", "framtida_antal_aktier": 2744.8, "aktuell_kurs": 49.53, "what_must_go_right": "x",
           "ev_musd": 15051.4,
           "fv": {"Base": {"forward_fcf_musd": 12.0, "target_yield": 12.0},
                  "Bear": {"forward_fcf_musd": 14.0, "target_yield": 14.0},
                  "Bull": {"forward_fcf_musd": 10.0, "target_yield": 10.0}},
           "csm": {"Base": {"fcf_musd": 3550.0}, "Bear": {"fcf_musd": 2781.0}, "Bull": {"fcf_musd": 4165.0}}}
    row.update(kw)
    return row


def test_fcf_in_the_wrong_unit_is_flagged_against_csm():
    w = fv.fcf_unit_warnings(_var_energi())
    assert len(w) == 3 and "Base: forward FCF 12 MUSD mot CSM:s Base-FCF 3 550 MUSD" in " ".join(w)
    sane = (("Base", 3550, 12), ("Bear", 2781, 14), ("Bull", 4165, 10))
    ok = _var_energi(fv={s: {"forward_fcf_musd": f, "target_yield": y} for s, f, y in sane})
    assert fv.fcf_unit_warnings(ok) == []


def test_fcf_unit_falls_back_to_borsdata_fcf_and_then_ev():
    row = _var_energi(csm={})
    assert "Börsdatas FCF (12 mån) 3 000 MUSD" in fv.fcf_unit_warnings(row, {"fcf_musd": 3000.0})[0]
    assert "under 0,2 % av EV 15 051 MUSD" in fv.fcf_unit_warnings(row)[0]
    assert fv.fcf_unit_warnings({"fv": {"Base": {"forward_fcf_musd": 12}}}) == []        # ingen referens — tyst


def test_a_local_currency_price_is_flagged_and_converted():
    ref = {"price": 49.53, "currency": "NOK", "fx_to_usd": 0.095}
    c = fv.price_currency(49.53, ref)
    assert c["currency"] == "NOK" and c["usd"] == pytest.approx(4.7054) and c["stored_is_local"]
    assert not fv.price_currency(4.7, ref)["stored_is_local"]
    assert fv.price_currency(49.53, {"price": 49.53, "currency": "USD", "fx_to_usd": 1.0}) is None
    assert fv.price_currency(49.53, None) is None


def test_the_checks_change_no_calculation():
    """Varningarna är bara text: fair value, marginal och grind räknas som förut."""
    row = _var_energi()
    before = fv.evaluate(row)
    fv.fcf_unit_warnings(row)
    fv.price_currency(row["aktuell_kurs"], {"price": 49.53, "currency": "NOK", "fx_to_usd": 0.095})
    assert fv.evaluate(row) == before


def producer_refresh_row(ccy: str = "NOK", price: float = 49.53, market_cap: float = 122000.0,
                         ev: float = 158436.0) -> dict:
    """Rick Rule-raden som det riktiga nattjobbet (sheets_refresh.refresh) bygger den — utan fx_to_usd och
    fcf_musd, som bara Durrett-arkets rader får. Inget nätverk: Börsdata är testernas fejk."""
    import sheets_refresh as sr
    from tests.test_sheets_refresh import _API, _sheets

    class _Api(_API):
        def __init__(self):
            super().__init__()
            self.prices[40] = price
            self.snaps[40] = {"ev_ebitda": 3.0, "net_debt_ebitda": 0.8, "market_cap": market_cap, "ev": ev}

        def get_instruments(self):
            return [dict(i, stockPriceCurrency=ccy, reportCurrency="USD") if i["insId"] == 40 else i
                    for i in super().get_instruments()]

    sheets = _sheets()
    sheets["producers"]["producers"] = [{"id": "r1", "ticker": "BOL.ST", "nd_ebitda": 0.7}]
    out = sr.refresh(_Api(), sheets)
    assert out["error"] is None, out["error"]
    return out["rows"]["producers:r1"]


def test_the_price_check_works_on_the_row_the_nightly_job_writes():
    """Granskningens fynd: producentraderna bär ingen fx_to_usd — växelkursen tas ur den fasta tabellen."""
    ref = producer_refresh_row()
    assert "fx_to_usd" not in ref and ref["currency"] == "NOK"
    c = fv.price_currency(49.53, ref)
    assert c is not None and c["usd"] == pytest.approx(4.7054) and c["fx"] == 0.095 and c["stored_is_local"]
    assert not fv.price_currency(4.7054, ref)["stored_is_local"]
    assert fv.price_currency(None, ref)["stored_is_local"] is False
    sek = fv.price_currency(300.0, producer_refresh_row("SEK", 300.0))
    assert sek["currency"] == "SEK" and sek["usd"] == pytest.approx(28.5)
    assert fv.price_currency(8.0, producer_refresh_row("USD", 8.0)) is None


def test_a_nok_price_that_has_moved_is_still_seen_as_nok():
    """Kursen har rört sig mer än 3 % sedan den skrevs in — men en NOK-kurs ligger fortfarande närmare NOK än USD."""
    ref = {"price": 49.53, "currency": "NOK"}
    assert fv.price_currency(52.0, ref)["stored_is_local"]            # +5 % — fortfarande NOK
    assert fv.price_currency(40.0, ref)["stored_is_local"]
    assert not fv.price_currency(5.2, ref)["stored_is_local"]         # en USD-kurs som rört sig
    assert not fv.price_currency(0.0, ref)["stored_is_local"]
    # Valutor nära USD (EUR 1,08) går inte att skilja från kursrörelser: bara 3 %-bandet gäller
    eur = {"price": 10.0, "currency": "EUR"}
    assert fv.price_currency(10.2, eur)["stored_is_local"] and not fv.price_currency(11.0, eur)["stored_is_local"]


def test_pence_and_unknown_currencies():
    """London i pence räknas om via GBP/100; en valuta utanför tabellen räknas inte om som om den vore USD."""
    gbx = fv.price_currency(1250.0, {"price": 1250.0, "currency": "GBX"})
    assert gbx["currency"] == "GBX" and gbx["usd"] == pytest.approx(15.875) and gbx["stored_is_local"]
    assert fv.price_currency(None, {"price": 1250.0, "currency": "GBp"})["currency"] == "GBX"
    jpy = fv.price_currency(1500.0, {"price": 1500.0, "currency": "JPY", "fx_to_usd": 1.0})
    assert jpy["usd"] is None and jpy["fx"] is None and jpy["stored_is_local"]     # tabellens 1,0-reserv litas inte på
    assert not fv.price_currency(10.0, {"price": 1500.0, "currency": "JPY"})["stored_is_local"]
    assert fv.fx_to_usd("nok") == 0.095 and fv.fx_to_usd("") is None and fv.fx_to_usd("XYZ") is None


def test_fcf_typed_in_thousands_is_flagged_against_the_company_size():
    """Utan CSM- eller Börsdata-FCF fångades bara för små tal; KUSD (1 000× för stort) syns mot EV/börsvärdet."""
    ref = producer_refresh_row()                                     # EV 15 051, börsvärde 11 590 MUSD
    kusd = _var_energi(csm={}, ev_musd=None, fv={s: {"forward_fcf_musd": f, "target_yield": y} for s, f, y in
                                                 (("Base", 3550000, 12), ("Bear", 2781000, 14), ("Bull", 4165000, 10))})
    w = fv.fcf_unit_warnings(kusd, ref)
    assert len(w) == 3 and "Base: forward FCF 3 550 000 MUSD är över 2× EV 15 051 MUSD" in " ".join(w)
    good = dict(kusd, fv={s: {"forward_fcf_musd": f, "target_yield": y} for s, f, y in
                          (("Base", 3550, 12), ("Bear", 2781, 14), ("Bull", 4165, 10))})
    assert fv.fcf_unit_warnings(good, ref) == []
    # Ett nettokassabolag med litet EV: börsvärdet är storleken, så en rimlig FCF varnas inte
    cash_rich = {"fv": {"Base": {"forward_fcf_musd": 60.0}}, "ev_musd": 10.0}
    assert fv.fcf_unit_warnings(cash_rich, {"currency": "USD", "mcap_musd": 400.0}) == []
    # Uppdateringens MUSD-tal för en valuta utanför tabellen står i fel enhet — de används inte
    assert fv.fcf_unit_warnings({"fv": {"Base": {"forward_fcf_musd": 12.0}}},
                                {"currency": "JPY", "ev_musd": 900000.0}) == []


# ── Andra granskningens fynd ────────────────────────────────────────────────
def test_a_sub_ore_price_does_not_crash_the_check():
    """0,0004 SEK × 0,095 avrundas till 0 USD — jämförelsen görs mot den oavrundade kursen (ingen division med 0)."""
    for local, ccy, p in ((0.0004, "SEK", 0.02), (0.0005, "SEK", 1.0), (0.0039, "GBX", 1.0), (0.0002, "NOK", 0.5)):
        c = fv.price_currency(p, {"price": local, "currency": ccy})
        assert c is not None and c["usd"] == 0.0 and isinstance(c["stored_is_local"], bool)


def test_todays_fx_is_preferred_and_a_wild_one_is_rejected():
    ref = {"price": 400.0, "currency": "SEK"}
    live = fv.price_currency(42.0, ref, (0.105, "2026-11-06"))
    assert live["usd"] == pytest.approx(42.0) and live["fx_src"] == "live" and live["fx_asof"] == "2026-11-06"
    assert not live["stored_is_local"]
    rough = fv.price_currency(42.0, ref)
    assert rough["usd"] == pytest.approx(38.0) and rough["fx_src"] == "tabell"
    wild = fv.price_currency(42.0, ref, (9.5, "2026-11-06"))        # inverterad kurs — datafel, tabellen gäller
    assert wild["fx_src"] == "tabell" and wild["usd"] == pytest.approx(38.0)
    jpy = fv.price_currency(None, {"price": 1500.0, "currency": "JPY"}, (0.0067, "2026-11-06"))
    assert jpy["usd"] == pytest.approx(10.05) and jpy["fx_src"] == "live"     # dagens kurs finns även utanför tabellen
    assert fv.fx_pair("NOK") == ("NOKUSD=X", 1.0) and fv.fx_pair("GBX") == ("GBPUSD=X", 100.0)
    assert fv.fx_pair("USD") is None and fv.fx_pair("") is None


def _kusd(fcf=(3550, 2781, 4165), csm=(3550, 2781, 4165), k_fv=1, k_csm=1):
    return {"fv": {s: {"forward_fcf_musd": f * k_fv, "target_yield": 12} for s, f in zip(("Base", "Bear", "Bull"), fcf)},
            "csm": {s: {"fcf_musd": c * k_csm} for s, c in zip(("Base", "Bear", "Bull"), csm)}}


def test_csm_in_thousands_is_not_trusted_as_the_reference():
    """CSM:s FCF skrivs också för hand: KUSD i både CSM och FV gav ingen varning, och omvänt pekade varningen på den
    rätta FV-siffran."""
    ref = producer_refresh_row()                                     # EV 15 051, börsvärde 11 590 MUSD
    both = fv.fcf_unit_warnings(_kusd(k_fv=1000, k_csm=1000), ref)
    assert len(both) == 6
    assert "Base: CSM:s Base-FCF 3 550 000 MUSD är över 2× EV 15 051 MUSD" in " ".join(both)
    assert "Base: forward FCF 3 550 000 MUSD är över 2× EV 15 051 MUSD" in " ".join(both)
    only_csm = fv.fcf_unit_warnings(_kusd(k_csm=1000), ref)
    assert len(only_csm) == 3 and all("CSM:s" in w and "är över 2×" in w for w in only_csm)
    only_fv = fv.fcf_unit_warnings(_kusd(k_fv=1000), ref)
    assert len(only_fv) == 3 and all("mot CSM:s" in w for w in only_fv)
    assert fv.fcf_unit_warnings(_kusd(), ref) == []


def test_fcf_in_millions_of_the_listing_currency_is_flagged():
    """MSEK/MNOK i MUSD-rutan (≈10× för stort) — samma valutaförväxling som kursen. Med CSM i MUSD syns det."""
    sek = producer_refresh_row("SEK", 330.0, market_cap=90000.0, ev=100000.0)     # 8 550 / 9 500 MUSD
    row = _kusd(fcf=(9000, 4000, 15000), csm=(855, 380, 1425))
    w = fv.fcf_unit_warnings(row, sek)
    assert len(w) == 3 and "Base: forward FCF 9 000 MUSD är 10,5× CSM:s Base-FCF 855 MUSD" in " ".join(w)
    assert "omräknat från SEK stämmer det. Står talet i miljoner SEK?" in w[0]
    assert fv.fcf_unit_warnings(_kusd(fcf=(855, 380, 1425), csm=(855, 380, 1425)), sek) == []
    # EUR (1,08) ligger för nära USD för att skilja från ett annat scenario — ingen sådan varning
    eur = dict(sek, currency="EUR")
    assert fv.fcf_unit_warnings(_kusd(fcf=(950, 420, 1500), csm=(855, 380, 1425)), eur) == []


def test_pence_rows_sizes_are_not_trusted():
    """Nattjobbet räknar om GBX med 1,0 (valutan saknas i dess tabell) — EV och börsvärde står i fel enhet."""
    gbx = producer_refresh_row("GBX", 250.0, market_cap=250000.0, ev=280000.0)
    assert fv.fcf_unit_warnings({"fv": {"Base": {"forward_fcf_musd": 350.0}}}, gbx) == []
    gbp = producer_refresh_row("GBP", 2.5, market_cap=2500.0, ev=2800.0)            # 3 175 / 3 556 MUSD
    kusd = {"fv": {"Base": {"forward_fcf_musd": 350000.0}}}
    assert "är över 2× EV 3 556 MUSD" in fv.fcf_unit_warnings(kusd, gbp)[0]


def test_a_bear_case_near_break_even_is_not_a_unit_error():
    usd = {"currency": "USD", "ev_musd": 1300.0, "mcap_musd": 1200.0}
    near = {"fv": {"Bear": {"forward_fcf_musd": 30.0}, "Base": {"forward_fcf_musd": 150.0}},
            "csm": {"Bear": {"fcf_musd": 0.5}, "Base": {"fcf_musd": 150.0}}}
    assert fv.fcf_unit_warnings(near, usd) == []
    assert fv.fcf_unit_warnings({"fv": {"Bear": {"forward_fcf_musd": 2.0}}, "csm": {"Bear": {"fcf_musd": -5.0}}},
                                usd) == []
    assert fv.fcf_unit_warnings(_var_energi(csm={}, fv={"Bear": {"forward_fcf_musd": 25.0}})) == []
    # Base och Bull i fel enhet fångas fortfarande, och Bear i tusental mot storleken
    wrong = _kusd(fcf=(150, 30, 260), csm=(150, 0.5, 260), k_fv=1000)
    assert {w.split(":")[0] for w in fv.fcf_unit_warnings(wrong, usd)} == {"Base", "Bear", "Bull"}
    # Utan storlek: en positiv referens visas aldrig som 0 MUSD
    w = fv.fcf_unit_warnings({"fv": {"Bear": {"forward_fcf_musd": 30.0}}, "csm": {"Bear": {"fcf_musd": 0.5}}})
    assert w and "CSM:s Bear-FCF 0,5 MUSD" in w[0]


# ── Tredje granskningens fynd ───────────────────────────────────────────────
def _sek_row(fcf, csm=None, shares=273.5, price=350.0):
    row = {"framtida_antal_aktier": shares, "aktuell_kurs": price,
           "fv": {s: {"forward_fcf_musd": f, "target_yield": 12} for s, f in zip(("Base", "Bear", "Bull"), fcf)}}
    if csm is not None:
        row["csm"] = {s: {"fcf_musd": c} for s, c in zip(("Base", "Bear", "Bull"), csm)}
    return row


def test_msek_fcf_is_seen_when_csm_is_in_msek_too_or_empty():
    """CSM i samma enhet (eller tom) gav ingen referens — bolagets storlek ur Börsdata, som ingen skrivit för hand,
    avslöjar en Base-FCF på 77 % av EV som omräknad från SEK blir 7 %."""
    sek = producer_refresh_row("SEK", 350.0, market_cap=95700.0, ev=110000.0)    # 9 092 / 10 450 MUSD
    for row in (_sek_row((8000, 2500, 15000), csm=(8000, 2500, 15000)), _sek_row((8000, 2500, 15000))):
        w = fv.fcf_unit_warnings(row, sek)
        assert any("Base: forward FCF 8 000 MUSD är 77 % av EV 10 450 MUSD — omräknat från SEK 7 %" in x for x in w), w
        assert fv.fcf_in_local_currency(row, sek) == "SEK"
    ok = _sek_row((760, 237.5, 1425), csm=(760, 237.5, 1425))
    assert fv.fcf_unit_warnings(ok, sek) == [] and fv.fcf_in_local_currency(ok, sek) is None
    nok = producer_refresh_row("NOK", 120.0, market_cap=240000.0, ev=260000.0)   # Hydro-likt: 22 800 / 24 700 MUSD
    assert fv.fcf_in_local_currency(_sek_row((16000, 6000, 30000)), nok) == "NOK"
    assert fv.fcf_in_local_currency(_sek_row((1520, 570, 2850)), nok) is None
    eur = dict(sek, currency="EUR")
    assert fv.fcf_in_local_currency(_sek_row((8000, 2500, 15000)), eur) is None


def test_the_local_band_does_not_flag_ordinary_model_gaps():
    """Ett ~10× gap mellan CSM (nuvarande, kapitaltungt år) och Lukacs (normaliserat) nära break-even är ingen enhet,
    och PLN (4×) ligger mitt i vanliga modellgap."""
    for c in ("SEK", "NOK"):
        ref = {"currency": c, "ev_musd": 1300.0, "mcap_musd": 1200.0}
        assert fv.fcf_unit_warnings({"fv": {"Bear": {"forward_fcf_musd": 30.0}}, "csm": {"Bear": {"fcf_musd": 3.0}}},
                                    ref) == []
    bol = {"currency": "SEK", "ev_musd": 10450.0, "mcap_musd": 9092.0}
    assert fv.fcf_unit_warnings({"fv": {"Bear": {"forward_fcf_musd": 500.0}}, "csm": {"Bear": {"fcf_musd": 50.0}}},
                                bol) == []
    kghm = {"currency": "PLN", "ev_musd": 9000.0, "mcap_musd": 8000.0}
    assert fv.fcf_unit_warnings(_kusd(fcf=(1500, 600, 2500), csm=(400, 150, 900)), kghm) == []
    # Samma modell skriven i MSEK fångas fortfarande
    sek = producer_refresh_row("SEK", 330.0, market_cap=90000.0, ev=100000.0)
    assert len(fv.fcf_unit_warnings(_kusd(fcf=(9000, 4000, 15000), csm=(855, 380, 1425)), sek)) == 3


def test_a_net_cash_producer_without_a_trusted_market_cap_is_not_flagged():
    """Pence-, Yahoo- och nya rader har inget betrott börsvärde — det uppskattas som aktier × kurs i USD i stället
    för att storleken blir bolagets lilla EV."""
    row = {"ev_musd": 80.0, "framtida_antal_aktier": 115.0, "aktuell_kurs": 3.3,
           "fv": {"Bull": {"forward_fcf_musd": 170.0}}, "csm": {"Bull": {"fcf_musd": 170.0}}}
    gbx = producer_refresh_row("GBX", 260.0, market_cap=30000.0, ev=8000.0)
    yahoo = {"ticker": "SLP.L", "price": 260.0, "currency": "GBX", "source": "yfinance"}
    for ref in (None, gbx, yahoo):
        assert fv.fcf_unit_warnings(row, ref) == [], ref
    kusd = dict(row, fv={"Bull": {"forward_fcf_musd": 170000.0}}, csm={})
    for ref in (None, gbx, yahoo):
        assert "över 2× börsvärdet (aktier × kurs)" in fv.fcf_unit_warnings(kusd, ref)[0]
    # Utan antal aktier finns ingen uppskattning — som förut mot EV
    assert fv.fcf_unit_warnings({"fv": {"Base": {"forward_fcf_musd": 60.0}}, "ev_musd": 10.0}) != []


# ── Fjärde granskningens fynd ───────────────────────────────────────────────
def test_a_hand_typed_msek_ev_does_not_hide_msek_fcf():
    """Kortets EV-ruta skrivs för hand: ett EV i MSEK (10× för stort) tystade alla MSEK-kontroller. Börsdatas EV
    går först."""
    sek = producer_refresh_row("SEK", 350.0, market_cap=95700.0, ev=110000.0)
    row = dict(_sek_row((8000, 2500, 15000), csm=(760, 237.5, 1425)), ev_musd=110000.0)
    assert len([w for w in fv.fcf_unit_warnings(row, sek) if "miljoner SEK" in w]) == 3
    assert fv.fcf_in_local_currency(row, sek) == "SEK"
    assert fv.fcf_in_local_currency(dict(_sek_row((8000, 2500, 15000)), ev_musd=110000.0), sek) == "SEK"
    ok = dict(_sek_row((760, 237.5, 1425), csm=(760, 237.5, 1425)), ev_musd=12000.0)
    assert fv.fcf_unit_warnings(ok, sek) == []
    # Utan Börsdatas EV (pence, Yahoo, ingen rad) gäller rutan som förut
    assert "under 0,2 % av EV 15 051 MUSD" in " ".join(fv.fcf_unit_warnings(_var_energi(csm={})))


def test_a_genuinely_cheap_nordic_producer_is_not_told_to_divide_its_fcf():
    """OKEA-likt: P/FCF ≈ 2,5 — Base-FCF 45 % av EV i MUSD, men under EBITDA. Börsdatas EV/EBITDA skiljer det från
    MNOK (≈10× över EBITDA)."""
    ref = dict(producer_refresh_row("NOK", 20.0, market_cap=2000.0, ev=3300.0), ev_ebitda=1.2)   # EBITDA ≈ 261 MUSD
    cheap = _sek_row((140, 60, 260), csm=(140, 60, 260), shares=100.0, price=20.0)
    assert fv.fcf_unit_warnings(cheap, ref) == [] and fv.fcf_in_local_currency(cheap, ref) is None
    mnok = _sek_row((1474, 632, 2737), shares=100.0, price=20.0)                                  # samma i MNOK
    assert fv.fcf_in_local_currency(mnok, ref) == "NOK"
    # Utan EV/EBITDA gäller yield-testet ensamt (som innan)
    assert fv.fcf_in_local_currency(cheap, dict(ref, ev_ebitda=None)) == "NOK"
    bol = dict(producer_refresh_row("SEK", 350.0, market_cap=95700.0, ev=110000.0), ev_ebitda=5.5)
    assert fv.fcf_in_local_currency(_sek_row((8000, 2500, 15000)), bol) == "SEK"
