"""
Tester för review_link.py — granskningsarken in i Copiloten.

Det viktiga: modulen ÖVERSÄTTER arkens egna utfall, den räknar inte om dem.
Varje test bygger en rad som arket självt skulle bedöma på ett känt sätt och
kontrollerar att Copiloten säger samma sak.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import controls as ctl
import review_link as rl


def _stores(**over):
    base = {"producers": {"producers": [], "royalty": []},
            "scoring": {"sprott": [], "durrett": []},
            "tiggre": {"candidates": [], "positions": [], "closed": [],
                       "parked": []},
            "insider": {"signals": []}}
    base.update(over)
    return base


def _rule_row(**over):
    row = {"ticker": "EQX", "price": 4250.0, "unit_cost": 2175.0,
           "jurisdiktion": True, "insyn": True, "kapitaldisciplin": True,
           "mine_life": 25.0}
    row.update(over)
    return row


# ── Grunderna ────────────────────────────────────────────────────────────────
def test_only_reviewed_strategies_have_a_sheet():
    for key in ("rule", "royalty", "sprott", "durrett", "tiggre", "insider"):
        assert rl.has_review(key), key
    for key in ("momentum", "quality", "wolf", "", None):
        assert not rl.has_review(key), key
    assert rl.review("momentum", "ABB", _stores()) is None


def test_a_missing_company_points_back_to_the_sheet():
    rev = rl.review("rule", "SAKNAS", _stores())
    assert rev["found"] is False
    assert rev["status"] == rl.MANUAL
    assert "Rick Rule" in rev["note"] and "screener → granskning" in rev["note"]


def test_the_lookup_ignores_case():
    stores = _stores(producers={"producers": [_rule_row()], "royalty": []})
    assert rl.find_row("rule", "eqx", stores)["ticker"] == "EQX"


# ── Rick Rule ────────────────────────────────────────────────────────────────
def test_a_five_of_five_producer_passes():
    stores = _stores(producers={"producers": [_rule_row()], "royalty": []})
    rev = rl.review("rule", "EQX", stores)
    assert rev["status"] == rl.PASS
    assert "5/5" in rev["note"] and "Köpkandidat" in rev["note"]


def test_a_dying_asset_fails_no_matter_the_score():
    """Arkets strykregel ska nå Copiloten: 5/5 med 4 års gruva är en passa."""
    stores = _stores(producers={"producers": [_rule_row(mine_life=4.0)],
                                "royalty": []})
    rev = rl.review("rule", "EQX", stores)
    assert rev["status"] == rl.FAIL
    assert "Döende tillgång" in rev["note"]


def test_a_three_pointer_is_watch_not_a_yes():
    stores = _stores(producers={"producers": [_rule_row(insyn=False,
                                                        jurisdiktion=False)],
                                "royalty": []})
    rev = rl.review("rule", "EQX", stores)
    assert rev["status"] == rl.MANUAL and "Bevaka" in rev["note"]


def test_an_unpriced_row_is_manual_with_instructions():
    stores = _stores(producers={"producers": [{"ticker": "EQX"}],
                                "royalty": []})
    rev = rl.review("rule", "EQX", stores)
    assert rev["status"] == rl.MANUAL
    assert "råvarupris" in rev["note"]


# ── Royalty C ────────────────────────────────────────────────────────────────
def test_royalty_buy_signal_passes_and_geo_warning_fails():
    buy = {"ticker": "FNV", "pnav_now": 1.0, "pnav_bottom": 0.95,
           "ev_now": 10.0, "ev_median": 12.0, "geo_now": 1.2, "geo_3y": 1.0}
    shrink = {"ticker": "SAND", "geo_now": 0.8, "geo_3y": 1.0}
    stores = _stores(producers={"producers": [], "royalty": [buy, shrink]})
    assert rl.review("royalty", "FNV", stores)["status"] == rl.PASS
    rev = rl.review("royalty", "SAND", stores)
    assert rev["status"] == rl.FAIL and "krymper" in rev["note"]


# ── Poängmodellen ────────────────────────────────────────────────────────────
def test_scoring_bands_translate_to_the_three_statuses():
    def row(pts):
        import scoring
        return {"ticker": "T",
                "factors": {f.key: pts for f in scoring.FACTORS}}
    for pts, want in ((2, rl.PASS), (0, rl.FAIL)):
        stores = _stores(scoring={"sprott": [row(pts)], "durrett": []})
        assert rl.review("sprott", "T", stores)["status"] == want, pts
    stores = _stores(scoring={"sprott": [],
                              "durrett": [{"ticker": "T", "factors": {}}]})
    rev = rl.review("durrett", "T", stores)
    assert rev["status"] == rl.MANUAL and "poängsatt" in rev["note"].lower()


# ── Tiggre ───────────────────────────────────────────────────────────────────
def test_tiggre_needs_every_gate_green():
    """Arkets egen regel: köp ENDAST när alla grindar är gröna."""
    incomplete = {"ticker": "LOB", "screen": {}, "factors": {},
                  "catalysts": []}
    stores = _stores(tiggre={"candidates": [incomplete], "positions": [],
                             "closed": [], "parked": []})
    rev = rl.review("tiggre", "LOB", stores)
    assert rev["status"] == rl.FAIL
    assert "Röda köpgrindar" in rev["note"]


def test_tiggre_finds_a_held_position_too():
    stores = _stores(tiggre={"candidates": [], "closed": [], "parked": [],
                             "positions": [{"ticker": "LOB"}]})
    assert rl.find_row("tiggre", "LOB", stores) is not None


# ── Insiderbevakaren ─────────────────────────────────────────────────────────
def test_insider_status_flow_translates():
    import insider as insider_mod
    buy = {"ticker": "IND", "insiders": 3, "role": insider_mod.ROLE_TOP,
           "amount": 1.5, "okar_25": True, "efter_fall": True,
           "aterkommande": True, "gate": insider_mod.GATE_YES, "trigger": "A"}
    stores = _stores(insider={"signals": [buy]})
    rev = rl.review("insider", "IND", stores)
    assert rev["status"] == rl.PASS and "KÖP" in rev["note"]

    noise = {"ticker": "BRUS", "insiders": 1,
             "role": insider_mod.ROLE_OTHER, "amount": 0.1}
    stores = _stores(insider={"signals": [noise]})
    assert rl.review("insider", "BRUS", stores)["status"] == rl.FAIL

    incomplete = {"ticker": "TOM", "insiders": 2}
    stores = _stores(insider={"signals": [incomplete]})
    rev = rl.review("insider", "TOM", stores)
    assert rev["status"] == rl.MANUAL and "Ofullständig" in rev["note"]


# ── Kontrollerna ─────────────────────────────────────────────────────────────
def test_the_ds_lock_is_a_hard_fail():
    row = _rule_row(**{f.key: 2 for f in ctl.DS_FIELDS})     # DS 10 — extrem
    findings = rl.control_findings(row, "rule")
    ds = [f for f in findings if f[1] == "DS"][0]
    assert ds[0] == rl.FAIL and "låst" in ds[2]


def test_a_financing_catalyst_downgrades_the_lock_to_a_note():
    row = _rule_row(**{f.key: 2 for f in ctl.DS_FIELDS},
                    fin_catalyst_text="Riktad emission",
                    fin_catalyst_date="2026-10")
    ds = [f for f in rl.control_findings(row, "rule") if f[1] == "DS"][0]
    assert ds[0] == rl.PASS or ds[0] == rl.FAIL  # låset släpper -> inte FAIL
    assert ds[0] == rl.PASS


def test_unassessed_controls_are_manual_only_when_required():
    """Proportionalitetsregeln: under 2 % krävs inte AQS/CSM."""
    small = _rule_row(position_pct=1.0)
    labels = [f[1] for f in rl.control_findings(small, "rule")]
    assert "AQS" not in labels and "CSM" not in labels
    assert "DS" in labels                       # DS krävs alltid för rule

    big = _rule_row(position_pct=4.0)
    findings = {f[1]: f[0] for f in rl.control_findings(big, "rule")}
    assert findings["AQS"] == rl.MANUAL and findings["CSM"] == rl.MANUAL


def test_a_csm_bear_flag_fails():
    row = _rule_row(position_pct=4.0, csm_kind=ctl.PRODUCER,
                    csm={ctl.BEAR: {"price": 50, "fcf_musd": -20},
                         ctl.BASE: {"price": 80, "fcf_musd": 10},
                         ctl.BULL: {"price": 120, "fcf_musd": 40}})
    findings = {f[1]: (f[0], f[2]) for f in rl.control_findings(row, "rule")}
    assert findings["CSM"][0] == rl.FAIL
    assert ctl.CSM_BEAR_FAIL in findings["CSM"][1]


# ── Prompten ─────────────────────────────────────────────────────────────────
def test_prompt_lines_carry_sheet_verdict_and_controls():
    stores = _stores(producers={"producers": [_rule_row(position_pct=4.0)],
                                "royalty": []})
    lines = rl.prompt_lines(rl.review("rule", "EQX", stores))
    assert any("Rick Rule" in ln and "PASS" in ln for ln in lines)
    assert any(ln.strip().startswith("AQS") for ln in lines)


def test_prompt_lines_for_a_missing_company_say_so():
    lines = rl.prompt_lines(rl.review("rule", "SAKNAS", _stores()))
    assert len(lines) == 1 and "SAKNAS" in lines[0]
    assert rl.prompt_lines(None) == []


# ── Underlaget — komponenterna, inte bara summan ─────────────────────────────
def test_detail_lines_carry_the_sheets_answered_fields():
    """Utan komponenterna bad modellen användaren kontrollera landrisk,
    kostnadsposition, ledning och kapitaldisciplin — de fyra frågor som
    redan VAR besvarade i arket. Summan utan delarna bjuder in frågan."""
    row = _rule_row(ev_ebitda=4.2, nd_ebitda=0.15, position_pct=4.0)
    lines = "\n".join(rl.detail_lines("rule", row))
    assert "marginal 48.8 %" in lines
    assert "jurisdiktion Ja" in lines and "insynsägande Ja" in lines
    assert "gruvlivslängd 25 år" in lines
    assert "EV/EBITDA 4.2" in lines


def test_detail_lines_name_the_weak_aqs_factors():
    row = _rule_row(aqs_kostnad=1, aqs_livslangd=2, aqs_metallurgi=1,
                    aqs_capex=1, aqs_jurisdiktion=1, aqs_infrastruktur=0,
                    aqs_management=0, aqs_expansion=0)
    lines = "\n".join(rl.detail_lines("rule", row))
    assert "AQS svagast (0 p)" in lines
    assert "Infrastruktur / tillstånd" in lines


def test_detail_lines_include_lukacs_fv_when_filled():
    row = _rule_row(fcf_kvalitet="B", framtida_antal_aktier=1200.0,
                    aktuell_kurs=5.0,
                    fv={"Base": {"forward_fcf_musd": 900, "target_yield": 9}})
    lines = "\n".join(rl.detail_lines("rule", row))
    assert "fair value Base 8.33/aktie" in lines
    assert "säkerhetsmarginal 40 %" in lines


def test_prompt_lines_append_the_details():
    stores = _stores(producers={"producers": [_rule_row()], "royalty": []})
    lines = rl.prompt_lines(rl.review("rule", "EQX", stores), "rule")
    assert any("Disciplinfrågorna" in ln for ln in lines)
    # och strategin kan härledas ur arknamnet när den inte skickas
    lines2 = rl.prompt_lines(rl.review("rule", "EQX", stores))
    assert any("Disciplinfrågorna" in ln for ln in lines2)


def test_unfilled_fields_show_as_dashes_not_zeros():
    """R/P är olje- och gasmåttet — orört ska det stå som streck, inte 0."""
    lines = "\n".join(rl.detail_lines("rule", _rule_row(rp_ratio=0.0)))
    assert "R/P –" in lines
