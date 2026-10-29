"""
review/missing — "Saknas för köpbeslut" härleds ur arkens egna grindar.

En rad som klarar alla grindar har inga luckor; en tom rad listar exakt de indata grindarna behöver. En grind som
går att pröva men faller är ett svar, inte en lucka. Trösklarna läses ur arken — flyttas en gräns följer listan.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import controls as ctl  # noqa: E402
import insider as ins  # noqa: E402
import scorecard as sc  # noqa: E402
import scoring as sco  # noqa: E402
import tiggre as tig  # noqa: E402
from asymmetry import fields as afields  # noqa: E402
from confidence.data import validation as cval  # noqa: E402
from review import catalog as cat  # noqa: E402
from review import missing as mis  # noqa: E402

DS0 = {f.key: 0 for f in ctl.DS_FIELDS}
DS_KEYS = set(DS0)
AQS2 = {f.key: 2 for f in ctl.AQS_FIELDS}
AQS_KEYS = set(AQS2)
FACTORS2 = {f.key: 2 for f in sco.FACTORS}

RULE_OK = {"id": "p1", "ticker": "BOL", "price": 3000.0, "unit_cost": 1200.0, "mine_life": 12.0, "jurisdiktion": True,
           "kapitaldisciplin": True, "insyn": False, "ev_musd": 500.0, "nav_musd": 900.0, "position_pct": 1.5, **DS0}
ROYALTY_OK = {"id": "r1", "ticker": "FNV", "pnav_now": 1.1, "pnav_bottom": 1.0, "ev_now": 8.0, "ev_median": 10.0,
              "geo_now": 0.05, "geo_3y": 0.04, "level": 1}
SPROTT_OK = {"id": "s1", "ticker": "JR", "cash": 30.0, "burn": 10.0, "factors": dict(FACTORS2), **DS0}
DURRETT_OK = {"id": "d1", "ticker": "AU", "mcap": 500.0, "profit": 100.0, "factors": dict(FACTORS2), **DS0}
_CSM_DEV = {s: {"price": 2000.0, "nav_musd": 400.0, "financing_need": 0.0} for s in ctl.SCENARIOS_3}
TIGGRE_OK = {"id": "c1", "ticker": "LOB", "screen": {"fs": True, "permits": True, "funded": False}, "mcap": 100.0,
             "nav": 500.0, "downside": -40.0, "factors": {"stadium": 2, "finansiering": 2, "manniskor": 2,
                                                          "jurisdiktion": 2},
             "catalysts": [{"name": "FID", "date": "2027-01"}, {"name": "Byggstart", "date": "2027-06"}],
             "csm": _CSM_DEV, **DS0, **AQS2}
INSIDER_OK = {"id": "i1", "ticker": "EKTA-B", "insiders": 3, "role": ins.ROLE_TOP, "amount": 1500.0,
              "gate": ins.GATE_YES, "trigger": "A", "cluster_avg": 10.0, "price_now": 11.0}


def _paths(items) -> set:
    return {i.path for i in items}


def test_rows_that_pass_every_gate_have_nothing_missing():
    assert tig.buy_gates(TIGGRE_OK) and all(ok for _l, ok, _d in tig.buy_gates(TIGGRE_OK))
    assert ins.status(INSIDER_OK) == ins.S_BUY
    for sheet, row in (("rule", RULE_OK), ("royalty", ROYALTY_OK), ("sprott", SPROTT_OK), ("durrett", DURRETT_OK),
                       ("tiggre", TIGGRE_OK), ("insider", INSIDER_OK)):
        assert mis.missing_for(sheet, row) == [], sheet
    oil = {k: v for k, v in RULE_OK.items() if k != "mine_life"}
    assert mis.missing_for("rule", dict(oil, rp_ratio=12.0)) == []     # olja/gas: R/P räcker


def test_a_ready_scorecard_card_has_nothing_missing():
    entry = sc.collect({"producenter": [dict(RULE_OK, name="Boliden")]})[0]
    card = {"strategy": "producenter", "position_pct_total": 1.5, "valuation": "4× EV/EBITDA", "trigger": "Q1",
            "sell_rule": "Stopp −15 %", **{g.key: True for g in sc.GATES}}
    assert sc.gate_state(card, entry)["ready"]
    assert mis.missing_for("scorecard", card, entry=entry) == []


def test_blank_rows_list_exactly_the_inputs_the_gates_need():
    assert _paths(mis.missing_for("rule", {})) == {"price", "unit_cost", "mine_life", "jurisdiktion",
                                                   "kapitaldisciplin", "insyn", "ev_musd", "nav_musd",
                                                   "position_pct"} | DS_KEYS
    assert _paths(mis.missing_for("royalty", {})) == {"pnav_now", "pnav_bottom", "ev_now", "ev_median", "geo_now",
                                                      "geo_3y"}
    factors = {f"factors.{f.key}" for f in sco.FACTORS}
    assert _paths(mis.missing_for("sprott", {})) == factors | {"cash", "burn"} | DS_KEYS
    assert _paths(mis.missing_for("durrett", {})) == factors | {"mcap", "profit"} | DS_KEYS
    csm = {f"csm.{s}.{k}" for s in ctl.SCENARIOS_3 for k in ("price", "nav_musd", "financing_need")}
    assert _paths(mis.missing_for("tiggre", {})) == (
        {"screen.fs", "screen.permits", "screen.funded", "mcap", "nav", "downside", "factors.stadium",
         "factors.finansiering", "factors.manniskor", "factors.jurisdiktion", "catalysts.0.name", "catalysts.0.date",
         "catalysts.1.name", "catalysts.1.date"} | DS_KEYS | AQS_KEYS | csm)
    assert _paths(mis.missing_for("insider", {})) == {"insiders", "amount", "role", "cluster_avg", "price_now"}
    conf = mis.missing_for("confidence", {"stage": "developer"})
    assert _paths(conf) == {f"fields.{k}" for k in cval._CORE_BY_STAGE["developer"]} | {""}   # "" = råvaran
    asym = mis.missing_for("asymmetry", {"stage": "producer"})
    assert _paths(asym) == {f"fields.{f.key}" for f in afields.economy("producer") + afields.confidence_core("producer")}


def test_every_missing_path_is_a_catalog_field():
    for sheet, row, kw in (("rule", {}, {}), ("rule", {"position_pct": 3.0, "nd_ebitda": 2.0}, {}), ("royalty", {}, {}),
                           ("royalty", {}, {"position_pct": 3.0}), ("sprott", {}, {}), ("durrett", {}, {}),
                           ("tiggre", {}, {}), ("insider", {"insiders": 3, "role": "VD/CFO", "amount": 2000}, {}),
                           ("scorecard", {}, {}), ("confidence", {"stage": "explorer"}, {}),
                           ("asymmetry", {"stage": "royalty"}, {})):
        fields = {f.path for f in cat.fields_for(sheet, row, kw.get("stage"), kw.get("position_pct"))}
        for it in mis.missing_for(sheet, row, **kw):
            assert it.sheet == sheet and it.reason and it.label and it.gate
            assert it.path == "" or it.path in fields, (sheet, it.path)


def test_a_gate_that_can_be_tested_and_fails_is_an_answer_not_a_gap():
    thin = dict(RULE_OK, price=100.0, unit_cost=99.0, insyn=False)       # marginalen är räknad — och tunn
    assert "price" not in _paths(mis.missing_for("rule", thin))
    tg = dict(TIGGRE_OK, screen={"fs": False, "permits": False, "funded": False}, downside=-400.0)
    assert not tig.buy_gates(tg)[0][1] and not tig.buy_gates(tg)[1][1]   # grovsållning och U/N faller …
    assert not {"screen.fs", "mcap", "nav", "downside"} & _paths(mis.missing_for("tiggre", tg))   # … men är besvarade
    assert "pnav_bottom" in _paths(mis.missing_for("royalty", dict(ROYALTY_OK, pnav_bottom=0.0)))  # noll delar inte
    assert "price" in _paths(mis.missing_for("rule", dict(RULE_OK, price=0.0)))


def test_ds_partial_and_high_ds_needs_the_financing_catalyst():
    part = dict(RULE_OK, ds_capex=None)
    part.pop("ds_capex")
    items = {i.path: i for i in mis.missing_for("rule", part)}
    assert set(items) == {"ds_capex"} and "räknas som 0" in items["ds_capex"].reason
    high = dict(RULE_OK, ds_runway=2, ds_capex=2, ds_warrants=2)
    assert ctl.ds_blocks_buy(high)
    items = {i.path: i for i in mis.missing_for("rule", high)}
    assert set(items) == {"fin_catalyst_text", "fin_catalyst_date"} and items["fin_catalyst_text"].reason == ctl.ds_note(high)
    done = dict(high, fin_catalyst_text="Riktad emission", fin_catalyst_date="2027-02")
    assert mis.missing_for("rule", done) == []


def test_lukacs_and_deleveraging_over_the_limit():
    big = dict(RULE_OK, position_pct=3.0, nd_ebitda=1.5, **AQS2,
               csm={s: {"price": 3000.0, "fcf_musd": 50.0} for s in ctl.SCENARIOS_3})
    p = _paths(mis.missing_for("rule", big))
    assert p == {"fv.Base.forward_fcf_musd", "fv.Base.target_yield", "framtida_antal_aktier", "aktuell_kurs",
                 "what_must_go_right", "ar_till_lag_skuld"}
    filled = dict(big, framtida_antal_aktier=50.0, aktuell_kurs=10.0, what_must_go_right="Kopparpris 4 USD/lb",
                  ar_till_lag_skuld=2.0, fv={"Base": {"forward_fcf_musd": 100.0, "target_yield": 12.0}})
    assert mis.missing_for("rule", filled) == []                 # en låg MoS är ett svar — inte en lucka här
    assert mis.missing_for("rule", dict(big, fcf_kvalitet="D", aktuell_kurs=10.0,
                                        what_must_go_right="x", ar_till_lag_skuld=2.0)) == []   # klass D: ingen FV


def test_insider_flow_asks_for_the_next_step():
    run = dict(INSIDER_OK, gate="", trigger="")
    assert ins.status(run) == ins.S_RUN_GATE and _paths(mis.missing_for("insider", run)) == {"gate"}
    wait = dict(INSIDER_OK, trigger="")
    assert ins.status(wait) == ins.S_WAIT_TRIGGER and _paths(mis.missing_for("insider", wait)) == {"trigger"}
    noise = dict(INSIDER_OK, insiders=1, role=ins.ROLE_OTHER, amount=100.0, gate="", trigger="")
    assert ins.status(noise) == ins.S_NOISE and mis.missing_for("insider", noise) == []


def test_tiggre_lists_only_the_catalyst_slots_still_needed():
    one = dict(TIGGRE_OK, catalysts=[{"name": "FID", "date": "2027-01"}, {"name": "Byggstart"}])
    assert _paths(mis.missing_for("tiggre", one)) == {"catalysts.1.date"}
    none_ = dict(TIGGRE_OK, catalysts=[])
    want = {f"catalysts.{i}.{part}" for i in (0, 1) for part in ("name", "date")}
    assert _paths(mis.missing_for("tiggre", none_)) == want


def test_scorecard_position_brings_the_controls_the_buy_gate_needs():
    assert mis.missing_for("royalty", ROYALTY_OK) == []                  # kortet självt har inga kontroller
    assert DS_KEYS <= _paths(mis.missing_for("royalty", ROYALTY_OK, position_pct=1.0))
    p = _paths(mis.missing_for("sprott", SPROTT_OK, position_pct=3.0))
    assert AQS_KEYS <= p and "csm.Bear.fcf_musd" in p


def test_scorecard_without_candidate_data_is_a_general_gap():
    items = mis.missing_for("scorecard", {"strategy": "producenter"}, entry=None)
    general = [i for i in items if not i.path]
    assert any("Kandidatdata saknas" in i.reason for i in general)
    assert {"position_pct_total", "sell_rule", "trigger"} <= _paths(items)


def test_thresholds_are_read_from_the_rules(monkeypatch):
    big = dict(RULE_OK, position_pct=3.0)
    assert AQS_KEYS <= _paths(mis.missing_for("rule", big))
    monkeypatch.setattr(ctl, "FULL_WORK_MIN_PCT", 5.0)
    assert not AQS_KEYS & _paths(mis.missing_for("rule", big))
    monkeypatch.setattr(sco, "SPROTT_RUNWAY_MIN_MONTHS", 30)
    assert "30 mån" in {i.path: i.reason for i in mis.missing_for("sprott", {})}["cash"]
