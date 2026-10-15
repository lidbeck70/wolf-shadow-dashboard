"""
Thor — Arkens dom: domslutet (ren funktion) ur granskningen, kontrollerna, scorecardet, motorn och cykeldatan;
cykeldatan med fejkade motorer; Copilot-prompten; och Regim-fliken för en arkstrategi under AppTest (inget nätverk).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_arken as fxa  # noqa: E402
from thor import dom  # noqa: E402
from thor import dossier as td  # noqa: E402
from thor import ui as tui  # noqa: E402

CTX = {"rotation_key": "koppar",
       "rotation": {"status": "AGERA", "sum": 13, "max": 15, "month": "2026-10", "why": "hat, case och katalysator",
                    "warnings": []},
       "valhall": {"view": "temat", "status": "GRÖN", "phase": "STARK", "cykel": "TIDIG"},
       "blindspot": {"cykel": "TIDIG", "hat": 61.0, "pct": 22.0, "score": 30.0, "rank": 1, "n": 2},
       "berserk": {"phase": "STARK", "driver": "HG=F", "close": 4.1, "date": "2026-10-14"},
       "fiat": {"level": "Neutral", "value": 55.0, "as_of": "2026-09"}, "errors": {}}
MISSING = ["Trigger definierad", "Position och säljregel satta"]
READY = lambda card, entry=None: {"ready": True, "missing": []}  # noqa: E731
NOT_READY = lambda card, entry=None: {"ready": False, "missing": list(MISSING)}  # noqa: E731


class _Score:
    def __init__(self, total, rating="GOOD"):
        self.total, self.rating, self.band = total, rating, rating


class _Analysis:
    def __init__(self, rec, why="skäl"):
        self.recommendation, self.recommendation_why = rec, why
        self.case, self.confidence = _Score(85.0, "PICK"), _Score(75.0, "GOOD")


def _sheets(status="PASS", controls=(("PASS", "DS", "2/10 — Låg"),), found=True, key="rule", cards=None):
    rev = {"found": found, "sheet": "Rick Rule", "status": status, "note": f"4/5 p — {status}",
           "controls": list(controls), "row": {"ticker": "BOL", "price": 9000.0}}
    return {"reviews": {key: rev} if found else {}, "confidence_store": None, "positions": [],
            "scorecard": cards if cards is not None else {"Rick Rule|BOL.ST": {}}}


def _judge(sheets, ctx=CTX, strategy="rule", scorecard=READY, rec="BUY CANDIDATE", company=object()):
    return dom.judge(strategy, sheets, ctx, "BOL.ST", scorecard_fn=scorecard,
                     company_fn=lambda store, t: company, analyze_fn=lambda c: _Analysis(rec))


# ── Domslutet ───────────────────────────────────────────────────────────────
def test_kop_when_the_sheet_says_buy_and_everything_is_green():
    d = _judge(_sheets())
    assert d.word == dom.KOP and d.color == dom.WORD_COLOR[dom.KOP] and not d.by_status(dom.MANUAL)
    assert [c.key for c in d.checks] == ["review", "ctl_DS", "scorecard", "engine", "rotation", "valhall", "blindspot",
                                         "berserk"]
    assert d.detail.startswith("8 PASS · 0 MANUAL · 0 FAIL") and d.review_key == "rule"
    labels = [f[0] for f in d.facts]
    assert labels == ["Case · Confidence", "Triple Signal", "Valhall · temat", "Blindspot", "BERSERK", "Fiat-miljö"]


def test_bevaka_when_the_sheet_passes_but_manual_items_remain():
    d = _judge(_sheets(controls=[("MANUAL", "AQS", "Ej bedömd")]), scorecard=NOT_READY, rec="WATCH")
    assert d.word == dom.BEVAKA and "kvar att göra" in d.sentence and "Kontrollen AQS" in d.sentence
    assert {c.key for c in d.by_status(dom.MANUAL)} == {"ctl_AQS", "scorecard", "engine"}
    assert "saknar: Trigger definierad" in next(c.note for c in d.checks if c.key == "scorecard")


@pytest.mark.parametrize("sheets,kw,who", [
    (_sheets(status="FAIL"), {}, "Granskningen i Rick Rule"),
    (_sheets(controls=[("FAIL", "DS", "DS 7/10 — köp låst")]), {}, "Kontrollen DS"),
    (_sheets(), {"rec": "REJECT"}, "Confidence/Durrett-motorn"),
    (_sheets(), {"ctx": {**CTX, "rotation": {**CTX["rotation"], "status": "Vila", "sum": 6}}},
     "Triple Signal (rotationen)"),
    (_sheets(), {"ctx": {**CTX, "valhall": {**CTX["valhall"], "cykel": "TOPP"}}}, "Valhall-kartan"),
])
def test_nej_on_any_fail(sheets, kw, who):
    d = _judge(sheets, **kw)
    assert d.word == dom.NEJ and d.sentence.startswith(who) and d.by_status(dom.FAIL)[0].label == who


def test_avvakta_when_not_reviewed_or_not_in_the_sheet():
    d = _judge(_sheets(status="MANUAL"))
    assert d.word == dom.AVVAKTA and d.sentence.startswith("Granskningen i Rick Rule")
    none = _judge(_sheets(found=False))
    assert none.word == dom.AVVAKTA and "finns inte i arket" in none.sentence and none.checks[0].status == dom.MANUAL
    hav = _judge({}, strategy="hav", company=None)
    assert hav.word == dom.AVVAKTA and "Håvarna är en lista" in hav.sentence and hav.review_key is None
    assert not any(c.key == "engine" for c in hav.checks)                         # ingen Confidence-rad för håven


def test_review_fallbacks_and_soft_signals():
    roy = {"found": True, "sheet": "Royalty C", "status": "PASS", "note": "ok", "controls": []}
    royalty = dom.judge("rule", {"reviews": {"royalty": roy}, "scorecard": {}}, {}, "FNV", scorecard_fn=READY,
                        company_fn=lambda s, t: None)
    assert royalty.review_key == "royalty" and royalty.checks[1].key == "ctl" and royalty.checks[1].status == dom.PASS
    assert royalty.word == dom.BEVAKA                                              # scorecard saknas → MANUAL
    insider = {"found": True, "sheet": "Insiderbevakaren", "status": "PASS", "note": "8/10 p", "controls": []}
    ins = dom.judge("insider", {"reviews": {"insider": insider}, "scorecard": {}},
                    {"valhall": {"view": "Norden", "status": "RÖD", "phase": "SVAG", "cykel": "TIDIG"}, "errors": {}},
                    "BOL.ST", scorecard_fn=READY, company_fn=lambda s, t: None)
    v = next(c for c in ins.checks if c.key == "valhall")
    assert v.status == dom.MANUAL and "arkets egen regel avgör" in v.note and ins.word == dom.BEVAKA
    soft = _judge(_sheets(), ctx={**CTX, "rotation": None, "blindspot": {**CTX["blindspot"], "cykel": "TOPP"},
                                  "berserk": {**CTX["berserk"], "phase": "BAISSE"}})
    by = {c.key: c for c in soft.checks}
    assert by["rotation"].status == dom.MANUAL and "inte betygsatt" in by["rotation"].note
    assert by["blindspot"].status == dom.MANUAL and by["berserk"].status == dom.MANUAL and soft.word == dom.BEVAKA
    turn = _judge(_sheets(), ctx={**CTX, "berserk": {**CTX["berserk"], "phase": "BAISSE · VÄNDER"}})
    assert "⚡ vändningen" in next(c.note for c in turn.checks if c.key == "berserk") and turn.word == dom.KOP


def test_prompt_lists_the_verdict_checks_and_facts():
    d = _judge(_sheets(controls=[("MANUAL", "AQS", "Ej bedömd")]), scorecard=NOT_READY)
    text = dom.dom_prompt(d, "BOL.ST", "Boliden")
    assert text.startswith("Kandidat: BOL.ST (Boliden)\nStrategi: Rick Rule\nArkens dom: BEVAKA")
    assert "Underlag som PASSERADE (6)" in text and "Underlag som kräver MANUELL kontroll (2)" in text
    assert "Underlag som FÖLL (0):\n  (inga)" in text and "Triple Signal: AGERA 13/15" in text
    assert "motsäg aldrig domslutet" in text and "Nivåer: inga satta här" in text


# ── Cykeldatan ──────────────────────────────────────────────────────────────
def test_cycle_context_reads_every_engine_and_survives_failures(monkeypatch):
    from test_thor_ravara import _prices, patch_engines
    patch_engines(monkeypatch)
    ident = td.identify("BOL.ST")
    assert ident.theme == "koppar"
    import test_thor_candidates as fx
    rotation = {"month": "2026-10", "grades": {"koppar": {"hatred": 5, "fundamentals": 4, "catalyst": 4,
                                                          "case_intact": True}}}
    ctx = dom.cycle_context(ident, {"board": fx.BOARD}, {"rotation": rotation}, _prices,
                            fiat=lambda ccy: {"currency": ccy, "level": "Neutral", "value": 55.0, "as_of": "2026-09"})
    assert ctx["rotation_key"] == "koppar" and ctx["rotation"]["status"] == "AGERA"
    assert ctx["blindspot"]["cykel"] == "TIDIG" and ctx["blindspot"]["rank"] == 1 and ctx["fiat"]["level"] == "Neutral"
    assert ctx["berserk"] is None or "phase" in ctx["berserk"]
    assert ctx["errors"] == {} or set(ctx["errors"]) <= {"BERSERK", "Valhall"}
    broken = dom.cycle_context(ident, {}, {}, None, theme_board=lambda: (_ for _ in ()).throw(RuntimeError("nere")),
                               fiat=lambda ccy: (_ for _ in ()).throw(RuntimeError("fred nere")))
    assert "nere" in broken["errors"]["Odins blindspot"] and "fred nere" in broken["errors"]["Debasement"]
    assert broken["valhall"] is None


# ── Fliken ──────────────────────────────────────────────────────────────────
def _script():
    import streamlit as st
    from thor import dossier as d
    from thor import ui as t
    ident = d.identify("BOL.ST")
    sheets = d.sheets_for("BOL.ST", st.session_state["_stores"])
    strategy = st.session_state.get("_strategy", "rule")
    t._tab_regim(st, {"board": st.session_state.get("_board")}, ident, strategy, sheets, [])


def test_regim_tab_shows_the_verdict_for_a_sheet_strategy(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_thor_ravara import patch_engines
    import test_thor_candidates as fx
    patch_engines(monkeypatch)
    monkeypatch.setattr(dom, "cycle_context", lambda ident, data, sheets, prices=None, **kw: CTX)
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["_stores"], at.session_state["_board"] = fxa.STORES, fx.BOARD
    at.run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    for part in ("ARKENS DOM — VAD SKA JAG GÖRA?", ">NEJ<", "Granskningen i Rick Rule", "Passa", "Kontrollen DS",
                 "Scorecardet", "Triple Signal", "Valhall-kartan", "Odins blindspot", "BERSERK-fasen", "UNDERLAGET",
                 "COPILOTEN", "Fiat-miljö", "INNEHAV"):
        assert part in text, part
    labels = [b.label for b in at.button]
    assert "🤖 Be Copiloten kommentera" in labels and any("Copiloten" in lbl for lbl in labels)
    assert any("Det du fyllt i" in e.label for e in at.expander)
    assert "Modellen är inte påslagen" in " ".join(c.value for c in at.caption)


def test_regim_tab_still_uses_the_engine_layout_for_other_strategies(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from thor import regim
    seen = []
    monkeypatch.setattr(regim, "render",
                        lambda st, ident, strategy, data, ohlcv=None, cands=None: seen.append(strategy))
    monkeypatch.setattr(dom, "render", lambda *a, **k: seen.append("dom"))
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["_stores"], at.session_state["_strategy"] = {}, "viking_nine"
    at.run()
    assert not at.exception and seen == ["viking_nine"]
