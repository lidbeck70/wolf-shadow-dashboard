"""
LIDSKJALV PR 23 — Regler & Guider, ägarbeslut 12 (docs/design_lidskjalv.md §4.10, §7): ovtlyr/ui/rules_page.py är inte
fryst — ordningen ändras, varje regeltext står ordagrant.

  · Handelsregler: kortet "Hårda regler" först (ovanför RISKMODELL) med exakt playbookens hårda regler, ritade med
    _rule_row; i Entry/Exit-listan behåller den hårda regeln sitt nummer men visar bara en pekare dit. Varje regeltext,
    förklaring, panelguide och fallgrop står på sidan exakt så många gånger som i playbooken själv (en gång när
    texten är unik) — för VARJE playbook i LEARNING_ORDER
  · Fusklapp: rutan HÅRDA REGLER före rutnätet; varje fusklappsvärde och hård regel exakt en gång
  · Kom igång: Börja här → Veckorutinen → en expander per playbook (rules_start_<nyckel>) i LEARNING_ORDER, och
    antalet i ingressen ur len(LEARNING_ORDER)
  · Årshjulet: "om 1 dag" i singular, "om 2 dagar" i plural
"""
import dataclasses
import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import routines                                    # noqa: E402
import strategy_rules as sr                        # noqa: E402
from ovtlyr.ui import rules_page as rp             # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def _page_app():
    import streamlit as st

    from ovtlyr.ui import rules_page as rp
    {"rules": rp._page_rules, "cheat": rp._page_cheatsheet, "start": rp._page_start,
     "routines": rp._page_routines}[st.session_state.get("_pg", "rules")]()


def _run(page: str, monkeypatch=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_page_app, default_timeout=60)
    at.session_state["_pg"] = page
    at.run()
    assert not at.exception, at.exception
    return at


def _html(at) -> str:
    """Allt sidan skrev som markdown (och st.info) i ritordning."""
    return "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(i.value for i in at.info)


def _flat(at) -> list:
    """Sidans noder i ritordning: ('expander', etikett) och ('markdown', text)."""
    out = []

    def walk(node):
        kids = getattr(node, "children", None)
        if isinstance(kids, dict):
            if node.type == "expander":
                out.append(("expander", node.label))
            for i in sorted(kids):
                walk(kids[i])
        elif node.type == "markdown":
            out.append(("markdown", node.value))
    walk(at.main)
    return out


def _first(flat, needle, kind="markdown") -> int:
    return next(i for i, (k, v) in enumerate(flat) if k == kind and needle in v)


def _rules_corpus(pb) -> str:
    """Allt Handelsregler ritar ur playbooken, varje fält EN gång — referensen för hur ofta en text får förekomma
    (några texter står i två fält i datan, t.ex. 'Max 10 % per aktie' både som regel och i riskmodellen)."""
    r = pb.risk
    parts = [pb.name, pb.tagline, pb.where, pb.level.upper(), pb.support.upper(), r.risk_per_trade,
             r.position_size, r.max_positions, r.stop, r.targets, *pb.workflow, *pb.pitfalls, pb.note]
    for x in pb.entry + pb.exit + pb.mindset:
        parts += [x.text, x.explanation, x.panel_guide]
    return "\n".join(parts)


def _cheat_corpus(pb) -> str:
    return "\n".join([pb.name] + [s for pair in pb.cheatsheet for s in pair]
                     + [x.text for x in pb.entry + pb.exit if x.hard])


def _lists(pb):
    return (("Entry", pb.entry, pb.color), ("Exit", pb.exit, rp._RED), ("Mindset", pb.mindset, rp._YELLOW))


# ── Handelsregler ────────────────────────────────────────────────────────────
def test_every_rule_text_appears_exactly_once_on_handelsregler_for_every_playbook():
    at = _run("rules")
    for key in sr.LEARNING_ORDER:
        pb = sr.PLAYBOOKS[key]
        at.radio(key="rules_pb_rules").set_value(pb.name).run()
        assert not at.exception, (key, at.exception)
        html, ref = _html(at), _rules_corpus(pb)
        texts = [t for r in pb.entry + pb.exit + pb.mindset for t in (r.text, r.explanation, r.panel_guide)]
        texts += list(pb.pitfalls)
        for t in texts:
            assert html.count(t) == ref.count(t), (key, html.count(t), ref.count(t), t[:60])
        assert sum(ref.count(t) == 1 for t in texts) >= len(texts) - 3, key     # nästan alla är unika: en gång
        # och varje regel ritas i sin helhet (nummer, text, chip, förklaring, 📍) exakt en gång
        for _label, rules, color in _lists(pb):
            for r in rules:
                assert html.count(rp._rule_row(r, color)) == 1, (key, r.number, r.text[:40])


def test_the_hard_card_comes_first_and_holds_exactly_the_hard_rules():
    at = _run("rules")
    for key in sr.LEARNING_ORDER:
        pb = sr.PLAYBOOKS[key]
        at.radio(key="rules_pb_rules").set_value(pb.name).run()
        md = [m.value for m in at.markdown]
        hard = [r for _l, rules, _c in _lists(pb) for r in rules if r.hard]
        assert hard, key                                         # varje playbook i dag har hårda regler
        card_i = next(i for i, v in enumerate(md) if f"id='{rp.HARD_ID}'" in v)
        risk_i = next(i for i, v in enumerate(md) if ">RISKMODELL<" in v)
        line_i = next(i for i, v in enumerate(md) if f"📍 {pb.where}" in v)
        assert line_i < card_i < risk_i, key                     # väljaren och playbook-raden, sedan kortet
        card = md[card_i]
        assert card == rp._hard_card_html(pb)
        assert card.count("HÅRD REGEL</span>") == len(hard), key
        for label, rules, color in _lists(pb):
            for r in rules:
                assert (rp._rule_row(r, color) in card) == r.hard, (key, label, r.number)
            if any(r.hard for r in rules):                       # märkt med listan den kommer ur
                assert f">{label}</div>" in card, (key, label)
        # listorna: samma numrering, den hårda regelns plats är en pekare till kortet
        entry = next(v for v in md if rp._rule_slot(pb.entry[0], pb.color) in v)
        exits = next(v for v in md if rp._rule_slot(pb.exit[0], rp._RED) in v)
        for body, rules, color in ((entry, pb.entry, pb.color), (exits, pb.exit, rp._RED)):
            for r in rules:
                slot = rp._rule_slot(r, color)
                assert body.count(slot) == 1
                assert f"{r.number}.</span>" in slot
                assert (rp.HARD_POINTER in slot) == r.hard and (r.text in slot) != r.hard
            assert body.count(rp.HARD_POINTER) == sum(r.hard for r in rules)
            assert [x.number for x in rules] == list(range(1, len(rules) + 1))      # obruten numrering


def test_no_hard_rules_means_no_card_and_plain_rows():
    pb = sr.PLAYBOOKS["momentum"]
    soft = dataclasses.replace(
        pb, entry=tuple(dataclasses.replace(r, hard=False) for r in pb.entry),
        exit=tuple(dataclasses.replace(r, hard=False) for r in pb.exit))
    assert rp._hard_card_html(soft) == "" and rp._hard_groups(soft) == []
    for r in soft.entry:
        assert rp._rule_slot(r, soft.color) == rp._rule_row(r, soft.color)


def test_the_pointer_jumps_to_the_card():
    pb = sr.PLAYBOOKS["momentum"]
    r = next(x for x in pb.entry if x.hard)
    slot = rp._rule_slot(r, pb.color)
    assert f"data-jump='{rp.HARD_ID}'" in slot and "min-height:44px" in slot
    assert rp._hard_card_html(pb).startswith(f"<div id='{rp.HARD_ID}'")
    assert "scroll-margin-top:72px" in rp._hard_card_html(pb)          # landar under den klistrade flikraden


# ── Fusklapp ─────────────────────────────────────────────────────────────────
def test_fusklapp_shows_its_hard_box_before_the_grid_and_every_value_once():
    at = _run("cheat")
    for key in sr.LEARNING_ORDER:
        pb = sr.PLAYBOOKS[key]
        at.radio(key="rules_pb_cheat").set_value(pb.name).run()
        assert not at.exception, (key, at.exception)
        md = [m.value for m in at.markdown]
        box = next(i for i, v in enumerate(md) if "HÅRDA REGLER — BRYTS ALDRIG" in v)
        grid = next(i for i, v in enumerate(md) if "grid-template-columns:repeat(auto-fit" in v)
        assert box < grid, key
        html, ref = _html(at), _cheat_corpus(pb)
        hard = [r.text for r in pb.entry + pb.exit if r.hard]
        assert md[box].count("✕</span>") == len(hard), key
        for t in hard + [v for _l, v in pb.cheatsheet]:
            assert html.count(t) == ref.count(t), (key, t[:60])


# ── Kom igång ────────────────────────────────────────────────────────────────
@pytest.fixture
def expander_keys(monkeypatch):
    import streamlit
    calls = []
    real = streamlit.expander

    def _spy(label, *a, **kw):
        calls.append((label, kw.get("key")))
        return real(label, *a, **kw)
    monkeypatch.setattr(streamlit, "expander", _spy)
    return calls


def test_kom_igang_order_borja_har_veckorutinen_then_one_expander_per_playbook(expander_keys):
    at = _run("start")
    flat = _flat(at)
    n = len(sr.LEARNING_ORDER)
    intro = flat[_first(flat, "KOM IGÅNG")][1]
    assert f"{n} strategier, olika svårighetsgrad. Lär dig en i taget — börja överst." in intro
    assert "Fyra strategier" not in intro
    start = _first(flat, "👉 Börja här:")
    week = _first(flat, "VECKORUTINEN — MOMENTUM SWING")
    exps = [i for i, (k, _v) in enumerate(flat) if k == "expander"]
    assert len(exps) == n
    assert start < week < exps[0]
    assert week < _first(flat, "ÅRSHJULET</b> här ovanför") < exps[0]          # årshjulsnoten hör till rutinen
    labels = [flat[i][1] for i in exps]
    for i, (key, label) in enumerate(zip(sr.LEARNING_ORDER, labels), start=1):
        pb = sr.PLAYBOOKS[key]
        assert label == f"{i}\\. {pb.name} · {pb.level.upper()}", label
    assert [k for _l, k in expander_keys] == [f"rules_start_{k}" for k in sr.LEARNING_ORDER]


def test_each_kom_igang_expander_holds_the_old_card_texts_verbatim():
    at = _run("start")
    bodies = {e.label: "\n".join(c.value for c in e.children.values() if c.type == "markdown") for e in at.expander}
    for i, key in enumerate(sr.LEARNING_ORDER, start=1):
        pb = sr.PLAYBOOKS[key]
        body = bodies[f"{i}\\. {pb.name} · {pb.level.upper()}"]
        assert body == rp._start_card(pb)
        for text in (pb.tagline, pb.idea, pb.horizon, pb.risk.risk_per_trade, pb.where,
                     rp._level_badge(pb.level), rp._support_badge(pb)):
            assert text in body, (key, text[:40])
        if pb.source:
            assert f"Källa:</b> {pb.source}" in body
        if pb.support != sr.SUPPORT_FULL and pb.support_note:
            assert pb.support_note in body
    # och inget kort står kvar utanför expandrarna
    outside = "\n".join(v for k, v in _flat(at) if k == "markdown" and v not in bodies.values())
    for key in sr.LEARNING_ORDER:
        assert sr.PLAYBOOKS[key].idea not in outside


# ── Årshjulet ────────────────────────────────────────────────────────────────
def _quiet_day(days_until: int) -> datetime.date:
    """En dag utan rutin, där nästa rutin infaller om days_until dagar (ur routines egen kalender)."""
    d = datetime.date(2026, 1, 1)
    for _ in range(800):
        if not routines.due_today(d) and routines.agenda(d)[0]["days_until"] == days_until:
            return d
        d += datetime.timedelta(days=1)
    raise AssertionError(f"ingen dag med nästa rutin om {days_until} dagar")


@pytest.mark.parametrize("left, word", [(1, "om 1 dag."), (2, "om 2 dagar.")])
def test_arshjulet_says_om_1_dag_in_the_singular(monkeypatch, left, word):
    day = _quiet_day(left)

    class _Day(datetime.date):
        @classmethod
        def today(cls):
            return day
    monkeypatch.setattr(rp, "_date", _Day)
    at = _run("routines")
    line = next(m.value for m in at.markdown if "Ingen rutin i dag" in m.value)
    assert f"{routines.agenda(day)[0]['routine'].title} {word}</span>" in line
    assert "om 1 dagar" not in _html(at)
    assert rp._om_dagar(1) == "om 1 dag" and rp._om_dagar(3) == "om 3 dagar" and rp._om_dagar(0) == "om 0 dagar"
