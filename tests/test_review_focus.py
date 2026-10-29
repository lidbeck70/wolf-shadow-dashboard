"""
Arken (PR B3) — enklare att fylla i, bara yta: inga regler, trösklar eller grindar är ändrade.

Fokusläget (ett bolag i taget, "Visa alla", Thors arkknappar öppnar arket på bolaget), raden först i korten —
'x/y fält ifyllda' och den dämpat röda "Saknas för köpbeslut", ritad sist så att den läser körningens värden, medan
kortets rubrik står still (en ny etikett fällde ihop kortet) — kriteriet i 0–2-rullgardinerna, synliga etiketter i
CSM-matrisen, Lukacs-scenarierna och Tiggres katalysatorer, Wolf Asymmetry-knapparna "Uppdatera arket",
Durrett-arkets "Visa bara saknade / köpkritiska", nivån i Royalty-kortet, "Durrett →" till Poängmodellen och
🧭 Durrett som egen knapp, Rick Rules position som förslag i Scorecardet och extraktionens förslag utan API-nyckel.
"""
import copy
import inspect
import json
import os
import re
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import controls as ctl  # noqa: E402
import insider  # noqa: E402
import scoring  # noqa: E402
import storage  # noqa: E402
import tiggre  # noqa: E402
from review import catalog as cat  # noqa: E402
from review import focus  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISIBLE, COLLAPSED = 0, 2                    # LabelVisibilityMessage
DS0 = {f.key: 0 for f in ctl.DS_FIELDS}
STORES = {
    "producers": {"producers": [{"id": "p1", "ticker": "BOL", "name": "Boliden", "commodity": "Koppar",
                                 "date": "2026-09-01", "position_pct": 5.0},
                                {"id": "p2", "ticker": "NEM", "name": "Newmont", "commodity": "Guld",
                                 "date": "2026-09-01"}],
                  "royalty": [{"id": "r1", "ticker": "FNV", "name": "Franco", "date": "2026-09-01"},
                              {"id": "r2", "ticker": "WPM", "name": "Wheaton", "level": 2, "date": "2026-09-01"}]},
    "scoring": {"sprott": [{"id": "s1", "ticker": "JR", "name": "Junior", "commodity": "Guld",
                            "date": "2026-09-01", "factors": {}}],
                "durrett": [{"id": "d1", "ticker": "BOL", "name": "Boliden", "commodity": "Guld",
                             "date": "2026-09-01", "factors": {}}]},
    "insider": {"signals": [{"id": "i1", "ticker": "EKTA-B.ST", "name": "Elekta", "found": "2026-09-01",
                             "role": insider.ROLE_OTHER, "gate": insider.GATE_BLANK, "trigger": ""}]},
    "tiggre": {"candidates": [{"id": "c1", "ticker": "LOB", "name": "Lobo", "added": "2026-09-01", "screen": {},
                               "downside": -40.0, "factors": {},
                               "catalysts": [{"id": "k1", "name": "FID", "date": "2027-03",
                                              "status": tiggre.CAT_DELIVERED}]}],
               "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "scorecard": {"cards": {}},
    "confidence": {"companies": {}, "commodity_overrides": {}},
}


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def page(monkeypatch):
    stores = copy.deepcopy(STORES)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import gist_storage
    import refresh_ui
    import screens_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    monkeypatch.setenv("B3_TEST_ROOT", ROOT)

    def run(module: str, fn: str, kw: dict = None, state: dict = None):
        from streamlit.testing.v1 import AppTest
        monkeypatch.setenv("B3_MOD", module)
        monkeypatch.setenv("B3_FN", fn)
        monkeypatch.setenv("B3_KW", json.dumps(kw or {}))

        def app():
            import importlib
            import json as _j
            import os as _o
            import sys as _s
            _s.path.insert(0, _o.environ["B3_TEST_ROOT"])
            getattr(importlib.import_module(_o.environ["B3_MOD"]), _o.environ["B3_FN"])(**_j.loads(_o.environ["B3_KW"]))

        at = AppTest.from_function(app, default_timeout=90)
        for k, v in (state or {}).items():
            at.session_state[k] = v
        at.run()
        assert not at.exception, (module, at.exception)
        return at
    return run


def _cards(at, marker: str) -> list:
    return [e for e in at.expander if marker in e.label]


def _count(card) -> tuple:
    """(ifyllda, totalt) ur raden först i kortet."""
    m = re.search(r"(\d+)/(\d+) fält ifyllda", card.markdown[0].value)
    assert m, card.markdown[0].value
    return int(m.group(1)), int(m.group(2))


# ── Fokusläget ───────────────────────────────────────────────────────────────
def test_focus_matches_every_ticker_form():
    assert focus.matches("BOL", "BOL.ST") and focus.matches("bol.st", "BOL") and not focus.matches("BOL", "NEM")
    assert focus.pick(["NEM", "BOL"], "BOL.ST") == "BOL" and focus.pick(["NEM"], "BOL") is None
    assert focus.pick(["BOL"], "") is None and not focus.matches("", "BOL")


def test_focus_opens_one_company_and_visa_alla_brings_the_list_back(page):
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"}, {focus.FOCUS: "BOL.ST"})
    cards = _cards(at, "/5 p")
    assert [c.label.split(" ·")[0] for c in cards] == ["BOL"] and cards[0].proto.expanded
    assert at.selectbox(key="arken_focus_rule").value == "BOL"
    at.button(key="arken_focus_rule_all").click().run()
    assert [c.label.split(" ·")[0] for c in _cards(at, "/5 p")] == ["BOL", "NEM"]
    assert not any(c.proto.expanded for c in _cards(at, "/5 p")) and at.session_state[focus.FOCUS] == ""
    at.selectbox(key="arken_focus_rule").select("NEM").run()
    assert [c.label.split(" ·")[0] for c in _cards(at, "/5 p")] == ["NEM"] and at.session_state[focus.FOCUS] == "NEM"
    # fokus följer med till nästa ark — där bolaget inte finns visas alla, och det sägs
    at2 = page("producers", "render_producers_page", {"sheet": "Royalty C"}, {focus.FOCUS: "NEM"})
    assert len(_cards(at2, "nivå")) == 2
    assert any("NEM finns inte i arket — visar alla bolag." in c.value for c in at2.caption)


def test_focus_works_in_every_sheet_and_writes_nothing(page):
    for module, fn, kw, sheet, ticker, marker in (
            ("scoring", "render_scoring_page", {}, "sprott", "JR", "/10 p"),
            ("tiggre", "render_tiggre_page", {}, "tiggre", "LOB", "poäng"),
            ("insider", "render_insider_page", {}, "insider", "EKTA-B.ST", "/10 p")):
        at = page(module, fn, kw, {focus.FOCUS: ticker})
        cards = _cards(at, marker)
        assert len(cards) == 1 and cards[0].proto.expanded, (module, [c.label for c in cards])
        store = {"scoring": "scoring", "tiggre": "tiggre", "insider": "insider"}[module]
        assert _same(at.session_state[store], STORES[store]), module


# ── Raden först i kortet ─────────────────────────────────────────────────────
@pytest.mark.parametrize("module,fn,kw,sheet,row_of,marker", [
    ("producers", "render_producers_page", {"sheet": "Rick Rule"}, "rule",
     lambda s: s["producers"]["producers"][0], "BOL ·"),
    ("producers", "render_producers_page", {"sheet": "Royalty C"}, "royalty",
     lambda s: s["producers"]["royalty"][0], "FNV ·"),
    ("scoring", "render_scoring_page", {}, "sprott", lambda s: s["scoring"]["sprott"][0], "JR ·"),
    ("tiggre", "render_tiggre_page", {}, "tiggre", lambda s: s["tiggre"]["candidates"][0], "LOB"),
    ("insider", "render_insider_page", {}, "insider", lambda s: s["insider"]["signals"][0], "EKTA-B.ST"),
])
def test_the_first_line_counts_filled_fields_and_names_what_is_missing(page, module, fn, kw, sheet, row_of, marker):
    at = page(module, fn, kw)
    card = next(e for e in at.expander if e.label.startswith(marker) or marker in e.label.split("·")[0])
    done, total, _m = cat.progress(sheet, row_of(STORES))
    assert "ifyllda" not in card.label                                    # rubriken är som förr
    first = card.markdown[0].value
    assert f"{done}/{total} fält ifyllda" in first, first
    assert "Saknas för köpbeslut:" in first and "#c44545" in first, first


# (sida, argument, kortets början, widget, nytt värde) — ett fält som inte rör rubrikens poäng eller dom
EDITS = [
    ("producers", "render_producers_page", {"sheet": "Rick Rule"}, "BOL ·", ("number_input", "pr_c_p1"), 1200.0),
    ("producers", "render_producers_page", {"sheet": "Royalty C"}, "FNV ·", ("number_input", "ro_pn_r1"), 0.5),
    ("scoring", "render_scoring_page", {}, "JR ·", ("number_input", "sc_cash_s1"), 20.0),
    ("tiggre", "render_tiggre_page", {}, "🔍  LOB", ("number_input", "tg_nav_c1"), 500.0),
    ("insider", "render_insider_page", {}, "EKTA-B.ST", ("number_input", "ins_a_i1"), 5000.0),
    ("scorecard", "render_scorecard_page", {}, "📋  BOL", ("text_input", "sc_sell_boliden|BOL"), "Stopp −20 %"),
]


@pytest.mark.parametrize("module,fn,kw,marker,widget,value", EDITS)
def test_an_edit_keeps_the_card_label_and_the_line_follows_in_the_same_run(
        page, module, fn, kw, marker, widget, value):
    """Rubriken byter inte text när ett fält fylls i (då fälldes kortet ihop), och raden först i kortet räknar
    med värdet redan i samma körning — de platta arken körs inte om efter en ändring."""
    at = page(module, fn, kw)
    card = next(e for e in at.expander if e.label.startswith(marker))
    label, (done, total) = card.label, _count(card)
    getattr(at, widget[0])(key=widget[1]).set_value(value).run()
    assert not at.exception, at.exception
    card = next(e for e in at.expander if e.label.startswith(marker))
    assert card.label == label
    assert _count(card) == (done + 1, total)


def test_the_missing_line_drops_a_field_in_the_run_it_is_filled(page):
    """Granskningens fall: AISC ifylld — lagret har talet och raden slutar be om det i samma körning."""
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"})
    nem = next(e for e in at.expander if e.label.startswith("NEM"))
    assert "Kostnad per enhet (AISC/C1)" in nem.markdown[0].value
    at.number_input(key="pr_c_p2").set_value(1200.0).run()
    assert at.session_state["producers"]["producers"][1]["unit_cost"] == 1200.0
    nem = next(e for e in at.expander if e.label.startswith("NEM"))
    line = nem.markdown[0].value
    assert "Kostnad per enhet" not in line and "Råvarupris nu" in line and "1/" in line, line


def test_the_missing_line_names_the_gate_inputs_and_says_when_nothing_is_missing(page):
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"})
    nem = next(e for e in at.expander if e.label.startswith("NEM"))
    line = nem.markdown[0].value
    assert "Råvarupris nu" in line and "Kostnad per enhet" in line and "Gruvlivslängd" in line
    assert "Poängen går inte att räkna" in line                                   # skälet i hovringstexten
    done = {"id": "p9", "ticker": "AEM", "name": "Agnico", "commodity": "Guld", "price": 3000.0,
            "unit_cost": 1200.0, "mine_life": 15.0, "jurisdiktion": True, "kapitaldisciplin": True, "insyn": False,
            "ev_musd": 900.0, "nav_musd": 1500.0, "position_pct": 1.0, **DS0}
    at2 = page("producers", "render_producers_page", {"sheet": "Rick Rule"},
               {"producers": {"producers": [done], "royalty": []}})
    card = next(e for e in at2.expander if e.label.startswith("AEM"))
    assert "Inget saknas för köpbeslut" in card.markdown[0].value


def test_scorecard_and_durrett_cards_have_the_line(page, monkeypatch):
    at = page("scorecard", "render_scorecard_page", {})
    card = next(e for e in at.expander if "BOL" in e.label and "producenter" in e.label)
    assert "ifyllda" not in card.label and "fält ifyllda" in card.markdown[0].value
    assert "Saknas för köpbeslut:" in card.markdown[0].value
    assert at.session_state["scorecard"]["cards"] == {}                            # orört kort skrivs inte
    import durrett_cases as dcs
    from confidence import store as cs
    data = cs.default()
    cs.put(data, dcs.gold_producer())
    at = page("engines.durrett.ui", "render_durrett_page", {}, {"confidence": data})
    card = next(e for e in at.expander if e.label.startswith("GPR · Guld"))
    assert "ifyllda" not in card.label and "fält ifyllda" in card.markdown[0].value
    assert "Saknas för köpbeslut:" in card.markdown[0].value


# ── Rullgardinerna och etiketterna ───────────────────────────────────────────
def test_score_dropdowns_show_the_criterion_and_still_store_the_number(page):
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"}, {focus.FOCUS: "BOL"})
    f = ctl.DS_FIELDS[0]
    sb = at.selectbox(key=f"ds_p1_{f.key}")
    assert sb.options == ["–", f"0 — {f.zero}", f"1 — {f.one}", f"2 — {f.two}"]
    a = ctl.AQS_FIELDS[0]
    assert at.selectbox(key=f"aqs_p1_{a.key}").options[3] == f"2 — {a.two}"
    sb.select(1).run()
    assert at.session_state["producers"]["producers"][0][f.key] == 1
    at = page("scoring", "render_scoring_page", {})
    fac = scoring.FACTORS[0]
    want = [f"0 — {fac.zero}", f"1 — {fac.one}", f"2 — {fac.two}"]
    assert at.selectbox(key=f"sc_f_s1_{fac.key}").options[1:] == want
    at = page("tiggre", "render_tiggre_page", {})
    assert at.selectbox(key="tg_f_c1_finansiering").options == ["–", "0 — Ofinansierat", "1 — Delvis", "2 — Löst"]
    assert at.selectbox(key="tg_f_c1_jurisdiktion").options == ["–", "0", "1", "2 — Toppjurisdiktion"]
    assert at.selectbox(key="tg_f_c1_stadium").options == ["–", "0", "1", "2"]     # ingen nivåtext att visa


def test_factor_levels_come_from_the_help_text_only():
    lv = {k: tiggre.factor_levels(h) for k, _l, h in tiggre.FACTORS}
    assert lv["finansiering"] == {2: "Löst", 1: "Delvis", 0: "Ofinansierat"}
    assert lv["jurisdiktion"] == {2: "Toppjurisdiktion"} and lv["stadium"] == {} and lv["manniskor"] == {}


def test_csm_lukacs_and_catalyst_inputs_have_visible_labels(page):
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"}, {focus.FOCUS: "BOL"})
    for key in ("csm_p1_Bear_price", "csm_p1_Bear_fcf", "fv_p1_Base_fcf", "fv_p1_Base_yield"):
        ni = at.number_input(key=key)
        assert ni.proto.label_visibility.value == VISIBLE and "(" in ni.label, key
    assert {m.label for m in at.metric} >= {"FV/aktie (Base)", "Uppsida (Base)"}
    at = page("tiggre", "render_tiggre_page", {})
    assert at.text_input(key="tg_cat_name_c1").label == "Ny katalysator"
    for w in (at.text_input(key="tg_cat_name_c1"), at.text_input(key="tg_cat_date_c1"),
              at.selectbox(key="tg_catst_c1_k1"), at.text_input(key="tg_catact_c1_k1"),
              at.number_input(key="tg_catrea_c1_k1"), at.text_input(key="tg_catles_c1_k1")):
        assert w.proto.label_visibility.value == VISIBLE, w.key
    import controls_ui
    import lukacs_ui
    for fn in (controls_ui.render_csm, lukacs_ui.render_fv, tiggre._catalysts):
        assert 'label_visibility="collapsed"' not in inspect.getsource(fn), fn.__name__


def test_wolf_asymmetry_buttons_update_the_sheet_and_point_to_save():
    from asymmetry import ui as aui
    src = inspect.getsource(aui)
    assert aui.UPDATE == "Uppdatera arket" and "spara med 💾" in aui.SAVE_NOTE
    assert 'form_submit_button("Spara' not in src and src.count("form_submit_button(UPDATE)") == 2


# ── Durrett: bara saknade / köpkritiska ──────────────────────────────────────
def test_durrett_all_fields_toggle_shows_only_missing_buy_critical_fields(page):
    import durrett_cases as dcs
    from confidence import store as cs
    from review import missing as mis
    data = cs.default()
    company = dcs.gold_producer()
    cs.put(data, company)
    state = {"confidence": data, "durrett_row_sub_GPR": "Mer"}
    at = page("engines.durrett.ui", "render_durrett_page", {}, state)
    every = {t.key for t in at.text_input if str(t.key).startswith("cf_GPR_") and t.key.endswith("_v")}
    at.toggle(key="conf_only_GPR").set_value(True).run()
    assert not at.exception, at.exception
    shown = {t.key[len("cf_GPR_"):-2] for t in at.text_input if str(t.key).startswith("cf_GPR_") and t.key.endswith("_v")}
    shown |= {s.key[len("cf_GPR_"):-2] for s in at.selectbox if str(s.key).startswith("cf_GPR_") and s.key.endswith("_v")}
    want = {it.path.split(".", 1)[1] for it in mis.missing_for("confidence", company) + mis.missing_for("asymmetry", company)
            if it.path.startswith("fields.")}
    assert want and shown == want and len(shown) < len(every)
    assert any("köpkritiska fält saknas" in c.value for c in at.caption)


# ── Royalty: nivån i kortet ──────────────────────────────────────────────────
def test_royalty_level_is_editable_in_the_card_and_never_written_on_open(page):
    at = page("producers", "render_producers_page", {"sheet": "Royalty C"})
    sb = at.selectbox(key="ro_lvl_r1")
    assert sb.value is None and sb.options[0] == "– (ej satt)" and "Nivå 1" in sb.options[1]
    assert _same(at.session_state["producers"], STORES["producers"])               # att öppna skriver inget
    assert at.selectbox(key="ro_lvl_r2").value == 2 and "– (ej satt)" not in at.selectbox(key="ro_lvl_r2").options
    sb.select(3).run()
    assert at.session_state["producers"]["royalty"][0]["level"] == 3


# ── Thor: knapparna öppnar arket på bolaget ──────────────────────────────────
def test_thor_sheet_buttons_open_poangmodellen_for_durrett_and_offer_the_10_step_sheet():
    from thor import arken
    from thor import ui as tui
    got = {k: (label, path, extra) for k, label, path, extra in
           arken.sheet_buttons(tui.SHEET_LINK, {"reviews": {"durrett": {}}})}
    assert got["durrett"] == ("Durrett →", "review/poangmodell", {scoring.WHICH_KEY: scoring.WHICH[1]})
    assert got["sprott"][1] == "review/poangmodell" and got["sprott"][2] == {scoring.WHICH_KEY: scoring.WHICH[0]}
    assert got["durrett10"] == ("🧭 Durrett →", arken.DURRETT_10, None) and got["rule"][2] is None
    plain = {k for k, *_ in arken.sheet_buttons(tui.SHEET_LINK, {"reviews": {}})}
    assert plain == {"rule", "royalty", "sprott"}                                 # de tre första alltid
    assert "durrett10" in {k for k, *_ in arken.sheet_buttons(tui.SHEET_LINK, {"reviews": {}, "confidence": {"x": 1}})}
    from ui import nav
    assert nav.resolve(arken.DURRETT_10) == ["review", "🧭 Durrett"]
    assert nav.resolve(tui.SHEET_LINK["durrett"]) == ["review", "Poängmodell"]


def _thor_script():
    import streamlit as st
    from thor import arken as a
    from thor import dossier as d
    from thor import ui as t
    ident = d.identify("BOL.ST")
    a.render(st, {}, ident, d.sheets_for("BOL.ST", st.session_state["_stores"]), [], t.SHEET_LINK, t._jump,
             strategy="durrett")


def test_thor_durrett_button_sets_focus_and_the_model(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    stores = {"scoring": copy.deepcopy(STORES["scoring"])}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    at = AppTest.from_function(_thor_script, default_timeout=120)
    at.session_state["_stores"] = stores
    at.run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button]
    assert "Durrett →" in labels and "🧭 Durrett →" in labels
    at.button(key="thor_sheet_durrett").click().run()
    assert at.session_state["nav_goto"] == "review/poangmodell"
    assert at.session_state[scoring.WHICH_KEY] == "Durrett (snabbpoäng)" and at.session_state[focus.FOCUS] == "BOL.ST"
    at.button(key="thor_sheet_durrett10").click().run()
    assert at.session_state["nav_goto"] == "review/durrett"


def test_poangmodellen_opens_on_the_model_and_company_thor_set(page):
    at = page("scoring", "render_scoring_page", {}, {scoring.WHICH_KEY: scoring.WHICH[1], focus.FOCUS: "BOL.ST"})
    cards = _cards(at, "/10 p")
    assert [c.label.split(" ·")[0] for c in cards] == ["BOL"] and cards[0].proto.expanded


# ── Scorecardet: Rick Rules position som förslag ─────────────────────────────
RULE_HINT = "Rick Rule-kortet: tänkt position"


def test_scorecard_offers_the_rick_rule_position_and_writes_it_only_on_anvand(page):
    at = page("scorecard", "render_scorecard_page", {}, {"scorecard": {"cards": {
        "boliden|BOL": {"strategy": "producenter"}}}})
    key = "boliden|BOL"
    assert any(f"{RULE_HINT} 5 % av total" in c.value for c in at.caption)
    assert at.session_state["scorecard"]["cards"][key] == {"strategy": "producenter"}  # bara ett förslag
    at.button(key=f"sc_userule_{key}").click().run()
    assert not at.exception, at.exception
    card = at.session_state["scorecard"]["cards"][key]
    assert card["position_pct_total"] == 5.0 and card["strategy"] == "producenter"
    assert at.number_input(key=f"sc_post_{key}").value == 5.0
    assert not any(RULE_HINT in c.value for c in at.caption)


def test_the_rick_rule_position_is_offered_only_for_producenter_and_until_the_card_has_a_size(page):
    """Förslaget gäller Rick Rule-beslutet: inte när kortet gäller en annan strategi (här Durrett, kandidatens
    första), och inte när du redan valt en storlek."""
    at = page("scorecard", "render_scorecard_page", {})
    key = "boliden|BOL"
    assert at.selectbox(key=f"sc_strat_{key}").value == "durrett"
    assert not any(RULE_HINT in c.value for c in at.caption)
    assert at.session_state["scorecard"]["cards"] == {}
    at.selectbox(key=f"sc_strat_{key}").select("producenter").run()
    assert any(RULE_HINT in c.value for c in at.caption)
    at.number_input(key=f"sc_post_{key}").set_value(2.0).run()
    assert not at.exception, at.exception
    assert not any(RULE_HINT in c.value for c in at.caption)
    assert at.session_state["scorecard"]["cards"][key]["position_pct_total"] == 2.0     # ditt val står


# ── Extraktionen utan API-nyckel ─────────────────────────────────────────────
def _parsed(**fields):
    return {"fields": {k: {"value": v, "unit": "", "page": 3, "quote": "q", "confidence": "high"}
                       for k, v in fields.items()}}


def test_stored_extraction_is_shown_and_usable_without_an_api_key(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ai import openai_client as oc
    monkeypatch.setattr(oc, "configured", lambda: False)
    monkeypatch.setenv("B3_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["B3_TEST_ROOT"])
        import streamlit as st
        import extract_ui
        row = st.session_state.setdefault("row", {"id": "r1", "ticker": "KDK", "name": "Kodiak"})
        extract_ui.render_extractor("rule", row, widget_keys={"unit_cost": "pr_c_r1"})

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["xt_doc:KDK"] = {"parsed": _parsed(aisc=1250), "doc": "PFS 2026", "sheet": "Tiggre",
                                      "model": "m", "pages": 12, "chars": 5000, "when": "2026-09-23 10:00"}
    at.run()
    assert not at.exception, at.exception
    assert "Extrahera" not in [b.label for b in at.button] and not at.get("file_uploader")
    assert any("OPENAI_API_KEY saknas" in c.value for c in at.caption)
    at.button(key="xt_rule_r1_use_unit_cost").click().run()
    assert at.session_state["row"]["unit_cost"] == 1250.0


def test_confidence_extractor_shows_stored_proposals_without_an_api_key(monkeypatch):
    import durrett_cases as dcs
    from streamlit.testing.v1 import AppTest
    from ai import openai_client as oc
    from confidence import store as cs
    monkeypatch.setattr(oc, "configured", lambda: False)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, default))
    monkeypatch.setenv("B3_TEST_ROOT", ROOT)
    data = cs.default()
    company = dcs.gold_producer()
    company.fields.pop("aisc", None)
    cs.put(data, company)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["B3_TEST_ROOT"])
        import streamlit as st
        from confidence import store as cs_
        from confidence import ui as cui
        d = st.session_state["_data"]
        cui.render_extractor(d, cs_.get(d, "GPR"))

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["_data"] = data
    at.session_state["xt_doc:GPR"] = {"parsed": _parsed(aisc=1180), "doc": "MD&A", "sheet": "Rick Rule",
                                      "model": "m", "pages": 4, "chars": 900, "when": "2026-09-23 10:00"}
    at.run()
    assert not at.exception, at.exception
    assert "Extrahera" not in [b.label for b in at.button]
    at.button(key="cf_xt_GPR_use_aisc").click().run()
    assert not at.exception, at.exception
    assert at.session_state["_data"]["companies"]["GPR"]["fields"]["aisc"]["value"] == 1180.0
