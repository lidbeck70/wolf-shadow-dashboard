"""
LIDSKJALV PR 21 — ARKEN: domen överst i kortet (§4.7 Scorecard, §4.8 Rick Rule och Royalty C, §3.5.14 gate_meter).

Bara ordning och utseende: korten sorteras för visningen (klara först, flest kryss, ticker) ur gate_state, rubriken
börjar med tickern och expandrarna är nycklade, domen står i kortets första markdown tillsammans med räknaren och
"Saknas", "Registrera beslut" direkt under den, remsorna räknar det som redan räknats och fiatmiljön ligger hopfälld.
gate_state, GATES, poängen och det sparade är orörda; att öppna sidan skriver ingenting.
"""
import copy
import json
import os
import sys
from collections import Counter
from datetime import date

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import producers as pr  # noqa: E402
import scorecard as sc  # noqa: E402
import storage  # noqa: E402
from review import card as rcard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS0 = {"ds_runway": 0, "ds_capex": 0, "ds_warrants": 0, "ds_aktier_3ar": 0, "ds_historik": 0}

# Kandidaterna (samma sorts rader som test_thor_scorecard_entry och test_review_focus): BOL klar, NEM tre kryss,
# AEM ett, EQX, LOB, FNV och WPM inget kort alls. Rick Rule: BOL Köpkandidat, NEM Bevaka, AEM Passa, EQX ofullständig
BOL = {"id": "a1", "ticker": "BOL", "name": "Boliden", "commodity": "Koppar", "price": 9000.0, "unit_cost": 6000.0,
       "ev_ebitda": 5.0, "nd_ebitda": 0.5, "mine_life": 12.0, "rp_ratio": 15.0, "jurisdiktion": True,
       "kapitaldisciplin": True, "insyn": True, **DS0}
NEM = {"id": "a2", "ticker": "NEM", "name": "Newmont", "commodity": "Guld", "price": 3000.0, "unit_cost": 1200.0,
       "mine_life": 15.0, "jurisdiktion": True, **DS0}
AEM = {"id": "a3", "ticker": "AEM", "name": "Agnico", "commodity": "Guld", "price": 2000.0, "unit_cost": 1900.0,
       "mine_life": 10.0, "insyn": True}
EQX = {"id": "a4", "ticker": "EQX", "name": "Equinox", "commodity": "Guld"}             # pris och kostnad saknas
GATE_KEYS = [g.key for g in sc.GATES if g.key != "inga_roda_flaggor"]
CARDS = {"boliden|BOL": {**{k: True for k in GATE_KEYS}, "strategy": "producenter", "position_pct_total": 1.5},
         "newmont|NEM": {**{k: True for k in GATE_KEYS[:3]}, "strategy": "producenter", "position_pct_total": 1.0},
         "agnico|AEM": {GATE_KEYS[0]: True, "strategy": "producenter"}}
WPM = {"id": "r2", "ticker": "WPM", "name": "Wheaton", "level": 1, "pnav_now": 1.05, "pnav_bottom": 1.0,
       "ev_now": 20.0, "ev_median": 25.0, "geo_now": 0.0022, "geo_3y": 0.0020}
FNV = {"id": "r1", "ticker": "FNV", "name": "Franco", "level": 1}
STORES = {
    "producers": {"producers": [BOL, NEM, AEM, EQX], "royalty": [FNV, WPM]},
    "scoring": {"sprott": [], "durrett": []},
    "insider": {"signals": []},
    "tiggre": {"candidates": [{"id": "c1", "ticker": "LOB", "name": "Lobo", "added": "2026-09-01", "screen": {},
                               "downside": -40.0, "factors": {}, "catalysts": []}],
               "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "scorecard": {"cards": CARDS},
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
    monkeypatch.setenv("ARKEN_TEST_ROOT", ROOT)
    expanders = []                                  # (etikett, nyckel, expanded) för varje st.expander-anrop
    real = streamlit.expander

    def _spy(label, expanded=False, **kw):
        expanders.append((label, kw.get("key"), expanded))
        return real(label, expanded=expanded, **kw)
    monkeypatch.setattr(streamlit, "expander", _spy)

    def run(module: str, fn: str, kw: dict = None, state: dict = None):
        from streamlit.testing.v1 import AppTest
        monkeypatch.setenv("ARKEN_MOD", module)
        monkeypatch.setenv("ARKEN_FN", fn)
        monkeypatch.setenv("ARKEN_KW", json.dumps(kw or {}))

        def app():
            import importlib
            import json as _j
            import os as _o
            import sys as _s
            _s.path.insert(0, _o.environ["ARKEN_TEST_ROOT"])
            getattr(importlib.import_module(_o.environ["ARKEN_MOD"]),
                    _o.environ["ARKEN_FN"])(**_j.loads(_o.environ["ARKEN_KW"]))

        expanders.clear()
        at = AppTest.from_function(app, default_timeout=90)
        for k, v in (state or {}).items():
            at.session_state[k] = v
        at.run()
        assert not at.exception, (module, at.exception)
        at.expander_calls = list(expanders)
        at.expander_log = expanders                 # levande: växer med varje ny körning (töm före .run())
        return at
    return run


def _sc_cards(at) -> list:
    return [e for e in at.expander if e.label.startswith(("📋  ", "✅  "))]


def _ticker(label: str) -> str:
    """Tickern i en scorecard-rubrik: ikonen och två blanksteg först, sedan 'TICKER · …'."""
    return label.split("  ", 1)[1].split(" ·")[0]


def _entries_states(stores=STORES):
    entries = sc.collect(sc.sources_from_stores(stores))
    cards = stores["scorecard"]["cards"]
    return entries, {e["key"]: sc.gate_state(sc._shown(cards.get(e["key"], {}), e), e) for e in entries}


# ── Sorteringen: bara gate_state ─────────────────────────────────────────────
def test_display_order_is_ready_then_gates_passed_then_ticker():
    entries = [{"key": k, "ticker": t} for k, t in (("a", "ZZZ"), ("b", "AAA"), ("c", "MMM"), ("d", "BBB"))]
    states = {"a": {"ready": False, "checks": {"x": True, "y": True, "z": False}},
              "b": {"ready": False, "checks": {"x": True, "y": False, "z": False}},
              "c": {"ready": True, "checks": {"x": True, "y": True, "z": True}},
              "d": {"ready": False, "checks": {"x": True, "y": True, "z": False}}}
    before = copy.deepcopy((entries, states))
    assert [e["ticker"] for e in sc.display_order(entries, states)] == ["MMM", "BBB", "ZZZ", "AAA"]
    assert (entries, states) == before                                     # läser bara
    assert sc.passed({"checks": {"a": True, "b": False, "c": None}}) == 1 and sc.passed({}) == 0


def test_the_page_sorts_by_gate_state_and_writes_nothing(page):
    at = page("scorecard", "render_scorecard_page")
    entries, states = _entries_states()
    want = [e["ticker"] for e in sc.display_order(entries, states)]
    assert [_ticker(e.label) for e in _sc_cards(at)] == want
    assert want[0] == "BOL" and states["boliden|BOL"]["ready"]              # klar först
    passed = [sc.passed(states[e["key"]]) for e in sc.display_order(entries, states)][1:]
    assert passed == sorted(passed, reverse=True)
    assert _same(at.session_state["scorecard"], STORES["scorecard"])        # att öppna skriver inget


def test_the_order_follows_gate_state_and_nothing_else(page, monkeypatch):
    """Byt grinden mot en attrapp: ordningen följer attrappens kryss — sorteringen har ingen egen regel."""
    real = sc.gate_state
    fake_passed = {"BOL": 1, "NEM": 6, "AEM": 4, "EQX": 0, "LOB": 6}

    def fake(card, entry=None):
        s = real(card, entry)
        n = fake_passed.get((entry or {}).get("ticker"), 0)
        checks = {g.key: i < n for i, g in enumerate(sc.GATES)}
        return {**s, "checks": checks, "missing": [g.label for g in sc.GATES if not checks[g.key]],
                "ready": n == len(sc.GATES)}
    monkeypatch.setattr(sc, "gate_state", fake)
    at = page("scorecard", "render_scorecard_page")
    assert [_ticker(e.label) for e in _sc_cards(at)] == ["LOB", "NEM", "AEM", "BOL", "EQX", "FNV", "WPM"]
    assert {m.label: m.value for m in at.metric}["Klara för köp"] == "0"


def test_live_card_reads_the_cards_own_widget_values_and_writes_nothing():
    entries, _s = _entries_states()
    nem = next(e for e in entries if e["ticker"] == "NEM")
    card = copy.deepcopy(CARDS["newmont|NEM"])
    k = nem["key"]
    assert sc.live_card(card, nem, {}) == sc._shown(card, nem)                        # inga widgetvärden: som förut
    session = {f"sc_g_{k}_trigger_definierad": True, f"sc_g_{k}_strategi_aktiv": False, f"sc_post_{k}": 3.0,
               f"sc_strat_{k}": "finns-inte", "sc_g_annat|X_trigger_definierad": True}
    live = sc.live_card(card, nem, session)
    assert live["trigger_definierad"] is True and live["strategi_aktiv"] is False
    assert live["position_pct_total"] == 3.0 and live["strategy"] == "producenter"     # okänd strategi: kortets
    assert card == CARDS["newmont|NEM"]                                                # läser bara


def test_a_tick_moves_the_head_the_order_and_the_counts_in_the_same_run(page):
    """Ett kryss syns i rubriken, ordningen, remsan och st.metric i samma körning som i mätaren — inte ett steg efter."""
    at = page("scorecard", "render_scorecard_page")
    k = "newmont|NEM"
    head = next(e.label for e in _sc_cards(at) if _ticker(e.label) == "NEM")
    assert head.endswith(" · 4/7 kryss")
    at.checkbox(key=f"sc_g_{k}_sakerhetsmarginal").check().run()
    assert not at.exception, at.exception
    card = next(e for e in _sc_cards(at) if _ticker(e.label) == "NEM")
    assert card.label.endswith(" · 5/7 kryss") and "5/7 kryss" in card.markdown[0].value
    for g in ("trigger_definierad", "position_saljregel"):
        at.checkbox(key=f"sc_g_{k}_{g}").check().run()
        assert not at.exception, at.exception
    cards = _sc_cards(at)
    assert [_ticker(c.label) for c in cards][:2] == ["BOL", "NEM"]                    # två klara, ticker
    card = cards[1]
    assert card.label.startswith("✅  NEM · ") and sc.READY in card.markdown[0].value
    assert [b.key for b in card.button if str(b.key).startswith("sc_dec_")] == [f"sc_dec_{k}"]
    assert {m.label: m.value for m in at.metric}["Klara för köp"] == "2"
    assert "2 KLARA FÖR KÖP" in _strip(at, "KÖPGRINDEN · ANTAL")
    stored = at.session_state["scorecard"]["cards"][k]
    assert stored["sakerhetsmarginal"] and stored["trigger_definierad"] and stored["position_saljregel"]


# ── Rubrikerna och de nycklade expandrarna ───────────────────────────────────
def test_scorecard_heads_start_with_the_ticker_and_the_ready_word_is_green(page):
    at = page("scorecard", "render_scorecard_page")
    entries, states = _entries_states()
    by = {e["ticker"]: e for e in entries}
    for card in _sc_cards(at):
        t = _ticker(card.label)
        e, s = by[t], states[by[t]["key"]]
        assert card.label.split("  ", 1)[1].startswith(f"{t} · {e['name'] or '—'} · {', '.join(e['strategies'])} · ")
        if s["ready"]:
            assert card.label.startswith("✅  ") and card.label.endswith(":green[KLAR FÖR KÖP]")
        else:
            assert card.label.startswith("📋  ") and card.label.endswith(f" · {sc.passed(s)}/7 kryss")
            assert ":green[" not in card.label and ":orange[" not in card.label   # ingen tröskelfärg under 7/7


def test_every_card_expander_is_keyed(page):
    at = page("scorecard", "render_scorecard_page")
    calls = {label: key for label, key, _x in at.expander_calls}
    entries, _s = _entries_states()
    for card in _sc_cards(at):
        key = next(e["key"] for e in entries if e["ticker"] == _ticker(card.label))
        assert calls[card.label] == f"sc_card_{key}"
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"})
    calls = {label: key for label, key, _x in at.expander_calls}
    for row in STORES["producers"]["producers"]:
        label = next(e.label for e in at.expander if e.label.startswith(f"{row['ticker']} · "))
        assert calls[label] == f"pr_card_{row['id']}"
    at = page("producers", "render_producers_page", {"sheet": "Royalty C"})
    calls = {label: key for label, key, _x in at.expander_calls}
    for row in STORES["producers"]["royalty"]:
        label = next(e.label for e in at.expander if e.label.startswith(f"{row['ticker']} · "))
        assert calls[label] == f"ro_card_{row['id']}"


def _card_calls(at, prefix: str) -> list:
    """(nyckel, expanded) för korten med prefixet i den senaste körningen."""
    return [(key, x) for _l, key, x in at.expander_log if str(key or "").startswith(prefix)]


@pytest.mark.parametrize("module, fn, kw, sheet, prefix, ident, ticker", [
    ("scorecard", "render_scorecard_page", {}, "scorecard", "sc_card_", "boliden|BOL", "BOL"),
    ("producers", "render_producers_page", {"sheet": "Rick Rule"}, "rule", "pr_card_", "a2", "NEM"),
    ("producers", "render_producers_page", {"sheet": "Royalty C"}, "royalty", "ro_card_", "r2", "WPM"),
])
def test_picking_a_ticker_in_fokus_remounts_its_card_so_it_opens(page, module, fn, kw, sheet, prefix, ident, ticker):
    """Ett val i Fokus ska öppna kortet i webbläsaren, inte bara i protot.

    En nycklad expander (on_change='ignore') får ett block-id av nyckeln, och webbläsaren läser sedan sitt eget
    sparade öppna/stängda läge och bryr sig inte om ett nytt expanded=. AppTest visar bara protot (expanded=True
    även när webbläsaren håller kortet stängt), så testet låser det som får webbläsaren att lyssna: medan ett bolag
    står i fokus har kortet en annan nyckel (suffixet _fokus) än i listan, och "Visa alla" ger tillbaka listans."""
    at = page(module, fn, kw)
    listed = _card_calls(at, prefix)
    assert (f"{prefix}{ident}", False) in listed and all(not k.endswith("_fokus") and not x for k, x in listed)
    at.expander_log.clear()
    at.selectbox(key=f"arken_focus_{sheet}").select(ticker).run()
    assert not at.exception, at.exception
    assert _card_calls(at, prefix) == [(f"{prefix}{ident}_fokus", True)]       # ny nyckel: kortet monteras om, öppet
    at.expander_log.clear()
    at.button(key=f"arken_focus_{sheet}_all").click().run()
    assert not at.exception, at.exception
    assert _card_calls(at, prefix) == listed                                    # listans nycklar igen, hopfällda


def test_rick_rule_heads_keep_the_prefix_and_colour_only_the_verdict_word(page):
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"})
    ranked = {r["row"]["ticker"]: r for r in pr.ranked_producers(STORES["producers"]["producers"])}
    colour = {pr.P_BUY: "green", pr.P_WATCH: "orange", pr.P_PASS: "gray"}
    seen = set()
    for t, r in ranked.items():
        label = next(e.label for e in at.expander if e.label.startswith(f"{t} · "))
        score = r["score"] if r["score"] is not None else "–"
        prefix = f"{t} · {r['row'].get('commodity') or '—'} · {score}/5 p · "
        assert label.startswith(prefix), label
        rest = label[len(prefix):]
        if r["verdict"]:
            assert rest.startswith(f":{colour[r['verdict'].label]}[{r['verdict'].label}]"), label
            seen.add(r["verdict"].label)
        else:
            assert rest == pr.INCOMPLETE, label                             # 'ofullständig' ofärgad
    assert seen == {pr.P_BUY, pr.P_WATCH, pr.P_PASS}                        # fixturen täcker alla tre


def test_royalty_heads_are_unchanged(page):
    at = page("producers", "render_producers_page", {"sheet": "Royalty C"})
    for r in pr.ranked_royalty(STORES["producers"]["royalty"]):
        row = r["row"]
        assert f"{row['ticker']} · nivå {row['level']} · {r['verdict'].label}" in [e.label for e in at.expander]


# ── Mätaren ──────────────────────────────────────────────────────────────────
def _segments(html: str) -> list:
    return [chunk.split("background:", 1)[1].split(";", 1)[0] for chunk in html.split("<span title=")[1:]]


def test_gate_meter_colours_each_segment_by_its_own_value():
    labels = [g.label for g in sc.GATES]
    html = sc.gate_meter([True, False, None, True, "x", False, True], labels)
    assert _segments(html) == [sc.GREEN, sc.RED, sc.METER_HATCH, sc.GREEN, sc.METER_HATCH, sc.RED, sc.GREEN]
    assert "repeating-linear-gradient(45deg,#6b7280 0 3px,transparent 3px 6px)" == sc.METER_HATCH
    assert "3/7 kryss" in html and f"{labels[2]} — Saknas" in html and f"{labels[1]} — ✗" in html
    assert "flex:1 1 0" in html and "height:8px" in html and "gap:2px" in html


def test_gate_meter_has_no_thresholds():
    labels = [g.label for g in sc.GATES]
    for n in range(8):
        html = sc.gate_meter([i < n for i in range(7)], labels)
        assert _segments(html) == [sc.GREEN] * n + [sc.RED] * (7 - n)
        assert sc.AMBER not in html and "orange" not in html
        assert f"{n}/7 kryss" in html


def test_gate_meter_names_exactly_what_gate_state_misses():
    entries, states = _entries_states()
    for e in entries:
        s = states[e["key"]]
        html = sc.gate_meter([s["checks"][g.key] for g in sc.GATES], [g.label for g in sc.GATES])
        if s["missing"]:
            assert f"saknas: {', '.join(s['missing'])}" in html
        else:
            assert "saknas" not in html and "7/7 kryss" in html


# ── Domen överst i kortet ────────────────────────────────────────────────────
def test_the_verdict_sits_in_the_first_markdown_with_the_count(page):
    at = page("scorecard", "render_scorecard_page")
    for card in _sc_cards(at):
        first = card.markdown[0].value
        assert "sc-meter" in first and "fält ifyllda" in first, first
        if card.label.startswith("✅"):
            assert sc.READY in first and sc.NO_TRADE not in first
        else:
            assert sc.NO_TRADE in first and "Saknas för köpbeslut:" in first and "#c44545" in first
        text = " ".join(m.value for m in card.markdown)
        assert text.count(sc.READY) + text.count(sc.NO_TRADE) == 1          # den gamla rutan längst ned är borta
    for sheet, prefix in (("Rick Rule", "BOL · "), ("Royalty C", "WPM · ")):
        at = page("producers", "render_producers_page", {"sheet": sheet})
        card = next(e for e in at.expander if e.label.startswith(prefix))
        first = card.markdown[0].value
        assert "arken-verdict" in first and "fält ifyllda" in first, first
        assert " ".join(m.value for m in card.markdown).count("arken-verdict") == 1


def test_rick_rule_and_royalty_verdicts_are_the_sheets_own_words(page):
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"})
    for r in pr.ranked_producers(STORES["producers"]["producers"]):
        card = next(e for e in at.expander if e.label.startswith(f"{r['row']['ticker']} · "))
        first = card.markdown[0].value
        if r["verdict"]:
            assert f"{r['score']}/5" in first and r["verdict"].label in first and r["verdict"].why in first
        else:
            assert "–/5" in first and "Fyll i pris och kostnad" in first
    at = page("producers", "render_producers_page", {"sheet": "Royalty C"})
    for r in pr.ranked_royalty(STORES["producers"]["royalty"]):
        card = next(e for e in at.expander if e.label.startswith(f"{r['row']['ticker']} · "))
        assert r["verdict"].label in card.markdown[0].value and r["verdict"].why in card.markdown[0].value


def test_registrera_beslut_only_when_ready_and_right_under_the_verdict(page):
    at = page("scorecard", "render_scorecard_page")
    for card in _sc_cards(at):
        top = list(card.children.values())[0]                                 # toppen: först i kortet
        kids = list(top.children.values())
        assert kids[0].type == "markdown" and "fält ifyllda" in kids[0].value
        decisions = [b for b in card.button if str(b.key).startswith("sc_dec_")]
        if card.label.startswith("✅"):
            assert kids[1].type == "button" and kids[1].key == "sc_dec_boliden|BOL"
            assert kids[1].label == "Registrera beslut" and len(decisions) == 1
        else:
            assert len(kids) == 1 and not decisions
    at.button(key="sc_dec_boliden|BOL").click().run()
    assert not at.exception, at.exception
    assert at.session_state["scorecard"]["cards"]["boliden|BOL"]["last_decision"] == date.today().isoformat()


# ── Remsorna ─────────────────────────────────────────────────────────────────
def _strip(at, kicker: str) -> str:
    return next(m.value for m in at.markdown if "wolf-verdict-strip" in m.value and kicker in m.value)


def test_scorecard_strip_counts_are_the_metrics(page):
    at = page("scorecard", "render_scorecard_page")
    metric = {m.label: m.value for m in at.metric}
    strip = _strip(at, "KÖPGRINDEN · ANTAL")
    assert f"{metric['Klara för köp']} KLARA FÖR KÖP" in strip and f"{metric['Kandidater']} kandidater" in strip
    assert metric == {"Kandidater": "7", "Klara för köp": "1"} and sc.NO_TRADE in strip
    assert sc.GREEN in strip


def test_rick_rule_strip_counts_are_the_ranked_verdicts(page):
    rows = STORES["producers"]["producers"]
    ranked = pr.ranked_producers(rows)
    want = Counter(r["verdict"].label if r["verdict"] else pr.INCOMPLETE for r in ranked)
    words = pr.strip_words(ranked, pr.SHEET_RULE)
    assert {w.split(" ", 1)[1]: int(w.split(" ", 1)[0]) for w, _c in words} == dict(want)
    colours = {w.split(" ", 1)[1]: c for w, c in words}
    assert colours == {pr.P_BUY: pr.GREEN, pr.P_WATCH: pr.AMBER, pr.P_PASS: pr.GREY, pr.INCOMPLETE: pr.MUTED}
    assert pr.strip_words(ranked[:1], pr.SHEET_RULE)[0] == (f"1 {pr.P_BUY}", pr.GREEN)
    assert pr.strip_words([], pr.SHEET_RULE) == [(f"0 {pr.P_BUY}", pr.MUTED)]       # köpordet står alltid med
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"})
    strip = _strip(at, "BEDÖMNINGAR · ANTAL")
    for w, _c in words:
        assert w in strip
    caps = " ".join(c.value for c in at.caption)
    assert "Marginal ≥ 40 % ger 2 p, ≥ 25 % ger 1 p." in caps                       # poängnoten står kvar


def test_royalty_strip_counts_are_the_ranked_signals(page):
    ranked = pr.ranked_royalty(STORES["producers"]["royalty"])
    words = pr.strip_words(ranked, pr.SHEET_ROYALTY)
    assert words == [(f"1 {pr.R_BUY}", pr.ROYALTY_COLOR[pr.R_BUY]), (f"1 {pr.R_NEUTRAL}", pr.ROYALTY_COLOR[pr.R_NEUTRAL])]
    at = page("producers", "render_producers_page", {"sheet": "Royalty C"})
    strip = _strip(at, "BEDÖMNINGAR · ANTAL")
    assert all(w in strip for w, _c in words)
    assert _same(at.session_state["producers"], STORES["producers"])               # att öppna skriver inget


# ── Fiatmiljön hopfälld ──────────────────────────────────────────────────────
def _review_tab():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["ARKEN_TEST_ROOT"])
    import wolf_panel
    wolf_panel._tab_review()


def test_the_fiat_environment_is_a_collapsed_background_expander(page, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from fiat_debasement import environment as env
    from ui import nav
    monkeypatch.setattr(env, "current", lambda ccy="USD", compute=None: {
        "currency": "USD", "value": 62.0, "as_of": "2026-10-01", "level": "Neutral"})
    for sub in ("Rick Rule", "Royalty C"):
        at = AppTest.from_function(_review_tab, default_timeout=120)
        at.session_state[nav.STATE_KEY["review"]] = sub
        at.run()
        assert not at.exception, at.exception
        fiat = at.expander[0]                                                  # samma plats: först på sidan
        assert fiat.label == "🌍 Bakgrund — fiatmiljön (styr inga beslut)" and fiat.proto.expanded is False
        assert any("FIAT ENVIRONMENT USD" in m.value for m in fiat.markdown)     # innehållet oförändrat
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    block = src.split('with st.expander("🌍 Bakgrund — fiatmiljön (styr inga beslut)", expanded=False', 1)[1]
    assert block.split("\n", 2)[1].strip() == "render_fiat_environment()"


# ── Stilen ───────────────────────────────────────────────────────────────────
def test_arken_css_targets_the_keys_and_lifts_small_text():
    css = rcard.CSS
    for sel in ('[class*="st-key-sc_card_"]', '[class*="st-key-pr_card_"]', '[class*="st-key-ro_card_"]',
                '[class*="st-key-sc_g_"] [data-testid="stCheckbox"] label', ".st-key-kpis_scorecard",
                '[class*="st-key-sc_post_"]', ".st-key-arken_fiat summary"):
        assert sel in css, sel
    assert "min-height: 48px" in css and "font-size: 12px !important" in css
    assert '[style*="font-size:0.72rem"]' in css and '[style*="font-size:0.68rem"]' in css
    assert "<" not in css.replace("<style>", "")
