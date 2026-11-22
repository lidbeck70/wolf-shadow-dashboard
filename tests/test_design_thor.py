"""
LIDSKJALV PR 20 — Thor: listan och analysen beslut först (docs/design_lidskjalv.md §4.5, §4.6, §3.5.14, §8).

  · listläget: huvudraden (🔨 THOR · 📡 thor_refresh · ⓘ 'Om listan' med underraden och fotnoten ordagrant), hämtad-
    raden, lägets remsa, sökraden (thor_ticker och ⚙ Filter med thor_pick/thor_segment), pillsraden med 🌍 Marknad
    och raderna som tryckkort med samma nycklar, etiketter och kort-HTML som förut — i den ordningen
  · analysläget: navraden (◀ thor_back och thor_ticker) före huvudkortet, och ingen lägesremsa eller Körningar per
    strategi ovanför verdiktet — listans ram står under analysen, med 📡 som läser om före analysen
  · Gistens läsfel står EN gång på sidan, i båda lägena; ett körningsfel står en gång (i Körningar per strategi)
  · ⚙ Filter visar hur många av Screening och Segment som inte står på förvalet; popovrarnas innehåll (ritat i en
    portal utanför sidan) får sina 44 px på widgetarnas egna nycklar
  · Morgonrundan öppnar Thor i listläget (routine.advance stänger en öppen analys) — bara navigering
  · snapshot_items: bara 1 mån och 3 mån färgas efter tecknet; ett saknat tal blir Saknas
Inget nätverk: test_thor_ui:s stubbar.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_ui as tu  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import cards  # noqa: E402
from thor import ui as tui  # noqa: E402
from ui import routine, tokens  # noqa: E402

GIST = "Gisten gick inte att läsa"


def _app(monkeypatch, state=None, data=None):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    if data is not None:
        monkeypatch.setattr(tui, "load", lambda loader=None: dict(data))
    at = AppTest.from_function(tu._script, default_timeout=120)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at


def _flat(at) -> list:
    """Sidans noder i ritordning: ('block', typ, id, etikett) och ('el', typ, nyckel, text) — utan stilblocken."""
    out = []

    def walk(node):
        kids = getattr(node, "children", None)
        if isinstance(kids, dict):
            proto = getattr(node, "proto", None)
            kind, label, bid = node.type, str(getattr(node, "label", "") or ""), ""
            try:                                       # allmänna block (behållare, popover) bär BlockProto med id
                bid = proto.id
                if proto.WhichOneof("type") == "popover":
                    label = proto.popover.label
            except (AttributeError, ValueError):       # Expander, Tab, Column har egna proto utan id
                pass
            out.append(("block", kind, bid, label))
            for i in sorted(kids):
                walk(kids[i])
        else:
            try:
                text = node.label if getattr(node, "label", None) else node.value
            except Exception:                          # element utan värde (t.ex. ett tomt block)
                text = ""
            text = text if isinstance(text, str) else ""
            if "<style>" not in text:                  # sidans dolda stilblock är ingen ritad rad
                out.append(("el", node.type, getattr(node, "key", None) or "", text))
    walk(at.main)
    return out


def _idx(flat, pred) -> int:
    return next(i for i, n in enumerate(flat) if pred(n))


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown if "<style>" not in m.value)


# ── Listläget (§4.5) ─────────────────────────────────────────────────────────
def test_list_mode_draws_head_strip_search_and_tap_rows_in_order(monkeypatch):
    at = _app(monkeypatch)
    at.run()
    assert not at.exception, at.exception
    flat = _flat(at)
    head = _idx(flat, lambda n: n[0] == "block" and n[2].endswith("-head_thor"))
    title = _idx(flat, lambda n: n[1] == "markdown" and "🔨 THOR" in n[3])
    refresh = _idx(flat, lambda n: n[2] == "thor_refresh")
    about = _idx(flat, lambda n: n[1] == "popover" and n[2].endswith("-thor_about"))
    meta = _idx(flat, lambda n: n[1] == "markdown" and "hämtad nu · 32 rader" in n[3])
    strip = _idx(flat, lambda n: n[1] == "markdown" and "thor-lage" in n[3])
    field = _idx(flat, lambda n: n[2] == "thor_ticker")
    filt = _idx(flat, lambda n: n[1] == "popover" and n[2].endswith("-thor_filter_pop"))
    pick = _idx(flat, lambda n: n[2] == "thor_pick")
    pills = _idx(flat, lambda n: n[2] == "thor_filters")
    markets = _idx(flat, lambda n: n[1] == "popover" and n[2].endswith("-thor_market_pop"))
    first = _idx(flat, lambda n: str(n[2]).startswith("thor_open_"))
    fresh = _idx(flat, lambda n: n[1] == "expander" and n[3].startswith("Körningar per strategi"))
    assert head < title < refresh < about < meta < strip < field < filt < pick < pills < markets < first < fresh
    assert flat[about][3] == "ⓘ" and flat[filt][3] == "⚙ Filter" and flat[markets][3] == "🌍 Marknad"
    pop = " ".join(n[3] for n in flat[about + 1:about + 4])                       # Om listan: ordagrant
    assert tui.SUBTITLE in pop and tui.FOOTNOTE in pop
    assert _text(at).count(tui.FOOTNOTE) == 1                                      # fotnoten bara i ⓘ
    assert "🌍 Marknad" in flat[markets][3] and flat[_idx(flat, lambda n: n[2] == "thor_markets")][0] == "el"


def test_rows_are_tap_cards_with_the_old_keys_labels_and_card_html(monkeypatch):
    at = _app(monkeypatch)
    at.run()
    rows = {(c.strategy, c.ticker): c for c in at.session_state["thor_data"]["rows"]}
    flat = _flat(at)
    opens = [i for i, n in enumerate(flat) if str(n[2]).startswith("thor_open_")]
    assert len(opens) == tui.PAGE
    for i in opens:
        key, label = flat[i][2], flat[i][3]
        assert label == "Öppna →"
        assert flat[i - 2][0] == "block" and flat[i - 2][2].endswith(f"-tapc_{key}")   # kortet ÄR knappen
        assert flat[i - 1][1] == "markdown"
        strategy, ticker = next((s, t) for (s, t) in rows if key == f"thor_open_{s}_{t}")
        assert flat[i - 1][3] == cards.candidate_card(rows[(strategy, ticker)])  # kortets HTML oförändrad
    at.button(key="thor_open_berserk_BOL.ST").click().run()
    assert at.session_state["thor_ticker"] == "BOL.ST" and at.session_state["thor_strategy"] == "BERSERK"


def test_filter_popover_counts_non_default_choices_and_keeps_the_keys(monkeypatch):
    at = _app(monkeypatch)
    at.run()
    assert at.selectbox(key="thor_pick").value == tui.ALL and at.selectbox(key="thor_segment").value == tui.SEGMENT_ALL
    at.selectbox(key="thor_segment").select("technology").run()
    labels = [n[3] for n in _flat(at) if n[1] == "popover"]
    assert "⚙ Filter · 1" in labels
    at.selectbox(key="thor_pick").select("Viking Nine").run()
    assert "⚙ Filter · 2" in [n[3] for n in _flat(at) if n[1] == "popover"]
    at.pills(key="thor_markets").set_value(["Norden"]).run()
    assert "🌍 Marknad · 1" in [n[3] for n in _flat(at) if n[1] == "popover"]


def _css_selectors(css: str) -> list:
    """(selektor, regelkropp) för varje enskild selektor i ett stilblock — kommentarer bort, @media-skalet öppnas."""
    import re
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(s.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css) for s in sel.split(",")]


def test_popover_contents_are_44px_by_the_widgets_own_keys():
    """⚙ Filter och 🌍 Marknad ritar innehållet i en portal (stPopoverBody) utanför sidan — varken inne i
    .st-key-thor_page eller i triggerns nyckel. Bara widgetarnas egna nycklar följer med dit, så 44 px-reglerna (§8.7)
    för marknadspillsen, Screening/Segment-fälten med ⌄ och Segments hjälpikon måste börja på dem."""
    rules = _css_selectors(tui._CSS)

    def sized(key: str, part: str, prop: str = "min-height: 44px") -> bool:
        return any(s.startswith(f".st-key-{key} ") and part in s and prop in body for s, body in rules)

    assert sized("thor_markets", '[data-testid="stButtonGroup"] button')
    for k in ("thor_pick", "thor_segment"):
        assert sized(k, ".react-aria-ComboBox > div")
        assert sized(k, ".react-aria-ComboBox button") and sized(k, ".react-aria-ComboBox button", "min-width: 44px")
    assert sized("thor_segment", '[data-testid="stTooltipHoverTarget"] button', "min-width: 44px")
    assert sized("thor_segment", '[data-testid="stTooltipHoverTarget"] button')


def test_round_pill_sits_in_the_head_row_and_the_footer_last(monkeypatch):
    monkeypatch.setattr(routine, "_today", lambda: "2026-11-21")
    at = _app(monkeypatch, {"morning": {"date": "2026-11-21", "seen": ["home"]}})
    at.run()
    flat = _flat(at)
    head = _idx(flat, lambda n: n[0] == "block" and n[2].endswith("-head_thor"))
    strip = _idx(flat, lambda n: n[1] == "markdown" and "thor-lage" in n[3])
    pill = _idx(flat, lambda n: n[2] == "morning_top_thor")
    assert head < pill < strip
    assert [b.key for b in at.button][-1] == "morning_next_thor"


# ── Analysläget (§4.6) ──────────────────────────────────────────────────────
def test_analysis_draws_the_nav_row_first_and_no_list_chrome_above_the_verdict(monkeypatch):
    at = _app(monkeypatch, {"thor_ticker": "bol.st"})
    at.run()
    assert not at.exception, at.exception
    flat = _flat(at)
    nav = _idx(flat, lambda n: n[0] == "block" and n[2].endswith("-thor_nav"))
    back = _idx(flat, lambda n: n[2] == "thor_back")
    field = _idx(flat, lambda n: n[2] == "thor_ticker")
    header = _idx(flat, lambda n: n[1] == "markdown" and "finns i dag i" in n[3])
    strategy = _idx(flat, lambda n: n[2] == "thor_strategy")
    verdict = _idx(flat, lambda n: n[1] == "markdown" and "VAD SKA JAG GÖRA?" in n[3])
    storlek = _idx(flat, lambda n: n[1] == "markdown" and "thor-storlek" in n[3])
    watch = _idx(flat, lambda n: n[2] == "thor_watch_add")
    snap = _idx(flat, lambda n: n[1] == "markdown" and "wolf-kstrip" in n[3] and "thor-lage" not in n[3])
    tabs = _idx(flat, lambda n: n[1] == "tab")
    kand = _idx(flat, lambda n: n[1] == "expander" and n[3] == "◀ Kandidater (32)")
    strip = _idx(flat, lambda n: n[1] == "markdown" and "thor-lage" in n[3])
    fresh = _idx(flat, lambda n: n[1] == "expander" and n[3].startswith("Körningar per strategi"))
    refresh = _idx(flat, lambda n: n[2] == "thor_refresh")
    assert nav < back < field < header < strategy < verdict < storlek < watch < snap < tabs < kand < strip < fresh
    assert fresh < refresh                                                         # 📡 nås under analysen
    assert flat[back][3] == "◀ Tillbaka till listan" and at.button(key="thor_back").proto.type == "tertiary"
    above = flat[:verdict]
    assert not [n for n in above if "thor-lage" in n[3] or n[3].startswith("Körningar per strategi")
                or n[2] in ("thor_refresh", "morning_top_thor")]
    assert not [n for n in flat if n[0] == "block" and n[2].endswith("-head_thor")]   # ingen listrubrik
    assert not [b for b in at.button if str(b.key).startswith("morning_")]
    assert at.text_input(key="thor_ticker").value == "bol.st"                      # fältet behåller värdet


def test_the_ticker_field_keeps_its_value_between_the_modes(monkeypatch):
    at = _app(monkeypatch)
    at.run()
    at.text_input(key="thor_ticker").set_value("BOL.ST").run()
    assert not at.exception and "VAD SKA JAG GÖRA?" in _text(at)
    assert at.text_input(key="thor_ticker").value == "BOL.ST"
    at.button(key="thor_back").click().run()
    assert at.text_input(key="thor_ticker").value == "" and "32 KANDIDATER" in _text(at)


def test_refresh_below_the_analysis_reloads_before_it_is_drawn(monkeypatch):
    calls = []
    at = _app(monkeypatch, {"thor_ticker": "bol.st"})
    monkeypatch.setattr(tui, "load", lambda loader=None: calls.append(1) or dict(tu.DATA))
    at.run()
    assert len(calls) == 1
    at.run()
    assert len(calls) == 1                                                         # inget läses om av sig självt
    at.button(key="thor_refresh").click().run()
    assert not at.exception and len(calls) == 2 and "VAD SKA JAG GÖRA?" in _text(at)
    assert "_thor_reload" not in at.session_state


# ── Felen en gång (§3.5.19, §8.15) ───────────────────────────────────────────
def _unreadable():
    return {**{k: None for k in tc.BLOBS}, tc.UNREADABLE: True, "regime": None, "universe": {}, "segment_map": {},
            "loaded": "nu"}


def test_the_gist_error_is_printed_exactly_once_in_both_modes(monkeypatch):
    at = _app(monkeypatch, data=_unreadable())
    at.run()
    assert not at.exception, at.exception
    text = _text(at)
    assert text.count(GIST) == 1 and "Kör skanningarna i GitHub Actions" not in text
    warn = next(m.value for m in at.markdown if "thor-warn" in m.value)
    assert GIST in warn                                                            # i den bärnstensfärgade raden
    at = _app(monkeypatch, {"thor_ticker": "bol.st"}, data=_unreadable())
    at.run()
    assert not at.exception, at.exception
    assert _text(at).count(GIST) == 1


def test_a_failed_run_is_said_once_and_kept_in_the_freshness_box(monkeypatch):
    err = "yfinance: connection refused"
    data = {**tu.DATA, "arc": {**tu.DATA["arc"], "wolf": {"generated": "2026-10-14T16:50", "rows": [], "error": err}}}
    at = _app(monkeypatch, data=data)
    at.run()
    assert not at.exception, at.exception
    assert _text(at).count(err) == 1
    box = next(e for e in at.expander if e.label.startswith("Körningar per strategi"))
    assert err in " ".join(m.value for m in box.markdown) and "Wolf" in " ".join(m.value for m in box.markdown)
    assert any("körningar gamla/saknas/fel — se Körningar per strategi nedan" in m.value for m in at.markdown)
    at.selectbox(key="thor_pick").select("Wolf").run()
    text = _text(at)
    assert text.count(err) == 1 and "Körningen fallerade för Wolf" in text      # tomma listan: namnet, inte felet


def test_a_broken_file_is_still_on_the_page_inside_the_box(monkeypatch):
    at = _app(monkeypatch, data={**tu.DATA, "screens": {"generated": "2026-10-14", "screens": 7}})
    at.run()
    box = next(e for e in at.expander if e.label.startswith("Körningar per strategi"))
    assert "Kunde inte läsa screens (AttributeError" in " ".join(m.value for m in box.markdown)


# ── Morgonrundan öppnar listan (PR 19:s öppna punkt) ─────────────────────────
def test_the_round_opens_thor_in_list_mode(monkeypatch):
    monkeypatch.setattr(routine, "_today", lambda: "2026-11-21")
    state = {"thor_ticker": "BOL.ST"}
    routine.advance("laget", state)
    assert state["nav_goto"] == "thor" and state["thor_ticker"] == ""
    state = {"thor_ticker": "BOL.ST"}
    routine.advance("home", state)                                                 # nästa steg är Läget: orört
    assert state["thor_ticker"] == "BOL.ST" and state["nav_goto"] == "odin/laget"
    state = {}
    routine.advance("laget", state)
    assert "thor_ticker" not in state                                              # inget läggs till


# ── Ögonblicksbilden (§4.6) ─────────────────────────────────────────────────
def test_snapshot_items_sign_colour_only_the_changes():
    snap = {"price": 312.5, "as_of": "2026-10-14", "ret_1m_pct": 4.2, "ret_3m_pct": -1.0, "atr_pct": 2.1,
            "dist_ema50_pct": 3.0, "dist_ema200_pct": -2.0, "from_high_pct": -8.0}
    items = cards.snapshot_items(snap)
    assert [i[0] for i in items] == ["Kurs", "1 mån", "3 mån", "ATR", "EMA50", "EMA200", "52 v"]
    assert items[0][1] == "312.50" and items[1][1] == "+4.2 %" and items[6][1] == "-8.0 %"
    signed = {i[0]: i[4] for i in items if len(i) > 4}
    assert signed == {"1 mån": 4.2, "3 mån": -1.0}                                 # bara förändringarna
    assert {i[2] for i in items} == {tokens.TEXT}                                  # nivåer och avstånd i TEXT
    from ui.components import kpi_strip
    html = kpi_strip(items)
    assert tokens.GREEN in html and tokens.RED in html
    gap = cards.snapshot_items({**snap, "ret_3m_pct": None})
    assert gap[2][1] is None and "Saknas" in kpi_strip(gap) and "None" not in kpi_strip(gap)


def test_lage_strip_keeps_the_items_and_shows_a_missing_map_as_saknas():
    items = tui.lage_items(tu.DATA, tu._rows())
    html = cards.lage_strip(items)
    assert "Kandidater 32 · 13 KÖP" in html and html.count("wolf-tile") == 4
    empty = cards.lage_strip(tui.lage_items({}, []))
    assert "Saknas" in empty and "Norden —" in empty                               # källans ord i aria-label


def test_no_text_under_12px_in_the_page_source():
    import re
    src = open(tui.__file__, encoding="utf-8").read()
    page = src[src.index("# ── Sidan"):src.index("# ── Sidoflikarna")]
    sizes = [float(v) * (16 if u == "rem" else 1) for v, u in re.findall(r"font-size:\s*([0-9.]+)(px|rem)\b", page)]
    assert sizes and min(sizes) >= 12
    assert "st.columns([5, 1])" not in page and "c1.button(\"📡 Uppdatera\"" not in page


def _vn_data():
    """test_thor_ui:s data med Nordens marknadslager svagt (1/3) — då skriver vn_market_notes en rad."""
    vn = dict(tu.DATA.get("viking_nine") or {})
    markets = dict(vn.get("markets") or {})
    nd = dict(markets.get("norden") or {})
    nd["market_layer"] = [{"label": "Trend", "status": "FAIL"}, {"label": "Signal", "status": "FAIL"},
                          {"label": "Breadth", "status": "PASS"}]
    markets["norden"] = nd
    vn["markets"] = markets
    return {**tu.DATA, "viking_nine": vn}


def test_viking_nines_market_notes_stay_visible_once_outside_the_freshness_box(monkeypatch):
    """Marknadsläget är inte ett fel: raden står synlig under listan (inte inne i Körningar per strategi), en gång."""
    data = _vn_data()
    lines = tc.vn_market_notes(data["viking_nine"])
    assert lines
    for state in ({}, {"thor_ticker": "ERIC-B.ST"}):
        at = _app(monkeypatch, state=state, data=data)
        at.run()
        assert not at.exception, at.exception
        flat = _flat(at)
        fresh = _idx(flat, lambda n: n[1] == "expander" and n[3].startswith("Körningar per strategi"))
        hits = [i for i, n in enumerate(flat) if n[1] == "markdown" and lines[0][:40] in n[3]]
        assert len(hits) == 1 and hits[0] < fresh, (state, hits, fresh)


def test_the_stale_label_rule_beats_the_shells_important_expander_colour():
    """Skalets expanderfärg är !important (ui/css.py) — bärnstensetiketten måste vara skarpare och !important."""
    css = tui._CSS_STALE
    assert ".st-key-thor_fresh [data-testid=\"stExpander\"] summary p" in css
    assert tokens.AMBER in css and "!important" in css
