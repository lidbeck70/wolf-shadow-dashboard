"""LIDSKJALV PR 18: Dagens besked på I DAG — antalen är exakt sina källor, Saknas när Gisten fallerar (felet en gång),
'🔒 Låst' bakom låset och kvalarnas tal ur samma storlek som Storlek-raden."""
import re
from datetime import datetime

import pandas as pd
import pytest

from tabs import home_today as ht
from thor import candidates as tc

TODAY = pd.Timestamp("2026-10-23")
GIST_ERR = "Gisten gick inte att läsa (nätverk eller token) — inga körningar kunde hämtas; tryck 📡 Uppdatera"


def _c(t, cls, strat="viking_nine", status="KÖP", entry=100.0, stop=92.0, **kw):
    return tc.Candidate(ticker=t, name=t, strategy=strat, status=status, status_class=cls, score=50, entry=entry,
                        stop=stop, **kw)


def _board(window=True, on_way=2):
    plan = {"window_days": 5, "window": [
        {"level": "komplex", "key": "guld", "label": "Guld", "kind": "VÄNDER", "confirmed": True,
         "confirmed_since": "2026-10-22", "deep": [{"ticker": "AAA.ST"}], "long": [{"ticker": "BBB.ST"}]},
        {"level": "tema", "key": "gammal", "label": "Gammal", "kind": "VÄNDER", "confirmed": True,
         "confirmed_since": "2026-09-01", "deep": [], "long": []}]} if window else {}
    return {"turn_plan": plan, "turns": [{"confirmed": True}] + [{"confirmed": False}] * on_way}


def _state(**verdicts):
    return {"swing": {"verdicts": {t: {"verdict": v, "reason": f"skäl {t}"} for t, v in verdicts.items()}}}


ROWS = [_c("K1.ST", tc.KOP), _c("K2.ST", tc.KOP, "wolf"), _c("W1.ST", tc.BEVAKA), _c("W2.ST", tc.BEVAKA),
        _c("W3.ST", tc.BEVAKA), _c("H1.ST", tc.HALL), _c("S1.ST", tc.SALJ)]
FIXTURES = {
    "tom": {"board": {}, "thor": {"rows": []}, "alert_state": {}},
    "full": {"board": _board(), "thor": {"rows": ROWS},
             "alert_state": _state(**{"A.ST": "SELL", "B.ST": "DELVINST — sälj halva", "C.ST": "WATCH",
                                      "D.ST": "HOLD", "E.ST": "BEVAKA"})},
    "inget_fönster": {"board": _board(window=False, on_way=0), "thor": {"rows": ROWS[2:]},
                      "alert_state": _state(**{"D.ST": "HÅLL"})},
    "trasigt": {"board": {"turn_plan": "x", "turns": "x"}, "thor": {"rows": None}, "alert_state": {"swing": "x"}},
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_counts_equal_their_sources(name):
    d = FIXTURES[name]
    due = ["r1", "r2"] if name == "full" else []
    n = ht.counts(d, unlocked=True, today=TODAY, due=due)
    n_buy, n_watch, _top = ht.qualifiers(d["thor"]["rows"])
    words = [w for _t, w, _why in ht.holdings_lines(d["alert_state"])]
    assert n["turn"] == len(ht.turn_cards(d["board"], TODAY))
    assert n["on_way"] == ht.on_the_way(d["board"])
    assert (n["buy"], n["watch"]) == (n_buy, n_watch)
    assert n["sell"] == words.count("SÄLJ") + words.count("MINSKA") and n["vaksam"] == words.count("VAKSAM")
    assert n["due"] == len(due) and n["locked"] is False and n["gist"] is None
    assert all(v is None or isinstance(v, int) for k, v in n.items() if k not in ("locked", "gist"))


def test_counts_of_the_full_fixture_and_the_sentence():
    n = ht.counts(FIXTURES["full"], today=TODAY, due=["r1", "r2"])
    assert (n["turn"], n["on_way"], n["buy"], n["watch"], n["sell"], n["vaksam"], n["due"]) == (1, 2, 2, 3, 2, 2, 2)
    cards = ht.turn_cards(FIXTURES["full"]["board"], TODAY)
    assert ht.besked(n, cards, FIXTURES["full"]["board"]) == ("⚡ VÄNDNING — Guld · dag 1 av 5", False)
    calm = ht.counts(FIXTURES["inget_fönster"], today=TODAY, due=[])            # bara BEVAKA och HÅLL: allt noll
    assert ht.besked(calm, [], FIXTURES["inget_fönster"]["board"]) == (ht.NOTHING_TODAY, False)
    buys = {**FIXTURES["inget_fönster"], "thor": {"rows": ROWS}}
    n_buys = ht.counts(buys, today=TODAY, due=[])
    assert ht.besked(n_buys, [], buys["board"]) == ("Ingen vändning i köpfönstret", False)


def test_all_zero_says_nothing_today_but_never_about_an_unknown_count():
    n = ht.counts(FIXTURES["tom"], today=TODAY, due=[])
    assert ht.besked(n, [], {}) == (ht.NOTHING_TODAY, False) == ("Inga KÖP, SÄLJ eller vändningar i dag", False)
    on_way = ht.counts({**FIXTURES["tom"], "board": _board(window=False, on_way=1)}, today=TODAY, due=[])
    assert ht.besked(on_way, [], _board(window=False, on_way=1))[0] == "Ingen vändning i köpfönstret · 1 på väg"
    locked = ht.counts(FIXTURES["tom"], unlocked=False, today=TODAY, due=[])
    assert locked["sell"] is None and locked["locked"] is True
    assert ht.besked(locked, [], {})[0] == "Ingen vändning i köpfönstret"         # SÄLJ okänt: inget "inga SÄLJ"
    no_board = ht.counts({"thor": {"rows": []}, "alert_state": {}}, today=TODAY, due=[])
    assert no_board["turn"] is None                                                 # kartan saknas: Saknas, inte 0
    # …och meningen säger då inte "Ingen vändning" bredvid Saknas-rutan, utan att kartan saknas (grått)
    assert ht.besked(no_board, [], None) == (ht.MAP_MISSING, True)
    assert ht.MAP_MISSING == "Saknas — Valhalls karta gick inte att läsa"
    locked_no_board = ht.counts({"thor": {"rows": []}, "alert_state": {}}, unlocked=False, today=TODAY, due=[])
    assert ht.besked(locked_no_board, [], None) == (ht.MAP_MISSING, True)


def test_home_css_wraps_turn_buttons_and_keeps_list_rows_flush():
    """Kolumnregeln (≤1023 px) staplar varje stHorizontalBlock i home_cols — vändningens knapprad har högre
    specificitet och radbryts; listradernas gap 0 vinner över sidans 8 px-regel."""
    import re
    from tabs import home
    css = home._CSS
    assert re.search(r'\.st-key-home_page \.st-key-home_cols \[class\*="st-key-home_turnrow_"\][^{]*\{[^}]*'
                     r'flex-direction: row; flex-wrap: wrap', css)
    assert '.st-key-home_page [data-testid="stVerticalBlock"].st-key-home_kval' in css
    assert css.index('[data-testid="stVerticalBlock"].st-key-home_kval') > css.index(':not([class*="st-key-tapc_"])')


def test_counts_read_the_routines_when_the_page_has_not(monkeypatch):
    import routines
    days = []
    monkeypatch.setattr(routines, "due_today", lambda day: days.append(day) or ["a", "b", "c"])
    assert ht.counts(FIXTURES["tom"], today=TODAY)["due"] == 3 and days == [TODAY.date()]
    monkeypatch.setattr(routines, "due_today", lambda day: 1 / 0)
    assert ht.counts(FIXTURES["tom"], today=TODAY)["due"] is None
    assert ht.counts(FIXTURES["tom"], today=TODAY, due=None)["due"] is None


def test_gist_failure_gives_saknas_and_the_reason_once():
    d = {"board": None, "thor": {"rows": [], "errors": {"gist": GIST_ERR}}, "alert_state": None}
    n = ht.counts(d, today=TODAY, due=[])
    assert n["buy"] is None and n["watch"] is None and n["turn"] is None and n["gist"] == GIST_ERR
    assert ht.besked(n, [], None) == ("Saknas — Gisten gick inte att läsa", True)
    reason = ht.gist_reason(GIST_ERR)
    assert reason == "(nätverk eller token) — inga körningar kunde hämtas; tryck 📡 Uppdatera"
    assert "Gisten gick inte att läsa" not in reason
    assert ht.gist_reason("timeout") == "timeout" and ht.gist_error({"thor": {"errors": "x"}}) is None


def test_counts_do_no_arithmetic_on_prices():
    src = ht.counts.__code__.co_names
    assert not {"entry", "stop", "target", "size", "_sized"} & set(src)


# ── size_parts: samma storlek som size_line ──────────────────────────────────
def _numbers(parts):
    return [m for p in parts for m in re.findall(r"\d[\d ]*(?:\.\d+)? (?:st|kr)", p)]


@pytest.mark.parametrize("c", [
    _c("ABC.ST", tc.KOP, "viking_nine"), _c("DEF.ST", tc.KOP, "wolf", entry=88.2, stop=82.5),
    _c("GHI.OL", tc.KOP, "momentum", status="KÖP · GUL regim", entry=142.0, stop=128.0),
    _c("JKL.ST", tc.KOP, "ember", entry=64.0, stop=60.1), _c("NEM", tc.KOP, "deep", entry=31.5, stop=27.0),
    _c("RUL.ST", tc.KOP, "rule", entry=50.0, stop=45.0)])
def test_size_parts_are_the_numbers_of_the_size_line(c):
    line, parts = ht.size_line(c, 100000), ht.size_parts(c, 100000)
    assert parts, line
    halved = "GUL regim" in c.status
    for num in _numbers(parts):
        if halved and num.endswith(" st"):
            # sk.halved halverar antalet aktier; Storlek-radens formeltext visar talet före halveringen + "halverat"
            shares = {int(x) for x in re.findall(r"(\d+) st", line)}
            assert any(m // 2 == int(num[:-3]) for m in shares), (num, line)
        else:
            assert num in line, (num, line)
    if halved:
        assert "halverat" in parts and "halverat" in line
    if not _numbers(parts):                                     # varken risk-% eller tak: köp och stopp
        assert parts[:2] == [f"köp {c.entry:.2f}", f"stopp {c.stop:.2f}"] and f"köp {c.entry:.2f}" in line


def test_size_parts_for_the_viking_nine_row_and_missing_levels():
    from thor import storlek as sk
    c = _c("ABC.ST", tc.KOP, "viking_nine")
    sz = sk.size(100.0, 92.0, 100000, sk.risk_pct("viking_nine"), sk.cap_pct("viking_nine"))
    risk = f"risk {sz['risk_amt']:,.0f} kr".replace(",", " ")
    assert ht.size_parts(c, 100000)[:3] == [f"{sz['risk_shares']} st", risk, f"tak {sz['cap_shares']} st"]
    bad = _c("X", tc.KOP, entry=None)
    assert ht.size_parts(bad, 100000) == [] and "nivåer saknas" in ht.size_line(bad, 100000)


# ── Sidan: rutorna, Saknas en gång, låset ────────────────────────────────────
def _page(d, unlocked=True, now=datetime(2026, 10, 23, 8, 0)):
    from streamlit.testing.v1 import AppTest

    def app():
        from tabs import home
        home.tab_home()

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["home_today"] = {**d, "loaded": now.strftime("%Y-%m-%d %H:%M")}
    at.session_state["settings"] = {"capital": 100000.0}
    at.session_state["holdings_unlocked"] = unlocked
    return at


@pytest.fixture
def quiet(monkeypatch):
    """Ingen laddning och ingen klocka ur servern: sidan ritar det som ligger i sessionen."""
    import routines
    monkeypatch.setattr(ht, "load", lambda loader=None: (_ for _ in ()).throw(AssertionError("laddade")))
    monkeypatch.setattr(ht, "now_se", lambda: datetime(2026, 10, 23, 8, 30))
    monkeypatch.setattr(routines, "due_today", lambda day: [])


def _text(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def test_gist_failure_on_the_page_says_saknas_and_prints_the_error_once(quiet):
    at = _page({"board": None, "thor": {"rows": [], "errors": {"gist": GIST_ERR}}, "alert_state": None}).run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "Saknas — Gisten gick inte att läsa" in text
    assert text.count("Gisten gick inte att läsa") == 1
    assert "inga körningar kunde hämtas" in text and text.count("inga körningar kunde hämtas") == 1


def test_tiles_show_the_counts_and_open_their_pages(quiet):
    d = {**FIXTURES["full"], "board": _board()}
    at = _page(d).run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "Dagens besked · antal i dag" in text and "⚡ VÄNDNING — Guld · dag 1 av 5" in text
    assert "3 BEVAKA" in text and "2 VAKSAM" in text and "rutiner i dag" in text and "2 på väg" in text
    for key, slug in (("home_tile_turn", "odin/kartan"), ("home_tile_buy", "thor"),
                      ("home_tile_sell", "portfolio/positioner"), ("home_tile_due", "more/regler-guider/arshjulet"),
                      ("home_laget", "odin/laget"), ("home_holdings", "portfolio/positioner"),
                      ("home_goal", "portfolio/mot-malet"), ("home_thor", "thor")):
        at.button(key=key).click().run()
        assert not at.exception, at.exception
        assert at.session_state["nav_goto"] == slug, key
    from ui import nav
    assert nav.resolve("more/regler-guider/arshjulet") == ["more", "Regler & Guider", "🗓 ÅRSHJULET"]
    labels = {b.key: b.label for b in at.button}
    assert labels["home_refresh"] == "📡 Uppdatera" and labels["home_laget"] == "Läget i Odin →"
    assert labels["home_buy_viking_nine_K1.ST"] == "K1.ST → Thor (Viking Nine)"
    assert labels["home_turn_0_deep_AAA.ST"] == "Deep Contrarian: AAA.ST → Thor"
    assert labels["home_holdings"] == "Positioner →" and labels["home_goal"] == "Mot målet →"


def test_locked_holdings_show_laast_and_no_holding(quiet):
    d = {**FIXTURES["full"], "board": _board()}
    at = _page(d, unlocked=False).run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "🔒 Låst" in text and "🔒 Innehaven är låsta — lås upp under PORTFÖLJ → Positioner" in text
    assert "VAKSAM" not in text and "skäl" not in text                    # inga innehav, inga skäl, inget antal
    assert ht.counts(d, unlocked=False, today=TODAY, due=[])["sell"] is None


def test_home_block_is_a_thin_wrapper_over_section():
    import inspect

    from tabs import home
    src = inspect.getsource(home._block)
    assert "section(" in src and "font-size:10px" not in inspect.getsource(home)
    whole = inspect.getsource(home)
    assert "ODIN · ODIN" not in whole and "{nav.TOP_LABEL[key]} · {title}" not in whole


# ── Stilattributen: typsnittstoken med citattecken bryter inte style='…' ──────
def test_no_single_quoted_style_interpolates_a_font_token():
    """FONT_UI/FONT_MONO bär egna ' — inne i style='…' stänger de attributet och stilen tappas."""
    import inspect

    from tabs import home
    from ui import routine
    for mod in (home, ht, routine):
        bad = re.findall(r"style='[^']*\{FONT_", inspect.getsource(mod))
        assert not bad, (mod.__name__, bad)


def _styles(html: str) -> list:
    """(tagg, attribut) för varje element i en HTML-snutt, som webbläsaren läser dem."""
    from html.parser import HTMLParser

    out = []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            out.append((tag, dict(attrs)))
    P().feed(html)
    return out


def test_rendered_styles_keep_their_fonts_and_the_turn_headline_its_weight(quiet):
    d = {**FIXTURES["full"], "board": _board()}
    at = _page(d).run()
    assert not at.exception, at.exception
    junk = {"space", "grotesk',", "jetbrains", "mono',", "sans-serif;", "monospace;"}
    head = None
    for m in at.markdown:
        for tag, attrs in _styles(str(m.value)):
            assert not junk & set(attrs), (tag, attrs)                  # inga attribut ur ett avbrutet style
            style = attrs.get("style") or ""
            for decl in re.findall(r"font-family:([^;]*)", style):
                assert decl.strip(), (tag, style)                       # font-family fick sitt värde
            if "VÄNDNING" in str(m.value) and "font-size:18px" in style:
                head = style
    assert head and "font-weight:900" in head and "Space Grotesk" in head
    assert "JetBrains Mono" in _text(at) and "?p=odin/laget" in _text(at)
