"""
Thor-granskningen 2026-10-23, PR C — listan: Momentum följer regimljuset (RÖD = SPÄRRAD, GUL = halv storlek, saknat =
BEVAKA), Viking Nine-dedupe låter den färskaste raden vinna, BERSERK-rader får bolagsnamn ur registret (inte temats
etikett), Momentum-tickrar utan karta märks i stället för att tyst bli '.ST', en oläsbar Gist skiljs från 'ingen
körning', och en trasig karta kraschar inte sidan. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
import test_thor_ui as tu  # noqa: E402
from berserk import signals as bsg  # noqa: E402
from berserk import themes as th  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import ui as tui  # noqa: E402


def _momentum(regime):
    return {(c.ticker): c for c in tc.from_momentum(fx.MOMENTUM, fx.CTX, regime=regime)}


def test_momentum_follows_the_regime_light():
    red = _momentum({"regime": "RÖD"})["ERIC-B.ST"]
    assert red.status_class == tc.SPARRAD and red.status == "rad 1 · setup A · RÖD regim" and "inga nya swingköp" in red.why
    yellow = _momentum({"regime": "GUL"})["ERIC-B.ST"]
    assert yellow.status_class == tc.KOP and "GUL regim" in yellow.status and "halv storlek" in yellow.why
    none = _momentum(None)["ERIC-B.ST"]
    assert none.status_class == tc.BEVAKA and "regimljuset saknas" in none.why
    green = _momentum({"regime": "GRÖN"})
    assert green["ERIC-B.ST"].status_class == tc.KOP and green["ABB.ST"].status_class == tc.BEVAKA   # utan setup A: som förut
    assert tc.summary(list(_momentum({"regime": "RÖD"}).values()))["buy"] == 0


def test_momentum_ticker_prefers_the_rows_yahoo_form_and_marks_guesses():
    top = [{"ticker": "NHY", "yf": "NHY.OL", "rank": 1, "setupA": True, "price": 60.0},
           {"ticker": "NOVO B", "rank": 2, "setupA": True, "price": 700.0}]
    blob = {"generated": fx.GEN, "top": top}
    rows = {c.ticker: c for c in tc.from_momentum(blob, fx.CTX, regime={"regime": "GRÖN"})}
    assert "NHY.OL" in rows and "antagen" not in rows["NHY.OL"].why
    assert "NOVO-B.ST" in rows and "antagen .ST" in rows["NOVO-B.ST"].why                     # gissningen syns


def test_viking_nine_freshest_row_wins_the_dedupe():
    board = {"generated": "2026-10-13T16:50:00+00:00", "board": fx.BOARD["board"],
             "swing": {"segments": {"technology": {"strategy": "Viking Nine", "picks": [
                 {"ticker": "HEXA-B.ST", "name": "Hexagon", "category": "GOLDEN TICKET", "ovtlyr_nine": "9/9",
                  "nine_weighted": 90.0, "entry": 110.0, "stop": 104.0, "rr": 2.4, "bar_date": "2026-10-13"}]}}}}
    scan = {"generated": "2026-10-15T16:50:00+00:00", "rows": [
        {"ticker": "HEXA-B.ST", "name": "Hexagon", "category": "READY", "ovtlyr_nine": "9/9", "nine_weighted": 90.0,
         "segment": "technology", "bar_date": "2026-10-15", "reason": "WAIT FOR CLOSE"}]}
    (c,) = [r for r in tc.build({"board": board, "viking_nine": scan}, today="2026-10-15") if r.strategy == "viking_nine"]
    assert c.status == "READY" and c.status_class == tc.BEVAKA and c.data_asof == "2026-10-15"
    (c2,) = [r for r in tc.build({"board": board, "viking_nine": {"generated": scan["generated"], "rows": []}},
                                 today="2026-10-15") if r.strategy == "viking_nine"]
    assert c2.status == "GOLDEN TICKET" and c2.asof == "2026-10-13"       # skanningen saknar aktien: kartans rad kvar, med sitt datum


def test_berserk_rows_get_the_company_name_not_the_theme_label():
    row = {"ticker": "BOL.ST", "status": "KÖP", "setup": bsg.S2, "label": th.label("koppar"), "theme": "koppar",
           "complex": "basmetaller", "region": "Norden", "close": 300.0, "stop": 280.0, "why": ["x"]}
    blob = {"generated": fx.GEN, "rows": [row]}
    (c,) = tc.from_berserk(blob, fx.CTX)
    assert c.name == "" and c.theme == "koppar"                                                # aldrig 'Koppar' som namn
    (c,) = tc.from_berserk(blob, {**fx.CTX, "names": {"BOL.ST": "Boliden"}})
    assert c.name == "Boliden"
    assert tc.filter_rows([c], search="boliden") == [c]
    rows = tc.build({"berserk": blob}, names={"BOL.ST": "Boliden"})
    assert [r.name for r in rows if r.strategy == "berserk"] == ["Boliden"]


def test_unreadable_gist_is_an_error_not_an_empty_run():
    blobs = tc.load_blobs(loader=lambda name, fb: fb, ca_loader=lambda mode: None, ember_loader=lambda: None)
    assert blobs[tc.UNREADABLE]
    errors = {}
    assert tc.build(blobs, errors=errors) == [] and "gist" in errors
    ok = tc.load_blobs(loader=lambda name, fb: fx.BLOBS.get({v: k for k, v in tc.BLOBS.items()}[name]) or fb,
                       ca_loader=lambda mode: None, ember_loader=lambda: None)
    assert not ok[tc.UNREADABLE]


def test_unreadable_gist_is_said_on_the_page(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    empty = {**{k: None for k in tc.BLOBS}, tc.UNREADABLE: True, "regime": None, "universe": {}, "segment_map": {}, "loaded": "nu"}
    monkeypatch.setattr(tui, "load", lambda loader=None: dict(empty))
    at = AppTest.from_function(tu._script)
    at.run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Gisten gick inte att läsa" in text and "Kör skanningarna i GitHub Actions" not in text


def test_malformed_board_does_not_crash_the_page(monkeypatch):
    from streamlit.testing.v1 import AppTest
    assert tc.segment_context({"board": {"segments": "x"}}) == {"segments": {}, "themes": {}}
    assert tc.segment_context({"board": {"segments": [None, {"nordic": {}}, {"key": "energy", "themes": 3}]}})["segments"] == {
        "energy": {"status": None, "cykel": None, "phase": None, "global_status": None}}
    tu._patch(monkeypatch)
    for bad in ({"board": {"segments": "x"}}, {"board": {"segments": [None, 1]}}):
        monkeypatch.setattr(tui, "load", lambda loader=None, bad=bad: {**tu.DATA, "board": bad})
        at = AppTest.from_function(tu._script)
        at.run(timeout=60)
        assert not at.exception, at.exception
        assert "KANDIDATER" in " ".join(m.value for m in at.markdown)
