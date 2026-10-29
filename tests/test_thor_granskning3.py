"""
Thor-granskningen 3 (T1–T8, T10, T12, T14–T16): rätt bolag när en bar ticker och en nordisk notering delar bas,
ärlig storlek (planering utan köpsignal, Logga köp bara för färska KÖP-rader), London i pence, arkens bara nordiska
tickers, temafiltret för rader utan eget tema, gamla körningar märkta i analysblocket, Viking Nines USA-del behåller
sitt datum, Valhall-temat ur sin egen segmentrad, Asymmetrins cache och läge, alla registrets suffix, hemmamarknadens
etikett och momentumrader utan ticker. Inga regler ändras. Inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
import test_thor_ui as tu  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import cards  # noqa: E402
from thor import dossier as td  # noqa: E402
from thor import ticker as tt  # noqa: E402
from thor import ui as tui  # noqa: E402
from universe import registry as ur  # noqa: E402


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _page(monkeypatch, rows=None, data=None, today="2026-10-15"):
    """Thor-sidan med test_thor_ui:s stubbar; raderna byggs med ett fast datum (eller ges direkt)."""
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    if data is not None:
        monkeypatch.setattr(tui, "load", lambda loader=None: dict(data))
    monkeypatch.setattr(tui, "rebuild", lambda d: {**d, "errors": {},
                                                   "rows": list(rows) if rows is not None else tc.build(d, today=today)})
    return AppTest.from_function(tu._script)


# ── T1: en bar rad vinner över den nordiska noteringen med samma bas ──────────
TEL_ROWS = tc.mark_stale(tc.dedupe([
    tc.Candidate("TEL", "TE Connectivity", "wolf", "MODERATE", tc.BEVAKA, score=56.0, score_text="70/125",
                 asof="2026-10-14"),
    tc.Candidate("TEL.OL", "Telenor", "wolf", "ENTRY · BULL", tc.KOP, score=88.0, score_text="110/125", entry=150.0,
                 stop=140.0, nordic=True, asof="2026-10-14")]), today="2026-10-15")


def test_bare_row_wins_over_the_nordic_namesake():
    assert [c.ticker for c in TEL_ROWS] == ["TEL.OL", "TEL"]                       # KÖP först i listan
    assert [c.ticker for c in td.candidates_for(TEL_ROWS, "TEL")] == ["TEL.OL", "TEL"]   # kontraktet oförändrat
    assert [c.ticker for c in td.bare_rows(td.candidates_for(TEL_ROWS, "TEL"), "TEL")] == ["TEL"]
    assert td.bare_rows(TEL_ROWS, "TEL.OL") == [] and td.bare_rows(TEL_ROWS, "") == []
    d = td.assemble("TEL", rows=TEL_ROWS)
    assert d["ticker"]["yf"] == "TEL" and [c["ticker"] for c in d["candidates"]] == ["TEL"]
    assert td.assemble("eric b", rows=[tc.Candidate("ERIC-B.ST", "Ericsson", "momentum", "rad 1", tc.KOP)])[
        "ticker"]["yf"] == "ERIC-B.ST"                                              # utan bar rad: som förut


def test_opening_the_bare_row_shows_its_own_company(monkeypatch):
    at = _page(monkeypatch, rows=TEL_ROWS)
    at.run(timeout=120)
    at.button(key="thor_open_wolf_TEL").click().run(timeout=120)
    assert not at.exception, at.exception
    assert at.session_state["thor_ticker"] == "TEL"
    head = next(m.value for m in at.markdown if "finns i dag i" in m.value)
    assert ">TEL</b>" in head and "TEL.OL" not in head and "MODERATE" in head
    text = _text(at)
    assert "Wolf: MODERATE" in text and "Wolf: ENTRY · BULL" not in text and "STORLEK" not in text
    assert "Logga köp" not in [b.label for b in at.button]


# ── T2: storleken för planering, Logga köp bara för färska KÖP-rader ─────────
def _momentum_page(monkeypatch, regime="GRÖN", today="2026-10-15"):
    data = {**tu.DATA, "regime": {"regime": regime, "generated": "2026-10-14T04:17"}}
    at = _page(monkeypatch, data=data, today=today)
    at.session_state["thor_ticker"] = "ERIC-B.ST"
    at.session_state["thor_strategy"] = "Momentum Swing"
    at.session_state["_thor_strategy_set"] = True
    at.run(timeout=120)
    assert not at.exception, at.exception
    return at, _text(at), next(b for b in at.button if b.label == "Logga köp")


def test_blocked_row_keeps_the_size_for_planning_but_not_the_buy(monkeypatch):
    _at, text, log = _momentum_page(monkeypatch, regime="RÖD")
    assert ">NEJ<" in text and "STORLEK" in text and "köp 80.00" in text            # storleken syns för planering
    assert "Ingen köpsignal (NEJ: rad 1 · setup A · RÖD regim) — storleken är bara för planering" in text
    assert log.disabled


def test_fresh_buy_row_can_be_logged(monkeypatch):
    _at, text, log = _momentum_page(monkeypatch)
    assert ">KÖP<" in text and "STORLEK" in text and "Ingen köpsignal" not in text and "GAMMAL KÖRNING" not in text
    assert not log.disabled


def test_stale_buy_row_is_marked_and_cannot_be_logged(monkeypatch):
    _at, text, log = _momentum_page(monkeypatch, today="2026-10-30")
    assert ">KÖP<" in text and "GAMMAL KÖRNING — körd 2026-10-14, äldre än 4 dagar" in text   # T6 i verdiktkortet
    assert "Gammal körning (körd 2026-10-14, äldre än 4 dagar) — storleken är bara för planering" in text
    assert log.disabled


# ── T3: London i pence, Europa och Kanada i rätt valuta ─────────────────────
def test_london_is_sized_in_pence_and_europe_canada_in_their_currency():
    import positions
    from thor import storlek as sk
    ccy, fx_ = sk.fx_for("GLEN.L")
    assert ccy == "GBp" and fx_ == pytest.approx(positions.FX_TO_SEK["GBP"] * 0.01)
    sz = sk.size(450, 420, 100000, 2, 6, fx=fx_, currency=ccy)
    assert sz["risk_shares"] == 498 and sz["cap_shares"] == 99 and "köp 450.00 GBp" in sk.row_text("GLEN.L", "Wolf", sz)
    assert sk.fx_for("UMI.BR")[0] == "EUR" and sk.fx_for("SEM.LS")[0] == "EUR" and sk.fx_for("XX.NE")[0] == "CAD"
    assert sk.fx_for("NEM") == ("USD", positions.FX_TO_SEK["USD"]) and sk.fx_for("BOL.ST") == ("SEK", 1.0)
    held = positions.valuation({"ticker": "GLEN.L", "entry_price": 450.0, "shares": 100})
    assert held["value_sek"] == round(450 * 0.01 * positions.FX_TO_SEK["GBP"] * 100, 0)    # samma enhet som förut
    assert positions.valuation({"ticker": "UMI.BR", "entry_price": 30.0, "shares": 10})["currency"] == "EUR"


# ── T4: arkens bara nordiska tickers ────────────────────────────────────────
def _universe():
    m = ur.Member
    return ur.Universe(members={
        "BOL.ST": m("BOL.ST", "Boliden", ur.NORDEN, bd_ticker="BOL"),
        "ERIC-B.ST": m("ERIC-B.ST", "Ericsson B", ur.NORDEN, bd_ticker="ERIC B"),
        "TEL.OL": m("TEL.OL", "Telenor", ur.NORDEN, bd_ticker="TEL"),
        "TEL": m("TEL", "TE Connectivity", ur.USA)})


def test_sheet_map_gives_nordic_forms_but_never_a_us_member():
    smap = _universe().sheet_map()
    assert smap["BOL"] == "BOL.ST" and smap["ERIC B"] == smap["ERIC-B"] == "ERIC-B.ST"
    assert "TEL" not in smap                                                        # TE Connectivity i USA
    assert ur.nordic_form("bol", smap) == "BOL.ST" and ur.nordic_form("eric b", smap) == "ERIC-B.ST"
    assert ur.nordic_form("NEM", smap) == "NEM" and ur.nordic_form("AEM.TO", smap) == "AEM.TO"
    assert ur.nordic_form("TEL", smap) == "TEL" and ur.nordic_form(" bol ", None) == "BOL"


SHEET_BOARD = {"generated": fx.GEN, "board": fx.BOARD["board"], "buys": {"themes": {"koppar": {
    "quality": {"picks": [{"ticker": "BOL", "name": "Boliden", "source": "Rick Rule-arket", "score": 4,
                           "verdict": "Köpkandidat", "nordic": False}]},
    "leverage": {"strategy": "Tiggre", "picks": []}}}}}
SHEET_SCREENS = {"generated": fx.GEN, "screens": {"rule": {"label": "Överlevarna (Rule)", "sheet": "Rick Rule",
                                                           "rows": [{"ticker": "BOL.ST", "name": "Boliden"}], "new": []}}}


def test_saved_board_with_a_bare_nordic_ticker_gives_one_nordic_row():
    blobs = {"board": SHEET_BOARD, "screens": SHEET_SCREENS}
    before = [(c.ticker, c.nordic) for c in tc.build(blobs, today="2026-10-15") if c.strategy == "rule"]
    assert sorted(before) == [("BOL", False), ("BOL.ST", True)]                     # felet: två rader, en som USA
    rows = tc.build(blobs, today="2026-10-15", sheet_map=_universe().sheet_map())
    (c,) = [r for r in rows if r.strategy == "rule"]
    assert c.ticker == "BOL.ST" and c.status_class == tc.KOP and c.nordic and ur.market_of(c.ticker) == ur.NORDEN
    assert [r.ticker for r in tui.shown_rows(rows, "Rick Rule", ["bara Norden"])] == ["BOL.ST"]
    assert [r.ticker for r in tui.shown_rows(rows, "Rick Rule", [], ["Norden"])] == ["BOL.ST"]


def test_board_source_maps_bare_sheet_tickers_before_the_region_rule():
    from valhall import picks as vp
    sheets = {"producers": {"producers": [
        {"ticker": "BOL", "name": "Boliden", "commodity": "Zink", "price": 3.0, "unit_cost": 1.5, "ev_ebitda": 5.0,
         "nd_ebitda": 0.5, "mine_life": 12.0, "rp_ratio": 15.0, "jurisdiktion": True, "kapitaldisciplin": True,
         "insyn": True}], "royalty": []}}
    theme = vp.theme_of_commodity("Zink")
    assert vp.theme_quality(sheets, None, [])[theme]["picks"] == []                   # bar 'BOL' räknades som USA
    picks = vp.theme_quality(sheets, None, [], to_yf=_universe().sheet_map())[theme]["picks"]
    assert [(p["ticker"], p["nordic"]) for p in picks] == [("BOL.ST", True)]
    built = vp.build([], None, sheets, analyze=lambda c: None, to_yf=_universe().sheet_map())
    assert built["themes"][theme]["quality"]["picks"][0]["ticker"] == "BOL.ST"
    src = open(os.path.join(tu.ROOT, "valhall_scan.py"), encoding="utf-8").read()
    assert "to_yf=sheet_map" in src and "reg_u.sheet_map()" in src                  # kartan byggs med registret


def test_load_carries_the_registers_sheet_map():
    import markets as mk
    nordic = [{"insId": 3, "ticker": "BOL", "name": "Boliden", "marketId": 1, "sectorId": 7, "stockPriceCurrency": "SEK"}]
    u = ur.build(table=mk.FALLBACK, nordic=nordic, global_=[], include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    out = tui.load(lambda name, fb: saved.get(name, fb))
    assert out["sheet_map"]["BOL"] == "BOL.ST"
    assert tui.load(lambda name, fb: fb)["sheet_map"] == {}


# ── T5: temafiltret för rader utan eget tema ────────────────────────────────
def test_theme_filter_matches_theme_less_rows_on_the_companys_theme():
    def row(t, th=None, strategy="hav"):
        return SimpleNamespace(ticker=t, theme=th, family="håv", strategy=strategy, status_class=tc.TRAFF,
                               nordic=False, segment=None, new=False)
    rows = [row("NXE"), row("DNN"), row("CCJ", "uran"), row("UUUU", strategy="wolf"), row("PALL"), row("SBSW"),
            row("NEM"), row("CCO.TO"), row("XYZ"), row("UEC", "guld")]
    pick = lambda seg: [c.ticker for c in tui.shown_rows(rows, tui.ALL, [], None, seg)]  # noqa: E731
    assert pick("tema:uran") == ["NXE", "DNN", "CCJ", "UUUU", "CCO.TO"]              # CCO.TO ur EMBER-listan
    assert pick("tema:palladium") == ["PALL", "SBSW"]                                # SBSW ur temalistan som förut
    assert "UEC" not in pick("tema:uran") and "UEC" in pick("tema:guld")             # radens eget tema vinner
    assert all(c.theme in (None, "uran", "guld") for c in rows)                      # raderna ändras inte


# ── T6: gammal körning i analysblocket ──────────────────────────────────────
def test_stale_verdict_keeps_its_word_but_is_grey_and_marked():
    fresh = {(c.strategy, c.ticker): c for c in tc.build(fx.BLOBS, today="2026-10-15")}
    stale = {(c.strategy, c.ticker): c for c in tc.build(fx.BLOBS, today="2026-10-28")}
    key = ("viking_nine", "HEXA-B.ST")
    v = cards.verdict_for(stale[key], "viking_nine")
    assert ">KÖP<" in v and "GAMMAL KÖRNING — körd 2026-10-14, äldre än 4 dagar" in v
    assert f"border:2px solid {cards.GREY}66" in v and cards.GREEN not in v               # grått, inte grönt
    ok = cards.verdict_for(fresh[key], "viking_nine")
    assert "GAMMAL" not in ok and "körd 2026-10-14" in ok and f"border:2px solid {cards.GREEN}66" in ok
    head = cards.header_card({"yf": "HEXA-B.ST"}, [stale[key]])
    assert "gammal · 2026-10-14" in head and "gammal" not in cards.header_card({"yf": "HEXA-B.ST"}, [fresh[key]])


# ── T7: Viking Nines USA-del behåller sitt datum efter ⚡ Skanna nu ───────────
def test_viking_nine_live_keeps_the_usa_parts_own_date(monkeypatch):
    import viking_nine_scan as vns
    from thor import live
    monkeypatch.setattr(vns, "universes", lambda usa=True, **kw: {"norden": {"BOL.ST": ("Boliden", "materials")}, "usa": {}})
    monkeypatch.setattr(vns, "run", lambda members=None, dry_run=False, **kw: {"payload": {
        "generated": "2026-10-28T16:00:00+00:00", "rows": [{"ticker": "BOL.ST", "category": "READY", "market": "norden"}],
        "markets": {"norden": {"funnel": {}, "rows": [{"ticker": "BOL.ST", "category": "READY", "market": "norden"}]},
                    "usa": {"funnel": {}, "rows": []}}}})
    nvda = {"ticker": "NVDA", "category": "GOLDEN TICKET", "market": "usa", "bar_date": "2026-10-20"}
    old = {"viking_nine": {"generated": "2026-10-20T16:00:00+00:00", "rows": [nvda],
                           "markets": {"usa": {"funnel": {"universe": 900}, "rows": [nvda]}}}}

    def rows_of(blob):
        return {c.ticker: c for c in tc.mark_stale(tc.from_viking_nine(None, fx.CTX, blob), today="2026-10-28")}
    assert rows_of(old["viking_nine"])["NVDA"].stale                                 # före: gammal
    once = live._viking_nine(old)
    r = rows_of(once)
    assert r["NVDA"].asof == "2026-10-20" and r["NVDA"].stale and r["BOL.ST"].asof == "2026-10-28"
    twice = live._viking_nine({"viking_nine": once})                                 # andra trycket: datumet står kvar
    assert rows_of(twice)["NVDA"].asof == "2026-10-20"
    plain = {"generated": "2026-10-14T16:00:00+00:00", "rows": [{"ticker": "X.ST"}]}  # rad utan market: skanningens datum
    assert rows_of(plain)["X.ST"].asof == "2026-10-14"


# ── T8: temats Valhall-läge ur temats egen segmentrad ───────────────────────
KAFFE_BOARD = {"generated": fx.GEN, "board": {"segments": [
    {"key": "consumer_disc", "nordic": {"status": "RÖD", "cykel": "TOPP", "phase": "SVAG"},
     "global": {"status": "RÖD", "cykel": "TOPP", "phase": "SVAG"}, "themes": []},
    {"key": "consumer_staples", "nordic": {"status": "GUL", "cykel": "SEN", "phase": "STARK"},
     "global": {"status": "GUL", "cykel": "SEN", "phase": "STARK"},
     "themes": [{"theme": "kaffe", "state": {"status": "GRÖN", "cykel": "TIDIG", "phase": "BAISSE · VÄNDER"}}]}]}}


def test_theme_state_is_read_from_the_themes_own_segment(monkeypatch):
    from test_thor_ravara import _prices, patch_engines
    from thor import dom
    patch_engines(monkeypatch)
    sbux = tt.Ticker(raw="SBUX", yf="SBUX", segment="consumer_disc", theme="kaffe")     # sektorn Sällanköp, temat kaffe
    tag = td.valhall_tag(KAFFE_BOARD["board"], sbux)
    assert (tag["t_status"], tag["t_cykel"]) == ("GRÖN", "TIDIG") and tag["g_status"] == "RÖD"   # Norden/globalt: egna
    seg = td.cycle(sbux, "rule", KAFFE_BOARD, phase_fn=lambda t: (None, ""))["segment"]
    assert seg["view"] == "temat" and seg["status"] == "GRÖN" and seg["cykel"] == "TIDIG"
    ctx = dom.cycle_context(sbux, {"board": KAFFE_BOARD}, {}, _prices, theme_board=lambda: [],
                            fiat=lambda ccy: None)
    assert ctx["valhall"]["view"] == "temat" and ctx["valhall"]["cykel"] == "TIDIG"
    checks, _facts = dom._cycle_checks("rule", ctx)
    assert next(c for c in checks if c.key == "valhall").status == dom.PASS          # inte 'cykeltopp' (NEJ)
    src = open(os.path.join(tu.ROOT, "thor", "regim.py"), encoding="utf-8").read()
    assert "valhall_tag(inner, ident)" in src                                        # Long Screener samma uppslag


def test_register_theme_fills_segment_and_complex():
    ident = tt.Ticker(raw="XYZ", yf="XYZ")
    assert tui.fill_from_universe(ident, ur.Member("XYZ", "Xyz Coffee", ur.USA, theme="kaffe"))
    assert ident.theme == "kaffe" and ident.segment == "consumer_staples" and ident.segment_label == "Dagligvaror"
    assert ident.complex == "agri" and ident.sector_etf is None                     # sektor-ETF:n (regelinput) orörd
    seg = td.cycle(ident, "rule", KAFFE_BOARD, phase_fn=lambda t: (None, ""))["segment"]
    assert seg["view"] == "temat" and seg["status"] == "GRÖN"
    own = tt.Ticker(raw="Y", yf="Y", segment="consumer_disc")
    tui.fill_from_universe(own, ur.Member("Y", "Y", ur.USA, theme="kaffe"))
    assert own.segment == "consumer_disc"                                            # eget segment står kvar


# ── T10 och T12: Asymmetrin ────────────────────────────────────────────────
def test_asymmetry_does_not_cache_a_not_found_answer():
    from thor import asymmetri as ta

    class _St:
        session_state: dict = {}
    calls = []

    def fetch(ticker, api=None):
        calls.append(ticker)
        return {"ticker": ticker, "source": "—", "prices": [], "price": None}
    st = _St()
    ta.cached(st, "XYZ.ST", fetch_fn=fetch)
    ta.cached(st, "XYZ.ST", fetch_fn=fetch)
    assert calls == ["XYZ.ST", "XYZ.ST"] and "XYZ.ST" not in st.session_state[ta.CACHE_KEY]


def test_asymmetry_mode_is_chosen_per_ticker(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from asymmetry import quick_config as qc
    from asymmetry import quick_data
    import test_thor_asymmetri as fa
    monkeypatch.setattr(quick_data, "fetch", lambda t, api=None, **kw: fa.copper_co() if t == "CU.X" else fa.value_co())
    monkeypatch.setattr(tui, "_api", lambda: None)
    at = AppTest.from_function(fa._script)
    at.session_state["_ticker"] = "CU.X"
    at.run(timeout=120)
    at.radio(key="thor_asym_mode_CU.X").set_value(qc.MODE_COMMODITY).run(timeout=120)
    at.session_state["_ticker"] = "VAL.ST"
    at.run(timeout=120)
    assert not at.exception, at.exception
    assert at.radio(key="thor_asym_mode_VAL.ST").value == qc.MODE_AUTO              # Råvara följer inte med
    assert "VÄRDEDRIVARE" in _text(at)


# ── T14 och T15: registrets suffix och hemmamarknadens etikett ─────────────
def test_base_and_forms_know_every_register_suffix():
    assert tt.base("TTE.PA") == "TTE" and tt.forms("RHM.DE") == ["RHM.DE", "RHM"] and tt.base("SHOP.NE") == "SHOP"
    assert tt.base("X.CN") == "X" and tt.base("UMI.BR") == "UMI" and tt.base("ERIC-B.ST") == "ERIC-B"
    assert all(s in tt.SUFFIXES for m in ur.MARKETS for s in m.suffixes if s)
    rows = [tc.Candidate("TTE.PA", "TotalEnergies", "rule", "TRÄFF", tc.TRAFF)]
    assert [c.ticker for c in td.candidates_for(rows, "TTE")] == ["TTE.PA"]


def test_home_market_label_comes_from_the_register_but_the_rule_input_does_not():
    for t, label in (("AZN.L", "^FTSE"), ("RHM.DE", "^STOXX50E"), ("NESN.SW", "^STOXX50E"), ("SHOP.TO", "^GSPTSE"),
                     ("BOL.ST", "^OMX"), ("EQNR.OL", tt.home_index("EQNR.OL")), ("NEM", "SPY")):
        assert tt.home_label(t) == label, t
    azn = tt.identify("AZN.L")
    assert azn.home_label == "^FTSE" and azn.home_index == "SPY" and azn.market == "SPY"   # Alpha Regimes riktmärke orört
    assert "^FTSE · SPY" in cards.header_card(azn.as_dict(), [])
    seen = []
    cyc = td.cycle(azn, "quality", None, lambda t: (seen.append(t), ({"phase": "HOPE", "confidence": 50.0}, None))[1])
    assert cyc["benchmark"]["index"] == "^FTSE" and seen == ["AZN.L", "^FTSE"]


# ── T16: momentumrad utan ticker ───────────────────────────────────────────
def test_momentum_row_without_a_ticker_is_skipped():
    blob = {"generated": fx.GEN, "top": [{"rank": 3}, {"rank": 4, "ticker": "  "}, {"rank": 5, "yf": "NHY.OL"},
                                         {"rank": 1, "ticker": "ERIC B", "setupA": True, "price": 80.0}]}
    rows = tc.from_momentum(blob, fx.CTX, {"ERIC B": "ERIC-B.ST"}, regime={"regime": "GRÖN"})
    assert sorted(c.ticker for c in rows) == ["ERIC-B.ST", "NHY.OL"] and ".ST" not in [c.ticker for c in rows]


def test_a_suffixed_ticker_only_borrows_rows_from_its_own_market():
    rows = [tc.Candidate("OR.TO", "Osisko", "royalty", "KÖP", tc.KOP), tc.Candidate("EQNR.OL", "Equinor", "wolf", "x", tc.HALL)]
    assert td.candidates_for(rows, "OR.PA") == []                       # L'Oréal får inte Osiskos rad
    assert [c.ticker for c in td.candidates_for(rows, "EQNR.ST")] == ["EQNR.OL"]       # samma marknad (Norden)
