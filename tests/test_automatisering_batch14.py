"""
Automatisering, batch 14 — 'Hämta nu' för nya rader (Rick Rule, Tiggre och Durrett).

Rader som lagts in i dag har ingen rad i sifferuppdateringen förrän nattjobbet körts efter 💾. 'Hämta nu' hämtar
bolagets rad på ditt klick (sheets_refresh.fetch_row, Yahoo som reserv) och lägger den i sessionen under radens
nyckel (refresh_ui.LIVE_KEY) — kortets förslag läser den före nattens blob (refresh_ui.current_blob).

1. refresh_ui: överlagringen (put_live/with_live/current_blob), raden under knappen med bolaget som matchades
   (fetched_line) och fetch_live/fetch_button. Bara i sessionen — aldrig till Gisten eller repot.
2. Rick Rule: 'Hämta nu' överst på kortet — EV/EBITDA, nettoskuld/EBITDA, råvarupris (dagens termin för kortets
   råvara), EV, FCF 12 mån, DS-runway … ur den hämtade raden. 'Hämta EV nu' skriver EV som förut och ger kortet
   resten av raden.
3. Tiggre: 'Hämta nu' under Börsvärde — börsvärde, kassa och burn (DS-runway), aktier.
4. Durrett: 'Hämta nu' (Wolf Asymmetrys flöde, asymmetry/fetch.fetch_to_session) — samma nyckel confidence:<TICKER>.

Varje klick drar Börsdata-kvot (står i knappens hjälp). Inget skrivs när en sida öppnas och inget sparas utan ditt
eget Använd + 💾. Inga regler, grindar, trösklar eller poängformler ändrade.
"""
import copy
import html
import os
import sys
from datetime import date

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refresh_ui as rui        # noqa: E402
import sheets_refresh as sr     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = "2026-11-09"

# fetch_row:s form (Durrett-/Wolf Asymmetry-arkets rad) för Newmont
NEM = {"ticker": "NEM", "ins_id": 501, "price": 52.3, "asof": TODAY, "currency": "USD", "source": "borsdata",
       "ev_ebitda": 6.2, "nd_ebitda": 0.4, "mcap_musd": 60000.0, "bd_name": "Newmont Corporation", "bd_ticker": "NEM",
       "report_currency": "USD", "cash_musd": 3000.0, "debt_musd": 8000.0, "burn_musd": 0.0, "shares_now_m": 1100.0,
       "shares_year": 2025, "roic_pct": 9.0, "fcf_yield_pct": 7.0, "ev_ebit": 9.5, "fx_to_usd": 1.0,
       "fx_table": "sheets_refresh.FX_TO_USD (fast tabell)", "ev_musd": 65000.0, "net_debt_musd": 5000.0,
       "revenue_musd": 19000.0, "fcf_musd": 4200.0, "ocf_musd": 8000.0, "pe": 14.0, "ps": 3.1, "rs_rank": 70.0,
       "ebitda_margin": 0.45}
GOLD_SPOT = {"price": 4012.3, "unit": "USD/oz", "asof": "2026-11-07", "ticker": "GC=F"}


def _md(at) -> str:
    return html.unescape(" ".join(m.value for m in at.markdown))


def _captions(at) -> str:
    return " ".join(c.value for c in at.caption)


def _buttons(at) -> set:
    return {b.key for b in at.button}


@pytest.fixture
def session(monkeypatch):
    """En egen session (en dict) för de rena testerna — överlagringen följer inte med till nästa test. Ingen natt
    (load_refresh tom) om testet inte sätter en egen; Gisten läses aldrig."""
    state: dict = {}
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(rui, "load_refresh", lambda: {})
    return state


# ═════════════════════════════════════════════════════════════════════════════
#  Rena funktioner
# ═════════════════════════════════════════════════════════════════════════════
def test_as_sheet_row_gives_each_sheet_the_nightly_shape():
    p = sr.as_sheet_row(NEM, "producers")
    assert p["fcf12_musd"] == 4200.0 and "fcf_musd" not in p                 # FCF 12 mån bara som referens
    assert p["ev_musd"] == 65000.0 and p["net_debt_musd"] == 5000.0 and p["ev_ebitda"] == 6.2
    assert not {"roic_pct", "fx_to_usd", "fx_table", "pe", "ps", "rs_rank", "ebitda_margin", "ev_ebit"} & set(p)
    t = sr.as_sheet_row(NEM, "tiggre")
    assert not {"ev_musd", "fcf_musd", "net_debt_musd", "revenue_musd", "ocf_musd", "fcf12_musd", "roic_pct"} & set(t)
    assert (t["mcap_musd"], t["cash_musd"], t["burn_musd"], t["shares_now_m"]) == (60000.0, 3000.0, 0.0, 1100.0)
    c = sr.as_sheet_row(NEM, "confidence")
    assert c == NEM and c is not NEM
    assert "fcf_musd" in NEM and "roic_pct" in NEM                            # indata orörd
    assert sr.as_sheet_row(None, "producers") is None


def test_as_sheet_row_matches_the_nightly_producer_row():
    """Samma nycklar som nattjobbet lägger på en Rick Rule-rad (utom råvarupris, direktavkastning och median, som
    natten hämtar på annat sätt) — så att kortets förslag läser 'Hämta nu' exakt som natten."""
    from test_sheets_refresh import _API, _sheets
    night = sr.refresh(_API(), _sheets())["rows"]["producers:r1"]
    fetched = sr.as_sheet_row(sr.fetch_row(_API(), night["ticker"], night["ins_id"]), "producers")
    extra_night = {"commodity_price", "commodity_unit", "commodity_asof", "commodity_ticker", "dividend_yield_pct",
                   "ev_ebitda_median"}
    assert set(night) - extra_night == set(fetched)
    for k in ("ev_musd", "fcf12_musd", "net_debt_musd", "mcap_musd", "price", "ev_ebitda", "bd_name"):
        assert fetched[k] == night[k], k


def test_overlay_goes_before_the_night_and_keeps_the_nights_commodity_price(session):
    night = {"generated": "x", "rows": {"producers:p1": {"ev_ebitda": 9.0, "asof": "2026-11-08", "commodity_price": 4000.0,
                                                         "commodity_unit": "USD/oz", "commodity_asof": "2026-11-08",
                                                         "commodity_ticker": "GC=F"},
                                        "producers:p2": {"ev_ebitda": 3.0}}}
    assert rui.with_live(night, {}) is night and rui.with_live(None, {}) is None
    rui.put_live("producers:p1", {"ev_ebitda": 6.2, "asof": TODAY})
    assert rui.is_live("producers:p1") and not rui.is_live("producers:p2")
    merged = rui.with_live(night, rui.live_rows())
    assert merged["rows"]["producers:p1"]["ev_ebitda"] == 6.2 and merged["rows"]["producers:p2"]["ev_ebitda"] == 3.0
    assert merged["rows"]["producers:p1"]["commodity_price"] == 4000.0                 # nattens termin, eget datum
    assert merged["rows"]["producers:p1"]["commodity_asof"] == "2026-11-08"
    assert merged["generated"] == "x" and night["rows"]["producers:p1"]["ev_ebitda"] == 9.0   # natten orörd
    rui.put_live("producers:p1", {"ev_ebitda": 6.2, "commodity_price": 4012.3, "commodity_ticker": "GC=F"})
    assert rui.with_live(night, rui.live_rows())["rows"]["producers:p1"]["commodity_price"] == 4012.3
    rui.put_live("tiggre:c9", {"mcap_musd": 50.0})                                    # ny rad utan natt
    assert rui.with_live({}, rui.live_rows())["rows"]["tiggre:c9"] == {"mcap_musd": 50.0}


def test_a_newer_night_wins_over_an_old_fetch(session, monkeypatch):
    """En session som stått öppen: nattens körning efter hämtningen går före den — och märks inte '(hämtat nu)'."""
    night = {"rows": {"producers:p1": {"ev_ebitda": 5.5, "asof": "2026-11-10"}}}
    monkeypatch.setattr(rui, "load_refresh", lambda: night)
    rui.put_live("producers:p1", {"ev_ebitda": 6.2, "asof": TODAY})
    assert rui.current_blob()["rows"]["producers:p1"]["ev_ebitda"] == 5.5 and not rui.is_live("producers:p1")
    night["rows"]["producers:p1"]["asof"] = TODAY                                     # samma dag: hämtningen gäller
    assert rui.current_blob()["rows"]["producers:p1"]["ev_ebitda"] == 6.2 and rui.is_live("producers:p1")
    rui.put_live("producers:p1", {"ev_ebitda": 6.4})                                  # ingen kurs, inget datum
    assert rui.current_blob()["rows"]["producers:p1"]["ev_ebitda"] == 6.4 and rui.is_live("producers:p1")


def test_current_blob_feeds_suggestion_from_the_session_first(session, monkeypatch):
    monkeypatch.setattr(rui, "load_refresh", lambda: {"rows": {"producers:p1": {"ev_ebitda": 9.0, "asof": "2026-11-08"}}})
    row = {"id": "p1", "name": "Newmont"}
    assert rui.suggestion(rui.current_blob(), "producers", row, "ev_ebitda")[0] == 9.0
    rui.put_live("producers:p1", sr.as_sheet_row(NEM, "producers"))
    assert rui.suggestion(rui.current_blob(), "producers", row, "ev_ebitda") == (6.2, TODAY, "USD")
    assert rui.load_refresh()["rows"]["producers:p1"]["ev_ebitda"] == 9.0               # nattens blob orörd
    assert set(session) == {rui.LIVE_KEY}                                              # bara sessionens överlagring


def test_fetched_line_always_names_the_matched_company():
    s = sr.as_sheet_row(NEM, "producers")
    line = rui.fetched_line(s, {"name": "Newmont", "ins_id": 501})
    assert line == f"🤖 Hämtat nu ur Börsdata {TODAY}: Newmont Corporation (NEM), kurs 52.3 USD."
    assert "saknar ins_id" in rui.fetched_line(s, {"name": "Newmont"})
    bad = rui.fetched_line(s, {"name": "Genovis"})
    assert "Newmont Corporation (NEM)" in bad and "Liknar inte 'Genovis' — kontrollera tickern." in bad
    gbx = rui.fetched_line(dict(s, currency="GBX", mcap_musd=None), {"name": "Newmont", "ins_id": 1})
    assert "Kursen står i GBX, som saknas i växelkurstabellen — inga omräknade MUSD-tal." in gbx
    nameless = rui.fetched_line({"price": 7.1, "asof": TODAY, "currency": "USD"}, {"ins_id": 1})
    assert "svaret har inget bolagsnamn (kurs 7.1 USD) — kontrollera att det är rätt bolag." in nameless
    assert rui.fetched_line(None, {}) == ""


def test_yahoo_rows_show_the_symbol_and_name():
    from asymmetry import fetch as af
    info = {"marketCap": 1.5e8, "currency": "CAD", "currentPrice": 0.85, "sharesOutstanding": 176e6,
            "totalCash": 2e7, "longName": "Abc Minerals Ltd"}
    y = af.yahoo_row("ABC", lambda sym: info if sym == "ABC.V" else {})
    assert y["yahoo"] == "ABC.V" and y["yahoo_name"] == "Abc Minerals Ltd"
    assert rui.source_name(y) == "Yahoo (ABC.V)" and rui.source_name(NEM) == "Börsdata"
    assert rui.source_name({"source": "yfinance", "price": 3.0}) == "Börsdata"          # nattens Yahoo-kurs som förut
    line = rui.fetched_line(sr.as_sheet_row(y, "tiggre"), {"name": "Abc Minerals"})
    assert line.startswith(f"🤖 Hämtat nu ur Yahoo (ABC.V) {date.today().isoformat()}: Abc Minerals Ltd, kurs 0.85 CAD.")
    assert "Yahoo provar tickern med börssuffix och kan träffa fel bolag." in line and "ins_id" not in line
    blob = {"rows": {"tiggre:c1": y}}
    assert rui.matched_note(blob, "tiggre", {"id": "c1", "name": "Zed Gold"}) == " (matchat: Abc Minerals Ltd)"
    assert rui.matched_note(blob, "tiggre", {"id": "c1", "name": "Abc Minerals"}) == ""


class _Api:
    pass


def test_fetch_live_stores_the_sheets_row_with_todays_spot(session, monkeypatch):
    import commodity_prices
    calls = []
    monkeypatch.setattr(sr, "fetch_row", lambda api, t, ins_id=None: calls.append((t, ins_id)) or dict(NEM))
    monkeypatch.setattr(commodity_prices, "spot", lambda name: GOLD_SPOT if name == "Guld" else None)
    row = {"id": "p1", "ticker": "nem", "name": "Newmont", "commodity": "Guld"}
    got, why = rui.fetch_live("producers", row, commodity="Guld", api=_Api())
    assert why == "" and calls == [("NEM", None)]
    assert got["fcf12_musd"] == 4200.0 and "roic_pct" not in got
    assert (got["commodity_price"], got["commodity_unit"], got["commodity_ticker"]) == (4012.3, "USD/oz", "GC=F")
    assert rui.live_rows()["producers:p1"] == got
    assert row == {"id": "p1", "ticker": "nem", "name": "Newmont", "commodity": "Guld"}     # raden orörd


def test_fetch_live_falls_back_to_yahoo_and_says_why(session, monkeypatch):
    from asymmetry import fetch as af
    real = af.yahoo_row
    info = {"marketCap": 1.5e8, "currency": "CAD", "currentPrice": 0.85, "longName": "Abc Minerals Ltd"}
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: real(t, lambda sym: info if sym == "ABC.V" else {}))
    monkeypatch.setattr(sr, "fetch_row", lambda api, t, ins_id=None: None)
    got, why = rui.fetch_live("tiggre", {"id": "c1", "ticker": "ABC"}, api=_Api())
    assert why == "Börsdata känner inte ABC." and got["yahoo"] == "ABC.V"
    assert got["mcap_musd"] == round(1.5e8 * sr.FX_TO_USD["CAD"] / 1e6, 1) and "fx_to_usd" not in got
    assert rui.is_live("tiggre:c1")
    # utan Yahoo (Hämta EV nu): inget hämtat, inget lagrat
    session.clear()
    got, why = rui.fetch_live("producers", {"id": "p1", "ticker": "ABC"}, api=_Api(), yahoo=False)
    assert got is None and why == "Börsdata känner inte ABC." and rui.live_rows() == {}
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: None)
    got, why = rui.fetch_live("tiggre", {"id": "c1", "ticker": "ABC"}, api=_Api())
    assert got is None and "Yahoo har inte heller ABC" in why and rui.live_rows() == {}

    def boom(api, t, ins_id=None):
        raise RuntimeError("429 kvoten slut")
    monkeypatch.setattr(sr, "fetch_row", boom)
    got, why = rui.fetch_live("producers", {"id": "p1", "ticker": "ABC"}, api=_Api(), yahoo=False)
    assert got is None and why == "Börsdata svarade inte: 429 kvoten slut"
    assert rui.fetch_live("producers", {"id": "p1"}, api=_Api()) == (None, "Raden saknar ticker.")


def test_fetch_to_session_puts_the_confidence_row_in_the_shared_overlay(session, monkeypatch):
    """Durrett och Wolf Asymmetry: samma nyckel confidence:<TICKER> — en hämtning syns på båda arken."""
    from asymmetry import fetch as af
    from confidence.data.models import CompanyInput
    monkeypatch.setattr(sr, "fetch_row", lambda api, t, ins_id=None: dict(NEM) if t == "NEM" else None)
    ok, msg = af.fetch_to_session(CompanyInput(ticker="NEM", name="Newmont"), api=_Api())
    assert ok and "tal ur Börsdata" in msg
    assert rui.live_rows()["confidence:NEM"]["roic_pct"] == 9.0 and rui.is_live("confidence:NEM")
    session.clear()
    ok, msg = af.fetch_to_session(CompanyInput(ticker="ZZZ"), api=_Api(), info_getter=lambda sym: {})
    assert not ok and "Yahoo har inte heller ZZZ" in msg and rui.live_rows() == {}


def test_buttons_say_they_draw_borsdata_quota():
    assert "drar Börsdata-kvot" in rui.FETCH_HELP and "💾" in rui.FETCH_HELP and "ins_id" in rui.FETCH_HELP


# ═════════════════════════════════════════════════════════════════════════════
#  Arksidorna (AppTest)
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def page(monkeypatch):
    """Arksidan med fejkad sifferuppdatering, Börsdata och lagring — inget nätverk. Sparningar till GitHub spelas in."""
    from streamlit.testing.v1 import AppTest
    import borsdata_api
    import gist_storage
    import judgment_hints as jh
    import lukacs_ui
    import screens_ui
    import storage

    saves: list = []
    fetched: list = []

    def make(name: str, blob: dict, stores: dict, rows: dict):
        monkeypatch.setattr(lukacs_ui, "_live_fx", lambda ccy: None)
        monkeypatch.setattr(gist_storage, "load_blob",
                            lambda n, default=None: blob if n == "sheets_refresh.json" else default)
        monkeypatch.setattr(rui, "load_refresh", lambda: blob)
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        monkeypatch.setattr(jh, "gather_for", lambda t: {})
        monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(stores.get(n, default)))
        monkeypatch.setattr(storage, "legacy_gist", lambda f: None)
        monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda *a, **k: _Api())

        def fetch_row(api, t, ins_id=None):
            fetched.append(t)
            r = rows.get(t)
            return dict(r) if r else None
        monkeypatch.setattr(sr, "fetch_row", fetch_row)

        def _save(n):
            import streamlit as st
            saves.append((n, copy.deepcopy(st.session_state[n])))
            return storage.SaveResult(n, "abc1234def", "https://x", "2026-11-08T10:00:00Z")
        monkeypatch.setattr(storage, "save_session", _save)

        def app(name, root):
            import sys as _s
            _s.path.insert(0, root)
            if name == "Rick Rule":
                import producers
                producers.render_producers_page(name)
            elif name == "Durrett":
                from engines.durrett.ui import render_durrett_page
                render_durrett_page()
            else:
                import tiggre
                tiggre.render_tiggre_page()
        return AppTest.from_function(app, args=(name, ROOT), default_timeout=90)

    make.saves = saves
    make.fetched = fetched
    return make


# ── Rick Rule ────────────────────────────────────────────────────────────────
RULE = {"id": "p1", "ticker": "NEM", "name": "Newmont", "commodity": "Guld", "date": TODAY, "position_pct": 3.0}


def _rule_store(**over):
    return {"producers": {"producers": [dict(RULE, **over)], "royalty": []}}


def _rule_row(at) -> dict:
    return at.session_state["producers"]["producers"][0]


def test_rule_new_row_gets_every_suggestion_from_fetch_now(page, monkeypatch):
    import commodity_prices
    monkeypatch.setattr(commodity_prices, "spot", lambda name: GOLD_SPOT if name == "Guld" else None)
    at = page("Rick Rule", {}, _rule_store(), {"NEM": NEM})
    at.run()
    assert not at.exception, at.exception
    assert "pr_live_p1" in _buttons(at) and "rf_producers_p1_ev_ebitda" not in _buttons(at)   # ingen natt än
    assert rui.LIVE_KEY not in at.session_state and not page.fetched                  # att öppna hämtar inget
    at.button(key="pr_live_p1").click().run()
    assert not at.exception, at.exception
    assert page.fetched == ["NEM"]
    md, cap = _md(at), _captions(at)
    assert (f"🤖 Hämtat nu ur Börsdata {TODAY}: Newmont Corporation (NEM), kurs 52.3 USD. Raden saknar ins_id — "
            f"matchat på tickern.") in cap
    assert f"🤖 Börsdata {TODAY} (hämtat nu): EV/EBITDA 6.2" in md
    assert f"🤖 Börsdata {TODAY} (hämtat nu): nettoskuld/EBITDA 0.40" in md
    assert f"🤖 Börsdata {TODAY} (hämtat nu): råvarupris 4,012.30 USD/oz" in md         # dagens termin för Guld
    assert f"🤖 Börsdata {TODAY} (hämtat nu): EV (MUSD) 65,000" in md
    assert f"🤖 Börsdata FCF 12 mån: 4 200 MUSD ({TODAY})" in md                        # referensraden vid FV
    assert f"Börsdata {TODAY}: positivt FCF 12 mån, burn 0 MUSD/år" in md                # DS-runwayn
    row = _rule_row(at)
    assert not {"ev_ebitda", "nd_ebitda", "price", "ev_musd", "ins_id"} & set(row)      # hämta skriver inget i raden
    assert not page.saves
    at.button(key="rf_producers_p1_ev_ebitda").click().run()
    assert not at.exception, at.exception
    assert _rule_row(at)["ev_ebitda"] == 6.2 and at.number_input(key="pr_ev_p1").value == 6.2
    assert "nd_ebitda" not in _rule_row(at) and not page.saves                         # bara det fältet, 💾 sparar


def test_rule_fetch_ev_now_still_writes_ev_and_feeds_the_card(page):
    at = page("Rick Rule", {}, _rule_store(), {"NEM": NEM})
    at.run()
    at.button(key="pr_evget_p1").click().run()
    assert not at.exception, at.exception
    assert _rule_row(at)["ev_musd"] == 65000.0 and at.number_input(key="pr_evm_p1").value == 65000.0
    assert f"EV 65,000 MUSD ur Börsdata ({TODAY})." in _captions(at)
    assert f"🤖 Börsdata {TODAY} (hämtat nu): EV/EBITDA 6.2" in _md(at)                 # resten av raden som förslag
    assert "ev_ebitda" not in _rule_row(at) and not page.saves


def test_rule_night_row_is_kept_until_you_fetch(page):
    blob = {"rows": {"producers:p1": {"ticker": "NEM", "asof": "2026-11-08", "ev_ebitda": 9.0, "bd_name": "Newmont"}}}
    at = page("Rick Rule", blob, _rule_store(), {"NEM": NEM})
    at.run()
    assert "🤖 Börsdata 2026-11-08: EV/EBITDA 9.0" in _md(at) and "hämtat nu" not in _md(at)
    at.button(key="pr_live_p1").click().run()
    assert not at.exception, at.exception
    assert f"🤖 Börsdata {TODAY} (hämtat nu): EV/EBITDA 6.2" in _md(at) and "EV/EBITDA 9.0" not in _md(at)


# ── Tiggre ───────────────────────────────────────────────────────────────────
def _tiggre_store(**cand):
    c = {"id": "c1", "ticker": "TGR", "name": "Tiger Gold", "screen": {}, "downside": -40.0, "factors": {},
         "catalysts": [], **cand}
    return {"tiggre": {"candidates": [c], "positions": [], "closed": [], "parked": []},
            "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0}}


TGR = {"ticker": "TGR", "ins_id": 9001, "price": 0.62, "asof": TODAY, "currency": "CAD", "source": "borsdata",
       "mcap_musd": 85.4, "bd_name": "Tiger Gold Corp", "bd_ticker": "TGR", "report_currency": "CAD",
       "cash_musd": 12.0, "burn_musd": 16.0, "shares_now_m": 190.0, "shares_year": 2025, "ev_musd": 70.0}


def test_tiggre_fetch_now_next_to_market_cap(page):
    stores = _tiggre_store(ins_id=9001)
    at = page("Tiggre", {}, stores, {"TGR": TGR})
    at.run()
    assert not at.exception, at.exception
    assert "tg_live_c1" in _buttons(at) and "rf_tiggre_c1_mcap" not in _buttons(at)
    assert at.session_state["tiggre"] == stores["tiggre"] and not page.fetched
    at.button(key="tg_live_c1").click().run()
    assert not at.exception, at.exception
    assert f"🤖 Hämtat nu ur Börsdata {TODAY}: Tiger Gold Corp (TGR), kurs 0.62 CAD." in _captions(at)
    md = _md(at)
    assert f"🤖 Börsdata {TODAY} (hämtat nu): börsvärde MUSD 85" in md
    assert (f"🤖 DS: Runway < 12 mån → 2 riskpoäng — Börsdata {TODAY}: kassa 12,0 MUSD ÷ burn 16,0 MUSD/år "
            f"(negativt FCF 12 mån) = runway 9 mån") in md
    assert at.session_state["tiggre"] == stores["tiggre"]                              # hämta skriver inget i arket
    at.button(key="rf_tiggre_c1_mcap").click().run()
    assert not at.exception, at.exception
    assert at.session_state["tiggre"]["candidates"][0]["mcap"] == 85.4
    assert at.number_input(key="tg_mcap_c1").value == 85.4 and not page.saves


def test_tiggre_yahoo_fallback_shows_the_symbol_it_used(page, monkeypatch):
    from asymmetry import fetch as af
    real = af.yahoo_row
    info = {"marketCap": 1.5e8, "currency": "CAD", "currentPrice": 0.85, "sharesOutstanding": 176e6,
            "longName": "Tiger Gold Corp"}
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: real(t, lambda sym: info if sym == "TGR.V" else {}))
    at = page("Tiggre", {}, _tiggre_store(), {})
    at.run()
    at.button(key="tg_live_c1").click().run()
    assert not at.exception, at.exception
    today = date.today().isoformat()
    cap = _captions(at)
    assert f"🤖 Hämtat nu ur Yahoo (TGR.V) {today}: Tiger Gold Corp, kurs 0.85 CAD." in cap
    assert "Yahoo provar tickern med börssuffix och kan träffa fel bolag." in cap
    assert f"🤖 Yahoo (TGR.V) {today} (hämtat nu): börsvärde MUSD 110" in _md(at)


def test_tiggre_unknown_currency_gives_no_converted_numbers(page):
    gbx = dict(TGR, currency="GBX", report_currency="GBX", mcap_musd=None, cash_musd=None, burn_musd=None,
               ev_musd=None)
    at = page("Tiggre", {}, _tiggre_store(), {"TGR": gbx})
    at.run()
    at.button(key="tg_live_c1").click().run()
    assert not at.exception, at.exception
    assert "Kursen står i GBX, som saknas i växelkurstabellen — inga omräknade MUSD-tal." in _captions(at)
    assert "rf_tiggre_c1_mcap" not in _buttons(at) and "börsvärde MUSD" not in _md(at)


def test_tiggre_nothing_found_says_so_and_stores_nothing(page, monkeypatch):
    from asymmetry import fetch as af
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: None)
    at = page("Tiggre", {}, _tiggre_store(), {})
    at.run()
    at.button(key="tg_live_c1").click().run()
    assert not at.exception, at.exception
    assert ("Börsdata känner inte TGR. Yahoo har inte heller TGR — vänta på nattens körning eller skriv talen själv."
            in _captions(at))
    assert rui.LIVE_KEY not in at.session_state


# ── Durrett ──────────────────────────────────────────────────────────────────
def _durrett_store():
    from confidence import store as cs
    from confidence.data.models import CompanyInput
    data = cs.default()
    cs.put(data, CompanyInput(ticker="NEM", name="Newmont", commodity="gold", stage="producer",
                              maturity="production"))
    return {"confidence": data, "producers": {}, "tiggre": {}}


def test_durrett_fetch_now_feeds_the_proposals(page):
    stores = _durrett_store()
    at = page("Durrett", {}, stores, {"NEM": NEM})
    at.run()
    assert not at.exception, at.exception
    assert "durrett_bd_go_NEM" in _buttons(at) and "durrett_rf_NEM_market_cap_musd" not in _buttons(at)
    assert "tryck Hämta nu" in _captions(at) and not page.fetched
    at.button(key="durrett_bd_go_NEM").click().run()
    assert not at.exception, at.exception
    assert page.fetched == ["NEM"]
    assert any(e.label.startswith(f"🤖 Börsdata {TODAY}:") and e.label.endswith("förslag (hämtade nu)")
               for e in at.expander)
    cap = _captions(at)
    assert f"🤖 Hämtat nu ur Börsdata {TODAY}: Newmont Corporation (NEM), kurs 52.3 USD." in cap
    assert "tickern matchades mot Börsdata-id 501" in cap                              # spara id:t med Använd
    assert at.session_state["confidence"]["companies"]["NEM"]["fields"] == {}           # hämta skriver inget
    assert "confidence:NEM" in at.session_state[rui.LIVE_KEY]                          # Wolf Asymmetrys nyckel
    at.button(key="durrett_rf_NEM_market_cap_musd").click().run()
    assert not at.exception, at.exception
    mc = at.session_state["confidence"]["companies"]["NEM"]["fields"]["market_cap_musd"]
    assert mc["value"] == 60000.0 and mc["pub_date"] == TODAY and not page.saves


# ═════════════════════════════════════════════════════════════════════════════
#  Rättelser: ett gammalt 'hittade inget' och Börsdata som inte svarar
# ═════════════════════════════════════════════════════════════════════════════
def _raise_429(api, t, ins_id=None):
    raise RuntimeError("429 kvoten slut")


def test_borsdata_error_gives_no_yahoo_and_keeps_the_rows(session, monkeypatch):
    """Börsdata svarar inte (429): ingen Yahoo på bara tickern — den kan vara ett annat bolag. Inget lagras, en
    tidigare hämtning står kvar. 'Känner inte tickern' ger Yahoo som förut."""
    from asymmetry import fetch as af
    asked = []
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: asked.append(t) or {"ticker": t, "source": "yfinance",
                                                                                    "yahoo": t, "mcap_musd": 9999.0})
    monkeypatch.setattr(sr, "fetch_row", _raise_429)
    rui.put_live("tiggre:c1", {"mcap_musd": 85.4})
    for row in ({"id": "c1", "ticker": "TGR", "ins_id": 9001}, {"id": "c1", "ticker": "TGR"}):
        got, why = rui.fetch_live("tiggre", row, api=_Api())
        assert (got, why) == (None, "Börsdata svarade inte: 429 kvoten slut")
        assert not asked and rui.live_rows() == {"tiggre:c1": {"mcap_musd": 85.4}}
    monkeypatch.setattr(sr, "fetch_row", lambda api, t, ins_id=None: None)
    got, why = rui.fetch_live("tiggre", {"id": "c1", "ticker": "TGR"}, api=_Api())
    assert asked == ["TGR"] and why == "Börsdata känner inte TGR." and rui.live_rows()["tiggre:c1"]["mcap_musd"] == 9999.0


def test_a_stored_row_clears_the_old_failure_for_that_row(session, monkeypatch):
    """Meddelandet ligger under radens nyckel: en rad som kommer in på annat sätt tar bort det, andra raders står kvar."""
    import borsdata_api
    from asymmetry import fetch as af
    monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda *a, **k: _Api())
    monkeypatch.setattr(sr, "fetch_row", lambda api, t, ins_id=None: None)
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: None)
    rui._fetch_click("producers", {"id": "p1", "ticker": "NEM"}, None, None)
    rui._fetch_click("producers", {"id": "p2", "ticker": "ZZZ"}, None, None)
    assert "Yahoo har inte heller NEM" in session[rui.MSG_KEY]["producers:p1"]
    rui.put_live("producers:p1", sr.as_sheet_row(NEM, "producers"))
    assert set(session[rui.MSG_KEY]) == {"producers:p2"}
    monkeypatch.setattr(sr, "fetch_row", lambda api, t, ins_id=None: dict(NEM, ticker=t))
    rui._fetch_click("producers", {"id": "p2", "ticker": "ZZZ"}, None, None)
    assert session[rui.MSG_KEY] == {} and rui.is_live("producers:p2")


def test_rule_fetch_ev_now_clears_the_old_fetch_now_failure(page, monkeypatch):
    """'Hämta nu' hittade inget, sedan svarar Börsdata på 'Hämta EV nu': det gamla 'hittade inget' försvinner och
    raden under 'Hämta nu' säger vilket bolag som matchades."""
    from asymmetry import fetch as af
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: None)
    rows: dict = {}
    at = page("Rick Rule", {}, _rule_store(), rows)
    at.run()
    at.button(key="pr_live_p1").click().run()
    assert not at.exception, at.exception
    assert "Yahoo har inte heller NEM" in _captions(at)
    rows["NEM"] = NEM                                                                   # nu känner Börsdata NEM
    at.button(key="pr_evget_p1").click().run()
    assert not at.exception, at.exception
    cap = _captions(at)
    assert "Yahoo har inte heller NEM" not in cap and "Börsdata känner inte NEM" not in cap
    assert (f"🤖 Hämtat nu ur Börsdata {TODAY}: Newmont Corporation (NEM), kurs 52.3 USD. Raden saknar ins_id — "
            f"matchat på tickern.") in cap
    assert f"EV 65,000 MUSD ur Börsdata ({TODAY})." in cap


def _yahoo_answers_the_bare_ticker(monkeypatch, ticker: str, name: str, **extra):
    """Yahoo har ett annat bolag på den nakna tickern — det som en reserv efter ett Börsdata-fel skulle ge."""
    from asymmetry import fetch as af
    real = af.yahoo_row
    info = {"marketCap": 4.2e9, "currency": "USD", "currentPrice": 31.0, "sharesOutstanding": 135e6,
            "totalCash": 9e8, "longName": name, **extra}
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: real(t, lambda sym: info if sym == ticker else {}))


def test_tiggre_borsdata_error_keeps_the_night_row(page, monkeypatch):
    night = {"rows": {"tiggre:c1": sr.as_sheet_row(dict(TGR, asof="2026-11-06"), "tiggre")}}
    at = page("Tiggre", night, _tiggre_store(ins_id=9001), {"TGR": TGR})
    at.run()
    assert not at.exception, at.exception
    before = {k for k in _buttons(at) if k.startswith("rf_")}
    assert {"rf_tiggre_c1_mcap", "rf_offer_ds_ds_c1_ds_runway"} <= before
    monkeypatch.setattr(sr, "fetch_row", _raise_429)
    _yahoo_answers_the_bare_ticker(monkeypatch, "TGR", "Tegra Holdings Inc")
    at.button(key="tg_live_c1").click().run()
    assert not at.exception, at.exception
    assert "Börsdata svarade inte: 429 kvoten slut" in _captions(at)
    assert {k for k in _buttons(at) if k.startswith("rf_")} == before                  # nattens förslag står kvar
    md = _md(at)
    assert "🤖 Börsdata 2026-11-06: börsvärde MUSD 85" in md and "hämtat nu" not in md and "Tegra" not in md
    assert rui.LIVE_KEY not in at.session_state


def test_rule_borsdata_error_keeps_the_night_row(page, monkeypatch):
    night = {"rows": {"producers:p1": sr.as_sheet_row(dict(NEM, asof="2026-11-06"), "producers")}}
    at = page("Rick Rule", night, _rule_store(), {"NEM": NEM})
    at.run()
    assert not at.exception, at.exception
    assert {"rf_producers_p1_ev_musd", "rf_producers_p1_nd_ebitda"} <= _buttons(at)
    monkeypatch.setattr(sr, "fetch_row", _raise_429)
    _yahoo_answers_the_bare_ticker(monkeypatch, "NEM", "Nemo Mining Inc")
    at.button(key="pr_live_p1").click().run()
    assert not at.exception, at.exception
    assert "Börsdata svarade inte: 429 kvoten slut" in _captions(at)
    assert {"rf_producers_p1_ev_musd", "rf_producers_p1_nd_ebitda"} <= _buttons(at)
    md = _md(at)
    assert "🤖 Börsdata 2026-11-06: EV (MUSD) 65,000" in md and "hämtat nu" not in md and "Nemo" not in md
    assert rui.LIVE_KEY not in at.session_state


# ═════════════════════════════════════════════════════════════════════════════
#  Rättelser 2: Börsdata-fel på en rad utan ins_id, och Yahoos EV/EBITDA märkt Yahoo
# ═════════════════════════════════════════════════════════════════════════════
class _DownApi:
    """Börsdata som inte svarar (nyckeln utgången, 401, 429 efter tre försök, nätet): instrumentlistan och
    tickeruppslaget kastar, den globala listan ger [] som den riktiga klienten gör vid fel."""

    def __init__(self):
        self.calls: list = []

    def get_instruments(self, force_refresh=False):
        self.calls.append("get_instruments")
        raise RuntimeError("401 Unauthorized")

    def resolve_instrument_id(self, ticker):
        self.calls.append("resolve_instrument_id")
        raise RuntimeError("401 Unauthorized")

    def get_global_instruments_list(self):
        self.calls.append("get_global_instruments_list")
        return []


class _UnknownApi:
    """Börsdata svarar men känner inte tickern."""

    def get_instruments(self, force_refresh=False):
        return []

    def resolve_instrument_id(self, ticker):
        return None

    def get_global_instruments_list(self):
        return []


def test_borsdata_error_without_ins_id_gives_no_yahoo(session, monkeypatch):
    """Raden saknar ins_id (➕ Nytt bolag): tickeruppslaget i fetch_row sväljer Börsdatas fel och svarar 'känner inte'.
    fetch_live hämtar instrumentlistan först, så felet står under knappen — ingen Yahoo på bara tickern, inget lagras,
    en tidigare hämtning står kvar. Svarar Börsdata men känner inte tickern ger det Yahoo som förut."""
    from asymmetry import fetch as af
    asked = []
    monkeypatch.setattr(af, "yahoo_row", lambda t, getter=None: asked.append(t) or {
        "ticker": t, "source": "yfinance", "yahoo": t, "yahoo_name": "Nemo Mining Inc", "ev_ebitda": 3.1})
    assert sr.fetch_row(_DownApi(), "NEM") is None                     # felet sväljs här — därav kontrollen
    api = _DownApi()
    got, why = rui.fetch_live("producers", {"id": "p1", "ticker": "NEM", "name": "Newmont"}, api=api)
    assert (got, why) == (None, "Börsdata svarade inte: 401 Unauthorized")
    assert not asked and api.calls == ["get_instruments"] and rui.LIVE_KEY not in session
    rui.put_live("tiggre:c1", {"mcap_musd": 85.4})
    got, why = rui.fetch_live("tiggre", {"id": "c1", "ticker": "TGR"}, api=_DownApi())
    assert (got, why) == (None, "Börsdata svarade inte: 401 Unauthorized")
    assert not asked and rui.live_rows() == {"tiggre:c1": {"mcap_musd": 85.4}}
    got, why = rui.fetch_live("producers", {"id": "p1", "ticker": "NEM"}, api=_UnknownApi())
    assert why == "Börsdata känner inte NEM." and asked == ["NEM"] and rui.is_live("producers:p1")


def test_rule_borsdata_error_without_ins_id_keeps_the_night_row(page, monkeypatch):
    """Rick Rule-raden från ➕ Nytt bolag (inget ins_id) har nattens rad. Börsdata svarar inte på 'Hämta nu' — med den
    riktiga fetch_row: felet står under knappen, nattens EV/EBITDA och EV står kvar, inget annat bolag ur Yahoo."""
    import borsdata_api
    real_fetch_row = sr.fetch_row
    night = {"rows": {"producers:p1": sr.as_sheet_row(dict(NEM, asof="2026-11-06"), "producers")}}
    at = page("Rick Rule", night, _rule_store(), {"NEM": NEM})
    monkeypatch.setattr(sr, "fetch_row", real_fetch_row)
    monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda *a, **k: _DownApi())
    _yahoo_answers_the_bare_ticker(monkeypatch, "NEM", "Nemo Mining Inc", enterpriseToEbitda=3.1)
    at.run()
    assert not at.exception, at.exception
    before = {k for k in _buttons(at) if k.startswith("rf_")}
    assert {"rf_producers_p1_ev_musd", "rf_producers_p1_ev_ebitda"} <= before
    at.button(key="pr_live_p1").click().run()
    assert not at.exception, at.exception
    assert "Börsdata svarade inte: 401 Unauthorized" in _captions(at)
    assert {k for k in _buttons(at) if k.startswith("rf_")} == before                  # nattens förslag står kvar
    md = _md(at)
    assert "🤖 Börsdata 2026-11-06: EV/EBITDA 6.2" in md and "🤖 Börsdata 2026-11-06: EV (MUSD) 65,000" in md
    assert "hämtat nu" not in md and "Nemo" not in md and "Yahoo" not in md
    assert rui.LIVE_KEY not in at.session_state and not page.saves


def test_yahoo_ev_ebitda_is_labelled_yahoo_in_the_valuation_suggestion(session):
    """'Hämta nu' med Yahoo-reserven: Scorecardets värderingsförslag för Rick Rule och Royalty C säger Yahoo och
    symbolen som träffades — inte '(Börsdata)'. Börsdatas rader märks som förut."""
    import scorecard
    from asymmetry import fetch as af
    info = {"marketCap": 1.5e9, "currency": "CAD", "currentPrice": 12.0, "enterpriseToEbitda": 4.3,
            "longName": "Abc Gold Corp"}
    y = af.yahoo_row("ABC", lambda sym: info if sym == "ABC.TO" else {})
    rui.put_live("producers:p1", sr.as_sheet_row(y, "producers"))
    entry = {"ticker": "ABC", "rows": {"producenter": {"id": "p1", "ticker": "ABC", "name": "Abc Gold"}}}
    ref = scorecard._valuation_ref(entry, "producenter")
    assert scorecard.valuation_suggestion(entry, "producenter", ref) == "EV/EBITDA 4,3× (Yahoo ABC.TO)"
    roy = {"ticker": "ABC", "rows": {"royalty": {"id": "p1", "ticker": "ABC"}}}
    assert scorecard.valuation_suggestion(roy, "royalty", scorecard._valuation_ref(roy, "royalty")) == (
        "EV/EBITDA 4,3× (Yahoo ABC.TO)")
    assert scorecard._ev_ebitda_part({"source": "yfinance", "ev_ebitda": 4.3}) == ["EV/EBITDA 4,3× (Yahoo)"]
    rui.put_live("producers:p1", sr.as_sheet_row(NEM, "producers"))
    assert scorecard.valuation_suggestion(entry, "producenter", scorecard._valuation_ref(entry, "producenter")) == (
        "EV/EBITDA 6,2× (Börsdata)")


def test_durrett_yahoo_fallback_labels_ev_ebitda_yahoo(page, monkeypatch):
    """Durrett-arket: Börsdata känner inte NEM, Yahoo svarar på den nakna tickern. Värderingsraden säger
    '(Yahoo NEM)' — inte '(Börsdata)'."""
    _yahoo_answers_the_bare_ticker(monkeypatch, "NEM", "Nemo Mining Inc", enterpriseToEbitda=3.1)
    at = page("Durrett", {}, _durrett_store(), {})
    at.run()
    assert not at.exception, at.exception
    at.button(key="durrett_bd_go_NEM").click().run()
    assert not at.exception, at.exception
    assert "confidence:NEM" in at.session_state[rui.LIVE_KEY]
    lines = " ".join(m.value for m in at.markdown if "Värdering:" in m.value)
    assert "EV/EBITDA 3,1× (Yahoo NEM)" in lines and "(Börsdata)" not in lines
    assert at.session_state["confidence"]["companies"]["NEM"]["fields"] == {} and not page.saves
