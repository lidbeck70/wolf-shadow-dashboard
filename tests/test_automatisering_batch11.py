"""
Automatisering, batch 11 — ins_id och namn följer med bolaget, och Tiggre-positionerna visar färska tal.

1. Tiggre KÖP → position: kandidatens ins_id följer med till registret (extras), och nattjobbet läser det.
2. Manuellt tillägg i Tiggre och Royalty C: ins_id och namn ur universum-registret på klicket, 'Börsdata: <namn>'
   syns. Holdings: namnet ur registret. Utan register blir raden som förut.
3. Tiggre-positioner: Avkastning, Kapital i risk och CSV:n på nattens kurs med datum. 'Antal saknas' i stället för
   0 kr, och beloppet i aktiens valuta.
4. Katalysatorer: 'förfallen sedan ÅÅÅÅ-MM — Levererad eller Försenad?' — statusen skrivs aldrig.
5. Royalty C: utspädningsraden under GEO ur nattens aktieantal.

Inget skrivs när en sida öppnas; inga grindar, trösklar eller formler ändrade.
"""
import copy
import logging
import os
import sys
from datetime import date

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import positions   # noqa: E402
import storage     # noqa: E402
from universe import registry as ur   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASOF = "2026-11-07"
LOBO = ur.Member("LOB.TO", "Lobo Mining Corp", ur.KANADA, ins_id=4242, currency="CAD")
FRANCO = ur.Member("FNV", "Franco-Nevada Corporation", ur.USA, ins_id=777, currency="USD")
LISTED = ur.Member("ABC", "Abc Royalties", ur.USA)                     # ur panelens listor: inget Börsdata-id


def _registry(monkeypatch, *members):
    """Universum-registret ur en fejkad Gist. Returnerar listan med filerna som lästes."""
    u = ur.Universe(members={m.ticker: m for m in members})
    saved: dict = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    calls: list = []

    def loader(name, fb):
        calls.append(name)
        return saved.get(name, fb)
    monkeypatch.setattr(ur, "default_loader", lambda: loader)
    return calls


# ── rena funktioner ──────────────────────────────────────────────────────────
def test_registry_identity_gives_ins_id_and_the_name_only_when_left_empty():
    import refresh_ui
    assert refresh_ui.registry_identity("LOB.TO", "", LOBO) == {
        "ins_id": 4242, "bd_name": "Lobo Mining Corp", "name": "Lobo Mining Corp"}
    assert refresh_ui.registry_identity("LOB.TO", "Lobo", LOBO) == {"ins_id": 4242, "bd_name": "Lobo Mining Corp"}
    assert refresh_ui.registry_identity("ABC", "", LISTED) == {"name": "Abc Royalties"}     # inget id att spara
    assert refresh_ui.registry_identity("ABC", "Abc", LISTED) == {}
    odd = ur.Member("X", "", ur.USA, ins_id=True)
    assert refresh_ui.registry_identity("X", "", odd) == {}


def test_registry_identity_without_a_registry_is_empty(monkeypatch):
    import refresh_ui
    assert refresh_ui.registry_identity("LOB.TO", "") == {}              # conftest: inget register i testerna
    _registry(monkeypatch, LOBO)
    assert refresh_ui.registry_identity("NOPE.TO", "") == {}             # okänd ticker
    assert refresh_ui.registry_identity("7203.T", "") == {}              # okänd marknad


def test_identity_note_names_the_match_and_warns_when_it_does_not_resemble():
    import refresh_ui
    assert refresh_ui.identity_note({"bd_name": "Lobo Mining Corp", "name": "Lobo Mining"}) == \
        "Börsdata: Lobo Mining Corp"
    assert refresh_ui.identity_note({"bd_name": "Lobo Mining Corp", "name": ""}) == "Börsdata: Lobo Mining Corp"
    assert refresh_ui.identity_note({"bd_name": "Lobo Mining Corp", "name": "Kinross"}) == \
        "Börsdata: Lobo Mining Corp — liknar inte 'Kinross'. Kontrollera tickern."
    assert refresh_ui.identity_note({"bd_name": "A $1 $2 Co"}) == "Börsdata: A \\$1 \\$2 Co"   # ingen matematik
    assert refresh_ui.identity_note({"name": "Lobo"}) == "" and refresh_ui.identity_note(None) == ""


@pytest.mark.parametrize("text, due", [
    ("2026-09", date(2026, 9, 30)), ("2026-09-15", date(2026, 9, 15)), ("2026/7", date(2026, 7, 31)),
    ("2026-Q3", date(2026, 9, 30)), ("Q3 2026", date(2026, 9, 30)), ("q1-2027", date(2027, 3, 31)),
    ("KV2 2026", date(2026, 6, 30)), ("2026 H1", date(2026, 6, 30)), ("H2 2026", date(2026, 12, 31)),
    ("2026", date(2026, 12, 31)), ("  2026-02  ", date(2026, 2, 28))])
def test_catalyst_due_reads_the_sheets_date_forms(text, due):
    import tiggre
    assert tiggre.catalyst_due(text) == due


@pytest.mark.parametrize("text", ["", None, "sen 2026", "2026-13", "2026-02-30", "Q5 2026", "H3 2026", "snart"])
def test_catalyst_due_gives_nothing_for_an_unreadable_date(text):
    import tiggre
    assert tiggre.catalyst_due(text) is None


def test_catalyst_overdue_only_for_waiting_and_first_delay_after_the_period():
    import tiggre as t
    today = date(2026, 11, 9)
    assert t.catalyst_overdue({"date": "2026-Q3", "status": t.CAT_WAITING}, today) == "2026-09"
    assert t.catalyst_overdue({"date": "2026-10-31", "status": t.CAT_LATE_1}, today) == "2026-10"
    assert t.catalyst_overdue({"date": "2026-09"}, today) == "2026-09"                      # utan status = Väntar
    assert t.catalyst_overdue({"date": "2026-11", "status": t.CAT_WAITING}, today) is None  # månaden pågår
    assert t.catalyst_overdue({"date": "2026-11-09", "status": t.CAT_WAITING}, today) is None
    for status in (t.CAT_DELIVERED, t.CAT_LATE_2, t.CAT_MISSED):
        assert t.catalyst_overdue({"date": "2025-01", "status": status}, today) is None
    assert t.catalyst_overdue({"date": "snart", "status": t.CAT_WAITING}, today) is None
    assert t.catalyst_overdue(None, today) is None


def test_price_currency_from_the_night_row_or_the_ticker():
    import tiggre as t
    assert t.price_currency("LOB.TO") == "CAD" and t.price_currency("LOB") == "USD"
    assert t.price_currency("LOB", "aud") == "AUD"
    assert t.price_currency("GGP.L") == "GBX" and t.price_currency("GGP.L", "GBP") == "GBX"   # London i pence


def test_risk_text_says_what_is_missing_and_uses_the_currency():
    import tiggre as t
    assert t.risk_text(1.0, 1.5, 0, False, "CAD") == "Antal saknas"           # förut '0 kr'
    assert t.risk_text(1.0, 1.5, None, False, "CAD") == "Antal saknas"
    assert t.risk_text(0.0, 1.5, 2000, False, "CAD") == "Entry saknas"
    assert t.risk_text(1.0, 1.5, 2000, False, "CAD") == "2 000 CAD"
    assert t.risk_text(1.0, 2.5, 2000, True, "CAD") == "0 CAD"                # free ride: insatsen är uttagen
    assert t.risk_text(1.0, 1.5, 2000, True, "CAD") == "2 000 CAD"            # halva såld men inte +100 %


def _pos(**kw):
    return {"id": "t1", "ticker": "LOB.TO", "name": "Lobo", "entry": 1.0, "current": 1.1, "shares": 0,
            "mcap": 50.0, "nav": 400.0, "half_sold": False, "triggers": {}, "catalysts": [], **kw}


def test_export_rows_use_the_nightly_price_and_its_date():
    import tiggre as t
    fresh = {"tiggre:t1": {"price": 2.1, "currency": "CAD", "asof": ASOF + "T06:00", "mcap_musd": 330.0}}
    row = t.export_rows([_pos(shares=1000)], fresh)[0]
    assert row["_px"] == 2.1 and row["_px_asof"] == ASOF and row["_ccy"] == "CAD"
    assert row["_ret"] == 110.0 and row["_risk"] == 1000.0
    assert row["_pnav"] == pytest.approx(0.82)                                  # färskt börsvärde, som sidan
    assert "+100 % — FREE RIDE" in row["_signals"] and "0.8–1× NAV — slutsälj i etapper" in row["_signals"]
    assert row["current"] == 1.1                                               # arkets sparade kurs står kvar
    cols = [c for c, _h in t.POS_CSV]
    assert {"_px", "_px_asof", "_ccy"} <= set(cols)


def test_export_rows_without_a_night_row_keep_the_sheets_numbers():
    import tiggre as t
    row = t.export_rows([_pos(shares=1000)], {})[0]
    assert row["_px"] == 1.1 and row["_px_asof"] is None and row["_ccy"] == "CAD"
    assert row["_ret"] == 10.0 and row["_pnav"] == pytest.approx(0.12) and row["_signals"] is None
    assert t.export_rows([_pos()], {})[0]["_risk"] is None                     # antal saknas: tomt, inte 0


def test_dilution_line_from_the_royalty_night_row():
    import producers as pr
    f = {"asof": ASOF, "shares_now_m": 192.4, "shares_3y_ago_m": 191.0, "shares_year": 2025}
    assert pr.dilution_line(f) == ("Utspädning: 192,40 milj. aktier mot 191,00 milj. för tre år sedan (2025 mot "
                                   "2022) = +0,7 %. Börsdata 2026-11-07.")
    shrink = pr.dilution_line({"shares_now_m": 90.0, "shares_3y_ago_m": 100.0})
    assert shrink == "Utspädning: 90,00 milj. aktier mot 100,00 milj. för tre år sedan = −10,0 %. Börsdata."
    for bad in ({}, None, {"shares_now_m": 10.0}, {"shares_now_m": 0, "shares_3y_ago_m": 10.0}):
        assert pr.dilution_line(bad) == ""


def test_the_nightly_job_writes_the_fields_the_dilution_line_reads():
    """report_fields ger shares_now_m, shares_3y_ago_m och shares_year — samma namn som raden läser."""
    import sheets_refresh as sr

    class _Api:
        def get_reports(self, ins_id, kind, max_count=1):
            if kind == "year":
                return [{"year": y, "numberOfShares": n} for y, n in ((2022, 100.0), (2023, 104.0), (2025, 110.0))]
            return []
    out = sr.report_fields(_Api(), 1)
    assert out["shares_now_m"] == 110.0 and out["shares_3y_ago_m"] == 100.0 and out["shares_year"] == 2025


# ── 1. _promote: ins_id följer med till registret ────────────────────────────
@pytest.fixture
def reg(monkeypatch):
    """Registret och Tiggre-arket mot ett tomt sessionsminne, utan nätverk. Sparningarna spelas in."""
    state: dict = {}
    saves: list = []
    seed = {"holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [],
                         "cash": 0}}
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(storage, "session_load",
                        lambda name, default=None, legacy_file=None:
                        state.setdefault(name, copy.deepcopy(seed.get(name, default))))

    def _save(name):
        saves.append((name, copy.deepcopy(state[name])))
        return storage.SaveResult(name, "abc1234def", "https://x", "2026-11-08T10:00:00Z")
    monkeypatch.setattr(storage, "save_session", _save)
    return state, saves


def test_promote_carries_the_candidates_ins_id_to_the_nightly_job(reg):
    import sheets_refresh as sr
    import tiggre
    state, _ = reg
    cand = {"id": "c1", "ticker": "LOB.TO", "name": "Lobo", "ins_id": 4242, "mcap": 50.0, "nav": 400.0,
            "catalysts": []}
    data = {"candidates": [cand], "positions": [], "closed": [], "parked": []}
    tiggre._promote(data, cand, 1.2)
    row = positions.load()["tiggre"][0]
    assert row["extras"]["ins_id"] == 4242 and row["ticker"] == "LOB.TO"            # i registrets extras
    refs = sr.collect_rows({"holdings": positions.load(), "tiggre": {}})
    assert [(r["key"], r["ins_id"]) for r in refs] == [(f"tiggre:{row['id']}", 4242)]   # via positions.to_view
    assert data["candidates"] == []


def test_promote_without_an_ins_id_adds_no_field(reg):
    import tiggre
    cand = {"id": "c1", "ticker": "LOB.TO", "name": "Lobo", "catalysts": []}
    tiggre._promote({"candidates": [cand], "positions": [], "closed": [], "parked": []}, cand, 1.2)
    assert "ins_id" not in positions.load()["tiggre"][0]["extras"]


def test_holdings_add_takes_the_name_from_the_registry_on_add(reg, monkeypatch):
    import holdings
    calls = _registry(monkeypatch, FRANCO)
    assert holdings._add_holding("long", "FNV", 150, "Materials", 10, "Royalty")
    assert positions.find("FNV")["name"] == "Franco-Nevada Corporation" and calls
    assert holdings._add_holding("long", "NEM", 40, "Materials", 5, "Quality")       # okänd: som förut
    assert positions.find("NEM")["name"] == ""


def test_holdings_add_without_a_registry_behaves_as_before(reg):
    import holdings
    assert holdings._add_holding("long", "FNV", 150, "Materials", 10, "Royalty")
    assert positions.find("FNV")["name"] == ""


# ── sidorna (AppTest) ────────────────────────────────────────────────────────
@pytest.fixture
def page(monkeypatch):
    """Arksidan med fejkad sifferuppdatering och lagring — inget nätverk. Sparningar spelas in."""
    from streamlit.testing.v1 import AppTest
    import gist_storage
    import refresh_ui
    import screens_ui

    saves: list = []

    def make(name: str, blob: dict, stores: dict):
        monkeypatch.setattr(gist_storage, "load_blob",
                            lambda n, default=None: blob if n == "sheets_refresh.json" else default)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(stores.get(n, default)))
        monkeypatch.setattr(storage, "legacy_gist", lambda f: None)

        def _save(n):
            import streamlit as st
            saves.append((n, copy.deepcopy(st.session_state[n])))
            return storage.SaveResult(n, "abc1234def", "https://x", "2026-11-08T10:00:00Z")
        monkeypatch.setattr(storage, "save_session", _save)

        def app(name, root):
            import sys as _s
            _s.path.insert(0, root)
            if name == "Tiggre":
                import tiggre
                tiggre.render_tiggre_page()
            else:
                import producers
                producers.render_producers_page(name)
        return AppTest.from_function(app, args=(name, ROOT), default_timeout=90)

    make.saves = saves
    return make


def _captions(at) -> str:
    return " ".join(c.value for c in at.caption)


def _markdown(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _empty_reg(**buckets):
    return {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0, **buckets}


def _tiggre_stores(candidates=(), held=()):
    return {"tiggre": {"candidates": list(candidates), "positions": [], "closed": [], "parked": []},
            "holdings": _empty_reg(tiggre=list(held))}


# 2. Manuellt tillägg
def test_tiggre_manual_add_takes_ins_id_and_name_from_the_registry_on_click(page, monkeypatch):
    calls = _registry(monkeypatch, LOBO)
    at = page("Tiggre", {}, _tiggre_stores())
    at.run()
    assert not at.exception, at.exception
    assert not calls                                                           # att öppna läser inget register
    at.text_input(key="tg_add_ticker").input("lob.to")
    at.button(key="tg_add_btn").click().run()
    assert not at.exception, at.exception
    new = at.session_state["tiggre"]["candidates"][-1]
    assert new["ticker"] == "LOB.TO" and new["ins_id"] == 4242
    assert new["name"] == "Lobo Mining Corp" and new["bd_name"] == "Lobo Mining Corp"
    assert "Börsdata: Lobo Mining Corp" in _captions(at)
    assert not page.saves                                                      # 💾 sparar, som förut


def test_tiggre_manual_add_keeps_your_name_and_shows_a_mismatch(page, monkeypatch):
    _registry(monkeypatch, LOBO)
    at = page("Tiggre", {}, _tiggre_stores())
    at.run()
    at.text_input(key="tg_add_ticker").input("LOB.TO")
    at.text_input(key="tg_add_name").input("Kinross")
    at.button(key="tg_add_btn").click().run()
    assert not at.exception, at.exception
    new = at.session_state["tiggre"]["candidates"][-1]
    assert new["name"] == "Kinross" and new["ins_id"] == 4242
    assert "Börsdata: Lobo Mining Corp — liknar inte 'Kinross'. Kontrollera tickern." in _captions(at)


def test_tiggre_manual_add_without_a_registry_is_as_before(page):
    at = page("Tiggre", {}, _tiggre_stores())
    at.run()
    at.text_input(key="tg_add_ticker").input("LOB.TO")
    at.button(key="tg_add_btn").click().run()
    assert not at.exception, at.exception
    new = at.session_state["tiggre"]["candidates"][-1]
    assert set(new) == {"id", "ticker", "name", "added", "screen", "downside", "factors", "catalysts"}
    assert "Börsdata:" not in _captions(at)


def test_royalty_manual_add_takes_ins_id_and_name_from_the_registry_on_click(page, monkeypatch):
    calls = _registry(monkeypatch, FRANCO)
    at = page("Royalty C", {}, {"producers": {"producers": [], "royalty": []}})
    at.run()
    assert not at.exception, at.exception
    assert not calls
    at.text_input(key="ro_new_t").input("fnv")
    at.button(key="ro_new_add").click().run()
    assert not at.exception, at.exception
    new = at.session_state["producers"]["royalty"][-1]
    assert new["ticker"] == "FNV" and new["ins_id"] == 777 and new["name"] == "Franco-Nevada Corporation"
    assert "Börsdata: Franco-Nevada Corporation" in _captions(at)


def test_royalty_manual_add_without_a_registry_is_as_before(page):
    at = page("Royalty C", {}, {"producers": {"producers": [], "royalty": []}})
    at.run()
    at.text_input(key="ro_new_t").input("FNV")
    at.button(key="ro_new_add").click().run()
    assert not at.exception, at.exception
    new = at.session_state["producers"]["royalty"][-1]
    assert set(new) == {"id", "ticker", "name", "level", "date"}


# 3. Tiggre-positionerna på nattens kurs
def _held(**extras):
    base = {"current": 1.1, "mcap": 50.0, "nav": 400.0, "half_sold": False, "triggers": {}, "catalysts": []}
    return {"id": "t1", "ticker": "LOB.TO", "name": "Lobo", "strategy": "Tiggre", "entry_price": 1.0,
            "shares": extras.pop("shares", 0), "entry_date": "2026-06-01", "source": "tiggre",
            "extras": {**base, **extras}}


def _metric(at, label):
    hits = [m for m in at.metric if m.label == label]
    assert len(hits) == 1, [m.label for m in at.metric]
    return hits[0].value


def test_tiggre_position_return_and_risk_on_the_nightly_price(page):
    blob = {"rows": {"tiggre:t1": {"ticker": "LOB.TO", "price": 1.5, "currency": "CAD", "asof": ASOF,
                                   "mcap_musd": 75.0, "source": "borsdata"}}}
    stores = _tiggre_stores(held=[_held()])
    at = page("Tiggre", blob, stores)
    at.run()
    assert not at.exception, at.exception
    assert _metric(at, "Avkastning") == "+50.0 %"                               # 1,5 mot entry 1,0 — inte 1,1
    assert _metric(at, "Kapital i risk") == "Antal saknas"                     # förut '0 kr'
    assert f"Avkastning och kapital i risk räknas på nattens kurs 1.5 CAD ({ASOF})." in _captions(at)
    assert at.number_input(key="tg_p_cur_t1").value == 1.1                     # rutan visar arkets kurs
    assert not page.saves                                                      # bara visning
    assert at.session_state["holdings"]["tiggre"][0]["extras"]["current"] == 1.1


def test_tiggre_position_risk_in_the_stocks_currency_and_zero_after_free_ride(page):
    blob = {"rows": {"tiggre:t1": {"ticker": "LOB.TO", "price": 1.5, "currency": "CAD", "asof": ASOF}}}
    at = page("Tiggre", blob, _tiggre_stores(held=[_held(shares=2000)]))
    at.run()
    assert not at.exception, at.exception
    assert _metric(at, "Kapital i risk") == "2 000 CAD"
    blob["rows"]["tiggre:t1"]["price"] = 2.5
    at2 = page("Tiggre", blob, _tiggre_stores(held=[_held(shares=2000, half_sold=True)]))
    at2.run()
    assert not at2.exception, at2.exception
    assert _metric(at2, "Kapital i risk") == "0 CAD" and _metric(at2, "Avkastning") == "+150.0 %"
    assert not page.saves


def test_tiggre_position_without_a_night_row_uses_kurs_nu(page):
    at = page("Tiggre", {}, _tiggre_stores(held=[_held(shares=500)]))
    at.run()
    assert not at.exception, at.exception
    assert _metric(at, "Avkastning") == "+10.0 %" and _metric(at, "Kapital i risk") == "500 CAD"
    assert "räknas på nattens kurs" not in _captions(at)


# 4. Katalysatorerna
def test_catalyst_badge_for_a_passed_date_and_the_status_is_never_written(page):
    cats = [{"id": "k1", "name": "Finansieringsbesked", "date": "2020-Q3", "status": "Väntar"},
            {"id": "k2", "name": "Byggstart", "date": "2020-03", "status": "Levererad"},
            {"id": "k3", "name": "FID", "date": "sent i år", "status": "Väntar"},
            {"id": "k4", "name": "First pour", "date": "2099-01", "status": "Väntar"}]
    cand = {"id": "c1", "ticker": "LOB.TO", "name": "Lobo", "screen": {}, "downside": -40.0, "factors": {},
            "catalysts": cats}
    at = page("Tiggre", {}, _tiggre_stores(candidates=[cand]))
    at.run()
    assert not at.exception, at.exception
    text = _markdown(at)
    assert "förfallen sedan 2020-09 — Levererad eller Försenad?" in text
    assert text.count("förfallen sedan") == 1                                  # inte Levererad, oläsbart eller framtid
    assert [c["status"] for c in at.session_state["tiggre"]["candidates"][0]["catalysts"]] == \
        ["Väntar", "Levererad", "Väntar", "Väntar"]


# 5. Royalty C: utspädningen under GEO
def test_royalty_dilution_line_under_geo_is_display_only(page):
    blob = {"rows": {"producers:r1": {"ticker": "FNV", "asof": ASOF, "shares_now_m": 192.4,
                                      "shares_3y_ago_m": 191.0, "shares_year": 2025}}}
    store = {"producers": {"producers": [], "royalty": [{"id": "r1", "ticker": "FNV", "name": "Franco", "level": 1,
                                                         "date": "2026-09-01"}]}}
    at = page("Royalty C", blob, store)
    at.run()
    assert not at.exception, at.exception
    assert ("Utspädning: 192,40 milj. aktier mot 191,00 milj. för tre år sedan (2025 mot 2022) = +0,7 %. "
            "Börsdata 2026-11-07.") in _captions(at)
    assert at.session_state["producers"] == store["producers"]                  # inget skrivet
    at2 = page("Royalty C", {}, store)
    at2.run()
    assert not at2.exception, at2.exception
    assert "Utspädning:" not in _captions(at2)
