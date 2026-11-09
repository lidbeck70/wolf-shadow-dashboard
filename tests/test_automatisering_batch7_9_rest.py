"""
Automatisering, batch 7 och 9 — de två sista bitarna (ägaren delegerade besluten).

A) Lukacs FV, 'År till låg skuld': en uppskattning som BARA är text — '≈ X år (nettoskuld ÷ Börsdatas FCF 12 mån,
   grovt)' ur nattens net_debt_musd och fcf12_musd (FCF via lukacs.ref_fcf_musd, samma valutaspärr för skulden),
   'nettokassa' när nettoskulden är ≤ 0, inget när ett tal saknas eller valutan är okänd. Ingen Använd, rutan fylls
   aldrig, grinden räknar som förut.
B) Scorecard, 'Screenern kvalar': kvalar bolaget i den senaste körningen av strategins egen screener (håven i
   screens.json, Insider-skanningen för Insider) ges ett förslag med Använd och körningens datum; annars texten
   'kvalade inte i senaste körningen (<datum>)'. Inget när strategin saknar screener. Arkets rad stämplas inte och
   rutan kryssas aldrig av sig själv.

Inget skrivs när sidan öppnas; inga grindar, trösklar eller formler ändrade.
"""
import copy
import json
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import insider as ins  # noqa: E402
import lukacs  # noqa: E402
import lukacs_ui  # noqa: E402
import scorecard as sc  # noqa: E402
import storage  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ══ A) År till låg skuld ═════════════════════════════════════════════════════
def night_row(net_debt_m=None, fcf_m=None, ccy: str = "USD", report: str = "USD") -> dict:
    """Rick Rule-raden som det riktiga nattjobbet (sheets_refresh.refresh) bygger den, ur Börsdatas nettoskuld och
    FCF (miljoner rapportvaluta). Inget nätverk: Börsdata är testernas fejk."""
    import sheets_refresh as sr
    from tests.test_sheets_refresh import _API, _sheets

    class _Api(_API):
        def __init__(self):
            super().__init__()
            self.snaps[40] = {"ev_ebitda": 3.0, "net_debt_ebitda": 1.4, "market_cap": 9000.0, "ev": 9500.0}
            if net_debt_m is not None:
                self.snaps[40]["net_debt_m"] = net_debt_m
            if fcf_m is not None:
                self.snaps[40]["fcf_m"] = fcf_m

        def get_instruments(self):
            return [dict(i, stockPriceCurrency=ccy, reportCurrency=report) if i["insId"] == 40 else i
                    for i in super().get_instruments()]

    sheets = _sheets()
    sheets["producers"]["producers"] = [{"id": "r1", "ticker": "BOL.ST", "nd_ebitda": 1.4}]
    out = sr.refresh(_Api(), sheets)
    assert out["error"] is None, out["error"]
    return out["rows"]["producers:r1"]


def test_years_is_net_debt_over_the_borsdata_fcf_of_the_night_row():
    ref = night_row(net_debt_m=475.0, fcf_m=190.0)
    assert ref["net_debt_musd"] == 475.0 and ref["fcf12_musd"] == 190.0
    assert lukacs_ui.delev_years_hint(ref) == "≈ 2,5 år (nettoskuld ÷ Börsdatas FCF 12 mån, grovt)"
    sek = night_row(net_debt_m=10000.0, fcf_m=2000.0, ccy="SEK", report="SEK")      # MSEK × 0,095 båda
    assert lukacs_ui.delev_years_hint(sek) == "≈ 5,0 år (nettoskuld ÷ Börsdatas FCF 12 mån, grovt)"


def test_net_cash_says_nettokassa():
    assert lukacs_ui.delev_years_hint(night_row(net_debt_m=-146.0, fcf_m=50.0)) == "nettokassa"
    assert lukacs_ui.delev_years_hint(night_row(net_debt_m=0.0)) == "nettokassa"         # FCF behövs inte då


def test_nothing_without_a_positive_fcf_or_the_numbers():
    for ref in (night_row(net_debt_m=475.0), night_row(net_debt_m=475.0, fcf_m=0.0),
                night_row(net_debt_m=475.0, fcf_m=-30.0), night_row(fcf_m=190.0), night_row(),
                None, {}, {"source": "yfinance", "price": 8.0, "currency": "USD"}):
        assert lukacs_ui.delev_years_hint(ref) == "", ref


def test_nothing_when_the_currency_is_unknown():
    zar = night_row(net_debt_m=475.0, fcf_m=190.0, report="ZAR")      # rapportvaluta utanför jobbets tabell
    assert zar["net_debt_musd"] is None and lukacs_ui.delev_years_hint(zar) == ""
    old = {"currency": "USD", "report_currency": "ZAR", "net_debt_musd": 475.0, "fcf12_musd": 190.0}   # äldre blob
    assert lukacs.ref_net_debt_musd(old) is None and lukacs_ui.delev_years_hint(old) == ""
    assert lukacs_ui.delev_years_hint({"currency": "GBX", "net_debt_musd": 475.0, "fcf12_musd": 190.0}) == ""
    assert lukacs_ui.delev_years_hint({"net_debt_musd": 475.0, "fcf12_musd": 190.0}) == ""   # ingen valuta alls
    assert lukacs_ui.delev_years_hint({"net_debt_musd": -5.0}) == ""


def test_net_debt_shares_the_fcf_guard():
    for ref in ({"currency": "USD", "net_debt_musd": 1.0, "fcf12_musd": 2.0},
                {"currency": "SEK", "report_currency": "SEK", "net_debt_musd": 1.0, "fcf12_musd": 2.0},
                {"currency": "USD", "report_currency": "ZAR", "net_debt_musd": 1.0, "fcf12_musd": 2.0},
                {"currency": "GBX", "net_debt_musd": 1.0, "fcf12_musd": 2.0}, None):
        assert (lukacs.ref_net_debt_musd(ref) is None) == (lukacs.ref_fcf_musd(ref) is None), ref


def test_short_and_long_estimates_are_readable():
    usd = {"currency": "USD"}
    assert lukacs_ui.delev_years_hint(dict(usd, net_debt_musd=1.0, fcf12_musd=100.0)).startswith("under 0,1 år")
    assert lukacs_ui.delev_years_hint(dict(usd, net_debt_musd=12345.0, fcf12_musd=1.0)).startswith("≈ 12 345,0 år")


def test_the_estimate_changes_no_gate():
    row = {"nd_ebitda": 1.4}
    before = lukacs.deleveraging_state(row["nd_ebitda"], row.get("ar_till_lag_skuld"))
    lukacs_ui.delev_years_hint({"currency": "USD", "net_debt_musd": 100.0, "fcf12_musd": 50.0})
    assert lukacs.deleveraging_state(row["nd_ebitda"], row.get("ar_till_lag_skuld")) == before
    assert before["gaps"] and "ar_till_lag_skuld" not in row


# ── Rick Rule-sidan (AppTest) ────────────────────────────────────────────────
RULE = {"id": "p1", "ticker": "EQX.TO", "name": "Equinox Gold", "commodity": "Guld", "date": "2026-09-01",
        "price": 4000.0, "unit_cost": 2000.0, "nd_ebitda": 1.4, "mine_life": 10.0, "position_pct": 3.0}
NIGHT = {"ticker": "EQX.TO", "price": 8.0, "currency": "USD", "report_currency": "USD", "asof": "2026-11-07",
         "source": "borsdata", "bd_name": "Equinox Gold Corp", "net_debt_musd": 475.0, "fcf12_musd": 190.0}
HOLDINGS = {"swing": [], "ovtlyr": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0, "long": []}


@pytest.fixture
def rule_page(monkeypatch):
    """Rick Rule-arket med fejkad lagring och sifferuppdatering — inget nätverk."""
    import gist_storage
    import refresh_ui
    import screens_ui

    def make(row=None, night=None):
        stores = {"producers": {"producers": [dict(RULE, **(row or {}))], "royalty": []},
                  "holdings": copy.deepcopy(HOLDINGS)}
        blob = {"rows": {"producers:p1": dict(NIGHT, **(night or {}))}}
        monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                            streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
        monkeypatch.setattr(storage, "load_error", lambda name: None)
        monkeypatch.setattr(storage, "is_dirty", lambda name: False)
        monkeypatch.setattr(storage, "last_saved", lambda name: None)
        monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        from streamlit.testing.v1 import AppTest

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import producers as p
            p.render_producers_page(sheet="Rick Rule")
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        at.stores = stores
        return at
    return make


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _unwritten(at) -> bool:
    """Arket som det sparades — bortsett från FV-modulens tomma scenarioskelett (setdefault sedan förut)."""
    rows = copy.deepcopy(at.session_state["producers"])
    for r in rows["producers"]:
        if isinstance(r.get("fv"), dict) and not any(r["fv"].values()):
            r.pop("fv")
    return rows == at.stores["producers"]


def test_the_card_shows_the_estimate_as_text_and_writes_nothing(rule_page):
    at = rule_page()
    md = _md(at)
    assert ("🤖 Börsdata 2026-11-07: ≈ 2,5 år (nettoskuld ÷ Börsdatas FCF 12 mån, grovt) — uppskattning, fylls "
            "inte i.") in md
    assert at.number_input(key="fv_p1_delev").value is None                 # rutan fylls inte
    assert not [b for b in at.button if "delev" in str(b.key) or "ar_till_lag_skuld" in str(b.key)]   # ingen Använd
    assert "År till låg skuld är inte ifyllt" in " ".join(e.value for e in at.error)   # grinden som förut
    assert _unwritten(at)


def test_net_cash_on_the_card(rule_page):
    at = rule_page(night={"net_debt_musd": -146.0})
    assert "🤖 Börsdata 2026-11-07: nettokassa — uppskattning, fylls inte i." in _md(at)
    assert _unwritten(at)


def test_nothing_on_the_card_when_data_or_currency_is_missing(rule_page):
    for night in ({"fcf12_musd": None}, {"net_debt_musd": None}, {"fcf12_musd": -20.0},
                  {"report_currency": "ZAR"}, {"currency": "GBX", "report_currency": None}):
        at = rule_page(night=night)
        assert "nettoskuld ÷" not in _md(at) and "nettokassa" not in _md(at), night


def test_no_estimate_when_the_debt_rule_does_not_apply(rule_page):
    at = rule_page(row={"nd_ebitda": 0.6})                                  # rutan visas inte: ingen rad heller
    assert "nettoskuld ÷" not in _md(at)


def test_a_typed_value_stays_and_the_estimate_is_still_text(rule_page):
    at = rule_page(row={"ar_till_lag_skuld": 4.0})
    assert at.number_input(key="fv_p1_delev").value == 4.0
    assert "≈ 2,5 år (nettoskuld ÷ Börsdatas FCF 12 mån, grovt)" in _md(at)
    assert at.session_state["producers"]["producers"][0]["ar_till_lag_skuld"] == 4.0
    assert "År till låg skuld 4 — kravet är under 3 år." in " ".join(e.value for e in at.error)


# ══ B) Screenern kvalar ══════════════════════════════════════════════════════
SCREENS = {"generated": "2026-11-07T05:12:00+00:00", "screens": {
    "rule": {"label": "Överlevarna (Rule)", "rows": [{"ticker": "BOL.ST", "name": "Boliden", "ins_id": 40}],
             "error": None},
    "sprott": {"label": "Optionalitet (Sprott)", "rows": [], "error": "Kräver Börsdata Pro+ global"},
    "durrett": {"label": "Durrett", "rows": [{"ticker": "GPR", "name": "Great Panther", "ins_id": 105}],
                "error": None},
    "tiggre": {"label": "Tiggre (sweet spot)", "rows": [{"ticker": "LOB.V", "name": "Lobo", "ins_id": 7}],
               "error": None},
    "royalty": {"label": "Royalty", "rows": [], "error": None}}}
SCAN = {"generated": "2026-11-06T17:01:00+00:00", "clusters": [
    {"ticker": "EVO.ST", "name": "Evolution", "ins_id": 77, "score": 7, "status": "Köpkandidat"}], "error": None}


def _entry(strategy: str, row: dict) -> dict:
    return sc.collect({strategy: [row]})[0]


def test_each_strategy_reads_its_own_screener():
    assert sc.SCREEN_KEY == {"producenter": "rule", "royalty": "royalty", "sprott": "sprott", "durrett": "durrett",
                             "confidence": "durrett", "tiggre": "tiggre"}
    import screens_scan
    assert set(sc.SCREEN_KEY.values()) <= set(screens_scan.SCREEN_BY_KEY)
    assert {screens_scan.SCREEN_BY_KEY[k].sheet for k in ("rule", "royalty", "tiggre")} == \
        {"Rick Rule", "Royalty C", "Tiggre"}
    assert sc.has_screener("insider") and not sc.has_screener("swing") and not sc.has_screener("")


def test_a_hit_in_the_latest_run_qualifies_with_its_date():
    e = _entry("producenter", {"id": "a1", "ticker": "BOL.ST", "name": "Boliden"})
    assert sc.screener_verdict(e, "producenter", SCREENS) == ("Håven Överlevarna (Rule)", "2026-11-07", True, "")
    gpr = _entry("confidence", {"ticker": "GPR", "name": "Great Panther"})          # 🧭 Durrett: Durrett-håven
    assert sc.screener_verdict(gpr, "confidence", SCREENS)[:3] == ("Håven Durrett", "2026-11-07", True)
    assert sc.screener_verdict(_entry("durrett", {"id": "d1", "ticker": "GPR"}), "durrett", SCREENS)[2] is True


def test_the_company_is_matched_on_ins_id_and_the_tickers_forms():
    lobo = _entry("tiggre", {"id": "c1", "ticker": "LOBO", "name": "Lobo", "ins_id": 7})     # annan ticker, samma id
    assert sc.screener_verdict(lobo, "tiggre", SCREENS)[2] is True
    bare = _entry("tiggre", {"id": "c1", "ticker": "LOB", "name": "Lobo"})                   # kortnamnet utan suffix
    assert sc.screener_verdict(bare, "tiggre", SCREENS)[2] is True
    other = _entry("producenter", {"id": "a1", "ticker": "BOL.ST", "name": "Boliden", "ins_id": 999})
    assert sc.screener_verdict(other, "producenter", SCREENS)[2] is False                  # annat bolag
    evo = _entry("insider", {"id": "i1", "ticker": "EVO", "name": "Evolution"})
    assert sc.screener_verdict(evo, "insider", SCAN) == ("Insider-skanningen", "2026-11-06", True, " (7/10 p)")


def test_not_qualifying_in_the_latest_run():
    e = _entry("royalty", {"id": "y1", "ticker": "FNV", "name": "Franco-Nevada"})
    assert sc.screener_verdict(e, "royalty", SCREENS) == ("Håven Royalty", "2026-11-07", False, "")
    e = _entry("insider", {"id": "i2", "ticker": "HEXA-B.ST", "name": "Hexagon"})
    assert sc.screener_verdict(e, "insider", SCAN) == ("Insider-skanningen", "2026-11-06", False, "")
    assert sc.screener_verdict(e, "insider", dict(SCAN, clusters=[]))[2] is False


def test_nothing_without_a_screener_a_run_or_when_the_screen_failed():
    e = _entry("sprott", {"id": "s1", "ticker": "XYZ"})
    assert sc.screener_verdict(e, "sprott", SCREENS) is None                               # håven felade
    assert sc.screener_verdict(e, "swing", SCREENS) is None                                # ingen screener
    for run in (None, {}, {"screens": SCREENS["screens"]}, {"generated": "2026-11-07"}):    # ingen körning/datum
        assert sc.screener_verdict(_entry("producenter", {"ticker": "BOL.ST"}), "producenter", run) is None, run
    assert sc.screener_verdict(_entry("insider", {"ticker": "EVO.ST"}), "insider", dict(SCAN, error="nät")) is None
    assert sc.screener_verdict(None, "producenter", SCREENS)[2] is False


def test_the_verdict_changes_no_gate():
    card = {"strategy": "producenter"}
    e = _entry("producenter", {"id": "a1", "ticker": "BOL.ST", "name": "Boliden"})
    before = sc.gate_state(card, e)
    sc.screener_verdict(e, "producenter", SCREENS)
    assert sc.gate_state(card, e) == before and before["checks"]["screener_kval"] is False and card == {
        "strategy": "producenter"}


# ── Scorecardet (AppTest) ────────────────────────────────────────────────────
RULE_ROW = {"id": "a1", "ticker": "BOL.ST", "name": "Boliden", "commodity": "Koppar", "price": 9000.0,
            "unit_cost": 6000.0, "ev_ebitda": 5.0, "nd_ebitda": 0.5, "mine_life": 12.0}
ROY_ROW = {"id": "y1", "ticker": "FNV", "name": "Franco-Nevada", "level": 2}
INS_ROW = {"id": "i1", "ticker": "EVO.ST", "name": "Evolution", "insiders": 1, "role": ins.ROLE_OTHER,
           "amount": 100.0, "cluster_avg": 50.0}
STORES = {
    "producers": {"producers": [RULE_ROW], "royalty": [ROY_ROW]},
    "scoring": {"sprott": [], "durrett": []},
    "insider": {"signals": [INS_ROW]},
    "tiggre": {"candidates": [], "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "allocator": {"values": {}, "positions": [], "peak": None, "current": None, "quarters_cash_high": 0,
                  "updated": ""},
    "scorecard": {"cards": {}},
    "confidence": {"companies": {}, "commodity_overrides": {}},
}
BOL, FNV, EVO = "boliden|BOL.ST", "franco-nevada|FNV", "evolution|EVO.ST"


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def page(monkeypatch):
    """Scorecardet ur en fejkad lagring och fejkade körningar — inget GitHub, ingen Gist."""
    stores = copy.deepcopy(STORES)
    runs = {"screens": copy.deepcopy(SCREENS), "scan": copy.deepcopy(SCAN)}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import gist_storage
    import refresh_ui
    import screens_ui
    monkeypatch.setattr(screens_ui, "load_screens", lambda: copy.deepcopy(runs["screens"]))
    monkeypatch.setattr(ins, "_load_auto_scan", lambda: copy.deepcopy(runs["scan"]))
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})

    def run(seed: dict = None, screens=None, scan=None):
        from streamlit.testing.v1 import AppTest
        stores.update(copy.deepcopy(seed or {}))
        if screens is not None:
            runs["screens"] = screens
        if scan is not None:
            runs["scan"] = scan

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import scorecard as s
            s.render_scorecard_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        return at
    return run


def _caps(at) -> str:
    return " ".join(c.value for c in at.caption)


def test_opening_the_page_shows_the_screeners_and_writes_nothing(page):
    at = page()
    caps, md = _caps(at), _md(at)
    assert "🤖 Håven Överlevarna (Rule) 2026-11-07: kvalar i senaste körningen — förslag, inte inskrivet." in caps
    assert "🤖 Insider-skanningen 2026-11-06: kvalar i senaste körningen (7/10 p) — förslag, inte inskrivet." in caps
    assert "🤖 Håven Royalty: kvalade inte i senaste körningen (2026-11-07)" in md
    keys = {b.key for b in at.button}
    assert f"sc_usescr_{BOL}" in keys and f"sc_usescr_{EVO}" in keys and f"sc_usescr_{FNV}" not in keys
    for k in (BOL, FNV, EVO):
        assert at.checkbox(key=f"sc_g_{k}_screener_kval").value is False            # aldrig kryssad av sig själv
    for name in ("scorecard", "producers", "insider"):
        assert _same(at.session_state[name], STORES[name]), name                      # inget skrivet, ingen stämpel


def test_anvand_ticks_the_box_and_the_suggestion_goes_away(page):
    at = page()
    at.button(key=f"sc_usescr_{BOL}").click().run()
    assert not at.exception, at.exception
    card = at.session_state["scorecard"]["cards"][BOL]
    assert card["screener_kval"] is True and card["strategy"] == "producenter"
    assert at.checkbox(key=f"sc_g_{BOL}_screener_kval").value is True
    assert "Håven Överlevarna (Rule) 2026-11-07: kvalar" not in _caps(at)
    assert set(at.session_state["scorecard"]["cards"]) == {BOL}
    assert _same(at.session_state["producers"], STORES["producers"])                  # arkets rad stämplas inte


def test_a_ticked_box_gets_no_suggestion(page):
    at = page(seed={"scorecard": {"cards": {BOL: {"strategy": "producenter", "screener_kval": True}}}})
    assert f"sc_usescr_{BOL}" not in {b.key for b in at.button}
    assert "Håven Överlevarna (Rule) 2026-11-07: kvalar" not in _caps(at)


def test_no_run_or_a_failed_screen_shows_nothing(page):
    at = page(screens={}, scan={})
    t = _caps(at) + _md(at)
    assert "kvalar i senaste" not in t and "kvalade inte" not in t
    assert not [b for b in at.button if str(b.key).startswith("sc_usescr_")]
    bad = copy.deepcopy(SCREENS)
    bad["screens"]["rule"]["error"] = "Börsdata svarade inte"
    at = page(screens=bad, scan=dict(SCAN, error="nät"))
    t = _caps(at) + _md(at)
    assert "Överlevarna" not in t and "Insider-skanningen" not in t
    assert "🤖 Håven Royalty: kvalade inte i senaste körningen (2026-11-07)" in _md(at)   # royalty-håven gick bra
