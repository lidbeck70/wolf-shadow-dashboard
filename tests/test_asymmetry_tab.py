"""
Wolf Asymmetry: tre poäng, ett kort, ett ark.

Fliken läser och skriver samma ark som Durrett (data/confidence.json) med en
strategi-tagg per bolag. Analys: 🚀 🛡️ 🎯 med band, en tunn rad, verdikt,
thesis killers, två hopfällda block. Ark: bara de fält poängen läser,
Börsdata, terminspris, extraktor. Det gamla data/asymmetry.json flyttas in
en gång.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from asymmetry import fetch, fields  # noqa: E402
from asymmetry import store as ast  # noqa: E402
from confidence import config as cfg  # noqa: E402
from confidence import store as cs  # noqa: E402
from ui import nav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = "🐺 Wolf Asymmetry"
BLOB = {"generated": "2026-09-25T06:00", "rows": {
    "confidence:GPR": {"ticker": "GPR", "ins_id": 105, "price": 6.4, "asof": "2026-09-25", "currency": "USD",
                       "mcap_musd": 1600.0, "ev_musd": 1520.0, "ev_ebitda": 4.0, "nd_ebitda": -0.3, "pe": 8.0,
                       "revenue_musd": 600.0, "fcf_musd": 110.0, "fx_to_usd": 1.0}}}


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── navigation ───────────────────────────────────────────────────────────────
def test_tab_is_wired_under_review_after_durrett():
    review = nav.options("review")
    assert review.index(TAB) == review.index("🧭 Durrett") + 1
    assert nav.resolve("granskning/wolf-asymmetry") == ["review", TAB]
    assert f'sub == "{TAB}"' in _src("wolf_panel.py")
    assert f"GRANSKNING → {TAB}" in _src(os.path.join("ovtlyr", "ui", "rules_page.py"))
    import sheets_refresh as sr
    assert "asymmetry" not in sr.SHEET_FILES and "asymmetry" not in sr._BUCKETS   # ett ark, inte två


# ── fälten: bara det poängen läser ───────────────────────────────────────────
def test_sheet_fields_exist_apply_to_stage_and_stay_small():
    for stage in ("producer", "royalty", "developer", "explorer"):
        keys = fields.all_keys(stage)
        assert len(keys) == len(set(keys))
        for k in keys:
            spec = cfg.FIELD_BY_KEY[k]
            assert not spec.stages or stage in spec.stages, (stage, k)
        hands_on = len(fields.economy(stage)) + len(fields.confidence_core(stage))
        assert hands_on <= 25, (stage, hands_on)                 # inte Durretts 136
    assert [f.key for f in fields.economy("producer")] == ["commodity_price", "production_current",
                                                           "production_unit", "aisc"]
    assert "npv_stress_price_musd" in [f.key for f in fields.economy("developer")]
    assert [f.key for f in fields.auto()] == ["market_cap_musd", "share_price", "basic_shares_m", "cash_musd", "debt_musd"]
    assert fields.futures_name("gold") == "guld" and fields.futures_name("uranium") is None
    # allt asymmetry-motorn läser finns i arket (utom default-antagandena)
    read = {"commodity_price", "aisc", "production_current", "annual_production", "npv_musd", "capex_musd",
            "breakeven_price", "irr_pct", "cash_musd", "debt_musd", "market_cap_musd", "npv_stress_price_musd"}
    assert read <= set(fields.all_keys("developer")) | set(fields.all_keys("producer"))


# ── lagret: samma ark som Durrett + strategi-tagg ────────────────────────────
def test_store_is_the_shared_confidence_store_with_a_strategy_tag():
    assert ast.STORE == cs.STORE == "confidence"
    d = ast.normalize(cs.default())
    cs.put(d, dcs.gold_producer())
    cs.put(d, dcs.copper_developer())
    ast.set_strategy(d, "gpr", "Viking")
    assert ast.strategy(d, "GPR") == "Viking" and ast.strategy(d, "CDV") == ast.NO_STRATEGY
    assert ast.tickers_for(d, "Viking") == ["GPR"] and ast.tickers_for(d, "Alla") == ["GPR", "CDV"]
    ast.set_strategy(d, "GPR", ast.NO_STRATEGY)
    assert d["strategies"] == {} and cs.normalize(d)["strategies"] == {}
    tags = ast.strategy_tags()
    assert tags[0] == ast.NO_STRATEGY and "Durrett" in tags and "Untagged" not in tags


def test_merge_legacy_moves_companies_once_without_overwriting():
    d = ast.normalize(cs.default())
    cs.put(d, dcs.gold_producer())
    legacy = {"companies": {"GPR": {"ticker": "GPR", "name": "gammal kopia"},
                            "CDV": dcs.copper_developer().as_dict()},
              "strategies": {"GPR": "Viking", "CDV": "Ember"},
              "commodity_overrides": {"copper": {"supply_balance_pct": {"value": -2}}}}
    assert ast.merge_legacy(d, legacy) == ["CDV"]
    assert cs.get(d, "GPR").name == "Test Gold Producer"
    assert ast.strategy(d, "CDV") == "Ember" and ast.strategy(d, "GPR") == "Viking"
    assert d["commodity_overrides"]["copper"]["supply_balance_pct"]["value"] == -2
    assert ast.merge_legacy(d, legacy) == [] and ast.merge_legacy(d, None) == []


def test_register_candidates_and_add():
    d = ast.normalize(cs.default())
    cs.put(d, dcs.gold_producer())
    rows = [{"ticker": "gpr", "name": "x", "strategy": "Viking"},
            {"ticker": "NYX", "name": "Nyx Gold", "strategy": "Momentum"},
            {"ticker": "NYX", "name": "dubblett", "strategy": "Wolf"},
            {"ticker": "ZZZ", "name": "Zed", "strategy": "Untagged"}]
    assert fetch.register_candidates(rows, d) == [("NYX", "Nyx Gold", "Momentum"), ("ZZZ", "Zed", ast.NO_STRATEGY)]
    c = fetch.add_from_register(d, "nyx", "Nyx Gold", "Momentum")
    assert c.ticker == "NYX" and c.fields == {} and ast.strategy(d, "NYX") == "Momentum"
    props = {k: p for k, p, _c in fetch.refresh_proposals(BLOB, dcs.gold_producer())}
    assert props["market_cap_musd"].value == 1600.0


# ── fliken ───────────────────────────────────────────────────────────────────
def _app(monkeypatch, conf: dict, legacy=None, register_rows=None, blob=None, futures=None):
    import streamlit as st
    import storage
    import positions
    import refresh_ui
    import commodity_prices

    stores = {"confidence": conf, "asymmetry": legacy, "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name) if stores.get(name) is not None else default))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: register_rows or [])
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob or {})
    monkeypatch.setattr(commodity_prices, "spot", lambda name: futures)
    monkeypatch.setenv("ASYM_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["ASYM_TEST_ROOT"])
        from asymmetry.ui import render_asymmetry_page
        render_asymmetry_page()

    return app


def _text(at) -> str:
    return (" ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
            + " " + " ".join(e.label for e in at.expander))


def _conf(*makers, tags=None):
    d = ast.normalize(cs.default())
    for mk in makers:
        cs.put(d, mk())
    for t, s in (tags or {}).items():
        ast.set_strategy(d, t, s)
    return d


def test_analysis_is_three_scores_on_one_card(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = _app(monkeypatch, _conf(dcs.gold_producer, dcs.copper_developer, dcs.missing_everything,
                                  tags={"GPR": "Viking", "CDV": "Ember"}))
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_mode"] = "Analys"
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="asym_mode").value == "Analys"
    assert [s.key for s in at.selectbox] == ["asym_pick"]          # en väljare, inget filter
    vals = {m.label: m.value for m in at.metric}
    assert list(vals) == ["🚀 Commodity Leverage", "🛡️ Margin of Safety", "🎯 Confidence",
                          "Asymmetri Bull/|Bear|", "Justerad uppsida", "Break-even-marginal"]
    assert vals["🚀 Commodity Leverage"] == "6/10" and vals["🛡️ Margin of Safety"] == "7/8"
    assert vals["Asymmetri Bull/|Bear|"] == "1.3×" and float(vals["🎯 Confidence"]) > 60
    labels = [e.label for e in at.expander]
    assert labels == ["Diagram", "Varför? — stegen bakom varje tal"]
    assert len(at.get("plotly_chart")) == 5
    assert "Råvarupris" in _text(at) and "Viking" in _text(at)
    assert at.selectbox(key="asym_pick").options[0] == "GPR · Viking"

    # tomt bolag: DATA_MISSING, inga diagram, ingen krasch, pekar på Ark
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_mode"] = "Analys"
    at.session_state["asym_pick"] = "NUL"
    at.run()
    assert not at.exception, at.exception
    vals = {m.label: m.value for m in at.metric}
    assert vals["🚀 Commodity Leverage"] == "DATA_MISSING" and vals["Justerad uppsida"] == "DATA_MISSING"
    assert len(at.get("plotly_chart")) == 0
    assert "DATA_MISSING:" in _text(at) and "fyll i under Ark" in _text(at)


def test_ark_shows_only_score_fields_and_writes_to_shared_store(monkeypatch):
    from streamlit.testing.v1 import AppTest

    fut = {"price": 3100.5, "unit": "USD/oz", "asof": "2026-09-26", "ticker": "GC=F"}
    app = _app(monkeypatch, _conf(dcs.gold_producer, tags={"GPR": "Viking"}), blob=BLOB, futures=fut)
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_mode"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    labels = [e.label for e in at.expander]
    assert [l.split(" ·")[0].split(" —")[0] for l in labels] == [
        "➕ Nytt bolag", "📥 Hämta från registret (Holdings)", "Identitet och strategi", "Börsdata fyller",
        "Ekonomi", "Confidence", "Confidence", "🤖 Läs ur presentationen / tekniska rapporten (AI-förslag"]
    assert any(l.startswith("Ekonomi") and l.endswith("3/4 ifyllda") for l in labels)   # GPR saknar produktionsenhet
    keys = {t.key for t in at.text_input}
    assert "cf_GPR_aisc_v" in keys and "cf_GPR_production_current_v" in keys
    assert "cf_GPR_reserve_proven_v" not in keys and "cf_GPR_grade_v" not in keys   # Durretts fält syns inte
    # Börsdata: Använd skriver till det gemensamma arket
    at.button(key="asym_rf_GPR_market_cap_musd").click().run()
    assert not at.exception, at.exception
    f = at.session_state["confidence"]["companies"]["GPR"]["fields"]
    assert f["market_cap_musd"]["value"] == 1600.0 and "Börsdata" in f["market_cap_musd"]["source"]
    # terminspriset som ASSUMPTION med källa och datum
    at.button(key="asym_fut_GPR").click().run()
    assert not at.exception, at.exception
    p = at.session_state["confidence"]["companies"]["GPR"]["fields"]["commodity_price"]
    assert p["value"] == 3100.5 and p["kind"] == "ASSUMPTION" and "GC=F" in p["source"] and p["pub_date"] == "2026-09-26"
    # ett ekonomifält
    at.text_input(key="cf_GPR_aisc_v").set_value("1400").run()
    at.button(key="FormSubmitter:asym_econ_GPR-Spara").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["companies"]["GPR"]["fields"]["aisc"]["value"] == 1400.0
    # strategi-taggen
    at.selectbox(key="asym_strat_GPR").set_value("Ember").run()
    at.button(key="FormSubmitter:asym_ident_GPR-Uppdatera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["strategies"]["GPR"] == "Ember"


def test_empty_sheet_opens_ark_and_register_import(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = _app(monkeypatch, _conf(), register_rows=[{"ticker": "NYX", "name": "Nyx Gold", "strategy": "Momentum"}])
    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="asym_mode").value == "⚡ Snabbkoll"               # standardläget
    at.radio(key="asym_mode").set_value("Ark").run()
    assert not at.exception, at.exception
    assert at.radio(key="asym_mode").value == "Ark" and not at.metric
    assert any("Inga bolag i arket" in i.value for i in at.info)
    at.button(key="asym_reg_NYX").click().run()
    assert not at.exception, at.exception
    s = at.session_state["confidence"]
    assert s["companies"]["NYX"]["name"] == "Nyx Gold" and s["strategies"]["NYX"] == "Momentum"
    at.text_input(key="asym_new_ticker").set_value("abc").run()
    at.selectbox(key="asym_new_strategy").set_value("Viking").run()
    at.button(key="FormSubmitter:asym_new-Lägg till").click().run()
    assert not at.exception, at.exception
    s = at.session_state["confidence"]
    assert "ABC" in s["companies"] and s["strategies"]["ABC"] == "Viking"


def test_uranium_has_no_futures_and_developer_sheet_shows_study_fields(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = _app(monkeypatch, _conf(dcs.copper_developer, dcs.lithium_explorer))
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_pick"] = "LEX"
    at.session_state["asym_mode"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    assert "har ingen termin på Yahoo" in _text(at) and not any(b.key == "asym_fut_LEX" for b in at.button)
    keys = {t.key for t in at.text_input}
    assert "cf_LEX_npv_musd_v" in keys and "cf_LEX_quarterly_burn_musd_v" in keys


def test_legacy_asymmetry_store_is_merged_once(monkeypatch):
    from streamlit.testing.v1 import AppTest

    legacy = {"companies": {"CDV": dcs.copper_developer().as_dict()}, "strategies": {"CDV": "Ember"},
              "commodity_overrides": {}}
    at = AppTest.from_function(_app(monkeypatch, _conf(dcs.gold_producer), legacy=legacy), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Flyttade CDV" in i.value for i in at.info)
    s = at.session_state["confidence"]
    assert "CDV" in s["companies"] and s["strategies"]["CDV"] == "Ember"
    at.run()
    assert not any("Flyttade" in i.value for i in at.info)


# ── Hämta från Börsdata nu ───────────────────────────────────────────────────
class _FakeApi:
    """Minsta möjliga Börsdata-klient: ett nordiskt bolag (FNV, USD)."""

    def get_instruments(self):
        return [{"insId": 105, "ticker": "FNV", "name": "Franco-Nevada", "stockPriceCurrency": "USD",
                 "reportCurrency": "USD"}]

    def get_global_instruments_list(self):
        return []

    def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
        return {105: {"market_cap": 30000.0, "ev_ebitda": 22.0, "net_debt_ebitda": -0.5, "roic": 0.12,
                      "p_fcf": 25.0, "ev_ebit": 30.0, "ev": 29500.0, "net_debt_m": -500.0, "revenue_m": 1200.0,
                      "fcf_m": 900.0, "ocf_m": 950.0, "pe": 40.0, "ps": 25.0, "rs_rank": 80.0, "ebitda_margin": 0.8}}

    def get_stockprices(self, ins_id, max_count=5):
        return [{"d": "2026-09-26", "c": 156.2}]

    def get_reports(self, ins_id, kind, max_count=7):
        return []


def test_fetch_row_builds_the_same_row_as_the_nightly_job(monkeypatch):
    import sheets_refresh as sr
    monkeypatch.setattr(sr, "resolve", lambda api, t, ins_id=None: 105 if t == "FNV" else None)
    row = sr.fetch_row(_FakeApi(), "fnv")
    assert row["ins_id"] == 105 and row["currency"] == "USD" and row["mcap_musd"] == 30000.0
    assert row["price"] == 156.2 and row["asof"] == "2026-09-26"
    assert row["roic_pct"] == 12.0 and row["fcf_yield_pct"] == 4.0 and row["ev_musd"] == 29500.0
    assert row["fx_to_usd"] == 1.0 and row["source"] == "borsdata"
    assert sr.fetch_row(_FakeApi(), "OKÄND") is None


def test_borsdata_now_returns_proposals_for_the_company(monkeypatch):
    import sheets_refresh as sr
    monkeypatch.setattr(sr, "resolve", lambda api, t, ins_id=None: 105 if t == "FNV" else None)
    c = dcs.royalty_company()
    c.ticker = "FNV"
    blob, props, msg = fetch.borsdata_now(c, api=_FakeApi())
    assert "confidence:FNV" in blob["rows"] and "tal ur Börsdata" in msg
    by = {k: p for k, p, _cur in props}
    assert by["market_cap_musd"].value == 30000.0 and by["share_price"].value == 156.2
    assert "Börsdata" in by["market_cap_musd"].source
    c.ticker = "NOPE"
    blob, props, msg = fetch.borsdata_now(c, api=_FakeApi())
    assert blob is None and props == [] and "känner inte NOPE" in msg


class _GlobalApi(_FakeApi):
    """Nordisk lista utan träff; global lista med FNV (NYSE) och AEM (TSX)."""

    def get_instruments(self):
        return [{"insId": 1, "ticker": "VOLV B", "stockPriceCurrency": "SEK", "reportCurrency": "SEK"}]

    def resolve_instrument_id(self, q):
        return 1 if q == "VOLV B" else None

    def get_global_instruments_list(self):
        return [{"insId": 9001, "ticker": "FNV", "name": "Franco-Nevada", "stockPriceCurrency": "USD",
                 "reportCurrency": "USD"},
                {"insId": 9002, "ticker": "AEM", "name": "Agnico Eagle", "stockPriceCurrency": "CAD",
                 "reportCurrency": "USD"}]

    def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
        assert scope == "global"
        return {i: {"market_cap": 30000.0, "ev_ebitda": 22.0} for i in ids}


def test_resolve_falls_back_to_the_global_list():
    import sheets_refresh as sr
    api = _GlobalApi()
    assert sr.resolve(api, "VOLV-B.ST") == 1
    assert sr.resolve(api, "FNV") == 9001 and sr.resolve(api, "AEM.TO") == 9002 and sr.resolve(api, "XXX") is None
    row = sr.fetch_row(api, "FNV")
    assert row["ins_id"] == 9001 and row["currency"] == "USD" and row["mcap_musd"] == 30000.0
    assert api._wolf_global_ticker_map == {"FNV": 9001, "AEM": 9002}


def test_yahoo_is_the_fallback_when_borsdata_lacks_the_company():
    infos = {"FNV.TO": {"marketCap": 5e10, "currency": "CAD", "financialCurrency": "USD", "currentPrice": 210.5,
                        "sharesOutstanding": 192_000_000, "totalCash": 1.2e9, "totalDebt": 0}}
    row = fetch.yahoo_row("FNV", lambda sym: infos.get(sym, {}))
    assert row["yahoo"] == "FNV.TO" and row["currency"] == "CAD"
    assert row["mcap_musd"] == round(5e10 * 0.73 / 1e6, 1) and row["cash_musd"] == 1200.0 and row["debt_musd"] == 0.0
    assert row["shares_now_m"] == 192.0 and row["source"] == "yfinance"
    assert fetch.yahoo_row("NOPE", lambda sym: {}) is None
    c = dcs.royalty_company()
    c.ticker = "FNV"
    blob, props, msg = fetch.fetch_now(c, api=_FakeApi(), info_getter=lambda sym: infos.get(sym, {}))
    by = {k: p for k, p, _cur in props}
    assert by["market_cap_musd"].value == row["mcap_musd"] and "Yahoo Finance (FNV.TO" in by["market_cap_musd"].source
    assert by["basic_shares_m"].value == 192.0 and "Yahoo" in msg and "känner inte FNV" in msg
    blob, props, msg = fetch.fetch_now(c, api=_FakeApi(), info_getter=lambda sym: {})
    assert blob is None and "Yahoo har inte heller FNV" in msg


def test_ark_button_fetches_from_borsdata_now(monkeypatch):
    from streamlit.testing.v1 import AppTest

    live = {"generated": "2026-09-27T05:00", "rows": {"confidence:GPR": {"ticker": "GPR", "ins_id": 105, "price": 7.1,
            "asof": "2026-09-27", "currency": "USD", "mcap_musd": 1775.0, "cash_musd": 130.0, "fx_to_usd": 1.0}}}
    calls = []

    def fake_now(company, api=None, info_getter=None):
        calls.append(company.ticker)
        from engines.durrett import refresh as dr
        return live, dr.proposals(live, company), "2 tal ur Börsdata (kurs 7.1 USD)"

    app = _app(monkeypatch, _conf(dcs.gold_producer))
    monkeypatch.setattr(fetch, "fetch_now", fake_now)
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_mode"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    assert "tryck Hämta nu" in _text(at)
    at.button(key="asym_bd_go_GPR").click().run()
    assert not at.exception, at.exception
    assert calls == ["GPR"] and "hämtade nu" in _text(at)
    at.button(key="asym_rf_GPR_market_cap_musd").click().run()
    assert not at.exception, at.exception
    f = at.session_state["confidence"]["companies"]["GPR"]["fields"]
    assert f["market_cap_musd"]["value"] == 1775.0 and f["market_cap_musd"]["pub_date"] == "2026-09-27"
