"""
Automatisering, batch 5 — råvarupriset nu på fler ark (nattfält ur Yahoo-terminer, 0 Börsdata-anrop).

1. sheets_refresh lägger commodity_price, commodity_unit, commodity_asof och commodity_ticker också på Durrett-arkets
   rader (råvaran via registrets rotation_key) och Poängmodellens rader. En hämtning per termin, delad med Rick Rule.
   Rader utan råvara (eller utan termin) får inget.
2. engines/durrett/refresh föreslår Råvarupris bara när nattens enhet är registrets (olja: ett fat = en boe, ägarens
   beslut; järnmalm TIO=F i USD/t). Wolf Asymmetry får samma förslag. Okänd kursvaluta stoppar inte priset (USD).
3. Poängmodellen (Durrett): 'Spot X USD/oz (datum) · vinst vid spot Y MUSD/år' vid Målpris; Durrett-arket: spot och
   motorns framtida vinst vid spot. Bara visning.
4. Rick Rule: marginalen med dagens pris bredvid den sparade. Bara visning.

Durrett-nattlarmet räknar på det sparade priset (ägarens beslut). Inget skrivs när en sida öppnas.
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import commodity_prices as cp        # noqa: E402
import durrett_cases as dcs          # noqa: E402
import producers                     # noqa: E402
import scoring                       # noqa: E402
import sheets_refresh as sr          # noqa: E402
from confidence import commodities as com   # noqa: E402
from engines.durrett import refresh as dr   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLD = {"commodity_price": 4012.3, "commodity_unit": "USD/oz", "commodity_asof": "2026-11-06",
        "commodity_ticker": "GC=F"}


def _night(**kw) -> dict:
    return dict(GOLD, **kw)


# ── Terminerna och registret (rent) ─────────────────────────────────────────
@pytest.mark.parametrize("key, name, ticker", [
    ("gold", "guld", "GC=F"), ("silver", "silver", "SI=F"), ("platinum", "platina", "PL=F"),
    ("palladium", "palladium", "PA=F"), ("copper", "koppar", "HG=F"), ("oil_gas", "olja", "CL=F"),
    ("natural_gas", "gas", "NG=F"), ("iron_ore", "jarnmalm", "TIO=F"),
])
def test_the_registry_rotation_key_is_the_futures_name_and_the_units_match(key, name, ticker):
    assert cp.registry_name(key) == name
    spec = cp.ticker_for(name)
    assert spec[0] == ticker
    assert cp.unit_fits(spec[1], com.REGISTRY[key].unit), (spec, com.REGISTRY[key].unit)


def test_commodities_without_futures_get_nothing():
    for key in ("uranium", "lithium", "zinc", "coal"):
        assert cp.ticker_for(cp.registry_name(key)) is None, key
    for key in ("rare_earth", "nickel", "", None, "okänt"):
        assert cp.registry_name(key) is None, key


def test_iron_ore_is_tio_and_only_where_the_registry_says_usd_per_tonne():
    assert com.REGISTRY["iron_ore"].unit == "USD/t"
    assert cp.ticker_for("Järnmalm") == cp.ticker_for("jarnmalm") == ("TIO=F", "USD/t")
    assert "62 % Fe" in cp.BASIS["TIO=F"] and cp.BASIS["NG=F"] == "Henry Hub"
    assert "järnmalm" not in cp.NOT_ON_YAHOO and "jarnmalm" not in cp.NOT_ON_YAHOO
    from asymmetry import fields
    assert fields.futures_name("iron_ore") == "jarnmalm"                  # 'Hämta termin' säger samma sak


def test_units_fit_only_when_they_are_the_same():
    assert cp.unit_fits("USD/oz", "USD/oz") and cp.unit_fits("USD/fat", "USD/boe")    # ett fat = en boe
    assert not cp.unit_fits("USD/boe", "USD/fat")                                     # bara åt det hållet
    assert not cp.unit_fits("USD/lb", "USD/lb U3O8") and not cp.unit_fits("USD/oz", "USD/lb")
    assert not cp.unit_fits("", "USD/oz") and not cp.unit_fits("USD/oz", None)


def test_spot_many_fetches_each_future_once(monkeypatch):
    calls = []

    def fake(name):
        calls.append(name)
        t, u = cp.ticker_for(name)
        return {"price": 100.0, "unit": u, "asof": "2026-11-06", "ticker": t}
    monkeypatch.setattr(cp, "spot", fake)
    out = cp.spot_many(["Järnmalm", "jarnmalm", "Guld", "guld", "Uran", "", None])
    assert sorted(cp.ticker_for(n)[0] for n in calls) == ["GC=F", "TIO=F"]
    assert set(out) == {"järnmalm", "jarnmalm", "guld"} and out["järnmalm"] is out["jarnmalm"]


# ── 1. Nattjobbet ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sheet, bucket, commodity, name", [
    ("producers", "producers", "Guld", "Guld"), ("scoring", "durrett", "Guld", "Guld"),
    ("scoring", "sprott", "Silver", "Silver"), ("confidence", "companies", "gold", "guld"),
    ("confidence", "companies", "uranium", "uran"), ("confidence", "companies", "", None),
    ("scoring", "durrett", "", None), ("producers", "royalty", "Guld", None), ("insider", "signals", "Guld", None),
    ("tiggre", "candidates", "Guld", None),
])
def test_which_rows_ask_for_a_commodity_price(sheet, bucket, commodity, name):
    assert sr.commodity_name({"sheet": sheet, "bucket": bucket, "row": {"commodity": commodity}}) == name


def _batch5_sheets():
    from test_sheets_refresh import _sheets
    sheets = _sheets()
    sheets["producers"]["producers"][0]["commodity"] = "Guld"
    sheets["producers"]["producers"].append({"id": "r2", "ticker": "ZZFE.AX", "commodity": "Järnmalm"})
    sheets["producers"]["royalty"][0]["commodity"] = "Guld"                  # Royalty C: inget pris
    sheets["scoring"]["durrett"][0]["commodity"] = "Guld"
    sheets["scoring"]["sprott"] = [{"id": "s1", "ticker": "ZZ.V", "commodity": ""}]
    sheets["confidence"]["companies"]["IOR"] = {"ticker": "IOR", "commodity": "iron_ore"}
    sheets["confidence"]["companies"]["URX"] = {"ticker": "URX", "commodity": "uranium"}
    sheets["confidence"]["companies"]["NOC"] = {"ticker": "NOC", "commodity": ""}
    return sheets


@pytest.fixture
def futures(monkeypatch):
    """Yahoo fejkat: guld och järnmalm svarar, varje anrop räknas. Inget nät för rader utan Börsdata-träff."""
    calls = []
    prices = {"GC=F": 4012.3, "TIO=F": 104.5}

    def fake(name):
        spec = cp.ticker_for(name)
        calls.append(spec[0] if spec else name)
        if not spec or spec[0] not in prices:
            return None
        return {"price": prices[spec[0]], "unit": spec[1], "asof": "2026-11-06", "ticker": spec[0]}
    monkeypatch.setattr(cp, "spot", fake)
    monkeypatch.setattr(sr, "_yf_close", lambda t, quiet=False, with_currency=False: (None, None))
    monkeypatch.setattr(sr, "_confidence_yf_price", lambda t, quiet=False: {"price": None, "asof": None})
    return calls


def test_the_night_puts_the_price_on_durrett_and_scoring_rows_once_per_future(futures):
    from test_sheets_refresh import _API
    out = sr.refresh(_API(), _batch5_sheets())
    assert out["error"] is None, out["error"]
    rows = out["rows"]
    assert sorted(futures) == ["GC=F", "TIO=F"]                             # en hämtning per termin, delad
    for key in ("producers:r1", "scoring:d1", "confidence:GPR"):
        r = rows[key]
        assert (r["commodity_price"], r["commodity_unit"], r["commodity_asof"], r["commodity_ticker"]) == \
            (4012.3, "USD/oz", "2026-11-06", "GC=F"), key
    for key in ("producers:r2", "confidence:IOR"):                         # Järnmalm / iron_ore: samma TIO=F
        assert (rows[key]["commodity_price"], rows[key]["commodity_unit"], rows[key]["commodity_ticker"]) == \
            (104.5, "USD/t", "TIO=F"), key
    for key in ("scoring:s1", "confidence:URX", "confidence:NOC", "producers:y1", "insider:i1", "tiggre:c1",
                "tiggre:p1"):
        assert not {k for k in rows[key] if k.startswith("commodity_")}, key


def test_the_commodity_price_costs_no_borsdata_calls(monkeypatch, futures):
    from collections import Counter
    from test_sheets_refresh import _Counting
    a1 = _Counting()
    sr.refresh(a1, _batch5_sheets())
    monkeypatch.setattr(cp, "spot", lambda name: None)                     # Yahoo svarar inte: inget pris
    a2 = _Counting()
    out = sr.refresh(a2, _batch5_sheets())
    assert Counter(a1.calls) == Counter(a2.calls)
    assert not [k for r in out["rows"].values() for k in r if k.startswith("commodity_")]


def test_the_durrett_alarm_keeps_the_saved_price():
    """Ägarens beslut: nattlarmet räknar på det sparade råvarupriset — nattens termin ändrar inte händelsen."""
    from test_sheets_refresh import _durrett_company
    row = dict(_durrett_company(), id="GPR")
    s = {"mcap_musd": 2000.0, "asof": "2026-09-10"}
    before = sr._durrett_engine_events(row, s)
    assert before and "köpregeln uppfylld" in before[0]["body"]
    for price in (1000.0, 99999.0):                                         # under AISC / orimligt högt
        assert sr._durrett_engine_events(row, dict(s, **_night(commodity_price=price))) == before


# ── 2. Durrett-arkets och Wolf Asymmetrys förslag ───────────────────────────
def _props(company, night, ticker=None):
    blob = {"generated": "2026-11-07", "rows": {f"confidence:{ticker or company.ticker}": night}}
    return {k: (p, cur) for k, p, cur in dr.proposals(blob, company)}


def test_durrett_gets_the_futures_price_as_a_proposal():
    c = dcs.gold_producer()                                                  # sparat råvarupris 3 000 USD/oz
    p, cur = _props(c, dict(_night(), price=6.0, currency="USD", asof="2026-11-07"))["commodity_price"]
    assert (p.value, p.unit, p.kind, p.source, p.pub_date, cur) == \
        (4012.3, "USD/oz", "ASSUMPTION", "Yahoo GC=F (termin)", "2026-11-06", 3000.0)
    assert "terminspris, inte spot eller realiserat pris" in p.note
    assert c.num("commodity_price") == 3000.0                               # förslaget skriver ingenting


def test_no_proposal_when_the_price_is_the_saved_one_or_for_another_commodity():
    c = dcs.gold_producer()
    assert "commodity_price" not in _props(c, _night(commodity_price=3000.0))
    assert "commodity_price" not in _props(c, _night(commodity_ticker="SI=F", commodity_price=48.2))
    c.commodity = ""
    assert "commodity_price" not in _props(c, _night())
    c.commodity = "uranium"                                                  # ingen termin
    assert "commodity_price" not in _props(c, _night(commodity_unit="USD/lb", commodity_ticker=""))
    assert "commodity_price" not in _props(dcs.gold_producer(), _night(commodity_price=None))


def test_no_proposal_when_the_unit_differs_from_the_registry():
    c = dcs.copper_developer()                                               # USD/lb
    hg = {"commodity_price": 4.52, "commodity_unit": "USD/lb", "commodity_asof": "2026-11-06",
          "commodity_ticker": "HG=F"}
    assert _props(c, hg)["commodity_price"][0].unit == "USD/lb"
    assert "commodity_price" not in _props(c, dict(hg, commodity_unit="USD/t"))     # t.ex. en ändrad termin


@pytest.mark.parametrize("key, night, unit, note", [
    ("oil_gas", {"commodity_price": 61.5, "commodity_unit": "USD/fat", "commodity_ticker": "CL=F"}, "USD/boe",
     "ett fat räknas som en boe"),
    ("natural_gas", {"commodity_price": 3.1, "commodity_unit": "USD/MMBtu", "commodity_ticker": "NG=F"}, "USD/MMBtu",
     "Henry Hub"),
    ("iron_ore", {"commodity_price": 104.5, "commodity_unit": "USD/t", "commodity_ticker": "TIO=F"}, "USD/t",
     "62 % Fe CFR Kina, per torrton"),
])
def test_oil_gas_and_iron_ore_follow_the_owner_decisions(key, night, unit, note):
    c = dcs.gold_producer()
    c.commodity = key
    p, _cur = _props(c, dict(night, commodity_asof="2026-11-06"))["commodity_price"]
    assert p.unit == unit and note in p.note and p.value == night["commodity_price"]


def test_an_unknown_share_currency_does_not_stop_the_commodity_price():
    """Batch 18: en kursvaluta utanför FX_TO_USD ger inga omräknade tal — råvarupriset står i USD och föreslås."""
    c = dcs.gold_producer()
    keys = set(_props(c, dict(_night(), price=15.2, currency="GBX", mcap_musd=197600.0), ticker="GPR"))
    assert "commodity_price" in keys and not keys & {"share_price", "market_cap_musd", "market_currency"}


def test_wolf_asymmetry_and_the_chatgpt_export_get_the_same_proposal():
    from asymmetry import fetch
    from review import export
    c = dcs.gold_producer()
    blob = {"generated": "2026-11-07", "rows": {"confidence:GPR": _night()}}
    assert "commodity_price" in {k for k, _p, _c in fetch.refresh_proposals(blob, c)}
    assert export._conf_proposals(blob, c.as_dict())["commodity_price"].source == "Yahoo GC=F (termin)"


def test_spot_earnings_run_the_engine_on_the_futures_price():
    c = dcs.gold_producer()
    e = dr.spot_earnings(_night(), c)
    assert (e["price"], e["unit"], e["asof"]) == (4012.3, "USD/oz", "2026-11-06")
    assert e["fcf_spot"] > e["fcf_saved"] > 0                                 # 4 012 mot 3 000 USD/oz
    assert c.num("commodity_price") == 3000.0                               # bolaget rörs inte
    assert dr.spot_earnings(_night(commodity_price=3000.0), c) is None       # samma pris: ingen rad
    assert dr.spot_earnings(None, c) is None


def test_durrett_spot_line_text():
    from engines.durrett.ui import spot_line
    c = dcs.gold_producer()
    txt = spot_line(c, _night())
    assert txt.startswith("🤖 Spot 4 012 USD/oz (2026-11-06) · framtida vinst vid spot ")
    assert "(på sparat pris " in txt and txt.endswith("terminspris, bara visning.")
    assert spot_line(c, _night(commodity_ticker="SI=F")) == ""


# ── 3. Poängmodellen (Durrett): spot vid Målpris ────────────────────────────
def test_scoring_spot_line():
    assert scoring.spot_profit_line(_night(), "Guld", 200.0, 1450.0) == \
        "🤖 Spot 4 012 USD/oz (2026-11-06) · vinst vid spot 512 MUSD/år — terminspris, bara visning."
    assert scoring.spot_profit_line(_night(), "Guld", None, 1450.0) == \
        "🤖 Spot 4 012 USD/oz (2026-11-06) — terminspris, bara visning."
    assert "spot under AISC, ingen vinst" in scoring.spot_profit_line(_night(), "Guld", 200.0, 5000.0)
    assert scoring.spot_profit_line(_night(), "Silver", 200.0, 1450.0) == ""        # råvaran bytt sedan natten
    assert scoring.spot_profit_line(_night(), "", 200.0, 1450.0) == ""
    cu = {"commodity_price": 4.52, "commodity_unit": "USD/lb", "commodity_ticker": "HG=F"}
    assert scoring.spot_profit_line(cu, "Koppar", 200.0, 3.0) == ""                 # räknaren står i $/oz
    assert scoring.spot_profit_line({}, "Guld", 200.0, 1450.0) == ""
    assert scoring.spot_profit_line(None, "Guld", 200.0, 1450.0) == ""


# ── 4. Rick Rule: marginalen med dagens pris ────────────────────────────────
def test_rule_spot_margin_line():
    assert producers.spot_margin_line(_night(), "Guld", 1450.0, 3000.0) == (
        "🤖 Med dagens pris 4 012 USD/oz (2026-11-06): marginal 63.9 % (2 p) — poängen räknas på det sparade priset.")
    assert "(1 p)" in producers.spot_margin_line(_night(), "Guld", 2900.0, None)    # 27.7 %
    assert producers.spot_margin_line(_night(), "Guld", 1450.0, 4012.3) == ""        # redan dagens pris
    assert producers.spot_margin_line(_night(), "Guld", None, 3000.0) == ""          # ingen kostnad
    assert producers.spot_margin_line(_night(), "Olja", 1450.0, 3000.0) == ""        # fel råvara
    assert producers.spot_margin_line({}, "Guld", 1450.0, 3000.0) == ""


HG = {"commodity_price": 4.52, "commodity_unit": "USD/lb", "commodity_asof": "2026-11-06", "commodity_ticker": "HG=F"}
NG = {"commodity_price": 3.10, "commodity_unit": "USD/MMBtu", "commodity_asof": "2026-11-06", "commodity_ticker": "NG=F"}
CU_NOTE = "🤖 Dagens pris 4,52 USD/lb står nog i en annan enhet än kortets — ingen marginal räknad."


def test_rule_spot_margin_line_skips_a_card_in_another_unit():
    """Fix 1: koppar per ton (LME) mot HG=F i USD/lb, eller gas per boe mot NG=F i USD/MMBtu, ger ingen marginal."""
    assert producers.spot_margin_line(HG, "Koppar", 5500.0, 9900.0) == CU_NOTE           # sparat pris per ton
    assert producers.spot_margin_line(HG, "Koppar", 5500.0, None) == CU_NOTE             # bara kostnaden per ton
    assert producers.spot_margin_line(HG, "Koppar", 2.5, 4.1).startswith(
        "🤖 Med dagens pris 4,52 USD/lb (2026-11-06): marginal 44.7 % (2 p)")             # samma enhet: som förut
    assert producers.spot_margin_line(NG, "Gas", 11.0, 24.0) == (                       # per boe, 7,7× terminen
        "🤖 Dagens pris 3,10 USD/MMBtu står nog i en annan enhet än kortets — ingen marginal räknad.")
    assert producers.spot_margin_line(NG, "Gas", 11.0, None) == ""                       # gas kräver sparat pris
    assert producers.spot_margin_line(NG, "Gas", 1.8, 3.5) == (
        "🤖 Med dagens pris 3,10 USD/MMBtu (2026-11-06): marginal 41.9 % (2 p) — poängen räknas på det sparade "
        "priset.")
    # guldet som förut, också utan sparat pris
    assert "marginal 63.9 % (2 p)" in producers.spot_margin_line(_night(), "Guld", 1450.0, 3000.0)
    assert "marginal 27.7 % (1 p)" in producers.spot_margin_line(_night(), "Guld", 2900.0, None)


# ── AppTest: korten visar raderna och skriver ingenting ─────────────────────
@pytest.fixture
def card(monkeypatch):
    """Rick Rule eller Poängmodellen (Durrett) med fejkad lagring och sifferuppdatering — inget nätverk."""
    import gist_storage
    import refresh_ui
    import screens_ui
    import storage

    def make(page, rows, blob):
        bucket = {"Rick Rule": ("producers", "producers"), "Durrett": ("scoring", "durrett")}[page]
        store = ({"producers": [dict(r) for r in rows], "royalty": []} if bucket[0] == "producers"
                 else {"sprott": [], "durrett": [dict(r) for r in rows]})
        stores = {bucket[0]: store}
        monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                            streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
        monkeypatch.setattr(storage, "load_error", lambda name: None)
        monkeypatch.setattr(storage, "is_dirty", lambda name: False)
        monkeypatch.setattr(storage, "last_saved", lambda name: None)
        monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        from streamlit.testing.v1 import AppTest

        def app(root, page):
            import sys as _s
            _s.path.insert(0, root)
            if page == "Rick Rule":
                import producers as p
                p.render_producers_page(sheet="Rick Rule")
            else:
                import scoring as s
                s.render_scoring_page()
        at = AppTest.from_function(app, args=(ROOT, page), default_timeout=90)
        if page == "Durrett":
            at.session_state[scoring.WHICH_KEY] = scoring.WHICH[1]
        at.run()
        assert not at.exception, at.exception
        at.rows_now = lambda: at.session_state[bucket[0]][bucket[1]]
        return at
    return make


def _captions(at) -> list:
    return [c.value for c in at.caption]


SC_ROW = {"id": "d1", "ticker": "NEM", "name": "Newmont", "commodity": "Guld", "date": "2026-09-01", "factors": {},
          "prod": 200.0, "aisc": 1450.0, "target": 3000.0}
RULE_ROW = {"id": "p1", "ticker": "NEM", "name": "Newmont", "commodity": "Guld", "date": "2026-09-01",
            "unit_cost": 1450.0, "price": 3000.0}


def test_scoring_card_shows_spot_by_the_target_price_and_writes_nothing(card):
    at = card("Durrett", [SC_ROW], {"rows": {"scoring:d1": _night()}})
    assert ("🤖 Spot 4 012 USD/oz (2026-11-06) · vinst vid spot 512 MUSD/år — terminspris, bara visning."
            in _captions(at))
    assert at.rows_now() == [SC_ROW]                                         # att öppna skriver inget
    plain = card("Durrett", [SC_ROW], {})
    assert {b.key for b in at.button} == {b.key for b in plain.button}      # ingen Använd
    assert not [c for c in _captions(plain) if c.startswith("🤖 Spot")]


def test_scoring_card_drops_the_line_when_the_commodity_changed(card):
    at = card("Durrett", [dict(SC_ROW, commodity="Silver")], {"rows": {"scoring:d1": _night()}})
    assert not [c for c in _captions(at) if c.startswith("🤖 Spot")]


def test_rule_card_shows_the_margin_with_todays_price_and_writes_nothing(card):
    at = card("Rick Rule", [RULE_ROW], {"rows": {"producers:p1": _night()}})
    assert ("🤖 Med dagens pris 4 012 USD/oz (2026-11-06): marginal 63.9 % (2 p) — poängen räknas på det sparade "
            "priset." in _captions(at))
    assert at.rows_now() == [RULE_ROW]
    plain = card("Rick Rule", [RULE_ROW], {})
    assert {b.key for b in at.button} - {b.key for b in plain.button} == {"rf_producers_p1_price"}   # bara förslaget


def test_rule_card_in_tonnes_shows_no_margin_with_the_pound_price(card):
    """Fix 1: ett kopparkort per ton får notisen, inte en marginal på -121 581 %. Att öppna skriver inget."""
    row = dict(RULE_ROW, commodity="Koppar", unit_cost=5500.0, price=9900.0)
    at = card("Rick Rule", [row], {"rows": {"producers:p1": HG}})
    caps = _captions(at)
    assert CU_NOTE in caps
    assert not [c for c in caps if c.startswith("🤖 Med dagens pris")]
    assert at.rows_now() == [row]


# ── AppTest: Durrett-arket ───────────────────────────────────────────────────
def test_durrett_sheet_offers_the_price_with_use_and_shows_the_spot_line(monkeypatch):
    import storage
    import refresh_ui
    import screens_ui
    from confidence import store as cs
    from streamlit.testing.v1 import AppTest

    data = cs.default()
    cs.put(data, dcs.gold_producer())
    stores = {"confidence": data, "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {"generated": "2026-11-07", "rows": {"confidence:GPR": dict(
        _night(), ticker="GPR", price=6.0, asof="2026-11-07", currency="USD", mcap_musd=1500.0)}})
    monkeypatch.setenv("DURRETT_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["DURRETT_TEST_ROOT"])
        from engines.durrett.ui import render_durrett_page
        render_durrett_page()

    at = AppTest.from_function(app, default_timeout=90)
    at.session_state["durrett_row_sub_GPR"] = "Arket"
    at.run()
    assert not at.exception, at.exception
    fields = lambda: at.session_state["confidence"]["companies"]["GPR"]["fields"]   # noqa: E731
    assert fields()["commodity_price"]["value"] == 3000.0                  # att öppna skriver inget
    text = " ".join(m.value for m in at.markdown)
    assert "Råvarupris nu: 4,012 USD/oz" in text and "terminspris, inte spot" in text
    assert any(c.startswith("🤖 Spot 4 012 USD/oz (2026-11-06) · framtida vinst vid spot ") for c in _captions(at))
    at.button(key="durrett_rf_GPR_commodity_price").click().run()
    assert not at.exception, at.exception
    p = fields()["commodity_price"]
    assert (p["value"], p["source"], p["kind"], p["pub_date"], p["unit"]) == \
        (4012.3, "Yahoo GC=F (termin)", "ASSUMPTION", "2026-11-06", "USD/oz")
    assert not [c for c in _captions(at) if c.startswith("🤖 Spot")]        # använt: raden borta
    assert "durrett_rf_GPR_commodity_price" not in {b.key for b in at.button}
