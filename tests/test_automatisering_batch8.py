"""
Automatisering, batch 8 — Durrett och Wolf Asymmetry: rensa förslagen och härled identitetsfälten.

1. engines/durrett/refresh.proposals hoppar över fält vars FieldSpec.stages inte omfattar bolagets stage, och negativa
   tal för fält utanför confidence/data/validation._MAY_BE_NEGATIVE (samma golv som refresh_ui.suggest lo=0). FCF,
   OCF och rörelsekapital får vara negativa (ägarens beslut) — valideringen godtar dem, och motorns burn ur −OCF läser
   dem. En explorer får alltså FCF −9,5 och OCF −7,0 men inte EV/EBITDA −3,2.
2. Identitet (Durrett och Asymmetry): 'Förslag: stage producer (Durrett-motorn, …)' med Använd när klassificeringen är
   säker (hög/medel) och säger något annat än fältet. Använd sparar och ritar om Identitet-formen.
3. Produktionsenhet ur råvarans prisenhet (Använd) och en varning när enheten inte passar prisenheten — samma kontroll
   som Durrett-motorn (Ctx.quantity), nu också i Asymmetry.
4. Burn per kvartal = nattens burn/år ÷ 4 (FCF-baserad som i dag), bara explorer/developer och bara när burn > 0.
5. Högsta resurskategori = den högsta kategorin med ett tal > 0 bland arkets reserver/resurser (Använd), härledd och
   sagd som härledd.

Inget skrivs när en sida öppnas; inga trösklar ändras.
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs                       # noqa: E402
import form_keys as fk                            # noqa: E402
from confidence import config as ccfg             # noqa: E402
from confidence import store as cs                # noqa: E402
from confidence.data.models import CompanyInput   # noqa: E402
from confidence.data.provenance import dp         # noqa: E402
from confidence.data.validation import errors, usable, validate   # noqa: E402
from engines.durrett import refresh as dr         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASOF = "2026-11-07"
_BD = dict(kind="ACTUAL", source="Börsdata", source_type="secondary", pub_date=ASOF)


def _blob(ticker: str, **row) -> dict:
    return {"generated": ASOF + "T06:00", "rows": {f"confidence:{ticker}": dict(row, ticker=ticker, asof=ASOF)}}


def _explorer(ticker="EXP.V", **kw) -> CompanyInput:
    return CompanyInput(ticker=ticker, name="Test Explorer", commodity="gold", stage="explorer",
                        maturity="exploration", **kw)


def _keys(blob, company) -> dict:
    return {k: p for k, p, _cur in dr.proposals(blob, company)}


# ── 1. förslagen följer stage och golvet ─────────────────────────────────────
def test_an_explorer_gets_fcf_and_ocf_but_not_ev_ebitda():
    """Planens exempel: EV/EBITDA −3,2, FCF −9,5 och OCF −7,0. EV/EBITDA, P/E och ROIC gäller bara producenter."""
    blob = _blob("EXP.V", currency="CAD", ev_ebitda=-3.2, fcf_musd=-9.5, ocf_musd=-7.0, pe=-4.0, roic_pct=-30.0,
                 ev_ebit=-2.0, ebitda_margin=-0.5, nd_ebitda=0.4, mcap_musd=50.0, cash_musd=12.0)
    c = _explorer()
    props = _keys(blob, c)
    assert props["free_cash_flow_musd"].value == -9.5 and props["operating_cash_flow_musd"].value == -7.0
    assert not set(props) & {"ev_ebitda", "pe", "roic_pct", "ev_ebit", "ebitda_margin_pct", "net_debt_ebitda"}
    assert {"market_cap_musd", "cash_musd"} <= set(props)
    for k, p in props.items():                                            # "Använd alla"
        c.set(k, p)
    assert errors(validate(c)) == []                                     # förr två valideringsfel
    assert usable(c, "free_cash_flow_musd") and usable(c, "operating_cash_flow_musd")


def test_a_producer_skips_negatives_outside_the_may_be_negative_set():
    """Negativt EBITDA ger negativ EV/EBITDA, P/E och EV/EBIT — rutan tar ≥ 0, så de föreslås inte. Nettoskuld/EBITDA
    (nettokassa), ROIC, marginal och kassaflöden får vara negativa och föreslås som förut."""
    blob = _blob("GPR", currency="USD", ev_ebitda=-3.2, pe=-4.0, ev_ebit=-2.0, nd_ebitda=-0.3, roic_pct=-5.0,
                 ebitda_margin=-0.1, fcf_musd=-20.0, ocf_musd=-4.0, cash_musd=-1.0, debt_musd=30.0, mcap_musd=1500.0)
    props = _keys(blob, dcs.gold_producer())
    assert not set(props) & {"ev_ebitda", "pe", "ev_ebit", "cash_musd"}
    assert props["roic_pct"].value == -5.0 and props["ebitda_margin_pct"].value == -10.0
    assert props["free_cash_flow_musd"].value == -20.0 and props["operating_cash_flow_musd"].value == -4.0
    assert props["debt_musd"].value == 30.0
    assert "net_debt_ebitda" not in props                                 # samma som det sparade (−0,3)
    zero = _keys(_blob("GPR", currency="USD", ev_ebitda=0.0, pe=0.0), dcs.gold_producer())
    assert zero["ev_ebitda"].value == 0.0 and zero["pe"].value == 0.0     # golvet är 0, inte > 0


def test_proposals_follow_each_fields_stages():
    """Varje föreslaget fält gäller bolagets stage — ett fält utanför ignoreras ändå av valideringen."""
    row = dict(currency="USD", ev_ebitda=4.0, nd_ebitda=0.5, pe=8.0, roic_pct=12.0, fcf_yield_pct=6.0, ev_ebit=6.0,
               ebitda_margin=0.4, revenue_musd=10.0, fcf_musd=-3.0, ocf_musd=-2.0, mcap_musd=90.0, burn_musd=12.0)
    for stage in ccfg.STAGES:
        c = CompanyInput(ticker="T1", commodity="gold", stage=stage)
        for k in _keys(_blob("T1", **row), c):
            spec = ccfg.FIELD_BY_KEY[k]
            assert not spec.stages or stage in spec.stages, (stage, k)
    assert "ev_ebitda" in _keys(_blob("T1", **row), CompanyInput(ticker="T1", stage="royalty"))


def test_validation_accepts_negative_cash_flows_and_working_capital_only():
    c = _explorer()
    for k, v in (("free_cash_flow_musd", -9.5), ("operating_cash_flow_musd", -7.0), ("working_capital_musd", -2.0),
                 ("cash_musd", -1.0), ("revenue_musd", -3.0)):
        c.set(k, dp(v, **_BD))
    bad = {i.field for i in errors(validate(c))}
    assert bad == {"cash_musd", "revenue_musd"}                           # bara de tre nya får vara negativa
    assert all(usable(c, k) for k in ("free_cash_flow_musd", "operating_cash_flow_musd", "working_capital_musd"))
    from review import catalog                                            # ChatGPT-underlaget följer samma golv
    lo = {f.path: f.lo for f in catalog.fields_for("confidence", c.as_dict(), stage="explorer")}
    assert lo["fields.free_cash_flow_musd"] is None and lo["fields.working_capital_musd"] is None
    assert lo["fields.cash_musd"] == 0.0


def test_the_engine_now_reads_the_burn_from_a_negative_ocf():
    """balance_sheet.annual_burn: −OCF när burn per kvartal saknas — förr ett valideringsfel, så aldrig läst."""
    from engines.durrett.engine import analyze
    c = _explorer()
    c.set("cash_musd", dp(14.0, **_BD))
    c.set("operating_cash_flow_musd", dp(-7.0, **_BD))
    a = analyze(c)
    assert a.metrics["cash_runway_years"]["value"] == pytest.approx(2.0)
    c.set("quarterly_burn_musd", dp(1.0, **_BD))                          # burn per kvartal går före (som förut)
    assert analyze(c).metrics["cash_runway_years"]["value"] == pytest.approx(3.5)


# ── 2. stage-förslaget ur klassificeringen ───────────────────────────────────
def _shell(revenue=None, ticker="SHL.V", **kw) -> CompanyInput:
    """Ett skal från Holdings/Snabbkoll: stage developer, mognad PEA (CompanyInput-standard)."""
    c = CompanyInput(ticker=ticker, name="Shell Mining", commodity="gold", **kw)
    if revenue is not None:
        c.set("revenue_musd", dp(revenue, **_BD))
    return c


def test_stage_offer_from_a_confident_classification_with_its_reasons():
    stage, conf, reasons = dr.stage_offer(_shell(600.0))
    assert (stage, conf) == ("producer", "high")
    assert reasons[0] == "omsättning 600 MUSD (≥ 5)"
    pilot = dr.stage_offer(_shell(5.0))                                   # pilotintäkt på 5 MUSD räcker för motorn
    assert pilot[0] == "producer" and "omsättning 5 MUSD (≥ 5)" in pilot[2]
    assert dr.stage_offer(_shell(4.9)) is None                            # developer (medel) = fältet
    early = dr.stage_offer(_shell(maturity="exploration"))
    assert early[:2] == ("explorer", "medium") and any("Prospektering" in r for r in early[2])
    prod = dr.stage_offer(_shell(stage="developer", maturity="production"))
    assert prod[:2] == ("producer", "medium")


@pytest.mark.parametrize("company", [
    dcs.gold_producer(), dcs.copper_developer(), dcs.lithium_explorer(), dcs.royalty_company(),
    dcs.missing_everything(),
    CompanyInput(ticker="LOW", commodity="gold", stage="producer", maturity="pea"),   # motorn: UNKNOWN (låg)
    CompanyInput(ticker="DEV", commodity="gold", stage="developer", maturity="pea"),
])
def test_no_stage_offer_when_the_engine_agrees_or_is_unsure(company):
    assert dr.stage_offer(company) is None


# ── 3. produktionsenheten ────────────────────────────────────────────────────
def _with_unit(company: CompanyInput, unit) -> CompanyInput:
    if unit is None:
        company.fields.pop("production_unit", None)
    else:
        company.set("production_unit", dp(unit, kind="ACTUAL", source="Årsredovisning"))
    return company


def test_production_unit_offer_is_the_unit_the_price_is_quoted_in():
    key, point, cur = dr.production_unit_offer(_with_unit(dcs.gold_producer(), None))
    assert (key, point.value, cur) == ("production_unit", "oz", None)
    assert point.kind == "ASSUMPTION" and point.note and "USD/oz" in point.source
    assert point.source_type == "" and point.pub_date is None             # inget datum/källtyp som blåser upp DQ
    for commodity, unit in (("copper", "lb"), ("uranium", "lb"), ("lithium", "t"), ("oil_gas", "boe"),
                            ("natural_gas", "MMBtu"), ("rare_earth", "kg"), ("diamonds", "ct")):
        c = CompanyInput(ticker="U", commodity=commodity)
        assert dr.production_unit_offer(c)[1].value == unit, commodity
    assert dr.production_unit_offer(CompanyInput(ticker="U", commodity="")) is None
    assert dr.production_unit_offer(CompanyInput(ticker="U", commodity="wheat")) is None      # USc/bu: ingen bas


@pytest.mark.parametrize("commodity, unit, conflict", [
    ("gold", "oz", False), ("gold", "koz", False), ("gold", "Moz", False), ("copper", "Mlb", False),
    ("oil_gas", "boe", False), ("lithium", "kt", False),
    ("copper", "t", True), ("gold", "lb", True), ("oil_gas", "oz", True), ("lithium", "lb", True),
    ("gold", "oz (attributable)", False), ("gold", "enheter", False),      # egen text: kan inte avgöras
])
def test_production_unit_conflict_uses_the_engines_unit_tables(commodity, unit, conflict):
    c = _with_unit(CompanyInput(ticker="U", commodity=commodity), unit)
    text = dr.production_unit_conflict(c)
    assert bool(text) is conflict
    offer = dr.production_unit_offer(c)
    if conflict:
        base = dr.production_unit_offer(CompanyInput(ticker="U", commodity=commodity))[1].value
        assert f"Produktionsenheten {unit} passar inte" in text and f"Förslag: {base}" in text
        assert offer[1].value == base and offer[2] == unit
    else:
        assert offer is None                                              # enheten passar, eller är din egen text


def test_the_conflict_matches_the_engines_quantity_check():
    """Samma tabeller som Ctx.quantity: en enhetskonflikt där ger ingen mängd i motorn."""
    from engines.durrett._base import Ctx
    c = _with_unit(dcs.copper_developer(), "t")
    c.fields.pop("resource_unit")
    ctx = Ctx(c)
    assert ctx.quantity("production_future") is None and any("enhetskonflikt" in x for x in ctx.log)
    assert dr.production_unit_conflict(c)
    _with_unit(c, "Mlb")
    assert Ctx(c).quantity("production_future") is not None and dr.production_unit_conflict(c) is None


# ── 4. burn per kvartal ──────────────────────────────────────────────────────
def test_quarterly_burn_is_the_nightly_burn_over_four_for_pre_revenue_only():
    blob = _blob("EXP.V", currency="CAD", burn_musd=12.0, fcf_musd=-12.0)
    p = _keys(blob, _explorer())["quarterly_burn_musd"]
    assert p.value == 3.0 and p.unit == "MUSD" and p.kind == "ESTIMATE" and p.pub_date == ASOF
    assert p.source.startswith("Börsdata") and "÷ 4" in p.note and "FCF" in p.note
    dev = CompanyInput(ticker="EXP.V", stage="developer")
    assert _keys(blob, dev)["quarterly_burn_musd"].value == 3.0
    for stage in ("producer", "royalty"):
        assert "quarterly_burn_musd" not in _keys(blob, CompanyInput(ticker="EXP.V", stage=stage))
    for burn in (0.0, None, "x"):                                         # positivt FCF (burn 0) eller inget tal
        assert "quarterly_burn_musd" not in _keys(_blob("EXP.V", burn_musd=burn), _explorer())
    same = _explorer()
    same.set("quarterly_burn_musd", dp(3.0, **_BD))
    assert "quarterly_burn_musd" not in _keys(blob, same)                 # samma tal: inget förslag


# ── 5. högsta resurskategori ─────────────────────────────────────────────────
def test_resource_category_is_the_highest_category_with_a_number():
    c = dcs.copper_developer()                                            # reserv proven 900 Mlb, fältet 'probable'
    key, point, cur = dr.resource_category_offer(c)
    assert (key, point.value, cur) == ("resource_category", "proven", "probable")
    basis = c.get("reserve_proven")
    assert (point.source_type, point.pub_date, point.kind) == (basis.source_type, basis.pub_date, basis.kind)
    assert point.source == f"Härlett ur arket: Reserv — Proven ({basis.source})"
    assert "härlett ur arkets egna tal" in point.note
    c.set("reserve_proven", dp(0, **_BD))                                 # 0 räknas inte: probable är högst …
    assert dr.resource_category_offer(c) is None                          # … och fältet har redan probable
    c.set("resource_category", dp("inferred", **_BD))
    _k, point, cur = dr.resource_category_offer(c)
    assert (point.value, cur) == ("probable", "inferred")                 # också nedåt i förhållande till fältet
    only = CompanyInput(ticker="I", commodity="gold")
    assert dr.resource_category_offer(only) is None                       # inga tal
    only.set("resource_inferred", dp(1.2e6, **_BD))
    assert dr.resource_category_offer(only)[1].value == "inferred"
    only.set("resource_indicated", dp(-5.0, **_BD))                       # ogiltigt (negativt) räknas inte
    assert dr.resource_category_offer(only)[1].value == "inferred"


def test_the_category_order_follows_the_confidence_weights():
    from review import catalog
    cats = [cat for _k, cat in dr.CATEGORY_FIELDS]
    weights = [ccfg.RESOURCE_CATEGORY_WEIGHT[c] for c in cats]
    assert weights == sorted(weights, reverse=True) and set(cats) < set(ccfg.RESOURCE_CATEGORY_WEIGHT)
    assert all(ccfg.FIELD_BY_KEY[k].stages == () for k, _c in dr.CATEGORY_FIELDS)
    assert "resource_category" in catalog._CONF_JUDGED                  # katalogen: en bedömning, inte ett nattfält


# ── AppTest: Wolf Asymmetry ──────────────────────────────────────────────────
def _asym_app(monkeypatch, conf: dict, blob=None):
    import storage
    import positions
    import refresh_ui
    import commodity_prices

    stores = {"confidence": conf, "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores[name]) if name in stores
                                                           else default))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: [])
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob or {})
    monkeypatch.setattr(commodity_prices, "spot", lambda name: None)
    monkeypatch.setenv("ASYM_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["ASYM_TEST_ROOT"])
        from asymmetry.ui import render_asymmetry_page
        render_asymmetry_page()
    return app


def _conf(*companies) -> dict:
    from asymmetry import store as ast
    d = ast.normalize(cs.default())
    for c in companies:
        cs.put(d, c)
    return d


def _captions(at) -> list:
    return [c.value for c in at.caption]


def _company(at, ticker) -> dict:
    return at.session_state["confidence"]["companies"][ticker]


def _ark(monkeypatch, conf, pick, blob=None, mode="Ark"):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_asym_app(monkeypatch, conf, blob), default_timeout=90)
    at.session_state["asym_mode"] = mode
    at.session_state["asym_pick"] = pick
    at.run()
    assert not at.exception, at.exception
    return at


def test_asymmetry_identity_offers_the_stage_and_use_redraws_the_form(monkeypatch):
    conf = _conf(_shell(600.0))
    at = _ark(monkeypatch, conf, "SHL.V")
    hint = [c for c in _captions(at) if c.startswith("Förslag: stage producer")]
    assert hint == ["Förslag: stage producer (Durrett-motorn, säkerhet hög): omsättning 600 MUSD (≥ 5); "
                    "stage-fält: developer · mognad: PEA"]
    assert _company(at, "SHL.V")["stage"] == "developer"                  # att öppna skriver inget
    assert fk.ident_box(at.selectbox, "asym_id_SHL.V", "stage").value == "developer"
    at.button(key="cf_stage_SHL.V").click().run()
    assert not at.exception, at.exception
    assert _company(at, "SHL.V")["stage"] == "producer"
    assert fk.ident_box(at.selectbox, "asym_id_SHL.V", "stage").value == "producer"   # formen ritas om
    assert "cf_stage_SHL.V" not in {b.key for b in at.button}
    at.button(key="FormSubmitter:asym_ident_SHL.V-Uppdatera").click().run()  # Uppdatera skriver inte tillbaka
    assert not at.exception, at.exception
    assert _company(at, "SHL.V")["stage"] == "producer"


def test_asymmetry_economy_offers_the_unit_and_warns_on_a_mismatch(monkeypatch):
    gpr = dcs.gold_producer()                                             # saknar produktionsenhet
    at = _ark(monkeypatch, _conf(gpr), "GPR")
    assert "Förslag: Produktionsenhet oz (Guld prissätts i USD/oz)" in _captions(at)
    assert "production_unit" not in _company(at, "GPR")["fields"]
    at.button(key="cf_dv_GPR_production_unit").click().run()
    assert not at.exception, at.exception
    p = _company(at, "GPR")["fields"]["production_unit"]
    assert p["value"] == "oz" and p["kind"] == "ASSUMPTION"
    assert fk.box(at.text_input, "cf_GPR_", "production_unit").value == "oz"
    assert "cf_dv_GPR_production_unit" not in {b.key for b in at.button}
    # en enhet som inte passar priset: varningen i Ekonomi och på Analys-kortet
    at = _ark(monkeypatch, _conf(_with_unit(dcs.gold_producer(), "lb")), "GPR")
    warn = "⚠ Produktionsenheten lb passar inte råvarans pris i USD/oz — produktion × pris blir fel. Förslag: oz."
    assert warn in _captions(at)
    assert "Förslag: Produktionsenhet oz (Guld prissätts i USD/oz), ersätter lb" in _captions(at)
    at = _ark(monkeypatch, _conf(_with_unit(dcs.gold_producer(), "lb")), "GPR", mode="Analys")
    assert any(c.startswith("⚠ Produktionsenheten lb passar inte") for c in _captions(at))
    at = _ark(monkeypatch, _conf(_with_unit(dcs.gold_producer(), "oz")), "GPR", mode="Analys")
    assert not any("Produktionsenheten" in c for c in _captions(at))


def test_asymmetry_core_offers_the_resource_category_from_the_sheet(monkeypatch):
    at = _ark(monkeypatch, _conf(dcs.copper_developer()), "CDV")
    assert ("Förslag: Högsta resurskategori proven (härlett ur arkets egna tal: Reserv — Proven 900 000 000 lb), "
            "ersätter probable") in _captions(at)
    assert _company(at, "CDV")["fields"]["resource_category"]["value"] == "probable"
    at.button(key="cf_dv_CDV_resource_category").click().run()
    assert not at.exception, at.exception
    p = _company(at, "CDV")["fields"]["resource_category"]
    assert p["value"] == "proven" and p["source"].startswith("Härlett ur arket: Reserv — Proven")
    assert fk.box(at.selectbox, "cf_CDV_", "resource_category").value == "proven"


def test_asymmetry_borsdata_offers_quarterly_burn_and_no_negative_multiples(monkeypatch):
    lex = dcs.lithium_explorer()                                          # burn per kvartal 2,5
    blob = _blob("LEX", currency="CAD", burn_musd=12.0, fcf_musd=-12.0, ocf_musd=-9.0, ev_ebitda=-3.2, pe=-4.0)
    at = _ark(monkeypatch, _conf(lex), "LEX", blob=blob)
    keys = {b.key for b in at.button}
    assert {"asym_rf_LEX_quarterly_burn_musd", "asym_rf_LEX_free_cash_flow_musd",
            "asym_rf_LEX_operating_cash_flow_musd"} <= keys
    assert not keys & {"asym_rf_LEX_ev_ebitda", "asym_rf_LEX_pe"}
    assert _company(at, "LEX")["fields"]["quarterly_burn_musd"]["value"] == 2.5
    at.button(key="asym_rf_LEX_quarterly_burn_musd").click().run()
    assert not at.exception, at.exception
    p = _company(at, "LEX")["fields"]["quarterly_burn_musd"]
    assert p["value"] == 3.0 and p["kind"] == "ESTIMATE" and p["source"].startswith("Börsdata")


def test_opening_the_asymmetry_sheet_writes_nothing(monkeypatch):
    companies = (_shell(600.0), dcs.copper_developer(), _with_unit(dcs.gold_producer(), "lb"))
    conf = _conf(*companies)
    before = copy.deepcopy(conf)
    for pick in ("SHL.V", "CDV", "GPR"):
        at = _ark(monkeypatch, conf, pick, blob=_blob(pick, burn_musd=40.0, fcf_musd=-40.0))
        assert at.session_state["confidence"]["companies"] == before["companies"], pick


# ── AppTest: Durrett-arket ───────────────────────────────────────────────────
def _durrett(monkeypatch, conf, sub, ticker, blob=None):
    import storage
    import refresh_ui
    import screens_ui
    from streamlit.testing.v1 import AppTest

    stores = {"confidence": conf, "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores[name]) if name in stores
                                                           else default))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob or {})
    monkeypatch.setenv("DURRETT_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["DURRETT_TEST_ROOT"])
        from engines.durrett.ui import render_durrett_page
        render_durrett_page()

    at = AppTest.from_function(app, default_timeout=90)
    at.session_state[f"durrett_row_sub_{ticker}"] = sub
    at.run()
    assert not at.exception, at.exception
    return at


def test_durrett_identity_offers_the_stage(monkeypatch):
    at = _durrett(monkeypatch, _conf(_shell(600.0)), "Mer", "SHL.V")
    assert any(c.startswith("Förslag: stage producer (Durrett-motorn, säkerhet hög)") for c in _captions(at))
    assert _company(at, "SHL.V")["stage"] == "developer"
    at.button(key="cf_stage_SHL.V").click().run()
    assert not at.exception, at.exception
    assert _company(at, "SHL.V")["stage"] == "producer"
    assert fk.ident_box(at.selectbox, "conf_id_SHL.V", "stage").value == "producer"


def test_durrett_all_fields_offer_unit_and_category(monkeypatch):
    cdv = _with_unit(dcs.copper_developer(), "t")
    at = _durrett(monkeypatch, _conf(cdv), "Mer", "CDV")
    caps = _captions(at)
    assert "⚠ Produktionsenheten t passar inte råvarans pris i USD/lb — produktion × pris blir fel. Förslag: lb." in caps
    assert {"cf_dv_CDV_production_unit", "cf_dv_CDV_resource_category"} <= {b.key for b in at.button}
    at.button(key="cf_dv_CDV_production_unit").click().run()
    assert not at.exception, at.exception
    assert _company(at, "CDV")["fields"]["production_unit"]["value"] == "lb"
    assert not any(c.startswith("⚠ Produktionsenheten") for c in _captions(at))


def test_durrett_sheet_offers_fcf_ocf_and_quarterly_burn_for_an_explorer(monkeypatch):
    lex = dcs.lithium_explorer()
    blob = _blob("LEX", currency="CAD", burn_musd=12.0, fcf_musd=-12.0, ocf_musd=-9.0, ev_ebitda=-3.2)
    at = _durrett(monkeypatch, _conf(lex), "Arket", "LEX", blob=blob)
    keys = {b.key for b in at.button}
    assert {"durrett_rf_LEX_quarterly_burn_musd", "durrett_rf_LEX_free_cash_flow_musd",
            "durrett_rf_LEX_operating_cash_flow_musd"} <= keys and "durrett_rf_LEX_ev_ebitda" not in keys
    at.button(key="durrett_rf_all_LEX").click().run()
    assert not at.exception, at.exception
    f = _company(at, "LEX")["fields"]
    assert f["quarterly_burn_musd"]["value"] == 3.0 and f["operating_cash_flow_musd"]["value"] == -9.0
    assert "ev_ebitda" not in f
    assert errors(validate(cs.get(at.session_state["confidence"], "LEX"))) == []
