"""
Automatisering, batch 6 — kopior ur Durrett-arket till granskningsarken (en gemensam hjälpare, crosssheet.py).

1. crosssheet: Durrett-bolaget hittas på tickerns former (thor.ticker.forms) och ger (värde, enhet, källa, datum).
   En datapunkt vars källa börjar med 'Granskningsarket' (confidence/prefill.py) föreslås aldrig tillbaka.
2. Rick Rule: Kostnad per enhet ← AISC bara när råvaran är kortets och enheten stämmer (AISC i USD/oz på ett
   kopparkort i USD/lb föreslås aldrig); Gruvlivslängd ← mine_life_years; NAV-paret NAV + metallpris i NPV, där
   Använd också fyller NAV-källan 'Durrett-arket: <källa>, <datum>'. Fast enhetsetikett vid kostnad och pris.
3. Tiggre: NAV ← Durrett nav_musd, annars npv_musd (märkt projekt-NPV, ägarandelen inte avräknad), annars Rick
   Rule-kortets NAV med dess källa. FS, tillstånd och finansiering föreslås bara som ja. CSM Base-NAV ← kortets NAV.
4. Poängmodellen (Durrett-kortet): AISC bara för guld i USD/oz.

Ägarens beslut: nav_musd vinner över npv_musd. Allt är förslag med Använd — inget skrivs när en sida öppnas.
Inga grindar, trösklar, poängformler eller storlekar ändrade.
"""
import copy
import html
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crosssheet as xs                                   # noqa: E402
from confidence import prefill                            # noqa: E402
from confidence import store as cs                        # noqa: E402
from confidence.data.models import CompanyInput           # noqa: E402
from confidence.data.provenance import dp                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AISC_TXT = "(all-in, inte C1; biproduktkrediter enligt källan)"


def _company(ticker="EQX", commodity="gold", stage="producer", maturity="pea", **points) -> CompanyInput:
    c = CompanyInput(ticker=ticker, commodity=commodity, stage=stage, maturity=maturity)
    for key, (value, meta) in points.items():
        c.set(key, dp(value, kind=meta.pop("kind", "ESTIMATE"), **meta))
    return c


def _p(value, source="FS 2023", pub_date="2023-01-15", unit="", **meta):
    return value, {"source": source, "pub_date": pub_date, "unit": unit, **meta}


GOLD = dict(aisc=_p(1450.0, "Årsrapport 2025", "2026-03-01", "USD/enhet"),
            mine_life_years=_p(12.0, unit="år"),
            nav_musd=_p(850.0, unit="MUSD"),
            npv_price_assumption=_p(2000.0, unit="USD/oz"))


def _store(*companies) -> dict:
    data = cs.default()
    for c in companies:
        cs.put(data, copy.deepcopy(c))
    return data


# ═════════════════════════════════════════════════════════════════════════════
#  1. Hjälparen (ren)
# ═════════════════════════════════════════════════════════════════════════════
def test_the_company_is_found_on_the_tickers_forms_exact_form_first():
    data = _store(_company("EQX"), _company("ERIC B", commodity="copper"))
    assert xs.company_for("EQX.TO", data).ticker == "EQX"                  # TSX-kortet når Durrett-raden 'EQX'
    assert xs.company_for("eqx", data).ticker == "EQX"
    assert xs.company_for("ERIC-B.ST", data).ticker == "ERIC B"
    both = _store(_company("EQX"), _company("EQX.TO", commodity="silver"))
    assert xs.company_for("EQX.TO", both).commodity == "silver"            # exakt form först
    for none in ("", None, "ZZZ.TO"):
        assert xs.company_for(none, data) is None
    assert xs.company_for("EQX", None) is None and xs.company_for("EQX", {}) is None


def test_copy_returns_value_unit_source_and_date():
    c = _company(**copy.deepcopy(GOLD))
    assert xs.copy(c, "aisc") == (1450.0, "USD/enhet", "Årsrapport 2025", "2026-03-01")
    assert tuple(xs.copy(c, "mine_life_years")) == (12.0, "år", "FS 2023", "2023-01-15")
    assert xs.copy(c, "cash_cost") is None and xs.copy(None, "aisc") is None


def test_a_point_that_came_from_the_review_sheets_is_never_offered_back():
    """prefill.from_producers/from_tiggre skriver 'Granskningsarket …' — ett förslag tillbaka vore en cirkel."""
    c = _company(stage="developer")
    rule = {"producers": [{"ticker": "EQX", "unit_cost": 1500.0, "mine_life": 9.0, "price": 2500.0}]}
    tig = {"candidates": [{"ticker": "EQX", "nav": 300.0, "screen": {"permits": True, "funded": True}}]}
    for key, point in {**prefill.from_producers(c, rule), **prefill.from_tiggre(c, tig)}.items():
        c.set(key, point)
    assert c.num("aisc") == 1500.0 and c.num("nav_musd") == 300.0 and c.truth("permits_granted") is True
    for key in ("aisc", "mine_life_years", "commodity_price", "nav_musd", "npv_musd", "permits_granted",
                "financing_committed"):
        assert xs.copy(c, key) is None, key
    assert xs.cost_offer(c, "Guld") is None and xs.life_offer(c) is None and xs.nav_offer(c, "Guld") is None
    assert xs.tiggre_nav_offer(c, None) is None and xs.screen_offers(c) == []
    # du skrev om talet i Durrett-arket: källan är din, då föreslås det
    c.set("aisc", dp(1400.0, kind="ACTUAL", source="Manuell (ersätter Granskningsarket Rick Rule)"))
    assert xs.cost_offer(c, "Guld")[0] == 1400.0


# ── Rick Rule: AISC bara i kortets råvara och enhet ──────────────────────────
def test_aisc_is_offered_for_the_same_commodity_in_the_cards_unit():
    c = _company(**copy.deepcopy(GOLD))
    assert xs.cost_offer(c, "Guld", "EQX.TO") == (
        1450.0, f"🤖 Durrett-arket (EQX): AISC 1 450 USD/oz {AISC_TXT} — Årsrapport 2025, 2026-03-01")
    assert xs.cost_offer(c, "Guld", "EQX")[1].startswith("🤖 Durrett-arket: AISC")   # samma ticker: ingen form


@pytest.mark.parametrize("durrett, unit, card", [
    ("gold", "USD/oz", "Koppar"),            # AISC i USD/oz på ett kopparkort (USD/lb) — vänder domen
    ("copper", "USD/oz", "Koppar"),          # rätt råvara, men datapunkten står i USD/oz
    ("copper", "USD/t", "Koppar"),
    ("copper", "CAD/lb", "Koppar"),          # annan valuta
    ("gold", "USD/oz", "Silver"),
    ("gold", "USD/oz", ""),                  # kortet saknar råvara: aldrig en gissning
    ("", "USD/oz", "Guld"),                  # Durrett-raden saknar råvara
    ("gold", "USD/oz", "Royalty"),
])
def test_aisc_is_never_offered_across_commodities_or_units(durrett, unit, card):
    c = _company(commodity=durrett, aisc=_p(1450.0, unit=unit))
    assert xs.cost_offer(c, card) is None


def test_aisc_on_a_copper_card_in_usd_per_lb():
    c = _company(commodity="copper", aisc=_p(2.35, "Q3 2026", "2026-10-30", "USD/lb"))
    assert xs.cost_offer(c, "Koppar")[0] == 2.35
    assert "AISC 2,35 USD/lb" in xs.cost_offer(c, "Koppar")[1]
    uranium = _company(commodity="uranium", aisc=_p(38.0, unit="USD/lb"))
    assert xs.cost_offer(uranium, "Uran")[0] == 38.0                        # 'USD/lb U3O8' är lb
    for blank in (0.0, -5.0, "x"):
        assert xs.cost_offer(_company(aisc=_p(blank)), "Guld") is None


def test_the_unit_check_follows_review_units():
    assert xs.unit_fits("", "USD/oz") and xs.unit_fits("USD/enhet", "USD/lb") and xs.unit_fits("US$/oz", "USD/oz")
    assert not xs.unit_fits("USD/oz", "USD/lb") and not xs.unit_fits("CAD/oz", "USD/oz")
    assert not xs.unit_fits("MUSD", "USD/oz") and not xs.unit_fits("USD/oz", "")
    assert xs.musd_fits("MUSD") and xs.musd_fits("") and not xs.musd_fits("MCAD") and not xs.musd_fits("USD bn")


def test_mine_life_and_the_unit_label():
    assert xs.life_offer(_company(**copy.deepcopy(GOLD)), "EQX") == (
        12.0, "🤖 Durrett-arket: gruvlivslängd 12 år — FS 2023, 2023-01-15")
    assert xs.life_offer(_company(mine_life_years=_p(0.0))) is None
    assert xs.unit_label("Koppar") == "USD/lb · Koppar" and xs.unit_label("Guld") == "USD/oz · Guld"
    assert xs.unit_label("Uran") == "USD/lb U3O8 · Uran"
    assert xs.unit_label("") == "" and xs.unit_label("Royalty") == "" and xs.unit_label(None) == ""


def test_an_old_or_unsourced_study_always_shows_what_is_missing():
    c = _company(mine_life_years=(9.0, {"source": "", "pub_date": None}))
    assert xs.life_offer(c, "EQX")[1] == "🤖 Durrett-arket: gruvlivslängd 9 år — källa saknas, datum saknas"


# ── NAV: nav_musd vinner, projekt-NPV märks ──────────────────────────────────
def test_nav_pair_with_price_and_source():
    pair = xs.nav_offer(_company(**copy.deepcopy(GOLD)), "Guld", "EQX")
    assert (pair.nav, pair.price, pair.source) == (850.0, 2000.0, "Durrett-arket: FS 2023, 2023-01-15")
    assert pair.text == ("🤖 Durrett-arket: NAV 850 MUSD vid metallpris 2 000 USD/oz — FS 2023, 2023-01-15. "
                         "Använd fyller också metallpris och NAV-källa.")


def test_nav_wins_over_npv():
    c = _company(stage="developer", nav_musd=_p(850.0), npv_musd=_p(1200.0))
    assert xs.nav_offer(c, "Guld").nav == 850.0
    assert xs.tiggre_nav_offer(c, {"ticker": "EQX", "nav_musd": 10.0})[0] == 850.0


def test_only_npv_is_labelled_project_npv_not_the_companys_nav():
    c = _company(stage="developer", npv_musd=_p(1200.0, "PFS 2019", "2019-05-01"), ownership_pct=_p(80.0, unit="%"),
                 npv_price_assumption=_p(1300.0, unit="USD/oz"))
    pair = xs.nav_offer(c, "Guld", "EQX")
    assert pair.nav == 1200.0 and pair.price == 1300.0
    assert pair.source == "Durrett-arket: projekt-NPV (100 %), PFS 2019, 2019-05-01"
    assert pair.text == ("🤖 Durrett-arket: projekt-NPV (100 %), inte bolagets NAV: 1 200 MUSD — ägarandelen är inte "
                         "avräknad (ägarandel 80 %) vid metallpris 1 300 USD/oz — PFS 2019, 2019-05-01. Använd fyller "
                         "också metallpris och NAV-källa.")
    hit = xs.tiggre_nav_offer(c, None, "EQX")
    assert hit == (1200.0, "🤖 Durrett-arket: projekt-NPV (100 %), inte bolagets NAV: 1 200 MUSD — ägarandelen är "
                           "inte avräknad (ägarandel 80 %) — PFS 2019, 2019-05-01")
    no_own = _company(stage="developer", npv_musd=_p(1200.0))
    assert "ägarandelen är inte avräknad —" in xs.tiggre_nav_offer(no_own, None)[1]


def test_nav_price_stays_out_on_another_commodity_or_unit():
    c = _company(**copy.deepcopy(GOLD))
    pair = xs.nav_offer(c, "Koppar")
    assert pair.nav == 850.0 and pair.price is None
    assert pair.text.endswith("Använd fyller också NAV-källa.") and "metallpris" not in pair.text
    other = _company(nav_musd=_p(850.0), npv_price_assumption=_p(2000.0, unit="USD/lb"))
    assert xs.nav_offer(other, "Guld").price is None


def test_nav_in_another_currency_is_not_offered():
    assert xs.nav_offer(_company(nav_musd=_p(850.0, unit="MCAD")), "Guld") is None
    falls = _company(stage="developer", nav_musd=_p(850.0, unit="MCAD"), npv_musd=_p(600.0, unit="MUSD"))
    assert xs.nav_offer(falls, "Guld").source.startswith("Durrett-arket: projekt-NPV (100 %)")


# ── Tiggre ───────────────────────────────────────────────────────────────────
def test_tiggre_nav_falls_back_to_the_rule_card_with_its_source():
    rule = {"id": "p1", "ticker": "EQX.TO", "nav_musd": 640.0, "nav_source": "NI 43-101 2025"}
    assert xs.tiggre_nav_offer(None, rule, "EQX.TO") == (
        640.0, "🤖 Rick Rule-kortet: NAV efter skatt 640 MUSD — NAV-källa: NI 43-101 2025")
    assert xs.tiggre_nav_offer(_company(), dict(rule, nav_source=""), "EQX") == (
        640.0, "🤖 Rick Rule-kortet (EQX.TO): NAV efter skatt 640 MUSD — NAV-källa: saknas på kortet")
    assert xs.tiggre_nav_offer(None, dict(rule, nav_musd=0.0)) is None
    assert xs.tiggre_nav_offer(None, None) is None


def test_tiggre_nav_reads_the_rule_card_only_when_durrett_has_no_nav(monkeypatch):
    reads = []
    rule = {"producers": [{"ticker": "TGR", "nav_musd": 640.0, "nav_source": "NI 43-101 2025"}]}
    monkeypatch.setattr(xs, "producers_data", lambda: reads.append(1) or rule)
    assert xs.tiggre_nav_for("TGR", _company("TGR", nav_musd=_p(420.0)))[0] == 420.0 and not reads
    looped = _company("TGR", stage="developer")
    for key, point in prefill.from_tiggre(looped, {"candidates": [{"ticker": "TGR", "nav": 300.0}]}).items():
        looped.set(key, point)                                             # NAV och NPV kom ur Tiggre: cirkel
    assert xs.tiggre_nav_for("TGR", looped) == (
        640.0, "🤖 Rick Rule-kortet: NAV efter skatt 640 MUSD — NAV-källa: NI 43-101 2025")
    assert xs.tiggre_nav_for("ZZZ", None) is None and len(reads) == 2


def test_the_rule_row_is_found_on_the_tickers_forms():
    data = {"producers": [{"ticker": "EQX", "nav_musd": 1.0}, {"ticker": "EQX.TO", "nav_musd": 2.0}],
            "royalty": [{"ticker": "FNV", "nav_musd": 3.0}]}
    assert xs.rule_row_for("EQX.TO", data)["nav_musd"] == 2.0
    assert xs.rule_row_for("eqx", data)["nav_musd"] == 1.0
    assert xs.rule_row_for("FNV", data) is None                            # Royalty C har ingen NAV-ruta
    assert xs.rule_row_for("", data) is None and xs.rule_row_for("EQX", None) is None


@pytest.mark.parametrize("maturity, fs", [("exploration", False), ("pea", False), ("pfs", False), ("dfs", True),
                                          ("fid", True), ("construction", True), ("production", True)])
def test_screen_offers_are_only_yes(maturity, fs):
    c = _company(stage="developer", maturity=maturity,
                 permits_granted=_p(True, "Pressmeddelande", "2025-06-01"), financing_committed=_p(False))
    offers = dict(xs.screen_offers(c, "EQX"))
    assert ("fs" in offers) is fs
    assert offers["permits"] == "🤖 Durrett-arket: nyckeltillstånd beviljade — Pressmeddelande, 2025-06-01"
    assert "funded" not in offers                                          # ett nej föreslås aldrig
    if fs:
        assert offers["fs"].startswith("🤖 Durrett-arket: mognad ") and offers["fs"].endswith(" → FS klar")


def test_screen_offer_texts():
    c = _company(stage="developer", maturity="dfs", financing_committed=_p("ja", "Låneavtal", "2026-02-01"))
    assert xs.screen_offers(c, "EQX") == [("fs", "🤖 Durrett-arket: mognad DFS/BFS → FS klar"),
                                          ("funded", "🤖 Durrett-arket: finansiering åtagen — Låneavtal, 2026-02-01")]
    assert xs.screen_offers(None) == []
    assert xs.csm_base_nav_text(120.0) == "🤖 Kortet: NAV (Base) 120 MUSD — kortets NAV efter skatt, samma definition"


# ── Poängmodellen ────────────────────────────────────────────────────────────
def test_scoring_aisc_is_gold_in_usd_per_oz_only():
    c = _company(**copy.deepcopy(GOLD))
    assert xs.scoring_aisc_offer(c, "Guld", "EQX") == (
        1450.0, f"🤖 Durrett-arket: AISC 1 450 USD/oz {AISC_TXT} — Årsrapport 2025, 2026-03-01")
    assert xs.scoring_aisc_offer(_company(commodity="silver", aisc=_p(20.0, unit="USD/oz")), "Silver") is None
    assert xs.scoring_aisc_offer(_company(commodity="copper", aisc=_p(2.0, unit="USD/lb")), "Koppar") is None
    assert xs.scoring_aisc_offer(c, "Silver") is None and xs.scoring_aisc_offer(c, "") is None
    assert xs.scoring_aisc_offer(_company(aisc=_p(1450.0, unit="USD/lb")), "Guld") is None


# ═════════════════════════════════════════════════════════════════════════════
#  Arksidorna (AppTest)
# ═════════════════════════════════════════════════════════════════════════════
def _md(at) -> str:
    return html.unescape(" ".join(m.value for m in at.markdown))


def _captions(at) -> list:
    return [c.value for c in at.caption]


def _buttons(at) -> set:
    return {b.key for b in at.button}


@pytest.fixture
def page(monkeypatch):
    """Arksidan med fejkad lagring (Durrett-arket i 'confidence') och utan sifferuppdatering — inget nätverk."""
    from streamlit.testing.v1 import AppTest
    import gist_storage
    import lukacs_ui
    import refresh_ui
    import screens_ui
    import storage

    saves: list = []

    def make(name: str, stores: dict, which=None):
        monkeypatch.setattr(lukacs_ui, "_live_fx", lambda ccy: None)
        monkeypatch.setattr(gist_storage, "load_blob", lambda n, default=None: default)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(stores.get(n, default)))
        monkeypatch.setattr(storage, "legacy_gist", lambda f: None)

        def _save(n):
            saves.append(n)
            return storage.SaveResult(n, "abc1234def", "https://x", "2026-11-09T10:00:00Z")
        monkeypatch.setattr(storage, "save_session", _save)

        def app(name, root):
            import sys as _s
            _s.path.insert(0, root)
            if name == "Rick Rule":
                import producers
                producers.render_producers_page(name)
            elif name == "Poängmodell":
                import scoring
                scoring.render_scoring_page()
            else:
                import tiggre
                tiggre.render_tiggre_page()
        at = AppTest.from_function(app, args=(name, ROOT), default_timeout=90)
        if which:
            import scoring
            at.session_state[scoring.WHICH_KEY] = scoring.WHICH_FOR[which]
        at.run()
        assert not at.exception, at.exception
        return at

    make.saves = saves
    return make


# ── Rick Rule ────────────────────────────────────────────────────────────────
RULE = {"id": "p1", "ticker": "EQX.TO", "name": "Equinox Gold", "commodity": "Guld", "date": "2026-09-01"}


def _rule_stores(row=None, *companies) -> dict:
    return {"producers": {"producers": [dict(row or RULE)], "royalty": []},
            "confidence": _store(*(companies or (_company(**copy.deepcopy(GOLD)),)))}


def _rule_row(at) -> dict:
    return at.session_state["producers"]["producers"][0]


def test_rule_card_offers_aisc_life_and_nav_and_writes_nothing_on_open(page):
    stores = _rule_stores()
    at = page("Rick Rule", stores)
    md = _md(at)
    assert f"🤖 Durrett-arket (EQX): AISC 1 450 USD/oz {AISC_TXT} — Årsrapport 2025, 2026-03-01" in md
    assert "🤖 Durrett-arket (EQX): gruvlivslängd 12 år — FS 2023, 2023-01-15" in md
    assert ("🤖 Durrett-arket (EQX): NAV 850 MUSD vid metallpris 2 000 USD/oz — FS 2023, 2023-01-15. Använd fyller "
            "också metallpris och NAV-källa.") in md
    assert _captions(at).count("USD/oz · Guld") == 3                       # kostnad, pris och metallpris i NAV
    assert _rule_row(at) == RULE and not page.saves                        # att öppna skriver inget
    assert {"rf_offer_xs_pr_cost_p1", "rf_offer_xs_pr_life_p1", "rf_offer_xs_pr_nav_p1"} <= _buttons(at)


def test_rule_card_use_writes_each_field_and_the_nav_source(page):
    at = page("Rick Rule", _rule_stores())
    at.button(key="rf_offer_xs_pr_cost_p1").click().run()
    assert not at.exception, at.exception
    assert _rule_row(at)["unit_cost"] == 1450.0 and at.number_input(key="pr_c_p1").value == 1450.0
    assert "rf_offer_xs_pr_cost_p1" not in _buttons(at)
    at.button(key="rf_offer_xs_pr_life_p1").click().run()
    assert not at.exception, at.exception
    assert _rule_row(at)["mine_life"] == 12.0 and at.number_input(key="pr_life_p1").value == 12.0
    at.button(key="rf_offer_xs_pr_nav_p1").click().run()
    assert not at.exception, at.exception
    row = _rule_row(at)
    assert (row["nav_musd"], row["nav_price"], row["nav_source"]) == (850.0, 2000.0,
                                                                      "Durrett-arket: FS 2023, 2023-01-15")
    assert at.number_input(key="pr_nav_p1").value == 850.0 and at.number_input(key="pr_navp_p1").value == 2000.0
    assert at.text_input(key="pr_navs_p1").value == "Durrett-arket: FS 2023, 2023-01-15"
    assert not {"rf_offer_xs_pr_cost_p1", "rf_offer_xs_pr_life_p1", "rf_offer_xs_pr_nav_p1"} & _buttons(at)
    at.run()                                                               # ritas om: inget skrivs tillbaka
    assert not at.exception, at.exception
    assert _rule_row(at)["nav_source"] == "Durrett-arket: FS 2023, 2023-01-15" and not page.saves


def test_rule_copper_card_never_gets_an_ounce_aisc(page):
    gold_aisc = _company(commodity="copper", aisc=_p(1450.0, unit="USD/oz"), nav_musd=_p(850.0),
                         npv_price_assumption=_p(2000.0, unit="USD/oz"))
    at = page("Rick Rule", _rule_stores(dict(RULE, commodity="Koppar"), gold_aisc))
    md = _md(at)
    assert "Durrett-arket (EQX): AISC" not in md and "rf_offer_xs_pr_cost_p1" not in _buttons(at)
    assert "🤖 Durrett-arket (EQX): NAV 850 MUSD — FS 2023, 2023-01-15. Använd fyller också NAV-källa." in md
    assert "USD/lb · Koppar" in _captions(at)
    at.button(key="rf_offer_xs_pr_nav_p1").click().run()
    assert not at.exception, at.exception
    assert _rule_row(at)["nav_musd"] == 850.0 and "nav_price" not in _rule_row(at)


def test_rule_card_on_another_commodity_gets_no_cost(page):
    at = page("Rick Rule", _rule_stores(dict(RULE, commodity="Koppar")))
    assert "Durrett-arket (EQX): AISC" not in _md(at) and "rf_offer_xs_pr_cost_p1" not in _buttons(at)
    assert "rf_offer_xs_pr_life_p1" in _buttons(at)                        # livslängden är inte i en prisenhet


def test_rule_card_never_gets_back_what_it_gave_durrett(page):
    c = _company()
    for key, point in prefill.from_producers(c, {"producers": [{"ticker": "EQX", "unit_cost": 1500.0,
                                                                "mine_life": 9.0}]}).items():
        c.set(key, point)
    at = page("Rick Rule", _rule_stores(None, c))
    assert "Durrett-arket" not in _md(at)
    assert not [k for k in _buttons(at) if str(k).startswith("rf_offer_xs_")]


def test_rule_card_without_a_durrett_row_is_unchanged(page):
    at = page("Rick Rule", _rule_stores(dict(RULE, ticker="ZZZ.TO")))
    assert "Durrett-arket" not in _md(at)
    assert not [k for k in _buttons(at) if str(k).startswith("rf_offer_xs_")]


# ── Tiggre ───────────────────────────────────────────────────────────────────
CAND = {"id": "c1", "ticker": "TGR", "screen": {}, "downside": -40.0, "factors": {}, "catalysts": []}
HOLDINGS = {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0}


def _tiggre_stores(cand=None, companies=(), rule=None) -> dict:
    out = {"tiggre": {"candidates": [dict(cand or CAND)], "positions": [], "closed": [], "parked": []},
           "holdings": HOLDINGS, "confidence": _store(*companies)}
    if rule is not None:
        out["producers"] = {"producers": [rule], "royalty": []}
    return out


def _cand(at) -> dict:
    return at.session_state["tiggre"]["candidates"][0]


DEV = _company("TGR", stage="developer", maturity="dfs", nav_musd=_p(420.0, "FS 2024", "2024-09-01"),
               permits_granted=_p(True, "Pressmeddelande", "2025-06-01"),
               financing_committed=_p(True, "Granskningsarket Tiggre"))


def test_tiggre_offers_durrett_nav_and_screening_yes_and_writes_nothing_on_open(page):
    stores = _tiggre_stores(companies=(DEV,))
    at = page("Tiggre", stores)
    md = _md(at)
    assert "🤖 Durrett-arket: NAV 420 MUSD — FS 2024, 2024-09-01" in md
    assert "🤖 Durrett-arket: mognad DFS/BFS → FS klar" in md
    assert "🤖 Durrett-arket: nyckeltillstånd beviljade — Pressmeddelande, 2025-06-01" in md
    assert "finansiering åtagen" not in md                                 # kom ur Tiggre: ingen cirkel
    assert _cand(at) == dict(CAND) and not page.saves
    at.button(key="rf_offer_xs_tg_nav_c1").click().run()
    assert not at.exception, at.exception
    assert _cand(at)["nav"] == 420.0 and at.number_input(key="tg_nav_c1").value == 420.0
    at.button(key="rf_offer_xs_tg_fs_c1").click().run()
    assert not at.exception, at.exception
    at.button(key="rf_offer_xs_tg_permits_c1").click().run()
    assert not at.exception, at.exception
    assert _cand(at)["screen"] == {"fs": True, "permits": True}
    assert at.checkbox(key="tg_sc_c1_fs").value and at.checkbox(key="tg_sc_c1_permits").value
    assert not at.checkbox(key="tg_sc_c1_funded").value
    assert not [k for k in _buttons(at) if str(k).startswith("rf_offer_xs_tg_")]


def test_tiggre_project_npv_is_labelled(page):
    dev = _company("TGR", stage="developer", npv_musd=_p(900.0, "PEA 2018", "2018-04-01"))
    at = page("Tiggre", _tiggre_stores(companies=(dev,)))
    assert ("🤖 Durrett-arket: projekt-NPV (100 %), inte bolagets NAV: 900 MUSD — ägarandelen är inte avräknad — "
            "PEA 2018, 2018-04-01") in _md(at)


def test_tiggre_nav_from_the_rule_card_shows_its_source(page):
    rule = {"id": "p1", "ticker": "TGR", "commodity": "Guld", "nav_musd": 640.0, "nav_source": "NI 43-101 2025"}
    at = page("Tiggre", _tiggre_stores(rule=rule))
    assert "🤖 Rick Rule-kortet: NAV efter skatt 640 MUSD — NAV-källa: NI 43-101 2025" in _md(at)
    at.button(key="rf_offer_xs_tg_nav_c1").click().run()
    assert not at.exception, at.exception
    assert _cand(at)["nav"] == 640.0
    assert at.session_state["producers"]["producers"][0] == rule           # Rick Rule-kortet rörs inte


def test_tiggre_csm_base_nav_from_the_cards_own_nav(page):
    at = page("Tiggre", _tiggre_stores(dict(CAND, nav=120.0)))
    assert "🤖 Kortet: NAV (Base) 120 MUSD — kortets NAV efter skatt, samma definition" in _md(at)
    assert "csm" not in _cand(at) and not page.saves
    at.button(key="rf_offer_xs_csm_csm_c1").click().run()
    assert not at.exception, at.exception
    assert _cand(at)["csm"] == {"Base": {"nav_musd": 120.0}}
    assert at.number_input(key="csm_c1_Base_nav").value == 120.0
    assert "rf_offer_xs_csm_csm_c1" not in _buttons(at)
    at.run()
    assert not at.exception, at.exception
    assert _cand(at)["csm"]["Base"]["nav_musd"] == 120.0


def test_tiggre_without_nav_or_durrett_has_no_new_offers(page):
    at = page("Tiggre", _tiggre_stores())
    assert not [k for k in _buttons(at) if str(k).startswith("rf_offer_xs_")]
    assert _cand(at) == dict(CAND)


# ── Poängmodellen (Durrett-kortet) ───────────────────────────────────────────
def _scoring_stores(commodity="Guld", company=None) -> dict:
    row = {"id": "q1", "ticker": "EQX", "commodity": commodity, "factors": {}}
    return {"scoring": {"sprott": [], "durrett": [row]},
            "confidence": _store(company or _company(**copy.deepcopy(GOLD)))}


def test_scoring_durrett_card_offers_gold_aisc(page):
    at = page("Poängmodell", _scoring_stores(), which="durrett")
    assert f"🤖 Durrett-arket: AISC 1 450 USD/oz {AISC_TXT} — Årsrapport 2025, 2026-03-01" in _md(at)
    assert "aisc" not in at.session_state["scoring"]["durrett"][0] and not page.saves
    at.button(key="rf_offer_xs_sc_aisc_q1").click().run()
    assert not at.exception, at.exception
    assert at.session_state["scoring"]["durrett"][0]["aisc"] == 1450.0
    assert at.number_input(key="sc_aisc_q1").value == 1450.0
    assert "rf_offer_xs_sc_aisc_q1" not in _buttons(at)


def test_scoring_durrett_card_on_silver_gets_nothing(page):
    silver = _company(commodity="silver", aisc=_p(22.0, unit="USD/oz"))
    at = page("Poängmodell", _scoring_stores("Silver", silver), which="durrett")
    assert "rf_offer_xs_sc_aisc_q1" not in _buttons(at) and "Durrett-arket: AISC" not in _md(at)
