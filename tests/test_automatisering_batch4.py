"""
Automatisering, batch 4 — rätt råvara från början (Rick Rule, Poängmodellen, Durrett och Wolf Asymmetry).

Ägarens beslut: den härledda råvaran FÖRESLÅS med Använd, den fylls aldrig i av sig själv.

1. thor.ticker.commodity_for(ticker, branch_id): theme_for utan arken → confidence.commodities → arkets namn.
   None när det är osäkert (bransch 17 och 18, polymetalliska bolag, teman med flera råvaror) — aldrig en gissning.
2. screens_ui.row_to_fields tar med branch_id ur håvens rad.
3. Håvens ➕ i Rick Rule och Poängmodellen skriver ingen råvara (förr alltid 'Guld'); alla läsare tål en tom.
4. Råvaruväljaren på korten ('—' = ingen) med 'Förslag: <råvara> (<källa>) · Använd'.
5. Identitet (Durrett/Wolf Asymmetry): tom råvara visar '—' i stället för uranium, så Uppdatera sparar aldrig uran
   av sig själv. Snabbkollens 'Lägg till i arket' skickar med ins_id och råvaran bara som förslag i Identitet.

Inget skrivs när en sida öppnas; inga grindar, trösklar, formler eller storlekar ändrade.
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import producers                     # noqa: E402
import screens_ui                    # noqa: E402
from confidence import commodities as com   # noqa: E402
from confidence import store as cs   # noqa: E402
from confidence.data.models import CompanyInput   # noqa: E402
from thor import ticker as tk        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. Härledningen (ren) ────────────────────────────────────────────────────
@pytest.mark.parametrize("ticker, branch, name, source", [
    ("EQNR.OL", None, "Olja", "BERSERK-listan"),           # olja + naturgas (TTF) är registrets 'Olja & gas'
    ("CCO.TO", None, "Uran", "EMBER-listan"),
    ("ZZOIL.TO", 2, "Olja", "Börsdata-bransch 2"),
    ("ZZURAN.TO", 7, "Uran", "Börsdata-bransch 7"),
    ("NEM", None, "Guld", "BERSERK-listan"),
    ("AG", None, "Silver", "BERSERK-listan"),
    ("EQT", None, "Gas", "BERSERK-listan"),
])
def test_commodity_for_follows_the_lists_and_the_branch(ticker, branch, name, source):
    assert tk.commodity_for(ticker, branch) == name
    assert tk.commodity_hint(ticker, branch) == (name, source)


@pytest.mark.parametrize("ticker, branch, why", [
    ("ZZ17.TO", 17, ""),                                                   # metaller: för grovt, inget alls
    ("ZZ18.TO", 18, "Börsdata-bransch 18 är både guld och silver"),
    ("BOL.ST", None, "flera råvaror (Koppar · Zink & nickel)"),
    ("LUN.TO", None, "flera råvaror (Koppar · Zink & nickel)"),            # temalistan: samma som Boliden
    ("WPM", None, "flera råvaror (Guld · Silver)"),
    ("OUT1V.HE", None, "flera råvaror (Zink & nickel · Stål)"),
    ("NHY.OL", None, "Aluminium finns inte i arkets lista"),
    ("", 2, ""),
    (None, None, ""),
])
def test_commodity_for_never_guesses(ticker, branch, why):
    assert tk.commodity_for(ticker, branch) is None
    assert tk.commodity_hint(ticker, branch) == (None, why)


def test_a_theme_is_a_commodity_only_when_it_is_one_commodity():
    assert tk.theme_commodity("koppar") == "copper"
    assert tk.theme_commodity("naturgas_eu") == "natural_gas"
    assert tk.theme_commodity("guld") == "gold"
    for many in ("zink_nickel", "litium", "godsel"):                       # zink/nickel, litium/kobolt/grafit, kali/fosfat
        assert tk.theme_commodity(many) is None, many
    for nothing in ("", None, "okänt"):
        assert tk.theme_commodity(nothing) is None


def test_the_registry_key_for_durrett_and_asymmetry_is_not_limited_to_the_rule_list():
    assert tk.commodity_suggestion("NHY.OL") == ("aluminum", "BERSERK-listan")
    assert tk.commodity_suggestion("EQNR.OL") == ("oil_gas", "BERSERK-listan")
    assert tk.commodity_suggestion("BOL.ST", theme="koppar", source="EMBER-temat")[0] is None   # ändå polymetalliskt
    assert tk.commodity_suggestion("ZZCU.V", theme="koppar", source="EMBER-temat") == ("copper", "EMBER-temat")


def test_the_derivation_never_reads_the_sheets(monkeypatch):
    """sheets=None: arkets eget (kanske felaktiga) svar får inte bli förslaget."""
    import valhall.picks as vp
    monkeypatch.setattr(vp, "_commodity_lookup", lambda sheets: {"EQNR.OL": "guld"})
    seen = []
    real = tk.theme_for

    def spy(t, sheets=None, branch_id=None):
        seen.append(sheets)
        return real(t, sheets, branch_id)
    monkeypatch.setattr(tk, "theme_for", spy)
    assert tk.commodity_for("EQNR.OL") == "Olja"
    assert seen == [None]


def test_the_sheet_names_are_the_rotations():
    assert tk.commodity_for("CCO.TO", names=producers.COMMODITIES) == "Uran"
    assert tk.commodity_hint("CCO.TO", names=("Guld", "Koppar")) == (None, "Uran finns inte i arkets lista")


# ── 2. Håvens rad tar med branch_id ──────────────────────────────────────────
def test_row_to_fields_carries_the_branch():
    row = {"ticker": "zzoil.to", "name": "Zz Oil", "ins_id": 1234, "branch_id": 2, "m": {"ev_ebitda": 3.0}}
    for key in ("rule", "sprott", "durrett"):
        out = screens_ui.row_to_fields(row, key)
        assert out["branch_id"] == 2 and out["ins_id"] == 1234 and "commodity" not in out, key
    assert "branch_id" not in screens_ui.row_to_fields({"ticker": "x"}, "rule")


# ── 3. Alla läsare tål en tom råvara ─────────────────────────────────────────
def test_every_reader_tolerates_an_empty_commodity(monkeypatch):
    import commodity_prices
    import controls_ui
    import cycle
    import valhall.picks as vp
    from review import units

    calls = []
    monkeypatch.setattr(commodity_prices, "spot", lambda name: calls.append(name))
    assert commodity_prices.spot_many(["", None]) == {} and calls == []          # natten hämtar inget pris
    assert units.commodity_unit({"commodity": ""}) == ""                          # enhetskontrollen: okänd
    sheets = {"producers": {"producers": [{"ticker": "EQNR.OL", "commodity": ""}]},
              "scoring": {"sprott": [{"ticker": "ZZ.V", "commodity": ""}], "durrett": []}}
    assert vp._commodity_lookup(sheets) == {}                                     # Thor/Valhall: inget arktema
    assert tk.theme_for("EQNR.OL", sheets) == ("olja", "BERSERK-listan")          # … listorna tar över
    assert controls_ui.spot_context({"commodity_price": 4000.0, "commodity_unit": "USD/oz"}, {"commodity": ""}) == ""
    assert cycle.commodity_for_ticker("EQNR.OL", sheets["producers"]) is None    # Copilot: 'Ingen råvara vald'


# ── 3–4. Rick Rule-sidan (AppTest) ───────────────────────────────────────────
SCREENS = {"generated": "2026-11-09T06:00:00", "screens": {k: {"label": k.title(), "rows": [
    {"ticker": "ZZOIL.TO", "name": "Zz Oil", "ins_id": 1234, "branch_id": 2, "universe": "global",
     "mcap_musd": 300.0, "m": {"ev_ebitda": 3.0, "nd_ebitda": 0.5}},
    {"ticker": "ZZAU.V", "name": "Zz Gold", "ins_id": 5678, "branch_id": 18, "universe": "global",
     "mcap_musd": 100.0, "m": {}}]} for k in ("rule", "sprott", "durrett")}}


@pytest.fixture
def sheet(monkeypatch):
    """Rick Rule eller Poängmodellen med fejkad lagring, håv och sifferuppdatering — inget nätverk."""
    import gist_storage
    import refresh_ui
    import storage

    def make(page="Rick Rule", rows=(), which=None):
        bucket = {"Rick Rule": ("producers", "producers"), "Sprott": ("scoring", "sprott"),
                  "Durrett": ("scoring", "durrett")}[page]
        store = ({"producers": [dict(r) for r in rows], "royalty": []} if bucket[0] == "producers"
                 else {"sprott": [], "durrett": [], bucket[1]: [dict(r) for r in rows]})
        stores = {bucket[0]: store}
        monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                            streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
        monkeypatch.setattr(storage, "load_error", lambda name: None)
        monkeypatch.setattr(storage, "is_dirty", lambda name: False)
        monkeypatch.setattr(storage, "last_saved", lambda name: None)
        monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
        monkeypatch.setattr(screens_ui, "load_screens", lambda: SCREENS)
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
        if page != "Rick Rule":
            import scoring
            at.session_state[scoring.WHICH_KEY] = scoring.WHICH[0 if page == "Sprott" else 1]
        at.run()
        assert not at.exception, at.exception
        at.stores, at.bucket = stores, bucket
        return at
    return make


def _rows(at) -> list:
    return at.session_state[at.bucket[0]][at.bucket[1]]


def _row(at, ticker) -> dict:
    return next(r for r in _rows(at) if r["ticker"] == ticker)


def _box(at, prefix):
    hits = [s for s in at.selectbox if str(s.key).startswith(prefix)]
    assert len(hits) == 1, [s.key for s in at.selectbox]
    return hits[0]


def _text(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [e.label for e in at.expander])


def _prefix(at) -> str:
    return "pr" if at.bucket[0] == "producers" else "sc"


@pytest.mark.parametrize("page", ["Rick Rule", "Sprott", "Durrett"])
def test_haven_add_writes_no_commodity_and_the_card_suggests_it(sheet, page):
    at = sheet(page)
    assert _rows(at) == []                                                   # att öppna skriver inget
    kp = "pr_producers" if page == "Rick Rule" else f"sc_{at.bucket[1]}"
    at.button(key=f"scr_add_{kp}_ZZOIL.TO").click().run()
    assert not at.exception, at.exception
    row = _row(at, "ZZOIL.TO")
    assert row["commodity"] == "" and row["branch_id"] == 2 and row["ins_id"] == 1234
    p = _prefix(at)
    box = _box(at, f"{p}_com_{row['id']}_")
    assert box.value == "" and box.options[0] == "—"
    assert "ZZOIL.TO · — ·" in _text(at)
    assert "Förslag: Olja (Börsdata-bransch 2)" in _text(at)
    at.button(key=f"rf_offer_{p}_com_{row['id']}").click().run()                # Använd
    assert not at.exception, at.exception
    assert _row(at, "ZZOIL.TO")["commodity"] == "Olja"
    assert _box(at, f"{p}_com_{row['id']}_").value == "Olja"
    assert f"rf_offer_{p}_com_{row['id']}" not in {b.key for b in at.button}   # använt: förslaget borta
    assert "ZZOIL.TO · Olja ·" in _text(at)


def test_haven_gold_or_silver_branch_gets_no_suggestion(sheet):
    at = sheet("Rick Rule")
    at.button(key="scr_add_pr_producers_ZZAU.V").click().run()
    assert not at.exception, at.exception
    row = _row(at, "ZZAU.V")
    assert row["commodity"] == "" and row["branch_id"] == 18
    assert ("Inget förslag på råvara: Börsdata-bransch 18 är både guld och silver — välj själv."
            in [c.value for c in at.caption])
    assert not [b.key for b in at.button if str(b.key).startswith("rf_offer_pr_com_")]


RULE_ROW = {"id": "p1", "ticker": "EQNR.OL", "name": "Equinor", "commodity": "Guld", "date": "2026-09-01"}


def test_rule_card_suggests_over_a_wrong_commodity_and_writes_nothing_on_open(sheet):
    at = sheet("Rick Rule", rows=[RULE_ROW])
    assert _rows(at) == [RULE_ROW]
    assert _box(at, "pr_com_p1_").value == "Guld"
    assert "Förslag: Olja (BERSERK-listan)" in _text(at)
    assert "rf_offer_pr_com_p1" in {b.key for b in at.button}


def test_rule_card_selector_writes_only_your_change(sheet):
    at = sheet("Rick Rule", rows=[dict(RULE_ROW, commodity="Olja")])
    assert "Förslag:" not in _text(at)                                       # redan rätt: inget förslag
    _box(at, "pr_com_p1_").set_value("Koppar").run()
    assert not at.exception, at.exception
    assert _row(at, "EQNR.OL")["commodity"] == "Koppar"
    assert _box(at, "pr_com_p1_").value == "Koppar"
    _box(at, "pr_com_p1_").set_value("").run()                                 # '—' = ingen
    assert not at.exception, at.exception
    assert _row(at, "EQNR.OL")["commodity"] == ""
    assert "EQNR.OL · — ·" in _text(at)


def test_rule_card_polymetallic_says_why_without_a_button(sheet):
    at = sheet("Rick Rule", rows=[dict(RULE_ROW, ticker="BOL.ST", name="Boliden", commodity="")])
    assert ("Inget förslag på råvara: flera råvaror (Koppar · Zink & nickel) — välj själv."
            in [c.value for c in at.caption])
    assert "rf_offer_pr_com_p1" not in {b.key for b in at.button}
    assert _rows(at)[0]["commodity"] == ""


def test_rule_card_keeps_a_commodity_outside_the_list(sheet):
    at = sheet("Rick Rule", rows=[dict(RULE_ROW, commodity="Nickel")])
    box = _box(at, "pr_com_p1_")
    assert box.value == "Nickel" and "Nickel" in box.options
    assert _rows(at)[0]["commodity"] == "Nickel"


def test_scoring_card_selector_and_suggestion(sheet):
    row = {"id": "s1", "ticker": "CCO.TO", "name": "Cameco", "commodity": "", "date": "2026-09-01", "factors": {}}
    at = sheet("Sprott", rows=[row])
    assert _rows(at) == [row]
    assert "Förslag: Uran (EMBER-listan)" in _text(at)
    at.button(key="rf_offer_sc_com_s1").click().run()
    assert not at.exception, at.exception
    assert _row(at, "CCO.TO")["commodity"] == "Uran"
    _box(at, "sc_com_s1_").set_value("Guld").run()
    assert not at.exception, at.exception
    assert _row(at, "CCO.TO")["commodity"] == "Guld"
    assert "Förslag: Uran (EMBER-listan)" in _text(at)                       # förslaget står kvar, skrivs inte


# ── 5. Identitet: '—' i stället för uranium, förslag med Använd (AppTest) ────
def _asym_app(monkeypatch, conf: dict, quick=None):
    import storage
    import positions
    import refresh_ui
    import commodity_prices
    from asymmetry import quick_data

    stores = {"confidence": conf, "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, stores.get(name) if stores.get(name) is not None
                                                           else default))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: [])
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(commodity_prices, "spot", lambda name: None)
    if quick is not None:
        monkeypatch.setattr(quick_data, "fetch", lambda t, **kw: dict(quick, ticker=t))
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


def _ident(at, ticker, field="commodity"):
    import form_keys as fk
    return fk.ident_box(at.selectbox, f"asym_id_{ticker}", field)


def test_identity_shows_a_dash_and_update_never_saves_uranium(monkeypatch):
    from streamlit.testing.v1 import AppTest
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    conf = _conf(CompanyInput(ticker="ZZ.V", name="Zz Junior"))
    at = AppTest.from_function(_asym_app(monkeypatch, conf), default_timeout=60)
    at.session_state["asym_mode"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    box = _ident(at, "ZZ.V")
    assert box.value == "" and box.options[0] == "—"
    assert at.session_state["confidence"]["companies"]["ZZ.V"]["commodity"] == ""
    at.button(key="FormSubmitter:asym_ident_ZZ.V-Uppdatera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["companies"]["ZZ.V"]["commodity"] == ""      # inte 'uranium'


def test_identity_suggestion_is_written_only_with_use(monkeypatch):
    from streamlit.testing.v1 import AppTest
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    conf = _conf(CompanyInput(ticker="EQNR.OL", name="Equinor", commodity="uranium"))   # tyst sparat förr
    at = AppTest.from_function(_asym_app(monkeypatch, conf), default_timeout=60)
    at.session_state["asym_mode"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    assert "Förslag: Olja & gas (BERSERK-listan)" in [c.value for c in at.caption]
    assert at.session_state["confidence"]["companies"]["EQNR.OL"]["commodity"] == "uranium"
    at.button(key="cf_com_EQNR.OL").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["companies"]["EQNR.OL"]["commodity"] == "oil_gas"
    assert _ident(at, "EQNR.OL").value == "oil_gas"                         # formens ruta ritas om med det sparade
    assert "cf_com_EQNR.OL" not in {b.key for b in at.button}


def test_identity_keeps_a_stored_value_outside_the_registry(monkeypatch):
    from streamlit.testing.v1 import AppTest
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    conf = _conf(CompanyInput(ticker="ZZ.V", name="Zz", commodity="vanadium"))
    at = AppTest.from_function(_asym_app(monkeypatch, conf), default_timeout=60)
    at.session_state["asym_mode"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    assert _ident(at, "ZZ.V").value == "vanadium"
    at.button(key="FormSubmitter:asym_ident_ZZ.V-Uppdatera").click().run()
    assert at.session_state["confidence"]["companies"]["ZZ.V"]["commodity"] == "vanadium"


def test_confidence_page_identity_has_the_same_dash(monkeypatch):
    """Durrett-arkets och Confidence-sidans Identitet är samma rutor (confidence.ui._identity_widgets)."""
    from streamlit.testing.v1 import AppTest
    import storage
    stores = {"confidence": _conf(CompanyInput(ticker="ZZ.V", name="Zz Junior"))}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, stores.get(name, default)))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setenv("ASYM_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["ASYM_TEST_ROOT"])
        from confidence.ui import render_confidence_page
        render_confidence_page()
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["conf_sub"] = "Indata"
    at.run()
    assert not at.exception, at.exception
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import form_keys as fk
    assert fk.ident_box(at.selectbox, "conf_id_ZZ.V", "commodity").value == ""
    at.button(key="FormSubmitter:conf_ident_ZZ.V-Uppdatera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["companies"]["ZZ.V"]["commodity"] == ""


def test_new_company_form_starts_on_the_dash(monkeypatch):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_asym_app(monkeypatch, _conf()), default_timeout=60)
    at.session_state["asym_mode"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="asym_new_commodity").value == ""


# ── 5. Snabbkollen: ins_id följer med, råvaran bara som förslag ─────────────
def test_quick_hint_uses_ember_without_yahoo_and_the_same_uncertainty_rules():
    from asymmetry import quick_ui
    seen = []

    def detect(t, auto=True):
        seen.append((t, auto))
        return {"CCO.TO": "uran", "BOL.ST": "koppar"}.get(t)
    assert quick_ui.sheet_commodity_hint({"yf_ticker": "CCO.TO"}, detect) == ("uranium", "EMBER-temat")
    assert seen == [("CCO.TO", False)]                                       # auto=False: inte Yahoos bransch
    assert quick_ui.sheet_commodity_hint({"yf_ticker": "BOL.ST"}, detect)[0] is None    # polymetalliskt
    assert quick_ui.sheet_commodity_hint({"yf_ticker": "EQNR.OL"}, detect) == ("oil_gas", "BERSERK-listan")
    assert quick_ui.sheet_commodity_hint({}, detect) == (None, "")


QUICK = {"name": "Cameco Corp", "source": "Börsdata", "ins_id": 4321, "yf_ticker": "CCO.TO",
         "prices": [("2026-11-06", 80.0), ("2026-11-07", 81.0)], "fetched": "2026-11-09 06:00 UTC",
         "commodity_px": {"commodity": "uran", "ticker": "", "prices": {}, "p0": None}}


def test_quick_add_writes_ins_id_and_only_suggests_the_commodity(monkeypatch):
    from streamlit.testing.v1 import AppTest
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    at = AppTest.from_function(_asym_app(monkeypatch, _conf(), quick=QUICK), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.text_input(key="asym_quick_ticker").set_value("cco.to")
    at.button(key="FormSubmitter:asym_quick_form-🔍 Analysera").click().run()
    assert not at.exception, at.exception
    at.button(key="asym_quick_add_CCO.TO").click().run()
    assert not at.exception, at.exception
    c = at.session_state["confidence"]["companies"]["CCO.TO"]
    assert c["ins_id"] == 4321 and c["name"] == "Cameco Corp"
    assert c["commodity"] == ""                                              # förslag, inte inskrivet
    assert any("Förslag på råvara: Uran (Använd under Identitet)." in s.value for s in at.success)
    at.session_state["asym_mode"] = "Ark"
    at.session_state["asym_pick"] = "CCO.TO"
    at.run()
    assert not at.exception, at.exception
    assert "Förslag: Uran (EMBER-temat)" in [x.value for x in at.caption]
    assert _ident(at, "CCO.TO").value == ""
    assert at.session_state["confidence"]["companies"]["CCO.TO"]["commodity"] == ""
    at.button(key="cf_com_CCO.TO").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["companies"]["CCO.TO"]["commodity"] == "uranium"
    assert _ident(at, "CCO.TO").value == "uranium"


def test_add_quick_ignores_a_bad_ins_id_and_never_writes_the_commodity():
    from asymmetry import ui as aui
    d = _conf()
    assert aui._add_quick(d, "zz.v", "Zz", ins_id="abc", commodity=None) is True
    assert d["companies"]["ZZ.V"]["ins_id"] is None
    assert aui._add_quick(d, "zz.v", "Zz") is False                          # fanns redan
    assert aui._add_quick(d, "yy.v", "Yy", ins_id=0) is True and d["companies"]["YY.V"]["ins_id"] is None
    assert aui._add_quick(d, "xx.v", "Xx", ins_id=True) is True and d["companies"]["XX.V"]["ins_id"] is None
    assert aui._add_quick(d, "ww.v", "Ww", ins_id=77, commodity=("copper", "EMBER-temat")) is True
    assert d["companies"]["WW.V"]["ins_id"] == 77
    assert all(c["commodity"] == "" for c in d["companies"].values())
    assert com.REGISTRY["copper"].label == "Koppar"
