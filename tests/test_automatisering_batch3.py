"""
Automatisering, batch 3 — Använd på DS-, AQS- och Poängmodellen-förslag som förut bara var text.

1. DS (controls_ui.render_ds): Runway, 'Aktier 3 år' och 'Historik' får Använd, med underlaget (tal, källa, datum,
   årsrapportens år och gränsen förslaget landar i). Burn 0 (positivt FCF 12 mån) blir '> 24 mån → 0 riskpoäng'
   (ägarens beslut). Rick Rule, Tiggre och Poängmodellen (Sprott/Durrett) får knapparna via render_ds.
2. AQS 'Livslängd / resurs' ur Rick Rule-kortets gruvlivslängd (controls.aqs_life_suggestion, fältets tryckta
   gränser), med Använd — och texten säger att talet är reservernas livslängd, inte resursens.
3. Poängmodellen faktor 1: Använd på runwayförslaget.
4. Poängmodellen faktor 4: Använd bara när talet kommer ur Durrett-arket; utdragets tal står som text med källa.

Inget skrivs när en sida öppnas — bara på klicket. Inga grindar, trösklar eller poängformler ändrade.
"""
import copy
import html
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                     # durrett_cases

import controls as ctl         # noqa: E402
import controls_ui as cu        # noqa: E402
import judgment_hints as jh     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASOF = "2026-11-07"


def _md(at) -> str:
    return html.unescape(" ".join(m.value for m in at.markdown))


def _captions(at) -> str:
    return " ".join(c.value for c in at.caption)


def _buttons(at) -> set:
    return {b.key for b in at.button}


# ═════════════════════════════════════════════════════════════════════════════
#  Rena funktioner
# ═════════════════════════════════════════════════════════════════════════════
def test_aqs_life_follows_the_fields_printed_limits():
    f = [x for x in ctl.AQS_FIELDS if x.key == "aqs_livslangd"][0]
    assert (f.zero, f.one, f.two) == ("< 5 år", "5–10 år", "> 10 år")             # gränserna som står i fältet
    assert ctl.aqs_life_suggestion(4.9) == 0
    assert ctl.aqs_life_suggestion(5) == 1 and ctl.aqs_life_suggestion(10.0) == 1
    assert ctl.aqs_life_suggestion(10.5) == 2 and ctl.aqs_life_suggestion("14") == 2
    for blank in (None, "", 0, 0.0, -3, "x"):                                       # 0 = ofylld ruta, som strykregeln
        assert ctl.aqs_life_suggestion(blank) is None, blank


def test_burn_zero_is_more_than_24_months():
    years, note = cu.refresh_runway({"cash_musd": 30.0, "burn_musd": 0.0, "asof": ASOF + "T06:00"})
    assert math.isinf(years) and ctl.ds_runway_suggestion(years) == 0
    assert note == f"Börsdata {ASOF}: positivt FCF 12 mån, burn 0 MUSD/år, kassa 30,0 MUSD"
    years, note = cu.refresh_runway({"burn_musd": 0.0})                             # kassan behövs inte
    assert math.isinf(years) and note == "Börsdata: positivt FCF 12 mån, burn 0 MUSD/år"
    assert cu.refresh_runway({"cash_musd": 30.0, "burn_musd": None}) == (None, "")
    assert cu.refresh_runway({"cash_musd": 12.0, "burn_musd": 8.0})[0] == pytest.approx(1.5)    # som förut


def test_card_runway_shows_the_cards_own_numbers():
    years, note = cu.card_runway(30.0, 12.0)
    assert years == pytest.approx(2.5)
    assert note == "kortets kassa 30,0 MUSD ÷ burn 12,0 MUSD/år = runway 30 mån"
    for cash, burn in ((30.0, 0.0), (None, 5.0), (30.0, None), (-1.0, 5.0)):      # som rutan 'Runway'
        assert cu.card_runway(cash, burn) == (None, "")


def test_share_basis_names_the_report_year_and_warns_for_a_decrease():
    assert (cu._shares_basis((27.3, ASOF, ""), 2025, 3)
            == f"Börsdata {ASOF}: antalet aktier i årsrapporten 2025 är +27 % mot 2022")
    assert (cu._shares_basis((-12.0, ASOF, ""), 2025, 3)
            == f"Börsdata {ASOF}: antalet aktier i årsrapporten 2025 är -12 % mot 2022 — en minskning kan vara "
               f"en omvänd split")
    assert cu._shares_basis((60.0, "", ""), None, 5) == "Börsdata: antalet aktier är +60 % på fem år"


def test_owner_offer_only_for_a_number_from_the_durrett_sheet():
    assert jh.owner_offer({"insider_pct": (12.0, "Durrett-arket")}) == (2, 12.0, "Durrett-arket")
    assert jh.owner_offer({"insider_pct": (7.5, jh.SHEET_SOURCE)}) == (1, 7.5, "Durrett-arket")
    assert jh.owner_offer({"insider_pct": (12.0, "utdrag ur DFS 2025")}) is None    # AI-utdraget: bara text
    assert jh.owner_offer({}) is None and jh.owner_offer(None) is None
    assert jh.owner_basis(12.0) == "insynsägande 12 % (över 10 % → 2 p, 5–10 % → 1 p, under 5 % → 0 p)"
    assert jh.owner_basis(7.5).startswith("insynsägande 7,5 %") and jh.owner_basis(None) == ""


def test_the_durrett_sheet_source_is_the_one_gather_uses():
    from confidence import store as cs
    from confidence.data.provenance import dp
    import durrett_cases as dcs
    data = cs.default()
    c = dcs.gold_producer()
    c.set("insider_ownership_pct", dp(8.0, kind="ACTUAL", source="MD&A"))
    cs.put(data, c)
    g = jh.gather("GPR", data, None)
    assert g["insider_pct"] == (8.0, jh.SHEET_SOURCE)
    assert jh.owner_offer(g) == (1, 8.0, "Durrett-arket")


# ═════════════════════════════════════════════════════════════════════════════
#  DS-sektionen ensam (AppTest)
# ═════════════════════════════════════════════════════════════════════════════
def _ds_app():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["B3_ROOT"])
    import streamlit as st
    import controls_ui
    row = st.session_state.setdefault("row", {"id": "p1", "name": "Boliden"})
    rw, note = controls_ui.sheet_runway("producers", "p1")
    controls_ui.render_ds(row, "p1", rw, sheet="producers", runway_note=note)


def _run_ds(monkeypatch, ref):
    from streamlit.testing.v1 import AppTest
    import refresh_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {"rows": {"producers:p1": ref}})
    monkeypatch.setenv("B3_ROOT", ROOT)
    at = AppTest.from_function(_ds_app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


NIGHT = {"asof": ASOF, "cash_musd": 12.0, "burn_musd": 8.0, "shares_year": 2025,
         "shares_growth_3y_pct": -12.0, "shares_growth_5y_pct": 60.0}
DS_KEYS = {"ds_runway", "ds_capex", "ds_warrants", "ds_aktier_3ar", "ds_historik"}


def test_ds_suggestions_show_their_basis_and_write_nothing(monkeypatch):
    at = _run_ds(monkeypatch, NIGHT)
    md = _md(at)
    assert (f"🤖 DS: Runway 12–24 mån → 1 riskpoäng — Börsdata {ASOF}: kassa 12,0 MUSD ÷ burn 8,0 MUSD/år "
            f"(negativt FCF 12 mån) = runway 18 mån") in md
    assert (f"🤖 DS: Aktier 3 år stabilt → 0 riskpoäng — Börsdata {ASOF}: antalet aktier i årsrapporten 2025 är "
            f"-12 % mot 2022 — en minskning kan vara en omvänd split") in md
    assert (f"🤖 DS: Historik serieutspädare → 2 riskpoäng — Börsdata {ASOF}: antalet aktier i årsrapporten 2025 är "
            f"+60 % mot 2020. Grova gränser på 5 år: +20 % blandad, över +50 % serieutspädare") in md
    assert {"rf_offer_ds_ds_p1_ds_runway", "rf_offer_ds_ds_p1_ds_aktier_3ar",
            "rf_offer_ds_ds_p1_ds_historik"} <= _buttons(at)
    assert not DS_KEYS & set(at.session_state["row"])                              # förslag, inte ifyllt
    at.run()                                                                        # en omritning skriver inte heller
    assert not DS_KEYS & set(at.session_state["row"])
    assert "Förslag, inte ifyllt" not in _captions(at)                             # texten blev en knapp


def test_ds_use_writes_only_that_field_and_its_dropdown(monkeypatch):
    at = _run_ds(monkeypatch, NIGHT)
    at.button(key="rf_offer_ds_ds_p1_ds_runway").click().run()
    assert not at.exception, at.exception
    row = at.session_state["row"]
    assert row["ds_runway"] == 1 and not (DS_KEYS - {"ds_runway"}) & set(row)
    assert at.selectbox(key="ds_p1_ds_runway").value == 1
    assert "rf_offer_ds_ds_p1_ds_runway" not in _buttons(at)                       # använt: förslaget borta
    assert {"rf_offer_ds_ds_p1_ds_aktier_3ar", "rf_offer_ds_ds_p1_ds_historik"} <= _buttons(at)
    at.button(key="rf_offer_ds_ds_p1_ds_historik").click().run()
    assert not at.exception, at.exception
    assert at.session_state["row"]["ds_historik"] == 2 and ctl.ds_total(at.session_state["row"]) == 3
    assert at.selectbox(key="ds_p1_ds_historik").value == 2


def test_ds_burn_zero_is_offered_as_more_than_24_months(monkeypatch):
    at = _run_ds(monkeypatch, {"asof": ASOF, "cash_musd": 30.0, "burn_musd": 0.0})
    assert (f"🤖 DS: Runway > 24 mån → 0 riskpoäng — Börsdata {ASOF}: positivt FCF 12 mån, burn 0 MUSD/år, "
            f"kassa 30,0 MUSD") in _md(at)
    assert "ds_runway" not in at.session_state["row"]
    at.button(key="rf_offer_ds_ds_p1_ds_runway").click().run()
    assert not at.exception, at.exception
    assert at.session_state["row"]["ds_runway"] == 0                               # 0 är ett svar
    assert ctl.ds_total(at.session_state["row"]) == 0


def test_ds_no_suggestion_for_an_answered_field(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import refresh_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {"rows": {"producers:p1": NIGHT}})
    monkeypatch.setenv("B3_ROOT", ROOT)
    at = AppTest.from_function(_ds_app, default_timeout=60)
    at.session_state["row"] = {"id": "p1", "ds_runway": 2, "ds_aktier_3ar": 1}
    at.run()
    assert not at.exception, at.exception
    assert "🤖 DS: Runway" not in _md(at) and "🤖 DS: Aktier 3 år" not in _md(at)
    assert "🤖 DS: Historik" in _md(at)
    assert at.session_state["row"] == {"id": "p1", "ds_runway": 2, "ds_aktier_3ar": 1}


# ═════════════════════════════════════════════════════════════════════════════
#  Arksidorna (AppTest)
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def page(monkeypatch):
    """Arksidan med fejkad sifferuppdatering och lagring — inget nätverk. Sparningar till GitHub spelas in."""
    from streamlit.testing.v1 import AppTest
    import gist_storage
    import lukacs_ui
    import refresh_ui
    import screens_ui
    import storage

    saves: list = []

    def make(name: str, blob: dict, stores: dict, gathered=None):
        monkeypatch.setattr(lukacs_ui, "_live_fx", lambda ccy: None)
        monkeypatch.setattr(gist_storage, "load_blob",
                            lambda n, default=None: blob if n == "sheets_refresh.json" else default)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        monkeypatch.setattr(jh, "gather_for", lambda t: copy.deepcopy(gathered or {}))
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
        return at

    make.saves = saves
    return make


# ── Rick Rule: AQS 'Livslängd / resurs' och DS ───────────────────────────────
RULE = {"id": "p1", "ticker": "EQX.TO", "name": "Equinox Gold", "commodity": "Guld", "date": "2026-09-01",
        "price": 4000.0, "unit_cost": 2000.0, "nd_ebitda": 0.4, "position_pct": 3.0}


def _rule_store(**over):
    return {"producers": {"producers": [dict(RULE, **over)], "royalty": []}}


def _rule_row(at) -> dict:
    return at.session_state["producers"]["producers"][0]


@pytest.mark.parametrize("life, pts, crit", [(8.0, 1, "5–10 år"), (12.0, 2, "> 10 år"), (3.0, 0, "< 5 år")])
def test_rule_aqs_life_is_offered_from_the_cards_mine_life(life, pts, crit, page):
    at = page("Rick Rule", {}, _rule_store(mine_life=life))
    at.run()
    assert not at.exception, at.exception
    yrs = f"{life:g}"
    assert (f"🤖 Kortet: Livslängd / resurs {pts} ({crit}) — ur Gruvlivslängd {yrs} år: reservernas livslängd "
            f"(LOM), inte hela resursen") in _md(at)
    assert at.selectbox(key="aqs_p1_aqs_livslangd").value == "–"
    assert "aqs_livslangd" not in _rule_row(at) and not page.saves                # att öppna skriver inget
    at.button(key="rf_offer_aqs_aqs_p1_aqs_livslangd").click().run()
    assert not at.exception, at.exception
    assert _rule_row(at)["aqs_livslangd"] == pts
    assert at.selectbox(key="aqs_p1_aqs_livslangd").value == pts
    assert "rf_offer_aqs_aqs_p1_aqs_livslangd" not in _buttons(at)
    assert ctl.aqs_total(_rule_row(at)) == pts


@pytest.mark.parametrize("over", [{}, {"mine_life": 0.0}, {"mine_life": 8.0, "aqs_livslangd": 2}])
def test_rule_aqs_life_needs_a_number_and_an_open_question(over, page):
    at = page("Rick Rule", {}, _rule_store(**over))
    at.run()
    assert not at.exception, at.exception
    assert "Livslängd / resurs" not in _md(at)
    assert "rf_offer_aqs_aqs_p1_aqs_livslangd" not in _buttons(at)


def test_rule_ds_runway_use_sets_the_points(page):
    blob = {"rows": {"producers:p1": {"ticker": "EQX.TO", "asof": ASOF, "cash_musd": 30.0, "burn_musd": 40.0}}}
    at = page("Rick Rule", blob, _rule_store())
    at.run()
    assert not at.exception, at.exception
    assert (f"🤖 DS: Runway < 12 mån → 2 riskpoäng — Börsdata {ASOF}: kassa 30,0 MUSD ÷ burn 40,0 MUSD/år "
            f"(negativt FCF 12 mån) = runway 9 mån") in _md(at)
    assert "ds_runway" not in _rule_row(at)
    at.button(key="rf_offer_ds_ds_p1_ds_runway").click().run()
    assert not at.exception, at.exception
    assert _rule_row(at)["ds_runway"] == 2 and at.selectbox(key="ds_p1_ds_runway").value == 2
    assert not page.saves                                                          # 💾 sparar, som förut


# ── Tiggre: DS ───────────────────────────────────────────────────────────────
def test_tiggre_ds_runway_use_writes_only_on_the_click(page):
    stores = {"tiggre": {"candidates": [{"id": "c1", "ticker": "TGR", "screen": {}, "downside": -40.0,
                                         "factors": {}, "catalysts": []}],
                         "positions": [], "closed": [], "parked": []},
              "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [],
                           "cash": 0}}
    blob = {"rows": {"tiggre:c1": {"ticker": "TGR", "asof": ASOF, "cash_musd": 12.0, "burn_musd": 0.0}}}
    at = page("Tiggre", blob, stores)
    at.run()
    assert not at.exception, at.exception
    assert f"🤖 DS: Runway > 24 mån → 0 riskpoäng — Börsdata {ASOF}: positivt FCF 12 mån" in _md(at)
    assert at.session_state["tiggre"] == stores["tiggre"]                          # inget ifyllt vid öppning
    at.button(key="rf_offer_ds_ds_c1_ds_runway").click().run()
    assert not at.exception, at.exception
    assert at.session_state["tiggre"]["candidates"][0]["ds_runway"] == 0
    assert not page.saves


# ── Poängmodellen: faktor 1, faktor 4 och DS ur kortet ───────────────────────
def _scoring_store(bucket="sprott", **row):
    r = {"id": "q1", "ticker": "AAA", "commodity": "Guld", "factors": {}, **row}
    return {"scoring": {"sprott": [], "durrett": [], bucket: [r]}}


def _scoring_row(at, bucket="sprott") -> dict:
    return at.session_state["scoring"][bucket][0]


def _open(page, stores, bucket="sprott", blob=None, gathered=None):
    import scoring
    at = page("Poängmodell", blob or {}, stores, gathered=gathered)
    at.session_state[scoring.WHICH_KEY] = scoring.WHICH_FOR[bucket]
    at.run()
    assert not at.exception, at.exception
    return at


def test_factor_one_offers_the_runway_points_with_its_numbers(page):
    at = _open(page, _scoring_store(cash=30.0, burn=12.0))
    md = _md(at)
    assert ("🤖 Kortet: Balansräkning & runway 2 p (Nettokassa, runway > 24 mån) — kortets kassa 30,0 MUSD ÷ "
            "burn 12,0 MUSD/år = runway 30 mån. Nettokassa och emissionshistorik kan flytta den.") in md
    # DS-runwayn ur samma kort, med samma underlag
    assert ("🤖 DS: Runway > 24 mån → 0 riskpoäng — kortets kassa 30,0 MUSD ÷ burn 12,0 MUSD/år = runway 30 mån"
            in md)
    assert _scoring_row(at)["factors"] == {} and "ds_runway" not in _scoring_row(at)
    assert not page.saves
    at.button(key="rf_offer_sc_f_q1_balans").click().run()
    assert not at.exception, at.exception
    assert _scoring_row(at)["factors"] == {"balans": 2}
    assert at.selectbox(key="sc_f_q1_balans").value == 2
    assert "rf_offer_sc_f_q1_balans" not in _buttons(at)
    assert "ds_runway" not in _scoring_row(at)                                     # bara den faktorn
    assert not page.saves                                                          # 💾 sparar, som förut


def test_factor_one_set_by_you_keeps_the_text_and_gets_no_button(page):
    at = _open(page, _scoring_store(cash=6.0, burn=12.0, factors={"balans": 2}))
    assert "Runwayen motsvarar 0 p på faktor 1" in _captions(at)
    assert "rf_offer_sc_f_q1_balans" not in _buttons(at)
    assert _scoring_row(at)["factors"] == {"balans": 2}


@pytest.mark.parametrize("bucket", ["sprott", "durrett"])
def test_factor_four_from_the_durrett_sheet_gets_use(bucket, page):
    at = _open(page, _scoring_store(bucket), bucket, gathered={"insider_pct": (12.0, "Durrett-arket")})
    assert ("🤖 Durrett-arket: Ägare & management 2 p (Insyn > 10 % eller Sprott/Van Eck bland ägarna) — "
            "insynsägande 12 % (över 10 % → 2 p, 5–10 % → 1 p, under 5 % → 0 p)") in _md(at)
    assert _scoring_row(at, bucket)["factors"] == {}
    at.button(key="rf_offer_sc_f_q1_agare").click().run()
    assert not at.exception, at.exception
    assert _scoring_row(at, bucket)["factors"] == {"agare": 2}
    assert at.selectbox(key="sc_f_q1_agare").value == 2
    assert "rf_offer_sc_f_q1_agare" not in _buttons(at)


def test_factor_four_from_the_ai_extract_stays_text_with_its_source(page):
    at = _open(page, _scoring_store(), gathered={"insider_pct": (12.0, "utdrag ur DFS 2025")})
    assert ("Förslag: Ägare & management 2 p — insynsägande 12 % (utdrag ur DFS 2025). Kryssa själv."
            in _captions(at))
    assert "rf_offer_sc_f_q1_agare" not in _buttons(at)
    assert "🤖 Durrett-arket" not in _md(at)


def test_factor_four_set_by_you_is_never_overwritten_from_here(page):
    at = _open(page, _scoring_store(factors={"agare": 0}), gathered={"insider_pct": (12.0, "Durrett-arket")})
    assert "Förslag: Ägare & management 2 p — insynsägande 12 % (Durrett-arket)" in _captions(at)
    assert "rf_offer_sc_f_q1_agare" not in _buttons(at)
    assert _scoring_row(at)["factors"] == {"agare": 0}


def test_scoring_ds_share_history_gets_use(page):
    blob = {"rows": {"scoring:q1": {"asof": ASOF, "shares_growth_3y_pct": 27.3, "shares_year": 2025}}}
    at = _open(page, _scoring_store("durrett"), "durrett", blob=blob)
    assert (f"🤖 DS: Aktier 3 år > +25 % → 2 riskpoäng — Börsdata {ASOF}: antalet aktier i årsrapporten 2025 är "
            f"+27 % mot 2022") in _md(at)
    at.button(key="rf_offer_ds_ds_q1_ds_aktier_3ar").click().run()
    assert not at.exception, at.exception
    assert _scoring_row(at, "durrett")["ds_aktier_3ar"] == 2
    assert at.selectbox(key="ds_q1_ds_aktier_3ar").value == 2
