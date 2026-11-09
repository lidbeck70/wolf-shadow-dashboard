"""
Batch 2 i automatiseringsplanen — nattfälten syns i arken, bara som förslag och referenser.

- refresh_ui.suggest säger vilket bolag Börsdata matchade när namnet inte liknar radens ('matchat: …').
- Rick Rule-/Royalty-kortet visar Börsdatas FCF (12 mån) i MUSD vid FCF-rutorna — bara text, ingen knapp, och
  aldrig ett tal i en annan valuta.
- DS-förslagen säger vilket års aktieantal de räknar på (shares_year).
Inget skrivs när sidan öppnas; Använd är det enda som skriver.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import refresh_ui as rui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Namnjämförelsen (ren) ────────────────────────────────────────────────────
def test_names_that_resemble_each_other():
    for own, bd in (("Boliden", "Boliden AB"), ("Var Energi", "Vår Energi ASA"), ("Kinross", "Kinross Gold Corp"),
                    ("Franco Nevada", "Franco-Nevada Corp"), ("Hexagon", "Hexagon B"), ("GENO", "Genovis AB"),
                    ("Agnico Eagle Mines", "Agnico Eagle")):
        assert rui.name_resembles(own, bd), (own, bd)


def test_names_that_do_not():
    for own, bd in (("Genovis", "General Oceans"), ("Gold Fields", "Gold Resource Corp"),
                    ("Equinor Energy", "Aker Energy")):
        assert not rui.name_resembles(own, bd), (own, bd)
    assert rui.name_resembles("", "General Oceans") and rui.name_resembles("Genovis", None)   # inget att jämföra


def test_one_letter_names_and_nordic_letters_are_not_called_a_mismatch():
    """Granskningen: 'H&M B' (Börsdatas eget namn, som Insider-raden får) har bara enbokstavsdelar och fick
    '(matchat: H&M B)' mot sig själv; ø/æ delas inte av NFKD ('Maersk' mot 'A.P. Møller - Mærsk B')."""
    import refresh_ui as rui

    def note(own, bd):
        return rui.matched_note({"rows": {"insider:s1": {"bd_name": bd}}}, "insider", {"id": "s1", "name": own})

    for own, bd in (("H&M B", "H&M B"), ("H&M", "H&M B"), ("h&m b", "H&M B"), ("Maersk", "A.P. Møller - Mærsk B"),
                    ("A.P. Moller-Maersk", "A.P. Møller - Mærsk B"), ("Leroy", "Lerøy Seafood Group"),
                    ("Orsted", "Ørsted")):
        assert note(own, bd) == "", (own, bd)
    assert note("B&O", "Bang & Olufsen") == ""                             # initialerna
    assert note("Z", "H&M B") == " (matchat: H&amp;M B)"                  # olika bokstäver: fortfarande en notis
    assert note("H&M B", "Boliden AB") == " (matchat: Boliden AB)"
    # bokstäverna någonstans i namnet räcker inte (bekräftelserundan): andra bolag får notisen
    for own, bd in (("B&O", "Boliden"), ("B&O", "Bouvet"), ("B&O", "Bonava B"), ("H&M", "HMS Networks"),
                    ("H&M", "North Media"), ("H&M", "HM Holdings")):
        assert note(own, bd).startswith(" (matchat: "), (own, bd)


def test_matched_note_only_when_the_name_differs_or_is_missing():
    blob = {"rows": {"insider:s1": {"price": 20.0, "bd_name": "General Oceans"},
                     "insider:s2": {"price": 20.0},
                     "insider:s3": {"price": 1.0, "bd_name": "A&B <Mining> $1"}}}
    assert rui.matched_note(blob, "insider", {"id": "s1", "name": "Genovis"}) == " (matchat: General Oceans)"
    assert rui.matched_note(blob, "insider", {"id": "s1", "name": "General Oceans ASA"}) == ""
    assert rui.matched_note(blob, "insider", {"id": "s1"}) == " (matchat: General Oceans)"     # raden saknar namn
    assert rui.matched_note(blob, "insider", {"id": "s2", "name": "Genovis"}) == ""            # äldre blob
    assert rui.matched_note(blob, "insider", {"id": "x", "name": "Genovis"}, row_id="s1") != ""
    assert rui.matched_note(blob, "insider", {"id": "s3", "name": "Z"}) == " (matchat: A&amp;B &lt;Mining&gt; &#36;1)"
    assert rui.matched_note({}, "insider", {"id": "s1"}) == ""


# ── Förslaget i arket (AppTest) ──────────────────────────────────────────────
_BLOB = {"rows": {"insider:s1": {"price": 20.0, "asof": "2026-11-06", "currency": "NOK",
                                 "bd_name": "General Oceans", "bd_ticker": "GENO"}}}


def _suggest_app():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["B2_ROOT"])
    import streamlit as st
    import refresh_ui
    row = st.session_state.setdefault("row", {"id": "s1", "name": _o.environ.get("B2_NAME", ""), "price_now": 10.0})
    refresh_ui.suggest("insider", row, "price_now", "price", "kurs", None, with_currency=True)


def _run_suggest(monkeypatch, name):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(rui, "load_refresh", lambda: _BLOB)
    monkeypatch.setenv("B2_ROOT", ROOT)
    monkeypatch.setenv("B2_NAME", name)
    at = AppTest.from_function(_suggest_app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_suggest_names_the_matched_company_when_it_differs(monkeypatch):
    at = _run_suggest(monkeypatch, "Genovis")
    html = " ".join(m.value for m in at.markdown)
    assert "kurs 20 NOK (matchat: General Oceans)" in html
    assert at.session_state["row"]["price_now"] == 10.0                  # inget skrivet när sidan öppnas
    at.button(key="rf_insider_s1_price_now").click().run()
    assert at.session_state["row"]["price_now"] == 20.0                  # bara Använd skriver


def test_suggest_is_unchanged_when_the_name_matches(monkeypatch):
    at = _run_suggest(monkeypatch, "General Oceans ASA")
    html = " ".join(m.value for m in at.markdown)
    assert "kurs 20 NOK" in html and "matchat" not in html


# ── Börsdatas FCF vid kortets FCF-rutor (AppTest) ───────────────────────────
def _csm_app():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["B2_ROOT"])
    import streamlit as st
    import controls_ui
    row = st.session_state.setdefault("row", {"id": "p1", "name": "Great Panther"})
    controls_ui.render_csm(row, "p1", with_fv=True, sheet="producers")


def _run_csm(monkeypatch, ref):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(rui, "load_refresh", lambda: {"rows": {"producers:p1": ref}})
    monkeypatch.setenv("B2_ROOT", ROOT)
    at = AppTest.from_function(_csm_app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at, " ".join(m.value for m in at.markdown)


_USD_REF = {"price": 8.0, "asof": "2026-11-07", "currency": "USD", "report_currency": "USD", "source": "borsdata",
            "fcf12_musd": 1234.0, "bd_name": "Great Panther Mining"}


def test_the_card_shows_borsdata_fcf_next_to_both_fcf_inputs(monkeypatch):
    at, html = _run_csm(monkeypatch, _USD_REF)
    line = "🤖 Börsdata FCF 12 mån: 1 234 MUSD (2026-11-07) — referens, inte normaliserad."
    assert html.count(line) == 2                                         # CSM:s FCF-rutor och Lukacs forward FCF
    assert "matchat" not in html
    assert "csm" not in at.session_state["row"] and "fcf12_musd" not in at.session_state["row"]   # bara visning


def test_the_card_line_shows_negative_fcf_and_a_differing_match(monkeypatch):
    _at, html = _run_csm(monkeypatch, dict(_USD_REF, fcf12_musd=-35.0, bd_name="Other Company"))
    assert "🤖 Börsdata FCF 12 mån: −35 MUSD (2026-11-07) (matchat: Other Company) — referens" in html


def test_the_card_never_shows_fcf_in_another_currency(monkeypatch):
    _at, html = _run_csm(monkeypatch, dict(_USD_REF, report_currency="ZAR"))      # jobbet räknade om med 1,0
    assert "Börsdata FCF 12 mån" not in html
    _at, html = _run_csm(monkeypatch, dict(_USD_REF, fcf12_musd=None))
    assert "Börsdata FCF 12 mån" not in html


# ── DS: vilket års aktieantal (AppTest) ─────────────────────────────────────
def _ds_app():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["B2_ROOT"])
    import streamlit as st
    import controls_ui
    row = st.session_state.setdefault("row", {"id": "p1", "name": "Boliden"})
    controls_ui.render_ds(row, "p1", sheet="producers")


def _run_ds(monkeypatch, ref):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(rui, "load_refresh", lambda: {"rows": {"producers:p1": ref}})
    monkeypatch.setenv("B2_ROOT", ROOT)
    at = AppTest.from_function(_ds_app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at, " ".join(c.value for c in at.caption)


def test_ds_suggestions_name_the_year_of_the_share_count(monkeypatch):
    at, caps = _run_ds(monkeypatch, {"asof": "2026-11-07", "shares_growth_3y_pct": 12.0, "shares_growth_5y_pct": 30.0,
                                     "shares_year": 2025})
    assert "antalet aktier i årsrapporten 2025 är +12 % mot 2022" in caps
    assert "antalet aktier i årsrapporten 2025 är +30 % mot 2020" in caps
    assert "ds_aktier_3ar" not in at.session_state["row"]                 # förslag, inte ifyllt


def test_ds_suggestions_without_the_year_read_as_before(monkeypatch):
    _at, caps = _run_ds(monkeypatch, {"asof": "2026-11-07", "shares_growth_3y_pct": 12.0})
    assert "antalet aktier är +12 % mot för tre år sedan" in caps
