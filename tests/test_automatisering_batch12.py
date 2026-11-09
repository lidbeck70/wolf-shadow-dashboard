"""
Automatisering, batch 12 — strömbrytarens tal ur fördelningens summa.

1. Under Topp/Nuvarande: '🤖 Fördelningens summa (fälten ändrade <datum>): Nuvarande X SEK · Använd'. Summan är
   fälten under Fördelningsmodellen (som 'Fyll i ur registret' fyller). Använd skriver bara Nuvarande och sparar
   som sidan (sessionen, 💾 till repot).
2. 'Ny topp?' med Använd bara när summan är större än den sparade toppen.
3. En rad om vad summan består av, vilka fält som är tomma och att det som inte står i fälten inte räknas.

Strömbrytarens gränser och uträkning är orörda. Inget skrivs när sidan öppnas. Sidan ritas bara upplåst.
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allocator as al   # noqa: E402
import storage           # noqa: E402
from portfolio import private_lock   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _alloc(values=None, peak=200_000.0, current=180_000.0, asof=""):
    d = al._default()
    d["values"].update(values or {})
    d.update(peak=peak, current=current)
    if asof:
        d["values_asof"] = asof
    return d


# ── rena funktioner ──────────────────────────────────────────────────────────
def test_allocation_total_is_the_sum_of_the_fields_the_page_shows():
    assert al.allocation_total({"royalty": 100_000.0, "kassa": 50_000.0}) == 150_000.0
    # samma golv som procenten: tomt, text, negativt och okända nycklar räknas inte
    assert al.allocation_total({"royalty": None, "swing": "", "insider": -5.0, "kassa": "x",
                                "okänd": 99.0, "durrett": 10.0}) == 10.0
    assert al.allocation_total(None) == 0.0 and al.allocation_total({}) == 0.0


def test_total_parts_names_the_filled_and_the_empty_fields():
    filled, empty = al.total_parts({"royalty": 100_000.0, "kassa": 50_000.0})
    assert filled == [("Royaltykärnan", 100_000.0), ("Kassa", 50_000.0)]
    assert empty == ["Producenter", "Optionalitet", "Durrett", "Momentum-swing", "Insider", "Långsiktigt"]


def test_total_note_says_what_is_in_the_sum_and_what_is_not():
    note = al.total_note({"royalty": 100_000.0, "kassa": 50_000.0})
    assert note.startswith("Summan är fälten under Fördelningsmodellen: Royaltykärnan 100,000 · Kassa 50,000 SEK.")
    assert "Tomma fält: Producenter, Optionalitet, Durrett, Momentum-swing, Insider, Långsiktigt." in note
    assert "Det som inte står i fälten (t.ex. fysiska metaller, fondkonton) räknas inte" in note
    assert "ser ut som ett fall" in note and "Registret" not in note
    assert al.total_note({}) == "" and al.total_note(None) == ""


def test_total_note_names_an_empty_field_the_register_has_money_in():
    reg = {"royalty": 90_000.0, "insider": 26_000.0, "kassa": 0.0}
    note = al.total_note({"royalty": 100_000.0}, reg)
    assert "Registret har belopp i Insider — Fyll i ur registret nedan." in note
    full = {s.key: 1.0 for s in al.SLEEVES}
    assert "Registret" not in al.total_note(full, reg) and "Tomma fält" not in al.total_note(full, reg)


def test_breaker_suggestions_offer_current_and_only_a_higher_top():
    assert al.breaker_suggestions(_alloc()) == {}                                  # inga fält ifyllda
    assert al.breaker_suggestions(None) == {}
    vals = {"royalty": 100_000.0, "kassa": 50_000.0}
    assert al.breaker_suggestions(_alloc(vals)) == {"current": 150_000.0}         # under toppen: ingen ny topp
    assert al.breaker_suggestions(_alloc(vals, current=150_000.0)) == {}          # redan inskrivet
    assert al.breaker_suggestions(_alloc(vals, peak=150_000.0, current=150_000.0)) == {}   # lika med toppen
    assert al.breaker_suggestions(_alloc(vals, peak=120_000.0)) == {"current": 150_000.0, "peak": 150_000.0}
    assert al.breaker_suggestions(_alloc(vals, peak=0.0, current=0.0)) == {"current": 150_000.0, "peak": 150_000.0}
    assert al.breaker_suggestions(_alloc(vals, peak=None, current=None)) == {"current": 150_000.0, "peak": 150_000.0}


def test_the_breaker_rules_are_untouched():
    assert [(lo, hi, label) for lo, hi, label, _c, _a in al.BREAKER_LEVELS] == [
        (0.0, 10.0, "NORMAL"), (10.0, 20.0, "SKÄRPT"), (20.0, float("inf"), "HALVERAD RISK")]
    assert al.drawdown_pct(200_000, 150_000) == 25.0 and al.breaker_state(25.0)[0] == "HALVERAD RISK"


# ── sidan ────────────────────────────────────────────────────────────────────
@pytest.fixture
def page(monkeypatch):
    """Allokeringen i AppTest med ett fejkat lager: inget GitHub, ingen Gist."""
    import refresh_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    saves: list = []
    monkeypatch.setattr(storage, "save_session", lambda name: saves.append(name))

    def make(alloc: dict, holdings=None):
        stores = {"allocator": alloc, "holdings": holdings or {"cash": 0}}
        monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                            stores.setdefault(name, copy.deepcopy(default)))
        from streamlit.testing.v1 import AppTest

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import allocator
            allocator.render_allocator_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=60)
        at.run()
        assert not at.exception, at.exception
        at.stores, at.saves = stores, saves
        return at
    return make


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _caps(at) -> str:
    return " ".join(c.value for c in at.caption)


def _btn(at, key):
    found = [b for b in at.button if b.key == key]
    return found[0] if found else None


def test_page_offers_current_from_the_sum_and_writes_nothing_on_open(page):
    vals = {"royalty": 100_000.0, "kassa": 50_000.0}
    at = page(_alloc(vals, asof="2026-11-08"))
    md = _md(at)
    assert "🤖 Fördelningens summa (fälten ändrade 2026-11-08): Nuvarande 150,000 SEK" in md
    assert "Ny topp?" not in md                                       # 150 000 < toppen 200 000
    assert "Summan är fälten under Fördelningsmodellen: Royaltykärnan 100,000 · Kassa 50,000 SEK." in _caps(at)
    assert "räknas inte" in _caps(at)
    a = at.stores["allocator"]
    assert (a["current"], a["peak"], a["values_asof"]) == (180_000.0, 200_000.0, "2026-11-08")   # orört
    assert at.saves == []
    assert "SKÄRPT" in md                                             # 10 % från toppen, som förut

    _btn(at, "rf_offer_al_cur_total").click().run()
    assert not at.exception, at.exception
    a = at.stores["allocator"]
    assert a["current"] == 150_000.0 and a["peak"] == 200_000.0 and a["values"]["royalty"] == 100_000.0
    assert at.number_input(key="al_cur").value == 150_000.0
    assert at.session_state["allocator"]["current"] == 150_000.0     # sparat som sidan sparar: sessionen
    assert at.saves == []                                            # 💾 sparar till repot, inte Använd
    md = _md(at)
    assert "HALVERAD RISK" in md and "fall från topp 25.0 %" in md    # samma gränser, nytt tal
    assert _btn(at, "rf_offer_al_cur_total") is None                 # förslaget är inskrivet


def test_page_offers_a_new_top_only_when_the_sum_is_higher(page):
    vals = {"royalty": 150_000.0, "swing": 60_000.0, "kassa": 40_000.0}       # 250 000 > toppen 200 000
    at = page(_alloc(vals))
    md = _md(at)
    assert "🤖 Fördelningens summa: Ny topp? 250,000 SEK — större än sparade toppen 200,000" in md
    assert "🤖 Fördelningens summa: Nuvarande 250,000 SEK" in md                # inget datum sparat
    _btn(at, "rf_offer_al_peak_total").click().run()
    assert not at.exception, at.exception
    a = at.stores["allocator"]
    assert a["peak"] == 250_000.0 and a["current"] == 180_000.0               # bara toppen
    assert at.number_input(key="al_peak").value == 250_000.0
    assert _btn(at, "rf_offer_al_peak_total") is None
    assert _btn(at, "rf_offer_al_cur_total") is not None


def test_no_top_saved_says_so(page):
    at = page(_alloc({"kassa": 50_000.0}, peak=0.0, current=0.0))
    assert "Ny topp? 50,000 SEK — ingen topp sparad" in _md(at)


def test_no_suggestion_without_filled_fields_or_when_already_typed(page):
    at = page(_alloc())
    assert "Fördelningens summa" not in _md(at) and "Summan är fälten" not in _caps(at)
    assert _btn(at, "rf_offer_al_cur_total") is None and _btn(at, "rf_offer_al_peak_total") is None
    at = page(_alloc({"kassa": 50_000.0}, peak=100_000.0, current=50_000.0))
    assert _btn(at, "rf_offer_al_cur_total") is None and _btn(at, "rf_offer_al_peak_total") is None
    assert "Summan är fälten under Fördelningsmodellen: Kassa 50,000 SEK." in _caps(at)   # raden står kvar


def test_opening_the_page_adds_no_date_or_other_field(page):
    seed = _alloc({"royalty": 100_000.0, "kassa": 50_000.0})
    before = copy.deepcopy(seed)
    at = page(seed)
    assert "values_asof" not in at.stores["allocator"] and at.stores["allocator"] == before
    assert "🤖 Fördelningens summa: Nuvarande 150,000 SEK" in _md(at)          # inget datum: inget i parentes
    assert "values_asof" not in al._default()


def test_typing_a_field_updates_the_sum_and_its_date_in_the_same_run(page, monkeypatch):
    monkeypatch.setattr(al, "_today", lambda: "2026-11-01")
    at = page(_alloc({"kassa": 50_000.0}, peak=300_000.0, current=50_000.0))
    assert _btn(at, "rf_offer_al_cur_total") is None
    at.number_input(key="al_v_royalty").set_value(100_000.0).run()
    assert not at.exception, at.exception
    assert at.stores["allocator"]["values_asof"] == "2026-11-01"
    assert "🤖 Fördelningens summa (fälten ändrade 2026-11-01): Nuvarande 150,000 SEK" in _md(at)
    assert at.stores["allocator"]["current"] == 50_000.0                      # bara förslag


def test_fill_from_register_stamps_the_date(page, monkeypatch):
    monkeypatch.setattr(al, "_today", lambda: "2026-11-01")
    holdings = {"long": [{"id": "q1", "ticker": "VOLV-B.ST", "strategy": "Quality", "entry_price": 300.0,
                          "shares": 100}], "cash": 20_000}
    at = page(_alloc(peak=0.0, current=0.0), holdings)
    assert "Registret" not in _caps(at)                       # inga fält ifyllda: ingen rad
    [b for b in at.button if b.label == "Fyll i ur registret"][0].click().run()
    assert not at.exception, at.exception
    a = at.stores["allocator"]
    assert a["values"]["langsiktigt"] == 30_000.0 and a["values"]["kassa"] == 20_000.0
    assert a["values_asof"] == "2026-11-01" and a["current"] == 0.0
    assert "🤖 Fördelningens summa (fälten ändrade 2026-11-01): Nuvarande 50,000 SEK" in _md(at)


def test_an_empty_field_with_register_money_is_named(page):
    holdings = {"momentum": [{"id": "m1", "ticker": "VOLV-B.ST", "strategy": "Momentum", "entry_price": 260.0,
                              "shares": 100}], "cash": 0}
    at = page(_alloc({"kassa": 50_000.0}), holdings)
    assert "Registret har belopp i Momentum-swing — Fyll i ur registret nedan." in _caps(at)


def test_a_failing_suggestion_never_takes_the_page_down(page, monkeypatch):
    def boom(_data):
        raise RuntimeError("x")
    monkeypatch.setattr(al, "breaker_suggestions", boom)
    at = page(_alloc({"kassa": 50_000.0}))
    assert not at.error and "FÖRDELNINGSMODELLEN" in _md(at) and "POSITIONSTAK PER BOLAG" in _md(at)
    assert "Fördelningens summa" not in _md(at)


def test_today_is_an_iso_date():
    from datetime import date
    assert date.fromisoformat(al._today())


# ── låset ────────────────────────────────────────────────────────────────────
def test_locked_allocation_shows_no_sum(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import refresh_ui
    seed = {"allocator": _alloc({"royalty": 123_456.0, "kassa": 50_000.0}), "holdings": {"cash": 0}}
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: copy.deepcopy(seed.get(name, default)))
    monkeypatch.setattr(storage, "legacy_gist", lambda *a, **k: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})

    def app(root):
        import sys as _s
        _s.path.insert(0, root)
        import wolf_panel
        wolf_panel._tab_portfolio()

    def texts(at):
        return " ".join(getattr(e, "value", "") or "" for coll in (at.markdown, at.caption, at.info, at.warning)
                        for e in coll)

    at = AppTest.from_function(app, args=(ROOT,), default_timeout=120)
    at.session_state["sub_portfolio"] = "Allokering"
    at.run()
    assert not at.exception, at.exception
    t = texts(at)
    assert "låsta" in t and "Fördelningens summa" not in t and "173,456" not in t and "123,456" not in t

    at = AppTest.from_function(app, args=(ROOT,), default_timeout=120)
    at.session_state["sub_portfolio"] = "Allokering"
    at.session_state[private_lock.STATE] = True
    at.run()
    assert not at.exception, at.exception
    assert "Nuvarande 173,456 SEK" in texts(at)
