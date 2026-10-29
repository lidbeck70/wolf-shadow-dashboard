"""
Panelens siffror: rätt bas, rätt valuta, rätt kapital och svensk tid.

F4  Allokeringens positionstak mäts mot registret inklusive kassan när inga delar är ifyllda (sidan och CSV:n).
F5  Positioner summerar i SEK — varje innehav växlas med registrets kurstabell (London i pence).
F6  Mot målet och 🏠 I dag räknar journalens årstakt på samma startkapital som Journal-sidan.
F8  Journalens Avg R:R och snitt-R per strategi räknar bara affärer som har ett R — '—' när ingen har det.
F11 Metallernas finhalt följer vald metall.
F14 Startsidans datum, rutiner och vändningens dag räknas i svensk tid, inte serverns UTC.
"""
import copy
import os
import re
import sys
from datetime import date, datetime, timezone

import pandas as pd
import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import journal_stats as js   # noqa: E402
import positions             # noqa: E402
import storage               # noqa: E402
import trade_journal as tj   # noqa: E402
from tabs import home_today as ht  # noqa: E402

FX = positions.FX_TO_SEK


def _stores(monkeypatch, seed: dict) -> None:
    """storage.session_load ur en fejkad lagring, ingen GitHub."""
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(seed.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)


def _text(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


# ── F4 · Allokeringens tak mot registret + kassan ────────────────────────────
HUSQ = {"long": [{"id": "h1", "ticker": "HUSQ-B.ST", "strategy": "Quality", "entry_price": 32.53, "shares": 70}],
        "cash": 21000.0}


def test_cap_total_takes_typed_sleeves_first_then_the_register_with_cash(monkeypatch):
    import allocator as a
    _stores(monkeypatch, {"holdings": HUSQ})
    monkeypatch.setattr(streamlit, "session_state", {})
    rows = a.derived_positions({})
    assert a._cap_total({}, rows) == 2277.0 + 21000.0                 # positionerna + kassan
    assert a._cap_total({k: 0.0 for k in a.SLEEVE_BY_KEY}, rows) == 23277.0
    assert a._cap_total({"langsiktigt": 50000.0, "kassa": 50000.0}, rows) == 100000.0   # ifyllt går först
    assert a.position_breaches(rows, a._cap_total({}, rows)) == []     # 9,8 % under taket 10 %


def test_allocator_page_and_csv_measure_the_cap_against_register_cash(monkeypatch):
    """Riktiga registret: HUSQ 70 × 32,53 och 21 000 kr kassa — 9,8 %, inte '100 % — Trimma ner'."""
    from streamlit.testing.v1 import AppTest
    import csv_export
    import refresh_ui
    _stores(monkeypatch, {"holdings": HUSQ})
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    exported: dict = {}
    monkeypatch.setattr(csv_export, "download_button",
                        lambda rows, columns, stem, *a, **k: exported.__setitem__(stem, rows))

    def app():
        import allocator
        allocator.render_allocator_page()

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert not [e.value for e in at.error if "över taket" in e.value]
    assert any("HUSQ-B.ST är 9.8 % — nära taket 10 %" in w.value for w in at.warning)
    assert any("9.8 %" in m.value and "tak" in m.value for m in at.markdown)
    row = exported["positioner"][0]
    assert row["ticker"] == "HUSQ-B.ST" and row["_pct"] == 9.8 and row["_state"] == "nära taket"


# ── F5 · Positioner i SEK ────────────────────────────────────────────────────
QUALITY = {"long": [
    {"id": "q1", "ticker": "EQX", "strategy": "Quality", "entry_price": 8.0, "shares": 1000},
    {"id": "q2", "ticker": "VOLV-B.ST", "strategy": "Quality", "entry_price": 280.0, "shares": 100},
    {"id": "q3", "ticker": "AZN.L", "strategy": "Quality", "entry_price": 11000.0, "shares": 10}],
    "cash": 0}
PRICES = {"EQX": 10.0, "VOLV-B.ST": 300.0, "AZN.L": 12000.0}      # USD · SEK · pence (GBp)


def test_sek_factor_converts_with_the_register_table_and_pence_for_london():
    assert positions.sek_factor("EQX") == FX["USD"] and positions.sek_factor("VOLV-B.ST") == 1.0
    assert positions.sek_factor("AZN.L") == pytest.approx(FX["GBP"] / 100)
    assert positions.sek_factor("AZN.L", "GBX") == pytest.approx(FX["GBP"] / 100)
    assert positions.sek_factor("AU.TO") == FX["CAD"]
    # valuation räknar som förut — nu genom samma faktor
    v = positions.valuation({"id": "l1", "ticker": "RIO.L", "shares": 10, "entry_price": 5000.0}, {})
    assert v["value_sek"] == round(50.0 * 10 * FX["GBP"], 0)


def test_positioner_totals_footer_and_ten_percent_rule_are_in_sek(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import holdings
    _stores(monkeypatch, {"holdings": QUALITY})
    monkeypatch.setattr(holdings, "_fetch_live_price",
                        lambda ticker, period="6mo": {"price": PRICES[ticker], "change_1d": 0.0, "df": None})
    monkeypatch.setattr(holdings, "_get_regime_signal", lambda t, s: None)

    def app():
        import holdings
        holdings._render_summary_header()
        holdings._render_strategy_section("Quality")

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, at.exception
    eqx, volv, azn = 10.0 * 1000 * FX["USD"], 300.0 * 100, 12000.0 / 100 * 10 * FX["GBP"]
    total = eqx + volv + azn
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Invested"] == f"{total:,.0f} SEK"                  # 151 302, inte 160 000 i blandad valuta
    assert metrics["Portfolio Total"] == f"{total:,.0f} SEK"
    text = _text(at)
    assert f"{total:,.0f} SEK" in text                                  # sektionens "Value"
    pnl = 2.0 * 1000 * FX["USD"] + 20.0 * 100 + 1000.0 / 100 * 10 * FX["GBP"]
    assert f"{pnl:+,.0f} SEK" in text                                   # sektionens "P&L"
    warn = [w.value for w in at.warning if ">10% position" in w.value]
    assert warn and f"EQX ({eqx / total * 100:.1f}%)" in warn[0]        # 69,6 % — förut 6,3 % och ej flaggad
    assert f"AZN.L ({azn / total * 100:.1f}%)" in warn[0]               # 10,6 % — förut 75 %
    assert any("12000" in m.value or "12,000" in m.value for m in at.markdown)   # kursen visas i pence som förut


# ── F6 · samma startkapital överallt ─────────────────────────────────────────
TRADES = [{"id": f"t{i}", "ticker": "VOLV-B.ST", "strategy": "wolf", "entry_date": "2025-10-01",
           "exit_date": "2025-11-03", "entry_price": 100.0, "exit_price": 110.0, "shares": 15,
           "pnl_amount": 1500.0, "pnl_pct": 10.0, "r_multiple": None} for i in range(20)]


def test_start_capital_is_the_journals(monkeypatch):
    _stores(monkeypatch, {"settings": {"capital": 300000.0}})
    monkeypatch.setattr(streamlit, "session_state", {})
    assert tj.start_capital() == tj._start_capital() == 300000.0
    monkeypatch.setattr(streamlit, "session_state", {})
    _stores(monkeypatch, {"settings": {"capital": None}})
    assert tj.start_capital() == tj.STARTING_CAPITAL


def test_mot_malet_shows_the_journals_cagr_on_the_settings_capital(monkeypatch):
    from streamlit.testing.v1 import AppTest
    _stores(monkeypatch, {"settings": {"capital": 300000.0}})
    on_capital, on_default = js.realized_cagr(TRADES, 300000.0), js.realized_cagr(TRADES)
    assert on_capital != on_default

    def app():
        from portfolio import mot_malet
        mot_malet.render_mot_malet_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["goal_data"] = {"trades": TRADES, "paper": None}
    at.run()
    assert not at.exception, at.exception
    text = _text(at)
    assert f"{on_capital:+.1f} %" in text and f"{on_default:+.1f} %" not in text
    # samma tal som Journal-sidans KPI med samma inställning
    monkeypatch.setattr(streamlit, "session_state", {})
    assert tj._cagr_of(pd.DataFrame(TRADES)) == on_capital


# ── F8 · snitt-R utan affärer som saknar R ───────────────────────────────────
def test_avg_r_ignores_trades_without_r_and_says_dash_when_none_has_one():
    none_r = pd.DataFrame([{"pnl_pct": 5.0, "pnl_amount": 500.0, "exit_date": "2026-01-02", "strategy": "wolf",
                            "r_multiple": None}] * 3)
    k = tj._compute_kpis(none_r)
    assert k["avg_rr"] is None and tj._fmt_r(k["avg_rr"]) == "—"
    assert tj._strategy_stats(none_r, "wolf")["avg_r"] is None
    mixed = pd.DataFrame([
        {"pnl_pct": 10.0, "pnl_amount": 1000.0, "exit_date": "2026-01-02", "strategy": "wolf", "r_multiple": 2.0},
        {"pnl_pct": -5.0, "pnl_amount": -500.0, "exit_date": "2026-01-03", "strategy": "wolf", "r_multiple": -1.0},
        {"pnl_pct": 4.0, "pnl_amount": 400.0, "exit_date": "2026-01-04", "strategy": "wolf", "r_multiple": None}])
    assert tj._compute_kpis(mixed)["avg_rr"] == 0.5                    # (2 − 1) / 2 — inte (2 − 1 + 0) / 3
    assert tj._strategy_stats(mixed, "wolf")["avg_r"] == 0.5
    assert tj._compute_kpis(pd.DataFrame())["avg_rr"] is None


def test_journal_page_shows_dash_for_avg_r_when_no_trade_has_one(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(tj, "load_journal", lambda: [dict(t) for t in TRADES])

    def app():
        import trade_journal
        trade_journal.render_trade_journal_page()

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, at.exception
    kpi = next(m.value for m in at.markdown if "Avg R:R" in m.value)
    assert re.search(r"margin-top:4px;\">—</div>", kpi), kpi            # förut 0.0 — alla 20 saknar R
    assert "Avg R: <b>—</b>" in _text(at)                               # Wolf-kortet


# ── F11 · finhalten följer metallen ──────────────────────────────────────────
def test_metals_purity_default_follows_the_chosen_metal(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import market_prices
    monkeypatch.setattr(market_prices, "close", lambda s, p="1y": None)          # inget spotpris — inget nät
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: {"rows": []})

    def app():
        from portfolio import metals
        metals.render_metals_section()

    at = AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception
    assert at.number_input(key="metals_purity_guld").value == 0.9999
    at.selectbox(key="metals_metal").select("silver").run()
    assert at.number_input(key="metals_purity_silver").value == 0.999
    at.button(key="metals_add").click().run()
    assert not at.exception
    row = at.session_state["metals"]["rows"][-1]
    assert row["metal"] == "silver" and row["purity"] == 0.999


# ── F14 · svensk tid ─────────────────────────────────────────────────────────
class _UtcKlocka(datetime):
    """Servern (UTC) står på 23:30 den 28 oktober — i Sverige är det redan torsdag den 29:e (00:30, normaltid)."""
    UTC = datetime(2026, 10, 28, 23, 30, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.UTC.astimezone(tz) if tz else cls.UTC.replace(tzinfo=None)


def test_now_se_is_swedish_time_not_server_utc(monkeypatch):
    monkeypatch.setattr(ht, "datetime", _UtcKlocka)
    assert ht.now_se() == datetime(2026, 10, 29, 0, 30)
    # Sommartid: 22:30 UTC den 22 oktober är 00:30 den 23:e — vändningen bekräftad 22:a är på dag 1, inte "i dag"
    monkeypatch.setattr(_UtcKlocka, "UTC", datetime(2026, 10, 22, 22, 30, tzinfo=timezone.utc))
    assert ht.now_se() == datetime(2026, 10, 23, 0, 30)
    board = {"turn_plan": {"window_days": 5, "window": [
        {"level": "komplex", "key": "basmetaller", "label": "Basmetaller", "kind": "VÄNDER", "confirmed": True,
         "confirmed_since": "2026-10-22", "deep": [], "long": [], "etf": None}]}}
    assert ht.turn_headline(ht.turn_cards(board)[0]) == "⚡ VÄNDNING — Basmetaller · dag 1 av 5"
    # hämtad 00:10 svensk tid, klockan 00:30 svensk tid: samma dag, 20 minuter — ingen omläsning
    assert not ht.needs_reload("2026-10-23 00:10")


def test_load_stamps_swedish_time(monkeypatch):
    import positions
    from odin import laget
    from portfolio import mot_malet as mm
    from thor import ui as tui
    monkeypatch.setattr(ht, "datetime", _UtcKlocka)
    monkeypatch.setattr(tui, "load", lambda loader=None: {})                # ingen nätväg: Thor, Odin, journalen
    monkeypatch.setattr(tui, "rebuild", lambda data: {"rows": []})
    monkeypatch.setattr(laget, "load", lambda: {})
    monkeypatch.setattr(positions, "all_positions", lambda: [])
    monkeypatch.setattr(mm, "load", lambda: {"paper": None, "trades": []})
    assert ht.load(loader=lambda name, default=None: None)["loaded"] == "2026-10-29 00:30"


def test_home_date_routines_and_goal_line_use_swedish_time_and_the_journal_capital(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import routines
    monkeypatch.setattr(ht, "datetime", _UtcKlocka)
    _stores(monkeypatch, {"settings": {"capital": 300000.0}})
    days: list = []
    monkeypatch.setattr(routines, "due_today", lambda day: days.append(day) or [])
    monkeypatch.setattr(ht, "load", lambda loader=None: {"trades": TRADES, "paper": None, "loaded": "2026-10-29 00:20"})

    def app():
        from tabs import home
        home.tab_home()

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "torsdag 29 oktober 2026" in text and "onsdag 28 oktober" not in text
    assert days == [date(2026, 10, 29)]
    assert f"journalen {js.realized_cagr(TRADES, 300000.0):+.1f} %" in text      # samma kapital som Journal-sidan
