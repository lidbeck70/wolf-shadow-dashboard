"""
Automatisering, batch 16 — positionerna: stopp, mål och mäklarens tal.

1. Holdings redigeringsruta: Stopp och Mål på positionen (journalen räknar R ur stoppen), med förslag per strategi ur
   regler som redan finns — Insider (klustersnittet × 0,85), Momentum (−10 %), Wolf (entry − 2,5 × ATR14), EMA200-
   reglerna som 'nivå i dag X (datum)', arkstrategierna 'ingen kursstopp'.
2. Mäklarimporten: FIFO-snitt och antal för rader som redan finns, ett konto i taget, med Använd.
3. Sektor i Holdings: Börsdatas sektor på tryck, med Använd.
4. Swing: 'Ur topp 40' och 'Stängt under MA50' som förslag ur screenern (och kurslagret) — kryssas bara med Använd.

Inget skrivs när en sida öppnas; ingen grind, tröskel, formel eller storlek ändras.
"""
import copy
import logging
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from journal_import.fifo import OpenPosition   # noqa: E402
from portfolio import position_hints as ph     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = date.today()


def _ohlc(n: int = 260, start: float = 100.0, end: float = 130.0, end_day: str = "2026-11-06") -> pd.DataFrame:
    idx = pd.bdate_range(end=end_day, periods=n)
    close = pd.Series(np.linspace(start, end, n), index=idx)
    return pd.DataFrame({"Open": close, "High": close + 2.0, "Low": close - 2.0, "Close": close,
                         "Volume": 1000.0})


# ── 1. Stopp per strategi (rena funktioner) ──────────────────────────────────
def test_momentum_stop_is_entry_less_the_rules_ten_percent():
    import swing_verdict as sv
    h = ph.stop_hint({"strategy": "Momentum", "entry_price": 250.0})
    assert h["value"] == pytest.approx(250.0 * (1 + sv.STOP_PCT / 100)) == pytest.approx(225.0)
    assert h["source"] == "Momentum-regeln" and h["unit"] == "" and "−10 %" in h["basis"]
    assert ph.stop_hint({"strategy": "Momentum", "entry_price": 0}) is None          # ingen entry, inget förslag


def test_wolf_stop_uses_wolfs_own_atr_and_multiplier_and_is_dated():
    from strategies import wolf
    df = _ohlc()
    h = ph.stop_hint({"strategy": "Wolf", "entry_price": 120.0}, df)
    atr = float(wolf._atr(df, wolf.DEFAULT_PARAMS["atr_period"]).iloc[-1])
    assert h["value"] == pytest.approx(round(120.0 - wolf.DEFAULT_PARAMS["atr_mult"] * atr, 2))
    assert h["unit"] == "(2026-11-06)" and h["basis"] == "2,5 × ATR14 2026-11-06"
    assert ph.stop_hint({"strategy": "Wolf", "entry_price": 120.0}, None) is None      # utan kurser: inget
    assert ph.stop_hint({"strategy": "Wolf", "entry_price": 120.0}, df.head(10)) is None
    assert ph.stop_hint({"strategy": "Wolf", "entry_price": 3.0}, df) is None          # stopp under noll


@pytest.mark.parametrize("strategy", ["Quality", "Alpha"])
def test_ema200_rules_suggest_todays_level_with_its_date(strategy):
    df = _ohlc()
    h = ph.stop_hint({"strategy": strategy, "entry_price": 120.0}, df)
    level = float(df["Close"].ewm(span=200).mean().iloc[-1])          # samma EMA200 som signalkolumnen
    assert h["value"] == pytest.approx(round(level, 2))
    assert h["label"] == "nivå i dag" and h["unit"] == "(2026-11-06)" and h["basis"] == "EMA200 2026-11-06"
    assert ph.stop_hint({"strategy": strategy, "entry_price": 120.0}, df.tail(150)) is None    # < 200 staplar


def test_insider_stop_is_the_sheets_stop_for_the_latest_cluster():
    import insider
    rows = [{"ticker": "BOL", "cluster_avg": 100.0, "found": "2026-10-01"},
            {"ticker": "BOL.ST", "cluster_avg": 110.0, "found": "2026-11-01"},
            {"ticker": "SAND", "cluster_avg": 50.0, "found": "2026-11-05"}]
    h = ph.stop_hint({"strategy": "Insider", "ticker": "BOL.ST", "entry_price": 120.0}, None, rows)
    assert h["value"] == pytest.approx(insider.stop_price(rows[1])) == pytest.approx(93.5)
    assert "klustersnittet 110 × 0,85" in h["note"] and "2026-11-01" in h["note"]
    assert ph.stop_hint({"strategy": "Insider", "ticker": "ERIC-B.ST"}, None, rows) is None
    assert ph.stop_hint({"strategy": "Insider", "ticker": "BOL.ST"}, None, None) is None


@pytest.mark.parametrize("strategy", ["Rule", "Royalty", "Tiggre", "Durrett", "Sprott", "Deep Contrarian"])
def test_sheet_strategies_get_the_no_price_stop_text(strategy):
    h = ph.stop_hint({"strategy": strategy, "entry_price": 10.0}, _ohlc())
    assert h["value"] is None and h["text"].startswith("Ingen kursstopp att föreslå.")


@pytest.mark.parametrize("strategy", ["Viking", "Viking Nine", "BERSERK", "Ember", "Untagged"])
def test_strategies_without_a_decided_stop_get_nothing(strategy):
    assert ph.stop_hint({"strategy": strategy, "entry_price": 10.0}, _ohlc()) is None


def test_update_by_row_id_hits_the_right_row_when_a_ticker_stands_twice(monkeypatch):
    """Två rader med samma ticker i en hink (Swing-köp på nytt id): Använd skriver den rad förslaget gällde."""
    import streamlit
    import positions
    import storage
    state: dict = {}
    seed = {"holdings": {"momentum": [{"id": "a", "ticker": "VOLV-B.ST", "strategy": "Momentum", "entry_price": 250},
                                      {"id": "b", "ticker": "VOLV-B.ST", "strategy": "Momentum", "entry_price": 260}]}}
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        state.setdefault(name, copy.deepcopy(seed.get(name, default))))
    monkeypatch.setattr(storage, "save_session", lambda name: storage.SaveResult(name, "abc1234", "u", "t"))
    assert positions.update("VOLV-B.ST", "momentum", row_id="b", stop=234.0)
    rows = {r["id"]: r for r in state["holdings"]["momentum"]}
    assert rows["b"]["stop"] == 234.0 and rows["a"]["stop"] is None
    assert positions.update("VOLV-B.ST", "momentum", target=300.0)          # utan id: som förut, första raden
    assert {r["id"]: r for r in state["holdings"]["momentum"]}["a"]["target"] == 300.0
    assert not positions.update("VOLV-B.ST", "momentum", row_id="zz", stop=1.0)


def test_the_journal_reads_r_from_a_stop_set_on_the_position():
    """Stoppen på positionen är den journal_bridge räknar R ur — ingen ändring i bryggan behövs."""
    import journal_bridge
    closed = {"id": "p1", "ticker": "SAND.ST", "strategy": "Wolf", "entry_price": 100.0, "exit_price": 120.0,
              "shares": 10, "stop": 90.0, "entry_date": "2026-10-01", "exit_date": "2026-11-01"}
    assert journal_bridge.trade_from_closed(closed)["r_multiple"] == 2.0
    assert journal_bridge.trade_from_closed({**closed, "stop": None})["r_multiple"] is None


# ── 3. Sektor (ren funktion) ─────────────────────────────────────────────────
def test_sector_hint_names_the_matched_company_and_warns_on_a_mismatch():
    good = ph.sector_hint(5, {5: "Material"}, "Boliden", "Boliden AB")
    assert good == {"value": "Material", "note": "Börsdata: Boliden"}
    bad = ph.sector_hint("5", {5: "Material"}, "Bolagret", "Volvo")
    assert bad["value"] == "Material" and "liknar inte 'Volvo'" in bad["note"]
    assert ph.sector_hint(None, {5: "Material"}) is None and ph.sector_hint(6, {5: "Material"}) is None


# ── 2. Mäklarens tal (rena funktioner) ───────────────────────────────────────
def _op(account="nordnet_isk", isin="SE0000107419", ticker="VOLV-B.ST", name="Volvo B", qty=100.0, avg=250.5,
        ccy="SEK"):
    return OpenPosition(account=account, isin=isin, ticker=ticker, name=name, earliest_entry_date="2026-01-02",
                        qty=qty, avg_price=avg, currency=ccy)


@pytest.mark.parametrize("ticker, avg, ccy, value, note", [
    ("VOLV-B.ST", 250.5, "SEK", 250.5, ""),
    ("EQNR.OL", 280.0, "NOK", 280.0, ""),
    ("AAPL", 150.0, "SEK", None, "AAPL noteras i USD, inte föreslagen"),      # Nordnets fil: kontots valuta
    ("RIO.L", 5020.0, "GBX", 5020.0, ""),
    ("RIO.L", 5020.0, "GBp", 5020.0, ""),
    ("RIO.L", 50.2, "GBP", 5020.0, "50,2 GBP = 5020 pence"),
    ("RIO.L", 5020.0, "SEK", None, "London noteras i pence"),
])
def test_broker_price_only_in_the_rows_own_unit(ticker, avg, ccy, value, note):
    got, text = ph.broker_price(ticker, avg, ccy)
    assert got == (pytest.approx(value) if value is not None else None)
    assert (note in text) if note else text == ""


def test_broker_hints_keep_accounts_apart_and_show_the_matched_name():
    rows = [{"id": "r1", "ticker": "VOLV-B.ST", "name": "Volvo", "strategy": "Quality", "_bucket": "long"},
            {"id": "r2", "ticker": "SAND.ST", "name": "Sandvik", "strategy": "Wolf", "_bucket": "swing"}]
    ops = [_op(), _op(account="nordea_ask", qty=40.0, avg=260.0),
           _op(isin="SE0000148884", ticker="SAND.ST", name="Boliden", qty=12.5, avg=200.0),
           _op(isin="X", ticker="ERIC-B.ST", name="Ericsson B")]          # inte i registret: inget förslag
    hints = ph.broker_hints(rows, ops)
    volvo = [h for h in hints if h["row"]["id"] == "r1"]
    got = [(h["account"], h["qty"], h["price"]) for h in volvo]
    assert got == [("nordnet_isk", 100, 250.5), ("nordea_ask", 40, 260.0)]
    assert all(h["accounts"] == 2 for h in volvo)                      # två konton, aldrig summerade
    sand = next(h for h in hints if h["row"]["id"] == "r2")
    assert sand["qty"] == 12.5 and sand["accounts"] == 1
    assert "liknar inte 'Sandvik'" in sand["warn"]                       # ISIN-kopplingen syns
    assert not [h for h in hints if h["row"]["ticker"] == "ERIC-B.ST"]


# ── 4. Swing (rena funktioner) ───────────────────────────────────────────────
def _screener(age_days: int = 1, top=None) -> dict:
    top = top if top is not None else [
        {"ticker": "VOLV B", "rank": 3, "dist_ma50": 0.04},
        {"ticker": "ERIC B", "rank": 12, "dist_ma50": -0.021}]
    return {"generated": (TODAY - timedelta(days=age_days)).isoformat() + "T06:00", "top": top}


def test_rank_hint_only_for_tickers_outside_a_fresh_top_40():
    scr = _screener()
    assert ph.rank_hint("VOLV-B.ST", scr) is None
    h = ph.rank_hint("SAND.ST", scr)
    assert h["note"] == "finns inte bland de 40 i listan" and h["asof"] == scr["generated"][:10]
    assert ph.rank_hint("SAND.ST", _screener(age_days=30)) is None      # gammal lista föreslår inget
    assert ph.rank_hint("SAND.ST", {"generated": TODAY.isoformat(), "top": []}) is None
    assert ph.rank_hint("SAND.ST", None) is None
    assert ph.rank_hint("VOLV-B.ST", _screener(top=[{"ticker": "VOLV B", "rank": 41}]))["note"] == "rank 41"


def test_ma50_from_the_screener_and_from_closes():
    scr = _screener()
    assert ph.ma50_from_screener("VOLV-B.ST", scr) == (None, True)          # över MA50: svaret är nej
    h, found = ph.ma50_from_screener("ERIC-B.ST", scr)
    assert found and h["note"] == "kurs −2,1 % mot MA50"
    assert ph.ma50_from_screener("SAND.ST", scr) == (None, False)          # utanför listan: kurslagret
    assert ph.ma50_from_screener("ERIC-B.ST", _screener(age_days=30)) == (None, False)
    falling = _ohlc(start=130.0, end=100.0)["Close"]
    h = ph.ma50_from_closes(falling)
    assert h["asof"] == "2026-11-06" and h["note"].startswith("stängning 100 under MA50")
    assert ph.ma50_from_closes(_ohlc()["Close"]) is None                     # stigande: över MA50
    assert ph.ma50_from_closes(falling.tail(20)) is None and ph.ma50_from_closes(pd.Series(dtype=float)) is None


# ── sidorna (AppTest) ────────────────────────────────────────────────────────
@pytest.fixture
def stores(monkeypatch):
    """Lagringen fejkad: storage.load_json ur en dict, sparningar spelas in. Inget nätverk."""
    import gist_storage
    import refresh_ui
    import storage

    saves: list = []
    data: dict = {}
    monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(data.get(n, default)))
    monkeypatch.setattr(storage, "legacy_gist", lambda f: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda n, default=None: default)

    def _save(n):
        import streamlit as st
        saves.append((n, copy.deepcopy(st.session_state[n])))
        return storage.SaveResult(n, "abc1234def", "https://x", "2026-11-08T10:00:00Z")
    monkeypatch.setattr(storage, "save_session", _save)
    data["_saves"] = saves
    return data


def _reg(**buckets) -> dict:
    base = {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0}
    base.update(buckets)
    return base


def _row(rid, ticker, strategy, entry, **kw):
    return {"id": rid, "ticker": ticker, "name": kw.pop("name", ""), "strategy": strategy, "entry_price": entry,
            "shares": kw.pop("shares", 10), "entry_date": "2026-10-01", "sector": kw.pop("sector", "Unknown"),
            **kw}


def _saved_rows(stores) -> dict:
    out = {}
    for n, s in stores["_saves"]:
        if n == "holdings":
            for b in ("swing", "long", "momentum", "tiggre", "ovtlyr"):
                for r in s[b]:
                    out[r["id"]] = r
    return out


# 1 och 3. Holdings
@pytest.fixture
def holdings_page(monkeypatch, stores):
    from streamlit.testing.v1 import AppTest
    import earnings_calendar
    import holdings
    import risk_dashboard

    df = _ohlc()
    monkeypatch.setattr(holdings, "_fetch_live_price", lambda t, period="6mo": {"price": 130.0, "change_1d": 0.0,
                                                                                "df": df})
    monkeypatch.setattr(holdings, "_get_regime_signal", lambda t, s: None)
    monkeypatch.setattr(holdings, "_fetch_returns_for_correlation", lambda t, period="6mo": pd.DataFrame())
    monkeypatch.setattr(risk_dashboard, "render_risk_dashboard", lambda d: None)
    monkeypatch.setattr(earnings_calendar, "render_earnings_calendar", lambda d: None)

    def make():
        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import holdings as h
            h.render_holdings_page()
        return AppTest.from_function(app, args=(ROOT,), default_timeout=90)
    make.df = df
    return make


def _captions(at) -> str:
    return " ".join(c.value for c in at.caption)


def _markdown(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_holdings_stop_suggestions_per_strategy_save_only_on_click(holdings_page, stores):
    from strategies import wolf
    stores["holdings"] = _reg(
        swing=[_row("w1", "SAND.ST", "Wolf", 120.0)],
        long=[_row("q1", "VOLV-B.ST", "Quality", 110.0), _row("i1", "BOL.ST", "Insider", 120.0)],
        momentum=[_row("m1", "ERIC-B.ST", "Momentum", 250.0)],
        tiggre=[_row("t1", "LOB.TO", "Tiggre", 1.2)])
    stores["insider"] = {"signals": [{"ticker": "BOL", "cluster_avg": 110.0, "found": "2026-11-01"}]}
    at = holdings_page()
    at.run()
    assert not at.exception, at.exception
    assert not [w for w in at.warning if "Error displaying" in w.value]
    md, caps = _markdown(at), _captions(at)
    atr = float(wolf._atr(holdings_page.df, 14).iloc[-1])
    wolf_stop = round(120.0 - 2.5 * atr, 2)
    assert f"🤖 Wolf-regeln: stopp {wolf_stop:g} (2026-11-06)".replace(".", ",") in md
    assert "🤖 EMA200-regeln: nivå i dag" in md and "(2026-11-06) — under EMA200 halveras positionen" in md
    assert "🤖 Momentum-regeln: stopp 225 — entry 250 − 10 %" in md
    assert "🤖 Insiderbevakaren: stopp 93,5 — klustersnittet 110 × 0,85" in md
    assert "Ingen kursstopp att föreslå. Regeln: Ingen kursstop" in caps
    assert not stores["_saves"]                                               # att öppna sparar inget

    at.button(key="rf_offer_hold_stop_swing_SAND.ST").click().run()
    assert not at.exception, at.exception
    saved = _saved_rows(stores)
    assert len(stores["_saves"]) == 1 and saved["w1"]["stop"] == pytest.approx(wolf_stop)
    assert saved["w1"]["extras"]["stop_basis"] == "2,5 × ATR14 2026-11-06"   # regeln och dagen följer med
    assert saved["m1"]["stop"] is None and saved["q1"]["stop"] is None        # bara den position du tryckte på
    assert at.number_input(key="edit_stop_swing_SAND.ST").value == pytest.approx(wolf_stop)
    assert not [b for b in at.button if b.key == "rf_offer_hold_stop_swing_SAND.ST"]   # samma tal: döljs
    assert "Stoppen sattes ur 2,5 × ATR14 2026-11-06." in _captions(at)


def test_holdings_save_writes_stop_target_and_sector_by_hand(holdings_page, stores):
    stores["holdings"] = _reg(momentum=[_row("m1", "ERIC-B.ST", "Momentum", 250.0)])
    at = holdings_page()
    at.run()
    assert not at.exception, at.exception
    at.number_input(key="edit_stop_momentum_ERIC-B.ST").set_value(230.0)
    at.number_input(key="edit_target_momentum_ERIC-B.ST").set_value(300.0)
    at.text_input(key="edit_sec_momentum_ERIC-B.ST").set_value("Informationsteknik")
    at.run()
    assert not stores["_saves"]                                               # inget utan Save
    at.button(key="save_edit_momentum_ERIC-B.ST").click().run()
    assert not at.exception, at.exception
    row = _saved_rows(stores)["m1"]
    assert (row["stop"], row["target"], row["sector"]) == (230.0, 300.0, "Informationsteknik")
    assert row["extras"].get("stop_basis") == ""                             # egen stopp: ingen regel bakom


def test_holdings_sector_is_fetched_on_click_and_saved_with_use(holdings_page, stores, monkeypatch):
    import holdings
    calls = []

    def fake(ticker, own=""):
        calls.append(ticker)
        return {"value": "Industri", "note": "Börsdata: Sandvik"}
    monkeypatch.setattr(holdings, "bd_sector", fake)
    stores["holdings"] = _reg(swing=[_row("w1", "SAND.ST", "Wolf", 120.0, name="Sandvik")])
    at = holdings_page()
    at.run()
    assert not at.exception, at.exception
    assert not calls and not [b for b in at.button if b.key == "rf_offer_hold_sec_w1"]   # inget hämtat vid öppning
    at.button(key="hold_sec_w1_get").click().run()
    assert calls == ["SAND.ST"] and not stores["_saves"]
    assert "🤖 Börsdata: sektor Industri — Börsdata: Sandvik" in _markdown(at)
    at.button(key="rf_offer_hold_sec_w1").click().run()
    assert not at.exception, at.exception
    assert _saved_rows(stores)["w1"]["sector"] == "Industri"
    assert at.text_input(key="edit_sec_swing_SAND.ST").value == "Industri"


def test_bd_sector_without_a_key_says_so(monkeypatch):
    import borsdata_api
    import holdings

    class _Api:
        is_configured = False
    monkeypatch.setattr(borsdata_api, "get_api", lambda: _Api())
    assert "nyckeln saknas" in holdings.bd_sector("SAND.ST")["msg"]


def test_bd_sector_from_the_registry_member(monkeypatch):
    import borsdata_api
    import holdings
    import refresh_ui
    from universe import registry as ur

    class _Api:
        is_configured = True

        def get_sectors(self):
            return [{"id": 3, "name": "Industri"}, {"id": 5, "name": "Material"}]
    monkeypatch.setattr(borsdata_api, "get_api", lambda: _Api())
    monkeypatch.setattr(refresh_ui, "_registry_member",
                        lambda t: ur.Member("SAND.ST", "Sandvik", ur.NORDEN, sector_id=3))
    assert holdings.bd_sector("SAND.ST", "Sandvik AB") == {"value": "Industri", "note": "Börsdata: Sandvik"}


# 2. Mäklarimporten
@pytest.fixture
def broker_page(stores):
    from streamlit.testing.v1 import AppTest

    def make(ops: list, unlocked: bool = True):
        def app(root, ops_):
            import sys as _s
            _s.path.insert(0, root)
            import trade_journal
            from journal_import.fifo import OpenPosition as _OP
            trade_journal._broker_suggestions([_OP(**o) for o in ops_])
        at = AppTest.from_function(app, args=(ROOT, [o.__dict__ for o in ops]), default_timeout=60)
        at.session_state["holdings_unlocked"] = unlocked
        return at
    return make


def test_broker_numbers_are_offered_per_account_and_saved_only_on_click(broker_page, stores):
    stores["holdings"] = _reg(long=[_row("r1", "VOLV-B.ST", "Quality", 240.0, name="Volvo", shares=90)],
                              swing=[_row("r2", "AAPL", "Wolf", 140.0, name="Apple", shares=5)])
    at = broker_page([_op(), _op(account="nordea_ask", qty=40.0, avg=260.0),
                      _op(isin="US0378331005", ticker="AAPL", name="Apple Inc", qty=5.0, avg=150.0)])
    at.run()
    assert not at.exception, at.exception
    md, caps = _markdown(at), _captions(at)
    assert "🤖 Mäklaren (nordnet_isk): snittkurs 250.5 SEK" in md and "🤖 Mäklaren (nordnet_isk): antal 100 st" in md
    assert "🤖 Mäklaren (nordea_ask): snittkurs 260 SEK" in md and "🤖 Mäklaren (nordea_ask): antal 40 st" in md
    assert "finns på 2 konton — förslaget gäller det här kontot, inte summan" in caps
    assert "AAPL noteras i USD, inte föreslagen" in caps                       # Nordnets SEK är kontots valuta
    assert not [b for b in at.button if b.key.startswith("rf_offer_imp_r2_") and b.key.endswith("_avg")]
    assert not stores["_saves"]
    at.button(key="rf_offer_imp_r1_nordnet_isk_SE0000107419_avg").click().run()
    assert not at.exception, at.exception
    rows = _saved_rows(stores)
    assert len(stores["_saves"]) == 1 and rows["r1"]["entry_price"] == 250.5 and rows["r1"]["shares"] == 90
    at.button(key="rf_offer_imp_r1_nordnet_isk_SE0000107419_qty").click().run()
    assert _saved_rows(stores)["r1"]["shares"] == 100 and _saved_rows(stores)["r1"]["entry_price"] == 250.5


def test_broker_numbers_stay_hidden_while_the_holdings_are_locked(broker_page, stores):
    stores["holdings"] = _reg(long=[_row("r1", "VOLV-B.ST", "Quality", 240.0, name="Volvo")])
    at = broker_page([_op()], unlocked=False)
    at.run()
    assert not at.exception, at.exception
    assert "Innehaven är låsta" in _captions(at)
    assert "VOLV-B.ST" not in _captions(at) + _markdown(at) and not list(at.button)
    assert "holdings" not in at.session_state                                 # registret lästes inte ens


def test_broker_numbers_that_already_match_say_so(broker_page, stores):
    stores["holdings"] = _reg(long=[_row("r1", "VOLV-B.ST", "Quality", 250.5, name="Volvo", shares=100)])
    at = broker_page([_op()])
    at.run()
    assert not at.exception, at.exception
    assert "✓ Snittkurs och antal stämmer med raden." in _captions(at) and not list(at.button)


# 4. Swing
@pytest.fixture
def swing_page(monkeypatch, stores):
    from streamlit.testing.v1 import AppTest
    import market_prices
    closes: dict = {}
    monkeypatch.setattr(market_prices, "close", lambda t, period="1y": closes.get(t, pd.Series(dtype=float)))

    def make(screener):
        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import swing
            swing.render_swing_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=60)
        at.session_state["_wolf_screener_cache"] = screener
        return at
    make.closes = closes
    return make


def _swing_stores(stores, **flags):
    stores["holdings"] = _reg(momentum=[
        {"id": pid, "ticker": t, "strategy": "Momentum", "entry_price": 250.0, "shares": 0,
         "entry_date": "2026-10-01", "extras": {"current": 260.0, "setup": "A", **flags.get(pid, {})}}
        for pid, t in (("m1", "VOLV-B.ST"), ("m2", "ERIC-B.ST"), ("m3", "SAND.ST"))])
    stores["swing"] = {"market": {"aboveMA200": True, "checked": "2026-11-01"},
                       "checklist": {"week": "", "done": []}, "watchlist": [], "positions": [], "closed": []}


def test_swing_rule_boxes_get_suggestions_and_tick_only_on_click(swing_page, stores):
    _swing_stores(stores)
    swing_page.closes["SAND.ST"] = _ohlc(start=130.0, end=100.0)["Close"]     # utanför listan: kurslagret
    scr = _screener()
    at = swing_page(scr)
    at.run()
    assert not at.exception, at.exception
    md = _markdown(at)
    day = scr["generated"][:10]
    assert f"🤖 Screenern {day}: Ur topp 40 (regel 3): ja — finns inte bland de 40 i listan" in md
    assert f"🤖 Screenern {day}: Stängt under MA50 (regel 1): ja — kurs −2,1 % mot MA50" in md
    assert "🤖 Kursen 2026-11-06: Stängt under MA50 (regel 1): ja — stängning 100 under MA50" in md
    keys = {b.key for b in at.button}
    assert {"rf_offer_swing_rank_m3", "rf_offer_swing_ma50_m2", "rf_offer_swing_ma50_m3"} <= keys
    assert not keys & {"rf_offer_swing_rank_m1", "rf_offer_swing_rank_m2", "rf_offer_swing_ma50_m1"}
    assert not at.checkbox(key="swing_rank_m3").value and not stores["_saves"]
    at.button(key="rf_offer_swing_rank_m3").click().run()
    assert not at.exception, at.exception
    assert at.checkbox(key="swing_rank_m3").value
    rows = _saved_rows(stores)
    assert len(stores["_saves"]) == 1 and rows["m3"]["extras"]["outOfRank"] is True
    assert not rows["m3"]["extras"].get("belowMA50") and not rows["m2"]["extras"].get("belowMA50")
    assert "rf_offer_swing_rank_m3" not in {b.key for b in at.button}


def test_swing_never_offers_an_untick_and_skips_a_stale_list(swing_page, stores):
    _swing_stores(stores, m3={"outOfRank": True, "belowMA50": True})
    at = swing_page(_screener(age_days=30))
    at.run()
    assert not at.exception, at.exception
    keys = {b.key for b in at.button}
    assert not {k for k in keys if k.startswith("rf_offer_swing_rank_")}         # gammal lista: inget rank-förslag
    assert "rf_offer_swing_ma50_m3" not in keys                                  # redan kryssad: inget
    assert at.checkbox(key="swing_rank_m3").value and at.checkbox(key="swing_ma50_m3").value
    assert not stores["_saves"]


def test_swing_without_screener_or_prices_shows_nothing(swing_page, stores):
    _swing_stores(stores)
    at = swing_page(None)
    at.session_state["_swing_screener_tried"] = True
    at.run()
    assert not at.exception, at.exception
    assert not [b for b in at.button if b.key.startswith("rf_offer_swing_")]
