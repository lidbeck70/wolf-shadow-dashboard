"""
📥 Uppdatera från mäklaren (portfolio/broker_sync, broker_sync_ui) — påhittade filer, inget nätverk.

1. Nordeas innehavslista (.xlsx): rubrikraden letas upp, kassakonton hoppas över, fondens snittkurs ur
   anskaffningsvärdet när den saknas.
2. Nordnets transaktionsexport (.csv): innehaven ur FIFO.
3. Planen: ny / ändrad / stämmer / saknas / bara text — konton slås ihop bara när alla kontona är uppladdade.
4. apply: en sparning, bara antal och snittkurs (stopp, mål, strategi orörda); nya rader som Untagged.
5. Sidan: inget skrivs när rutan visas eller en fil laddas upp — bara knapparna skriver.
"""
import copy
import io
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from portfolio import broker_sync as bs   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD = ("Type", "AccountKey", "Account", "POA", "FREE/PENSION", "Value on account level",
        "Value change on account level", "ISIN", "MIC", "CURRENCY", "NAME", "HOLDINGS", "PRICE", "PRICETIME",
        "PRICETYPE", "Value change (currency)", "Value change (%)", "PRICEFACTOR", "FX", "Average purchase price",
        "A.FX", "Purchase value of current holdings", "Current Value", "Value in base currency", "Base currency")


def nordea_xlsx(rows) -> bytes:
    """Nordeas 'My holdings' i samma form: tidsstämpel, rubrikrad, rader. rows: (isin, mic, ccy, namn, antal,
    snitt, anskaffningsvärde)."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Holdings"
    ws.append(["01.01.2026 08:00:00"])
    ws.append(list(HEAD))
    for isin, mic, ccy, name, qty, avg, cost in rows:
        d = dict.fromkeys(HEAD)
        d.update({"Type": "Custody", "AccountKey": "ASK:1", "Account": "ASK Testkonto", "ISIN": isin, "MIC": mic,
                  "CURRENCY": ccy, "NAME": name, "HOLDINGS": qty, "PRICE": 1.0, "Average purchase price": avg,
                  "Purchase value of current holdings": cost, "Base currency": "NOK"})
        ws.append([d[h] for h in HEAD])
    ws.append(["CashAccount", "KS:1"] + [None] * (len(HEAD) - 2))
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def nordnet_csv(trades) -> bytes:
    """Nordnets transaktionsexport: UTF-16, tabb, 27 kolumner. trades: (datum, typ, namn, isin, antal, kurs)."""
    lines = ["\t".join(f"c{i}" for i in range(27))]
    for day, typ, name, isin, qty, px in trades:
        cols = [""] * 27
        cols[2], cols[5], cols[6], cols[7] = day, typ, name, isin
        cols[8], cols[9], cols[12], cols[13] = str(qty).replace(".", ","), str(px).replace(".", ","), "SEK", "0"
        lines.append("\t".join(cols))
    return ("\n".join(lines) + "\n").encode("utf-16")


LIST_HEAD = ("Namn", "Valuta", "Antal", "GAV", "Idag %", "Senaste kurs", "Belåningsv.\xa0SEK", "Värde",
             "Värde\xa0SEK", "Avkast. %", "Avkast.\xa0SEK", "ISIN", "Market")


def nordnet_list(rows, last=None) -> bytes:
    """Nordnets innehavslista: UTF-16, tabb, decimalkomma, hårda mellanslag i rubrikerna. rows: (namn, valuta, antal,
    GAV, ISIN). Senaste kurs = GAV (eller last), värdet = antal × kurs."""
    def sv(x):
        return str(x).replace(".", ",")
    lines = ["\t".join(LIST_HEAD)]
    for name, ccy, qty, gav, isin in rows:
        px = (last or {}).get(name, gav)
        value = (last or {}).get(f"{name}:värde", qty * px)
        lines.append("\t".join([name, ccy, sv(qty), sv(gav), "0", sv(px), "0", sv(value), sv(value), "0", "0", isin,
                                "Nasdaq"]))
    return ("\n".join(lines) + "\n").encode("utf-16")


FILE = nordea_xlsx([("SE0000000001", "XSTO", "SEK", "Testbolag B", 70, 32.5271, 2000.0),
                    ("NO0000000002", "XOSL", "NOK", "Oljebolaget", 605, 12.0916, 7300.0),
                    ("SE0000000003", "FONDANO", "NOK", "Silverfonden", 20, 0.0, 8000.0)])
BD = {"SE0000000001": "TEST-B.ST", "NO0000000002": "OLJE.OL"}


# 1. Nordea
def test_nordea_holdings_list_is_read_from_its_header_row():
    broker, hs, errors = bs.read_file("My_holdings.xlsx", FILE)
    assert broker == bs.NORDEA and errors == []
    assert [(h.isin, h.qty, h.currency) for h in hs] == [("SE0000000001", 70, "SEK"), ("NO0000000002", 605, "NOK"),
                                                         ("SE0000000003", 20, "NOK")]
    assert hs[0].avg_price == pytest.approx(32.5271) and hs[0].mic == "XSTO"
    assert hs[2].avg_price == pytest.approx(400.0)              # fonden: 8000 NOK ÷ 20 i kontots basvaluta


def test_an_unreadable_pdf_or_other_file_says_what_to_upload():
    b, hs, errors = bs.read_file("transaction-export.pdf", b"%PDF-1.4")
    assert b is None and hs == [] and errors
    _b, hs, errors = bs.read_file("x.json", b"{}")
    assert hs == [] and "Okänd filtyp" in errors[0]


# 2. Nordnet
def test_nordnet_holdings_come_from_fifo_over_the_transactions():
    data = nordnet_csv([("2026-01-02", "KÖPT", "Volvo B", "SE0000115446", 10, 100.0),
                        ("2026-02-02", "KÖPT", "Volvo B", "SE0000115446", 10, 120.0),
                        ("2026-03-02", "SÅLT", "Volvo B", "SE0000115446", 5, 130.0)])
    broker, hs, _errors = bs.read_file("transactions-and-notes-export.csv", data)
    assert broker == bs.NORDNET and len(hs) == 1
    assert hs[0].qty == 15 and hs[0].avg_price == pytest.approx((5 * 100 + 10 * 120) / 15, rel=1e-4)


# 3. Planen
def _row(rid, ticker, shares, price, strategy="Untagged", **kw):
    return {"id": rid, "ticker": ticker, "name": kw.pop("name", ""), "strategy": strategy, "shares": shares,
            "entry_price": price, "notes": kw.pop("notes", ""), "extras": kw.pop("extras", {}), **kw}


def _hold():
    _b, hs, _e = bs.read_file("h.xlsx", FILE)
    return bs.resolve(hs, [], BD)


def test_plan_sorts_new_changed_same_missing_and_text():
    rows = [_row("r1", "TEST-B.ST", 60, 32.53, name="Testbolag B"),          # antal ändrat
            _row("r2", "OLJE.OL", 605, 12.09),                                # stämmer
            _row("r3", "GONE.OL", 10, 50.0, notes="import · nordea_ask · NOK"),   # hos Nordea, inte i filen
            _row("r4", "NEM", 5, 50.0)]                                       # inget konto: rörs inte
    by = {c.ticker or c.name: c for c in bs.plan(rows, _hold(), {bs.NORDEA})}
    assert by["TEST-B.ST"].kind == bs.CHANGED and (by["TEST-B.ST"].old_qty, by["TEST-B.ST"].qty) == (60, 70)
    assert by["OLJE.OL"].kind == bs.SAME
    assert by["GONE.OL"].kind == bs.MISSING
    assert "NEM" not in by
    assert by["Silverfonden"].kind == bs.NEW and by["Silverfonden"].ticker is None


def test_a_ticker_on_two_rows_or_on_an_account_not_uploaded_is_text_only():
    rows = [_row("a", "TEST-B.ST", 10, 30.0, strategy="Quality"), _row("b", "TEST-B.ST", 10, 30.0, strategy="Wolf")]
    c = next(c for c in bs.plan(rows, _hold(), {bs.NORDEA}) if c.ticker == "TEST-B.ST")
    assert c.kind == bs.INFO and "2 rader" in c.notes[-1]
    rows = [_row("a", "TEST-B.ST", 100, 30.0, extras={"broker_accounts": ["nordea_ask:ASK Testkonto",
                                                                          "nordnet_isk:nordnet_isk"]})]
    c = next(c for c in bs.plan(rows, _hold(), {bs.NORDEA}) if c.ticker == "TEST-B.ST")
    assert c.kind == bs.INFO and "Nordnet ISK" in c.notes[-1]


def test_two_uploaded_accounts_are_summed_and_the_price_only_in_the_listing_currency():
    hs = _hold()
    data = nordnet_csv([("2026-01-02", "KÖPT", "Testbolag B", "SE0000000001", 30, 40.0)])
    _b, nn, _e = bs.read_file("t.csv", data)
    hs = bs.resolve(hs + nn, [], BD)
    c = next(c for c in bs.plan([], hs, {bs.NORDEA, bs.NORDNET}) if c.ticker == "TEST-B.ST")
    assert c.kind == bs.NEW and c.qty == 100 and "summa av 2 konton" in c.notes
    assert c.price == pytest.approx((70 * 32.5271 + 30 * 40) / 100, abs=0.01)


def test_the_average_is_not_offered_at_or_below_the_rows_stop():
    rows = [_row("r1", "TEST-B.ST", 70, 30.0, stop=33.0)]
    c = next(c for c in bs.plan(rows, _hold(), {bs.NORDEA}) if c.ticker == "TEST-B.ST")
    assert c.price is None and any("stoppen" in n for n in c.notes)


def test_resolve_prefers_the_rows_saved_isin_then_borsdata_then_a_matching_name():
    rows = [_row("r1", "MINE.ST", 1, 1.0, extras={"isin": "SE0000000001"}), _row("r2", "OLJ.OL", 1, 1.0,
                                                                                 name="Oljebolaget")]
    _b, hs, _e = bs.read_file("h.xlsx", FILE)
    hs = bs.resolve(hs, rows, {})
    want = [("MINE.ST", "ISIN sparat på raden"), ("OLJ.OL", "namnet liknar radens")]
    assert [(h.ticker, h.how) for h in hs[:2]] == want


def test_isin_map_uses_the_market_suffix():
    import markets as mk
    table = mk.current()
    mid = next(m.id for m in table.values() if m.suffix == ".ST" and m.kind == mk.STOCK)
    assert bs.isin_map([{"isin": "SE0000000001", "ticker": "TEST B", "marketId": mid}], table) == \
        {"SE0000000001": "TEST-B.ST"}


# 4. apply
@pytest.fixture
def registry(monkeypatch):
    import streamlit
    import storage
    state: dict = {}
    saves: list = []
    seed = {"holdings": {"swing": [], "ovtlyr": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0,
                         "long": [_row("r1", "TEST-B.ST", 60, 30.0, strategy="Quality", stop=25.0, target=60.0,
                                       notes="egen anteckning")]}}
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        state.setdefault(name, copy.deepcopy(seed.get(name, default))))
    monkeypatch.setattr(storage, "save_session",
                        lambda name: saves.append(copy.deepcopy(state[name])) or storage.SaveResult(name, "a", "u", "t"))
    return state, saves


def test_apply_writes_quantity_and_price_once_and_leaves_stop_target_strategy(registry):
    state, saves = registry
    import positions
    changes = bs.plan(positions.all_positions(), _hold(), {bs.NORDEA})
    picked = [c for c in changes if c.kind in (bs.NEW, bs.CHANGED)]
    done, skipped, err = bs.apply(picked, {"new:SE0000000003": "silv.fund"}, today="2026-11-15")
    assert err is None and len(saves) == 1                                   # en sparning
    assert sorted(done) == ["OLJE.OL", "SILV.FUND", "TEST-B.ST"] and skipped == []
    rows = {r["ticker"]: r for r in state["holdings"]["long"]}
    t = rows["TEST-B.ST"]
    assert (t["shares"], t["entry_price"], t["stop"], t["target"], t["strategy"], t["notes"]) == \
        (70, 32.53, 25.0, 60.0, "Quality", "egen anteckning")
    assert t["extras"]["isin"] == "SE0000000001" and t["extras"]["broker_accounts"] == ["nordea_ask:ASK Testkonto"]
    assert rows["OLJE.OL"]["strategy"] == "Untagged" and rows["OLJE.OL"]["shares"] == 605
    assert rows["OLJE.OL"]["source"] == "broker" and rows["OLJE.OL"]["entry_date"] == "2026-11-15"
    assert rows["SILV.FUND"]["entry_price"] == 400.0


def test_apply_skips_a_new_holding_without_a_ticker_and_writes_nothing_when_nothing_is_done(registry):
    _state, saves = registry
    fund = bs.Change(bs.NEW, "new:X", None, "Silverfonden", "SE0000000003", ("nordea_ask:A",), qty=1, price=1.0)
    done, skipped, err = bs.apply([fund], {})
    assert done == [] and skipped == ["Silverfonden: ticker saknas"] and err is None and saves == []


# 5. Sidan
@pytest.fixture
def page(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import gist_storage
    import refresh_ui
    import storage
    saves: list = []
    data = {"holdings": {"swing": [], "ovtlyr": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0,
                         "long": [_row("r1", "TEST-B.ST", 60, 30.0, strategy="Quality", name="Testbolag B",
                                       extras={"broker_accounts": ["nordea_ask:ASK Testkonto"]}),
                                  _row("r3", "GONE.OL", 10, 50.0, notes="import · nordea_ask · NOK")]}}
    monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(data.get(n, default)))
    monkeypatch.setattr(storage, "legacy_gist", lambda f: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda n, default=None: default)

    def _save(n):
        import streamlit as st
        saves.append((n, copy.deepcopy(st.session_state[n])))
        return storage.SaveResult(n, "abc1234def", "https://x", "2026-11-15T10:00:00Z")
    monkeypatch.setattr(storage, "save_session", _save)

    from portfolio import broker_sync_ui as ui

    class F(io.BytesIO):
        name = "My_holdings.xlsx"

    def make(with_file: bool):
        monkeypatch.setattr(ui, "_uploader", (lambda: [F(FILE)]) if with_file else (lambda: []))

        def app(root, bd):
            import sys as _s
            _s.path.insert(0, root)
            import streamlit as st
            from portfolio import broker_sync_ui as u
            st.session_state.setdefault(u.BD_MAP, bd)
            u.render_broker_sync()
        return AppTest.from_function(app, args=(ROOT, BD), default_timeout=60)
    make.saves = saves
    return make


def _box(at, key):
    """Kryssrutan för ett kort — nyckeln bär förslaget efter kortets nyckel ('bsync_ck_row:r1:1:70:32.53')."""
    hits = [c for c in at.checkbox if c.key.startswith(f"bsync_ck_{key}:")]
    assert len(hits) == 1, [c.key for c in at.checkbox]
    return hits[0]


def test_the_box_writes_nothing_until_a_button_is_pressed(page):
    at = page(False)
    at.run()
    assert not at.exception, at.exception
    at = page(True)
    at.run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "antal 60 → **70**" in _box(at, "row:r1").label and "TEST\\-B\\.ST" in _box(at, "row:r1").label
    assert "Silverfonden" in _box(at, "new:SE0000000003").label and "GONE.OL" in md
    # fonden saknar ticker: kryssad men räknas inte — knappen säger hur många som faktiskt sparas
    assert at.button(key="bsync_apply").label == "✅ Uppdatera markerade (2)"
    assert page.saves == []                                                   # visa + ladda upp = inget sparat
    at.button(key="bsync_apply").click().run()
    assert not at.exception, at.exception
    assert len(page.saves) == 1
    long = {r["ticker"]: r for r in page.saves[-1][1]["long"]}
    assert long["TEST-B.ST"]["shares"] == 70 and long["TEST-B.ST"]["strategy"] == "Quality"
    assert "OLJE.OL" in long and "Silverfonden" not in {r["name"] for r in long.values()}   # utan ticker: hoppas
    at.button(key="bsync_close_r3").click().run()
    assert not at.exception, at.exception
    closed = page.saves[-1][1]["closed"]
    assert [r["ticker"] for r in closed] == ["GONE.OL"] and closed[0]["exit_reason"] == "Såld enligt mäklaren"


# ── Granskningens fynd ───────────────────────────────────────────────────────
def test_the_journals_isin_map_is_not_used_and_a_borsdata_ticker_must_resemble_the_rows_name():
    """Journalens ISIN-karta kopplar Volvo B (SE0000115446) till ATCO-A.ST — den används inte. Börsdatas ticker på
    en rad med ett helt annat namn blir bara text."""
    xlsx = nordea_xlsx([("SE0000115446", "XSTO", "SEK", "Volvo B", 200, 245.1, 49020.0)])
    rows = [_row("a", "ATCO-A.ST", 30, 150.0, name="Atlas Copco A"),
            _row("v", "VOLV-B.ST", 200, 245.1, name="Volvo B")]
    _b, hs, _e = bs.read_file("h.xlsx", xlsx)
    h = bs.resolve(hs, rows, None)[0]                                         # inte ATCO-A.ST ur kartan
    assert (h.ticker, h.how) == ("VOLV-B.ST", bs.BY_NAME)
    _b, hs, _e = bs.read_file("h.xlsx", xlsx)
    by = {c.ticker: c for c in bs.plan(rows, bs.resolve(hs, rows, {"SE0000115446": "VOLV-B.ST"}), {bs.NORDEA})}
    assert by["VOLV-B.ST"].kind == bs.SAME and "ATCO-A.ST" not in by
    _b, hs, _e = bs.read_file("h.xlsx", xlsx)
    wrong = bs.plan(rows, bs.resolve(hs, rows, {"SE0000115446": "ATCO-A.ST"}), {bs.NORDEA})
    c = next(c for c in wrong if c.ticker == "ATCO-A.ST")
    assert c.kind == bs.INFO and "liknar inte" in c.notes[-1]


def test_the_same_unknown_isin_on_two_accounts_is_one_card_with_the_summed_quantity():
    hs = bs.read_file("h.xlsx", nordea_xlsx([("SE0012673267", "XSTO", "SEK", "Evolution", 10, 900.0, 9000.0)]))[1]
    hs += bs.read_file("t.csv", nordnet_csv([("2026-01-02", "KÖPT", "Evolution", "SE0012673267", 5, 1200.0)]))[1]
    news = [c for c in bs.plan([], bs.resolve(hs, [], {}), {bs.NORDEA, bs.NORDNET}) if c.kind == bs.NEW]
    assert len(news) == 1 and news[0].key == "new:SE0012673267" and news[0].qty == 15
    assert "summa av 2 konton" in news[0].notes and news[0].price == pytest.approx(1000.0)


def test_a_name_only_matches_on_the_same_words_and_currency_and_is_not_ticked():
    xlsx = nordea_xlsx([("NO0003067902", "XOSL", "NOK", "Hexagon Composites ASA", 100, 30.0, 3000.0),
                        ("SE0011090018", "XSTO", "SEK", "Holmen AB ser. B", 21, 317.58, 6669.0)])
    rows = [_row("h", "HEXA-B.ST", 50, 100.0, name="Hexagon AB"),
            _row("o", "HOLM-B.ST", 20, 317.58, name="Holmen B")]
    hs = bs.resolve(bs.read_file("h.xlsx", xlsx)[1], rows, {})
    assert [h.ticker for h in hs] == [None, "HOLM-B.ST"]
    c = next(c for c in bs.plan(rows, hs, {bs.NORDEA}) if c.ticker == "HOLM-B.ST")
    assert c.kind == bs.CHANGED and c.pick is False and any("gissad ur namnet" in n for n in c.notes)


def test_an_unreadable_or_trade_less_nordnet_csv_does_not_count_as_uploaded():
    empty = nordnet_csv([])
    assert bs.read_file("t.csv", empty)[0] is None
    assert bs.read_file("t.csv", "skräp".encode("utf-8"))[0] is None
    assert bs.read_file("x.xlsx", nordea_xlsx([])[:0] or b"PK")[0] is None
    linked = {"broker_accounts": ["nordea_ask:ASK Testkonto", "nordnet_isk:nordnet_isk"]}
    rows = [_row("e", "OLJE.OL", 150, 12.0, extras=linked),
            _row("s", "SINCH.ST", 500, 30.0, notes="import · nordnet_isk · SEK")]
    by = {c.ticker: c for c in bs.plan(rows, _hold(), {bs.NORDEA})}
    assert by["OLJE.OL"].kind == bs.INFO and "SINCH.ST" not in by


def test_nordnet_newest_first_same_day_round_trip_leaves_no_holding():
    data = nordnet_csv([("2026-03-02", "SÅLT", "Sinch", "SE0016101844", 500, 31.0),
                        ("2026-03-02", "KÖPT", "Sinch", "SE0016101844", 500, 30.0),
                        ("2026-01-02", "KÖPT", "Investor B", "SE0015811963", 20, 250.0)])
    _b, hs, _e = bs.read_file("t.csv", data)
    assert [(h.name, h.qty) for h in hs if h.qty] == [("Investor B", 20)]
    assert [(h.name, h.sold) for h in hs if not h.qty] == [("Sinch", True)]    # helt såld: beviset för 'såld?'


def test_two_isins_on_one_ticker_are_never_summed():
    hs = bs.read_file("h.xlsx", FILE)[1]
    hs = bs.resolve(hs, [], {"SE0000000001": "SAME.ST", "NO0000000002": "SAME.ST"})
    c = next(c for c in bs.plan([], hs, {bs.NORDEA}) if c.ticker == "SAME.ST")
    assert c.kind == bs.INFO and "2 olika ISIN" in c.notes[-1]


def test_a_typed_ticker_that_already_stands_in_another_bucket_links_the_isin_instead_of_a_new_row(registry):
    state, saves = registry
    import positions
    positions.load()
    state["holdings"]["momentum"].append(_row("k", "KOG.OL", 30, 300.0, strategy="Momentum"))
    fund = bs.Change(bs.NEW, "new:NO0003043309", None, "Kongsberg Gruppen", "NO0003043309", ("nordea_ask:A",),
                     qty=30, price=300.0)
    done, skipped, err = bs.apply([fund], {"new:NO0003043309": "kog.ol"})
    assert err is None and skipped == [] and done[0].startswith("KOG.OL (kopplad")
    assert [r["ticker"] for r in state["holdings"]["long"]] == ["TEST-B.ST"]          # ingen ny rad
    mom = state["holdings"]["momentum"][0]
    assert mom["extras"]["isin"] == "NO0003043309" and mom["shares"] == 30 and mom["strategy"] == "Momentum"


def test_a_later_file_for_the_same_account_replaces_the_earlier_one(page, monkeypatch):
    from portfolio import broker_sync_ui as ui

    class Old(io.BytesIO):
        name = "My_holdings.xlsx"

    class New(io.BytesIO):
        name = "My_holdings (1).xlsx"
    old = nordea_xlsx([("SE0000000001", "XSTO", "SEK", "Testbolag B", 60, 30.0, 1800.0)])
    new = nordea_xlsx([("SE0000000001", "XSTO", "SEK", "Testbolag B", 70, 32.5271, 2277.0)])
    at = page(True)
    monkeypatch.setattr(ui, "_uploader", lambda: [Old(old), New(new)])
    at.run()
    assert not at.exception, at.exception
    assert any("ersätter My_holdings.xlsx" in w.value for w in at.warning)
    assert "antal 60 → **70**" in _box(at, "row:r1").label                  # inte 130


# ── Kontrollrundans fynd ─────────────────────────────────────────────────────
def test_nordnet_holdings_in_foreign_stocks_match_their_row_by_name_although_the_file_says_sek():
    """Nordnets fil anger kontots valuta (SEK) — valutan prövas inte för Nordnet, annars såg NVIDIA och Equinor
    sålda ut."""
    data = nordnet_csv([("2026-01-02", "KÖPT", "NVIDIA", "US67066G1040", 20, 1300.0),
                        ("2026-01-02", "KÖPT", "Equinor", "NO0010096985", 30, 280.0)])
    rows = [_row("n", "NVDA", 20, 130.0, name="NVIDIA", notes="import · nordnet_isk · SEK"),
            _row("e", "EQNR.OL", 30, 270.0, name="Equinor", notes="import · nordnet_isk · SEK")]
    hs = bs.resolve(bs.read_file("t.csv", data)[1], rows, {})
    assert {h.ticker for h in hs} == {"NVDA", "EQNR.OL"}
    kinds = {c.ticker: c.kind for c in bs.plan(rows, hs, {bs.NORDNET})}
    assert kinds == {"NVDA": bs.SAME, "EQNR.OL": bs.SAME}                       # ingen 'saknas', ingen snittkurs i SEK


def test_industry_words_count_in_the_name_lundin_gold_is_not_lundin_mining():
    xlsx = nordea_xlsx([("CA5503711080", "XSTO", "SEK", "Lundin Gold Inc", 100, 200.0, 20000.0),
                        ("CA5503721063", "XSTO", "SEK", "Lundin Mining Corporation", 300, 90.0, 27000.0)])
    both = [_row("g", "LUG.ST", 100, 200.0, name="Lundin Gold", notes="import · nordea_ask · SEK"),
            _row("m", "LUMI.ST", 300, 90.0, name="Lundin Mining", notes="import · nordea_ask · SEK")]
    hs = bs.resolve(bs.read_file("h.xlsx", xlsx)[1], both, {})
    assert [h.ticker for h in hs] == ["LUG.ST", "LUMI.ST"]
    only_gold = both[:1]
    hs = bs.resolve(bs.read_file("h.xlsx", xlsx)[1], only_gold, {})
    assert [h.ticker for h in hs] == ["LUG.ST", None]                          # Mining gissas inte på Golds rad


def test_one_isin_on_two_accounts_gets_one_ticker_and_one_card():
    xlsx = nordea_xlsx([("NO0010096985", "XOSL", "NOK", "Equinor", 100, 280.0, 28000.0)])
    csv = nordnet_csv([("2026-01-02", "KÖPT", "Equinor ASA", "NO0010096985", 50, 290.0)])
    rows = [_row("e", "EQNR.OL", 150, 283.0, name="Equinor")]
    hs = bs.read_file("h.xlsx", xlsx)[1] + bs.read_file("t.csv", csv)[1]
    hs = bs.resolve(hs, rows, {})
    assert [h.ticker for h in hs] == ["EQNR.OL", "EQNR.OL"]
    cards = [c for c in bs.plan(rows, hs, {bs.NORDEA, bs.NORDNET}) if c.ticker == "EQNR.OL"]
    assert len(cards) == 1 and cards[0].qty == 150 and cards[0].pick is False


def test_a_row_named_like_a_holding_without_ticker_is_not_flagged_as_sold():
    xlsx = nordea_xlsx([("SE0012673267", "XSTO", "SEK", "Evolution", 10, 900.0, 9000.0)])
    rows = [_row("v", "EVO.ST", 10, 900.0, name="Evolution AB (publ)", notes="import · nordea_ask · SEK",
                 extras={})]
    hs = bs.resolve(bs.read_file("h.xlsx", xlsx)[1], [], {})                  # ingen rad att jämföra mot än
    out = bs.plan(rows, hs, {bs.NORDEA})
    assert [c.kind for c in out] == [bs.NEW] and "skriv EVO.ST" in out[0].notes[-1]


def test_a_journal_row_with_a_wrong_ticker_is_not_sold_when_borsdata_gives_the_right_one():
    """Journalens karta gav raden 'ATCO-A.ST' namnet 'Volvo B'. Börsdata säger VOLV-B.ST: ingen 'såld?' på raden
    och inget förkryssat nytt kort (det vore en dubblett)."""
    csv = nordnet_csv([("2026-01-02", "KÖPT", "Volvo B", "SE0000115446", 40, 260.0)])
    rows = [_row("j3", "ATCO-A.ST", 40, 260.0, strategy="Quality", name="Volvo B", stop=230.0,
                 notes="import · nordnet_isk · SEK")]
    hs = bs.resolve(bs.read_file("t.csv", csv)[1], rows, {"SE0000115446": "VOLV-B.ST"})
    out = bs.plan(rows, hs, {bs.NORDNET})
    assert [c.kind for c in out] == [bs.NEW]
    assert out[0].ticker == "VOLV-B.ST" and out[0].pick is False and "ATCO-A.ST" in out[0].notes[-1]


def test_a_wrong_journal_row_matched_as_text_still_unticks_the_new_card():
    """Volvo B på raden ATCO-A.ST (Journalens karta) och Atlas Copco A hos mäklaren: Atlas-kortet blir text mot
    'Volvo B'-raden, och det nya VOLV-B.ST-kortet kryssas inte i — annars blev Volvo dubbelräknad."""
    csv = nordnet_csv([("2026-01-02", "KÖPT", "Volvo B", "SE0000115446", 40, 260.0),
                       ("2026-01-02", "KÖPT", "Atlas Copco A", "SE0017486889", 30, 170.0)])
    rows = [_row("j", "ATCO-A.ST", 40, 260.0, name="Volvo B", notes="import · nordnet_isk · SEK")]
    hs = bs.resolve(bs.read_file("t.csv", csv)[1], rows, {"SE0000115446": "VOLV-B.ST",
                                                          "SE0017486889": "ATCO-A.ST"})
    by = {c.ticker: c for c in bs.plan(rows, hs, {bs.NORDNET})}
    assert by["ATCO-A.ST"].kind == bs.INFO
    assert by["VOLV-B.ST"].kind == bs.NEW and by["VOLV-B.ST"].pick is False and "ATCO-A.ST" in by["VOLV-B.ST"].notes[-1]


def test_swapped_journal_tickers_never_show_a_held_position_as_sold():
    """SEB A och Sandvik ombytta av Journalens karta; Sandvik är såld. Raden 'SAND.ST' med namnet 'SEB A' visas inte
    som 'såld?' — mäklaren har SEB A."""
    csv = nordnet_csv([("2026-01-02", "KÖPT", "SEB A", "SE0000148884", 100, 150.0)])
    rows = [_row("s", "SAND.ST", 100, 150.0, name="SEB A", notes="import · nordnet_isk · SEK"),
            _row("e", "SEB-A.ST", 50, 210.0, name="Sandvik", notes="import · nordnet_isk · SEK")]
    hs = bs.resolve(bs.read_file("t.csv", csv)[1], rows, {"SE0000148884": "SEB-A.ST"})
    out = bs.plan(rows, hs, {bs.NORDNET})
    assert not [c for c in out if c.kind == bs.MISSING and c.ticker == "SAND.ST"]
    assert not [c for c in out if c.kind in (bs.NEW, bs.CHANGED) and c.pick]


# ── Femte rundans fynd ───────────────────────────────────────────────────────
def test_a_row_without_known_account_is_not_cut_to_one_brokers_count_by_default():
    """Equinor 100 hos Nordea på en handskriven rad om 150 (resten hos Nordnet): bara Nordeas fil uppladdad →
    ändringen 150 → 100 kryssas inte i. Med Nordnets innehavslista kryssas den (hela innehavet syns); med Nordnets
    transaktioner inte — ett antal ur köp och sälj kryssas aldrig i."""
    xlsx = nordea_xlsx([("NO0010096985", "XOSL", "NOK", "Equinor", 100, 280.0, 28000.0)])
    csv = nordnet_csv([("2026-01-02", "KÖPT", "Equinor", "NO0010096985", 40, 290.0)])
    rows = [_row("h1", "EQNR.OL", 150, 283.33, stop=240.0)]
    bd = {"NO0010096985": "EQNR.OL"}
    c = bs.plan(rows, bs.resolve(bs.read_file("h.xlsx", xlsx)[1], rows, bd), {bs.NORDEA})[0]
    assert c.kind == bs.CHANGED and c.pick is False and "Nordnet ISK" in c.notes[-1]
    both = bs.read_file("h.xlsx", xlsx)[1] + bs.read_file("t.csv", csv)[1]
    c = bs.plan(rows, bs.resolve(both, rows, bd), {bs.NORDEA, bs.NORDNET})[0]
    assert c.kind == bs.CHANGED and c.qty == 140 and c.pick is False and bs.TX_NOTE in c.notes
    lst = nordnet_list([("Equinor", "NOK", 40, 290.0, "NO0010096985")])
    both = bs.read_file("h.xlsx", xlsx)[1] + bs.read_file("aktier_kontonummer-11112222_2026-11-15.csv", lst)[1]
    c = bs.plan(rows, bs.resolve(both, rows, bd), {bs.NORDEA, bs.NORDNET})[0]
    assert c.kind == bs.CHANGED and c.qty == 140 and c.pick is True


def test_the_name_guard_checks_every_brokers_spelling_whatever_the_upload_order():
    xlsx = nordea_xlsx([("SE0000148884", "XSTO", "SEK", "Skandinaviska Enskilda Banken ser. A", 50, 140.0, 7000.0)])
    csv = nordnet_csv([("2026-01-02", "KÖPT", "SEB A", "SE0000148884", 100, 150.0)])
    rows = [_row("s", "SAND.ST", 100, 150.0, name="SEB A", notes="import · nordnet_isk · SEK")]
    for files in ((("h.xlsx", xlsx), ("t.csv", csv)), (("t.csv", csv), ("h.xlsx", xlsx))):
        hs = [h for n, d in files for h in bs.read_file(n, d)[1]]
        c = next(c for c in bs.plan(rows, bs.resolve(hs, rows, {"SE0000148884": "SEB-A.ST"}),
                                    {bs.NORDEA, bs.NORDNET}) if c.ticker == "SEB-A.ST")
        assert c.kind == bs.NEW and c.pick is False and "SAND.ST" in c.notes[-1]


# ── Sjätte rundans fynd ──────────────────────────────────────────────────────
SEB_XLSX = nordea_xlsx([("SE0000148884", "XSTO", "SEK", "Skandinaviska Enskilda Banken ser. A", 50, 140.0, 7000.0)])
SEB_CSV = nordnet_csv([("2026-01-02", "KÖPT", "SEB A", "SE0000148884", 100, 150.0)])


def test_a_box_ticked_earlier_starts_over_when_the_plan_unticks_the_card(page, monkeypatch):
    """Nordea först (SEB-A.ST nytt, kryssat), sedan Nordnet-filen: kortet ska nu inte kryssas (heter som raden
    SAND.ST). Rutan får ny nyckel och börjar om från planen — ingen gammal bock och inget sparat."""
    import storage
    from portfolio import broker_sync_ui as ui
    data = {"holdings": {"swing": [], "ovtlyr": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0,
                         "long": [_row("s", "SAND.ST", 100, 150.0, name="SEB A",
                                       notes="import · nordnet_isk · SEK")]}}
    monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(data.get(n, default)))
    monkeypatch.setitem(BD, "SE0000148884", "SEB-A.ST")

    class X(io.BytesIO):
        name = "My_holdings.xlsx"

    class C(io.BytesIO):
        name = "transactions.csv"
    at = page(True)
    monkeypatch.setattr(ui, "_uploader", lambda: [X(SEB_XLSX)])
    at.run()
    assert not at.exception, at.exception
    assert _box(at, "new:SEB-A.ST").value is True
    monkeypatch.setattr(ui, "_uploader", lambda: [X(SEB_XLSX), C(SEB_CSV)])
    at.run()
    assert not at.exception, at.exception
    assert _box(at, "new:SEB-A.ST").value is False
    assert at.button(key="bsync_apply").label == "✅ Uppdatera markerade (0)" and page.saves == []


def test_a_changed_card_named_like_a_row_that_is_not_held_is_not_ticked():
    rows = [_row("n", "SEB-A.ST", 50, 140.0, name="Skandinaviska Enskilda Banken ser. A",
                 extras={"isin": "SE0000148884", "broker_accounts": ["nordea_ask:ASK Testkonto"]}),
            _row("s", "SAND.ST", 100, 150.0, name="SEB A", notes="import · nordnet_isk · SEK")]
    hs = bs.read_file("h.xlsx", SEB_XLSX)[1] + bs.read_file("t.csv", SEB_CSV)[1]
    c = next(c for c in bs.plan(rows, bs.resolve(hs, rows, {}), {bs.NORDEA, bs.NORDNET}) if c.ticker == "SEB-A.ST")
    assert c.kind == bs.CHANGED and c.qty == 150 and c.pick is False and "SAND.ST" in c.notes[-1]


def test_a_nordnet_split_makes_the_count_uncertain_and_never_sold():
    """NVIDIA köpt före splitten 10:1: FIFO ser bara köpet (10). Ändringen 100 → 10 kryssas inte i, och efter en
    försäljning visas raden inte som 'såld?'."""
    split = [("2024-01-02", "KÖPT", "NVIDIA", "US67066G1040", 10, 500.0),
             ("2024-06-10", "SPLIT INLÄGG VP", "NVIDIA", "US67066G1040", 100, 0.0),
             ("2024-06-10", "SPLIT UTTAG VP", "NVIDIA", "US67066G1040", 10, 0.0)]
    rows = [_row("n", "NVDA", 100, 50.0, name="NVIDIA", notes="import · nordnet_isk · SEK",
                 extras={"isin": "US67066G1040"})]
    hs = bs.resolve(bs.read_file("t.csv", nordnet_csv(split))[1], rows, {})
    c = bs.plan(rows, hs, {bs.NORDNET})[0]
    assert c.kind == bs.CHANGED and c.qty == 10 and c.pick is False and bs.SHAKY_NOTE in c.notes
    sold = split + [("2025-03-01", "SÅLT", "NVIDIA", "US67066G1040", 60, 120.0)]
    hs = bs.resolve(bs.read_file("t.csv", nordnet_csv(sold))[1], rows, {})
    assert not [c for c in bs.plan(rows, hs, {bs.NORDNET}) if c.kind == bs.MISSING]


# ── Sjunde rundans fynd ──────────────────────────────────────────────────────
def test_a_borsdata_ticker_on_a_text_only_card_does_not_retick_the_changed_card():
    xlsx = nordea_xlsx([("SE0000148884", "XSTO", "SEK", "Skandinaviska Enskilda Banken ser. A", 50, 140.0, 7000.0),
                        ("SE0000667891", "XSTO", "SEK", "Sandvik AB", 20, 220.0, 4400.0)])
    rows = [_row("n", "SEB-A.ST", 50, 140.0, name="Skandinaviska Enskilda Banken ser. A",
                 extras={"isin": "SE0000148884", "broker_accounts": ["nordea_ask:ASK Testkonto"]}),
            _row("s", "SAND.ST", 100, 150.0, name="SEB A", notes="import · nordnet_isk · SEK")]
    hs = bs.read_file("h.xlsx", xlsx)[1] + bs.read_file("t.csv", SEB_CSV)[1]
    bd = {"SE0000148884": "SEB-A.ST", "SE0000667891": "SAND.ST"}
    out = bs.plan(rows, bs.resolve(hs, rows, bd), {bs.NORDEA, bs.NORDNET})
    c = next(c for c in out if c.ticker == "SEB-A.ST")
    assert c.kind == bs.CHANGED and c.pick is False and "SAND.ST" in c.notes[-1]


def test_an_uncertain_nordnet_holding_without_ticker_is_not_ticked():
    split = [("2024-01-02", "KÖPT", "NVIDIA", "US67066G1040", 10, 500.0),
             ("2024-06-10", "SPLIT INLÄGG VP", "NVIDIA", "US67066G1040", 100, 0.0),
             ("2024-06-10", "SPLIT UTTAG VP", "NVIDIA", "US67066G1040", 10, 0.0)]
    c = bs.plan([], bs.resolve(bs.read_file("t.csv", nordnet_csv(split))[1], [], {}), {bs.NORDNET})[0]
    assert c.kind == bs.NEW and c.ticker is None and c.pick is False


# ── Åttonde rundans fynd ─────────────────────────────────────────────────────
def test_shares_moved_into_nordnet_make_the_same_isin_at_nordea_uncertain():
    """40 Investor B flyttade in till Nordnet (INLÄGG VP, inget köp i filen), 30 hos Nordea, en rad om 70: ändringen
    70 → 30 kryssas inte i."""
    xlsx = nordea_xlsx([("SE0015811963", "XSTO", "SEK", "Investor AB ser. B", 30, 251.0, 7530.0)])
    csv = nordnet_csv([("2026-01-02", "KÖPT", "Volvo B", "SE0000115446", 10, 260.0),
                       ("2025-05-02", "INLÄGG VP", "Investor B", "SE0015811963", 40, 0.0)])
    rows = [_row("h2", "INVE-B.ST", 70, 250.0, strategy="Quality", name="Investor B", stop=200.0)]
    hs = bs.read_file("h.xlsx", xlsx)[1] + bs.read_file("t.csv", csv)[1]
    out = bs.plan(rows, bs.resolve(hs, rows, {"SE0015811963": "INVE-B.ST", "SE0000115446": "VOLV-B.ST"}),
                  {bs.NORDEA, bs.NORDNET})
    c = next(c for c in out if c.ticker == "INVE-B.ST")
    assert c.kind == bs.CHANGED and c.qty == 30 and c.pick is False and bs.SHAKY_NOTE in c.notes


# ── Nordnet som PDF (avräkningsnotor) ────────────────────────────────────────
def _note(action, isin, name, fills, day="2026-06-03", fx=None, kop=None, depa="11112222", nbsp=False):
    """En påhittad avräkningsnota i Nordnets layout (pypdf-ordning: Avräkning före avsluten)."""
    sp = " " if nbsp else " "
    lines = [f"{q} {p} {c} US EXCHANGE NQ 15:30:01.017" for q, p, c in fills]
    total = sum(float(q) * float(str(p).replace(sp, "").replace(" ", "").replace(",", ".")) for q, p, _c in fills)
    rate = fx or 1.0
    kop = kop if kop is not None else total * rate
    kop_txt = f"{kop:,.2f}".replace(",", "X").replace(".", ",").replace("X", sp)
    return "\n".join(["Nr 1", "AVRÄKNINGSNOTA I KOMMISSION Utskriftsdatum 2026-06-03", f"Depå: {depa}",
                      "Depåinnehavare: (bort)", f"Affärsdatum: {day}",
                      f"Enligt uppdrag har vi för Er räkning {action} följande värdepapper: ",
                      f"X (ISIN {isin})  - {name}", "Avräkning"]
                     + ([f"Valutakurs USD/SEK {str(rate).replace('.', ',')}"] if fx else [])
                     + ["Antal Pris Valuta Marknad Tid"] + lines
                     + [f"Köpeskilling (SEK) {kop_txt}", "Courtage (SEK) 9,00"])


def _move(head, isin, name, qty, depa="11112222"):
    return "\n".join([f"{head} Utskriftsdatum 2026-11-15", f"Depå: {depa}", f"Värdepapper: {name}",
                      f"ISIN: {isin}", "Affärsdatum: 2025-12-10", f"Antal: {qty}", "Belopp: 0,00 SEK"])


def test_nordnet_contract_notes_become_transactions_checked_against_the_purchase_sum():
    pages = [_note("köpt", "US1252691001", "CF Industries", [(3, "114,0000", "USD")], fx=9.413876),
             _note("köpt", "SE0000000009", "Dyrbolaget", [(2, "1 120,5000", "SEK"), (13000, "0,0300", "SEK")],
                   nbsp=True),
             _note("sålt", "US1252691001", "CF Industries", [(1, "120,0000", "USD")], day="2026-07-01", fx=9.5),
             _note("köpt", "US0000000001", "Fel AB", [(5, "10,0000", "USD")], fx=10.0, kop=999.0),
             _move("SPLIT INLÄGG VP", "US67066G1040", "NVIDIA", 100),
             _move("UTDELNING", "US1252691001", "CF Industries", 2)]
    txs, shaky, errors = bs.parse_nordnet_notes(pages)
    assert [(t.action, t.isin, t.qty, round(t.price, 4)) for t in txs] == [
        ("BUY", "US1252691001", 3, 114.0), ("BUY", "SE0000000009", 13002, round((2 * 1120.5 + 13000 * 0.03) / 13002, 4)),
        ("SELL", "US1252691001", 1, 120.0), ("BUY", "US0000000001", 5, 10.0)]
    assert txs[0].account == "depå …2222 · notor" and txs[0].currency == "USD"        # hela numret sparas aldrig
    assert set(shaky) == {"US0000000001", "US67066G1040"}                  # fel summa, split — inte utdelningen
    assert any("stämmer inte med köpeskillingen" in e for e in errors)


def test_a_nordnet_pdf_is_read_through_its_pages(monkeypatch):
    pages = [_note("köpt", "US1252691001", "CF Industries", [(3, "114,0000", "USD")], fx=9.4),
             _note("köpt", "SE0000115446", "Volvo B", [(10, "250,0000", "SEK")], depa="33334444"),
             _note("sålt", "SE0000115446", "Volvo B", [(10, "260,0000", "SEK")], day="2026-07-01", depa="33334444")]
    monkeypatch.setattr(bs, "_pdf_pages", lambda data: pages)
    broker, hs, errors = bs.read_file("transaction-export.pdf", b"%PDF-1.4 x")
    assert broker == bs.NORDNET and errors == []
    assert [(h.isin, h.qty, h.account) for h in hs if h.qty] == [("US1252691001", 3, "depå …2222 · notor")]
    assert [(h.name, h.sold) for h in hs if not h.qty] == [("Volvo B", True)]
    monkeypatch.setattr(bs, "_pdf_pages", lambda data: [_move("UTDELNING", "US1252691001", "CF", 2)])
    assert bs.read_file("x.pdf", b"%PDF")[0] is None                       # inga köp/sälj: inte läst


def test_a_sale_in_nordnets_transactions_is_text_only(monkeypatch):
    """Nordnets fil är en period: en rad hos Nordnet utan något i filen kan vara köpt före perioden — ingenting. En
    rad vars bolag filen visar sålt får bara text ('stäng själv'), aldrig 'såld?' med Stäng."""
    pages = [_note("köpt", "US1252691001", "CF Industries", [(3, "114,0000", "USD")], fx=9.4),
             _note("köpt", "SE0000115446", "Volvo B", [(10, "250,0000", "SEK")]),
             _note("sålt", "SE0000115446", "Volvo B", [(10, "260,0000", "SEK")], day="2026-07-01")]
    monkeypatch.setattr(bs, "_pdf_pages", lambda data: pages)
    rows = [_row("v", "VOLV-B.ST", 10, 250.0, name="Volvo B", notes="import · nordnet_isk · SEK"),
            _row("o", "OLD.ST", 5, 100.0, name="Gammalt Bolag", notes="import · nordnet_isk · SEK")]
    hs = bs.resolve(bs.read_file("t.pdf", b"%PDF x")[1], rows, {"US1252691001": "CF"})
    out = {c.ticker: c for c in bs.plan(rows, hs, {bs.NORDNET})}
    assert out["VOLV-B.ST"].kind == bs.INFO and out["VOLV-B.ST"].notes == [bs.TX_SOLD_NOTE]
    assert "OLD.ST" not in out and not [c for c in out.values() if c.kind == bs.MISSING]


def test_dividends_in_the_nordnet_csv_do_not_make_a_stock_uncertain():
    data = nordnet_csv([("2026-01-02", "KÖPT", "Equinor", "NO0010096985", 50, 280.0),
                        ("2026-05-20", "UTDELNING", "Equinor", "NO0010096985", 50, 3.5)])
    hs = bs.read_file("t.csv", data)[1]
    assert [(h.qty, h.unsure) for h in hs if h.qty] == [(50, False)]


def test_sells_before_the_period_are_one_summary_line():
    data = nordnet_csv([("2026-02-02", "SÅLT", "Gammal", "SE0000000077", 10, 50.0),
                        ("2026-02-03", "SÅLT", "Gammal", "SE0000000077", 5, 50.0),
                        ("2026-01-02", "KÖPT", "Ny", "SE0000000088", 5, 10.0)])
    _b, _hs, errors = bs.read_file("t.csv", data)
    assert len(errors) == 1 and errors[0].startswith("1 bolag har försäljningar utan köp i filen") and "Gammal" in errors[0]


# ── Nordnets innehavslista ───────────────────────────────────────────────────
LIST_FILE = "aktier_kontonummer-11112222_2026-11-15 2.csv"


def test_nordnets_holdings_list_is_read_straight_off_with_the_account_from_the_file_name():
    data = nordnet_list([("Testbolaget", "SEK", 70, 32.5, "SE0000000001"),
                         ("Kanadagruvan", "CAD", 2400, 0.2664, "")])
    broker, hs, errors, kind = bs.read(LIST_FILE, data)
    assert (broker, kind, errors) == (bs.NORDNET, bs.LIST, [])
    assert [(h.name, h.qty, h.avg_price, h.currency, h.isin, h.account, h.snap, h.unsure) for h in hs] == [
        ("Testbolaget", 70, 32.5, "SEK", "SE0000000001", "depå …2222", True, False),
        ("Kanadagruvan", 2400, 0.2664, "CAD", "", "depå …2222", True, False)]
    assert bs.read_file(LIST_FILE, data)[0] == bs.NORDNET                       # tre värden som förut
    assert {h.account for h in bs.read("innehav.csv", data)[1]} == {bs.NORDNET}  # utan kontonummer: Nordnet i stort
    broker, hs, errors, _k = bs.read(LIST_FILE, nordnet_list([]))
    assert broker == bs.NORDNET and hs == [] and errors == ["Nordnet: inga innehav i innehavslistan."]
    assert bs.read("t.csv", nordnet_csv([("2026-01-02", "KÖPT", "X", "SE0000000001", 1, 1.0)]))[3] == bs.TX


def test_a_list_row_whose_value_does_not_add_up_is_uncertain_and_not_ticked():
    data = nordnet_list([("Testbolaget", "SEK", 70, 32.5, "SE0000000001")], last={"Testbolaget:värde": 9999.0})
    _b, hs, errors, _k = bs.read(LIST_FILE, data)
    assert hs[0].unsure and "stämmer inte med värdet" in errors[0]
    c = bs.plan([], bs.resolve(hs, [], BD), {bs.NORDNET})[0]
    assert c.kind == bs.NEW and c.pick is False and bs.LIST_SHAKY_NOTE in c.notes


def _reads(*files):
    """combine tar (filnamn, mäklare, innehav, sort) — som rutan."""
    return [(name, b, hs, kind) for name, data in files for b, hs, _e, kind in [bs.read(name, data)]]


def test_the_list_gives_nordnets_counts_and_the_transactions_only_lend_an_isin_by_the_exact_name():
    lst = nordnet_list([("Kanadagruvan", "CAD", 2400, 0.27, ""), ("Dubbelnamn", "USD", 5, 10.0, "")])
    tx = nordnet_csv([("2026-01-02", "KÖPT", "kanadagruvan ", "CA0000000001", 1000, 2.0),
                      ("2026-01-02", "KÖPT", "Dubbelnamn", "US0000000001", 1, 1.0),
                      ("2026-01-03", "KÖPT", "Dubbelnamn", "US0000000002", 1, 1.0),
                      ("2026-01-02", "KÖPT", "Bara i transaktionerna", "SE0000000099", 7, 1.0)])
    for files in ([("t.csv", tx), (LIST_FILE, lst)], [(LIST_FILE, lst), ("t.csv", tx)]):    # ordningen spelar ingen roll
        hs, uploaded, msgs = bs.combine(_reads(*files))
        assert uploaded == {bs.NORDNET}
        assert [(h.name, h.qty, h.isin) for h in hs] == [("Kanadagruvan", 2400, "CA0000000001"),
                                                         ("Dubbelnamn", 5, "")]          # två ISIN: inget lånas
        assert msgs == [("caption", "Nordnets innehavslista ger antalen; transaktionsfilen används bara för ISIN "
                                    "(1 rader utan ISIN fick det via samma namn).")]


def test_two_nordnet_transaction_files_are_never_summed(monkeypatch):
    csv = nordnet_csv([("2026-01-02", "KÖPT", "CF Industries", "US1252691001", 3, 114.0)])
    monkeypatch.setattr(bs, "_pdf_pages", lambda data: [_note("köpt", "US1252691001", "CF Industries",
                                                              [(3, "114,0000", "USD")], fx=9.4)])
    hs, _u, msgs = bs.combine(_reads(("t.csv", csv), ("t.pdf", b"%PDF x")))
    assert [(h.isin, h.qty) for h in hs if h.qty] == [("US1252691001", 3)]                    # inte 6
    assert msgs == [("warning", "t.pdf ersätter t.csv för Nordnet ISK — Nordnets transaktionsfiler räknas aldrig "
                                "ihop.")]


def test_a_nordnet_list_without_account_number_replaces_the_other_nordnet_lists():
    a = nordnet_list([("Testbolaget", "SEK", 70, 32.5, "SE0000000001")])
    b = nordnet_list([("Testbolaget", "SEK", 80, 33.0, "SE0000000001")])
    hs, _u, msgs = bs.combine(_reads((LIST_FILE, a), ("innehav.csv", b)))
    assert [h.qty for h in hs] == [80] and "innehav.csv ersätter " + LIST_FILE in msgs[0][1]
    hs, _u, msgs = bs.combine(_reads(("innehav.csv", b), (LIST_FILE, a)))
    assert [h.qty for h in hs] == [70]
    other = nordnet_list([("Annat", "SEK", 5, 1.0, "SE0000000002")])
    hs, _u, msgs = bs.combine(_reads((LIST_FILE, a), ("aktier_kontonummer-33334444_2026-11-15.csv", other)))
    assert [h.qty for h in hs] == [70, 5] and msgs == []                                      # två depåer: båda


def test_a_nordnet_row_missing_from_the_list_is_sold_question_only_for_a_depa_the_list_covers():
    lst = nordnet_list([("Testbolaget", "SEK", 70, 32.5, "SE0000000001")])
    rows = [_row("a", "GONE.ST", 10, 5.0, extras={"broker_accounts": ["nordnet_isk:depå …2222"]}),
            _row("b", "ELSE.ST", 10, 5.0, extras={"broker_accounts": ["nordnet_isk:depå …9999"]}),
            _row("c", "JOUR.ST", 10, 5.0, notes="import · nordnet_isk · SEK")]
    hs = bs.resolve(bs.read(LIST_FILE, lst)[1], rows, BD)
    miss = {c.ticker: c.notes for c in bs.plan(rows, hs, {bs.NORDNET}) if c.kind == bs.MISSING}
    assert miss == {"GONE.ST": ["finns inte i Nordnets innehavslista — såld?"]}   # Journalens rad: okänd depå
    hs = bs.resolve(bs.read("innehav.csv", lst)[1], rows, BD)                 # listan utan kontonummer: ingenting
    assert not [c for c in bs.plan(rows, hs, {bs.NORDNET}) if c.kind == bs.MISSING]


def test_a_name_match_from_the_list_checks_the_currency():
    rows = [_row("n", "NVDA", 5, 100.0, name="NVIDIA")]
    hs = bs.resolve(bs.read(LIST_FILE, nordnet_list([("NVIDIA", "USD", 5, 100.0, "")]))[1], rows, {})
    assert (hs[0].ticker, hs[0].how) == ("NVDA", bs.BY_NAME)
    hs = bs.resolve(bs.read(LIST_FILE, nordnet_list([("NVIDIA", "SEK", 5, 100.0, "")]))[1], rows, {})
    assert hs[0].ticker is None


def test_the_brokers_name_is_saved_on_the_row_and_finds_it_next_time_without_isin(registry):
    state, _saves = registry
    import positions
    lst = nordnet_list([("Kanadagruvan", "CAD", 2400, 0.27, "")])
    hs = bs.resolve(bs.read(LIST_FILE, lst)[1], positions.all_positions(), {})
    new = [c for c in bs.plan(positions.all_positions(), hs, {bs.NORDNET}) if c.kind == bs.NEW]
    assert new[0].names == ("Kanadagruvan",)
    done, _s, err = bs.apply(new, {new[0].key: "kan.v"}, today="2026-11-15")
    assert err is None and done == ["KAN.V"]
    row = next(r for r in state["holdings"]["long"] if r["ticker"] == "KAN.V")
    assert row["extras"]["broker_names"] == ["Kanadagruvan"] and row["extras"]["broker_accounts"] == \
        ["nordnet_isk:depå …2222"]
    again = nordnet_list([("Kanadagruvan", "CAD", 3000, 0.25, "")])
    hs = bs.resolve(bs.read(LIST_FILE, again)[1], positions.all_positions(), {})
    assert (hs[0].ticker, hs[0].how) == ("KAN.V", bs.SAVED_NAME)
    c = next(c for c in bs.plan(positions.all_positions(), hs, {bs.NORDNET}) if c.ticker == "KAN.V")
    assert c.kind == bs.CHANGED and c.qty == 3000 and c.pick is True and not c.notes


# ── Nordnets transaktioner: antal filen inte kan förklara kryssas inte i (granskningen av PDF-stödet) ──
def test_a_trim_of_a_position_held_before_the_file_is_never_sold_with_a_close_button(monkeypatch):
    pages = [_note("köpt", "SE0015811963", "Investor B", [(20, "250,0000", "SEK")], day="2026-03-02"),
             _note("sålt", "SE0015811963", "Investor B", [(20, "270,0000", "SEK")], day="2026-05-04")]
    monkeypatch.setattr(bs, "_pdf_pages", lambda data: pages)
    held = _row("i", "INVE-B.ST", 100, 210.0, name="Investor B", entry_date="2024-05-01",
                extras={"isin": "SE0015811963", "broker_accounts": ["nordnet_isk:depå …2222"]})
    hs = bs.resolve(bs.read_file("t.pdf", b"%PDF x")[1], [held], {})
    assert not [c for c in bs.plan([held], hs, {bs.NORDNET}) if c.kind == bs.MISSING]
    bigger = dict(held, shares=20)                                     # filen säljer 20, raden har 20 men från 2024
    assert not [c for c in bs.plan([bigger], hs, {bs.NORDNET}) if c.kind == bs.MISSING]
    synced = dict(bigger, entry_date="2026-11-15")                     # registreringsdagen säger inget om köpet
    assert [(c.kind, c.notes) for c in bs.plan([synced], hs, {bs.NORDNET})] == [(bs.INFO, [bs.TX_SOLD_NOTE])]


def test_the_sale_of_another_share_class_is_no_proof():
    csv = nordnet_csv([("2026-01-02", "KÖPT", "Atlas Copco B", "SE0011166628", 10, 150.0),
                       ("2026-02-02", "SÅLT", "Atlas Copco B", "SE0011166628", 10, 160.0)])
    rows = [_row("a", "ATCO-A.ST", 10, 170.0, name="Atlas Copco A", notes="import · nordnet_isk · SEK")]
    hs = bs.resolve(bs.read_file("t.csv", csv)[1], rows, {})
    assert not [c for c in bs.plan(rows, hs, {bs.NORDNET}) if c.kind == bs.MISSING]


def test_a_count_from_nordnets_transactions_is_never_ticked_up_or_down():
    rows = [_row("i", "INVE-B.ST", 100, 210.0, name="Investor B", entry_date="2024-05-01",
                 extras={"isin": "SE0015811963", "broker_accounts": ["nordnet_isk:nordnet_isk"]})]
    cases = [([("2026-02-02", "KÖPT", "Investor B", "SE0015811963", 20, 250.0)], 20),       # ned
             ([("2026-02-02", "KÖPT", "Investor B", "SE0015811963", 120, 260.0)], 120)]     # upp
    for trades, want in cases:
        c = bs.plan(rows, bs.resolve(bs.read_file("t.csv", nordnet_csv(trades))[1], rows, {}), {bs.NORDNET})[0]
        assert c.kind == bs.CHANGED and (c.old_qty, c.qty) == (100, want) and c.pick is False
        assert bs.TX_NOTE in c.notes


def test_a_new_card_counted_from_nordnet_transactions_is_not_ticked():
    csv = nordnet_csv([("2026-02-02", "KÖPT", "Testbolag B", "SE0000000001", 20, 30.0)])
    c = bs.plan([], bs.resolve(bs.read_file("t.csv", csv)[1], [], BD), {bs.NORDNET})[0]
    assert c.kind == bs.NEW and c.ticker == "TEST-B.ST" and c.pick is False and bs.TX_NOTE in c.notes
    lst = nordnet_list([("Testbolag B", "SEK", 20, 30.0, "SE0000000001")])
    c = bs.plan([], bs.resolve(bs.read(LIST_FILE, lst)[1], [], BD), {bs.NORDNET})[0]
    assert c.kind == bs.NEW and c.pick is True and bs.TX_NOTE not in c.notes        # innehavslistan: säkert


def test_the_box_reads_nordnets_list_and_takes_the_isin_from_the_transactions(page, monkeypatch):
    from portfolio import broker_sync_ui as ui

    class Lst(io.BytesIO):
        name = LIST_FILE

    class Tx(io.BytesIO):
        name = "transactions.csv"
    lst = nordnet_list([("Kanadagruvan", "CAD", 2400, 0.27, "")])
    tx = nordnet_csv([("2026-01-02", "KÖPT", "Kanadagruvan", "CA0000000001", 1000, 2.0)])
    at = page(True)
    monkeypatch.setattr(ui, "_uploader", lambda: [Lst(lst), Tx(tx)])
    at.run()
    assert not at.exception, at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "Nordnet ISK (innehavslista), 1 innehav" in caps and "används bara för ISIN" in caps
    box = _box(at, "new:CA0000000001")                                         # ISIN:et lånat via namnet
    assert "2400 st" in box.label.replace("\\", "") and box.value is True
    assert not at.warning and page.saves == []


# ── Granskningen av innehavslistan ───────────────────────────────────────────
def test_nordeas_count_never_cuts_away_a_nordnet_part_no_nordnet_list_shows():
    """Equinor 50 hos Nordnet (2024) och 30 hos Nordea på samma rad: Nordnets transaktioner för en period utan
    Equinor visar inte Nordnet-delen — bara text, aldrig 80 → 30."""
    xlsx = nordea_xlsx([("NO0010096985", "XOSL", "NOK", "Equinor", 30, 280.0, 8400.0)])
    csv = nordnet_csv([("2026-09-02", "KÖPT", "Volvo B", "SE0000115446", 5, 250.0)])
    rows = [_row("e", "EQNR.OL", 80, 283.0, name="Equinor",
                 extras={"broker_accounts": ["nordea_ask:ASK Testkonto", "nordnet_isk:nordnet_isk"]})]
    hs = bs.read_file("h.xlsx", xlsx)[1] + bs.read_file("t.csv", csv)[1]
    c = next(c for c in bs.plan(rows, bs.resolve(hs, rows, {"NO0010096985": "EQNR.OL"}), {bs.NORDEA, bs.NORDNET})
             if c.ticker == "EQNR.OL")
    assert c.kind == bs.INFO and c.notes[-1] == bs.NN_GAP_NOTE
    rows[0]["extras"]["broker_accounts"][1] = "nordnet_isk:depå …2222"       # listan för depån visar att den är borta
    lst = nordnet_list([("Volvo B", "SEK", 5, 250.0, "SE0000115446")])
    hs = bs.read_file("h.xlsx", xlsx)[1] + bs.read(LIST_FILE, lst)[1]
    c = next(c for c in bs.plan(rows, bs.resolve(hs, rows, {"NO0010096985": "EQNR.OL"}), {bs.NORDEA, bs.NORDNET})
             if c.ticker == "EQNR.OL")
    assert c.kind == bs.CHANGED and (c.old_qty, c.qty) == (80, 30) and c.pick is True


def test_each_nordnet_list_shows_only_its_own_table():
    """'aktier_…' och 'fonder_…' för samma depå läggs ihop; en fondrad visas inte som såld av aktielistan."""
    aktier = nordnet_list([("Testbolaget", "SEK", 70, 32.5, "SE0000000001")])
    fonder = nordnet_list([("Indexfonden", "SEK", 12.5, 100.0, "SE0000000055")])
    hs, _u, msgs = bs.combine(_reads((LIST_FILE, aktier), ("fonder_kontonummer-11112222_2026-11-15.csv", fonder)))
    assert [(h.name, h.account) for h in hs] == [("Testbolaget", "depå …2222"), ("Indexfonden", "depå …2222 · fonder")]
    assert msgs == []
    fund = _row("f", "INDEX.FUND", 12.5, 100.0, extras={"broker_accounts": ["nordnet_isk:depå …2222 · fonder"]})
    hs = bs.resolve(bs.read(LIST_FILE, aktier)[1], [fund], BD)
    assert not [c for c in bs.plan([fund], hs, {bs.NORDNET}) if c.kind == bs.MISSING]


def test_a_list_line_without_isin_or_ticker_keeps_a_row_in_its_currency_from_looking_sold():
    lst = nordnet_list([("NVIDIA", "USD", 20, 120.0, "")])
    nvda = _row("n", "NVDA", 20, 120.0, extras={"isin": "US67066G1040", "broker_accounts": ["nordnet_isk:depå …2222"]})
    volvo = _row("v", "VOLV-B.ST", 5, 250.0,
                 extras={"isin": "SE0000115446", "broker_accounts": ["nordnet_isk:depå …2222"]})
    out = {c.ticker: c for c in bs.plan([nvda, volvo], bs.resolve(bs.read(LIST_FILE, lst)[1], [nvda, volvo], {}),
                                        {bs.NORDNET}) if c.ticker}
    assert out["NVDA"].kind == bs.INFO and "NVIDIA" in out["NVDA"].notes[0]          # text, ingen Stäng
    assert out["VOLV-B.ST"].kind == bs.MISSING                                       # svensk aktie: listan har ISIN
    xlsx = nordea_xlsx([("US67066G1040", "XNAS", "USD", "NVIDIA Corp", 10, 110.0, 1100.0)])
    accts = ["nordea_ask:ASK Testkonto", "nordnet_isk:depå …2222"]
    both = dict(nvda, shares=30, extras={"isin": "US67066G1040", "broker_accounts": accts})
    hs = bs.read_file("h.xlsx", xlsx)[1] + bs.read(LIST_FILE, lst)[1]
    c = next(c for c in bs.plan([both], bs.resolve(hs, [both], {}), {bs.NORDEA, bs.NORDNET}) if c.ticker == "NVDA")
    assert c.kind == bs.CHANGED and (c.old_qty, c.qty) == (30, 10) and c.pick is False and "NVIDIA" in c.notes[-1]


# ── Bekräftelserundan ────────────────────────────────────────────────────────
def test_an_isin_lent_from_the_transactions_does_not_turn_off_the_saved_broker_name():
    """AMD kopplad via Nordnets sparade namn (listan har inget ISIN). Laddas transaktionerna upp samtidigt får
    listans rad ISIN — raden hittas ändå via namnet och visas aldrig som 'såld?'."""
    amd = _row("a", "AMD", 12, 150.0, extras={"broker_names": ["Advanced Micro Devices"],
                                              "broker_accounts": ["nordnet_isk:depå …2222"]})
    lst = nordnet_list([("Advanced Micro Devices", "USD", 12, 150.0, "")])
    tx = nordnet_csv([("2026-10-02", "KÖPT", "Advanced Micro Devices", "US0079031078", 2, 160.0)])
    hs, up, _m = bs.combine(_reads((LIST_FILE, lst), ("t.csv", tx)))
    assert hs[0].isin == "US0079031078"
    hs = bs.resolve(hs, [amd], {})
    assert (hs[0].ticker, hs[0].how) == ("AMD", bs.SAVED_NAME)
    out = bs.plan([amd], hs, up)
    assert [(c.kind, c.ticker) for c in out] == [(bs.SAME, "AMD")]
    other = dict(amd, extras=dict(amd["extras"], isin="US0000000009"))      # raden har ett annat ISIN: ingen koppling
    assert bs.resolve(bs.combine(_reads((LIST_FILE, lst), ("t.csv", tx)))[0], [other], {})[0].ticker is None
    hs = [bs.Holding(bs.NORDNET, "depå …2222", "", "Advanced Micro Devices", "USD", 12, 150.0, snap=True)]
    assert not [c for c in bs.plan([other], hs, up) if c.kind == bs.MISSING]   # skyddsnätet: namnet finns kvar


def test_a_row_without_known_account_is_not_cut_to_nordeas_count_when_nordnet_is_only_transactions():
    xlsx = nordea_xlsx([("NO0010096985", "XOSL", "NOK", "Equinor", 100, 280.0, 28000.0)])
    csv = nordnet_csv([("2026-09-02", "KÖPT", "Volvo B", "SE0000115446", 5, 250.0)])
    rows = [_row("e", "EQNR.OL", 150, 283.0, name="Equinor")]
    hs = bs.resolve(bs.read_file("h.xlsx", xlsx)[1] + bs.read_file("t.csv", csv)[1], rows,
                    {"NO0010096985": "EQNR.OL"})
    c = next(c for c in bs.plan(rows, hs, {bs.NORDEA, bs.NORDNET}) if c.ticker == "EQNR.OL")
    assert c.kind == bs.CHANGED and c.pick is False and "Nordnet ISK" in c.notes[-1]


def test_a_row_without_known_account_is_not_cut_when_the_nordnet_part_may_be_a_list_line_without_isin():
    xlsx = nordea_xlsx([("US8740391003", "XNYS", "USD", "Taiwan Semiconductor Mfg", 30, 150.0, 4500.0)])
    lst = nordnet_list([("Taiwan Semiconductor Manufacturing Co Ltd", "USD", 50, 140.0, "")])
    rows = [_row("t", "TSM", 80, 145.0, name="Taiwan Semi")]
    hs = bs.resolve(bs.read_file("h.xlsx", xlsx)[1] + bs.read(LIST_FILE, lst)[1], rows, {"US8740391003": "TSM"})
    c = next(c for c in bs.plan(rows, hs, {bs.NORDEA, bs.NORDNET}) if c.ticker == "TSM")
    assert c.kind == bs.CHANGED and (c.old_qty, c.qty) == (80, 30) and c.pick is False
    assert any("Taiwan Semiconductor Manufacturing Co Ltd" in n for n in c.notes)


def test_only_the_aktier_list_is_the_aktier_table_whatever_the_other_file_names():
    aktier = nordnet_list([("Volvo B", "SEK", 5, 250.0, "SE0000115446")])
    etf = nordnet_list([("Indexfonden ETF", "SEK", 3, 100.0, "SE0000000055")])
    names = {"etf_er_kontonummer-11112222_2026-11-15.csv": "depå …2222 · etfer",
             "etf-er_kontonummer-11112222_2026-11-15.csv": "depå …2222 · etfer",
             "börshandlade_produkter_kontonummer-11112222_2026-11-15.csv": "depå …2222 · börshandladeprodukter",
             "fonder-kontonummer-11112222_2026-11-15.csv": "depå …2222 · fonder",
             "kontonummer-11112222_2026-11-15.csv": "depå …2222 · okänd tabell",
             "Aktier_kontonummer-11112222_2026-11-15 (1).csv": "depå …2222"}
    for name, acct in names.items():
        assert bs.read(name, etf)[1][0].account == acct, name
    volvo = _row("v", "VOLV-B.ST", 5, 250.0,
                 extras={"isin": "SE0000115446", "broker_accounts": ["nordnet_isk:depå …2222"]})
    for files in ([(LIST_FILE, aktier), ("etf_er_kontonummer-11112222_2026-11-15.csv", etf)],
                  [("etf_er_kontonummer-11112222_2026-11-15.csv", etf), (LIST_FILE, aktier)],
                  [("etf_er_kontonummer-11112222_2026-11-15.csv", etf)]):
        hs, up, msgs = bs.combine(_reads(*files))
        assert msgs == [] and not [c for c in bs.plan([volvo], bs.resolve(hs, [volvo], BD), up)
                                   if c.kind == bs.MISSING]


def test_an_account_from_nordnets_contract_notes_is_never_covered_by_a_list(monkeypatch):
    monkeypatch.setattr(bs, "_pdf_pages", lambda data: [_note("köpt", "SE0000000077", "XACT Bull",
                                                              [(10, "300,0000", "SEK")])])
    acct = bs.read_file("t.pdf", b"%PDF x")[1][0].account
    bull = _row("b", "XBULL.ST", 10, 300.0, extras={"broker_accounts": [f"nordnet_isk:{acct}"]})
    lst = nordnet_list([("Testbolaget", "SEK", 70, 32.5, "SE0000000001")])
    assert not [c for c in bs.plan([bull], bs.resolve(bs.read(LIST_FILE, lst)[1], [bull], BD), {bs.NORDNET})
                if c.kind == bs.MISSING]


def test_a_row_on_two_nordnet_depas_is_not_cut_by_one_depas_list():
    rows = [_row("e", "EQNR.OL", 150, 283.0, name="Equinor",
                 extras={"isin": "NO0010096985", "broker_accounts": ["nordnet_isk:depå …1111",
                                                                     "nordnet_isk:depå …2222"]})]
    lst = nordnet_list([("Equinor", "NOK", 100, 280.0, "NO0010096985")])
    hs = bs.resolve(bs.read(LIST_FILE, lst)[1], rows, {})
    c = bs.plan(rows, hs, {bs.NORDNET})[0]
    assert c.kind == bs.INFO and c.notes[-1] == bs.NN_GAP_NOTE
    other = nordnet_list([("Volvo B", "SEK", 5, 250.0, "SE0000115446")])      # depå …1111:s lista utan Equinor
    hs, up, _m = bs.combine(_reads((LIST_FILE, lst), ("aktier_kontonummer-33331111_2026-11-15.csv", other)))
    c = next(c for c in bs.plan(rows, bs.resolve(hs, rows, {}), up) if c.ticker == "EQNR.OL")
    assert c.kind == bs.CHANGED and (c.old_qty, c.qty) == (150, 100) and c.pick is True


# ── Bekräftelserunda 3 ───────────────────────────────────────────────────────
def test_a_row_without_known_account_is_not_cut_by_a_nordnet_list_that_does_not_show_it():
    """XACT 150 för hand: 100 hos Nordea, 50 i Nordnets ETF-tabell. Aktielistan (eller fondlistan, eller en annan
    depås lista) visar inte ETF:en — 150 → 100 kryssas inte i. Med ETF-listan blir summan rätt och kryssad."""
    xlsx = nordea_xlsx([("SE0000000066", "XSTO", "SEK", "XACT OMXS30", 100, 300.0, 30000.0)])
    rows = [_row("x", "XACT-OMXS30.ST", 150, 300.0, name="XACT OMXS30")]
    bd = {"SE0000000066": "XACT-OMXS30.ST"}
    aktier = nordnet_list([("Volvo B", "SEK", 5, 250.0, "SE0000115446")])
    for name in (LIST_FILE, "fonder_kontonummer-11112222_2026-11-15.csv", "aktier_kontonummer-33334444_2026-11-15.csv"):
        hs, up, _m = bs.combine([("h.xlsx", *bs.read("h.xlsx", xlsx)[:2], bs.LIST)] + _reads((name, aktier)))
        c = next(c for c in bs.plan(rows, bs.resolve(hs, rows, bd), up) if c.ticker == "XACT-OMXS30.ST")
        assert c.kind == bs.CHANGED and (c.old_qty, c.qty) == (150, 100) and c.pick is False, name
    etf = nordnet_list([("XACT OMXS30", "SEK", 50, 300.0, "SE0000000066")])
    hs, up, _m = bs.combine([("h.xlsx", *bs.read("h.xlsx", xlsx)[:2], bs.LIST)]
                            + _reads((LIST_FILE, aktier), ("etf_er_kontonummer-11112222_2026-11-15.csv", etf)))
    c = next(c for c in bs.plan(rows, bs.resolve(hs, rows, bd), up) if c.ticker == "XACT-OMXS30.ST")
    assert c.kind == bs.SAME and c.qty == 150
