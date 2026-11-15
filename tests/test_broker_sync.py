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


def test_a_pdf_or_other_file_says_what_to_upload():
    _b, hs, errors = bs.read_file("transaction-export.pdf", b"%PDF-1.4")
    assert hs == [] and "CSV" in errors[0]
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


def test_the_box_writes_nothing_until_a_button_is_pressed(page):
    at = page(False)
    at.run()
    assert not at.exception, at.exception
    at = page(True)
    at.run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    labels = {c.key: c.label for c in at.checkbox}
    assert "antal 60 → **70**" in labels["bsync_ck_row:r1"] and "TEST\\-B\\.ST" in labels["bsync_ck_row:r1"]
    assert "Silverfonden" in labels["bsync_ck_new:SE0000000003"] and "GONE.OL" in md
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
    assert [(h.name, h.qty) for h in hs] == [("Investor B", 20)]


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
    labels = {c.key: c.label for c in at.checkbox}
    assert "antal 60 → **70**" in labels["bsync_ck_row:r1"]                  # inte 130


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
    ändringen 150 → 100 kryssas inte i. Båda filerna → den kryssas (hela innehavet syns)."""
    xlsx = nordea_xlsx([("NO0010096985", "XOSL", "NOK", "Equinor", 100, 280.0, 28000.0)])
    csv = nordnet_csv([("2026-01-02", "KÖPT", "Equinor", "NO0010096985", 40, 290.0)])
    rows = [_row("h1", "EQNR.OL", 150, 283.33, stop=240.0)]
    bd = {"NO0010096985": "EQNR.OL"}
    c = bs.plan(rows, bs.resolve(bs.read_file("h.xlsx", xlsx)[1], rows, bd), {bs.NORDEA})[0]
    assert c.kind == bs.CHANGED and c.pick is False and "Nordnet ISK" in c.notes[-1]
    both = bs.read_file("h.xlsx", xlsx)[1] + bs.read_file("t.csv", csv)[1]
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
