"""
👀 Övervakningen — sidan (portfolio/watch_ui.py) i AppTest och navigationen: PORTFÖLJ → 👀 Övervakning direkt efter
Positioner och utanför låset, målet 'watch', flikguiden. Läget läses ur Thors rader, inget nätverk: lagret går mot
ett repo i minnet (storage.load_json/save_json) och Thors data är testernas filer.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import storage  # noqa: E402
import test_thor_ui as tu  # noqa: E402
import watchlist as wl  # noqa: E402
from thor import candidates as tc  # noqa: E402
from ui import nav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SESSION_LOAD = storage.session_load            # den riktiga — test_thor_ui._patch byter den mot en tom läsare


def fake_repo(monkeypatch, files=None) -> dict:
    """data/<namn>.json i minnet: storage läser och sparar här i stället för GitHub (med den riktiga
    session_load, så att sparraden och 'osparat' fungerar som i panelen). Returnerar repot."""
    repo = {k: json.loads(json.dumps(v)) for k, v in (files or {}).items()}

    def _load(name, default=None):
        return json.loads(json.dumps(repo[name])) if name in repo else default

    def _save(name, data, sha=None):
        repo[name] = json.loads(json.dumps(data, default=str))
        return storage.SaveResult(name, "abc1234def", "https://x", "2026-10-29T10:00:00+00:00")

    monkeypatch.setattr(storage, "session_load", _SESSION_LOAD)
    monkeypatch.setattr(storage, "load_json", _load)
    monkeypatch.setattr(storage, "save_json", _save)
    return repo


def thor_data(today="2026-10-15") -> dict:
    """Thors data som sidan delar: testernas filer och raderna byggda på ett fast datum (inget blir GAMMAL)."""
    data = dict(tu.DATA)
    data["rows"], data["errors"] = tc.build(tu.DATA, today=today), {}
    return data


def _page():
    from portfolio.watch_ui import render_watch_page
    render_watch_page()


def _app(monkeypatch, items=None, data=None):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    repo = fake_repo(monkeypatch, {"watchlist": wl.normalize({"items": json.loads(json.dumps(items or []))})})
    at = AppTest.from_function(_page)
    at.session_state["thor_data"] = data if data is not None else thor_data()
    return at, repo


ITEMS = [{"id": "w1", "ticker": "HEXA-B.ST", "strategy": "viking_nine", "added": "2026-10-20", "source": "thor",
          "at_add": {"status": "READY", "class": "BEVAKA"}},
         {"id": "w2", "ticker": "NOPE.ST", "strategy": "wolf", "added": "2026-10-21", "levels": {"entry": 10, "stop": 9}},
         {"id": "w3", "ticker": "ABB.ST", "strategy": "momentum", "added": "2026-10-22", "note": "rapport 5/11"}]


# ── Navigationen ─────────────────────────────────────────────────────────────
def test_the_tab_sits_under_portfolio_right_after_positioner():
    opts = nav.options("portfolio")
    assert opts[opts.index("Positioner") + 1] == "👀 Övervakning"
    assert nav.target_slug("watch") == "portfolio/overvakning"
    assert nav.resolve("portfolio/overvakning") == ["portfolio", "👀 Övervakning"]
    cards = dict(nav.HOME_ZONES[[k for k, _t, _c in nav.HOME_ZONES].index("portfolio")][2])
    names = [n for n, _d in nav.HOME_ZONES[[k for k, _t, _c in nav.HOME_ZONES].index("portfolio")][2]]
    assert names[names.index("Positioner") + 1] == "👀 Övervakning" and "nära köpläge" in cards["👀 Övervakning"]
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    row = next(r for r in _PANEL_GUIDE if r[0] == "PORTFÖLJ → 👀 Övervakning")
    assert nav.link_for(row[0]) == "?p=portfolio/overvakning"
    from ui.links import go
    state: dict = {}
    go("watch", state=state)
    assert state == {"nav_goto": "portfolio/overvakning"}


def _tab_portfolio():
    import wolf_panel
    wolf_panel._tab_portfolio()


def test_the_panel_routes_the_tab_outside_the_positions_lock(monkeypatch):
    """Positioner står bakom lösenordet; Övervakningen ritas utan det (listan säger bara vad som bevakas)."""
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    fake_repo(monkeypatch, {"watchlist": {"items": []}})
    at = AppTest.from_function(_tab_portfolio)
    at.session_state["thor_data"] = thor_data()
    at.session_state[nav.STATE_KEY["portfolio"]] = "👀 Övervakning"
    at.run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "👀 ÖVERVAKNING" in text and "innehav är låsta" not in text and not at.text_input
    at.radio(key=nav.STATE_KEY["portfolio"]).set_value("Positioner").run(timeout=60)
    assert "innehav är låsta" in " ".join(m.value for m in at.markdown)          # låset står kvar på Positioner


# ── Rena funktioner ──────────────────────────────────────────────────────────
def test_table_rows_and_missing_reasons():
    from portfolio import watch_ui as wu
    data = thor_data()
    items = wl.normalize({"items": json.loads(json.dumps(ITEMS))})["items"]
    t = wu.table_rows(items, data["rows"], data)
    assert [r["Läge"] for r in t] == ["KÖP", "—", "BEVAKA"]
    assert t[0]["Strategins ord"] == "GOLDEN TICKET · 8/9 · 86" and t[0]["Entry/Stopp"] == "110.00 / 104.00"
    assert t[0]["Vid tillägg"] == "BEVAKA · READY" and t[0]["Körd"] == "2026-10-14" and t[0]["Sedan"] == "2026-10-20"
    assert t[1]["Strategins ord"] == wl.NOT_IN_RUN and t[1]["Entry/Stopp"] == "10.00 / 9.00 (egna)"
    assert t[1]["Vid tillägg"] == "ingen rad" and t[1]["Avstånd"] == "—"
    assert t[2]["Avstånd"] == "rad 5 — setup A saknas" and t[2]["Anteckning"] == "rapport 5/11"
    priced = wu.table_rows(items[:1], data["rows"], data, {"HEXA-B.ST": (115.5, "2026-10-28")})
    assert priced[0]["Kurs"] == "115.50" and priced[0]["Till entry"] == "+5.0 %" and priced[0]["Över stopp"] == "+11.1 %"
    old = thor_data(today="2026-10-30")
    assert wu.table_rows(items[:1], old["rows"], old)[0]["Läge"] == "KÖP · GAMMAL"
    assert wu.table_rows(items[:1], old["rows"], old)[0]["Avstånd"] == "KÖP i en gammal körning — GOLDEN TICKET"
    # varje del en gång: BERSERK:s status och siffra delar setupen, ögonblicksbildens klass står i statusen
    bol = wl.normalize({"items": [{"id": "b1", "ticker": "BOL.ST", "strategy": "berserk", "at_add": {
        "class": "KÖP", "status": "KÖP · S2 Cykelvändning"}}]})["items"]
    (row,) = wu.table_rows(bol, data["rows"], data)
    assert row["Strategins ord"] == "KÖP · S2 Cykelvändning · basmetaller" and row["Vid tillägg"] == "KÖP · S2 Cykelvändning"
    assert wu.joined("KÖP", "KÖP — logga i journalen", "", None) == "KÖP — logga i journalen"
    assert wu.joined("BEVAKA", "READY") == "BEVAKA · READY" and wu.joined("", None) == ""
    assert "#6b7280" in wu.lage_color("KÖP · GAMMAL") and "#2d8a4e" in wu.lage_color("KÖP")
    assert wu.missing_text("wolf", {"errors": {"gist": "nere"}}) == "körningarna gick inte att läsa"
    assert wu.missing_text("wolf", {"arc": {"wolf": {"rows": [], "error": "licens saknas"}}}) == \
        "körningen fallerade (licens saknas)"


def test_near_buy_takes_the_strategies_own_watch_rows_in_list_order():
    from portfolio import watch_ui as wu
    groups = dict(wu.near_buy(thor_data()["rows"]))
    assert all(c.status_class == tc.BEVAKA for g in groups.values() for c in g)
    assert [c.ticker for c in groups["momentum"]] == ["ABB.ST"] and "berserk" in groups and "hav" not in groups
    assert list(groups) == [k for k in __import__("thor.registry", fromlist=["ORDER"]).ORDER if k in groups]
    assert all(len(g) <= 1 for g in dict(wu.near_buy(thor_data()["rows"], top=1)).values())


def test_last_prices_uses_the_latest_close():
    from portfolio import watch_ui as wu
    s = pd.Series([100.0, 115.5], index=pd.to_datetime(["2026-10-27", "2026-10-28"]))
    got = wu.last_prices(["HEXA-B.ST", "NOPE.ST"], closes=lambda t, p: {"HEXA-B.ST": s, "NOPE.ST": pd.Series([], dtype=float)})
    assert got == {"HEXA-B.ST": (115.5, "2026-10-28")}


# ── Sidan ────────────────────────────────────────────────────────────────────
def test_empty_page_says_how_to_add_and_pins_from_near_buy(monkeypatch):
    at, repo = _app(monkeypatch)
    at.run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "ÖVERVAKNING" in text and "Inget bevakas än" in text and "Nära köp nu (automatiskt)" in text
    assert not at.dataframe
    pin = at.button(key="watch_pin_momentum_ABB.ST")
    assert pin.label == "👀" and not pin.disabled
    pin.click().run(timeout=60)
    assert not at.exception
    items = repo["watchlist"]["items"]
    assert [(i["ticker"], i["strategy"], i["source"], i["at_add"]["class"], i["at_add"]["status"]) for i in items] == \
        [("ABB.ST", "momentum", "auto", tc.BEVAKA, "rad 5")]
    assert any("ABB.ST lagd i Övervakning (Momentum Swing)." == t.value for t in at.toast)
    assert at.button(key="watch_pin_momentum_ABB.ST").disabled                     # bevakas: ✓ och avstängd
    assert at.dataframe[0].value["Aktie"].tolist() == ["ABB.ST"]


def test_page_lists_status_opens_thor_and_removes(monkeypatch):
    at, repo = _app(monkeypatch, ITEMS)
    before = json.dumps(repo, sort_keys=True)
    at.run(timeout=60)
    assert not at.exception
    assert json.dumps(repo, sort_keys=True) == before                             # ritningen skriver inget läge
    assert "Du har osparade ändringar" not in " ".join(m.value for m in at.markdown)
    df = at.dataframe[0].value
    assert df["Läge"].tolist() == ["KÖP", "—", "BEVAKA"] and df["Strategi"].tolist() == ["Viking Nine", "Wolf", "Momentum Swing"]
    assert list(df.columns)[:9] == ["Aktie", "Strategi", "Läge", "Strategins ord", "Avstånd", "Entry/Stopp", "Körd",
                                    "Sedan", "Vid tillägg"]
    labels = [b.label for b in at.button]
    assert "HEXA-B.ST → Thor (Viking Nine)" in labels
    at.button(key="watch_open_w1").click().run(timeout=60)
    assert at.session_state["nav_goto"] == "thor" and at.session_state["thor_ticker"] == "HEXA-B.ST"
    assert at.session_state["thor_strategy"] == "Viking Nine"
    at.button(key="watch_del_w2").click().run(timeout=60)
    assert not at.exception
    assert [i["id"] for i in repo["watchlist"]["items"]] == ["w1", "w3"]
    assert at.dataframe[0].value["Aktie"].tolist() == ["HEXA-B.ST", "ABB.ST"]
    assert any("NOPE.ST borttagen" in t.value for t in at.toast)


def _popovers(at) -> dict:
    """{etikett: [(knappnyckel, knappetikett)]} för sidans popovers — AppTest ritar innehållet även när de är stängda."""
    out: dict = {}

    def walk(node):
        for ch in (getattr(node, "children", None) or {}).values():
            if getattr(ch, "type", None) == "popover":
                out[ch.proto.popover.label] = [(b.key, b.label) for b in ch.button]
            walk(ch)
    walk(at.main)
    return out


def test_remove_names_the_stock_and_sits_behind_a_confirmation(monkeypatch):
    """Mobilen staplar raderna: '✕ Ta bort' säger vilken aktie, och tar inte bort något förrän du bekräftat inuti."""
    at, repo = _app(monkeypatch, ITEMS)
    at.run(timeout=60)
    assert not at.exception
    boxes = _popovers(at)
    assert boxes["✕ Ta bort NOPE.ST"] == [("watch_del_w2", "Ja, ta bort NOPE.ST (Wolf)")]
    assert set(boxes) == {"✕ Ta bort HEXA-B.ST", "✕ Ta bort NOPE.ST", "✕ Ta bort ABB.ST"}
    deletes = [b.key for b in at.button if (b.key or "").startswith("watch_del_")]
    assert sorted(deletes) == sorted(k for v in boxes.values() for k, _l in v)      # ingen ta-bort utanför bekräftelsen
    assert not [b for b in at.button if b.label.startswith("✕")]
    at.run(timeout=60)                                                             # att öppna rutan ändrar inget
    assert len(repo["watchlist"]["items"]) == 3
    at.button(key="watch_del_w2").click().run(timeout=60)
    assert [i["id"] for i in repo["watchlist"]["items"]] == ["w1", "w3"] and "✕ Ta bort NOPE.ST" not in _popovers(at)


def test_saved_levels_show_in_the_table_in_the_same_run(monkeypatch):
    """Spara nivåer är en callback: tabellen ovanför visar de nya egna nivåerna direkt, inte först vid nästa klick."""
    at, repo = _app(monkeypatch, ITEMS)
    at.run(timeout=60)
    at.selectbox(key="watch_lv_pick").select("w2").run(timeout=60)
    assert at.dataframe[0].value["Entry/Stopp"].tolist()[1] == "10.00 / 9.00 (egna)"
    at.number_input(key="watch_lv_entry_w2").set_value(12.5)
    at.number_input(key="watch_lv_stop_w2").set_value(11.0)
    at.text_input(key="watch_lv_note_w2").set_value("vänta på rapporten")
    at.button(key="watch_lv_save_w2").click().run(timeout=60)
    assert not at.exception
    df = at.dataframe[0].value
    assert df["Entry/Stopp"].tolist()[1] == "12.50 / 11.00 (egna)" and df["Anteckning"].tolist()[1] == "vänta på rapporten"
    saved = next(i for i in repo["watchlist"]["items"] if i["id"] == "w2")
    assert saved["levels"] == {"entry": 12.5, "stop": 11.0, "target": None}
    assert any(t.value == "NOPE.ST: egna nivåer sparade." for t in at.toast)
    assert at.number_input(key="watch_lv_entry_w2").value == 12.5                    # fälten står kvar


def test_two_rows_with_the_same_id_render_and_remove_separately(monkeypatch):
    """En handskriven fil med samma id två gånger: sidan ritas (inga dubbla nycklar) och ta bort tar rätt rad."""
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    repo = fake_repo(monkeypatch, {"watchlist": {"items": [
        {"id": "same", "ticker": "HEXA-B.ST", "strategy": "viking_nine", "added": "2026-10-20"},
        {"id": "same", "ticker": "ABB.ST", "strategy": "momentum", "added": "2026-10-21"}]}})
    at = AppTest.from_function(_page)
    at.session_state["thor_data"] = thor_data()
    at.run(timeout=60)
    assert not at.exception
    assert at.dataframe[0].value["Aktie"].tolist() == ["HEXA-B.ST", "ABB.ST"]
    opens = [b.key for b in at.button if (b.key or "").startswith("watch_open_")]
    assert len(opens) == len(set(opens)) == 2 and "watch_open_same" in opens
    other = next(k for k in opens if k != "watch_open_same").removeprefix("watch_open_")
    at.button(key=f"watch_del_{other}").click().run(timeout=60)
    assert not at.exception
    assert [(i["id"], i["ticker"]) for i in repo["watchlist"]["items"]] == [("same", "HEXA-B.ST")]


def test_a_stored_file_in_old_form_is_not_unsaved_on_first_render(monkeypatch):
    """Gemener och inget id i filen: läsningen normaliserar, men det är ingen ändring — ingen 'osparade ändringar'
    och inget skrivs. En riktig ändring (✕ Ta bort) sparar den normaliserade formen."""
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    repo = fake_repo(monkeypatch, {"watchlist": {"items": [
        {"ticker": "hexa-b.st", "strategy": "viking_nine", "added": "2026-10-20"},
        {"ticker": "abb.st", "strategy": "momentum", "added": "2026-10-21"}]}})
    at = AppTest.from_function(_page)
    at.session_state["thor_data"] = thor_data()
    at.run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Du har osparade ändringar" not in text and "Inga osparade ändringar" in text
    assert repo["watchlist"]["items"][0]["ticker"] == "hexa-b.st"                    # ritningen skriver inget
    assert at.dataframe[0].value["Aktie"].tolist() == ["HEXA-B.ST", "ABB.ST"]
    at.run(timeout=60)
    assert "Du har osparade ändringar" not in " ".join(m.value for m in at.markdown)
    abb = next(b.key for b in at.button if (b.key or "").startswith("watch_del_") and "ABB.ST" in b.label)
    at.button(key=abb).click().run(timeout=60)
    assert [(i["ticker"], bool(i["id"])) for i in repo["watchlist"]["items"]] == [("HEXA-B.ST", True)]


def test_prices_on_demand_and_own_levels(monkeypatch):
    import market_prices
    at, repo = _app(monkeypatch, ITEMS)
    s = pd.Series([100.0, 115.5], index=pd.to_datetime(["2026-10-27", "2026-10-28"]))
    monkeypatch.setattr(market_prices, "closes", lambda tickers, period="1y": {"HEXA-B.ST": s})
    at.run(timeout=60)
    assert "Kurs" not in at.dataframe[0].value.columns                            # inga kurser utan knapptryck
    at.button(key="watch_prices").click().run(timeout=60)
    df = at.dataframe[0].value
    assert df["Kurs"].tolist()[0] == "115.50" and df["Till entry"].tolist()[0] == "+5.0 %"
    assert "watch_prices" not in repo["watchlist"]                                 # kurser sparas aldrig
    at.selectbox(key="watch_lv_pick").select("w3").run(timeout=60)
    at.number_input(key="watch_lv_entry_w3").set_value(480.0).run(timeout=60)
    at.button(key="watch_lv_save_w3").click().run(timeout=60)
    assert not at.exception
    saved = next(i for i in repo["watchlist"]["items"] if i["id"] == "w3")
    assert saved["levels"] == {"entry": 480.0, "stop": None, "target": None} and saved["note"] == "rapport 5/11"


def test_unreadable_runs_and_stale_rows_are_said_plainly(monkeypatch):
    at, _repo = _app(monkeypatch, ITEMS, data={"rows": [], "errors": {"gist": "nere"}, "loaded": "nu"})
    at.run(timeout=60)
    assert not at.exception
    df = at.dataframe[0].value
    assert set(df["Strategins ord"]) == {"körningarna gick inte att läsa"}
    assert "Gisten gick inte att läsa" in " ".join(m.value for m in at.markdown)
    at2, _ = _app(monkeypatch, ITEMS, data=thor_data(today="2026-10-30"))
    at2.run(timeout=60)
    assert at2.dataframe[0].value["Läge"].tolist()[0] == "KÖP · GAMMAL"


def test_page_reads_thors_data_once_and_shares_it(monkeypatch):
    """Utan Thors data i sessionen läser sidan körningarna själv och lägger dem där Thor läser dem."""
    from streamlit.testing.v1 import AppTest
    from thor import ui as tui
    tu._patch(monkeypatch)
    fake_repo(monkeypatch, {"watchlist": wl.normalize({"items": json.loads(json.dumps(ITEMS[:1]))})})
    calls = []
    monkeypatch.setattr(tui, "load", lambda loader=None: calls.append(1) or dict(tu.DATA))
    at = AppTest.from_function(_page)
    at.run(timeout=60)
    assert not at.exception and len(calls) == 1
    assert at.session_state["thor_data"]["rows"]
    at.run(timeout=60)
    assert len(calls) == 1                                                         # andra ritningen läser inte om
    at.button(key="watch_refresh").click().run(timeout=60)
    assert len(calls) == 2
