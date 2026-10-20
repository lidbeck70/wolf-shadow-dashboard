"""
Universum-registret — marknaderna, medlemmarna ur Börsdata-instrument (fejkade) och panelens egna listor,
sammanslagningen, bloben (index + fil per marknad), förvalet per strategi, jobbet, arbetsflödet, EMBER-rättningen
och Thor-kopplingen (marknadsfiltret, momentumradernas suffix, registret som reserv). Inget nätverk.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import markets as mk  # noqa: E402
import universe_scan as us  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import ui as tui  # noqa: E402
from universe import registry as ur  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORDIC = [{"insId": 1, "name": "Ericsson B", "ticker": "ERIC B", "marketId": 1, "sectorId": 6, "branchId": 50,
           "instrumentType": 0, "stockPriceCurrency": "SEK"},
          {"insId": 2, "name": "OMXS30", "ticker": "OMXS30", "marketId": 7},                       # index — hoppas
          {"insId": 3, "name": "Boliden", "ticker": "BOL", "marketId": 1, "sectorId": 7, "branchId": 20,
           "stockPriceCurrency": "SEK"},
          {"insId": 4, "name": "Equinor", "ticker": "EQNR", "marketId": 9, "sectorId": 3, "branchId": 1,
           "stockPriceCurrency": "NOK"},
          "skräp", {"name": "utan ticker", "marketId": 1}]
GLOBAL = [{"insId": 10, "name": "Teck Resources", "ticker": "TECK B", "marketId": 35, "sectorId": 7, "branchId": 20,
           "stockPriceCurrency": "CAD"},
          {"insId": 11, "name": "OTC Co", "ticker": "OTCX", "marketId": 34},                        # OTC — hoppas
          {"insId": 12, "name": "Siemens", "ticker": "SIE", "marketId": 39},                        # Tyskland — utanför
          {"insId": 13, "name": "Newmont", "ticker": "NEM", "marketId": 32, "sectorId": 7, "branchId": 18,
           "stockPriceCurrency": "USD"}]


# ── Marknaderna ─────────────────────────────────────────────────────────────
def test_markets_and_suffix_rules():
    assert [m.key for m in ur.MARKETS] == ["norden", "usa", "kanada", "london", "australien"]
    assert ur.DEFAULT == ("norden",) and ur.COMMODITY_MARKETS == ("norden", "usa", "kanada")
    assert ur.BY_KEY["norden"].index == "OMXS30" and ur.BY_KEY["kanada"].index == "^GSPTSE"
    for t, key in (("BOL.ST", "norden"), ("EQNR.OL", "norden"), ("NOVO-B.CO", "norden"), ("NOKIA.HE", "norden"),
                   ("NEM", "usa"), ("AEM.TO", "kanada"), ("IPT.V", "kanada"), ("X.CN", "kanada"), ("RIO.L", "london"),
                   ("BHP.AX", "australien"), ("SIE.DE", ""), ("", "")):
        assert ur.market_of(t) == key, t
    assert ur.market_for_country("Sverige") == "norden" and ur.market_for_country("canada") == "kanada"
    assert ur.market_for_country("Tyskland") is None and ur.market_for_country(None) is None
    assert ur.label("usa") == "USA" and ur.label("x") == "x"


def test_strategy_defaults_follow_the_family():
    assert ur.markets_for("berserk") == ur.COMMODITY_MARKETS and ur.markets_for("rule") == ur.COMMODITY_MARKETS
    assert ur.markets_for("hav") == ur.COMMODITY_MARKETS and ur.markets_for("ember") == ur.COMMODITY_MARKETS
    for key in ("viking_nine", "wolf", "momentum", "insider", "long", "quality", "deep", "okänd"):
        assert ur.markets_for(key) == ur.DEFAULT, key
    rows = {r["Strategi"]: r["Marknader"] for r in ur.strategy_rows()}
    assert rows["BERSERK"] == "Norden · USA · Kanada" and rows["Viking Nine"] == "Norden"


# ── Medlemmarna ─────────────────────────────────────────────────────────────
def test_from_instruments_keeps_stock_lists_in_the_register_markets():
    mem = {m.ticker: m for m in ur.from_instruments(NORDIC + GLOBAL, mk.FALLBACK, "Börsdata")}
    assert set(mem) == {"ERIC-B.ST", "BOL.ST", "EQNR.OL", "TECK-B.TO", "NEM"}             # index, OTC, .DE borta
    e = mem["ERIC-B.ST"]
    assert (e.name, e.market, e.country, e.bd_ticker, e.ins_id, e.market_id) == ("Ericsson B", "norden", "Sverige",
                                                                                 "ERIC B", 1, 1)
    assert e.sector_id == 6 and e.branch_id == 50 and e.instrument_type == 0 and e.currency == "SEK"
    assert mem["BOL.ST"].theme == "koppar" and mem["EQNR.OL"].theme == "olja"                 # BERSERK-listan
    assert mem["NEM"].theme == "guld" and mem["TECK-B.TO"].market == "kanada" and mem["TECK-B.TO"].currency == "CAD"
    assert all(m.sources == ("Börsdata",) and m.kind == ur.STOCK for m in mem.values())


def test_from_lists_reads_the_panels_own_lists_without_dead_tickers():
    mem = ur.merge(ur.from_lists())
    assert len(mem.members) >= 400 and set(mem.counts()) == set(ur.BY_KEY)
    assert all(m.market in ur.BY_KEY for m in mem.members.values())
    bol, nem, su = mem.get("BOL.ST"), mem.get("NEM"), mem.get("SU.TO")
    assert bol.market == "norden" and bol.theme == "koppar" and "BERSERK" in bol.sources and bol.name == "Boliden"
    assert nem.market == "usa" and nem.theme == "guld" and "Contrarian Alpha US/CA" in nem.sources
    assert su.market == "kanada" and su.theme == "olja"
    assert mem.get("GOLD") is None and mem.get("B") is not None                              # omdöpt (dead_tickers)
    assert mem.get("XLE").kind == ur.ETF and mem.get("ERIC-B.ST").kind == ur.STOCK
    assert {"BERSERK", "Viking Nine-backtestet", "Wolf-screenern", "EMBER", "ticker_universe",
            "Contrarian Alpha US/CA", "Heatmap", "Long Screener"} <= set(mem.stats)


def test_build_merges_borsdata_first_then_lists_and_notes_a_missing_global_licence():
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=[], include_lists=True)
    bol = u.get("BOL.ST")
    assert bol.sources[0] == "Börsdata" and "BERSERK" in bol.sources and bol.ins_id == 3 and bol.theme == "koppar"
    assert u.get("NEM").sources[0] != "Börsdata" and u.notes and "global tom" in u.notes[0]
    full = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=GLOBAL, include_lists=False)
    assert full.notes == [] and set(full.members) == {"ERIC-B.ST", "BOL.ST", "EQNR.OL", "TECK-B.TO", "NEM"}
    assert full.stats == {"Börsdata": 3, "Börsdata global": 2} and full.counts()["usa"] == 1
    assert full.tickers(["norden"]) == ["BOL.ST", "EQNR.OL", "ERIC-B.ST"]
    assert full.tickers(themes=["koppar"]) == ["BOL.ST"] and full.tickers(commodity_only=True) == ["BOL.ST", "EQNR.OL",
                                                                                                   "NEM"]
    assert full.for_strategy("berserk") == ["BOL.ST", "EQNR.OL", "NEM"] and full.for_strategy("insider") == [
        "BOL.ST", "EQNR.OL", "ERIC-B.ST"]
    assert full.bd_map(["norden"]) == {"BOL": "BOL.ST", "EQNR": "EQNR.OL", "ERIC B": "ERIC-B.ST"}


def test_blob_roundtrip_save_and_load():
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=GLOBAL, include_lists=False)
    idx = ur.to_index(u)
    assert idx["members"] == 5 and idx["counts"]["norden"] == 3 and idx["bd_map"]["ERIC B"] == "ERIC-B.ST"
    assert [m["key"] for m in idx["markets"]] == [m.key for m in ur.MARKETS]
    rows = ur.to_rows(u, "norden")
    assert rows["columns"] == list(ur.COLUMNS) and len(rows["rows"]) == 3
    back = ur.from_rows([rows, ur.to_rows(u, "usa")], idx)
    assert set(back.members) == {"BOL.ST", "EQNR.OL", "ERIC-B.ST", "NEM"} and back.generated == u.generated
    assert back.get("ERIC-B.ST").ins_id == 1 and back.get("ERIC-B.ST").sources == ("Börsdata",)
    saved = {}
    out = ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    assert set(out) == {"universe.json"} | {f"universe_{m.key}.json" for m in ur.MARKETS} and all(out.values())
    loaded = ur.load(loader=lambda name, fb: saved.get(name, fb))
    assert set(loaded.members) == set(u.members) and loaded.stats == u.stats
    assert ur.load(["norden"], loader=lambda name, fb: saved.get(name, fb)).counts()["usa"] == 0
    assert ur.load(loader=lambda name, fb: fb) is None and ur.load_index(lambda n, fb: fb) is None
    assert [r["Marknad"] for r in ur.summary_rows(idx)] == ["Norden", "USA", "Kanada", "London", "Australien"]
    assert ur.summary_rows(idx)[0]["Bolag"] == 3 and ur.summary_rows(None)[0]["Bolag"] == 0


# ── Jobbet ──────────────────────────────────────────────────────────────────
def test_runner_builds_logs_and_saves():
    saved = {}
    out = us.run(nordic=NORDIC, global_=GLOBAL, table=mk.FALLBACK, save=lambda n, d: saved.setdefault(n, d) is not None)
    assert out["members"] >= 5 and out["counts"]["norden"] >= 3 and "Börsdata" in out["stats"]
    assert len(out["saved"]) == 1 + len(ur.MARKETS) and all(out["saved"].values())
    assert "universe_norden.json" in saved and saved["universe.json"]["bd_map"]["EQNR"] == "EQNR.OL"
    quiet = us.run(nordic=[], global_=[], table=mk.FALLBACK, no_save=True)
    assert quiet["saved"] == {} and quiet["notes"] and quiet["members"] >= 400          # listorna räcker


def test_workflow_and_ember_source_fix():
    src = open(os.path.join(ROOT, ".github", "workflows", "universe-scan.yml"), encoding="utf-8").read()
    assert "workflow_dispatch" in src and "python universe_scan.py" in src and "BORSDATA_API_KEY" in src
    assert "schedule" in src and "GIST_TOKEN" in src
    scan = open(os.path.join(ROOT, "scheduled_scan.py"), encoding="utf-8").read()
    from ember import universe as eu
    assert ("universe_source=SOURCE_BOTH" in scan or "universe_source=SOURCE_REGISTRY" in scan)
    assert 'universe_source="both"' not in scan
    assert eu.SOURCE_BOTH == "Båda" and eu.build_universe("both")[0] == []        # 'both' var ett okänt val


# ── Thor ────────────────────────────────────────────────────────────────────
def test_thor_reads_the_index_filters_by_market_and_maps_momentum_tickers():
    import test_thor_candidates as fx
    idx = {"generated": "2026-10-15T05:17:00+00:00", "members": 5, "counts": {"norden": 3}, "stats": {"Börsdata": 3},
           "notes": [], "bd_map": {"EQNR": "EQNR.OL", "ERIC B": "ERIC-B.ST"}}
    calls = []

    def loader(name, fb):
        calls.append(name)
        return idx if name == "universe.json" else fb
    data = tui.load(loader)
    assert data["universe"] is idx and "universe.json" in calls
    rows = tc.build({**fx.BLOBS, "momentum": {"generated": "2026-10-14", "top": [
        {"rank": 1, "ticker": "EQNR", "name": "Equinor", "price": 300.0, "setupA": True},
        {"rank": 2, "ticker": "ERIC B", "name": "Ericsson", "price": 80.0}]}}, momentum_map=idx["bd_map"])
    mom = {c.ticker for c in rows if c.strategy == "momentum"}
    assert "EQNR.OL" in mom and "ERIC-B.ST" in mom and "EQNR.ST" not in mom
    assert tui.MARKET_LABELS == {"Norden": "norden", "USA": "usa", "Kanada": "kanada", "London": "london",
                                 "Australien": "australien"}
    shown = tui.shown_rows(rows, tui.ALL, [], ["USA"])
    assert shown and all(ur.market_of(c.ticker) == "usa" for c in shown)
    assert len(tui.shown_rows(rows, tui.ALL, [], [])) == len(rows)
    nordic = tui.shown_rows(rows, tui.ALL, [], ["Norden", "Kanada"])
    assert nordic and all(ur.market_of(c.ticker) in ("norden", "kanada") for c in nordic)


def test_universe_member_and_fill_from_universe():
    from thor.ticker import identify
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=GLOBAL, include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)

    class _St:
        session_state: dict = {}
    st = _St()
    calls = []

    def loader(name, fb):
        calls.append(name)
        return saved.get(name, fb)
    m = tui.universe_member(st, "TECK-B.TO", loader)
    assert m.name == "Teck Resources" and calls == ["universe.json", "universe_kanada.json"]
    assert tui.universe_member(st, "AEM.TO", loader) is None and len(calls) == 2           # marknaden i sessionen
    assert tui.universe_member(st, "SIE.DE", loader) is None and len(calls) == 2
    ident = identify("TECK-B.TO")
    assert ident.name == "" and tui.fill_from_universe(ident, m) and ident.name == "Teck Resources"
    assert ident.currency == "CAD" and ident.theme is None                                  # inget tema i registret
    nem = identify("NEM")
    assert tui.fill_from_universe(nem, u.get("NEM")) and nem.name == "Newmont"
    assert nem.theme == "guld" and nem.theme_source == "BERSERK-listan"                     # listans tema vinner
    newco = {"insId": 14, "name": "New Gold Co", "ticker": "NEWCO", "marketId": 32, "branchId": 18,
             "stockPriceCurrency": "USD"}
    u2 = ur.build(table=mk.FALLBACK, nordic=[], global_=[newco], include_lists=False)
    fresh = identify("NEWCO")
    assert fresh.theme is None and tui.fill_from_universe(fresh, u2.get("NEWCO"))
    assert fresh.theme == "guld" and fresh.theme_source == "universum-registret" and fresh.theme_label == "Guld"
    assert not tui.fill_from_universe(fresh, None)


def _script():
    from thor import ui as t
    t.render_thor_page()


def test_page_shows_the_universe_box_and_market_pills(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_thor_ui import DATA, _patch
    _patch(monkeypatch)
    idx = {"generated": "2026-10-15T05:17:00+00:00", "members": 559, "counts": {"norden": 170, "usa": 284},
           "stats": {"BERSERK": 232}, "notes": ["Börsdata global tom"], "bd_map": {}}
    monkeypatch.setattr(tui, "load", lambda loader=None: {**DATA, "universe": idx})
    at = AppTest.from_function(_script)
    at.run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "559 bolag" in text and "FÖRVALET PER STRATEGI" in text and "Börsdata global tom" in " ".join(
        m.value for m in at.markdown)
    assert any("Universum-registret" in e.label for e in at.expander) and len(at.dataframe) == 3   # + körningar per strategi
    monkeypatch.setattr(tui, "load", lambda loader=None: {**DATA, "universe": None})
    at = AppTest.from_function(_script)
    at.run(timeout=60)
    assert not at.exception and "har inte byggts än" in " ".join(m.value for m in at.markdown)
