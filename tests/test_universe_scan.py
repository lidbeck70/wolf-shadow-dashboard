"""
Skannrarna mot universum-registret — hjälparen universe/scan (urvalet: strategins marknader, teman, huvudlistorna,
kurerade först, taket) och BERSERK: skanningens lista byggs ur listorna plus registrets råvarubolag, theme_of
känner registrets bolag, och utan register är allt som förut. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import berserk_scan as bs  # noqa: E402
import markets as mk  # noqa: E402
from berserk import universe as uv  # noqa: E402
from universe import registry as ur  # noqa: E402
from universe import scan as uscan  # noqa: E402


def _inst(iid, name, ticker, mid, branch=None, ccy="SEK"):
    return {"insId": iid, "name": name, "ticker": ticker, "marketId": mid, "branchId": branch,
            "stockPriceCurrency": ccy}


NORDIC = [_inst(1, "Ericsson B", "ERIC B", 1, 50), _inst(2, "Boliden", "BOL", 1, 20),
          _inst(3, "Equinor", "EQNR", 9, 1, "NOK"), _inst(4, "Oljebolag FN", "OLJA", 4, 2),       # First North (4)
          _inst(5, "Guldgruvan", "GULD", 3, 18)]
GLOBAL = [_inst(10, "Newmont", "NEM", 32, 18, "USD"), _inst(11, "Teck Resources", "TECK B", 35, 20, "CAD"),
          _inst(12, "Junior Gold", "JG", 36, 18, "CAD"), _inst(13, "Kinross", "K", 35, 18, "CAD"),   # TSXV (36)
          _inst(14, "Siemens", "SIE", 39, 30, "EUR")]


def _universe(include_lists=False):
    return ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=GLOBAL, include_lists=include_lists)


def _saved(u):
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return lambda name, fb: saved.get(name, fb)


# ── Urvalet ─────────────────────────────────────────────────────────────────
def test_select_keeps_main_boards_and_themes_in_the_strategy_markets():
    u = _universe()
    got = [m.ticker for m in uscan.select(u, ur.COMMODITY_MARKETS, commodity_only=True)]
    assert got == ["BOL.ST", "EQNR.OL", "GULD.ST", "NEM", "K.TO"]           # marknad för marknad, sedan bokstav
    assert "OLJA.ST" not in got and "JG.V" not in got and "SIE.DE" not in got   # First North, TSX Venture, Tyskland
    assert "ERIC-B.ST" not in got and "TECK-B.TO" not in got                # utan råvarutema
    assert [m.ticker for m in uscan.select(u, ur.COMMODITY_MARKETS, commodity_only=True, main_boards=False)] == [
        "BOL.ST", "EQNR.OL", "GULD.ST", "OLJA.ST", "NEM", "JG.V", "K.TO"]
    assert [m.ticker for m in uscan.select(u, ("norden",))] == ["BOL.ST", "EQNR.OL", "ERIC-B.ST", "GULD.ST"]
    assert [m.ticker for m in uscan.select(u, ur.COMMODITY_MARKETS, themes={"guld"})] == ["GULD.ST", "NEM", "K.TO"]
    assert [m.ticker for m in uscan.select(u, ur.COMMODITY_MARKETS, commodity_only=True, max_members=2)] == [
        "BOL.ST", "EQNR.OL"]
    assert uscan.MAIN_BOARD_IDS == {1, 2, 3, 9, 14, 20, 32, 33, 35, 38}


def test_curated_members_come_first_and_bypass_the_board_rule():
    u = _universe(include_lists=True)
    sel = uscan.select(u, ur.COMMODITY_MARKETS, commodity_only=True, max_members=50)
    assert all(uscan.is_curated(m) for m in sel)                           # kurerade först — taket nås av dem
    assert uscan.is_curated(u.get("BOL.ST")) and not uscan.is_curated(u.get("GULD.ST"))
    allsel = uscan.select(u, ur.COMMODITY_MARKETS, commodity_only=True)
    tickers = [m.ticker for m in allsel]
    assert "SU.TO" in tickers and "GULD.ST" in tickers and tickers.index("SU.TO") < tickers.index("GULD.ST")
    text = uscan.describe(allsel)
    assert text.startswith(f"{len(allsel)} bolag · Norden ") and "kurerade" in text and "bara Börsdata" in text


def test_members_for_reads_the_register_or_returns_none():
    u = _universe()
    loader = _saved(u)
    got = uscan.members_for("berserk", loader=loader)
    assert [m.ticker for m in got] == ["BOL.ST", "EQNR.OL", "GULD.ST", "NEM", "K.TO"]   # råvarumarknader, bara tema
    assert [m.ticker for m in uscan.members_for("insider", loader=loader)] == ["BOL.ST", "EQNR.OL", "ERIC-B.ST",
                                                                               "GULD.ST"]
    assert uscan.tickers_for("berserk", loader=loader, themes={"olja"}) == ["EQNR.OL"]
    assert uscan.members_for("berserk", loader=lambda name, fb: fb) is None     # registret saknas
    assert uscan.members_for("berserk", universe=ur.Universe()) is None


# ── BERSERK ─────────────────────────────────────────────────────────────────
def test_berserk_universe_adds_register_members_on_top_of_the_lists(monkeypatch):
    monkeypatch.setattr(uv, "REGISTRY", {})
    base = bs.lists()
    assert "BOL.ST" in base and "XLE" in base and uv.theme_of("GULD.ST") == ""
    got = bs.universe(loader=_saved(_universe()))
    assert got[:len(base)] == base and got[len(base):] == ["GULD.ST", "K.TO"]       # BOL, EQNR, NEM finns i listorna
    assert uv.theme_of("GULD.ST") == "guld" and uv.theme_of("K.TO") == "guld" and uv.kind_of("K.TO") == "producent"
    assert set(uv.REGISTRY) == {"GULD.ST", "K.TO"} and uv.theme_of("EQNR.OL") == "olja"
    assert uv.register({"GULD.ST": "guld", "X.ST": "okänt", "BOL.ST": "koppar"}) == 0  # redan känd / okänt tema
    assert "X.ST" not in uv.REGISTRY and "BOL.ST" not in uv.REGISTRY
    assert uv.by_theme(["GULD.ST", "BOL.ST"]) == {"guld": ["GULD.ST"], "koppar": ["BOL.ST"]}


def test_berserk_universe_without_register_is_the_lists(monkeypatch):
    monkeypatch.setattr(uv, "REGISTRY", {})
    assert bs.universe(loader=lambda name, fb: fb) == bs.lists()
    monkeypatch.setattr(uscan, "members_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nere")))
    assert bs.universe(loader=lambda name, fb: fb) == bs.lists() and uv.REGISTRY == {}


def test_berserk_extra_is_capped(monkeypatch):
    monkeypatch.setattr(uv, "REGISTRY", {})
    monkeypatch.setattr(bs, "EXTRA_MAX", 1)
    got = bs.universe(loader=_saved(_universe()))
    assert got[len(bs.lists()):] == ["GULD.ST"] and set(uv.REGISTRY) == {"GULD.ST"}
