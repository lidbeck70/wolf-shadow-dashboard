"""
PR 7 av panelgenomgången — "Global"-presetet och CAGR-taket.

Contrarian-motorns include_global fanns i konfigurationen men lästes aldrig:
"Global" i panelen körde exakt samma skanning som "Norden". Nu läggs
/instruments/global ovanpå Norden, nyckeltalen för de raderna hämtas ur den
globala KPI-screenern (den nordiska ger inget för globala id:n) och
statistiken visar hur många globala rader som kom med. Samma screener-fel
fanns i sheets_refresh för Tiggre/Durrett-raderna på TSX/ASX.

CAGR-fliken kapade stora universum till de 200 första i bokstavsordning —
"Alla marknader" blev A–D — och kapade FÖRE landsfiltret.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import markets as mk                                   # noqa: E402
import contrarian_alpha.engine as eng                  # noqa: E402


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── Fejk-API med Norden + global lista ───────────────────────────────────────
class _Api:
    is_configured = True

    def __init__(self, global_rows=None):
        self.calls = []
        self._global = global_rows if global_rows is not None else [
            {"insId": 910001, "ticker": "FCX", "name": "Freeport", "marketId": 32,
             "instrumentType": 1, "branchId": 10, "sectorId": 3},
            {"insId": 910002, "ticker": "CCO", "name": "Cameco", "marketId": 35,
             "instrumentType": 1, "branchId": 10, "sectorId": 3},
            {"insId": 910003, "ticker": "SPX", "name": "S&P 500", "marketId": 7,
             "instrumentType": 1},                      # indexlista → bort
            {"insId": 910004, "ticker": "XYZ", "name": "Okänd", "marketId": 999,
             "instrumentType": 1},                      # okänd marknad → bort
            {"insId": 910005, "ticker": "ETF1", "name": "ETF", "marketId": 32,
             "instrumentType": 8},                      # ej aktie → bort
        ]

    def get_instruments(self):
        return [{"insId": 5, "ticker": "BOL", "name": "Boliden", "instrumentType": 1,
                 "marketId": 1, "branchId": 10, "sectorId": 3}]

    def get_global_instruments_list(self):
        self.calls.append("global")
        return list(self._global)

    def get_branches(self):
        return [{"id": 10, "name": "Gruvor"}]

    def get_sectors(self):
        return [{"id": 3, "name": "Råvaror"}]

    def get_markets(self):
        return []

    def get_countries(self):
        return []


def setup_function(_f):
    mk.reset()


def teardown_function(_f):
    mk.reset()


# ── Universum ────────────────────────────────────────────────────────────────
def test_global_preset_adds_global_stock_lists_on_top_of_nordic():
    api = _Api()
    cfg = eng.PipelineConfig(market_ids=[1], include_global=True)
    uni = eng._build_universe(cfg, api)
    by = {u["ticker"]: u for u in uni}
    assert by["BOL.ST"]["ins_id"] == 5 and "scope" not in by["BOL.ST"]
    assert by["FCX"]["ins_id"] == 910001 and by["FCX"]["scope"] == "global"
    assert by["CCO.TO"]["scope"] == "global"                     # suffix ur marknadstabellen
    assert by["FCX"]["sector_name"] == "Råvaror"                 # samma metadata som Norden
    assert not {"SPX", "XYZ", "ETF1"} & set(by)                  # index/okänt/ej aktie
    assert api.calls.count("global") == 1                        # EN hämtning per bygge


def test_norden_preset_is_unchanged_and_never_touches_the_global_list():
    api = _Api()
    uni = eng._build_universe(eng.PipelineConfig(market_ids=[1]), api)
    assert [u["ticker"] for u in uni] == ["BOL.ST"]
    assert api.calls == []
    assert eng.PipelineConfig().include_global is False


def test_missing_global_licence_falls_back_to_nordic_only():
    api = _Api(global_rows=[])
    uni = eng._build_universe(eng.PipelineConfig(market_ids=[1], include_global=True), api)
    assert [u["ticker"] for u in uni] == ["BOL.ST"]


def test_manual_ticker_resolves_against_the_global_list_when_enabled():
    cfg = eng.PipelineConfig(market_ids=[], include_global=True, manual_tickers=["FCX", "BOL"])
    by = {u["ticker"]: u for u in eng._build_universe(cfg, _Api())}
    assert by["FCX"]["ins_id"] == 910001 and by["FCX"]["scope"] == "global"
    assert by["BOL.ST"]["ins_id"] == 5 and "scope" not in by["BOL.ST"]
    # utan flaggan är FCX fortfarande en suffixlös yfinance-rad
    cfg2 = eng.PipelineConfig(market_ids=[], manual_tickers=["FCX"])
    assert eng._build_universe(cfg2, _Api())[0]["ins_id"] is None


# ── Nyckeltal ur rätt screener ───────────────────────────────────────────────
def test_batch_fundamentals_route_global_ids_to_the_global_screener():
    class _Fund:
        is_configured = True

        def __init__(self):
            self.calls = []

        def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
            self.calls.append((tuple(ids), scope))
            return {i: {"ins_id": i, "roic": 0.2} for i in ids}

    api = _Fund()
    out = eng._batch_fetch_fundamentals([920001, 920002, 920003], api, global_ids={920002})
    assert sorted(api.calls) == [((920001, 920003), "nordic"), ((920002,), "global")]
    assert out[920002]["roic"] == 0.2 and out[920001]["roic"] == 0.2

    # cachat under egen nyckel — andra varvet frågar inte API:t igen
    api2 = _Fund()
    eng._batch_fetch_fundamentals([920002], api2, global_ids={920002})
    assert api2.calls == []


def test_api_scope_global_reads_the_global_kpi_endpoint(monkeypatch):
    import borsdata_api as bd
    api = bd.BorsdataAPI(api_key="x")
    paths = []

    def _get(path, **kw):
        paths.append(path)
        return {"values": [{"i": 77, "n": 12.0}]}

    monkeypatch.setattr(api, "_get", _get)
    snap = api.get_fundamentals_snapshot_fast([77], scope="global")
    assert paths and all(p.startswith("/instruments/global/kpis/") for p in paths)
    assert snap[77]["roe"] == 0.12                              # samma divisor som Norden
    paths.clear()
    api.get_fundamentals_snapshot_fast([77])
    assert paths and all(p.startswith("/instruments/kpis/") for p in paths)


def test_sheets_refresh_fetches_global_rows_from_the_global_screener():
    src = _src("sheets_refresh.py")
    assert 'get_fundamentals_snapshot_fast(missing_meta, scope="global")' in src


# ── Panelen ──────────────────────────────────────────────────────────────────
def test_pipeline_result_reports_global_count_and_ui_shows_it():
    assert eng.PipelineResult.__dataclass_fields__["global_count"].default == 0
    src = _src("contrarian_alpha/ui.py")
    assert '"Globalt"' in src and "Pro+ global" in src
    from contrarian_alpha.ui import _MARKETS
    assert _MARKETS["Global"]["include_global"] is True
    assert _MARKETS["Norden"]["include_global"] is False


# ── CAGR-taket ───────────────────────────────────────────────────────────────
def test_cagr_cap_keeps_curated_names_not_the_alphabet():
    from cagr.cagr_loader import cap_tickers
    meta = {t: {"country": "USA"} for t in ("AAA", "AAB", "AAC", "ZZZ", "FCX", "NEM")}
    out = cap_tickers(meta, 3, priority=["NEM", "FCX"])
    assert list(out) == ["NEM", "FCX", "AAA"]
    assert cap_tickers(meta, 10) == meta                        # under taket: orört
    # standardprioriteten är det nordiska registret + de kurerade råvarulistorna
    big = {f"T{i}": {} for i in range(300)}
    big["BOL.ST"] = {"country": "Sweden"}
    assert "BOL.ST" in cap_tickers(big, 200)


def test_cagr_filters_country_before_capping():
    src = _src("cagr/cagr_streamlit.py")
    assert "sorted(tickers_meta.keys())[:MAX_ALPHA_TICKERS]" not in src
    assert src.index('country_choice != "All" and market_choice != "UCITS ETFs"') \
        < src.index("cap_tickers(tickers_meta, MAX_ALPHA_TICKERS)")
