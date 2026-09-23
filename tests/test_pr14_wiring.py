"""
PR 14 — koppla det som redan hämtas till arken.

Fyra kopplingar, inga nya datakällor utom Yahoo-terminerna för råvarupris:
EV och operativt kassaflöde saknades i snabb-snapshoten (Durrett-arkets
förslag kom aldrig), Tiggre-kandidatens börsvärde räknades men visades bara
för positionerna, Rick Rule-arkets råvarupris skrevs för hand fast panelen
hämtar terminspriser, och Lukacs "aktuell kurs" var inte kopplad till
sifferuppdateringen.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── 1. EV och OCF i snabb-snapshoten ─────────────────────────────────────────
def test_fast_snapshot_fetches_ev_and_operating_cash_flow(monkeypatch):
    import borsdata_api as bd
    api = bd.BorsdataAPI(api_key="x")
    seen = []

    def _get(path, **kw):
        seen.append(path)
        kid = int(path.split("/")[3])
        return {"values": [{"i": 7, "n": float(kid)}]}

    monkeypatch.setattr(api, "_get", _get)
    snap = api.get_fundamentals_snapshot_fast([7])[7]
    assert snap["ev"] == bd.KPI["ev"] and snap["ocf_m"] == bd.KPI["ocf_m"]
    assert f"/instruments/kpis/{bd.KPI['ev']}/last/latest" in seen
    # sheets_refresh läser just de nycklarna för Durrett-arket
    assert '("ev", "ev_musd", fx)' in _src("sheets_refresh.py")
    assert '("ocf_m", "ocf_musd", rfx)' in _src("sheets_refresh.py")


# ── 2. Råvarupriset ──────────────────────────────────────────────────────────
def test_commodity_tickers_cover_yahoo_futures_and_nothing_else():
    import commodity_prices as cp
    assert cp.ticker_for("Guld") == ("GC=F", "USD/oz")
    assert cp.ticker_for("koppar") == ("HG=F", "USD/lb")
    assert cp.ticker_for("Olja") == ("CL=F", "USD/fat")
    assert cp.ticker_for("Uran") is None and cp.ticker_for("Litium") is None   # inte på Yahoo
    from rotation import COMMODITIES
    for c in COMMODITIES:                     # varje råvara är antingen kopplad eller medvetet inte
        assert cp.ticker_for(c.name) is not None or c.key in cp.NOT_ON_YAHOO, c.name


def test_spot_reads_the_last_close_once_per_commodity(monkeypatch):
    import commodity_prices as cp
    import pandas as pd
    calls = []

    class _T:
        def __init__(self, ticker):
            calls.append(ticker)

        def history(self, period="5d", auto_adjust=True):
            return pd.DataFrame({"Close": [4.1, 4.25]},
                                index=pd.to_datetime(["2026-09-19", "2026-09-22"]))

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_T))
    out = cp.spot_many(["Koppar", "koppar", "Uran", "Guld", None])
    assert set(out) == {"koppar", "guld"} and calls.count("HG=F") == 1
    assert out["koppar"] == {"price": 4.25, "unit": "USD/lb", "asof": "2026-09-22", "ticker": "HG=F"}


def test_refresh_job_puts_the_commodity_price_on_producer_rows(monkeypatch):
    import sheets_refresh as sr
    import commodity_prices as cp
    from test_sheets_refresh import _API, _sheets
    monkeypatch.setattr(cp, "spot_many", lambda names: {
        "koppar": {"price": 4.25, "unit": "USD/lb", "asof": "2026-09-22", "ticker": "HG=F"}})
    sheets = _sheets()
    sheets["producers"]["producers"][0]["commodity"] = "Koppar"
    out = sr.refresh(_API(), sheets)
    row = out["rows"]["producers:r1"]
    assert row["commodity_price"] == 4.25 and row["commodity_unit"] == "USD/lb"
    assert "commodity_price" not in out["rows"]["producers:y1"]            # royalty-raden: inget pris
    # och arket föreslår det på fältet "Råvarupris nu"
    assert '_suggest("producers", row, "price", "commodity_price"' in _src("producers.py")


# ── 3. Förslagsmekaniken: underdict och enhet ────────────────────────────────
def test_suggestion_supports_row_id_and_unit_field():
    import refresh_ui as rui
    blob = {"rows": {"producers:r1": {"price": 12.5, "currency": "CAD", "asof": "2026-09-22",
                                      "commodity_price": 4.25, "commodity_unit": "USD/lb"}}}
    assert rui.suggestion(blob, "producers", {"id": "r1"}, "price") == (12.5, "2026-09-22", "CAD")
    assert rui.suggestion(blob, "producers", {}, "price", row_id="r1") == (12.5, "2026-09-22", "CAD")
    assert rui.suggestion(blob, "producers", {"id": "r1"}, "commodity_price",
                          unit_field="commodity_unit") == (4.25, "2026-09-22", "USD/lb")
    assert rui.suggestion(blob, "producers", {"id": "zzz"}, "price") is None


# ── 4. Tiggre-kandidaten och Lukacs ──────────────────────────────────────────
def test_tiggre_candidate_and_lukacs_price_are_wired():
    assert '_suggest("tiggre", cand, "mcap", "mcap_musd"' in _src("tiggre.py")
    import inspect
    import lukacs_ui, controls_ui
    assert "sheet" in inspect.signature(lukacs_ui.render_fv).parameters
    assert "sheet" in inspect.signature(controls_ui.render_csm).parameters
    assert controls_ui.SHEET_BY_STRATEGY["producenter"] == "producers"
    assert 'refresh_ui.suggest(sheet, row, "aktuell_kurs", "price"' in _src("lukacs_ui.py")
