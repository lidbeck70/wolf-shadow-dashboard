"""
Contrarian Alpha: "USA & Kanada" ur Börsdatas globala lista, storleksfilter,
reserv till den kurerade listan, täckning per fält, "saknas" mot "ej
meningsfullt" och Yahoo-fyllning av valutafria nyckeltal. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha import engine as eng  # noqa: E402

US_CA = [32, 33, 35, 36, 37]
_TABLE = eng._markets.current()


class _Api:
    is_configured = True

    def __init__(self, glob):
        self._glob = glob

    def get_instruments(self):
        return [{"insId": 1, "ticker": "BOL", "name": "Boliden", "marketId": 1, "branchId": 17, "sectorId": 7}]

    def get_global_instruments_list(self):
        return self._glob

    def get_branches(self):
        return [{"id": 17, "name": "Gruv - Industrimetaller"}, {"id": 18, "name": "Gruv - Guld & Silver"},
                {"id": 94, "name": "Telekomtjänster"}]

    def get_sectors(self):
        return [{"id": 7, "name": "Material"}, {"id": 9, "name": "Telekom"}]


def _g(iid, tk, mid, bid, ccy):
    return {"insId": iid, "ticker": tk, "name": tk, "marketId": mid, "branchId": bid, "sectorId": 7,
            "instrumentType": 1, "stockPriceCurrency": ccy}


GLOB = [_g(101, "NEM", 32, 18, "USD"), _g(102, "AEM", 35, 18, "CAD"), _g(103, "TINY", 36, 18, "CAD"),
        _g(104, "OTCX", 34, 18, "USD"), _g(105, "VZ", 32, 94, "USD"), _g(106, "RIO", 2000, 17, "GBP")]


def test_global_rows_are_limited_to_us_canada_markets(monkeypatch):
    monkeypatch.setattr(eng._markets, "load", lambda api=None, **kw: _TABLE)
    cfg = eng.PipelineConfig(market_ids=[], include_global=True, global_market_ids=US_CA)
    uni = eng._build_universe(cfg, _Api(GLOB))
    tickers = {u["ticker"] for u in uni}
    assert {"NEM", "AEM.TO", "TINY.V", "VZ"} <= tickers
    assert not any(t.startswith("OTCX") or t.startswith("RIO") for t in tickers)   # OTC och London ute
    assert not any(t.startswith("BOL") for t in tickers)                           # Norden av
    assert all(u.get("scope") == "global" for u in uni)
    nem = next(u for u in uni if u["ticker"] == "NEM")
    assert nem["branch_name"] == "Gruv - Guld & Silver"


def test_market_cap_in_usd():
    assert eng.market_cap_musd({"market_cap": 1000.0}, {"marketId": 35}) == 730.0          # CAD via marknad
    assert eng.market_cap_musd({"market_cap": 1000.0}, {"stockPriceCurrency": "USD"}) == 1000.0
    assert eng.market_cap_musd({}, {"marketId": 32}) is None


def _run(monkeypatch, glob, **cfg_kw):
    monkeypatch.setattr(eng, "_BORSDATA_AVAILABLE", True)
    monkeypatch.setattr(eng, "BorsdataAPI", lambda: _Api(glob), raising=False)
    monkeypatch.setattr(eng._markets, "load", lambda api=None, **kw: _TABLE)
    snaps = {101: {"market_cap": 50000.0, "fcf_m": 900.0, "ebitda_margin": 0.4, "roic": 0.12},
             102: {"market_cap": 40000.0, "fcf_m": -10.0, "ebitda_margin": -0.1},
             103: {"market_cap": 20.0}}
    monkeypatch.setattr(eng, "_batch_fetch_fundamentals",
                        lambda ids, api, global_ids=None: {i: dict(snaps.get(i, {})) for i in ids})
    monkeypatch.setattr(eng, "_fetch_price_df", lambda t, i, api: None)
    monkeypatch.setattr(eng, "_batch_valuation_data", lambda scan, s, api: {})
    seen = []

    def fake_single(ticker, ins_id, inst_info, fund_snap, price_df, branch_name, sector_name, config, api, **kw):
        seen.append(ticker)
        return eng.ContrairianAlphaResult(ticker=ticker, ins_id=ins_id, name=ticker, market="US",
                                          sector=sector_name, branch=branch_name, composite_score=60.0)
    monkeypatch.setattr(eng, "_run_single_ticker", fake_single)
    cfg = eng.PipelineConfig(mode="deep_contrarian", market_ids=[], include_global=True,
                             global_market_ids=US_CA, max_fund_workers=1, max_price_workers=1, **cfg_kw)
    return eng.run_pipeline(cfg), seen


def test_us_canada_pipeline_commodity_and_size(monkeypatch):
    res, seen = _run(monkeypatch, GLOB, min_market_cap_musd=50.0, static_fallback=True)
    assert sorted(seen) == ["AEM.TO", "NEM"]                     # VZ ej råvara, TINY för liten
    out = {r.ticker: r for r in res.eliminated}
    assert out["VZ"].elimination_stage == "COMMODITY"
    assert out["TINY.V"].elimination_stage == "SIZE" and "14.6" not in out["TINY.V"].elimination_reason
    assert "< 50" in out["TINY.V"].elimination_reason
    assert res.commodity_passed == 3 and res.size_passed == 2 and not res.static_fallback_used
    assert res.pass_rates["size"] == "2/3 (67%)" and res.pass_rates["hate"].endswith("/2 (100%)")
    cov = res.field_coverage
    assert cov["Börsvärde"] == (2, 2) and cov["FCF"] == (2, 2) and cov["ROIC"] == (1, 2)


def test_static_fallback_when_global_is_empty(monkeypatch):
    res, seen = _run(monkeypatch, [], static_fallback=True)
    assert res.static_fallback_used and res.config.universe == "us_ca_resource"
    assert len(seen) > 20                                        # den kurerade listan skannades
    res2, _ = _run(monkeypatch, [], static_fallback=False)
    assert not res2.static_fallback_used


def test_meaningless_vs_missing_and_yahoo_fill():
    snap = {"market_cap": 100.0, "fcf_m": -5.0, "ebitda_margin": -0.2, "revenue_m": 10.0, "roic": -0.3,
            "debt_to_equity": 0.4}
    miss, nm = eng.missing_fields(snap)
    assert nm == ["Nettoskuld/EBITDA (negativ EBITDA)", "P/FCF (negativt FCF)", "EV/EBITDA (negativ EBITDA)"]
    assert miss == []
    snap2 = {"market_cap": 100.0, "fcf_m": 50.0}
    info = {"ebitdaMargins": 0.35, "debtToEquity": 47.0, "enterpriseToEbitda": 6.2, "ebitda": 1e9,
            "totalDebt": 2e9, "totalCash": 5e8}
    filled = eng.yahoo_fill(snap2, "NEM", lambda sym: info)
    assert filled == ["EBITDA-marginal", "D/E", "EV/EBITDA", "Nettoskuld/EBITDA"]
    assert snap2["ebitda_margin"] == 0.35 and snap2["debt_to_equity"] == 0.47
    assert snap2["ev_ebitda"] == 6.2 and snap2["net_debt_ebitda"] == 1.5
    # negativ EBITDA: inga multiplar ur Yahoo heller
    snap3 = {}
    eng.yahoo_fill(snap3, "X", lambda sym: {"ebitdaMargins": -0.1, "ebitda": -5e7, "enterpriseToEbitda": -3.0})
    assert "ev_ebitda" not in snap3 and "net_debt_ebitda" not in snap3 and snap3["ebitda_margin"] == -0.1
    assert eng.yahoo_fill({"ebitda_margin": 0.1, "debt_to_equity": 0.2, "ev_ebitda": 5.0, "net_debt_ebitda": 1.0},
                          "Y", lambda sym: 1 / 0) == []          # inget att fylla → ingen fråga
    assert eng.yahoo_fill({}, "Z", lambda sym: 1 / 0) == []      # Yahoo faller → tomt, ingen krasch


def test_ui_preset_and_texts():
    from contrarian_alpha import ui
    p = ui._MARKETS["USA & Kanada"]
    assert p["global_market_ids"] == US_CA and p["include_global"] and p["static_fallback"]
    assert p["min_market_cap_musd"] == 50.0
    assert "US/CA Resource" not in ui._MARKETS
