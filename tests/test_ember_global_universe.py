"""
EMBER-universumet: Norden + USA, Kanada och Australien ur Börsdata global
(råvarubranscher, börsvärde ≥ golvet, inte OTC) + de kurerade US/INTL-listorna.
Förut: Norden + 111 statiska US/INTL-tickers, ingen Australien.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import markets  # noqa: E402
from ember import universe as eu  # noqa: E402


class _Api:
    is_configured = True

    def __init__(self, glob=None, mcap=None):
        self.glob = glob if glob is not None else _GLOB
        self.mcap = mcap if mcap is not None else _MCAP

    def get_markets(self):
        return [{"id": 32, "name": "NYSE", "countryId": 5, "exchangeName": "NYSE"},
                {"id": 34, "name": "OTC", "countryId": 5, "exchangeName": "OTC"},
                {"id": 35, "name": "Toronto", "countryId": 6, "exchangeName": "TSX"},
                {"id": 36, "name": "TSX Venture", "countryId": 6, "exchangeName": "TSX Venture"},
                {"id": 60, "name": "ASX", "countryId": 23, "exchangeName": "ASX"},
                {"id": 38, "name": "London", "countryId": 7, "exchangeName": "LSE"}]

    def get_countries(self):
        return [{"id": 5, "name": "USA"}, {"id": 6, "name": "Kanada"}, {"id": 23, "name": "Australien"},
                {"id": 7, "name": "England"}]

    def get_global_instruments_list(self):
        return self.glob

    def get_kpi_screener_global(self, kpi_id, calc_group="last", calc="latest"):
        return [{"i": i, "n": v} for i, v in self.mcap.items()]


_GLOB = [
    {"insId": 1, "ticker": "NEM", "marketId": 32, "branchId": 18, "stockPriceCurrency": "USD"},      # guld, USA
    {"insId": 2, "ticker": "AAPL", "marketId": 32, "branchId": 70, "stockPriceCurrency": "USD"},     # inte råvara
    {"insId": 3, "ticker": "TINY", "marketId": 32, "branchId": 16, "stockPriceCurrency": "USD"},     # för liten
    {"insId": 4, "ticker": "OTCG", "marketId": 34, "branchId": 18, "stockPriceCurrency": "USD"},     # OTC
    {"insId": 5, "ticker": "CNQ", "marketId": 35, "branchId": 2, "stockPriceCurrency": "CAD"},       # olja, Toronto
    {"insId": 6, "ticker": "JUN", "marketId": 36, "branchId": 16, "stockPriceCurrency": "CAD"},      # Venture, okänt mcap
    {"insId": 7, "ticker": "BHP", "marketId": 60, "branchId": 17, "stockPriceCurrency": "AUD"},      # ASX
    {"insId": 8, "ticker": "PDN", "marketId": 60, "branchId": 7, "stockPriceCurrency": "AUD"},       # uran, ASX
    {"insId": 9, "ticker": "RIO", "marketId": 38, "branchId": 17, "stockPriceCurrency": "GBP"},      # London — inte med
    {"insId": 10, "ticker": "WFG", "marketId": 35, "branchId": 21, "stockPriceCurrency": "CAD"},     # skog, med i EMBER
    {"insId": 11, "ticker": "SMALLCA", "marketId": 35, "branchId": 2, "stockPriceCurrency": "CAD"},  # 350 MCAD ≈ 255 MUSD
    {"insId": 12, "ticker": "LAC", "name": "Lithium Americas Corp", "marketId": 32, "branchId": 15,
     "stockPriceCurrency": "USD"},                                                                   # kemi men litium
    {"insId": 13, "ticker": "PPG", "name": "PPG Industries", "marketId": 32, "branchId": 15,
     "stockPriceCurrency": "USD"},                                                                   # vanlig kemi
]
_MCAP = {1: 50000.0, 2: 3e6, 3: 120.0, 4: 900.0, 5: 90000.0, 7: 200000.0, 8: 900.0, 9: 90000.0,
         10: 8000.0, 11: 350.0, 12: 900.0, 13: 30000.0}


def test_global_picks_us_canada_australia_commodity_names_above_the_floor():
    try:
        tickers, per, err = eu.fetch_global_commodity_tickers(_Api())
    finally:
        markets.reset()
    assert err == ""
    assert tickers == sorted(["NEM", "CNQ.TO", "JUN.V", "BHP.AX", "PDN.AX", "WFG.TO", "LAC"])
    assert per == {"USA": 2, "Kanada": 3, "Australien": 2}
    # inte råvara, för liten, OTC, London, CAD-omräkning under golvet, vanlig kemi
    for t in ("AAPL", "TINY", "OTCG", "RIO.L", "SMALLCA.TO", "PPG"):
        assert t not in tickers


def test_without_global_licence_it_says_so():
    try:
        tickers, per, err = eu.fetch_global_commodity_tickers(_Api(glob=[]))
    finally:
        markets.reset()
    assert tickers == [] and "Pro+ global" in err


def test_floor_can_be_turned_off():
    try:
        tickers, _, _ = eu.fetch_global_commodity_tickers(_Api(), min_mcap_musd=0)
    finally:
        markets.reset()
    assert "TINY" in tickers and "SMALLCA.TO" in tickers and "AAPL" not in tickers


def test_build_universe_adds_the_global_names(monkeypatch):
    monkeypatch.setattr(eu, "fetch_nordic_commodity_tickers", lambda: (["BOL.ST"], True, ""))
    monkeypatch.setattr(eu, "fetch_global_commodity_tickers",
                        lambda: (["NEM", "BHP.AX"], {"USA": 1, "Australien": 1}, ""))
    tickers, stats = eu.build_universe(eu.SOURCE_AUTO, use_prefilter=False)
    assert tickers[:3] == ["BOL.ST", "NEM", "BHP.AX"]
    assert tickers.count("NEM") == 1                          # NEM finns även i den kurerade listan
    assert stats.global_raw == 2 and stats.global_by_country == {"USA": 1, "Australien": 1}
    assert stats.total_before_prefilter == len(tickers)


def test_build_universe_reports_a_missing_licence(monkeypatch):
    monkeypatch.setattr(eu, "fetch_nordic_commodity_tickers", lambda: ([], True, ""))
    monkeypatch.setattr(eu, "fetch_global_commodity_tickers",
                        lambda: ([], {}, "globala instrument saknas (Börsdata Pro+ global krävs)"))
    tickers, stats = eu.build_universe(eu.SOURCE_AUTO, use_prefilter=False)
    assert stats.global_raw == 0 and "Pro+" in stats.global_error
    assert len(tickers) == len(eu.US_INTL_CURATED)            # de kurerade finns kvar


def test_critical_metals_are_in_the_curated_list_with_themes():
    from ember.config import TICKER_THEME_MAP
    for t in ("LYC.AX", "ILU.AX", "ARU.AX", "ALB", "SQM", "PLS.AX", "MIN.AX", "LTR.AX", "SBSW", "PPLT", "PALL"):
        assert t in eu.US_INTL_CURATED, t
    assert TICKER_THEME_MAP["LYC.AX"] == "sallsynta" and TICKER_THEME_MAP["ALB"] == "sallsynta"
    assert TICKER_THEME_MAP["SBSW"] == "guld" and TICKER_THEME_MAP["PALL"] == "guld"
