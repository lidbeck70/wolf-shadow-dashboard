"""
Samma Börsdata-ticker i två länder: GENO är Genovis i Sverige och General Oceans i Norge. Yahoo-suffixet avgör
landet (Börsdatas countryId), så GENO.ST blir Genovis och GENO.OL General Oceans — förut vann den som lästes
sist. Reservtabellen för marknader följer Börsdatas /markets (proben 2026-10-27). Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import borsdata_api as bd  # noqa: E402
import markets as mk  # noqa: E402

INSTRUMENTS = [
    {"insId": 298, "ticker": "GENO", "name": "Genovis", "marketId": 4, "countryId": 1},
    {"insId": 2621, "ticker": "GENO", "name": "General Oceans", "marketId": 9, "countryId": 2},
    {"insId": 1035, "ticker": "ARR", "name": "Arribatec", "marketId": 9, "countryId": 2},
    {"insId": 236, "ticker": "VOLV B", "name": "Volvo B", "marketId": 1, "countryId": 1},
    {"insId": 906, "ticker": "NOVO B", "name": "Novo Nordisk", "marketId": 20, "countryId": 4},
    {"insId": 686, "ticker": "NOKIA", "name": "Nokia", "marketId": 14, "countryId": 3},
]


class _API(bd.BorsdataAPI):
    def __init__(self):
        self._instruments = None
        self._ticker_map = None
        self._id_map = None

    def _get(self, path, params=None, use_cache=True):
        return {"instruments": INSTRUMENTS}


def test_suffix_picks_the_company_in_that_country():
    assert bd.match_by_country(INSTRUMENTS, "GENO.ST") == 298
    assert bd.match_by_country(INSTRUMENTS, "geno.ol") == 2621
    assert bd.match_by_country(INSTRUMENTS, "VOLV-B.ST") == 236 and bd.match_by_country(INSTRUMENTS, "NOVO-B.CO") == 906
    assert bd.match_by_country(INSTRUMENTS, "ARR.ST") is None          # inget svenskt ARR → gamla uppslaget
    assert bd.match_by_country(INSTRUMENTS, "GENO") is None and bd.match_by_country(INSTRUMENTS, "AAPL") is None
    api = _API()
    assert api.resolve_instrument_id("GENO.ST") == 298 and api.resolve_instrument_id("GENO.OL") == 2621
    assert api.resolve_instrument_id("NOKIA.HE") == 686 and api.resolve_instrument_id("ARR.ST") == 1035


def test_bd_client_resolves_by_country_too(monkeypatch):
    from utils import bd_api
    monkeypatch.setattr(bd_api, "_cached_instruments", lambda key: INSTRUMENTS)
    client = bd_api.BDClient.__new__(bd_api.BDClient)
    client.api_key = "x"
    assert client._resolve_id("GENO.ST") == 298 and client._resolve_id("GENO.OL") == 2621


def test_fallback_market_table_matches_borsdata():
    for mid, suffix in ((4, ".ST"), (31, ".ST"), (9, ".OL"), (27, ".OL"), (78, ".OL"), (14, ".HE"), (30, ".HE"),
                        (20, ".CO"), (23, ".CO"), (48, ".CO")):
        assert mk.FALLBACK[mid].suffix == suffix, mid
    assert mk.to_yf("NOKIA", 14, mk.FALLBACK) == "NOKIA.HE" and mk.to_yf("NOVO B", 20, mk.FALLBACK) == "NOVO-B.CO"
    assert {mk.FALLBACK[i].kind for i in (7, 8, 13, 19)} == {mk.INDEX}
