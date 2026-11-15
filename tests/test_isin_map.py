"""
Journalens ISIN-karta (journal_import/ticker_map.py) — kontrollerad mot Börsdatas instrumentlistor 2026-11-15.

Den gamla kartan kopplade de flesta nordiska ISIN till fel bolag (SEB A ↔ Sandvik, Volvo B → Atlas Copco A,
Mowi → Telenor, …), så Journalens mäklarimport gav affärer och positioner fel ticker. Här står paren som Börsdata
gav, så att ingen gammal post smyger tillbaka.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from journal_import.ticker_map import ISIN_MAP, resolve_ticker   # noqa: E402

BORSDATA_2026_11_15 = {
    "SE0017486889": "ATCO-A.ST", "SE0017486897": "ATCO-B.ST", "SE0000115446": "VOLV-B.ST",
    "SE0000148884": "SEB-A.ST", "SE0000106270": "HM-B.ST", "SE0007100599": "SHB-A.ST",
    "SE0015811963": "INVE-B.ST", "SE0000695876": "ALFA.ST", "SE0000667891": "SAND.ST",
    "SE0000112724": "SCA-B.ST", "SE0000667925": "TELIA.ST", "SE0000108656": "ERIC-B.ST",
    "SE0000108227": "SKF-B.ST", "SE0000171100": "SSAB-A.ST", "SE0020050417": "BOL.ST",
    "SE0016101844": "SINCH.ST", "NO0010096985": "EQNR.OL", "NO0010063308": "TEL.OL", "NO0003054108": "MOWI.OL",
    "NO0005052605": "NHY.OL", "NO0010161896": "DNB.OL", "NO0003733800": "ORK.OL", "NO0003053605": "STB.OL",
    "NO0010208051": "YAR.OL", "NO0010716582": "AKSO.OL", "NO0010310956": "SALM.OL", "FI0009000681": "NOKIA.HE",
    "FI0009007132": "FORTUM.HE", "FI0009013296": "NESTE.HE", "FI4000552500": "SAMPO.HE",
    "DK0062498333": "NOVO-B.CO", "DK0010244508": "MAERSK-B.CO", "DK0060634707": "RBREW.CO",
    "US0378331005": "AAPL", "US67066G1040": "NVDA",
}

# ISIN som den gamla kartan gav fel bolag (eller som nu är ett annat bolag) — får aldrig komma tillbaka
OLD_WRONG = ("SE0000107419", "SE0011166610", "SE0000193120", "SE0000652216", "SE0005994832", "SE0000872305",
             "SE0000869646", "SE0001116761", "SE0000101584", "SE0000454746", "NO0010721050", "NO0010096922",
             "NO0003078900", "NO0010081235", "NO0003440709", "NO0010140502", "NO0010209331", "NO0003399084",
             "NO0003399001", "FI0009008403", "FI0009013403", "FI0009000202", "DK0060534915", "DK0010234467",
             "DK0060079531")


def test_the_map_agrees_with_borsdata():
    for isin, ticker in BORSDATA_2026_11_15.items():
        assert ISIN_MAP.get(isin) == ticker, isin
        assert resolve_ticker(isin) == ticker


def test_no_old_wrong_isin_is_left_and_no_ticker_has_two_isins():
    assert not [i for i in OLD_WRONG if i in ISIN_MAP]
    tickers = list(ISIN_MAP.values())
    assert len(tickers) == len(set(tickers))
