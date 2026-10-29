"""
Segmentens största bolag med kassaflöde (universe/segment_top.py): Börsdata KPI 50 (börsvärde → USD) och KPI 62 > 0
(kassaflöde), segmentet ur sektor/bransch, Norden/USA/Kanada på huvudlistorna (inte First North, TSX Venture, OTC),
ett aktieslag per bolag; temana ur temaprobens urval. Universum-registrets jobb sparar bloben och sparar registret
även när KPI-anropen fallerar. Probens avsnitt 'sectors'. Inget nätverk.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import markets as mk  # noqa: E402
import universe_scan as us  # noqa: E402
from universe import registry as ur  # noqa: E402
from universe import segment_top as ust  # noqa: E402
from valhall import segments as sgm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _inst(iid, name, ticker, mid, sector, branch=None, ccy="SEK", country=1):
    return {"insId": iid, "name": name, "ticker": ticker, "marketId": mid, "sectorId": sector, "branchId": branch,
            "stockPriceCurrency": ccy, "countryId": country}


NORDIC = [_inst(1, "Volvo A", "VOLV A", 1, 5), _inst(2, "Volvo B", "VOLV B", 1, 5),            # ett bolag, två slag
          _inst(3, "Atlas Copco A", "ATCO A", 1, 5), _inst(4, "Kongsberg", "KOG", 9, 5, ccy="NOK", country=2),
          _inst(5, "Förlustbolaget", "LOSS", 1, 5), _inst(6, "First North AB", "FNAB", 4, 5),
          _inst(7, "Castellum", "CAST", 1, 1, 75), _inst(8, "SEB A", "SEB A", 1, 1, 70),       # Fastighet vs Finans
          _inst(9, "Utan kassaflödesdata", "NOCF", 1, 5)]
GLOBAL = [_inst(20, "Caterpillar", "CAT", 32, 5, ccy="USD", country=5), _inst(21, "OTC Corp", "OTCC", 34, 5, ccy="USD", country=5),
          _inst(22, "CN Rail", "CNR", 35, 5, ccy="CAD", country=6), _inst(23, "Venture Ind", "VEN", 36, 5, ccy="CAD", country=6),
          _inst(24, "Rolls-Royce", "RR", 38, 5, ccy="GBP", country=7), _inst(25, "Sibanye", "SBSW", 32, 7, ccy="USD", country=30),
          _inst(26, "Impala ADR", "IMPUY", 34, 7, ccy="USD", country=30), _inst(27, "Johnson Matthey", "JMAT", 38, 7, ccy="GBP", country=7)]
MCAP = {1: 500_000, 2: 500_000, 3: 700_000, 4: 300_000, 5: 900_000, 6: 50_000, 7: 60_000, 8: 250_000, 9: 999_999,
        20: 150_000, 21: 900_000, 22: 100_000, 23: 90_000, 24: 80_000, 25: 5_000, 26: 9_000, 27: 3_000}
OCF = {1: 10, 2: 10, 3: 20, 4: 5, 5: -3, 6: 1, 7: 2, 8: 9, 20: 8, 21: 50, 22: 7, 23: 4, 24: 3, 25: 1, 26: 2, 27: 1}
FX = {"USD": 1.0, "SEK": 0.1, "NOK": 0.1, "CAD": 0.7, "GBP": 1.3}


def test_rank_segments_keeps_cash_flow_main_boards_and_one_share_class():
    out = ust.rank_segments(NORDIC, GLOBAL, mk.FALLBACK, MCAP, OCF, FX)
    assert list(out) == [s.key for s in sgm.SEGMENTS]
    assert all(list(per) == list(ust.MARKETS) for per in out.values())
    ind = out["industrials"]
    assert [r["ticker"] for r in ind["norden"]] == ["ATCO-A.ST", "VOLV-B.ST", "KOG.OL"]   # USD: 70k > 50k > 30k
    assert ind["norden"][0] == {"ticker": "ATCO-A.ST", "name": "Atlas Copco A", "mcap_usd": 70000, "ins_id": 3}
    assert [r["ticker"] for r in ind["usa"]] == ["CAT"] and [r["ticker"] for r in ind["kanada"]] == ["CNR.TO"]
    tickers = {r["ticker"] for per in out.values() for rows in per.values() for r in rows}
    for gone in ("LOSS.ST", "NOCF.ST", "FNAB.ST", "OTCC", "VEN.V", "RR.L", "VOLV-A.ST"):   # kassaflöde, listor, slag
        assert gone not in tickers, gone
    assert [r["ticker"] for r in out["real_estate"]["norden"]] == ["CAST.ST"]             # bransch 75 → Fastighet
    assert [r["ticker"] for r in out["financials"]["norden"]] == ["SEB-A.ST"]
    assert out["healthcare"] == {"norden": [], "usa": [], "kanada": []}
    assert len(ust.rank_segments(NORDIC, GLOBAL, mk.FALLBACK, MCAP, OCF, FX, n=1)["industrials"]["norden"]) == 1


def test_rank_segments_shows_the_b_share_and_the_home_listing():
    """Körningen 29 okt visade ERIC-A, VOLV-A, TEL2-A och NOKIA-SEK: Börsdata räknar börsvärdet på varje slags kurs,
    så A-aktien och valutalinjen kan se större ut. Bolaget rangordnas på sitt största slag, B/huvudnoteringen visas."""
    nordic = [_inst(1, "Ericsson A", "ERIC A", 1, 5), _inst(2, "Ericsson B", "ERIC B", 1, 5),
              _inst(3, "Nokia SEK", "NOKIA SEK", 1, 5), _inst(4, "Nokia", "NOKIA", 14, 5, ccy="EUR", country=3),
              _inst(5, "Mindre AB", "MIND", 1, 5),
              _inst(6, "SEB A", "SEB A", 1, 5), _inst(7, "SEB C", "SEB C", 1, 5)]       # A före C (andra körningen)
    mcap = {1: 300_000, 2: 250_000, 3: 330_000, 4: 30_000, 5: 270_000, 6: 200_000, 7: 210_000}
    ocf = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    out = ust.rank_segments(nordic, [], mk.FALLBACK, mcap, ocf, {"SEK": 0.1, "EUR": 1.1})
    assert [r["ticker"] for r in out["industrials"]["norden"]] == ["NOKIA.HE", "ERIC-B.ST", "MIND.ST", "SEB-A.ST"]
    assert out["industrials"]["norden"][1]["mcap_usd"] == 25_000      # det visade slagets eget värde
    assert [ust._class_rank(t) for t in ("ERIC B", "RCI.B", "SEB A", "CAST", "SEB C", "BRK.A")] == [0, 0, 1, 1, 2, 1]


def test_theme_tops_rank_the_probe_pool_without_otc():
    from universe import theme_probe as tp
    index = tp.instrument_index(NORDIC, GLOBAL, mk.FALLBACK)
    tops = ust.theme_tops(index, mk.FALLBACK, MCAP, OCF, FX)
    assert list(tops) == tp.map_themes()
    assert [r["ticker"] for r in tops["palladium"]] == ["SBSW", "JMAT.L"]                  # IMPUY (OTC) bort
    assert tops["palladium"][0] == {"ticker": "SBSW", "name": "Sibanye", "mcap_usd": 5000}
    assert tops["uran"] == []


class _Api:
    def __init__(self, kpis=None, fail=False):
        self.kpis, self.fail, self.calls = kpis if kpis is not None else {50: MCAP, 62: OCF}, fail, []

    def _kpi(self, kpi, ids):
        self.calls.append(kpi)
        if self.fail:
            raise RuntimeError("Börsdata nere")
        return [{"i": i, "n": v} for i, v in (self.kpis.get(kpi) or {}).items() if i in ids]

    def get_kpi_screener(self, kpi, calc_group="last", calc="latest"):
        return self._kpi(kpi, {i["insId"] for i in NORDIC})

    def get_kpi_screener_global(self, kpi, calc_group="last", calc="latest"):
        return self._kpi(kpi, {i["insId"] for i in GLOBAL})

    def get_markets(self):
        return []

    def get_countries(self):
        return [{"id": 1, "name": "Sverige"}, {"id": 2, "name": "Norge"}, {"id": 5, "name": "USA"},
                {"id": 6, "name": "Kanada"}, {"id": 7, "name": "England"}, {"id": 30, "name": "Sydafrika"}]

    def get_instruments(self):
        return NORDIC

    def get_global_instruments_list(self):
        return GLOBAL


def test_build_makes_the_blob_and_refuses_an_empty_answer():
    payload = ust.build(_Api(), NORDIC, GLOBAL, mk.FALLBACK, now="2026-10-26T05:17:00+00:00")
    assert payload["generated"] == "2026-10-26T05:17:00+00:00" and payload["source"] == "Börsdata KPI 50/62"
    assert payload["segments"]["industrials"]["usa"][0]["ticker"] == "CAT" and "palladium" in payload["themes"]
    assert any(line.startswith("Industri · Norden ATCO-A.ST, VOLV-B.ST, KOG.OL") for line in ust.describe(payload))
    with pytest.raises(ValueError):
        ust.build(_Api(kpis={}), NORDIC, GLOBAL, mk.FALLBACK)
    with pytest.raises(ValueError):
        ust.build(_Api(), [], [], mk.FALLBACK)


def test_universe_scan_saves_the_blob_and_the_register_survives_a_failure():
    saved = {}
    out = us.run(api=_Api(), nordic=NORDIC, global_=GLOBAL, table=mk.FALLBACK,
                 save=lambda n, d: saved.setdefault(n, d) is not None)
    assert out["saved"][ust.BLOB] is True and saved[ust.BLOB]["segments"]["industrials"]["norden"]
    assert all(out["saved"][f"universe_{m.key}.json"] for m in ur.MARKETS) and "universe.json" in saved
    saved.clear()
    bad = us.run(api=_Api(fail=True), nordic=NORDIC, global_=GLOBAL, table=mk.FALLBACK,
                 save=lambda n, d: saved.setdefault(n, d) is not None)
    assert ust.BLOB not in bad["saved"] and ust.BLOB not in saved                          # förra veckans lista står kvar
    assert "universe.json" in saved and all(bad["saved"].values())                          # registret sparat ändå
    dry = us.run(api=_Api(), nordic=NORDIC, global_=GLOBAL, table=mk.FALLBACK, no_save=True)
    assert dry["saved"] == {}


def test_probe_section_prints_the_tables_with_country():
    lines = []
    ust.probe(_Api(), out=lines.append)
    text = "\n".join(lines)
    assert "SEGMENTEN" in text and "CAT" in text and "SBSW" in text and "Sydafrika" in text and "Industri" in text
    src = open(os.path.join(ROOT, "borsdata_probe.py"), encoding="utf-8").read()
    assert '== "sectors"' in src and "segment_top.probe" in src
    wf = open(os.path.join(ROOT, ".github", "workflows", "borsdata-probe.yml"), encoding="utf-8").read()
    assert "sectors" in wf.split("options:")[1].split("\n")[0]
