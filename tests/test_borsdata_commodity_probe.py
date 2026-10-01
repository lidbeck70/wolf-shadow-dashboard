"""
Börsdata-proben, råvarudelen: listar instrumenten på råvaru-/valutamarknaderna
(76 Forex, 77 Nymex och andra av typen "other") med prishistorik och söker
namn efter uran, kol, zink, sällsynta metaller. Falsk API — inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import borsdata_probe as bp  # noqa: E402


class _FakeAPI:
    def get_markets(self):
        return [{"id": 1, "name": "Large Cap", "countryId": 1}, {"id": 76, "name": "Forex"},
                {"id": 77, "name": "Nymex"}, {"id": 90, "name": "Commodities"}]

    def get_countries(self):
        return [{"id": 1, "name": "Sverige"}]

    def get_instruments(self):
        return [{"insId": 1, "name": "Boliden", "ticker": "BOL", "marketId": 1, "instrumentType": 0},
                {"insId": 2, "name": "Uranium", "ticker": "UX", "marketId": 77, "instrumentType": 9},
                {"insId": 3, "name": "Zinc", "ticker": "ZN", "marketId": 90, "instrumentType": 9}]

    def get_global_instruments_list(self):
        return [{"insId": 2, "name": "Uranium", "ticker": "UX", "marketId": 77},
                {"insId": 4, "name": "EUR/USD", "ticker": "EURUSD", "marketId": 76}]

    def get_stockprices(self, ins_id, max_count=0):
        if ins_id == 3:
            raise RuntimeError("404")
        return [{"d": "2006-01-02", "c": 40.0}, {"d": "2026-09-30", "c": 80.5}] if ins_id == 2 else []


def test_lists_commodity_markets_with_history():
    lines = []
    rows = bp.probe_commodities(_FakeAPI(), out=lines.append)
    text = "\n".join(lines)
    assert [r["insId"] for r in rows] == [2, 3, 4]                 # inte Boliden, Uranium bara en gång
    by = {r["insId"]: r for r in rows}
    assert "2006-01-02 → 2026-09-30" in by[2]["history"] and "≈20.7 år" in by[2]["history"]
    assert "senast=80.5" in by[2]["history"]
    assert "FEL" in by[3]["history"] and by[4]["history"] == "pris: inga rader"
    assert "[76, 77, 90]" in text                                   # "Commodities" upptäcks som råvarumarknad
    assert "uran       → UX (2)" in text and "zinc       → ZN (3)" in text
    assert "coal       → INGEN TRÄFF" in text


def test_keyword_matching():
    assert bp._keyword_hits("Uranium U3O8") == ["uran"]
    assert "kol" not in bp._keyword_hits("Petrol")
    assert "coal" in bp._keyword_hits("Newcastle Coal Futures")
    assert "rare" in bp._keyword_hits("Rare Earth Oxide")


def test_workflow_offers_commodities_only():
    wf = open(os.path.join(ROOT, ".github", "workflows", "borsdata-probe.yml"), encoding="utf-8").read()
    assert "PROBE_SECTION" in wf and "default: commodities" in wf
    src = open(os.path.join(ROOT, "borsdata_probe.py"), encoding="utf-8").read()
    assert 'os.environ.get("PROBE_SECTION", "all") == "commodities"' in src
