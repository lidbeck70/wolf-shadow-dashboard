"""
Thor fas 2 — identiteten: tickerformer, Börsdata-uppslaget (nordiskt och globalt, med och utan klient),
segment, sektor-ETF, hemmaindex, Viking Nines marknad och råvarutemat med källa. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thor import ticker as tt  # noqa: E402


class Api:
    is_configured = True
    NORDIC = {3: {"insId": 3, "ticker": "BOL", "name": "Boliden", "sectorId": 7, "branchId": 17, "marketId": 1,
                  "stockPriceCurrency": "SEK"},
              5: {"insId": 5, "ticker": "FABG", "name": "Fabege", "sectorId": 1, "branchId": 75, "marketId": 1,
                  "stockPriceCurrency": "SEK"}}
    GLOBAL = [{"insId": 900, "ticker": "NEM", "name": "Newmont", "sectorId": 7, "branchId": 18}]

    def resolve_instrument_id(self, t):
        return {"BOL": 3, "FABG": 5}.get(t)

    def get_instrument_info(self, i):
        return self.NORDIC.get(i)

    def get_global_instruments_list(self):
        return self.GLOBAL


def test_forms_and_base():
    assert tt.forms("ERIC-B.ST") == ["ERIC-B.ST", "ERIC-B", "ERIC B"] and tt.base("EQNR.OL") == "EQNR"
    assert tt.forms("NEM") == ["NEM"] and tt.forms("eqx") == ["EQX"] and tt.base("AEM.TO") == "AEM"
    assert tt.forms("") == [] and tt.base(None) == ""


def test_identify_nordic_with_borsdata():
    t = tt.identify("bol.st", Api())
    assert t.yf == "BOL.ST" and t.name == "Boliden" and t.ins_id == 3 and t.currency == "SEK" and t.nordic
    assert t.region == "Norden" and t.segment == "materials" and t.segment_label == "Material"
    assert t.sector_etf == "XLB" and t.sector_source == "Börsdata · Material → XLB"
    assert t.home_index == "^OMX" and t.market == "OMXS30"
    assert t.theme == "koppar" and t.theme_source == tt.SOURCE_BERSERK and t.theme_label == "Koppar"
    assert t.complex == "basmetaller" and t.is_commodity and t.as_dict()["is_commodity"] is True


def test_identify_real_estate_branch_and_bare_form():
    import markets
    table = {1: markets.Market(id=1, name="Large Cap", country_id=1, country="Sverige", suffix=".ST")}
    t = tt.identify("FABG", Api(), table=table)
    assert t.yf == "FABG.ST" and t.segment == "real_estate" and t.theme is None and not t.is_commodity and t.nordic
    assert t.sector_etf == "XLF"                                    # Börsdata sektor 1 → XLF (Viking Nines tabell)
    bare = tt.identify("FABG", Api())                               # utan tabell lämnas formen
    assert bare.yf == "FABG" and bare.segment == "real_estate" and not bare.nordic


def test_identify_global_and_without_api():
    t = tt.identify("NEM", Api())
    assert t.ins_id == 900 and t.name == "Newmont" and t.theme == "guld" and t.segment == "materials"
    assert not t.nordic and t.home_index == "SPY" and t.market == "SPY" and t.sector_etf == "XLB"
    v = tt.identify("VOLV-B.ST")
    assert v.yf == "VOLV-B.ST" and v.ins_id is None and v.segment is None and v.sector_etf is None
    assert v.home_index == "^OMX" and v.market == "OMXS30" and v.nordic and v.theme is None and v.raw == "VOLV-B.ST"


def test_theme_sources_in_order(monkeypatch):
    sheets = {"producers": {"producers": [{"ticker": "EQX", "commodity": "Guld"}]}}
    assert tt.theme_for("EQX", sheets) == ("guld", tt.SOURCE_SHEET)
    assert tt.theme_for("eqx.to", sheets) == ("guld", tt.SOURCE_SHEET)          # arket skrev utan suffix
    from berserk import universe as buv
    monkeypatch.setattr(buv, "theme_of", lambda t: "")
    assert tt.theme_for("NEM") == ("guld", tt.SOURCE_EMBER)
    assert tt.theme_for("XYZ", branch_id=7) == ("uran", f"{tt.SOURCE_BRANCH} 7")
    assert tt.theme_for("XYZ", branch_id=99) == (None, "") and tt.theme_for("XYZ") == (None, "")


def test_market_rule_matches_viking_nine():
    import ovtlyr_nine as on
    for t in ("BOL.ST", "EQNR.OL", "NOVO-B.CO", "NOKIA.HE", "NEM", "AEM.TO", "RIO.L"):
        assert tt.market_for(t) == on.market_for(t), t


def test_lookup_without_client_key_or_network():
    assert tt.lookup(None, "BOL.ST") is None

    class Off:
        is_configured = False

    class Broken:
        def resolve_instrument_id(self, t):
            raise RuntimeError("nät")

    assert tt.lookup(Off(), "BOL.ST") is None and tt.lookup(Broken(), "BOL.ST") is None
    assert tt.identify("BOL.ST", Broken()).theme == "koppar"                     # uppslaget faller, identiteten står
