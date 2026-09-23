"""
PR 5 av panelgenomgången — universum.

Sex moduler hade var sin marknadstabell och sa olika saker om samma id;
ingen läste Börsdatas /markets. Ember hade två tickerkartor som sa emot
varandra för 16 tickers. Royalty-håven saknade branschfilter, Durrett och
Rule körde på fel geografi, okänd valuta räknades som USD, "mining" blev
gold_miner före koppar, och ett tjugotal listor bar döda tickers (GOLD,
MRO, MAG, X…) utan status.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import markets as mk          # noqa: E402
import dead_tickers as dead   # noqa: E402


# ── markets.py: en tabell ────────────────────────────────────────────────────
def test_from_api_derives_suffix_from_country_and_exchange_not_from_id():
    t = mk.from_api(
        [{"id": 1, "name": "Large Cap", "countryId": 1, "isIndex": False, "exchangeName": "Nasdaq Stockholm"},
         {"id": 7, "name": "OMX Index", "countryId": 1, "isIndex": True},
         {"id": 35, "name": "Toronto", "countryId": 6, "exchangeName": "TSX"},
         {"id": 36, "name": "TSX Venture", "countryId": 6, "exchangeName": "TSX Venture"},
         {"id": 60, "name": "ASX", "countryId": 30, "exchangeName": "ASX"},
         {"id": 76, "name": "Forex", "countryId": None}],
        [{"id": 1, "name": "Sverige"}, {"id": 6, "name": "Kanada"}, {"id": 30, "name": "Australien"}])
    assert (t[1].suffix, t[1].kind) == (".ST", mk.STOCK)
    assert t[7].kind == mk.INDEX
    assert (t[35].suffix, t[36].suffix) == (".TO", ".V")
    assert t[60].suffix == ".AX"                       # Australien finns med
    assert t[76].kind == mk.OTHER
    assert mk.nordic_stock_ids(t) == {1}


def test_to_yf_dashes_share_classes_and_leaves_unknown_markets_bare():
    assert mk.to_yf("SKF A", 1) == "SKF-A.ST"
    assert mk.to_yf("TECK B", 35) == "TECK-B.TO"
    assert mk.to_yf("EQNR", 9) == "EQNR.OL"
    assert mk.to_yf("CCJ", 32) == "CCJ"
    assert mk.to_yf("XYZ", 999) == "XYZ"
    assert mk.to_yf("XYZ") == "XYZ"
    assert not mk.is_stock(7) and not mk.is_stock(76) and not mk.is_stock(999)


def test_fallback_is_the_read_table_with_the_baltic_suffixes_corrected():
    assert mk.FALLBACK[53].suffix == ".RG" and mk.FALLBACK[54].suffix == ".VS"
    assert {m.id for m in mk.FALLBACK.values() if m.kind == mk.INDEX} == {7, 8, 13, 19, 28, 31}
    assert mk.FALLBACK[9].country == "Norge" and mk.FALLBACK[14].country == "Danmark"


def test_load_prefers_blob_then_api_then_fallback():
    class _Api:
        def get_markets(self):
            return [{"id": 5, "name": "X", "countryId": 2}]

        def get_countries(self):
            return [{"id": 2, "name": "Norge"}]

    class _Broken:
        pass

    mk.reset()
    blob = {"markets": mk.serialize({5: mk.Market(5, "Y", 4, "Finland", ".HE")})}
    t = mk.load(api=_Api(), blob=blob)
    assert t[5].suffix == ".HE"                                     # blob före API
    assert t[36].suffix == ".V"                                     # FALLBACK fyller ut
    assert mk.load(api=_Api())[5].suffix == ".OL"                  # API
    mk.reset()
    assert mk.load(api=_Broken(), path="/nonexistent/markets.json") == mk.FALLBACK  # tyst vidare
    assert mk.deserialize(mk.serialize(mk.FALLBACK)) == mk.FALLBACK
    mk.reset()


def test_the_six_copies_read_the_same_table():
    """Ingen modul får ha kvar en egen handskriven id→suffix-tabell."""
    import borsdata_api as bd
    import ticker_universe as tu
    assert set(bd.ALL_NORDIC_MARKETS) == mk.nordic_stock_ids(mk.FALLBACK)
    assert tu.MARKET_SUFFIX == {m.id: m.suffix for m in mk.FALLBACK.values() if m.kind == mk.STOCK}
    for path in ("screens_scan.py", "insider_scan.py", "ember/universe.py",
                 "contrarian_alpha/engine.py", "borsdata_api.py"):
        src = open(os.path.join(ROOT, path), encoding="utf-8").read()
        assert "_MARKET_SUFFIX = {" not in src and "_MARKET_SUFFIX: dict" not in src, path
    src = open(os.path.join(ROOT, "ticker_universe.py"), encoding="utf-8").read()
    assert "1: \".ST\", 2: \".ST\"" not in src


def test_scan_jobs_use_the_live_table_and_publish_it():
    import screens_scan as ss
    import insider_scan as isc
    assert ss.yahoo_ticker({"ticker": "KDK", "marketId": 36}, "global") == "KDK.V"
    assert isc.yf_ticker({"ticker": "VOLV B", "marketId": 1}) == "VOLV-B.ST"

    class _Api:
        def get_markets(self):
            return [{"id": 1, "name": "Large Cap", "countryId": 1}]

        def get_countries(self):
            return [{"id": 1, "name": "Sverige"}]

        def get_instruments(self):
            return [{"insId": 1, "ticker": "BOL", "name": "Boliden", "marketId": 1,
                     "branchId": 17, "countryId": 1, "stockPriceCurrency": "SEK"}]

        def get_kpi_screener(self, kid, g, c):
            return []

        def get_global_instruments_list(self):
            return []

    mk.reset()
    out = ss.scan(_Api())
    by_id = {m["id"]: m for m in out["markets"]}
    assert by_id[1]["suffix"] == ".ST" and by_id[1]["country"] == "Sverige"
    assert 36 in by_id                                              # FALLBACK fyller ut
    assert [r["ticker"] for r in out["screens"]["rule"]["rows"]] == []   # inga KPI:er
    mk.reset()


# ── Australien och håvarnas geografi ────────────────────────────────────────
def test_australia_resolves_from_countries_and_gets_asx_suffix():
    import ticker_universe as tu
    countries = [{"id": 1, "name": "Sverige"}, {"id": 30, "name": "Australien"}]
    assert tu.region_country_ids(["Australien"], countries) == {30}
    assert tu.region_country_ids(["Australien"]) == set()          # inget avläst id
    assert tu.region_country_ids(["Kanada"], countries) == {6}       # avläst id kvar
    assert "Australien" in tu.COUNTRY_REGIONS and "Australien" in tu.FALLBACK_TICKERS
    assert mk.suffix_for("Australia") == ".AX"


def test_screens_follow_the_masterguide_geography_and_industry():
    import screens_scan as ss
    by = ss.SCREEN_BY_KEY
    assert set(by["rule"].countries) == {ss.NORDIC, ss.CA, ss.US, ss.AU}
    assert set(by["durrett"].countries) == {ss.CA, ss.AU, ss.US}
    # Royalty: 70 % bruttomarginal räcker inte — mjukvara har det också.
    good = {"gross_margin": 80.0, "ebit_margin": 50.0, "nd_ebitda": 0.5}
    assert ss.royalty_check(good, {"branch_id": 18})[0] == []
    assert "inte råvarubransch" in ss.royalty_check(good, {"branch_id": 90})[0]
    c = ss.country_ids([{"id": 1, "name": "Sverige"}, {"id": 2, "name": "Norge"},
                        {"id": 30, "name": "Australia"}])
    assert c[ss.NORDIC] == {1, 2} and c[ss.AU] == {30}


def test_unknown_currency_leaves_market_cap_blank_instead_of_pretending_usd():
    import screens_scan as ss
    kpis = {"mcap": {1: 1000.0}, "revenue": {1: 100.0}}
    zar = ss.metrics_for({"insId": 1, "stockPriceCurrency": "ZAR"}, kpis)
    assert zar["mcap_musd"] == 55.0                                  # ZAR finns nu
    odd = ss.metrics_for({"insId": 1, "stockPriceCurrency": "XXX"}, kpis)
    assert odd["mcap_musd"] is None and odd["revenue_musd"] is None
    assert "MCap: saknas" in ss.tiggre_check(odd, {"branch_id": 16})[0]


# ── Ember: en karta ──────────────────────────────────────────────────────────
def test_ember_complex_map_is_derived_from_the_theme_map():
    from ember.config import TICKER_THEME_MAP, THEME_TO_COMPLEX
    from ember.regime import TICKER_COMPLEX_MAP, detect_complex
    assert set(TICKER_COMPLEX_MAP) == set(TICKER_THEME_MAP)
    for t, theme in TICKER_THEME_MAP.items():
        assert TICKER_COMPLEX_MAP[t] == THEME_TO_COMPLEX[theme], t
    assert detect_complex("CCJ") == "energi"            # uran är energi, inte agri
    assert detect_complex("ERO.TO") == "basmetaller"
    assert detect_complex("RIO.L") == "basmetaller" and TICKER_THEME_MAP["BHP.L"] == "koppar"
    assert detect_complex("MP") == "basmetaller"
    assert detect_complex("BOL.ST") is None             # okänt är okänt, inte energi
    assert "GOLD" not in TICKER_THEME_MAP and TICKER_THEME_MAP["B"] == "guld"
    assert "MRO" not in TICKER_THEME_MAP


def test_confidence_registry_agrees_with_ember_on_uranium_and_rare_earths():
    from confidence import commodities as com
    from ember.config import THEME_TO_COMPLEX
    assert com.REGISTRY["uranium"].ember_complex == THEME_TO_COMPLEX["uran"] == "energi"
    assert com.REGISTRY["rare_earth"].ember_complex == THEME_TO_COMPLEX["sallsynta"]
    assert "KOL" not in com.REGISTRY["coal"].proxies and "JJN" not in com.REGISTRY["nickel"].proxies


def test_detect_exposure_prefers_copper_uranium_and_gas_over_generic_mining():
    from alpha_regime.commodity_ratios import detect_exposure, EXPOSURE_TO_RATIO
    from alpha_regime.tactical_entry import _EXPOSURE_TO_ETF
    assert detect_exposure("Copper Mining") == "copper"
    assert detect_exposure("Gruvor & metaller") == "gold_miner"
    assert detect_exposure("Uranium") == "uranium"
    assert detect_exposure("Natural gas") == "gas"
    assert detect_exposure("Lithium mining") == "lithium"
    assert detect_exposure("Förnybar energi") is None    # "energi" är inte olja
    assert detect_exposure("Oil & Gas E&P") == "oil"
    for exp in ("uranium", "lithium", "gas"):
        assert exp in EXPOSURE_TO_RATIO and exp in _EXPOSURE_TO_ETF
    assert _EXPOSURE_TO_ETF["junior_miner"] == "GDXJ"


# ── Döda tickers ─────────────────────────────────────────────────────────────
def test_dead_ticker_registry_renames_and_drops():
    assert dead.alive(["GOLD", "MRO", "NEM", "B", "gold"]) == ["B", "NEM"]
    assert dead.alive_map({"GOLD": 1, "X": 2, "FCX": 3}) == {"B": 1, "FCX": 3}
    assert dead.status("MAG").startswith("delisted") and dead.status("NEM") is None
    assert dead.rename("ERICB.ST") == "ERIC-B.ST"


def test_hardcoded_lists_carry_no_dead_tickers():
    import ticker_universe as tu
    from ember.universe import US_INTL_CURATED
    from heatmap.heatmap_streamlit import US_TICKERS, CANADA_TICKERS
    import wolf_shadow_screener as wss
    from utils.presets import PRESET_PARAMS_BT, PRESET_LABELS
    from blindspot.config import BLINDSPOT_TICKERS
    from blindspot.classification.sector_map import TICKER_OVERRIDES
    from contrarian_alpha.ui import _MARKETS

    # (journal_import.ISIN_MAP ingår inte: gamla affärer får peka på en
    # avnoterad ticker — det är historik, inte ett universum.)
    pools = [t for lst in tu.FALLBACK_TICKERS.values() for t in lst] + US_INTL_CURATED
    pools += list(US_TICKERS) + list(CANADA_TICKERS) + list(wss.COMMODITY_TICKERS)
    pools += [t for m in wss.MARKETS.values() if isinstance(m, dict) for t in m]
    pools += list(PRESET_PARAMS_BT) + PRESET_LABELS + BLINDSPOT_TICKERS + list(TICKER_OVERRIDES)
    pools += _MARKETS["US"]["manual_tickers"]
    bad = sorted({t for t in pools if dead.status(t)})
    assert bad == [], bad
    assert "LUMI.ST" in wss.MARKETS["stockholm"]
    assert "B" in PRESET_PARAMS_BT and "GOLD" not in PRESET_PARAMS_BT


def test_labels_that_named_the_wrong_company_are_fixed():
    import wolf_shadow_screener as wss
    all_labels = {}
    for m in wss.MARKETS.values():
        if isinstance(m, dict):
            all_labels.update(m)
    assert "Lundin" not in all_labels.get("LUND-B.ST", "")
    assert all_labels.get("LUMI.ST", "").startswith("Lundin")
    assert "Outokumpu" not in all_labels.get("SSABBH.HE", "")
    src = open(os.path.join(ROOT, "long_trend", "long_trend_streamlit.py"), encoding="utf-8").read()
    assert '"LUMI.ST":   {"name": "Lundin Mining"' in src
    assert '"LUND-B.ST": {"name": "Lundbergföretagen B"' in src


def test_resource_csv_validator_flags_dead_tickers_and_accepts_share_classes():
    from scripts.validate_resource_universe import validate_resource_universe, _YF_TICKER_RE
    res = validate_resource_universe()
    assert res.ok, res.errors
    assert _YF_TICKER_RE.match("TECK-B.TO") and _YF_TICKER_RE.match("B")
    import csv, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "yf_ticker", "name", "country", "stage", "primary_commodity"])
        w.writerow(["GOLD", "GOLD", "Barrick", "US", "producer", "gold"])
        path = fh.name
    bad = validate_resource_universe(path)
    assert not bad.ok and any("renamed" in e for e in bad.errors)


# ── Swing-universumet ────────────────────────────────────────────────────────
def test_wolf_data_filters_on_instrument_type_and_nordic_markets():
    src = open(os.path.join(ROOT, "wolf_data.py"), encoding="utf-8").read()
    assert '"instrumentType"' in src and 'ins[type_col] == 0' in src
    assert 'ins["instrument"] == 0' not in src
    assert "nordic_stock_market_ids()" in src
