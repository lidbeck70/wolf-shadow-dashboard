"""
Tester för hat-poängens omdesign (Deep Contrarian-fixen).

Kärnan som testas: de sju komponenterna nås med Börsdata/yfinance/prisserien,
saknad data ger 0 poäng och räknas bort ur nåbart max (inga fabricerade
"moderata defaultpoäng"), och tröskeln 45 är därmed nåbar på riktigt.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha import hate as h
from contrarian_alpha import engine as eng


def _price(**over):
    base = {"close": 60.0, "sma200": 80.0, "high_52w": 120.0, "low_52w": 58.0,
            "avg_price_5y": 100.0, "avg_volume_20d": 3000.0,
            "avg_volume_6m": 10000.0}
    base.update(over)
    return base


# ── Komponenterna ────────────────────────────────────────────────────────────
def test_missing_data_scores_zero_not_a_fabricated_default():
    """Gamla motorn gav 21 defaultpoäng av tröskelns 45 för data som aldrig
    hämtades — grinden sorterade på brus. Saknat = 0 och omätt."""
    assert h._score_short_interest(None) == (0.0, False)
    assert h._score_valuation_depression(None) == (0.0, False)
    assert h._score_sector_outflow(None) == (0.0, False)
    assert h._score_volume_drought({}) == (0.0, False)
    assert h._score_sma200_gap({}) == (0.0, False)


def test_fi_registry_zero_is_real_data():
    """En nordisk aktie som saknas i FI-registret ligger under 0,5 %-golvet —
    det är RIKTIG låg blankning, inte saknad data."""
    pts, real = h._score_short_interest({"short_float_pct": 0.0,
                                         "source": "fi_registry"})
    assert pts == 0.0 and real is True


def test_valuation_depression_tiers():
    hist = list(range(1, 21))    # 1..20, 20 punkter
    pts, real = h._score_valuation_depression({"current": 1.5, "history": hist})
    assert (pts, real) == (15.0, True)      # billigaste tiondelen
    pts, _ = h._score_valuation_depression({"current": 21.0, "history": hist})
    assert pts == 0.0                        # dyraste läget
    # för tunn historik → omätt
    pts, real = h._score_valuation_depression({"current": 5, "history": [4, 6, 7]})
    assert (pts, real) == (0.0, False)
    # negativ multipel (förlustår) → omätt
    pts, real = h._score_valuation_depression({"current": -3, "history": hist})
    assert (pts, real) == (0.0, False)


def test_volume_drought_tiers():
    assert h._score_volume_drought(_price(avg_volume_20d=3000)) == (13.0, True)
    assert h._score_volume_drought(_price(avg_volume_20d=7000)) == (6.0, True)
    assert h._score_volume_drought(_price(avg_volume_20d=12000)) == (0.0, True)
    assert h._score_volume_drought(_price(avg_volume_6m=None)) == (0.0, False)


def test_sector_relative_prefers_universe_computation():
    pts, real = h._score_sector_outflow({"stock_vs_sector_3m": -25.0})
    assert (pts, real) == (15.0, True)
    pts, _ = h._score_sector_outflow({"stock_vs_sector_3m": 3.0})
    assert pts == 0.0
    # gamla ETF-nycklarna fungerar fortfarande (EODHD-vägen)
    pts, real = h._score_sector_outflow({"sector_vs_market_3m": -12.0})
    assert real is True and pts > 0


# ── Normaliseringen ──────────────────────────────────────────────────────────
def test_price_only_row_normalizes_against_reachable_max():
    """Nordisk rad utan ins_id: mätbart = pris (15+12+15) + volym (13) = 55.
    Poängen skalas mot 55, så en djupt hatad aktie NÅR tröskeln 45 — den
    gamla motorn krävde poäng från källor som aldrig hämtades."""
    r = h.calculate_hate_score(price_data=_price())
    raw = sum(v for k, v in r.breakdown.items() if k != "sentiment_bonus")
    assert r.score == round(min(100.0, raw / 55 * 100), 1)
    assert r.score >= h.HAT_THRESHOLD
    assert r.confidence == 0.55


def test_too_little_coverage_disables_normalization():
    """Skyddsräcket: under 50p mätbar rymd normaliseras inte — två mätta
    komponenter får inte blåsa upp sig till 100."""
    r = h.calculate_hate_score(price_data={"close": 60.0, "sma200": 80.0,
                                           "high_52w": 120.0, "low_52w": 58.0})
    # mätbart: sma 15 + 52w 12 = 27 + cykel omätt (ingen avg) → under 50
    assert "HATE_LOW_COVERAGE" in r.flags
    assert r.score <= 27.0


def test_full_data_uses_raw_hundred_scale():
    r = h.calculate_hate_score(
        price_data=_price(),
        short_data={"short_float_pct": 25.0},
        valuation_data={"current": 1.0, "history": list(range(1, 21))},
        sector_data={"stock_vs_sector_3m": -25.0},
    )
    assert r.confidence == 1.0
    assert set(r.breakdown) >= {"sma200_gap", "low_52w_proximity",
                                "cycle_position", "short_interest",
                                "valuation_depression", "volume_drought",
                                "sector_outflow"}


def test_sentiment_bonus_is_capped_and_optional():
    base = h.calculate_hate_score(price_data=_price())
    boosted = h.calculate_hate_score(
        price_data=_price(),
        analyst_data={"downgrades_90d": 9, "upgrades_90d": 0,
                      "consensus": "sell"},
        sentiment_data={"message_count": 0, "bear_ratio": 0.9,
                        "confidence": 1.0},
    )
    assert boosted.score >= base.score
    assert boosted.score - base.score <= h._MAX_BONUS + 0.11  # avrundning


def test_no_price_data_still_zero():
    r = h.calculate_hate_score(price_data={})
    assert r.score == 0.0 and "NO_PRICE_DATA" in r.flags


# ── Börsdata-integrationen ───────────────────────────────────────────────────
class _FakeAPI:
    is_configured = True

    def __init__(self):
        self.batch_calls = []

    def get_instruments(self):
        return [{"insId": 5, "ticker": "BOL", "name": "Boliden",
                 "instrumentType": 1, "marketId": 1, "branchId": 10,
                 "sectorId": 3},
                {"insId": 7, "ticker": "ERIC B", "name": "Ericsson B",
                 "instrumentType": 1, "marketId": 1, "branchId": 11,
                 "sectorId": 4}]

    def get_branches(self):
        return [{"id": 10, "name": "Gruvor"}, {"id": 11, "name": "Telekom"}]

    def get_sectors(self):
        return [{"id": 3, "name": "Råvaror"}, {"id": 4, "name": "Teknik"}]

    def get_kpi_history_batch(self, ins_ids, kpi_id):
        self.batch_calls.append((tuple(ins_ids), kpi_id))
        if kpi_id == 11:   # ev_ebitda: bara ins 5 har historik
            return {5: [{"y": 2016 + i, "v": float(4 + i)} for i in range(10)]}
        if kpi_id == 2:    # pe: reserven för ins 7
            return {7: [{"y": 2016 + i, "v": float(10 + i)} for i in range(10)]}
        return {}


def test_manual_tickers_resolve_to_borsdata():
    """'BOL' utan suffix ska bli Boliden med ins_id, sektor och .ST —
    'söker ej med mer än ett värde'-buggens andra halva."""
    cfg = eng.PipelineConfig(market_ids=[], manual_tickers=["BOL", "ERIC-B",
                                                            "AAPL"])
    uni = eng._build_universe(cfg, _FakeAPI())
    by = {u["ticker"]: u for u in uni}
    assert by["BOL.ST"]["ins_id"] == 5
    assert by["BOL.ST"]["sector_name"] == "Råvaror"
    assert by["ERIC-B.ST"]["ins_id"] == 7          # bindestreck → mellanslag
    assert by["AAPL"]["ins_id"] is None            # okänd → yfinance-väg


def test_batch_valuation_uses_ev_ebitda_then_pe():
    api = _FakeAPI()
    universe = [{"ticker": "BOL.ST", "ins_id": 5},
                {"ticker": "ERIC-B.ST", "ins_id": 7}]
    snaps = {5: {"ev_ebitda": 4.5}, 7: {"pe": 11.0}}
    out = eng._batch_valuation_data(universe, snaps, api)
    assert out[5]["metric"] == "ev_ebitda" and out[5]["current"] == 4.5
    assert out[7]["metric"] == "pe"
    # pe-batchen ska bara ha frågat om den tunna raden
    assert api.batch_calls[1][0] == (7,)


def test_short_positions_parsing(monkeypatch):
    import borsdata_api as bd
    api = bd.BorsdataAPI(api_key="x")
    monkeypatch.setattr(api, "_get", lambda path, **kw: {
        "list": [{"insId": 5, "positions": [{"position": 0.6},
                                            {"position": 1.2}]},
                 {"insId": 9, "position": 2.5},
                 {"insId": 11, "positions": []}]})
    out = api.get_short_positions()
    assert out == {5: 1.8, 9: 2.5}
