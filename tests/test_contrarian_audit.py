"""
Tester för Contrarian Alpha-granskningen (probe 2026-09-09).

Tre buggar som samverkade: Börsdatas bransch-id lästes som GICS-koder,
marginaler och skuldsättningsgrad kom i fel enhet, och ROIC-grinden i
deep-läget avrättade cykelbottnar. Plus det nya larmbenet.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha import necessity as nec
from contrarian_alpha import engine as eng
from contrarian_alpha.engine import PipelineConfig, _run_single_ticker
import alert_rules as ar


# ── Necessity: Börsdata-id är löpnummer, inte GICS ──────────────────────────
def test_borsdata_branch_ids_map_to_the_right_business():
    """G5EN (branch 55 Gaming & Spel) fick 90 p som 'Allmännyttiga tjänster';
    Boliden (branch 17) föll till fallback 40 och eliminerades."""
    assert nec.get_necessity_for_borsdata(branch_id=55, sector_id=8).score == 8
    assert nec.get_necessity_for_borsdata(branch_id=17, sector_id=7).score == 90
    assert nec.get_necessity_for_borsdata(branch_id=4, sector_id=3).score == 90   # Equinor
    assert nec.get_necessity_for_borsdata(branch_id=27, sector_id=5).score == 66  # QleanAir
    assert nec.get_necessity_for_borsdata(branch_id=7, sector_id=3).score == 98   # Uran


def test_borsdata_sector_fallback_then_name_then_fallback():
    # okänd bransch → sektor
    assert nec.get_necessity_for_borsdata(branch_id=999, sector_id=7).score == 78
    # inga id → namn
    assert nec.get_necessity_for_borsdata(branch_name="Guldgruvor").score == 90
    # ingenting → fallback
    assert nec.get_necessity_for_borsdata().score == nec.FALLBACK_SCORE.score


def test_every_borsdata_branch_and_sector_is_covered():
    """Probe-listan har branscher 1–94 och sektorer 1–10 — alla ska finnas,
    annars faller ett helt segment till fallback 40 igen."""
    assert set(nec.BORSDATA_BRANCH_MAP) == set(range(1, 95))
    assert set(nec.BORSDATA_SECTOR_MAP) == set(range(1, 11))
    # kärnan i strategin ligger över tröskeln, underhållning under
    core = (1, 2, 3, 4, 5, 7, 8, 10, 16, 17, 18)
    fluff = (54, 55, 56, 57, 89, 90, 91)
    assert all(nec.BORSDATA_BRANCH_MAP[b].score >= 60 for b in core)
    assert all(nec.BORSDATA_BRANCH_MAP[b].score < 60 for b in fluff)


def test_engine_uses_borsdata_map_for_borsdata_rows():
    """Samma inst_info som probe gav för G5EN: genom motorn ska den
    elimineras på NECESSITY — inte passera som kraftbolag."""
    cfg = PipelineConfig(universe="nordic", mode="deep_contrarian")
    inst = {"name": "G5 Entertainment", "marketId": 3, "instrumentType": 1,
            "sectorId": 8, "branchId": 55}
    r = _run_single_ticker("G5EN.ST", 383, inst, {}, None, "Gaming & Spel",
                           "Sällanköpsvaror", cfg, None)
    assert r.eliminated and r.elimination_stage == "NECESSITY"
    assert r.necessity_score == 8

    inst = {"name": "Boliden", "marketId": 1, "instrumentType": 1,
            "sectorId": 7, "branchId": 17}
    r = _run_single_ticker("BOL.ST", 40, inst, {}, None, "Gruv - Industrimetaller",
                           "Material", cfg, None)
    assert r.necessity_score == 90
    assert r.elimination_stage != "NECESSITY"


# ── Enheter ──────────────────────────────────────────────────────────────────
def test_fundamentals_dict_restores_percent_and_keeps_ratio():
    """snapshot_fast delar marginalen med 100 (57.1 → 0.571) — strength och
    kortet räknar i %. D/E är redan en kvot (0.88) och ska inte röras."""
    fund = eng._build_fundamentals_dict({"ebitda_margin": 0.5709,
                                         "debt_to_equity": 0.878})
    assert abs(fund["ebitda_margin"] - 57.09) < 0.01
    assert fund["debt_to_equity"] == 0.878


def test_snapshot_fast_no_longer_divides_debt_to_equity():
    import borsdata_api as bd
    src = open(bd.__file__, encoding="utf-8").read()
    assert '(KPI["debt_to_equity"], 100)' not in src
    assert '(KPI["debt_to_equity"], 1)' in src


# ── Deep-lägets grindar genom motorn ─────────────────────────────────────────
class _HistAPI:
    """Fake Börsdata: bara KPI-historik. roic (37) och nd/ebitda (42)."""
    is_configured = True

    def __init__(self, roic_hist=None, nd_hist=None):
        self._h = {37: roic_hist or [], 42: nd_hist or []}

    def get_kpi_history(self, ins_id, kpi_id, report_type, price_type):
        return [{"y": 2020 + i, "v": v} for i, v in enumerate(self._h.get(kpi_id, []))]

    def get_kpi_screener(self, *a, **k):
        return []

    def get_insider_transactions(self, ins_id):
        return []


def _deep_cfg():
    # hate_threshold 0 så raden (utan prisdata) når balansräkning + quality
    return PipelineConfig(universe="nordic", mode="deep_contrarian",
                          hate_threshold=0, market_ids=[])


def _bol_inst():
    return {"name": "Boliden", "marketId": 1, "instrumentType": 1,
            "sectorId": 7, "branchId": 17, "reportCurrency": "SEK",
            "stockPriceCurrency": "SEK"}


def _snap(**over):
    base = {"fcf_m": 4532.0, "ebitda_margin": 0.264, "total_equity_m": 80482.0,
            "total_assets_m": 151132.0, "revenue_m": 103657.0,
            "market_cap": 166214.0, "net_debt_ebitda": 0.7, "roic": 0.106}
    base.update(over)
    return base


def test_deep_leverage_gate_is_net_debt_to_ebitda():
    """Boliden: skuldsättningsgrad 0.88 fällde 'D/E < 0.6' med rätt enhet —
    men ND/EBITDA 0.7 är sund. Grinden är ND/EBITDA ≤ 3.0; nettokassa passerar."""
    r = _run_single_ticker("BOL.ST", 9101, _bol_inst(), _snap(net_debt_ebitda=0.7),
                           None, "Gruv - Industrimetaller", "Material",
                           _deep_cfg(), _HistAPI(roic_hist=[12, 14, 11]))
    assert r.elimination_stage != "BALANCE_SHEET", r.elimination_reason
    r = _run_single_ticker("X.ST", 9102, _bol_inst(), _snap(net_debt_ebitda=4.2),
                           None, "Gruv - Industrimetaller", "Material",
                           _deep_cfg(), _HistAPI(roic_hist=[12, 14, 11]))
    assert r.eliminated and r.elimination_stage == "BALANCE_SHEET"
    assert "Net Debt/EBITDA 4.2" in r.elimination_reason
    r = _run_single_ticker("Y.ST", 9103, _bol_inst(), _snap(net_debt_ebitda=-0.5),
                           None, "Gruv - Industrimetaller", "Material",
                           _deep_cfg(), _HistAPI(roic_hist=[12, 14, 11]))
    assert r.elimination_stage != "BALANCE_SHEET"


def test_deep_roic_gate_looks_through_the_cycle():
    """ROIC 6 % i botten men median 12 % genom cykeln → passera + ROIC_TROUGH.
    ROIC 6 % OCH median 5 % → eliminera. (20 av 22 föll på dagens ROIC.)"""
    trough = _run_single_ticker("T.ST", 9201, _bol_inst(), _snap(roic=0.06),
                                None, "Gruv - Industrimetaller", "Material",
                                _deep_cfg(), _HistAPI(roic_hist=[9, 14, 12]))
    assert trough.elimination_stage != "QUALITY_GATE", trough.elimination_reason
    assert "ROIC_TROUGH" in trough.all_flags

    weak = _run_single_ticker("W.ST", 9202, _bol_inst(), _snap(roic=0.06),
                              None, "Gruv - Industrimetaller", "Material",
                              _deep_cfg(), _HistAPI(roic_hist=[4, 6, 5]))
    assert weak.eliminated and weak.elimination_stage == "QUALITY_GATE"
    assert "median 5.0%" in weak.elimination_reason


def test_currency_mismatch_drops_market_cap_from_altman():
    """G5EN: fundamenta i USD, börsvärde i SEK → Altman 27.7. Vid valutamix
    tas börsvärdet bort (Altman räknar på resten) och raden flaggas."""
    inst = dict(_bol_inst(), reportCurrency="USD", stockPriceCurrency="SEK")
    r = _run_single_ticker("G.ST", 9301, inst, _snap(), None,
                           "Gruv - Industrimetaller", "Material",
                           _deep_cfg(), _HistAPI(roic_hist=[12, 14, 11]))
    assert "CURRENCY_MISMATCH" in r.all_flags
    same = _run_single_ticker("S.ST", 9302, _bol_inst(), _snap(), None,
                              "Gruv - Industrimetaller", "Material",
                              _deep_cfg(), _HistAPI(roic_hist=[12, 14, 11]))
    assert "CURRENCY_MISMATCH" not in same.all_flags


# ── Larmbenet ────────────────────────────────────────────────────────────────
def _ca(*tickers, ts="2026-09-09T15:00"):
    return {"timestamp": ts,
            "results": [{"ticker": t, "name": t, "composite_score": 61.5,
                         "rank": i + 1, "necessity_score": 90, "hat_score": 55,
                         "sector": "Material"} for i, t in enumerate(tickers)]}


def test_contrarian_alerts_on_entering_the_list_once():
    _a, state = ar.contrarian_alerts(_ca("BOL.ST"), None)
    assert _a == []
    alerts, state = ar.contrarian_alerts(_ca("BOL.ST", "LUMI.ST"), state)
    assert len(alerts) == 1 and "LUMI.ST" in alerts[0]["title"]
    assert "necessity 90" in alerts[0]["body"]
    alerts, _s = ar.contrarian_alerts(_ca("BOL.ST", "LUMI.ST"), state)
    assert alerts == []


def test_contrarian_unreadable_source_freezes_baseline():
    _a, state = ar.contrarian_alerts(_ca("BOL.ST"), None)
    _a2, kept = ar.contrarian_alerts({"timestamp": None, "results": []}, state)
    assert kept == state
    alerts, _s = ar.contrarian_alerts(_ca("BOL.ST"), kept)
    assert alerts == []


def test_evaluate_routes_contrarian_leg():
    regime = {"regime": "GRÖN", "rules": []}
    _a, state = ar.evaluate(regime, {"top": []}, {"positions": []}, [], None, {})
    alerts, state2 = ar.evaluate(regime, {"top": []}, {"positions": []}, [], state,
                                 {"contrarian": {"enabled": True,
                                                 "channels": ["email"]}},
                                 contrarian_data=_ca("BOL.ST"))
    kinds = {a["kind"]: a for a in alerts}
    assert kinds["contrarian_deep"]["channels"] == ["email"]
    assert "BOL.ST" in state2["contrarian"]["ranked"]


# ── Insiderregistret (holdings-API:t) ────────────────────────────────────────
def _recent(days_ago):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT00:00:00")


def test_insider_transactions_come_from_holdings_api(monkeypatch):
    """/insiders/{id} svarar 404; /holdings/insider?instList= är rätt väg
    (probe 2026-09-09) och svarar list/insId/values."""
    import borsdata_api as bd
    api = bd.BorsdataAPI(api_key="x")
    calls = []

    def fake_get(path, params=None, **kw):
        calls.append((path, params))
        return {"list": [{"insId": 40, "values": [{"shares": 1000, "transactionType": 19}]},
                         {"insId": 904, "values": [{"shares": -25, "transactionType": 25}]}]}
    monkeypatch.setattr(api, "_get", fake_get)
    out = api.get_insider_transactions(40)
    assert out == [{"shares": 1000, "transactionType": 19}]
    assert calls[0][0] == "/holdings/insider" and calls[0][1] == {"instList": "40"}


def test_insider_direction_comes_from_the_sign_of_shares():
    """Börsdatas typkoder är numeriska — riktningen sitter i tecknet på
    shares; incitamentsprogram (equityProgram) räknas inte som köp."""
    from contrarian_alpha import catalyst as cat

    class _API:
        def get_insider_transactions(self, ins_id):
            return [
                {"shares": 1000, "transactionType": 19, "transactionDate": _recent(30)},
                {"shares": 500, "transactionType": 18, "transactionDate": _recent(60),
                 "equityProgram": True},                       # program → ignoreras
                {"shares": -300, "transactionType": 25, "transactionDate": _recent(90)},
                {"shares": 9999, "transactionType": 19, "transactionDate": _recent(900)},  # >12m
            ]

    # unikt ins_id så cachen inte återanvänder ett tidigare test
    d = cat.fetch_insider_data(777001, _API())
    assert d["insider_buy_count"] == 1
    assert d["insider_sell_count"] == 1
    assert d["insider_net_bought_12m"] == 700


def test_insider_flag_only_when_nothing_could_be_measured():
    """Holdings-API:t ger köp/sälj men ingen ägarandel — det är inte
    'insiderdata saknas'. Flaggan ska bara sättas när båda delarna saknas."""
    from contrarian_alpha.catalyst import calculate_catalyst_score
    price = {"close": 10.0, "sma50": 9.0, "sma50_slope": 0.1, "current_volume": 100.0,
             "avg_volume_20d": 100.0, "std_volume_20d": 10.0,
             "close_history": [10.0, 9.8, 9.7, 9.9, 10.1]}
    with_tx = calculate_catalyst_score(price_data=price, ticker="X",
                                       insider_data={"insider_net_bought_12m": 700,
                                                     "insider_buy_count": 1,
                                                     "insider_sell_count": 1})
    assert "INSIDER_DATA_MISSING" not in with_tx.flags
    assert "INSIDER_OWNERSHIP_NA" in with_tx.flags
    none = calculate_catalyst_score(price_data=price, ticker="X", insider_data={})
    assert "INSIDER_DATA_MISSING" in none.flags
