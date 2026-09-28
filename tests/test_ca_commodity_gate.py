"""
Contrarian Alpha: råvarugrinden i Deep Contrarian-läget.

Strategiguiden säger "Råvarurelaterat: gruvbolag, guld, silver, olja, gas",
men nödvändighetsgrinden släppte in telekom, banker, livsmedel och
fastigheter (Infracom var etta). Grinden körs nu först i deep_contrarian;
Quality-läget är orört.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha import commodity_gate as cg  # noqa: E402
from contrarian_alpha import engine as eng  # noqa: E402
from contrarian_alpha.necessity import BORSDATA_BRANCH_MAP  # noqa: E402


def test_branch_ids_cover_oil_gas_coal_uranium_and_mining_only():
    assert set(cg.COMMODITY_BRANCH_IDS) == {1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 20}
    for bid in cg.COMMODITY_BRANCH_IDS:
        assert bid in BORSDATA_BRANCH_MAP, bid                     # samma löpnummer som necessity
    assert cg.by_branch(18) is True and cg.by_branch(5) is True
    assert cg.by_branch(94) is False                               # Telekomtjänster (Infracom)
    assert cg.by_branch(68) is False and cg.by_branch(63) is False # banker, livsmedel
    assert cg.by_branch(21) is False and cg.by_branch(60) is False # skog, jordbruk
    assert cg.by_branch(19) is False                               # ädelstenar
    assert cg.by_branch(9) is False                                # gasförsörjning är en utility
    assert cg.by_branch(None) is None and cg.by_branch("x") is None


def test_name_and_theme_fallbacks():
    assert cg.by_name("Gruv - Guld & Silver") is True and cg.by_name("Oil & Gas E&P") is True
    assert cg.by_name("Telekomtjänster") is False and cg.by_name("Skogsbolag") is False
    assert cg.by_name("Gasförsörjning") is False and cg.by_name("", None) is None
    themes = {"FCX": "koppar", "DBA": "agri", "MSFT": None}
    ok, label = cg.classify("FCX", theme_fn=themes.get)
    assert ok and label == "tema koppar"
    assert cg.classify("DBA", theme_fn=themes.get)[0] is False    # agri räknas inte här
    assert cg.classify("MSFT", theme_fn=themes.get) == (False, "okänd bransch")
    # bransch-id vinner över tema
    assert cg.classify("INFRA", 94, "Telekomtjänster", theme_fn=lambda t: "olja") == \
        (False, "bransch Telekomtjänster")
    assert cg.classify("BOL", 17, "Gruv - Industrimetaller") == (True, "Gruv – industrimetaller")


def test_missing_fields_are_listed_by_label():
    assert eng.missing_fields({"market_cap": 100.0, "fcf_m": 5.0, "roic": float("nan")}) == [
        "EBITDA-marginal", "Nettoskuld/EBITDA", "D/E", "ROIC", "P/FCF", "EV/EBITDA", "Omsättning"]
    assert eng.missing_fields(None)[0] == "Börsvärde"


def _universe():
    def row(t, bid, bname, sid=7):
        return {"ticker": t, "ins_id": hash(t) % 10_000, "branch_name": bname, "sector_name": "",
                "inst_info": {"name": t, "marketId": 1, "branchId": bid, "sectorId": sid}}
    return [row("INFRA.ST", 94, "Telekomtjänster", 9), row("BOL.ST", 17, "Gruv - Industrimetaller"),
            row("LUG.ST", 18, "Gruv - Guld & Silver"), row("EQNR.OL", 4, "Olja & Gas - Försäljning", 3),
            row("SHB.ST", 68, "Banker", 1), row("HOLM.ST", 21, "Skogsbolag")]


def _run(monkeypatch, mode):
    monkeypatch.setattr(eng, "_BORSDATA_AVAILABLE", False)
    monkeypatch.setattr(eng, "_build_universe", lambda cfg, api: _universe())
    monkeypatch.setattr(eng, "_batch_fetch_fundamentals", lambda ids, api, global_ids=None: {})
    monkeypatch.setattr(eng, "_fetch_price_df", lambda t, i, api: None)
    monkeypatch.setattr(eng, "_batch_valuation_data", lambda scan, snaps, api: {})
    seen = []

    def fake_single(ticker, ins_id, inst_info, fund_snap, price_df, branch_name, sector_name, config, api, **kw):
        seen.append(ticker)
        return eng.ContrairianAlphaResult(ticker=ticker, ins_id=ins_id, name=ticker, market="SE",
                                          sector=sector_name, branch=branch_name, composite_score=50.0)
    monkeypatch.setattr(eng, "_run_single_ticker", fake_single)
    res = eng.run_pipeline(eng.PipelineConfig(mode=mode, top_n=10, max_fund_workers=1, max_price_workers=1))
    return res, seen


def test_deep_contrarian_scans_only_commodities(monkeypatch):
    res, seen = _run(monkeypatch, "deep_contrarian")
    assert sorted(seen) == ["BOL.ST", "EQNR.OL", "LUG.ST"]         # data hämtas bara för råvaror
    out = {r.ticker: r for r in res.eliminated}
    assert out["INFRA.ST"].elimination_stage == "COMMODITY"
    assert "Ej råvara — bransch Telekomtjänster" == out["INFRA.ST"].elimination_reason
    assert {"SHB.ST", "HOLM.ST"} <= set(out)
    assert res.commodity_passed == 3 and res.pass_rates["commodity"] == "3/6 (50%)"
    assert res.pass_rates["necessity"].endswith("/3 (100%)")       # nödvändighet räknas mot råvarorna
    labels = {r.ticker: r.commodity_label for r in res.results}
    assert labels["LUG.ST"] == "Gruv – guld & silver" and labels["EQNR.OL"] == "Olja & gas – försäljning"


def test_quality_mode_is_unchanged(monkeypatch):
    res, seen = _run(monkeypatch, "quality")
    assert "INFRA.ST" in seen and res.commodity_passed is None
    assert "commodity" not in res.pass_rates
    assert not any(r.elimination_stage == "COMMODITY" for r in res.eliminated)

