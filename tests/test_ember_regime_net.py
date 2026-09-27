"""
Ember Regime dömer på netto (gröna − röda) i stället för att räkna gröna.
Gul och DATA_GAP är 0, inte nej. Två röda är AV oavsett.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ember import config as cfg  # noqa: E402
from ember import regime as rg  # noqa: E402


def _p(*statuses):
    return [rg.PillarResult(name=f"P{i}", status=s, value="", detail="") for i, s in enumerate(statuses)]


def test_verdict_by_net_not_green_count():
    # 3 gröna + 2 gula: PÅ (var SELEKTIV)
    v, text = rg._complex_verdict(_p("GREEN", "GREEN", "GREEN", "AMBER", "AMBER"), "adelmetaller")
    assert v == rg.VERDICT_PA and "netto +3" in text and "ÄDELMETALLER" in text
    # 2 gröna + 3 gula: netto +2 → SELEKTIV (var AV)
    v, _ = rg._complex_verdict(_p("GREEN", "GREEN", "AMBER", "AMBER", "AMBER"), "energi")
    assert v == rg.VERDICT_SELEKTIV
    # 4 gröna + 1 röd: PÅ (netto +3)
    assert rg._complex_verdict(_p("GREEN", "GREEN", "GREEN", "GREEN", "RED"), "energi")[0] == rg.VERDICT_PA
    # 3 gröna + 1 gul + 1 röd: netto +2 → SELEKTIV
    assert rg._complex_verdict(_p("GREEN", "GREEN", "GREEN", "AMBER", "RED"), "energi")[0] == rg.VERDICT_SELEKTIV
    # 2 gröna + 2 gula + 1 röd: netto +1 → SELEKTIV
    assert rg._complex_verdict(_p("GREEN", "GREEN", "AMBER", "AMBER", "RED"), "agri")[0] == rg.VERDICT_SELEKTIV
    # två röda: AV oavsett netto (3 gröna + 2 röda = +1)
    v, text = rg._complex_verdict(_p("GREEN", "GREEN", "GREEN", "RED", "RED"), "basmetaller")
    assert v == rg.VERDICT_AV and "Korten visas ändå" in text
    # 1 grön + 2 gula + 1 röd + 1 gap: netto 0 → SELEKTIV, gap i texten
    v, text = rg._complex_verdict(_p("GREEN", "AMBER", "AMBER", "RED", "DATA_GAP"), "energi")
    assert v == rg.VERDICT_SELEKTIV and "1 DATA_GAP" in text
    # allt gult: netto 0 → SELEKTIV, inte AV
    assert rg._complex_verdict(_p("AMBER", "AMBER", "AMBER", "AMBER", "AMBER"), "energi")[0] == rg.VERDICT_SELEKTIV
    # 1 grön + 4 gap: SELEKTIV (var AV)
    assert rg._complex_verdict(_p("GREEN", "DATA_GAP", "DATA_GAP", "DATA_GAP", "DATA_GAP"), "energi")[0] == rg.VERDICT_SELEKTIV


def test_counts_and_config():
    assert rg.regime_counts(_p("GREEN", "AMBER", "RED", "DATA_GAP", "GREEN")) == (2, 1, 1, 1)
    assert cfg.REGIME_PILLAR_POINTS == {"GREEN": 1, "AMBER": 0, "RED": -1, "DATA_GAP": 0}
    assert cfg.REGIME_PA_MIN > cfg.REGIME_SELEKTIV_MIN and cfg.REGIME_AV_RED_MIN == 2


def test_complex_result_carries_net_and_texts(monkeypatch):
    pillars = _p("GREEN", "GREEN", "GREEN", "AMBER", "AMBER")
    for fn in ("_ep_dxy", "_ep_xle_rs", "_ep_oil_trend", "_ep_gas_trend", "_ep_energy_breadth"):
        monkeypatch.setattr(rg, fn, (lambda p: (lambda: p))(pillars[0]))
    monkeypatch.setattr(rg, "_ep_oil_trend", lambda: pillars[3])
    monkeypatch.setattr(rg, "_ep_gas_trend", lambda: pillars[4])
    r = rg.compute_energi_regime.__wrapped__() if hasattr(rg.compute_energi_regime, "__wrapped__") \
        else rg.compute_energi_regime()
    assert r.green_count == 3 and r.red_count == 0 and r.net_score == 3 and r.verdict == rg.VERDICT_PA
    import strategy_rules as sr
    text = " ".join(r_.explanation for r_ in sr.EMBER_PB.entry)
    assert "Netto = gröna − röda" in text and "≥4 gröna" not in text
    assert "≥4 gröna" not in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             "ovtlyr", "ui", "rules_page.py"), encoding="utf-8").read().split("EMBER Regime")[1][:400]
