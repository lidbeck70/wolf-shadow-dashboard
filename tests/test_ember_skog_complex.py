"""
EMBER: skog är ett eget komplex (🌲 SKOG & MASSA) — SCA, Holmen och de andra skogsbolagen grindades förut mot
jordbrukskorgen DBA under AGRI & ÖVRIGT. Pelarna: WOOD-trend, WOOD/SPY, trävara (LBR=F), bostadsbyggande (XHB)
och skog-temats cykel. Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ember import config as cfg  # noqa: E402
from ember import regime as rg  # noqa: E402


def test_skog_maps_to_its_own_complex():
    assert cfg.THEME_TO_COMPLEX["skog"] == "skog" and cfg.COMMODITY_TO_COMPLEX["timber"] == "skog"
    assert cfg.COMPLEX_LABEL["skog"] == "SKOG & MASSA" and cfg.COMPLEX_DEFAULT_THEME["skog"] == "skog"
    assert cfg.THEME_TO_COMPLEX["vete"] == "agri" and cfg.THEME_TO_COMPLEX["agri"] == "agri"      # agri orört
    assert set(cfg.COMPLEX_LABEL) == {"energi", "adelmetaller", "basmetaller", "agri", "skog", "frakt"}
    from confidence.commodities import REGISTRY
    assert REGISTRY["timber"].ember_complex == "skog"


def _series(up: bool, n=300):
    idx = pd.bdate_range("2022-01-03", periods=n * 5)
    c = 100 + np.arange(len(idx)) * (0.05 if up else -0.05)
    return pd.Series(c, index=idx)


def test_skog_regime_uses_wood_lumber_and_housing(monkeypatch):
    seen = []

    def close(ticker, period):
        seen.append(ticker)
        return _series(up=True)

    monkeypatch.setattr(rg, "_close", close)
    monkeypatch.setattr(rg, "_theme_cycle", lambda key, label: rg.PillarResult(
        name=f"{label.upper()}-TEMA CYKEL", status="GREEN", value="Cykel: TIDIG", detail=key))
    getattr(rg.compute_skog_regime, "clear", lambda: None)()
    res = rg.compute_skog_regime()
    assert res.key == "skog" and res.label == "SKOG & MASSA" and len(res.pillars) == 5
    names = [p.name for p in res.pillars]
    assert names == ["WOOD vs 50V EMA + LUTNING", "WOOD/SPY RELATIV STYRKA", "TRÄVARA (LBR=F) vs 50V EMA",
                     "BOSTADSBYGGANDE (XHB) vs 50V EMA", "SKOG-TEMA CYKEL"]
    assert {"WOOD", "LBR=F", "XHB"} <= set(seen) and "DBA" not in seen
    assert res.pillars[0].status == "GREEN" and "skogs-upptrend" in res.pillars[0].detail
    getattr(rg.compute_skog_regime, "clear", lambda: None)()


def test_agri_pillar_texts_are_unchanged(monkeypatch):
    monkeypatch.setattr(rg, "_close", lambda t, p: _series(up=False))
    p = rg._ag_dba_trend()
    assert p.name == "DBA vs 50V EMA + LUTNING" and p.status == "RED" and p.detail == "DBA under 50V EMA → agri-nedtrend ⛔"
