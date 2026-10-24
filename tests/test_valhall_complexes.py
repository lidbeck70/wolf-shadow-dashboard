"""
K1 — Valhalls sex råvarukomplex (valhall/complexes.py): tema → komplex (EMBER:s indelning), bolag → komplex via
registrets tema eller Börsdatas bransch, komplexdrivaren som likaviktad korg av temadrivarna, och temaraderna ur
kartan grupperade per komplex. Inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import themes as bth  # noqa: E402
from valhall import complexes as vcx  # noqa: E402
from valhall import segments as sgm  # noqa: E402


def test_every_theme_belongs_to_exactly_one_complex():
    assert [k for k, _ in vcx.COMPLEXES] == ["energi", "basmetaller", "adelmetaller", "agri", "skog", "frakt"]
    grouped = [t for k in vcx.KEYS for t in vcx.themes_of(k)]
    assert sorted(grouped) == sorted(bth.THEMES) and len(grouped) == len(bth.THEMES)
    assert vcx.complex_of("skog") == "skog" and bth.complex_of("skog") == "agri"      # EMBER:s indelning, inte BERSERK:s
    assert vcx.complex_of("tank") == "frakt" and vcx.complex_of("guld") == "adelmetaller"
    assert vcx.complex_of("agri_korg") == "agri" and vcx.complex_of("x") is None and vcx.complex_of(None) is None
    for k in vcx.KEYS:
        assert k in vcx.LABEL and 0 < vcx.NECESSITY[k] <= 100
    # alla teman som ligger under Valhalls segment hittar ett komplex
    assert all(vcx.complex_of(t) for s in sgm.SEGMENTS for t in s.themes)


def test_company_to_complex_via_theme_then_branch():
    assert vcx.complex_for("koppar") == "basmetaller" and vcx.complex_for("koppar", 18) == "basmetaller"   # temat vinner
    assert vcx.complex_for(None, 18) == "adelmetaller" and vcx.complex_for("", 7) == "energi"
    assert vcx.complex_for(None, 41) == "frakt" and vcx.complex_for(None, 61) == "agri" and vcx.complex_for(None, 21) == "skog"
    assert vcx.complex_for(None, 15) is None and vcx.complex_for(None, "x") is None and vcx.complex_for() is None
    reg = SimpleNamespace(members={"BOL.ST": SimpleNamespace(theme="koppar", branch_id=17),
                                   "GOLD.ST": SimpleNamespace(theme="", branch_id=18),
                                   "ERIC-B.ST": SimpleNamespace(theme="", branch_id=23)})
    assert vcx.member_complexes(reg) == {"BOL.ST": "basmetaller", "GOLD.ST": "adelmetaller"}
    assert vcx.member_complexes(None) == {}


def test_basket_is_an_equal_weight_index_of_the_theme_drivers():
    idx = pd.bdate_range("2025-01-01", periods=300)
    up = pd.Series(np.linspace(100, 200, 300), index=idx)
    down = pd.Series(np.linspace(100, 50, 300), index=idx)
    drivers = {"koppar": ("HG=F", up), "stal": ("HRC=F", down), "olja": ("BZ=F", up)}
    syms, s = vcx.basket("basmetaller", drivers)
    assert syms == ["koppar:HG=F", "stal:HRC=F"] and s is not None and abs(s.iloc[0] - 100) < 1e-6
    assert s.iloc[-1] > 100 and s.iloc[-1] < up.iloc[-1]                           # mellan de två benen
    syms, s = vcx.basket("skog", drivers)                                            # inget ben alls
    assert syms == [] and s is None
    syms, s = vcx.basket("skog", {}, extra={"skog": up})                            # producentkorg som reserv
    assert syms == ["skog:producentkorg"] and s is not None
    syms, s = vcx.basket("frakt", {"tank": ("BWET", None), "torrbulk": ("BDRY", up)})
    assert syms == ["torrbulk:BDRY"]                                                # ett tema räcker


def test_theme_rows_group_by_complex_once():
    rows = [{"key": "energy", "themes": [{"theme": "olja", "label": "Olja", "state": {}}, {"theme": "tank", "state": {}}]},
            {"key": "materials", "themes": [{"theme": "guld", "state": {}}, {"theme": "skog", "state": {}}, "x"]},
            {"key": "consumer_staples", "themes": [{"theme": "lax", "state": {}}, {"theme": "olja", "state": {}}]}]
    g = vcx.complex_themes(rows)
    assert [t["theme"] for t in g["energi"]] == ["olja"] and [t["theme"] for t in g["frakt"]] == ["tank"]
    assert [t["theme"] for t in g["skog"]] == ["skog"] and [t["theme"] for t in g["agri"]] == ["lax"]
    assert g["basmetaller"] == [] and vcx.complex_themes(None) == {k: [] for k in vcx.KEYS}
