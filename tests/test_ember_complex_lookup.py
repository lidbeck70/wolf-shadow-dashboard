"""
Ember Regime: ett bolag får sitt komplex ur temakartan, ur ditt eget val
i fliken (data/ember.json) eller ur råvaran i arket — inte genom att du
redigerar ember/config.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from confidence import commodities as com  # noqa: E402
from confidence import store as cs  # noqa: E402
from ember import config as cfg  # noqa: E402
from ember import regime as rg  # noqa: E402


def test_every_sheet_commodity_has_a_complex():
    for key in com.REGISTRY:
        assert cfg.COMMODITY_TO_COMPLEX.get(key) in cfg.COMPLEX_LABEL, key


def _stores(monkeypatch, conf=None, ember=None):
    import streamlit as st
    import storage
    stores = {"confidence": conf if conf is not None else cs.default(), "ember": ember if ember is not None else {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name) if stores.get(name) is not None else default))
    saved = []
    monkeypatch.setattr(storage, "save_session", lambda name: saved.append(name) or type("R", (), {"ok": True})())
    return saved


def test_detect_complex_uses_sheet_commodity_and_overrides(monkeypatch):
    import streamlit as st
    conf = cs.default()
    visc = dcs.copper_developer()
    visc.ticker = "VISC.ST"
    cs.put(conf, visc)
    saved = _stores(monkeypatch, conf=conf)
    st.session_state.clear()
    assert rg.detect_complex("FCX") == "basmetaller"                 # temakartan
    assert rg.detect_complex("visc.st") == "basmetaller"             # råvaran i arket: copper
    assert rg.detect_complex("OKÄND") is None                        # okänt är okänt, aldrig energi
    assert rg.set_complex_override("OKÄND", "agri") and saved == ["ember"]
    assert rg.detect_complex("okänd") == "agri"                      # ditt val
    assert rg.complex_overrides() == {"OKÄND": "agri"}
    assert rg.set_complex_override("OKÄND", None)
    assert rg.detect_complex("OKÄND") is None
    # temakartan vinner över arket
    fcx = dcs.gold_producer()
    fcx.ticker = "FCX"
    cs.put(conf, fcx)
    assert rg.detect_complex("FCX") == "basmetaller"
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ember", "regime.py"),
               encoding="utf-8").read()
    assert "Lägg till tickern i ember/config.py" not in src            # inget "gå och redigera koden"
    assert "Kom ihåg" in src


def test_sector_etf_and_theme_follow_the_sheet_commodity(monkeypatch):
    import streamlit as st
    from ember import engine as e
    conf = cs.default()
    visc = dcs.copper_developer()
    visc.ticker = "VISC.ST"
    cs.put(conf, visc)
    lex = dcs.lithium_explorer()                                     # lithium → sallsynta → REMX
    cs.put(conf, lex)
    _stores(monkeypatch, conf=conf)
    st.session_state.clear()
    assert e.sector_etf_for("FCX") == "COPX" and e.sector_etf_for("CCJ") == "URA"      # temakartan
    assert rg.detect_theme("visc.st") == "koppar" and e.sector_etf_for("VISC.ST") == "COPX"
    assert e.sector_etf_for("LEX") == "REMX"
    assert e.sector_etf_for("OKÄND") == cfg.DEFAULT_SECTOR_ETF                          # GLD bara när okänt
    rg.set_complex_override("OKÄND", "energi")
    assert rg.detect_theme("OKÄND") == "olja" and e.sector_etf_for("OKÄND") == "XLE"    # valt komplex → bärande tema
    for key in com.REGISTRY:
        assert cfg.COMMODITY_TO_THEME[key] in cfg.EMBER_SECTOR_ETF, key
    for cx in cfg.COMPLEX_LABEL:
        assert cfg.COMPLEX_DEFAULT_THEME[cx] in cfg.EMBER_SECTOR_ETF


def test_theme_from_register_sector_text(monkeypatch):
    import streamlit as st
    import positions
    rows = [{"ticker": "DOFG.OL", "name": "DOF Group", "strategy": "Wolf", "sector": "Olja & offshore"},
            {"ticker": "NYX", "name": "x", "strategy": "Viking", "sector": "Unknown"}]
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: rows)
    _stores(monkeypatch)
    st.session_state.clear()
    assert rg.theme_from_sector_text("Olja & offshore") == "olja"
    assert rg.theme_from_sector_text("Guldgruva") == "guld" and rg.theme_from_sector_text("Bank") is None
    assert rg.register_sector("dofg.ol") == "Olja & offshore" and rg.register_sector("NYX") is None
    assert rg.detect_theme("DOFG.OL") == "olja" and rg.detect_complex("DOFG.OL") == "energi"
    from ember import engine as e
    assert e.sector_etf_for("DOFG.OL") == "XLE"
    assert rg.detect_theme("NYX") is None                              # Unknown ger inget tema
