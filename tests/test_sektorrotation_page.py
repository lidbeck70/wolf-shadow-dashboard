"""ODIN → Sektorrotation (odin/sektorrotation.py): fliken, raderna, valideringsmärket och Valhalls status."""
import pathlib

import numpy as np
import pandas as pd

from odin import sektorrotation as sr
from ui import nav

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _getter(sym, period):
    idx = pd.bdate_range("2021-10-01", "2026-10-23")
    c = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(sum(map(ord, sym))).normal(0.0003, 0.01, len(idx)))),
                  index=idx)
    return pd.DataFrame({"Close": c, "Volume": 1e6})


def test_tab_sits_in_odin_after_the_map():
    assert nav.options("odin")[:3] == ["Läget", "Kartan", "Sektorrotation"]
    assert nav.resolve(nav.target_slug("sector_rotation")) == ["odin", "Sektorrotation"]
    assert "render_sektorrotation_page" in (ROOT / "wolf_panel.py").read_text(encoding="utf-8")


def test_validation_label_follows_the_backtest():
    assert sr.validation(None)[0] == "EJ TESTAD"
    assert sr.validation({"verdict": {"R1": True, "R2": True, "S": False, "use": "x"}})[0] == "GODKÄND"
    assert sr.validation({"verdict": {"R1": False, "R2": True, "use": "x"}})[0] == "RISKFILTER"
    assert sr.validation({"verdict": {"R1": True, "R2": False}})[0] == "EJ VALIDERAD"


def test_valhall_status_maps_segments_to_funds():
    board = {"segments": [{"key": "energy", "global": {"status": "GRÖN"}}, {"key": "technology", "global": None},
                          "skräp"]}
    assert sr.valhall_status(board) == {"XLE": "GRÖN", "XLK": None}
    assert sr.valhall_status(None) == {} and sr.valhall_status(["x"]) == {}


def test_ui_rows_and_climb_text():
    rows = sr.ui_rows([{"ticker": "XLE", "label": "Energi", "phase": "2 Bekräftad ledare", "S": 90.0, "R12": 100.0,
                        "R6": 80.0, "R3": 90.0, "f1": True, "f2": False, "f3": True, "climb": 2.0}], {"XLE": "RÖD"})
    assert rows[0]["Filter"] == "✓✗✓" and rows[0]["Valhall"] == "RÖD" and rows[0]["Sektor"] == "Energi"
    assert sr.climb_text([("XLE", 4), ("XLB", 0), ("XLK", -3)]) == "upp: XLE +4   ·   ned: XLK -3"


def test_compute_live_ranks_sectors_and_producers():
    out = sr.compute_live(_getter)
    assert len(out["sectors"]) == 11 and len(out["producers"]) == 15
    assert out["sectors"][0]["S"] >= out["sectors"][-1]["S"]
    assert out["date"] and out["climb_1m"]
    assert sr.compute_live(lambda s, p: pd.DataFrame())["error"]


def test_page_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def app():
        import market_prices

        import gist_storage
        from odin import sektorrotation as page
        from tests.test_sektorrotation_page import _getter
        market_prices.ohlcv = _getter
        gist_storage.load_blob = lambda name, default=None: None
        page.render_sektorrotation_page()

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception
    assert any("EJ TESTAD" in m.value for m in at.markdown)
    assert len(at.dataframe) == 2
