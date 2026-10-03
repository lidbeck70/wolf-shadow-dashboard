"""
🐺 Fiat Debasement PR 5 — Wolf Debasement Index (percentil mot egen historik,
vikter som går att ändra) och scenarierna. Syntetiska serier — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiat_debasement import config as cfg  # noqa: E402
from fiat_debasement import index as fx  # noqa: E402
from tests.test_fiat_ui import ROOT, _app, fake_data  # noqa: E402,F401


def _monthly(start, end, pct, base=100.0):
    idx = pd.date_range(start, end, freq="MS")
    yrs = (idx - idx[0]).days / 365.25
    return pd.Series(base * (1 + pct / 100) ** np.asarray(yrs), index=idx)


def _comps(n_years=26):
    idx = pd.date_range("2000-01-01", periods=12 * n_years, freq="MS")
    rising = pd.Series(np.linspace(0, 10, len(idx)), index=idx)
    return {"gap": rising, "pp_loss": rising * 2, "debt": pd.Series(60.0, index=idx), "gold": pd.Series(rising.values[::-1], index=idx)}


def test_rolling_loss():
    cpi = _monthly("1995-01-01", "2026-08-01", 2.5)
    loss = fx.rolling_loss(cpi, 5)
    assert loss.index[0] == pd.Timestamp("2000-01-01")
    assert loss.iloc[-1] == pytest.approx((1 - 1 / 1.025 ** 5) * 100, abs=0.05)
    assert fx.rolling_loss(None) is None


def test_expanding_percentile_has_no_look_ahead():
    s = pd.Series(np.random.default_rng(1).normal(size=120), index=pd.date_range("2000-01-01", periods=120, freq="MS"))
    full = fx.expanding_percentile(s, min_obs=36)
    assert full.index[0] == s.index[35]
    for i in (40, 80, 119):
        cut = fx.expanding_percentile(s.iloc[:i + 1], min_obs=36)
        assert cut.iloc[-1] == pytest.approx(full.loc[s.index[i]])
    inc = pd.Series(range(50), index=pd.date_range("2000-01-01", periods=50, freq="MS"), dtype=float)
    assert fx.expanding_percentile(inc).iloc[-1] == pytest.approx(99.0)        # (49 + 0,5) / 50
    assert fx.expanding_percentile(inc.iloc[:10]) is None                     # för kort historik


def test_index_is_a_weighted_average_of_percentiles():
    res = fx.compute("SEK", _comps())
    assert res.value is not None and 0 <= res.value <= 100 and res.weight_share == 1.0
    rows = {r["key"]: r for r in res.rows}
    assert rows["gap"]["percentile"] == pytest.approx(rows["pp_loss"]["percentile"]) and rows["gap"]["percentile"] > 95
    assert rows["debt"]["percentile"] == pytest.approx(50.0) and rows["gold"]["percentile"] < 5
    assert [rows[k]["weight"] for k in ("gap", "pp_loss", "debt", "gold")] == [40, 30, 15, 15]
    assert sum(r["contribution"] for r in res.rows) == pytest.approx(res.value, abs=0.2)
    assert res.history.index[0] >= pd.Timestamp("2002-12-01")                 # 36 månader uppvärmning
    heavy_gold = fx.compute("SEK", _comps(), {"gap": 0, "pp_loss": 0, "debt": 0, "gold": 100})
    assert heavy_gold.value == pytest.approx(rows["gold"]["percentile"], abs=0.1)


def test_missing_components_are_reweighted_or_unavailable():
    comps = _comps()
    comps["gold"] = None
    res = fx.compute("USD", comps)
    assert res.value is not None and res.weight_share == pytest.approx(0.85)
    assert "Saknas" in res.note and "guld" in res.note
    gone = fx.compute("USD", {"gap": None, "pp_loss": None, "debt": _comps()["debt"], "gold": None})
    assert gone.value is None and "DATA UNAVAILABLE" in gone.note             # bara 15 % av vikten
    assert fx.normalize_weights({"gap": 1, "pp_loss": 1, "debt": 1, "gold": 1}) == \
        {"gap": 0.25, "pp_loss": 0.25, "debt": 0.25, "gold": 0.25}
    assert fx.normalize_weights({k: 0 for k in cfg.INDEX_COMPONENTS})["gap"] == 0.0


def test_index_uses_the_common_normalization_period():
    comps = _comps()
    early = pd.Series(1000.0, index=pd.date_range("1980-01-01", "1999-12-01", freq="MS"))
    comps["debt"] = pd.concat([early, comps["debt"]])                         # extremvärden före 2000 ignoreras
    res = fx.compute("SEK", comps)
    debt = next(r for r in res.rows if r["key"] == "debt")
    assert debt["percentile"] == pytest.approx(50.0) and debt["since"] == cfg.INDEX_SINCE


def test_scenarios_are_arithmetic_not_forecasts():
    r = fx.scenario(5.0, 2.0, 2.5, 10)
    assert r["monetary_gap"] == pytest.approx(3.0)
    assert r["purchasing_power_end"] == pytest.approx(100 / 1.025 ** 10, abs=0.1)        # ≈ 78,1
    assert r["money_per_output_end"] == pytest.approx(100 * (1.05 / 1.02) ** 10, abs=0.1)
    assert list(r["path"].index) == list(range(11)) and r["path"].iloc[0] == 100
    assert set(cfg.SCENARIOS) == {"BASE CASE", "BULLISH REAL ASSETS", "DEFENSIVE FIAT"}
    assert cfg.SCENARIOS["BULLISH REAL ASSETS"] == {"m2": 7.0, "gdp": 1.5, "cpi": 4.0}


def test_index_for_real_series(fake_data):
    from fiat_debasement import ui as fu
    res = {c: fu.index_for(c) for c in cfg.CURRENCIES}
    assert all(r.value is not None for r in res.values())
    rows = fu.overview_rows({}, res)
    assert all(r["cells"][0]["key"] == "index" and r["cells"][0]["text"] != fu.NA for r in rows)


def test_page_shows_index_and_scenarios(fake_data, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FD_TEST_ROOT", ROOT)
    at = AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    for part in ("WOLF DEBASEMENT INDEX", "Wolfpanel Composite Indicator", "It is not an official measure",
                 "<th>Debasement</th>", "SCENARIOS", "inte prognoser", "Komponenter för SEK"):
        assert part in html, part
    before = at.session_state["fd_w_gap"]
    at.number_input(key="fd_w_gap").set_value(0).run()
    assert not at.exception and at.session_state["fd_w_gap"] == 0 != before
    at.number_input(key="fd_sc_base_case_cpi").set_value(5.0).run()
    at.slider(key="fd_sc_years").set_value(20).run()
    assert not at.exception
