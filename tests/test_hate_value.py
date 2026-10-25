"""Hatat och billigt (valhall/hate_study.py V1/V2, hate_value_study.py) och hatkolumnen i Sektorrotation."""
import numpy as np
import pandas as pd

from odin import sektorrotation as sr
from valhall import hate_study as hs


def test_spec_matches_the_code():
    text = open("docs/forskning/hatat_och_billigt_spec.md", encoding="utf-8").read()
    assert hs.MID_RANKS == (10, 30) and (hs.V_MIN_N, hs.V_MIN_YEARS) == (30, 5)
    for w in ("rank 11–30", "minst 30 perioder i varje grupp", "minst 5 startår", "504 handelsdagar", "20261025"):
        assert w in text, w


def test_ranks_skip_companies_without_prices():
    members = {2020: list(range(40))}
    got = hs.ranks(members, set(range(40)) - {3, 12}, 10, 30)
    assert got[2020][0] == 11 and 12 not in got[2020] and len(got[2020]) == 20


def test_start_rows_relative_two_years():
    idx = pd.bdate_range("2009-01-01", periods=800)
    c = pd.Series(np.linspace(100, 300, 800), index=idx)
    m = pd.Series(np.linspace(100, 150, 800), index=idx)
    e = {"start": idx[5], "undervalued": True}
    r = hs.start_rows(c, m, [e])[0]
    a, b = 6, 6 + hs.HOLD
    assert abs(r["rel"] - ((c.iloc[b] / c.iloc[a] - 1) - (m.iloc[b] / m.iloc[a] - 1))) < 1e-12
    assert hs.start_rows(c, m, [{"start": idx[400]}]) == []


def _rows(gap, n=60, years=8):
    rng = np.random.default_rng(3)
    out = []
    for i in range(n):
        y = 2008 + i % years
        out.append({"year": y, "rel": float(rng.normal(gap, 0.05)), "undervalued": True})
        out.append({"year": y, "rel": float(rng.normal(-0.05, 0.05)), "undervalued": False})
    return out


def test_value_summary_verdicts():
    rng = np.random.default_rng(0)
    good = hs.value_summary(_rows(0.10), rng, 500)
    assert good["V1"]["pass"] and good["V2"]["pass"]
    only_v1 = hs.value_summary(_rows(-0.01), rng, 500)
    assert only_v1["V1"]["pass"] and not only_v1["V2"]["pass"]
    few = hs.value_summary(_rows(0.10, n=20), rng, 500)
    assert not few["V1"]["pass"] and not few["V2"]["pass"]


BLOB = {"nordic": {"variants": {"10": {"d1": {"Fastighet": {"median_m": 2.4}, "Energi": {"median_m": 4.6}}}},
                   "current": [{"name": "LUMO", "since": "2022-07-04", "months": 51.7, "sector": "Fastighet", "typical_m": 2.4},
                               {"name": "CAST", "since": "2024-01-16", "months": 33.1, "sector": "Fastighet", "typical_m": 2.4}]},
        "commodities": {"current": [{"name": "REMX", "since": "2026-10-06", "months": 0.7, "group": "producent",
                                     "typical_m": 4.2}]}}


def test_hate_by_sector_and_rows():
    h = sr.hate_by_sector(BLOB)
    assert h["Fastighet"]["count"] == 2 and h["Fastighet"]["longest"] == ("LUMO", 51.7)
    assert h["Energi"]["count"] == 0 and h["Energi"]["typical"] == 4.6
    rows = sr.hate_rows(BLOB)
    assert [r["Namn"] for r in rows] == ["LUMO", "CAST", "REMX"] and rows[0]["Långt över det typiska"] == "ja"
    assert sr.hate_by_sector(None) == {} and sr.hate_rows(["x"]) == []
    ui = sr.ui_rows([{"ticker": "XLRE", "label": "Fastighet"}], {}, h)
    assert ui[0]["Hatade storbolag"] == "2/10" and ui[0]["Längst hatad"].startswith("LUMO 52 mån")


def test_value_script_runs(monkeypatch):
    import markets

    import hate_value_study as hvs
    from tests.test_hate_study import _Api
    monkeypatch.setattr(markets, "load", lambda api=None: {})
    monkeypatch.setattr(markets, "nordic_stock_ids", lambda table=None: {1})
    saved = {}
    res = hvs.run(api=_Api(), today="2026-10-25", n_boot=100, save=lambda n, d: saved.setdefault(n, d) is not None)
    assert "error" not in res
    assert {"rank 11–30 (prövning)", "topp 10 (redan sett)"} <= set(res)
    assert res["rank 11–30 (prövning)"]["members"] <= 10 and "V1" in res["topp 10 (redan sett)"]
    assert hvs.BLOB in saved and any("V1" in x for x in hvs.report_lines(res))
