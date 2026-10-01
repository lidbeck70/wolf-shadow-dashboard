"""
REGIME → Råvaror → 🥇 Guldkvoter: råvaror (guld ÷ råvara) och index
(index ÷ guld) genom Guld/Silver-motorn, som lämnas orörd. Referens =
kvotens egen median, målkvoter = egna percentiler. Börsdata som reserv.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from gold_ratios import config as rc  # noqa: E402
from gold_ratios import data as grd  # noqa: E402
from gold_ratios import engine as gre  # noqa: E402
from gold_silver import engine as ge  # noqa: E402


def _flat(years: int, value: float, end="2026-09-30"):
    idx = pd.bdate_range(end=end, periods=years * 260)
    return pd.Series([value] * len(idx), index=idx, dtype=float)


def _ramp(years: int, start: float, stop: float, end="2026-09-30"):
    idx = pd.bdate_range(end=end, periods=years * 260)
    n = len(idx)
    return pd.Series([start + (stop - start) * i / (n - 1) for i in range(n)], index=idx, dtype=float)


PLAT, DOW = rc.PAIR_BY_KEY["platina"], rc.PAIR_BY_KEY["dow"]


# ── motorn ───────────────────────────────────────────────────────────────────
def test_orientation_commodity_vs_index():
    assert gre.orient(PLAT, 4000.0, 1600.0) == (4000.0, 1600.0)         # guld ÷ platina
    assert gre.orient(DOW, 4000.0, 44000.0) == (44000.0, 4000.0)        # Dow ÷ guld
    assert ge.ratio(*gre.orient(PLAT, 4000.0, 1600.0)) == 2.5
    assert ge.ratio(*gre.orient(DOW, 4000.0, 44000.0)) == 11.0


def test_the_gold_silver_engine_solves_the_other_side():
    # platina: kvot 2 vid guld 4000 → platina 2000
    rows = ge.revaluation_table(4000.0, 1600.0, [2.0], 2.0)
    assert rows[0].is_current and rows[0].ratio == 2.5 and rows[1].silver == 2000.0 and rows[1].change_pct == 25.0
    # Dow/guld: kvot 5 vid Dow 44000 → guld 8800
    rows = ge.revaluation_table(44000.0, 4000.0, [5.0], 5.0)
    assert rows[1].silver == 8800.0 and rows[1].is_reference and rows[1].formula == "44,000 / 5 = 8,800.00"


def test_targets_are_own_percentiles_and_reference_is_own_median():
    s = gre.ratio_series(PLAT, _flat(20, 2000.0), _ramp(20, 2000.0, 500.0))        # kvot 1 → 4
    full = gre.full_stats(s)
    labels = [lab for lab, _v in gre.targets(full)]
    assert labels == ["P10", "P25", "Median", "P75", "P90"]
    vals = dict(gre.targets(full))
    assert vals["P10"] < vals["Median"] < vals["P90"] and vals["Median"] == full.median
    assert gre.targets(None) == []


def test_anchor_grid_rounds_multiples_of_today():
    assert gre.anchor_grid(4189.0) == (3100.0, 4200.0, 5200.0, 6300.0, 8400.0)
    assert gre.anchor_grid(46213.0)[1] == 46000.0
    assert gre.anchor_grid(None) == () and gre.anchor_grid(0) == ()


def test_position_row_and_meaning():
    s = gre.ratio_series(DOW, _flat(12, 2000.0), _flat(12, 20000.0))              # Dow/guld = 10 hela vägen
    r = gre.position_row(DOW, s)
    assert r["current"] == 10.0 and r["median10"] == 10.0 and r["diff_pct"] == 0
    assert r["position"] == "Nära historisk median" and r["median_max"] == 10.0
    short = gre.position_row(PLAT, gre.ratio_series(PLAT, _flat(3, 2000.0), _flat(3, 1000.0)))
    assert short["current"] == 2.0 and short["median10"] is None                   # ingen 10-årsperiod
    assert gre.position_row(PLAT, None)["current"] is None
    assert "köper mycket platina" in gre.meaning(PLAT) and "kostar många uns guld" in gre.meaning(DOW)
    assert gre.unit_word(rc.PAIR_BY_KEY["koppar"]) == "lb"


def test_every_pair_is_complete():
    keys = [p["key"] for p in rc.PAIRS]
    assert {"platina", "palladium", "koppar", "olja", "brent", "naturgas", "vete", "kaffe", "kakao",
            "dow", "spx"} == set(keys)
    assert rc.PAIR_BY_KEY["dow"]["ticker"] == "^DJI" and rc.PAIR_BY_KEY["spx"]["ticker"] == "^GSPC"
    assert rc.PAIR_BY_KEY["vete"]["scale"] == 0.01 and rc.PAIR_BY_KEY["kaffe"]["scale"] == 0.01
    for p in rc.PAIRS:
        assert p["kind"] in (rc.COMMODITY, rc.INDEX) and p["unit"]


# ── data ─────────────────────────────────────────────────────────────────────
def test_fetch_scales_cents_and_falls_back_to_borsdata():
    yahoo = {"GC=F": _flat(2, 4000.0), "ZW=F": _flat(2, 550.0)}                     # vete i US-cent
    bd_calls = []

    def bd(ins_id):
        bd_calls.append(ins_id)
        return _flat(2, 1600.0) if ins_id == 21033 else None

    def getter(t, p):
        return yahoo.get(t, pd.Series(dtype=float))
    all_ = grd.fetch_all(getter, bd)
    vete = all_["vete"]
    assert vete["other"]["value"] == 5.5 and float(vete["series"].iloc[-1]) == pytest.approx(4000 / 5.5)
    plat = all_["platina"]
    assert plat["other"]["value"] == 1600.0 and "Börsdata" in plat["other"]["source"]
    assert float(plat["series"].iloc[-1]) == 2.5
    assert 21033 in bd_calls and 21031 not in bd_calls                               # guldet kom från Yahoo
    gas = all_["naturgas"]                                                           # ingen reserv → saknas
    assert gas["other"] is None and "saknas just nu" in gas["error"] and len(gas["series"]) == 0


def test_fetch_gold_from_borsdata_when_yahoo_is_empty():
    gold = grd.fetch_gold(lambda t, p: pd.Series(dtype=float), lambda i: _flat(1, 4100.0) if i == 21031 else None)
    assert gold["point"]["value"] == 4100.0 and "insId 21031" in gold["point"]["source"]


# ── sidan ────────────────────────────────────────────────────────────────────
def test_page_renders_overview_and_pairs(monkeypatch):
    from streamlit.testing.v1 import AppTest
    gold = {"series": _flat(22, 4000.0), "point": {"value": 4000.0, "source": "Yahoo Finance GC=F",
                                                    "date": "2026-09-30", "kind": "ACTUAL"}}
    yahoo = {"PL=F": _ramp(22, 4000.0, 1600.0), "^DJI": _ramp(22, 20000.0, 44000.0)}

    def fake_all(getter=None, bd_getter=None):
        def g(t, p):
            return yahoo.get(t, pd.Series(dtype=float))
        return {p["key"]: grd.fetch_pair(p["key"], gold, g, lambda i: None) for p in rc.PAIRS}
    monkeypatch.setattr(grd, "fetch_all", fake_all)
    monkeypatch.setenv("GR_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["GR_TEST_ROOT"])
        from gold_ratios.ui import render_gold_ratios_page
        render_gold_ratios_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "ALLA KVOTER" in html and "Guld / Platina" in html and "Dow Jones / Guld" in html
    assert "historik saknas" in html                                     # par utan data
    assert "GULD / PLATINA" in html and ">2.50<" in html and "1 uns guld = 2.50 oz platina" in html
    assert "PLATINA PER KVOT" in html and "REFERENS" in html and "inte ett fair value" in html
    assert "20 år" in html and "GULD × KVOT" in html
    assert any("PLATINA" in c.proto.spec and "VECKOVIS" in c.proto.spec for c in at.get("plotly_chart"))
    # Dow / guld: guldpriset räknas ut ur indexet
    at.selectbox(key="gr_pair").set_value("dow").run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "DOW JONES / GULD" in html and "Dow Jones = 11.00 uns guld" in html
    assert "GULD PER KVOT" in html and "Guld = Dow Jones / kvot" in html and "DOW JONES × KVOT" in html
    # egen inmatning
    at.number_input(key="gr_gold_dow").set_value(8800.0).run()
    html = " ".join(m.value for m in at.markdown)
    assert ">5.00<" in html and "egen inmatning · ASSUMPTION" in html


def test_navigation_and_guide():
    from ui import nav
    assert "🥇 Guldkvoter" in nav.options("regime/Råvaror")
    assert "🥇🥈 Guld/Silver" in nav.options("regime/Råvaror")                # Guld/Silver kvar
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert 'elif sub == "🥇 Guldkvoter":' in src and "render_gold_ratios_page" in src
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    assert any(t == "REGIME → Råvaror → 🥇 Guldkvoter" for t, _r, _u in _PANEL_GUIDE)
