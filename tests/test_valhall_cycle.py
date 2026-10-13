"""
Valhall fas 1 — segmenten, cykelmotorn (BERSERK-fas + Blindspot dag för dag), nordiska sektorindex
och kartan. Syntetiska kurser — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import themes as bth  # noqa: E402
from valhall import board as vb  # noqa: E402
from valhall import cycle  # noqa: E402
from valhall import nordic_index as ni  # noqa: E402
from valhall import segments as sgm  # noqa: E402

N = 3000
IDX = pd.bdate_range(end="2026-10-09", periods=N)


def _ser(values, idx=IDX):
    return pd.Series(np.asarray(values, dtype=float), index=idx)


CRASH_TURN = np.concatenate([np.linspace(100, 140, 2000), np.linspace(140, 70, 700), np.linspace(70, 85, 300)])
UP = np.linspace(50, 200, N)
DOWN = np.linspace(200, 60, N)


# ── Segmenten ───────────────────────────────────────────────────────────────
def test_eleven_segments_with_industrials():
    keys = [s.key for s in sgm.SEGMENTS]
    assert len(keys) == 11 and "industrials" in keys and len(set(keys)) == 11
    assert {s.spdr for s in sgm.SEGMENTS} == {"XLE", "XLB", "XLI", "XLF", "XLRE", "XLK", "XLV", "XLC", "XLY",
                                              "XLP", "XLU"}
    assert all(s.ucits and s.niche for s in sgm.SEGMENTS)
    assert all(sgm.segment_of_theme(t) for t in bth.THEMES)                    # alla BERSERK-teman har ett hem


def test_borsdata_sector_mapping_splits_real_estate():
    assert sgm.segment_for(3) == "energy" and sgm.segment_for(5) == "industrials" and sgm.segment_for(10) == "utilities"
    assert sgm.segment_for(1, 68) == "financials"                            # bank
    assert sgm.segment_for(1, 75) == "real_estate" and sgm.segment_for(1, 76) == "real_estate"
    assert sgm.segment_for(1) == "financials"                                 # okänd bransch → finans
    assert sgm.segment_for(None) is None and sgm.segment_for("x") is None
    from contrarian_alpha.necessity import BORSDATA_SECTOR_MAP
    for s in sgm.SEGMENTS:
        if s.key != "real_estate":
            assert s.necessity == BORSDATA_SECTOR_MAP[s.bd_sectors[0]].score, s.key


# ── Cykelmotorn ─────────────────────────────────────────────────────────────
def test_constants_match_blindspot():
    from blindspot import theme_board as tb
    assert (cycle.CYKEL_TIDIG, cycle.CYKEL_SEN, cycle.CYKEL_TOPP) == (tb._CYKEL_TIDIG, tb._CYKEL_SEN, tb._CYKEL_TOPP)
    assert (cycle.HAT_W_5Y_HIGH, cycle.HAT_W_VOL, cycle.HAT_W_RS) == (tb._HAT_W_5Y_HIGH, tb._HAT_W_VOL, tb._HAT_W_RS)
    for key, nec in vb.THEME_NECESSITY.items():
        spec = next((t for t in tb._THEMES if t.key == key), None)
        if spec is not None:
            assert nec == spec.necessity, key


def test_cykel_label_rules():
    assert cycle.cykel_label(95, -1) == cycle.TOPP and cycle.cykel_label(75, 1) == cycle.SEN
    assert cycle.cykel_label(20, 5) == cycle.TIDIG
    assert cycle.cykel_label(50, 0.5) == cycle.MITTEN and cycle.cykel_label(50, -0.5) == cycle.TIDIG
    assert cycle.cykel_label(50, None) == cycle.TIDIG and cycle.cykel_label(np.nan, 1) == cycle.NODATA


def test_status_rules():
    assert cycle.status_of(cycle.VANDER, cycle.TIDIG) == cycle.GRON
    assert cycle.status_of(cycle.STARK, cycle.MITTEN) == cycle.GRON
    assert cycle.status_of(cycle.UPPTREND, cycle.SEN) == cycle.GUL
    assert cycle.status_of(cycle.STARK, cycle.TOPP) == cycle.ROD
    assert cycle.status_of(cycle.BAISSE, cycle.TIDIG) == cycle.ROD and cycle.status_of(cycle.SVAG, cycle.MITTEN) == cycle.ROD
    assert cycle.status_of(cycle.NODATA, cycle.TIDIG) == cycle.NODATA


def test_cycle_frame_states():
    up = cycle.state(cycle.cycle_frame(_ser(UP)))
    assert up["phase"] == cycle.STARK and up["cykel"] == cycle.TOPP and up["status"] == cycle.ROD   # dyrt i toppen
    down = cycle.state(cycle.cycle_frame(_ser(DOWN)))
    assert down["phase"] == cycle.BAISSE and down["status"] == cycle.ROD and down["cykel"] == cycle.TIDIG
    turn = cycle.cycle_frame(_ser(CRASH_TURN))
    st = cycle.state(turn)
    assert st["phase"] == cycle.VANDER and st["cycle_start"] and st["status"] == cycle.GRON
    assert st["status_since"] is not None and st["status_since"] <= st["date"]
    assert turn["phase"].iloc[:150].eq(cycle.NODATA).all()                     # före SMA200 — ingen fas


def test_cycle_frame_is_point_in_time():
    close = _ser(CRASH_TURN + np.sin(np.arange(N) / 7.0) * 2)
    vol = _ser(1e6 + 1e5 * np.cos(np.arange(N) / 30.0))
    spy = _ser(np.linspace(100, 300, N))
    full = cycle.cycle_frame(close, vol, spy, 80)
    for k in (1300, 2200, 2800):
        cut = cycle.cycle_frame(close.iloc[:k + 1], vol.iloc[:k + 1], spy.iloc[:k + 1], 80)
        for col in ("phase", "cykel", "status"):
            assert full[col].iloc[k] == cut[col].iloc[-1], (col, k)
        for col in ("pct10", "hat", "blindspot"):
            assert full[col].iloc[k] == pytest.approx(cut[col].iloc[-1], nan_ok=True), (col, k)


def test_hat_components():
    close = _ser(np.r_[np.linspace(100, 100, 1500), np.linspace(100, 40, 1500)])     # faller 1 500 dagar
    spy = _ser(np.linspace(100, 200, N))
    vol = _ser(np.r_[np.full(N - 252, 2e6), np.full(252, 1e6)])                      # volymen halverad senaste året
    h = cycle.hat_frame(close, vol, spy).iloc[-1]
    high5 = close.iloc[-1260:].max()                                                  # femårstoppen
    assert h["from_high5"] == pytest.approx((high5 - 40) / high5 * 100, abs=0.1)
    assert h["vol_chg"] == pytest.approx(-50, abs=1) and h["rs_spy"] < 0 and h["hat"] > 90   # hatat på alla tre
    neutral = cycle.hat_frame(close).iloc[-1]                                         # utan volym och SPY
    expected_high = min(neutral["from_high5"] / 60, 1) * 100
    assert neutral["hat"] == pytest.approx(0.4 * expected_high + 0.3 * 50 + 0.3 * 50, abs=0.01)


# ── Nordiska sektorindex ────────────────────────────────────────────────────
def test_members_by_segment():
    inst = [{"ticker": "BOL", "marketId": 1, "sectorId": 7, "branchId": 17},
            {"ticker": "CAST", "marketId": 1, "sectorId": 1, "branchId": 75},
            {"ticker": "SEB A", "marketId": 1, "sectorId": 1, "branchId": 68},
            {"ticker": "AAPL", "marketId": 99, "sectorId": 6, "branchId": 1},
            {"ticker": "X", "marketId": 1, "sectorId": None}]
    m = ni.members(inst, {1}, lambda t, mid: t.replace(" ", "-") + ".ST")
    assert m["materials"] == ["BOL.ST"] and m["real_estate"] == ["CAST.ST"] and m["financials"] == ["SEB-A.ST"]
    assert m["technology"] == [] and len(m) == 11


def test_equal_weight_index_and_jump_guard():
    idx = pd.bdate_range("2026-01-01", periods=5)
    a = pd.Series([100, 110, 121, 121, 121.0], index=idx)                              # +10 %, +10 %
    b = pd.Series([50, 50, 50, 1.5, 1.5], index=idx)                                   # ojusterad split dag 4
    c = pd.Series([10, 9, 9, 9, 9.0], index=idx)                                       # −10 %
    i = ni.equal_weight_index({"A": a, "B": b, "C": c}, min_members=2)
    assert i.iloc[0] == 100 and i.iloc[1] == pytest.approx(100 * (1 + (0.10 + 0 - 0.10) / 3))
    assert i.iloc[3] == pytest.approx(i.iloc[2])                                       # splitten räknas inte
    assert ni.equal_weight_index({"A": a, "B": b}) is None                              # för få medlemmar


# ── Kartan ──────────────────────────────────────────────────────────────────
def _ohlcv(c):
    c = np.asarray(c, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=IDX)


def test_board_and_cycle_starts():
    data = {"SPY": _ohlcv(UP), "XLE": _ohlcv(CRASH_TURN), "XLK": _ohlcv(UP), "XLU": _ohlcv(DOWN)}
    drivers = {"olja": ("BZ=F", _ser(CRASH_TURN)), "koppar": ("HG=F", _ser(UP))}
    nordic = {"energy": {"index": _ser(CRASH_TURN), "members": 14, "with_data": 13}}
    b = vb.build(getter=lambda t, p: data.get(t), nordic=nordic, nordic_market=_ser(UP), drivers=drivers)
    rows = {r["key"]: r for r in b["segments"]}
    assert len(rows) == 11
    assert rows["energy"]["global"]["status"] == cycle.GRON and rows["energy"]["nordic"]["cycle_start"]
    assert rows["energy"]["nordic_members"] == 14 and rows["technology"]["global"]["cykel"] == cycle.TOPP
    assert rows["utilities"]["global"]["status"] == cycle.ROD
    assert rows["healthcare"]["global"]["status"] == cycle.NODATA and rows["healthcare"]["nordic"] is None
    olja = next(t for t in rows["energy"]["themes"] if t["theme"] == "olja")
    assert olja["state"]["cycle_start"] and olja["necessity"] == 90
    starts = vb.cycle_starts(b)
    assert {"segment": "Energi", "where": "global"} == {k: starts[0][k] for k in ("segment", "where")}
    assert any(s["where"] == "Norden" for s in starts) and any("Olja" in s["segment"] for s in starts)


def test_probe_workflow():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, ".github", "workflows", "valhall-probe.yml"), encoding="utf-8").read()
    assert "workflow_dispatch" in src and "python -m valhall.probe" in src and "BORSDATA_API_KEY" in src
