"""
K3 — konfigurationslagret utanpå Viking Nine (complex_profiles, complex_nine): profilerna bär bara injicerbara
fält, drivargrinden och drivarexiten räknas på BERSERK:s driver_frame, producentbredden och marknadsingångarna
byggs utanför motorn, evaluate_complex ger rader i evaluate_ticker-form och scan_complex delar marknad/drivare
per skanning. Motorn (ovtlyr_nine, viking_execution, viking_exit) rörs inte. Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import complex_nine as cn  # noqa: E402
import complex_profiles as cp  # noqa: E402
import ovtlyr_nine as on  # noqa: E402
import viking_execution as vx  # noqa: E402
from test_thor_regim import _prices  # noqa: E402
from valhall import complexes as vcx  # noqa: E402
from valhall import segments as sgm  # noqa: E402

IDX = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1500)
UP = pd.Series(np.linspace(100, 200, 1500), index=IDX)
DOWN = pd.Series(np.linspace(200, 100, 1500), index=IDX)


def test_profiles_cover_every_complex_and_segment_with_injectable_fields_only():
    assert [p.key for p in cp.complexes()] == list(vcx.KEYS) and len(cp.segments()) == len(sgm.SEGMENTS)
    for p in cp.complexes():
        assert p.is_complex and p.driver_gate == cp.GATE_TURN and p.breadth == cp.PRODUCERS and p.universe == cp.KOMPLEX
        assert p.risk_pct == 1.25 and p.max_position_pct == 20.0 and p.driver_theme and p.sector_symbol
    for p in cp.segments():
        assert not p.is_complex and p.driver_theme is None and p.driver_gate == cp.GATE_OFF
        assert p.risk_pct == 1.5 and p.max_position_pct == 25.0 and p.breadth == cp.LARGECAP
    assert cp.get("real_estate").sector_symbol == "XLRE" and cp.get("technology").sector_symbol == "XLK"
    assert cp.get("basmetaller").sector_symbol == "COPX" and cp.get("x") is None and cp.get(None) is None
    assert set(cp.as_dict(cp.get("energi"))) == set(cp.INJECTABLE)
    # inget av motorns låsta tal finns i profilen
    for forbidden in ("atr", "rr", "volume", "fg", "earnings"):
        assert not any(forbidden in f for f in cp.INJECTABLE)


def test_driver_gate_and_exit_follow_berserks_driver_frame(monkeypatch):
    from berserk import live
    p = cp.get("basmetaller")
    monkeypatch.setattr(live, "load_drivers", lambda themes, getter=None, today=None: {t: ("HG=F", UP) for t in themes})
    d = cn.driver_state(p)
    assert d["symbol"] == "HG=F" and d["turn"] and d["passed"] and d["phase"] in ("STARK", "UPPTREND", "BAISSE · VÄNDER")
    assert cn.driver_exit(p) == {"triggered": False, "detail": "HG=F: över EMA50"}
    monkeypatch.setattr(live, "load_drivers", lambda themes, getter=None, today=None: {t: ("HG=F", DOWN) for t in themes})
    d = cn.driver_state(p)
    assert not d["turn"] and d["passed"] is False and cn.DRIVER_BLOCK in d["detail"] and d["below_ema50"]
    assert cn.driver_exit(p)["triggered"]
    monkeypatch.setattr(live, "load_drivers", lambda themes, getter=None, today=None: {t: (None, None) for t in themes})
    d = cn.driver_state(p)
    assert d["passed"] is None and "ingen prisserie" in d["detail"] and cn.driver_exit(p) is None
    assert cn.driver_state(cp.get("technology")) is None and cn.driver_exit(cp.get("technology")) is None
    off = cp.ComplexProfile("x", "X", cp.KOMPLEX, "COPX", driver_theme="koppar", driver_gate=cp.GATE_OFF)
    monkeypatch.setattr(live, "load_drivers", lambda themes, getter=None, today=None: {t: ("HG=F", DOWN) for t in themes})
    assert cn.driver_state(off)["passed"] is True                                      # grinden av: aldrig spärr


def test_producer_breadth_and_market_inputs(monkeypatch):
    from berserk import universe as buv
    p = cp.get("basmetaller")
    members = {"koppar": ["BOL.ST", "LUMI.ST", "FCX"], "stal": ["SSAB-A.ST"]}
    monkeypatch.setattr(buv, "by_theme", lambda tickers=None: members)
    monkeypatch.setattr(buv, "kind_of", lambda t: "producent")
    monkeypatch.setattr(buv, "region_of", lambda t: "USA" if t == "FCX" else "Norden")
    frames = {"BOL.ST": _prices("BOL.ST"), "LUMI.ST": _prices("LUMI.ST"), "SSAB-A.ST": _prices("SSAB-A.ST")}
    b = cn.producer_breadth(p, getter=lambda t, per: frames.get(t))
    assert b is not None and 0 <= b.iloc[-1] <= 100 and len(b) > 200
    assert cn.producer_breadth(p, getter=lambda t, per: frames.get(t) if t == "BOL.ST" else None) is None   # < 3 bolag
    nm = {"close": _prices("OMXS30")["Close"], "source": "test", "breadth": pd.Series(55.0, index=IDX[-300:]),
          "breadth_source": "Large Cap"}
    monkeypatch.setattr(on, "nordic_market", lambda bars=on.NORDIC_BARS, getter=None: nm)
    kw = cn.market_inputs(p, getter=lambda t, per: frames.get(t))
    assert kw["market_label"].startswith("OMXS30") and "producentbredd Basmetaller" in kw["breadth_source"]
    kw = cn.market_inputs(cp.get("technology"), getter=lambda t, per: frames.get(t))
    assert kw["breadth_source"] == "Large Cap" and float(kw["breadth_pct"].iloc[-1]) == 55.0
    monkeypatch.setattr(on, "nordic_market", lambda bars=on.NORDIC_BARS, getter=None: None)
    kw = cn.market_inputs(p, getter=lambda t, per: None)
    assert "saknas" in kw["market_label"] and kw["breadth_source"] == "bredd saknas" and kw["breadth_pct"].empty


def _engine(monkeypatch, driver_passed=True):
    from berserk import live
    nm = {"close": _prices("OMXS30")["Close"], "source": "test", "breadth": pd.Series(60.0, index=IDX[-300:]),
          "breadth_source": "Large Cap"}
    monkeypatch.setattr(on, "nordic_market", lambda bars=on.NORDIC_BARS, getter=None: nm)
    monkeypatch.setattr(live, "load_drivers", lambda themes, getter=None, today=None:
                        {t: ("HG=F", UP if driver_passed else DOWN) for t in themes})
    monkeypatch.setattr(cn, "producer_breadth", lambda p, getter=None, now=None: None)
    return lambda t, per="1y": _prices(t)


def test_evaluate_complex_runs_the_unchanged_engine_with_the_profiles_inputs(monkeypatch):
    getter = _engine(monkeypatch)
    p = cp.get("basmetaller")
    row = cn.evaluate_complex("BOL.ST", p, getter=getter, earnings_getter=lambda t: None)
    assert row["error"] is None and row["profile"] == "basmetaller" and row["driver"]["symbol"] == "HG=F"
    n, d = row["nine"], row["decision"]
    assert isinstance(n, on.NineResult) and n.sector_etf == "COPX" and "Basmetaller" in n.market_label
    assert "komplexprofil" in n.sector_source and isinstance(d, vx.EntryDecision)
    assert row["category"] in ("GOLDEN TICKET", "READY", "DEVELOPING", "REJECTED")
    assert cn.evaluate_complex("NODATA", p, getter=lambda t, per: None)["error"].startswith("DATA UNAVAILABLE")
    # drivargrinden: en GOLDEN TICKET blir READY när drivaren inte vänt — annars orörd
    monkeypatch.setattr(vx, "watchlist_category", lambda passed, status=None: "GOLDEN TICKET")
    assert cn.evaluate_complex("BOL.ST", p, getter=getter)["category"] == "GOLDEN TICKET"
    getter = _engine(monkeypatch, driver_passed=False)
    row = cn.evaluate_complex("BOL.ST", p, getter=getter)
    assert row["category"] == "READY" and any(cn.DRIVER_BLOCK in r for r in row["decision"].reasons)
    # en position per komplex (held nycklas på profilen), aldrig på SPDR
    row = cn.evaluate_complex("BOL.ST", cp.get("basmetaller"), getter=_engine(monkeypatch), held={"basmetaller": "LUMI.ST"})
    assert row["sector_busy"] == "LUMI.ST" and row["category"] == "READY"
    row = cn.evaluate_complex("BOL.ST", cp.get("basmetaller"), getter=_engine(monkeypatch), held={"XLB": "LUMI.ST"})
    assert row["sector_busy"] is None


def test_scan_complex_shares_market_and_driver_across_the_candidates(monkeypatch):
    getter = _engine(monkeypatch)
    calls = {"market": 0, "driver": 0}
    real_mi, real_ds = cn.market_inputs, cn.driver_state

    def mi(p, g=None, now=None):
        calls["market"] += 1
        return real_mi(p, g, now)

    def ds(p, g=None, today=None):
        calls["driver"] += 1
        return real_ds(p, g, today)
    monkeypatch.setattr(cn, "market_inputs", mi)
    monkeypatch.setattr(cn, "driver_state", ds)
    tickers = ["BOL.ST", "LUMI.ST", "SSAB-A.ST", "OUT1V.HE"]
    closes = {t: _prices(t)["Close"] for t in tickers}
    res = cn.scan_complex(cp.get("basmetaller"), tickers, closes_getter=lambda ts: closes, max_candidates=2, getter=getter)
    assert res["profile"] == "basmetaller" and res["funnel"]["profile"] == "basmetaller"
    assert res["funnel"]["universe"] == 4 and len(res["rows"]) <= 2 and res["driver"]["symbol"] == "HG=F"
    assert calls["market"] == 1 and calls["driver"] == 1                      # en gång per skanning, inte per aktie
    assert all(r["profile"] == "basmetaller" for r in res["rows"])


# ── K4: Valhall kör Viking Nine per komplex bredvid BERSERK ─────────────────
def test_valhall_complex_candidates_scan_each_complex_with_its_profile():
    from valhall import cycle
    from valhall import swing as vsw
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=80)
    frame = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 50_000.0}, index=idx)
    data = {t: frame for t in ("BOL.ST", "LUMI.ST", "EQNR.OL", "SCA-B.ST", "ERIC-B.ST")}
    universe = {t: {"segment": "materials", "name": t.split(".")[0], "currency": "SEK"} for t in data}
    cmap = {"BOL.ST": "basmetaller", "LUMI.ST": "basmetaller", "EQNR.OL": "energi", "SCA-B.ST": "skog"}
    board = {"complexes": [{"key": "basmetaller", "state": {"status": cycle.GRON, "turn": {"kind": "VÄNDER", "days": 3}}}]}
    seen = []

    def scan(profile, tickers, closes_getter=None, max_candidates=8, getter=None, today=None, now=None, **kw):
        seen.append((profile.key, tuple(tickers), kw.get("held"), kw.get("earnings_getter") is not None))
        return {"rows": [], "funnel": {}, "profile": profile.key, "driver": {"symbol": "X", "detail": f"{profile.key}: STARK"}}
    out = vsw.complex_candidates(board, universe, data, cmap, scan_fn=scan, earnings_getter=lambda t: None)
    cx = out["complexes"]
    assert list(cx) == list(vcx.KEYS) and out["liquid"] == 5
    assert [s[0] for s in seen] == ["energi", "basmetaller", "skog"]                     # bara komplex med bolag
    assert dict((s[0], s[1]) for s in seen)["basmetaller"] == ("BOL.ST", "LUMI.ST") and all(s[2] == {} and s[3] for s in seen)
    assert cx["basmetaller"]["status"] == cycle.GRON and cx["basmetaller"]["turn"]["days"] == 3
    assert cx["basmetaller"]["strategy"] == "Viking Nine · Basmetaller" and cx["basmetaller"]["universe"] == 2
    assert cx["adelmetaller"]["universe"] == 0 and cx["adelmetaller"]["driver"] is None and cx["skog"]["driver"]["detail"] == "skog: STARK"


def test_complex_card_shows_the_profile_swing_row():
    from valhall import ui as vui
    c = {"key": "basmetaller", "label": "Basmetaller", "driver": "korg: koppar:HG=F", "state": {"status": "GRÖN"},
         "themes": [], "themes_turning": 1, "n_themes": 7}
    sw = {"strategy": "Viking Nine · Basmetaller", "picks": [{"ticker": "BOL.ST", "category": "READY"}], "universe": 6,
          "driver": {"detail": "HG=F: STARK"}}
    html = vui.complex_card(c, sw)
    assert "Swing (Viking Nine · Basmetaller)" in html and "BOL.ST" in html and "6 bolag" in html and "HG=F: STARK" in html
    assert "Swing (" not in vui.complex_card(c, None)
