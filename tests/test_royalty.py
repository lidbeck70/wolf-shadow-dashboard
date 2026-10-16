"""
Royalty — universumet (listan + Royalty-håvens royalty/streaming-träffar), playbookens grindar, rangen ur
Contrarian Alphas pelare, verdikten, körskriptet med fejkade delar och arbetsflödessteget. Inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import royalty_scan as rs  # noqa: E402
from royalty import engine as eng  # noqa: E402
from royalty import universe as ru  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENS = {"generated": "2026-10-16T06:55:00+00:00", "screens": {"royalty": {"label": "Royalty", "rows": [
    {"ticker": "FNV.TO", "name": "Franco-Nevada", "ins_id": 9001, "universe": "canada", "mcap_musd": 30000.0,
     "m": {"nd_ebitda": -0.5, "ev_ebitda": 28.0, "fcf": 900.0, "gross_margin": 93.0, "ebit_margin": 61.0},
     "notes": [ru.NAME_NOTE]},
    {"ticker": "BHP", "name": "BHP Group", "ins_id": 9002, "universe": "usa", "mcap_musd": 211000.0,
     "m": {"nd_ebitda": 0.4, "ev_ebitda": 7.0, "fcf": 12000.0, "gross_margin": 72.0, "ebit_margin": 45.0}, "notes": []},
    {"ticker": "NEWR.V", "name": "New Royalty Corp", "ins_id": 9003, "universe": "canada", "mcap_musd": 150.0,
     "m": {"nd_ebitda": 0.1, "ev_ebitda": 12.0, "fcf": 10.0, "gross_margin": 88.0, "ebit_margin": 50.0},
     "notes": [ru.NAME_NOTE]}]}}}


def _frame(n=320, start=100.0, step=-0.12, vol_start=2e6, vol_end=8e5, end="2026-10-15"):
    """Syntetisk OHLCV: linjär trend med lite brus; volymen glider från vol_start till vol_end."""
    idx = pd.bdate_range(end=end, periods=n)
    close = start + step * np.arange(n) + 1.5 * np.sin(np.arange(n) / 7.0)
    vol = np.linspace(vol_start, vol_end, n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": vol},
                        index=idx)


DOWN = _frame()                                    # ner 38 %, under SMA200, nära 52v-lägsta, volymtorka
UP = _frame(step=0.25, vol_start=8e5, vol_end=2e6)  # upp 80 %, långt över SMA200
GOOD_KPI = {"gross_margin": 90.0, "ebit_margin": 55.0, "nd_ebitda": 0.3, "fcf": 400.0, "ev_ebitda": 12.0,
            "p_fcf": 14.0, "pb": 2.0, "ps": 10.0, "mcap_musd": 5000.0}
GOOD_FUND = {"fcf": 400.0, "fcf_history": [400.0, 380.0, 350.0], "ebitda_margin": 70.0, "debt_to_equity": 0.1,
             "equity": 3000.0, "market_cap": 5000.0, "ev_ebitda": 12.0, "revenue": 600.0, "total_assets": 3500.0,
             "ebit": 330.0}
GOOD_VALUE = {"ev_ebitda": 12.0, "ev_ebitda_history": [20.0, 22.0, 18.0, 19.0, 21.0, 23.0, 17.0, 20.0, 24.0, 19.0],
              "p_fcf": 14.0, "p_fcf_history": [24.0, 26.0, 22.0, 25.0, 23.0, 27.0, 21.0, 24.0, 28.0, 23.0]}


def _company(ticker="FNV.TO", **kw):
    base = {"ticker": ticker, "name": ticker, "kind": "royalty", "theme": "guld", "source": ru.SOURCE_LIST,
            "ins_id": 9001, "market": "kanada", "country": "CA", "notes": [], "m": {}}
    return dict(base, **kw)


# ── Universumet ──────────────────────────────────────────────────────────────
def test_curated_list_loads_and_is_sane():
    from berserk import themes as th
    rows = ru.load_csv()
    tickers = [r["ticker"] for r in rows]
    assert len(rows) >= 20 and len(set(tickers)) == len(tickers) and "FNV.TO" in tickers and "PSK.TO" in tickers
    assert all(r["kind"] in ru.KINDS and r["source"] == ru.SOURCE_LIST for r in rows)
    assert all(r["theme"] is None or r["theme"] in th.THEMES for r in rows)          # BERSERK-vokabulär
    assert rows[0]["name"] == "Franco-Nevada" and rows[0]["country"] == "CA"


def test_screen_hits_need_the_royalty_name_note():
    rows = ru.from_screens(SCREENS)
    assert [r["ticker"] for r in rows] == ["FNV.TO", "NEWR.V"]                        # BHP är en producent
    r = rows[1]
    assert r["kind"] == ru.KIND_SCREEN and r["source"] == ru.SOURCE_SCREEN and r["ins_id"] == 9003 and r["country"] == "CA"
    assert r["m"]["gross_margin"] == 88.0 and r["m"]["mcap_musd"] == 150.0
    assert ru.from_screens(None) == [] and ru.from_screens({"screens": {"royalty": {"rows": ["skräp"]}}}) == []


def test_build_merges_list_and_screen_and_fills_from_register():
    reg = {"FNV.TO": SimpleNamespace(ins_id=9001, market="kanada", theme="guld", name="Franco-Nevada"),
           "NEWR.V": SimpleNamespace(ins_id=9003, market="kanada", theme="silver", name="New Royalty Corp"),
           "TPL": SimpleNamespace(ins_id=9010, market="usa", theme=None, name="Texas Pacific Land")}
    csv_rows = [{"ticker": "FNV.TO", "name": "Franco-Nevada", "kind": "royalty", "theme": "guld", "country": "CA",
                 "notes": [], "source": ru.SOURCE_LIST, "ins_id": None, "market": None, "m": {}},
                {"ticker": "TPL", "name": "Texas Pacific Land", "kind": "royalty", "theme": "olja", "country": "US",
                 "notes": [], "source": ru.SOURCE_LIST, "ins_id": None, "market": None, "m": {}}]
    rows = ru.build(csv_rows, SCREENS, SimpleNamespace(get=lambda t: reg.get(t)))
    by = {r["ticker"]: r for r in rows}
    assert list(by) == ["FNV.TO", "TPL", "NEWR.V"]                                     # listan först, håven efter
    fnv = by["FNV.TO"]
    assert fnv["source"] == f"{ru.SOURCE_LIST} + {ru.SOURCE_SCREEN}" and fnv["ins_id"] == 9001
    assert fnv["m"]["gross_margin"] == 93.0 and fnv["market"] == "kanada" and fnv["theme"] == "guld"
    assert by["TPL"]["ins_id"] == 9010 and by["TPL"]["market"] == "usa" and by["TPL"]["theme"] == "olja"
    assert by["NEWR.V"]["theme"] == "silver" and by["NEWR.V"]["kind"] == ru.KIND_SCREEN
    assert ru.describe(rows) == "3 bolag · listan 2 · håven 1 · med Börsdata-id 3"


# ── Grindarna och vakten ─────────────────────────────────────────────────────
def test_gates_are_the_playbooks():
    assert eng.gates(GOOD_KPI) == (True, [])
    ok, fails = eng.gates(dict(GOOD_KPI, gross_margin=65.0, nd_ebitda=2.0, fcf=-5.0))
    assert not ok and fails == ["bruttomarginal 65 % (kräver > 70 %)", "skuld/EBITDA 2.0 (kräver < 1.5)", "FCF negativt"]
    ok, fails = eng.gates({})
    assert not ok and fails == ["bruttomarginal saknas", "EBIT-marginal saknas", "skuld/EBITDA saknas", "FCF saknas"]
    assert eng.gates(dict(GOOD_KPI, ebit_margin=40.0))[1] == ["EBIT-marginal 40 % (kräver > 40 %)"]


def test_kpi_units_from_screen_and_snapshot():
    scr = eng.kpis_from_screen({"gross_margin": 93.0, "ebit_margin": 61.0, "nd_ebitda": -0.5, "fcf": 900.0})
    assert scr["gross_margin"] == 93.0 and scr["p_fcf"] is None
    snap = eng.kpis_from_snapshot({"gross_margin": 0.93, "operating_margin": 0.61, "net_debt_ebitda": -0.5,
                                   "fcf_m": 900.0, "ev_ebitda": 28.0, "p_fcf": 33.0, "market_cap": 41000.0})
    assert snap["gross_margin"] == 93.0 and snap["ebit_margin"] == 61.0 and snap["p_fcf"] == 33.0
    merged = eng.merge_kpis({"gross_margin": None, "ev_ebitda": 28.0}, {"gross_margin": 93.0, "ev_ebitda": 27.0})
    assert merged["gross_margin"] == 93.0 and merged["ev_ebitda"] == 28.0
    assert eng.fund_from_kpis({"fcf": 10.0, "ev_ebitda": 12.0, "pb": 1.0}) == {"fcf": 10.0, "ev_ebitda": 12.0}


def test_price_guard_and_verdict_thresholds():
    assert eng.price_guard({"close": 105.0, "sma200": 100.0}) == (True, 5.0)
    assert eng.price_guard({"close": 112.0, "sma200": 100.0}) == (False, 12.0)
    assert eng.price_guard({}) == (False, None)
    assert eng.verdict(True, 65.0) == eng.BUY and eng.verdict(True, 64.9) == eng.WATCH
    assert eng.verdict(True, 49.9) == eng.PASS and eng.verdict(False, 99.0) == eng.PASS
    assert eng.composite({"hate": 100, "value": 100, "quality": 100, "catalyst": 100}) == 100.0
    assert abs(sum(eng.WEIGHTS.values()) - 1.0) < 1e-9


# ── Pelarna och verdikten ────────────────────────────────────────────────────
def test_hated_cheap_royalty_is_a_candidate_and_loved_one_is_not():
    hated = eng.evaluate(_company(), DOWN, GOOD_KPI, GOOD_FUND, GOOD_VALUE)
    assert hated["gates"]["ok"] and hated["guard"]["ok"] and hated["guard"]["above_sma200_pct"] < 0
    assert hated["pillars"]["hate"] >= 55 and hated["pillars"]["value"] >= 50 and hated["pillars"]["quality"] > 0
    assert hated["composite"] >= eng.WATCH_MIN and hated["verdict"] in (eng.BUY, eng.WATCH) and hated["why"] == []
    assert hated["price"]["from_low_52w_pct"] is not None and hated["kpi"]["gross_margin"] == 90.0
    assert eng.MANUAL_NOTE in hated["notes"] and hated["regime"] in ("green", "orange", "red")
    loved = eng.evaluate(_company("UP.TO"), UP, GOOD_KPI, GOOD_FUND, GOOD_VALUE)
    assert not loved["guard"]["ok"] and loved["verdict"] == eng.PASS and "mot SMA200" in loved["why"][0]
    assert loved["pillars"]["hate"] < hated["pillars"]["hate"]


def test_failed_gates_and_missing_prices_are_pass_with_a_reason():
    weak = eng.evaluate(_company("LEV.TO"), DOWN, dict(GOOD_KPI, nd_ebitda=3.1), GOOD_FUND, GOOD_VALUE)
    assert weak["verdict"] == eng.PASS and weak["why"] == ["skuld/EBITDA 3.1 (kräver < 1.5)"] and weak["composite"] > 0
    blind = eng.evaluate(_company("NOPX.TO"), pd.DataFrame(), GOOD_KPI, GOOD_FUND, GOOD_VALUE)
    assert blind["verdict"] == eng.PASS and "INGEN_KURS" in blind["flags"] and blind["pillars"]["hate"] == 0.0
    assert blind["why"] == ["SMA200 saknas"] and blind["price"]["close"] is None and blind["regime"] is None
    thin = eng.evaluate(_company("THIN.V"), DOWN, GOOD_KPI, None, None)             # håvens rad: inga historiker
    assert thin["pillars"]["value"] == 0.0 and thin["pillars"]["quality"] == 0.0 and thin["gates"]["ok"]


def test_run_sorts_counts_and_keeps_going_on_errors():
    comps = [_company("A.TO"), _company("B.TO"), _company("C.TO"), _company("D.TO")]
    frames = {"A.TO": DOWN, "B.TO": UP, "C.TO": DOWN}

    def prices(t):
        if t == "D.TO":
            raise RuntimeError("Yahoo nere")
        return frames[t]

    def data(c):
        if c["ticker"] == "C.TO":
            return dict(GOOD_KPI, gross_margin=50.0), GOOD_FUND, GOOD_VALUE
        return GOOD_KPI, GOOD_FUND, GOOD_VALUE
    out = eng.run(comps, prices, data, today="2026-10-16")
    assert [r["ticker"] for r in out["rows"]][:1] == ["A.TO"] and len(out["rows"]) == 4
    assert out["rows"] == sorted(out["rows"], key=lambda r: (-r["composite"], r["ticker"]))
    assert out["errors"] == {"D.TO": "kurs: Yahoo nere"} and out["generated"] == "2026-10-16"
    assert out["stats"]["n"] == 4 and out["stats"]["buy"] + out["stats"]["watch"] + out["stats"]["pass"] == 4
    assert out["weights"] == eng.WEIGHTS and out["gates"] == eng.GATES and out["thresholds"] == {"buy": 65.0, "watch": 50.0}


# ── Körskriptet ──────────────────────────────────────────────────────────────
class _FakeAPI:
    def __init__(self):
        self.calls = []

    def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
        self.calls.append((scope, list(ids)))
        return {i: {"gross_margin": 0.90, "operating_margin": 0.55, "net_debt_ebitda": 0.3, "fcf_m": 400.0,
                    "ev_ebitda": 12.0, "p_fcf": 14.0, "market_cap": 5000.0, "ebitda_margin": 0.70,
                    "debt_to_equity": 0.1, "total_equity_m": 3000.0, "revenue_m": 600.0} for i in ids}

    def get_kpi_history(self, ins_id, kpi_id, report_type="year", price_type="mean"):
        """Tio år, äldst först som Börsdata — dagens multipel ligger under alla (billig mot sig själv)."""
        base = 20.0 if kpi_id == 11 else 24.0
        return [{"y": 2016 + i, "p": None, "v": base + (i % 4)} for i in range(10)]


def test_data_getter_uses_snapshot_history_and_screen_fallback():
    api = _FakeAPI()
    rows = [_company("FNV.TO", ins_id=9001, market="kanada"), _company("X.ST", ins_id=5, market="norden"),
            _company("NEWR.V", ins_id=None, m={"gross_margin": 88.0, "ebit_margin": 50.0, "nd_ebitda": 0.1, "fcf": 10.0})]
    hist = lambda iid, key: [20.0, 22.0] if key == "ev_ebitda" else []   # noqa: E731
    get = rs.data_getter(api, rows, history=hist)
    assert sorted(api.calls) == [("global", [9001]), ("nordic", [5])]
    kpi, fund, value = get(rows[0])
    assert kpi["gross_margin"] == 90.0 and kpi["ebit_margin"] == 55.0 and fund["fcf"] == 400.0 and fund["equity"] == 3000.0
    assert value == {"ev_ebitda": 12.0, "p_fcf": 14.0, "ev_ebitda_history": [20.0, 22.0]}
    kpi, fund, value = get(rows[2])                                               # bara håvens rad
    assert kpi["gross_margin"] == 88.0 and fund == {"fcf": 10.0} and value == {}
    kpi, fund, value = rs.data_getter(None, rows)(rows[2])                        # utan Börsdata
    assert kpi["nd_ebitda"] == 0.1 and value == {}


def test_scan_run_with_fakes_saves_the_blob():
    saved = {}
    frames = {"FNV.TO": DOWN, "NEWR.V": UP}
    reg = SimpleNamespace(get=lambda t: SimpleNamespace(ins_id=9001, market="kanada", theme="guld", name="Franco-Nevada")
                          if t == "FNV.TO" else None)
    csv_only = [{"ticker": "FNV.TO", "name": "Franco-Nevada", "kind": "royalty", "theme": "guld", "country": "CA",
                 "notes": [], "source": ru.SOURCE_LIST, "ins_id": None, "market": None, "m": {}}]
    import royalty.universe as uni
    real_build = uni.build
    uni.build = lambda c, s, r: real_build(csv_only, s, r)
    try:
        out = rs.run(api=_FakeAPI(), loader=lambda name, fb: SCREENS if name == rs.SCREENS_BLOB else fb,
                     getter=lambda t: frames.get(t, pd.DataFrame()), save=lambda n, d: saved.__setitem__(n, d) or True,
                     today="2026-10-16", register=reg)
    finally:
        uni.build = real_build
    p = out["payload"]
    assert out["saved"] and rs.BLOB_NAME in saved and p["universe_source"] == "listan + Royalty-håven + registret"
    assert [r["ticker"] for r in p["rows"]] == ["FNV.TO", "NEWR.V"] and p["rows"][0]["verdict"] in (eng.BUY, eng.WATCH)
    assert p["rows"][1]["verdict"] == eng.PASS and p["errors"] == {} and p["generated"] == "2026-10-16"
    dry = rs.run(api=None, loader=lambda name, fb: fb, getter=lambda t: pd.DataFrame(), save=lambda n, d: True,
                 today="2026-10-16", register=None, dry_run=True)
    assert not dry["saved"] and dry["payload"]["universe_source"] == "listan" and dry["payload"]["stats"]["n"] >= 20


def test_workflow_runs_royalty_after_the_screens():
    src = open(os.path.join(ROOT, ".github", "workflows", "scheduled-scan.yml"), encoding="utf-8").read()
    assert "python royalty_scan.py" in src and src.index("python screens_scan.py") < src.index("python royalty_scan.py")
    assert src.index("python royalty_scan.py") < src.index("python sheets_refresh.py")
