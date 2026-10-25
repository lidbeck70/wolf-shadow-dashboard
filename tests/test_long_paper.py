"""Papperskontona för långsidans cykeltest (valhall/long_paper.py): testets inträden framåt, två års horisont."""
import numpy as np
import pandas as pd

from valhall import long_paper as vlp
from valhall import long_study as lst
from valhall import paper as vpa


def _idx(n, start="2024-01-01"):
    return pd.bdate_range(start, periods=n)


def test_accounts_exist_and_are_long():
    for key in (vlp.E1_KEY, vlp.E3_KEY):
        assert vpa.STRATEGIES[key][1] == vpa.LONG
    assert set(vlp.ENTRY_KEY) == {lst.E1, lst.E3}                    # E2 tas inte med (prövades inte bättre)
    import valhall_long_study as vls
    assert set(vlp.PRODUCER_SEGMENT) == set(vls.PRODUCERS)          # samma serier som testet


def test_recent_events_uses_the_test_declustering():
    idx = _idx(600)
    ev = {lst.E1: [idx[100], idx[597]], lst.E3: [idx[400], idx[598]]}
    got = vlp.recent_events(ev, idx)
    assert got == {lst.E1: idx[597]}                                 # E3 dag 598 är inom 252 dagar efter dag 400
    assert vlp.recent_events({lst.E1: [idx[500]]}, idx) == {}        # för gammal


def test_nordic_top_ranks_in_sek_and_keeps_one_share_class():
    ins = [{"insId": 1, "ticker": "VOLV A", "name": "Volvo A", "marketId": 1, "sectorId": 5, "stockPriceCurrency": "SEK"},
           {"insId": 2, "ticker": "VOLV B", "name": "Volvo B", "marketId": 1, "sectorId": 5, "stockPriceCurrency": "SEK"},
           {"insId": 3, "ticker": "KNEBV", "name": "Kone", "marketId": 1, "sectorId": 5, "stockPriceCurrency": "EUR"},
           {"insId": 4, "ticker": "X", "name": "Utanför", "marketId": 9, "sectorId": 5, "stockPriceCurrency": "SEK"}]
    top = vlp.nordic_top({1: 500.0, 2: 500.0, 3: 100.0, 4: 9999.0}, ins, {1}, lambda t, m: t.replace(" ", "-") + ".ST")
    assert [t for t, _n in top["industrials"]] == ["KNEBV.ST", "VOLV-A.ST"]   # 100 EUR > 500 SEK; Volvo en gång


def test_signals_add_sector_large_caps_only_for_sector_funds():
    found = [("XLE", "energy", lst.SEKTOR, {lst.E1: pd.Timestamp("2026-10-20")}),
             ("GDX", "materials", lst.PRODUCENT, {lst.E3: pd.Timestamp("2026-10-21")})]
    top = {"energy": [("EQNR.OL", "Equinor"), ("AKRBP.OL", "Aker BP")], "materials": [("BOL.ST", "Boliden")]}
    sigs = vlp.signals(found, top)
    assert [(s["strategy"], s["ticker"]) for s in sigs] == [
        ("lang_e1", "XLE"), ("lang_e1", "EQNR.OL"), ("lang_e1", "AKRBP.OL"), ("lang_e3", "GDX")]
    assert sigs[0]["nordic"] is False and sigs[1]["nordic"] is True and sigs[0]["segment"] == "energy"


def _df(n, start="2024-01-01", px=100.0):
    idx = _idx(n, start)
    c = pd.Series(px, index=idx)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6})


def test_paper_sells_lang_accounts_after_two_years():
    df = _df(800, "2024-01-01")
    state = vpa.new_state("2024-01-01")
    sig = vpa._sig("lang_e1", "XLE", "XLE", "energy", None, False, "E1")
    state, _ = vpa.step(state, [sig], {}, lambda t: df.iloc[:2], today="2024-01-01")
    state, _ = vpa.step(state, [], {}, lambda t: df.iloc[:3], today="2024-01-03")
    assert len(state["positions"]) == 1
    end = df.index[-1]
    state, ev = vpa.step(state, [], {}, lambda t: df, today=end)
    assert not state["positions"] and state["closed"][0]["reason"].startswith("tidsstopp 2 år")
    held = (pd.Timestamp(state["closed"][0]["exit_date"]) - pd.Timestamp(state["closed"][0]["entry_date"])).days
    assert vpa.LANG_TEST_DAYS <= held < vpa.LONG_DAYS


def test_run_finds_a_fresh_event(monkeypatch):
    """Ett E1 som inträffar i dag ger signal för serien; utan händelse blir det inga signaler."""
    def fake_events(fr, kind):
        return {lst.E1: [fr.index[-1]], lst.E2: [], lst.E3: []}

    def getter(sym, period):
        rng = np.random.default_rng(sum(map(ord, sym)))
        c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))), index=_idx(800, "2023-01-02"))
        return pd.DataFrame({"Close": c, "Volume": 1e6})

    monkeypatch.setattr(lst, "entry_events", fake_events)
    sigs = vlp.run(getter)
    assert {s["ticker"] for s in sigs} >= {"XOP", "XLE", "GDX"} and all(s["strategy"] == "lang_e1" for s in sigs)
    monkeypatch.setattr(lst, "entry_events", lambda fr, kind: {lst.E1: [], lst.E2: [], lst.E3: []})
    assert vlp.run(getter) == []
