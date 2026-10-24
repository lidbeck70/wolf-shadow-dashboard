"""
K0 — papperets mätinstrument (analys_komplexstrategier.md §7.3): Valhalls Viking Nine-skanning får rapportdatum
(annars WAIT → READY → aldrig GOLDEN TICKET i kontot), Viking-exiten i papperet får hela playbooken på dagens stapel
(nine/order blocks/rapport), avnoterade positioner stängs i stället för att ligga öppna för evigt, cykelstart-
notisen nycklas på segmentets nyckel (Thors chip), och trunkering av stängda affärer räknas. Inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import valhall_scan  # noqa: E402
from test_valhall_paper import D, IDX, _board, _df, _sig  # noqa: E402
from valhall import paper as vpa  # noqa: E402


def test_valhall_scan_gives_the_real_viking_scan_a_report_date_getter(monkeypatch):
    import ovtlyr.ui.viking_screens as vs
    monkeypatch.setattr(vs, "_earnings", lambda t: None)
    assert "earnings_getter" in valhall_scan.scan_kwargs(None)
    assert valhall_scan.scan_kwargs(lambda *a, **k: {}) == {}           # ett fejkat scan_fn i tester får inget extra


def test_paper_viking_exit_gets_nine_blocks_and_report_date_on_todays_bar(monkeypatch):
    import viking_exit as vex
    data = {"VOLV-B.ST": _df(o=np.arange(100, 140, 1.0))}
    seen = []

    def fake(ticker, df, entry, entry_date, nine=None, ob_analysis=None, spy_df=None, earnings_date=None, today=None, **kw):
        seen.append((str(pd.Timestamp(today).date()), nine, ob_analysis, earnings_date))
        return SimpleNamespace(status=vex.HOLD, reasons=[])

    monkeypatch.setattr(vex, "evaluate_exit", fake)
    ctx = {"nine": lambda t, df: f"NINE {t} {len(df)}", "ob": lambda df: {"signal_bias": "HOLD"},
           "earnings": lambda t: "2026-03-01"}
    st, _ = vpa.step(None, [_sig("viking", "VOLV-B.ST")], _board(), lambda t: data[t].loc[:D[2]], today=D[2])
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[3]], today=D[3], exit_ctx=ctx)
    assert seen[-1][0] == D[3] and seen[-1][1].startswith("NINE VOLV-B.ST") and seen[-1][2] == {"signal_bias": "HOLD"}
    assert seen[-1][3] == "2026-03-01"
    seen.clear()
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[6]], today=D[6], exit_ctx=ctx)   # tre ikappdagar
    days = [s[0] for s in seen]
    assert days == [D[4], D[5], D[6]] and seen[-1][1] and all(s[1] is None for s in seen[:-1])   # bara dagens stapel
    broken = {"nine": lambda t, df: 1 / 0, "ob": None, "earnings": lambda t: 1 / 0}
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[7]], today=D[7], exit_ctx=broken)
    assert seen[-1][1] is None and seen[-1][3] is None                                       # fel → None, aldrig krasch
    assert vpa._exit_inputs(None, "X", data["VOLV-B.ST"]) == (None, None, None)


def test_position_without_prices_is_closed_after_ten_runs():
    data = {"A.ST": _df(o=np.arange(100, 140, 1.0))}
    st, _ = vpa.step(None, [_sig("quality", "A.ST")], _board(), lambda t: data[t].loc[:D[2]], today=D[2])
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[3]], today=D[3])
    assert st["positions"] and st["positions"][0]["last_close"] == 103.0
    gone = lambda t: None  # noqa: E731
    for i in range(vpa.NODATA_RUNS_MAX - 1):
        st, ev = vpa.step(st, [], _board(), gone, today=D[4 + i])
        assert st["positions"] and st["positions"][0]["nodata_runs"] == i + 1 and not ev
    st, ev = vpa.step(st, [], _board(), gone, today=D[4 + vpa.NODATA_RUNS_MAX - 1])
    assert not st["positions"] and st["closed"][0]["exit"] == 103.0 and "avnoterad" in st["closed"][0]["reason"]
    assert ev and ev[0]["ticker"] == "A.ST"
    # kursdata tillbaka innan gränsen: räknaren nollas
    st2, _ = vpa.step(None, [_sig("quality", "B.ST")], _board(), lambda t: _df(o=np.arange(100, 140, 1.0)).loc[:D[2]], today=D[2])
    st2, _ = vpa.step(st2, [], _board(), lambda t: _df(o=np.arange(100, 140, 1.0)).loc[:D[3]], today=D[3])
    st2, _ = vpa.step(st2, [], _board(), gone, today=D[4])
    st2, _ = vpa.step(st2, [], _board(), lambda t: _df(o=np.arange(100, 140, 1.0)).loc[:D[5]], today=D[5])
    assert st2["positions"][0]["nodata_runs"] == 0


def test_closed_truncation_is_counted():
    st = vpa.new_state(D[0])
    st["closed"] = [{"strategy": "quality", "ticker": f"T{i}", "exit_date": D[1], "entry": 1.0, "exit": 1.0,
                     "segment": "industrials"} for i in range(vpa.CLOSED_MAX + 5)]
    st, _ = vpa.step(st, [], _board(), lambda t: None, today=D[2])
    assert len(st["closed"]) == vpa.CLOSED_MAX and st["closed_dropped"] == 5
    assert vpa.results(st)["closed_dropped"] == 5 and vpa.results(vpa.new_state(D[0]))["closed_dropped"] == 0


def test_cycle_starts_carry_the_segment_key_and_thor_matches_it():
    from thor import cykel
    from valhall import board as vb
    from valhall import cycle
    seg = {"key": "materials", "label": "Material", "global": {"cycle_start": False},
           "nordic": {"cycle_start": True, "status_since": "2026-10-01"},
           "themes": [{"theme": "guld", "label": "Guld", "state": {"cycle_start": True, "status_since": "2026-09-01"}}]}
    starts = vb.cycle_starts({"segments": [seg]})
    assert starts[0] == {"segment": "Material", "key": "materials", "where": "Norden", "since": "2026-10-01"}
    assert starts[1]["key"] == "materials" and starts[1]["theme"] == "guld" and starts[1]["segment"] == "Material · Guld"
    blob = {"cycle_starts": starts}
    assert cykel.cycle_start(blob, "materials") == "Norden"                       # nyckeln — förut bara etiketten
    assert cykel.cycle_start({"cycle_starts": [{"segment": "materials", "where": "Norden"}]}, "materials") == "Norden"
    assert cykel.cycle_start(blob, "technology") is None and cycle.VANDER
