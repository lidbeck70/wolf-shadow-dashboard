"""
Skannrarna mot registret — Valhall: kartans instrumentrader kommer ur universum-registrets Norden (alla listor,
bara Börsdata-bolag) i Börsdatas form; utan register är det api.get_instruments() precis som förut. Inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import markets as mk  # noqa: E402
import valhall_scan as vsn  # noqa: E402
from test_valhall_swing import _cv  # noqa: E402
from universe import registry as ur  # noqa: E402

NORDIC = [{"insId": i, "ticker": t, "name": t, "marketId": 1, "sectorId": 5, "stockPriceCurrency": "SEK"}
          for i, t in enumerate(("A", "B", "C"), start=1)]
NORDIC.append({"insId": 100, "ticker": "OMXSPI", "name": "OMX Stockholm PI", "marketId": 7})


def _loader():
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=[], include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return saved, (lambda name, fb: saved.get(name, fb))


def test_instruments_from_the_register_or_borsdata():
    _saved, load = _loader()
    api = SimpleNamespace(get_instruments=lambda: [{"ticker": "Z", "marketId": 1}])
    rows, source = vsn.instruments_for(api, load)
    assert source == "universum-registret" and [r["ticker"] for r in rows] == ["A", "B", "C"]   # indexet är inte med
    assert rows[0] == {"insId": 1, "ticker": "A", "name": "A", "marketId": 1, "sectorId": 5, "branchId": None,
                       "stockPriceCurrency": "SEK", "market": "norden"}
    rows, source = vsn.instruments_for(api, lambda name, fb: fb)
    assert source == "Börsdata" and rows == [{"ticker": "Z", "marketId": 1}]
    rows, source = vsn.instruments_for(api, lambda name, fb: (_ for _ in ()).throw(RuntimeError("nere")))
    assert source == "Börsdata" and rows == [{"ticker": "Z", "marketId": 1}]


def test_run_reads_the_register(monkeypatch):
    saved, load = _loader()
    monkeypatch.setattr(mk, "load", lambda api=None, **k: {})
    monkeypatch.setattr(mk, "nordic_stock_ids", lambda table=None: {1})
    monkeypatch.setattr(mk, "to_yf", lambda t, m=None, table=None: t + ".ST")
    data = {t + ".ST": _cv(100, 1e5, n=300) for t in ("A", "B", "C")}
    calls = []
    api = SimpleNamespace(get_instruments=lambda: calls.append("api") or [])
    out = vsn.run(api=api, data_getter=lambda ts, p: {t: data[t] for t in ts if t in data}, getter=lambda t, p: None,
                  load=load, save=lambda f, d: saved.__setitem__(f, d) or True,
                  scan_fn=lambda *a, **k: {"rows": []}, nordic_market=None, sources={},
                  long_fundamentals=lambda ts: {})
    p = out["payload"]
    assert out["saved"] and p["universe_source"] == "universum-registret" and calls == []
    assert p["nordic"]["industrials"]["members"] == 3 and p["long"]["scanned"] == 3
