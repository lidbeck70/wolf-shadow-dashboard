"""
Skannrarna mot registret — Insider Norden: Nordens bolag ur universum-registret (alla listor, bara Börsdata-bolag)
som instrumentrader; samma kluster och poäng som förut; utan register är det Börsdatas lista. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import insider_scan as sc  # noqa: E402
import markets as mk  # noqa: E402
from test_insider_scan import TODAY, _FakeAPI  # noqa: E402
from universe import registry as ur  # noqa: E402
from universe import scan as uscan  # noqa: E402


def _loader(api):
    u = ur.build(table=mk.FALLBACK, nordic=api.get_instruments(), global_=[], include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return lambda name, fb: saved.get(name, fb)


def test_as_instruments_and_registry_nordic():
    api = _FakeAPI()
    rows = sc.registry_nordic(_loader(api))
    assert [r["ticker"] for r in rows] == ["EKTA B", "NOD"] and rows[0]["insId"] == 1 and rows[0]["marketId"] == 1
    assert rows[0]["stockPriceCurrency"] == "SEK" and rows[1]["market"] == "norden"      # index och okänd lista borta
    assert sc.registry_nordic(lambda name, fb: fb) is None
    assert uscan.as_instruments([ur.Member("X", "Lista", "usa", sources=("BERSERK",))]) == []   # utan insId


def test_scan_reads_the_register_and_scores_as_before():
    api = _FakeAPI()
    out = sc.scan(api, today=TODAY, loader=_loader(api))
    assert out["error"] is None and out["universe_source"] == "universum-registret" and out["universe"] == 2
    assert [c["ticker"] for c in out["clusters"]] == ["EKTA-B.ST"] and out["clusters"][0]["insiders"] == 3
    assert ("insider", [1, 2]) in api.calls
    plain = sc.scan(_FakeAPI(), today=TODAY, loader=lambda name, fb: fb)
    assert plain["universe_source"] == "Börsdata" and plain["universe"] == 2
    assert [c["ticker"] for c in plain["clusters"]] == ["EKTA-B.ST"]
    broken = sc.scan(_FakeAPI(), today=TODAY, loader=lambda name, fb: (_ for _ in ()).throw(RuntimeError("nere")))
    assert broken["universe_source"] == "Börsdata" and broken["error"] is None
