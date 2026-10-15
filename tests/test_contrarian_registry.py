"""
Skannrarna mot registret — Contrarian Alpha (den schemalagda körningen): Nordens aktielistor ur universum-
registret i stället för den frysta ALL_NORDIC_MARKETS; utan register eller med ett som inte går att läsa är det
ALL_NORDIC_MARKETS som förut. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import markets as mk  # noqa: E402
import scheduled_scan as ss  # noqa: E402
from borsdata_api import ALL_NORDIC_MARKETS  # noqa: E402
from universe import registry as ur  # noqa: E402
from universe import scan as uscan  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORDIC = [{"insId": 1, "ticker": "ERIC B", "name": "Ericsson", "marketId": 1, "stockPriceCurrency": "SEK"},
          {"insId": 2, "ticker": "FN", "name": "First North AB", "marketId": 4, "stockPriceCurrency": "SEK"},
          {"insId": 3, "ticker": "EQNR", "name": "Equinor", "marketId": 9, "stockPriceCurrency": "NOK"},
          {"insId": 4, "ticker": "OMXS30", "name": "Index", "marketId": 7}]
GLOBAL = [{"insId": 10, "ticker": "NEM", "name": "Newmont", "marketId": 32, "stockPriceCurrency": "USD"}]


def _loader(nordic=NORDIC, global_=GLOBAL):
    u = ur.build(table=mk.FALLBACK, nordic=nordic, global_=global_, include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return lambda name, fb: saved.get(name, fb)


def test_market_ids_come_from_the_register_including_small_lists():
    assert ss._nordic_market_ids(loader=_loader()) == [1, 4, 9]                   # index 7 och USA 32 inte med
    assert ss._nordic_market_ids(loader=lambda name, fb: fb) == list(ALL_NORDIC_MARKETS)
    assert ss._nordic_market_ids(loader=_loader(nordic=[])) == list(ALL_NORDIC_MARKETS)   # inga nordiska bolag


def test_a_broken_register_falls_back_to_the_frozen_table(monkeypatch):
    monkeypatch.setattr(uscan, "members_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nere")))
    assert ss._nordic_market_ids() == list(ALL_NORDIC_MARKETS)
    src = open(os.path.join(ROOT, "scheduled_scan.py"), encoding="utf-8").read()
    assert "market_ids=_nordic_market_ids()" in src and "market_ids=list(ALL_NORDIC_MARKETS)" not in src
