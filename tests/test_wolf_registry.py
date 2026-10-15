"""
Skannrarna mot registret — Momentum Swing (wolf_data): universumet begränsas till de Börsdata-bolag som finns i
universum-registrets Norden (alla listor); utan register är det Börsdatas lista precis som förut. Ingen nätverk —
Börsdata är fejkad via wolf_data._get och registret läses med en egen läsare.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import markets as mk  # noqa: E402
import wolf_data as w  # noqa: E402
from test_wolf_data import _fake_get  # noqa: E402
from universe import registry as ur  # noqa: E402

# Registret känner bara till AAA (Large Cap) och BBB (First North) — CCC och SNK finns hos Börsdata men inte här.
NORDIC = [{"insId": 1, "ticker": "AAA", "name": "Alfa AB", "marketId": 1, "stockPriceCurrency": "SEK"},
          {"insId": 2, "ticker": "BBB", "name": "Beta AB", "marketId": 4, "stockPriceCurrency": "SEK"},
          {"insId": 100, "ticker": "OMXSPI", "name": "OMX Stockholm PI", "marketId": 7}]


def _loader():
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=[], include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return lambda name, fb: saved.get(name, fb)


def _no_csv(tmp_path):
    w.CONFIG["UNIVERSE_CSV"] = str(tmp_path / "absent.csv")
    w._get = _fake_get


def test_registry_ins_ids(monkeypatch):
    monkeypatch.setattr(w, "REGISTRY_LOADER", _loader())
    assert w.registry_ins_ids() == {1, 2}                                   # First North med, indexet inte
    monkeypatch.setattr(w, "REGISTRY_LOADER", lambda name, fb: fb)
    assert w.registry_ins_ids() is None
    monkeypatch.setattr(w, "REGISTRY_LOADER", lambda name, fb: (_ for _ in ()).throw(RuntimeError("nere")))
    assert w.registry_ins_ids() is None


def test_build_universe_restricts_to_the_register(monkeypatch, tmp_path):
    _no_csv(tmp_path)
    monkeypatch.setattr(w, "REGISTRY_LOADER", _loader())
    uni = w.build_universe()
    assert list(uni["ticker"]) == ["AAA", "BBB"] and list(uni.columns) == ["insId", "name", "ticker"]


def test_build_universe_without_register_is_borsdatas_list(monkeypatch, tmp_path):
    _no_csv(tmp_path)
    monkeypatch.setattr(w, "REGISTRY_LOADER", lambda name, fb: fb)
    uni = w.build_universe()
    assert list(uni["ticker"]) == ["AAA", "BBB", "CCC", "SNK"]              # fond och index borta som förut
    monkeypatch.setattr(w, "REGISTRY_LOADER", lambda name, fb: (_ for _ in ()).throw(RuntimeError("nere")))
    assert list(w.build_universe()["ticker"]) == ["AAA", "BBB", "CCC", "SNK"]
