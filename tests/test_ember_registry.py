"""
Skannrarna mot registret — EMBER: universumkällan "Universum-registret" (registrets råvarubolag på huvudlistorna
i Norden, USA och Kanada, med temat registrerat i ember.regime), "Båda" som reserv när registret saknas, och
den schemalagda körningen. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ember import regime as er  # noqa: E402
from ember import universe as eu  # noqa: E402
from universe import registry as ur  # noqa: E402
from universe import scan as uscan  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBERS = [ur.Member("GULD.ST", "Guldgruvan", "norden", theme="guld", market_id=3, sources=("Börsdata",)),
           ur.Member("LAC", "Lithium Americas", "usa", theme="litium", market_id=32, sources=("Börsdata global",)),
           ur.Member("NTR.TO", "Nutrien", "kanada", theme="godsel", market_id=35, sources=("Börsdata global",)),
           ur.Member("FRO.OL", "Frontline", "norden", theme="tank", market_id=9, sources=("BERSERK",))]


def test_source_registry_uses_the_register_and_registers_themes(monkeypatch):
    monkeypatch.setattr(er, "_THEME_OVERLAY", {})
    seen = {}

    def fake_members_for(strategy, loader=None, themes=None, max_members=None, **kw):
        seen.update(strategy=strategy, themes=themes, max_members=max_members)
        return [m for m in MEMBERS if m.theme in themes]
    monkeypatch.setattr(uscan, "members_for", fake_members_for)
    tickers, stats = eu.build_universe(eu.SOURCE_REGISTRY, use_prefilter=False)
    assert seen["strategy"] == "ember" and seen["max_members"] == eu.REGISTRY_MAX
    assert {"guld", "olja", "uran", "agri"} <= seen["themes"] and "litium" in seen["themes"]
    assert tickers[:len(eu.EMBER_ETF_UNIVERSE)] == eu.EMBER_ETF_UNIVERSE
    assert tickers[len(eu.EMBER_ETF_UNIVERSE):] == ["GULD.ST", "LAC", "NTR.TO", "FRO.OL"]   # fas 8: tank = FRAKT
    assert stats.registry_raw == 4 and not stats.registry_missing and stats.total_before_prefilter == len(tickers)
    assert er.detect_theme("GULD.ST", auto=False) == "guld" and er.detect_theme("LAC", auto=False) == "litium"
    assert er.detect_theme("NTR.TO", auto=False) == "godsel"                   # BERSERKs vokabulär, inget alias
    assert er.detect_complex("FRO.OL") == "frakt" and er.detect_complex("NTR.TO") == "agri"
    assert er.register_themes({"X.ST": "finns-inte", "": "guld"}) == 0 and "X.ST" not in er._THEME_OVERLAY
    assert eu.SOURCE_REGISTRY in eu.ALL_SOURCES and eu.ALL_SOURCES[:3] == [eu.SOURCE_CURATED, eu.SOURCE_AUTO,
                                                                           eu.SOURCE_BOTH]


def test_source_registry_falls_back_to_both_when_the_register_is_missing(monkeypatch):
    monkeypatch.setattr(uscan, "members_for", lambda *a, **k: None)
    monkeypatch.setattr(eu, "fetch_nordic_commodity_tickers", lambda: (["BOL.ST"], True, ""))
    monkeypatch.setattr(eu, "fetch_global_commodity_tickers", lambda: (["NEM"], {"USA": 1}, ""))
    tickers, stats = eu.build_universe(eu.SOURCE_REGISTRY, use_prefilter=False)
    assert stats.registry_missing and stats.registry_raw == 0 and stats.source == eu.SOURCE_REGISTRY
    assert "BOL.ST" in tickers and "NEM" in tickers and set(eu.EMBER_STOCK_UNIVERSE) <= set(tickers)   # = Båda
    assert stats.nordic_raw == 1 and stats.global_raw == 1 and stats.us_intl_raw == len(eu.US_INTL_CURATED)


def test_registry_members_helper_and_scheduled_scan():
    assert eu.registry_members(loader=lambda name, fb: fb) is None
    src = open(os.path.join(ROOT, "scheduled_scan.py"), encoding="utf-8").read()
    assert "universe_source=SOURCE_REGISTRY" in src and 'universe_source="both"' not in src
    assert eu.REGISTRY_THEME_ALIAS == {} and eu.REGISTRY_MAX >= 100        # fas 8: EMBER har BERSERKs teman
