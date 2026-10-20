"""
Thor — "Skanna nu": kör en strategis skanning direkt från listan, med samma headless-funktion som den schemalagda
körningen, och ersätter strategins rader i sessionen. Inga regler ändras — samma motorer, samma förval.

Snabba skanningar körs live (Wolf, Viking, Viking Nine i Norden, EMBER, BERSERK, Insider, håvarna, Royalty).
De tunga (Quality/Deep Contrarian ~12 min, Momentum Swing ~12 min, Long Screener i Valhall-kartan) körs bara i
schemat; Thor säger då när de senast kördes.

RUNNERS: {strategi: (bloblnyckel i load_blobs, funktion(data) → ny blob)}. Funktionerna tar den gamla bloben
där den behövs (Arc: Nordens del byts, USA-delen behålls) och returnerar en blob i samma form som Gisten har.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

HEAVY = {"quality": "Quality Contrarian (~12 min)", "deep": "Deep Contrarian", "momentum": "Momentum Swing (~12 min)",
         "long": "Long Screener (Valhall-kartan)"}


def _api():
    from borsdata_api import get_api
    return get_api()


def _arc(part: str) -> Callable:
    def run(data: dict) -> dict:
        import arc_scan
        blob = dict((data or {}).get("arc") or {})
        blob[part] = arc_scan.run_wolf() if part == "wolf" else arc_scan.run_viking()
        return blob
    return run


def _viking_nine(data: dict) -> dict:
    """Norden live (steg 1 på hela registrets Norden, steg 2 på de 60 bästa); USA-raderna ur senaste körningen."""
    import viking_nine_scan as vns
    members = vns.universes(usa=False)
    try:
        from ovtlyr.ui.viking_screens import _earnings
    except Exception:                                             # pragma: no cover
        _earnings = None
    fresh = vns.run(members=members, dry_run=True, earnings_getter=_earnings, held={})["payload"]
    old = (data or {}).get("viking_nine") or {}
    usa = ((old.get("markets") or {}).get("usa")) or {"funnel": {}, "rows": []}
    fresh["markets"]["usa"] = usa
    fresh["rows"] = list(fresh["markets"]["norden"]["rows"]) + list(usa.get("rows") or [])
    return fresh


def _ember(data: dict) -> dict:
    from ember.cache import _setup_to_dict
    from ember.engine import run_ember_scan
    from ember.universe import SOURCE_REGISTRY
    res = run_ember_scan(universe_source=SOURCE_REGISTRY, use_prefilter=True)
    ts = res.timestamp.isoformat() if hasattr(res.timestamp, "isoformat") else str(res.timestamp)
    return {"timestamp": ts, "eligible": [_setup_to_dict(r) for r in res.eligible],
            "near_misses": [_setup_to_dict(r) for r in res.near_misses]}


def _berserk(data: dict) -> dict:
    import berserk_scan
    return berserk_scan.run(dry_run=True)["scan"]


def _insider(data: dict) -> dict:
    import insider_scan
    return insider_scan.scan(_api())


def _screens(data: dict) -> dict:
    import screens_scan
    return screens_scan.scan(_api())


def _royalty(data: dict) -> dict:
    import royalty_scan
    return royalty_scan.run(api=_api(), dry_run=True)["payload"]


RUNNERS: dict = {
    "wolf": ("arc", _arc("wolf")), "viking": ("arc", _arc("viking")), "viking_nine": ("viking_nine", _viking_nine),
    "ember": ("ember", _ember), "berserk": ("berserk", _berserk), "insider": ("insider", _insider),
    "hav": ("screens", _screens), "rule": ("screens", _screens), "durrett": ("screens", _screens),
    "tiggre": ("screens", _screens), "royalty": ("royalty", _royalty),
}


def can_run(strategy: Optional[str]) -> bool:
    return strategy in RUNNERS


def run(strategy: str, data: dict, runners: Optional[dict] = None) -> dict:
    """Kör strategins skanning och returnerar en ny data-dict (den gamla rörs inte) med bloben utbytt och
    data["live"][strategi] = tidsstämpel. Ett fel lyfts vidare — sidan visar det."""
    key, fn = (runners or RUNNERS)[strategy]
    out = dict(data or {})
    out[key] = fn(out)
    live = dict(out.get("live") or {})
    live[strategy] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    out["live"] = live
    return out


def freshness(rows: list, data: dict, today=None) -> list:
    """[(strategi, etikett, n, körd, läge)] för alla strategier — läge OK / GAMMAL / SAKNAS / LIVE."""
    from thor import candidates as tc
    from thor import registry as reg
    s = tc.summary(rows)["by_strategy"]
    live = (data or {}).get("live") or {}
    out = []
    for key, st in reg.STRATEGIES.items():
        b = s.get(key) or {}
        n, asof = b.get("n", 0), b.get("asof") or ""
        if key in live:
            state = "LIVE"
            asof = live[key]
        elif not n:
            state = "SAKNAS"
        elif b.get("stale"):
            state = "GAMMAL"
        else:
            state = "OK"
        out.append((key, st.label, n, asof, state))
    return out
