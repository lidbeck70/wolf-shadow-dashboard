"""
Thor — "Skanna nu": kör en strategis skanning direkt från listan, med samma headless-funktion som den schemalagda
körningen, och ersätter strategins rader i sessionen. Inga regler ändras — samma motorer, samma förval.

Snabba skanningar körs live: Viking Nine (dagens nordiska rader räknas om), EMBER (dagens rader räknas om),
Insider, håvarna, Royalty. De tunga körs bara i schemat och Thor säger när de senast kördes: Quality/Deep
Contrarian (~12 min), Momentum Swing (~12 min), Long Screener (Valhall-kartan), Arc Wolf/Viking (~6 min, fyra
marknader + registret) och BERSERK (~40 min, egen körning 21:35 UTC).

RUNNERS: {strategi: (bloblnyckel i load_blobs, funktion(data) → ny blob)}. Funktionerna tar den gamla bloben
där den behövs (Viking Nine: USA-delen behålls; EMBER: dagens tickers) och returnerar en blob i samma form som
Gisten har. En skanning som fallerar (skannerns eget error-fält, eller 0 rader där Gisten hade rader) lyfts som
LiveScanError — sidan visar felet och behåller Gistens rader; strategin stämplas inte LIVE.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

HEAVY = {"quality": "Quality Contrarian (~12 min)", "deep": "Deep Contrarian", "momentum": "Momentum Swing (~12 min)",
         "long": "Long Screener (Valhall-kartan)", "wolf": "Arc-skanningen (~6 min)", "viking": "Arc-skanningen (~6 min)",
         "berserk": "BERSERK (~40 min, egen körning 21:35 UTC)"}
# Strategier vars rader bara delvis kommer ur den blob som skannas live: resten (arken, kartan) behåller sitt datum.
PARTIAL = {"rule": "håven", "durrett": "håven", "tiggre": "håven", "viking_nine": "skanningen"}


class LiveScanError(RuntimeError):
    """Skanningen gick inte att lita på — Gistens rader behålls."""


def _api():
    from borsdata_api import get_api
    return get_api()


def _viking_nine(data: dict) -> dict:
    """Norden live: dagens nordiska rader räknas om (steg 1 + steg 2 på just dem); finns inga rader skannas hela
    registrets Norden som i schemat. USA-raderna ur senaste körningen behålls."""
    import viking_nine_scan as vns
    members = vns.universes(usa=False)
    old = (data or {}).get("viking_nine") or {}
    today_rows = ((old.get("markets") or {}).get("norden") or {}).get("rows") or []
    have = {str(r.get("ticker") or "") for r in today_rows}
    if have and members.get("norden"):
        members["norden"] = {t: v for t, v in members["norden"].items() if t in have} or members["norden"]
    try:
        from ovtlyr.ui.viking_screens import _earnings
    except Exception:                                             # pragma: no cover
        _earnings = None
    fresh = vns.run(members=members, dry_run=True, earnings_getter=_earnings, held={})["payload"]
    usa = ((old.get("markets") or {}).get("usa")) or {"funnel": {}, "rows": []}
    fresh["markets"]["usa"] = usa
    fresh["rows"] = list(fresh["markets"]["norden"]["rows"]) + list(usa.get("rows") or [])
    return fresh


def _ember(data: dict) -> dict:
    """Dagens EMBER-rader (köplägen + near-miss) räknas om; utan rader körs registret med förfiltret som i schemat."""
    from ember.cache import _setup_to_dict
    from ember.engine import run_ember_scan
    from ember.universe import SOURCE_REGISTRY
    old = (data or {}).get("ember") or {}
    tickers = [str(r.get("ticker")) for r in list(old.get("eligible") or []) + list(old.get("near_misses") or [])
               if r.get("ticker")]
    if tickers:
        res = run_ember_scan(tickers=tickers, universe_source=SOURCE_REGISTRY, use_prefilter=False)
    else:
        res = run_ember_scan(universe_source=SOURCE_REGISTRY, use_prefilter=True)
    ts = res.timestamp.isoformat() if hasattr(res.timestamp, "isoformat") else str(res.timestamp)
    return {"timestamp": ts, "eligible": [_setup_to_dict(r) for r in res.eligible],
            "near_misses": [_setup_to_dict(r) for r in res.near_misses]}


def _insider(data: dict) -> dict:
    import insider_scan
    return insider_scan.scan(_api())


def _screens(data: dict) -> dict:
    """Håvarna — med NY-märkena satta mot Gistens förra körning, som schemats main() gör."""
    import screens_scan
    return screens_scan.mark_first_seen(screens_scan.scan(_api()), (data or {}).get("screens"))


def _royalty(data: dict) -> dict:
    import royalty_scan
    return royalty_scan.run(api=_api(), dry_run=True)["payload"]


RUNNERS: dict = {
    "viking_nine": ("viking_nine", _viking_nine), "ember": ("ember", _ember), "insider": ("insider", _insider),
    "hav": ("screens", _screens), "rule": ("screens", _screens), "durrett": ("screens", _screens),
    "tiggre": ("screens", _screens), "royalty": ("royalty", _royalty),
}


def can_run(strategy: Optional[str]) -> bool:
    return strategy in RUNNERS


# ── Blobbarnas hälsa: rader och skannerns egna felfält ───────────────────────
def _health(key: str, blob, part: Optional[str] = None) -> tuple:
    """(antal rader, feltext eller None, generated) för en blob i Gistens form."""
    b = blob if isinstance(blob, dict) else {}

    def _d(x):
        return x if isinstance(x, dict) else {}

    def _n(x):
        return len(x) if isinstance(x, (list, tuple)) else 0

    if key == "arc":
        p = _d(b.get(part)) if part else {}
        return _n(p.get("rows")), p.get("error") or None, p.get("generated")
    if key == "viking_nine":
        mk = _d(b.get("markets"))
        errs = [f"{m}: {v.get('error')}" for m, v in mk.items() if isinstance(v, dict) and v.get("error")]
        return _n(b.get("rows")), ("; ".join(errs) or None), b.get("generated")
    if key == "ember":
        return _n(b.get("eligible")) + _n(b.get("near_misses")), b.get("error") or None, b.get("timestamp")
    if key == "berserk":
        miss, n = b.get("missing"), _n(b.get("rows"))
        return n, (f"saknar {miss}" if miss and not n else None), b.get("generated")
    if key == "insider":
        return _n(b.get("clusters")), b.get("error") or None, b.get("generated")
    if key == "screens":
        sc = {k: v for k, v in _d(b.get("screens")).items() if isinstance(v, dict)}
        n = sum(_n(v.get("rows")) for v in sc.values())
        errs = [f"{k}: {v.get('error')}" for k, v in sc.items() if v.get("error")]
        return n, ("; ".join(errs) if errs and len(errs) == len(sc) else None), b.get("generated")
    return _n(b.get("rows")), b.get("error") or None, b.get("generated")


BLOB_OF = {"wolf": ("arc", "wolf"), "viking": ("arc", "viking"), "viking_nine": ("viking_nine", None),
           "ember": ("ember", None), "berserk": ("berserk", None), "insider": ("insider", None),
           "hav": ("screens", None), "rule": ("screens", None), "durrett": ("screens", None),
           "tiggre": ("screens", None), "royalty": ("royalty", None)}


def blob_errors(data: dict) -> dict:
    """{strategi: (feltext, generated)} för strategier vars Gist-blob bär ett fel och inga rader — en körning som
    kördes men fallerade (nät, licens, nyckel) ska inte se ut som 'ingen körning'."""
    out = {}
    for strategy, (key, part) in BLOB_OF.items():
        n, err, gen = _health(key, (data or {}).get(key), part)
        if err and not n:
            out[strategy] = (str(err), str(gen or "")[:16])
    return out


def run(strategy: str, data: dict, runners: Optional[dict] = None) -> dict:
    """Kör strategins skanning och returnerar en ny data-dict (den gamla rörs inte) med bloben utbytt och
    data["live"][strategi] = tidsstämpel. Fallerar skanningen (error-fält, eller 0 rader där Gisten hade rader)
    lyfts LiveScanError och inget byts."""
    key, fn = (runners or RUNNERS)[strategy]
    out = dict(data or {})
    new = fn(out)
    part = BLOB_OF.get(strategy, (key, None))[1]
    n_new, err, _ = _health(key, new, part)
    n_old, _, old_gen = _health(key, out.get(key), part)
    if err and not n_new:
        raise LiveScanError(f"{err} — Gistens rader behålls")
    if not n_new and n_old:
        raise LiveScanError(f"skanningen gav 0 rader — Gistens {n_old} rader från {str(old_gen or '')[:16] or 'senaste körningen'} behålls")
    out[key] = new
    live = dict(out.get("live") or {})
    live[strategy] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    out["live"] = live
    return out


def freshness(rows: list, data: dict, today=None) -> list:
    """[(strategi, etikett, n, körd, läge)] för alla strategier — läge OK / GAMMAL / SAKNAS / FEL / LIVE.
    LIVE (håven) m.fl. betyder att bara en av strategins källor skannades live; övriga rader har kartans datum."""
    from thor import candidates as tc
    from thor import registry as reg
    s = tc.summary(rows)["by_strategy"]
    live = (data or {}).get("live") or {}
    errors = blob_errors(data)
    out = []
    for key, st in reg.STRATEGIES.items():
        b = s.get(key) or {}
        n, asof = b.get("n", 0), b.get("asof") or ""
        if key in live:
            state = f"LIVE ({PARTIAL[key]})" if key in PARTIAL else "LIVE"
            oldest = min((c.asof for c in rows if c.strategy == key and c.asof), default="")
            asof = f"{live[key]} · övriga rader {oldest}" if key in PARTIAL and oldest else live[key]
        elif not n and key in errors:
            state = "FEL"
            asof = errors[key][1]
        elif not n:
            state = "SAKNAS"
        elif b.get("stale"):
            state = "GAMMAL"
        else:
            state = "OK"
        out.append((key, st.label, n, asof, state))
    return out
