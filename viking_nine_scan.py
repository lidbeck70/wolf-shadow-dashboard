#!/usr/bin/env python3
"""
viking_nine_scan.py — headless Viking Nine över hela universum-registret, Norden och USA.

Samma två steg som ⚔️ Viking Nine-fliken (viking_screen.scan): steg 1 på stängningskurser över hela
universumet (aktiens Trend och Signal, rangordnat på avkastning tre månader), steg 2 full OVTLYR Nine och
Viking Execution på de bästa kandidaterna. Inga regler ändras — samma motor, samma trösklar.

  Norden  registrets nordiska bolag på huvudlistorna (Large/Mid/Small Cap, Oslo, Köpenhamn, Helsingfors)
  USA     samma universum som Arc-USA: NYSE/Nasdaq med börsvärde >= 2 000 MUSD (arc_scan.registry_usa)

Resultatet skrivs till Gisten som viking_nine_scan.json:
  {"generated", "markets": {"norden": {"funnel", "rows"}, "usa": {...}}, "rows": [...]}
där varje rad är Viking Nine-raden i loggens form (viking_screen.log_entry) plus namn, segment och kategori
(GOLDEN TICKET / READY / DEVELOPING). Thor läser filen; Valhall-kartans val per segment ligger kvar.

Env: BORSDATA_API_KEY (börsvärden för USA-filtret), GITHUB_TOKEN (gist-scope).
Flaggor: --dry-run (räkna, skriv inget till Gisten).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("viking_nine_scan")

BLOB_NAME = "viking_nine_scan.json"
MAX_CANDIDATES = {"norden": 60, "usa": 60}      # steg 2 per marknad (flikens förval är 40, taket 100)
KEEP = ("GOLDEN TICKET", "READY", "DEVELOPING")  # REJECTED och fel sparas inte


def universes(loader=None, usa_api=None, usa: bool = True) -> dict:
    """{marknad: {ticker: (namn, segment)}} ur registret — tomt för en marknad som inte går att läsa.
    usa=False hoppar över USA (Thors "Skanna nu" kör bara Norden live)."""
    from thor import candidates as tc
    from universe import registry as ur
    from universe import scan as uscan
    out = {"norden": {}, "usa": {}}
    try:
        reg = ur.load(None, loader)
    except Exception as exc:
        log.warning("Universum-registret gick inte att läsa (%s) — inget att skanna.", exc)
        return out
    if reg is None:
        log.warning("Universum-registret saknas i Gisten — inget att skanna.")
        return out
    segs = tc.segment_map_from_register(reg)
    for m in uscan.members_for("viking_nine", universe=reg, markets=("norden",)) or []:
        out["norden"][m.ticker] = (m.name or m.ticker, segs.get(m.ticker))
    if not usa:
        return out
    try:
        import arc_scan
        usa_reg = arc_scan.registry_usa(loader, usa_api) or {}
    except Exception as exc:
        log.warning("USA-universumet gick inte att räkna (%s) — USA hoppas över.", exc)
        usa_reg = {}
    for t, name in usa_reg.items():
        out["usa"][t] = (name, segs.get(t.upper()))
    return out


def scan_market(key: str, members: dict, scan_fn: Optional[Callable] = None, **kw) -> dict:
    """Viking Nine på en marknad → {"funnel", "rows"} med rader i loggens form."""
    import viking_screen as vs
    if scan_fn is None:
        scan_fn = vs.scan
    if not members:
        return {"funnel": {"universe": 0}, "rows": []}
    res = scan_fn(sorted(members), max_candidates=MAX_CANDIDATES.get(key, vs.MAX_CANDIDATES), **kw)
    rows = []
    for r in res.get("rows") or []:
        if r.get("category") not in KEEP:
            continue
        e = vs.log_entry(r, source="viking_nine_scan")
        if e is None:
            continue
        name, seg = members.get(r["ticker"], (r["ticker"], None))
        e.update(name=name, segment=seg, category=r.get("category"), market=key)
        for k in ("timestamp", "label", "source"):
            e.pop(k, None)
        rows.append(e)
    order = {c: i for i, c in enumerate(KEEP)}
    rows.sort(key=lambda e: (order.get(e["category"], 9), -(e.get("nine_weighted") or 0), e["ticker"]))
    funnel = {k: v for k, v in (res.get("funnel") or {}).items() if k != "momentum"}
    return {"funnel": funnel, "rows": rows}


def run(loader=None, usa_api=None, save=None, dry_run: bool = False, scan_fn: Optional[Callable] = None,
        members: Optional[dict] = None, **kw) -> dict:
    """Hela körningen. members = {marknad: {ticker: (namn, segment)}} ersätter registret (tester)."""
    members = members if members is not None else universes(loader, usa_api)
    payload = {"generated": datetime.now(tz=timezone.utc).isoformat(), "markets": {}, "rows": []}
    for key in ("norden", "usa"):
        m = members.get(key) or {}
        try:
            res = scan_market(key, m, scan_fn=scan_fn, **kw)
            res["error"] = None
        except Exception as exc:
            log.exception("Viking Nine %s fallerade: %s", key, exc)
            res = {"funnel": {"universe": len(m)}, "rows": [], "error": str(exc)}
        payload["markets"][key] = res
        payload["rows"] += res["rows"]
        cats = {c: sum(1 for r in res["rows"] if r["category"] == c) for c in KEEP}
        log.info("Viking Nine %s: %d bolag · steg 1 %s · %s", key, len(m), res["funnel"].get("passed", "—"),
                 " · ".join(f"{c} {n}" for c, n in cats.items()))
    if dry_run:
        log.info("[DRY-RUN] sparar inte till Gisten.")
        return {"payload": payload, "saved": False}
    if save is None:
        from gist_storage import save_blob as save
    saved = bool(save(BLOB_NAME, payload))
    log.info("Sparat till Gisten som %s." if saved else "Kunde inte spara %s — kontrollera GITHUB_TOKEN.", BLOB_NAME)
    return {"payload": payload, "saved": saved}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    import arc_scan
    try:
        from ovtlyr.ui.viking_screens import _earnings
    except Exception:                                             # pragma: no cover
        _earnings = None
    run(usa_api=arc_scan.borsdata_for_usa(), dry_run=args.dry_run, earnings_getter=_earnings, held={})
    return 0


if __name__ == "__main__":
    sys.exit(main())
