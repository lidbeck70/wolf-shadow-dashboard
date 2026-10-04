#!/usr/bin/env python3
"""
berserk_scan.py — 🪓 BERSERK headless: skanning, papperskonto och Discordlarm.

Körs av berserk-scan-workflowen vardagar efter USA:s stängning (då har alla
regioner en stängd dag). Samma regler som panelen:

  1. live.scan över hela universumet (Norden, USA, Kanada, London, Australien
     och råvaru-ETF:erna) på senaste stängda dag.
  2. paper.step förvaltar papperskontot: fyller gårdagens order på öppningen,
     stoppar/säljer med backtestets exitregler, lägger dagens KÖP som order.
  3. Larm till Discord — bara övergångar (KÖP PÅ ÖPPNING, KÖPT, SÄLJ PÅ
     ÖPPNING, SÅLD, STOPPAD, FLYTTA STOPP, SPÄRRAD). Inget nytt = inget larm.

Gisten: berserk_scan.json (dagens KÖP/BEVAKA, läses av panelen) och
berserk_paper.json (papperskontot). Riktiga order läggs alltid manuellt.

Env: BORSDATA_API_KEY (OMXS30 för Nordens marknadsgrind), GITHUB_TOKEN
     (gist-scope), DISCORD_WEBHOOK_URL.
Flaggor: --dry-run (räkna och skriv ut larmen, spara och skicka inget).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("berserk_scan")

SCAN_BLOB = "berserk_scan.json"
PAPER_BLOB = "berserk_paper.json"
ROW_KEYS = ("ticker", "status", "setup", "label", "theme", "complex", "region", "kind", "driver", "close", "stop",
            "position_pct", "risk_pct", "atr", "date", "why", "rsi2", "rvol", "divergence", "dd252")


def universe() -> list:
    from berserk import universe as uv
    return list(dict.fromkeys(t for lst in uv.LISTS.values() for t in lst))


def scan_payload(res: dict, paper_state: Optional[dict], paper_last_run: Optional[str] = None,
                 paper_summary: Optional[dict] = None) -> dict:
    from berserk import live
    from berserk import paper
    rows = live.sort_rows(res["rows"])
    sig = [{k: r.get(k) for k in ROW_KEYS} for r in rows if r.get("status") in (live.KOP, live.BEVAKA)]
    return {"generated": datetime.now(tz=timezone.utc).isoformat(), "when": res.get("when"),
            "tickers": len(rows), "missing": [r["ticker"] for r in rows if r.get("error")],
            "counts": {s: sum(1 for r in rows if r.get("status") == s) for s in (live.KOP, live.BEVAKA)},
            "drivers": res.get("drivers"), "rows": sig,
            "paper_last_run": paper_state.get("last_run") if paper_state else paper_last_run,
            "paper": paper.summary(paper_state) if paper_state else paper_summary}


def run(getter=None, nordic_provider=None, load=None, save=None, send=None, today=None, dry_run=False) -> dict:
    """Hela körningen med utbytbara delar (tester). Returnerar {scan, paper, events, messages, saved, sent}."""
    from berserk import live
    from berserk import paper
    if load is None or save is None:
        from gist_storage import load_blob, save_blob
        load, save = load or load_blob, save or save_blob
    if send is None:
        from alerts.engine import send_alert

        def send(msg):
            return send_alert(msg, channels=["discord"]).get("discord", False)

    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    tickers = universe()
    keep: dict = {}
    res = live.scan(tickers, getter=getter, today=today, nordic_provider=nordic_provider, keep=keep,
                    progress=lambda i, n, t: log.info("%d/%d %s", i, n, t) if i % 25 == 0 or i == n else None)
    prev_scan = load(SCAN_BLOB, None) or {}
    state = load(PAPER_BLOB, None)
    if state is None and prev_scan.get("paper_last_run"):
        # Papperskontot fanns men gick inte att läsa — starta inte om det och skriv inte över.
        log.error("%s gick inte att läsa (fanns senast %s) — hoppar över papperskontot den här gången.",
                  PAPER_BLOB, prev_scan["paper_last_run"])
        payload = scan_payload(res, None, prev_scan["paper_last_run"], prev_scan.get("paper"))
        saved = False if dry_run else bool(save(SCAN_BLOB, payload))
        return {"scan": payload, "paper": None, "events": [], "messages": [], "saved": saved, "sent": 0}
    state, events = paper.step(state, res["rows"], keep, today)
    msgs = paper.messages(events, state)
    payload = scan_payload(res, state)
    log.info("%d tickers · %d KÖP · %d BEVAKA · %d händelser · papper %.2f", payload["tickers"],
             payload["counts"]["KÖP"], payload["counts"]["BEVAKA"], len(events), paper.equity(state))
    if dry_run:
        for m in msgs:
            log.info("[DRY-RUN] larm:\n%s", m)
        return {"scan": payload, "paper": state, "events": events, "messages": msgs, "saved": False, "sent": 0}
    saved = bool(save(PAPER_BLOB, state)) & bool(save(SCAN_BLOB, payload))
    if not saved:
        log.error("Kunde inte spara till Gisten — kontrollera GITHUB_TOKEN (gist-scope).")
    sent = sum(1 for m in msgs if send(m))
    if msgs and sent < len(msgs):
        log.error("Discord: %d av %d meddelanden skickade — kontrollera DISCORD_WEBHOOK_URL.", sent, len(msgs))
    return {"scan": payload, "paper": state, "events": events, "messages": msgs, "saved": saved, "sent": sent}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
