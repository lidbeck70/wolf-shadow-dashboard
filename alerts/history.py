"""
alerts/history.py — larmloggen i Gisten (designförslaget PR 2).

alert_scan.py sparar de senaste LIMIT larmen till alert_log.json efter varje utskick, så loggen överlever att
panelen startar om och visar också larmen som de schemalagda körningarna skickat. Varje rad bär larmets mål
(ui/nav.TARGETS), så ALERTS och startsidan kan öppna rätt vy med ett tryck. Rena funktioner, inget nätverk.
"""
from __future__ import annotations

from typing import Optional

BLOB = "alert_log.json"
LIMIT = 50


def entry(alert: dict, now: str, delivered: Optional[bool] = None) -> dict:
    """En loggrad ur ett larm från alert_rules (kind, title, body, target, ticker, strategy)."""
    out = {"time": now, "kind": str(alert.get("kind") or ""), "title": str(alert.get("title") or ""),
           "body": str(alert.get("body") or "")}
    for k in ("target", "ticker", "strategy"):
        if alert.get(k):
            out[k] = str(alert[k])
    if delivered is not None:
        out["delivered"] = bool(delivered)
    return out


def append(log, alerts: list, now: str, delivered: Optional[dict] = None, limit: int = LIMIT) -> list:
    """Gamla loggen + de nya larmen, nyast sist, högst limit rader. delivered = {titel: bool} (valfritt)."""
    rows = [r for r in (log if isinstance(log, list) else []) if isinstance(r, dict)]
    rows += [entry(a, now, (delivered or {}).get(a.get("title"))) for a in alerts or []]
    return rows[-limit:]


def latest(log, n: int = 8) -> list:
    """De n senaste raderna, nyast först."""
    rows = [r for r in (log if isinstance(log, list) else []) if isinstance(r, dict)]
    return list(reversed(rows[-n:]))
