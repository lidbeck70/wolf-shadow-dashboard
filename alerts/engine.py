"""
alerts/engine.py
================
Central dispatcher.  Routes an alert message to one or more named channels
and returns a per-channel result dict so callers can log failures without
crashing.

Usage
-----
    from alerts.engine import send_alert

    results = send_alert(
        "NVDA — entry signal fired (score 92)",
        channels=["discord", "webhook"],
        metadata={"ticker": "NVDA", "score": 92},
    )
    # results == {"discord": True, "webhook": False, ...}

Supported channel names
-----------------------
    "discord"   alerts/channels/discord.py
    "email"     alerts/channels/email.py
    "webhook"   alerts/channels/webhook.py
"""

from __future__ import annotations

import importlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHANNEL_MODULE_MAP: Dict[str, str] = {
    "discord": "alerts.channels.discord",
    "email":   "alerts.channels.email",
    "webhook": "alerts.channels.webhook",
}

# In-memory alert log — ring buffer, capped at _LOG_MAX entries.
# Read by tabs/alerts.py to display the recent alert history.
ALERT_LOG: List[Dict[str, Any]] = []
_LOG_MAX = 200


def send_alert(
    message: str,
    channels: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """
    Send *message* to every channel in *channels*.

    Parameters
    ----------
    message  : Human-readable alert text.
    channels : List of channel names (see module docstring for valid names).
    metadata : Optional dict passed through to each channel's send() function.
               Channels may use it for structured payloads (embed fields, etc.).

    Returns
    -------
    dict mapping channel name → True (sent) / False (failed / skipped).
    """
    results: Dict[str, bool] = {}

    for name in channels:
        module_path = _CHANNEL_MODULE_MAP.get(name.lower())
        if module_path is None:
            logger.warning("alerts.engine: unknown channel %r — skipping", name)
            results[name] = False
            continue

        try:
            mod = importlib.import_module(module_path)
            ok  = mod.send(message, metadata=metadata)
            results[name] = bool(ok)
        except Exception as exc:
            logger.error("alerts.engine: channel %r raised %s: %s", name, type(exc).__name__, exc)
            results[name] = False

    # Append to in-memory log regardless of outcome.
    entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "message":   message,
        "channels":  list(channels),
        "results":   results,
        "metadata":  dict(metadata) if metadata else {},
    }
    ALERT_LOG.append(entry)
    if len(ALERT_LOG) > _LOG_MAX:
        del ALERT_LOG[: len(ALERT_LOG) - _LOG_MAX]

    return results
