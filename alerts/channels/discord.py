"""
alerts/channels/discord.py
===========================
Discord channel — posts a message to a Discord webhook.

Environment variables
---------------------
DISCORD_WEBHOOK_URL : Full Discord webhook URL.
                      Required; channel silently skips if absent.

Optional metadata keys
----------------------
username  : Override the webhook's display name.
avatar_url: Override the webhook's avatar.
embeds    : List of Discord embed dicts (raw API format).
            When present the plain *message* is sent as the embed description
            and other top-level fields (title, color) are applied if given.
title     : Embed title (used only when embeds not provided).
color     : Embed sidebar color as an integer (used only when embeds not provided).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENV_KEY = "DISCORD_WEBHOOK_URL"
_TIMEOUT = 10  # seconds


def send(message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    POST *message* to the Discord webhook configured in DISCORD_WEBHOOK_URL.

    Returns True on HTTP 2xx, False on any error.
    """
    webhook_url = os.environ.get(_ENV_KEY, "").strip()
    if not webhook_url:
        logger.warning("discord channel: %s not set — alert skipped", _ENV_KEY)
        return False

    meta = metadata or {}

    # Build the payload.
    payload: Dict[str, Any] = {}

    if meta.get("username"):
        payload["username"] = str(meta["username"])
    if meta.get("avatar_url"):
        payload["avatar_url"] = str(meta["avatar_url"])

    if meta.get("embeds"):
        # Caller supplied raw embed objects — pass them through unchanged.
        payload["embeds"] = meta["embeds"]
    elif meta.get("title") or meta.get("color") is not None:
        # Construct a simple embed from title + message + optional color.
        embed: Dict[str, Any] = {"description": message}
        if meta.get("title"):
            embed["title"] = str(meta["title"])
        if meta.get("color") is not None:
            embed["color"] = int(meta["color"])
        payload["embeds"] = [embed]
    else:
        # Plain text — Discord limits content to 2000 characters.
        payload["content"] = message[:2000]

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        logger.error("discord channel: HTTP %d — %s", exc.code, exc.reason)
        return False
    except OSError as exc:
        logger.error("discord channel: network error — %s", exc)
        return False

    if status not in range(200, 300):
        logger.error("discord channel: unexpected status %d", status)
        return False

    logger.debug("discord channel: alert sent (HTTP %d)", status)
    return True
