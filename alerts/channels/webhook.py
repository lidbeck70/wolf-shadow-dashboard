"""
alerts/channels/webhook.py
===========================
Generic HTTP webhook channel — POSTs a JSON payload to any URL.

Environment variables
---------------------
ALERT_WEBHOOK_URL : Default target URL.
                    Can be overridden per-call via metadata["url"].
ALERT_WEBHOOK_TOKEN : Optional Bearer token added as Authorization header.

Optional metadata keys
----------------------
url     : Override ALERT_WEBHOOK_URL for this specific call.
token   : Override ALERT_WEBHOOK_TOKEN for this specific call.
headers : Dict of additional HTTP headers to merge in.
payload : Dict merged into the POST body (takes precedence over defaults).

Default POST body
-----------------
{
    "message":  "<alert text>",
    "metadata": { ...caller metadata minus url/token/headers/payload keys... }
}
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENV_URL   = "ALERT_WEBHOOK_URL"
_ENV_TOKEN = "ALERT_WEBHOOK_TOKEN"
_TIMEOUT   = 10  # seconds

# Keys consumed by the channel itself — not forwarded in the body metadata.
_RESERVED_META_KEYS = {"url", "token", "headers", "payload"}


def send(message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    POST an alert to a configurable HTTP endpoint.

    Returns True on HTTP 2xx, False on any error.
    """
    meta = metadata or {}

    target_url = (
        str(meta.get("url", "")).strip()
        or os.environ.get(_ENV_URL, "").strip()
    )
    if not target_url:
        logger.warning("webhook channel: no URL configured (%s unset) — alert skipped", _ENV_URL)
        return False

    token = (
        str(meta.get("token", "")).strip()
        or os.environ.get(_ENV_TOKEN, "").strip()
    )

    # Build headers
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if isinstance(meta.get("headers"), dict):
        headers.update({str(k): str(v) for k, v in meta["headers"].items()})

    # Build body — forward metadata minus channel-internal keys
    forwarded_meta = {k: v for k, v in meta.items() if k not in _RESERVED_META_KEYS}
    body: Dict[str, Any] = {
        "message":  message,
        "metadata": forwarded_meta,
    }
    if isinstance(meta.get("payload"), dict):
        body.update(meta["payload"])

    raw_body = json.dumps(body, default=str).encode("utf-8")

    req = urllib.request.Request(
        target_url,
        data=raw_body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        logger.error("webhook channel: HTTP %d %s — url=%s", exc.code, exc.reason, target_url)
        return False
    except OSError as exc:
        logger.error("webhook channel: network error — %s", exc)
        return False

    if status not in range(200, 300):
        logger.error("webhook channel: unexpected status %d — url=%s", status, target_url)
        return False

    logger.debug("webhook channel: alert sent (HTTP %d) — url=%s", status, target_url)
    return True
