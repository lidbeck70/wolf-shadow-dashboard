"""
alerts/channels/email.py
========================
Email channel — placeholder implementation.

No real SMTP is wired up.  The function logs the alert at INFO level so it
appears in the application log and returns True so the engine counts it as
delivered.  Replace the body of _smtp_send() with real SMTP logic when
credentials are available.

Environment variables (for future SMTP integration)
----------------------------------------------------
EMAIL_FROM    : Sender address.
EMAIL_TO      : Comma-separated recipient addresses.
SMTP_HOST     : SMTP server hostname  (default: localhost).
SMTP_PORT     : SMTP server port      (default: 587).
SMTP_USER     : SMTP login username   (optional).
SMTP_PASSWORD : SMTP login password   (optional).

Optional metadata keys
----------------------
subject : Email subject line.  Defaults to "Wolf-Shadow Alert".
to      : Override EMAIL_TO for this specific alert.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _smtp_send(
    subject: str,
    body: str,
    to_addrs: list[str],
    from_addr: str,
) -> bool:
    """
    Placeholder — replace with smtplib logic when SMTP credentials exist.

    Expected implementation sketch::

        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = from_addr
        msg["To"]      = ", ".join(to_addrs)

        with smtplib.SMTP(host, port) as smtp:
            if user:
                smtp.starttls()
                smtp.login(user, password)
            smtp.sendmail(from_addr, to_addrs, msg.as_string())
    """
    logger.info(
        "email channel [PLACEHOLDER] to=%s subject=%r body=%r",
        to_addrs, subject, body[:120],
    )
    return True


def send(message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Send (placeholder) email alert.

    Returns True always unless required address configuration is missing
    and strict mode is needed — currently always True.
    """
    meta = metadata or {}

    subject  = str(meta.get("subject", "Wolf-Shadow Alert"))
    from_env = os.environ.get("EMAIL_FROM", "alerts@wolf-shadow.local").strip()
    to_env   = os.environ.get("EMAIL_TO",   "").strip()

    raw_to   = str(meta.get("to", to_env))
    to_addrs = [a.strip() for a in raw_to.split(",") if a.strip()]

    if not to_addrs:
        logger.warning("email channel: no recipient address configured — alert logged only")
        logger.info("email channel [NO RECIPIENT]: subject=%r body=%r", subject, message[:120])
        return True  # non-fatal; return True to avoid flooding error logs

    return _smtp_send(subject, message, to_addrs, from_env)
