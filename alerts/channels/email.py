"""
alerts/channels/email.py
========================
Email channel — riktig SMTP-leverans.

Var en attrapp som loggade och returnerade True — alltså exakt det mönster
som gömde sparbuggen: ett misslyckande som ser ut som en leverans. Nu skickas
mejlet på riktigt, och varje väg som INTE levererar returnerar False med en
logg som säger vad som saknas.

Environment variables
---------------------
EMAIL_FROM    : Avsändaradress (default: SMTP_USER).
EMAIL_TO      : Kommaseparerade mottagare. KRÄVS.
SMTP_HOST     : SMTP-server. KRÄVS (t.ex. smtp.gmail.com).
SMTP_PORT     : Port (default 587, STARTTLS).
SMTP_USER     : Inloggning (för Gmail: adressen; lösenordet är ett
                app-lösenord, inte kontolösenordet).
SMTP_PASSWORD : Lösenord.

Optional metadata keys
----------------------
subject : Ämnesrad. Default "Nordic Arc — larm".
to      : Skriv över EMAIL_TO för ett enskilt larm.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_SUBJECT = "Nordic Arc — larm"


def _smtp_send(subject: str, body: str, to_addrs: list[str],
               from_addr: str) -> bool:
    """Skicka via STARTTLS. False med tydlig logg på varje fel."""
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587") or 587)
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "")

    if not host:
        logger.warning("email channel: SMTP_HOST saknas — larmet skickades "
                       "INTE. Sätt SMTP_HOST/SMTP_USER/SMTP_PASSWORD.")
        return False

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)

    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.ehlo()
            if user:
                smtp.starttls()
                smtp.ehlo()
                smtp.login(user, password)
            smtp.sendmail(from_addr, to_addrs, msg.as_string())
        logger.info("email channel: skickat till %s (%r)", to_addrs, subject)
        return True
    except Exception as exc:
        logger.error("email channel: sändningen misslyckades (%s): %s",
                     type(exc).__name__, exc)
        return False


def send(message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Skicka larmet som mejl. False när konfigurationen saknas — ett
    oskickat mejl får inte räknas som levererat."""
    meta = metadata or {}

    subject = str(meta.get("subject", DEFAULT_SUBJECT))
    user = os.environ.get("SMTP_USER", "").strip()
    from_addr = os.environ.get("EMAIL_FROM", "").strip() or user or \
        "alerts@wolf-shadow.local"
    raw_to = str(meta.get("to", os.environ.get("EMAIL_TO", ""))).strip()
    to_addrs = [a.strip() for a in raw_to.split(",") if a.strip()]

    if not to_addrs:
        logger.warning("email channel: EMAIL_TO saknas — larmet skickades "
                       "INTE.")
        return False

    return _smtp_send(subject, message, to_addrs, from_addr)
