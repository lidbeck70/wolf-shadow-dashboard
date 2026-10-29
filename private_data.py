"""
private_data.py — kryptering av de privata arken i data/ (positionerna och de fysiska metallerna).

Repot är publikt, så data/<ark>.json på grenen panel-data kan läsas av vem som helst. De privata arken sparas
därför krypterade: filen innehåller bara {"_encrypted": "fernet-scrypt-v1", "data": "<chiffer>"}.

Nyckeln är en lösenfras i Streamlit Secrets (PANEL_DATA_KEY) och — för skannrarna som läser positionerna —
i GitHub Secrets med samma namn. Fernet-nyckeln härleds ur frasen med scrypt. Tappas frasen bort går arken inte
att läsa; den måste sparas på säkert ställe. Okrypterade äldre filer läses som förut och krypteras vid nästa
sparning.

Gisten är lika publik (id:t står i koden). Filerna där som bär innehaven (PRIVATE_BLOBS) sparas därför i samma
kuvert när nyckeln finns; utan nyckel går de i klartext som förut, med en varning i loggen (F12).
"""
from __future__ import annotations

import base64
import functools
import hashlib
import json
import logging
import os
from typing import Any, Optional

PRIVATE_STORES = frozenset({"holdings", "metals"})
# Gist-filerna med innehavens tickers: larmtillståndet (swing-verdikt per innehav), larmloggen, sifferuppdateringen
# (registrets kurser, Tiggre-positionernas händelser) och tradejournalen (antal och resultat i kronor).
PRIVATE_BLOBS = frozenset({"alert_state.json", "alert_log.json", "sheets_refresh.json", "journal_data.json"})
KEY_NAME = "PANEL_DATA_KEY"
SCHEME = "fernet-scrypt-v1"
_SALT = b"nordic-arc-panel-data-v1"
log = logging.getLogger("private_data")
_warned = False                        # klartextvarningen skrivs en gång per körning


class PrivateDataError(RuntimeError):
    """Kryptering eller dekryptering gick inte. Meddelandet är skrivet för att visas i UI:t."""


def passphrase() -> Optional[str]:
    """Lösenfrasen ur miljön (Actions) eller Streamlit Secrets — None när den saknas. Visas aldrig."""
    v = os.environ.get(KEY_NAME, "").strip()
    if v:
        return v
    try:
        import streamlit as st
        v = str(st.secrets.get(KEY_NAME, "") or "").strip()
    except Exception:
        v = ""
    return v or None


@functools.lru_cache(maxsize=4)
def _derived(phrase: str) -> bytes:
    """scrypt-nyckeln för frasen — ~40 ms, så den räknas en gång per process (varje Gist-läsning av en privat fil
    öppnar ett kuvert)."""
    return hashlib.scrypt(phrase.encode("utf-8"), salt=_SALT, n=2 ** 14, r=8, p=1, dklen=32)


def _fernet(phrase: str):
    from cryptography.fernet import Fernet
    return Fernet(base64.urlsafe_b64encode(_derived(phrase)))


def is_private(name: str) -> bool:
    return str(name or "") in PRIVATE_STORES


def is_encrypted(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("_encrypted") == SCHEME and isinstance(obj.get("data"), str)


def seal(data: Any, phrase: Optional[str] = None) -> dict:
    """Krypterar data till kuvertet. Kastar PrivateDataError när nyckeln saknas — privata ark sparas aldrig
    okrypterade."""
    phrase = phrase or passphrase()
    if not phrase:
        raise PrivateDataError(f"{KEY_NAME} saknas i Secrets — de privata arken sparas inte okrypterade i det publika "
                               f"repot. Lägg in en lång lösenfras som {KEY_NAME} (Streamlit: Settings → Secrets) och "
                               f"spara igen.")
    body = json.dumps(data, ensure_ascii=False, default=str, sort_keys=True).encode("utf-8")
    return {"_encrypted": SCHEME, "data": _fernet(phrase).encrypt(body).decode("ascii")}


def unseal(obj: Any, phrase: Optional[str] = None) -> Any:
    """Kuvert → data. Okrypterat innehåll returneras som det är. Kastar PrivateDataError när nyckeln saknas eller
    inte passar."""
    if not is_encrypted(obj):
        return obj
    phrase = phrase or passphrase()
    if not phrase:
        raise PrivateDataError(f"Arket är krypterat men {KEY_NAME} saknas i Secrets.")
    try:
        from cryptography.fernet import InvalidToken
        raw = _fernet(phrase).decrypt(obj["data"].encode("ascii"))
    except InvalidToken as exc:
        raise PrivateDataError(f"{KEY_NAME} passar inte arket — fel lösenfras?") from exc
    return json.loads(raw.decode("utf-8"))


def is_private_blob(filename: str) -> bool:
    return str(filename or "") in PRIVATE_BLOBS


def seal_blob(filename: str, data: Any) -> Any:
    """Det som skickas till Gisten: kuvertet för de privata filerna när nyckeln finns. Utan nyckel går datan i
    klartext som förut och en varning loggas en gång per körning — nyckelns värde loggas aldrig."""
    global _warned
    if not is_private_blob(filename):
        return data
    phrase = passphrase()
    if phrase:
        return seal(data, phrase)
    if not _warned:
        _warned = True
        log.warning("%s saknas — innehavsdelen (%s) publiceras i klartext i den publika Gisten tills %s läggs in "
                    "i GitHub Actions secrets (panelen: Streamlit Secrets).", KEY_NAME,
                    ", ".join(sorted(PRIVATE_BLOBS)), KEY_NAME)
    return data


def public_log(text: str) -> str:
    """Text till en logg som kan vara publik (Actions-loggarna i ett publikt repo). Med nyckeln satt döljs den —
    larm- och händelserubriker kan bära innehavens tickers. Utan nyckel som förut."""
    return "[dold]" if passphrase() else text
