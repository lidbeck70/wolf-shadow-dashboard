"""
private_data.py — kryptering av de privata arken i data/ (positionerna och de fysiska metallerna).

Repot är publikt, så data/<ark>.json på grenen panel-data kan läsas av vem som helst. De privata arken sparas
därför krypterade: filen innehåller bara {"_encrypted": "fernet-scrypt-v1", "data": "<chiffer>"}.

Nyckeln är en lösenfras i Streamlit Secrets (PANEL_DATA_KEY) och — för skannrarna som läser positionerna —
i GitHub Secrets med samma namn. Fernet-nyckeln härleds ur frasen med scrypt. Tappas frasen bort går arken inte
att läsa; den måste sparas på säkert ställe. Okrypterade äldre filer läses som förut och krypteras vid nästa
sparning.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any, Optional

PRIVATE_STORES = frozenset({"holdings", "metals"})
KEY_NAME = "PANEL_DATA_KEY"
SCHEME = "fernet-scrypt-v1"
_SALT = b"nordic-arc-panel-data-v1"


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


def _fernet(phrase: str):
    from cryptography.fernet import Fernet
    raw = hashlib.scrypt(phrase.encode("utf-8"), salt=_SALT, n=2 ** 14, r=8, p=1, dklen=32)
    return Fernet(base64.urlsafe_b64encode(raw))


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
