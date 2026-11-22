"""
storage_ui.py — sparraden som varje flik ritar.

En knapp, en osparat-varning och ett kvitto med committ-id. Skriven en gång
så att alla flikar beter sig likadant och ingen kan råka spara tyst.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

import storage
from ui.tokens import MUTED          # sparradens rad: 8:1 — DIM är bara för text ≥14 px

GREEN, AMBER, RED, DIM = "#2d8a4e", "#d4943a", "#c44545", "#8a8578"


def save_bar(name: str, label: str = "", key: Optional[str] = None) -> bool:
    """Ritar sparraden för ett lager. Returnerar True om något sparades.

    Anropas överst i fliken, efter storage.session_load().
    """
    key = key or f"save_{name}"
    err = storage.load_failed(name) or storage.load_error(name)   # läsfelet står kvar till omladdning
    if err:
        st.error(f"Kunde inte läsa sparad data för {label or name}: {err}\n\n"
                 f"Fliken visar tom data. Spara INTE förrän det här är löst — "
                 f"en sparning skulle skriva över det som ligger i GitHub.")

    dirty = storage.is_dirty(name)
    saved = False
    ok_msg = fail_msg = ""

    # EN rad (LIDSKJALV §3.5.17): [💾 Spara]  Inga osparade ändringar · senast 16:40 UTC — samma logik som förut,
    # kvittot och felet under raden i stället för i en smal kolumn
    with st.container(horizontal=True, vertical_alignment="center", key=f"{key}_row"):
        if st.button("💾 Spara", key=key, type="primary" if dirty else "secondary",
                     disabled=bool(err)):
            try:
                result = storage.save_session(name)
                ok_msg = f"Sparat — commit {result.short_sha}"
                saved = True
            except storage.StorageError as exc:
                fail_msg = (f"SPARNINGEN MISSLYCKADES: {exc}\n\n"
                            f"Ändringarna finns kvar i sessionen men är INTE "
                            f"sparade. Lämna inte fliken förrän det är löst.")

        if saved:
            pass
        elif dirty:
            st.markdown(
                f"<div style='color:{AMBER};font-size:13px;line-height:1.3;'>"
                f"⚠️ Du har osparade ändringar.</div>",
                unsafe_allow_html=True)
        else:
            info = storage.last_saved(name)
            when = (f" · senast <span style=\"font-family:'JetBrains Mono',monospace;\">{info['when'][11:16]}</span> UTC"
                    if info else "")
            st.markdown(
                f"<div style='color:{MUTED};font-size:13px;line-height:1.3;'>"
                f"Inga osparade ändringar{when}</div>",
                unsafe_allow_html=True)

    if ok_msg:
        st.success(ok_msg)
        st.toast("Sparat", icon="💾")
    if fail_msg:
        st.error(fail_msg)
    return saved


def footer() -> None:
    """Sidfotsraden: senast sparad, och vad som ligger osparat."""
    dirty = storage.dirty_stores()
    saved = storage.saved_stores()
    if not dirty and not saved:
        return

    parts = []
    for store, info in saved:
        parts.append(f"{store} {info['when'][11:16]} UTC ({info['sha']})")
    line = " · ".join(parts) if parts else "inget sparat i den här sessionen"

    st.markdown("<hr style='border-color:rgba(138,133,120,0.2);margin:18px 0 6px;'>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{DIM};font-size:0.72rem;'>Senast sparad: {line}</div>",
        unsafe_allow_html=True)
    if dirty:
        st.markdown(
            f"<div style='color:{AMBER};font-size:0.75rem;'>⚠️ Osparade "
            f"ändringar i: {', '.join(dirty)} — spara innan du lämnar "
            f"panelen.</div>", unsafe_allow_html=True)


def diagnostics() -> None:
    """Konfigurationskoll — visas i en expander så den går att felsöka från appen."""
    with st.expander("🔌 Lagring — status och konfiguration", expanded=False):
        for level, text in storage.check_config():
            if level == "error":
                st.error(text)
            elif level == "warning":
                st.warning(text)
            else:
                st.caption(text)
        st.caption("Data ligger i data/<flik>.json i repot. Varje sparning är "
                   "en commit, så historiken visar exakt vad som ändrades och "
                   "när.")
