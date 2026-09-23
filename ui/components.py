"""
ui/components.py — de byggstenar granskningsarken och flikarna delar.

Sex rubrikmönster, fem identiska _badge-kopior, sju "verdiktrutor" med samma
CSS och tolv raderaknappar utan bekräftelse — var och en kopierad mellan
filer. Här finns EN version av varje. Alla returnerar HTML (str) utom de
som ritar widgets (page_header, confirm_delete), så de går att sätta ihop.
"""

from __future__ import annotations

from html import escape
from typing import Optional

import streamlit as st

from ui.tokens import ACCENT, BG_ALT, BG_CARD, BORDER_FAINT, DIM, GOLD, TEXT


def page_header(title: str, subtitle: str = "", accent: str = GOLD,
                align: str = "center") -> None:
    """Sidhuvudet: rubriken ÄR flikens namn (så navigationen och sidan säger
    samma sak), en dämpad rad under. Ritas direkt under sparraden."""
    st.markdown(
        f"<div style='text-align:{align};padding:10px 0 4px;'>"
        f"<h2 style='color:{accent};letter-spacing:0.12em;margin:0;'>"
        f"{escape(title).upper()}</h2>"
        + (f"<p style='color:{DIM};font-size:0.78rem;margin:6px 0 0;'>{subtitle}</p>"
           if subtitle else "")
        + "</div>", unsafe_allow_html=True)


def section(label: str, color: str = ACCENT) -> str:
    """Liten versal sektionsetikett med understreck."""
    return (f"<div style='color:{color};font-size:0.7rem;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin:18px 0 8px;border-bottom:1px solid {color}33;"
            f"padding-bottom:4px;'>{escape(label)}</div>")


def badge(text: str, color: str) -> str:
    """Färgad pill: bedömning, status, band."""
    return (f"<span style='background:{color}22;color:{color};border:1px solid {color};"
            f"border-radius:4px;padding:2px 8px;font-size:0.78rem;font-weight:700;'>"
            f"{escape(str(text))}</span>")


def verdict_box(score: str, label: str, note: str = "", color: str = DIM) -> str:
    """Rutan som sammanfattar en rad: poäng, bedömning, en förklarande rad."""
    return (f"<div style='border:1px solid {color}55;background:{color}0d;"
            f"border-radius:8px;padding:10px 14px;margin:10px 0;'>"
            f"<span style='color:{color};font-weight:700;font-size:1.05rem;'>{escape(str(score))}</span>"
            f"<span style='color:{color};font-weight:700;margin-left:12px;'>{escape(str(label))}</span>"
            + (f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>{note}</div>" if note else "")
            + "</div>")


def kpi(label: str, value: str, color: str = ACCENT, caption: str = "") -> str:
    """Nyckeltalskort: etikett, stort värde, valfri bildtext."""
    return (f"<div style='background:{BG_ALT};border:1px solid {BORDER_FAINT};"
            f"border-left:3px solid {color};border-radius:8px;padding:14px 16px;'>"
            f"<div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;"
            f"color:{DIM};margin-bottom:6px;'>{escape(label)}</div>"
            f"<div style='font-size:16px;font-weight:700;color:{color};'>{escape(str(value))}</div>"
            + (f"<div style='font-size:11px;color:{DIM};margin-top:4px;'>{caption}</div>" if caption else "")
            + "</div>")


def card(title: str, body: str, color: str = ACCENT) -> str:
    """Navigations-/infokort: rubrik och en rad text."""
    return (f"<div style='background:{BG_ALT};border:1px solid {BORDER_FAINT};"
            f"border-left:2px solid {color};border-radius:8px;padding:14px;'>"
            f"<div style='font-size:13px;font-weight:700;color:{TEXT};margin-bottom:4px;'>{escape(title)}</div>"
            f"<div style='font-size:11px;color:{DIM};line-height:1.5;'>{body}</div></div>")


def confirm_delete(label: str, key: str, help: Optional[str] = None) -> bool:
    """Radera bakom en bekräftelse: en 🗑-popover med den riktiga knappen inuti.

    Returnerar True när användaren bekräftat. Ett klick på fel rad ska inte
    kosta ett bolag med tio ifyllda fält."""
    with st.popover(f"🗑 {label}", help=help or "Öppnar en bekräftelse — inget raderas direkt."):
        st.caption("Raderingen går inte att ångra i sessionen.")
        return st.button(f"Ja, {label.lower()}", key=f"{key}__confirm", type="primary")


def empty_state(text: str) -> None:
    """Tom lista, sagt vänligt och centrerat."""
    st.markdown(
        f"<div style='background:{BG_CARD};border:1px solid {BORDER_FAINT};border-radius:8px;"
        f"padding:16px;color:{DIM};font-size:0.78rem;text-align:center;'>{text}</div>",
        unsafe_allow_html=True)
