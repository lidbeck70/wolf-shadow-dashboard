"""
ui/theme.py
===========
Global branding and UI helpers for Nordic Alpha Systems.

This module is the single source of truth for colours, typography, and
reusable layout primitives.  Import from here instead of ui.css so that
a palette change in PALETTE propagates everywhere automatically.

Public API
----------
    PALETTE          — colour dict
    inject_css()     — injects the full CYBERPUNK_CSS stylesheet
    render_header()  — branded page header (wolf icon + tagline)
    render_footer()  — slim branded footer
    section_title()  — gold uppercase section label
    card()           — styled content card rendered via st.markdown
"""

from __future__ import annotations

import os
import base64
from typing import Optional

import streamlit as st

# Re-export the CSS injector and legacy helpers so callers can use
# `from ui.theme import inject_css, section_title` without touching ui.css.
from ui.css import inject_css, color_score, color_entry, tab_not_found  # noqa: F401

# ---------------------------------------------------------------------------
# Palette — single source of truth
# ---------------------------------------------------------------------------

PALETTE: dict = {
    # Backgrounds
    "bg":           "#0c0c12",
    "bg2":          "#14141e",
    "bg3":          "#1a1a28",
    "surface":      "#10101a",

    # Brand gold
    "gold":         "#c9a84c",
    "gold_dim":     "#8b7340",
    "gold_muted":   "rgba(201,168,76,0.40)",
    "gold_faint":   "rgba(201,168,76,0.12)",

    # Ice-blue / silver accents
    "silver":       "#b8c4d0",
    "ice_blue":     "#a0b4c8",
    "ice_faint":    "rgba(160,180,200,0.15)",

    # Semantic
    "green":        "#2d8a4e",
    "red":          "#c44545",
    "amber":        "#d4943a",
    "text":         "#e8e4dc",
    "text_dim":     "#8a8578",

    # Borders
    "border":       "rgba(201,168,76,0.15)",
    "border_hi":    "rgba(201,168,76,0.35)",
    "border_ice":   "rgba(160,180,200,0.20)",
}

_P = PALETTE  # shorthand used internally

# ---------------------------------------------------------------------------
# section_title — re-implemented here so it is the canonical version
# ---------------------------------------------------------------------------

def section_title(text: str, icon: str = "") -> None:
    """
    Render a gold uppercase section label with a thin underline.

    Parameters
    ----------
    text : Section label text.
    icon : Optional emoji/character prepended to the text.
    """
    prefix = f"{icon}&nbsp;&nbsp;" if icon else ""
    st.markdown(
        f'<p style="font-size:11px;letter-spacing:4px;text-transform:uppercase;'
        f'color:{_P["gold_muted"]};border-bottom:1px solid {_P["border"]};'
        f'padding-bottom:6px;margin-bottom:16px;">{prefix}{text}</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# card — styled content block
# ---------------------------------------------------------------------------

def card(
    content: str,
    title: Optional[str] = None,
    border_color: str = "",
    accent_color: Optional[str] = None,
    bg: str = "",
    padding: str = "16px 18px",
    margin_bottom: str = "12px",
) -> None:
    """
    Render an HTML content block styled as a card.

    Parameters
    ----------
    content       : Inner HTML string.
    title         : Optional card title rendered above content in gold.
    border_color  : Full border colour (defaults to palette border).
    accent_color  : If set, draws a 3-px left stripe in this colour.
    bg            : Card background (defaults to palette bg2).
    padding       : CSS padding.
    margin_bottom : Space below the card.
    """
    _bc = border_color or _P["border"]
    _bg = bg or _P["bg2"]
    _left = (
        f"border-left:3px solid {accent_color};"
        if accent_color else ""
    )
    _title_html = (
        f'<div style="color:{_P["gold"]};font-size:0.72rem;text-transform:uppercase;'
        f'letter-spacing:0.12em;margin-bottom:8px;font-weight:700;">{title}</div>'
        if title else ""
    )
    st.markdown(
        f'<div style="background:{_bg};border:1px solid {_bc};{_left}'
        f'border-radius:8px;padding:{padding};margin-bottom:{margin_bottom};">'
        f'{_title_html}{content}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# render_header — full branded page header
# ---------------------------------------------------------------------------

_HEADER_CSS = f"""
<style>
.nas-header {{
    background: linear-gradient(
        135deg,
        {_P["surface"]}  0%,
        #121020          50%,
        {_P["surface"]}  100%
    );
    border: 1px solid {_P["border"]};
    border-top: 3px solid {_P["gold"]};
    border-radius: 0 0 10px 10px;
    padding: 18px 32px 16px;
    margin: -1rem -1rem 1.5rem -1rem;
    position: relative;
    overflow: hidden;
}}
.nas-header::before {{
    content: "";
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(
            90deg, transparent, transparent 39px, rgba(201,168,76,0.025) 40px
        ),
        repeating-linear-gradient(
            0deg,  transparent, transparent 39px, rgba(201,168,76,0.025) 40px
        );
    pointer-events: none;
}}
.nas-header::after {{
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, {_P["gold"]}, transparent);
}}
.nas-brand {{
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    z-index: 1;
}}
.nas-wolf {{
    font-size: 2.4rem;
    line-height: 1;
    filter: drop-shadow(0 0 8px rgba(201,168,76,0.5));
}}
.nas-wordmark {{
    display: flex;
    flex-direction: column;
    gap: 2px;
}}
.nas-title {{
    font-family: 'Courier New', monospace;
    font-size: 1.5rem;
    font-weight: 900;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    background: linear-gradient(90deg, {_P["gold"]}, {_P["gold_dim"]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}}
.nas-tagline {{
    font-family: 'Courier New', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: {_P["gold_muted"]};
}}
.nas-badges {{
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
    position: relative;
    z-index: 1;
}}
.nas-badge {{
    font-family: 'Courier New', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    padding: 2px 10px;
    border-radius: 3px;
    border: 1px solid {_P["border"]};
    color: {_P["text_dim"]};
    background: rgba(255,255,255,0.03);
    white-space: nowrap;
}}
</style>
"""


def render_header() -> None:
    """
    Render the branded page header.

    Shows the banner image from assets/banner.jpg when present;
    falls back to the HTML wolf-icon header with tagline.
    """
    # Try banner image first (existing behaviour)
    try:
        _banner_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "assets", "banner.jpg"
        )
        if os.path.exists(_banner_path):
            with open(_banner_path, "rb") as _bf:
                _b64 = base64.b64encode(_bf.read()).decode()
            st.markdown(
                f"<div style='text-align:center;margin:-1rem -1rem 1rem -1rem;padding:0;'>"
                f"<img src='data:image/jpeg;base64,{_b64}' "
                f"style='width:100%;height:auto;border-radius:0 0 8px 8px;'/>"
                f"</div>",
                unsafe_allow_html=True,
            )
            return
    except Exception:
        pass

    # HTML wolf header fallback
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="nas-header">'
        '  <div class="nas-brand">'
        '    <div class="nas-wolf">🐺</div>'
        '    <div class="nas-wordmark">'
        '      <div class="nas-title">Nordic Alpha Systems</div>'
        '      <div class="nas-tagline">Born of Wolves, Made for Markets</div>'
        '    </div>'
        '    <div class="nas-badges">'
        '      <div class="nas-badge">SWING · POSITION · REGIME</div>'
        '      <div class="nas-badge">SENTIMENT · ALERTS · STRATEGY</div>'
        '    </div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# render_footer — slim branded footer
# ---------------------------------------------------------------------------

_FOOTER_HTML = (
    f'<div style="'
    f'margin-top:48px;padding:14px 24px;'
    f'border-top:1px solid {_P["border"]};'
    f'background:linear-gradient(0deg,{_P["surface"]},{_P["bg"]});'
    f'border-radius:8px 8px 0 0;'
    f'display:flex;justify-content:space-between;align-items:center;'
    f'flex-wrap:wrap;gap:8px;'
    f'">'
    # Left: wolf + tagline
    f'<div style="display:flex;align-items:center;gap:8px;">'
    f'  <span style="font-size:1rem;filter:drop-shadow(0 0 4px rgba(201,168,76,0.4));">🐺</span>'
    f'  <span style="font-family:\'Courier New\',monospace;font-size:0.62rem;'
    f'  letter-spacing:0.2em;text-transform:uppercase;color:{_P["gold_muted"]};">'
    f'  Born of Wolves, Made for Markets</span>'
    f'</div>'
    # Right: copyright
    f'<div style="font-family:\'Courier New\',monospace;font-size:0.58rem;'
    f'letter-spacing:0.1em;color:{_P["text_dim"]};">'
    f'Nordic Alpha Systems &nbsp;©&nbsp;2025'
    f'</div>'
    f'</div>'
)


def render_footer() -> None:
    """Render the slim branded footer at the bottom of the page."""
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)
