"""
ai/document.py — dokumentet till text med sidmarkeringar.

PDF via pypdf (lazy import: panelen startar utan paketet, då går bara
klistra-in-vägen). Varje sida inleds med "[Sida N]" så att modellen kan
ange sida per fynd och användaren kan slå upp citatet.
"""

from __future__ import annotations

import re


def pdf_to_text(data: bytes) -> str:
    """Hela PDF:en som text, "[Sida N]" före varje sida. Tomma sidor hoppas."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Paketet pypdf saknas — klistra in texten i stället, "
                           "eller lägg till pypdf i requirements.txt.") from exc
    import io
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = _tidy(txt)
        if txt:
            parts.append(f"[Sida {i}]\n{txt}")
    return "\n\n".join(parts)


def text_with_pages(text: str, chars_per_page: int = 3500) -> str:
    """Inklistrad text utan sidmarkeringar: dela i 'sidor' av fast längd så
    att sidhänvisningarna ändå pekar någonstans."""
    text = _tidy(text or "")
    if "[Sida " in text:
        return text
    parts = []
    for i in range(0, max(len(text), 1), chars_per_page):
        chunk = text[i:i + chars_per_page]
        if chunk.strip():
            parts.append(f"[Sida {i // chars_per_page + 1}]\n{chunk}")
    return "\n\n".join(parts)


def _tidy(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_count(text: str) -> int:
    return len(re.findall(r"\[Sida \d+\]", text or ""))
