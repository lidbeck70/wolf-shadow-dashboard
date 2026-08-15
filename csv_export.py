"""
csv_export.py — reservutgången ur panelen.

Migrationsspecen §0 och §7: varje flik ska kunna exporteras till CSV, och
knapparna ska ligga kvar även när Excel pensionerats. Skälet är konkret —
offshore utan uppkoppling går panelen inte att nå, och då är en nedladdad fil
skillnaden mellan att kunna följa reglerna och att inte kunna det.

Delade komponenter, byggda en gång:
  rows_to_csv(rows, columns)  -> str
  download_button(...)        -> knapp som laddar ner det
"""

from __future__ import annotations

import csv
import io
from datetime import date

import streamlit as st


def rows_to_csv(rows: list, columns: list) -> str:
    """CSV-text ur en lista dictar.

    columns: [(nyckel, rubrik), ...] — bestämmer både urval och ordning, så
    exporten är stabil även om lagringen får nya fält.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", lineterminator="\n")
    writer.writerow([header for _key, header in columns])
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        out = []
        for key, _header in columns:
            v = row.get(key)
            if isinstance(v, bool):
                v = "Ja" if v else "Nej"
            out.append("" if v is None else v)
        writer.writerow(out)
    return buf.getvalue()


def filename(stem: str, day: date | None = None) -> str:
    d = day or date.today()
    return f"{stem}_{d.isoformat()}.csv"


def download_button(rows: list, columns: list, stem: str, label: str = "⬇ CSV",
                    key: str | None = None) -> None:
    """Nedladdningsknapp för en flik. Tom lista ger en avstängd knapp."""
    data = rows_to_csv(rows, columns)
    st.download_button(
        label, data=data.encode("utf-8-sig"),   # BOM: Excel läser å/ä/ö rätt
        file_name=filename(stem), mime="text/csv",
        key=key or f"csv_{stem}", disabled=not rows,
        help="Semikolonseparerad, öppnas direkt i Excel. Reservutgången när "
             "panelen inte går att nå.")
