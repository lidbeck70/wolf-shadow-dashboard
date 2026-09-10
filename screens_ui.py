"""
screens_ui.py — "Håven — senaste körning" i granskningsarken.

Visar träffarna från screens_scan.py (Gisten: screens.json) för ett ark och
låter användaren lägga in ett bolag som arkrad med ett klick. Arket äger
raden efteråt; inget skrivs över. Rena hjälpare (screen_rows, row_to_fields)
utan Streamlit så de går att testa.
"""
from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

DIM, TEXT = "#8a8578", "#e8e4dc"


@st.cache_data(ttl=600, show_spinner=False)
def load_screens() -> dict:
    try:
        from gist_storage import load_blob
        d = load_blob("screens.json", None)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def screen_rows(blob: dict, key: str) -> tuple:
    """(rader, label, error, tidsstämpel) för en håv ur bloben."""
    s = ((blob or {}).get("screens") or {}).get(key) or {}
    ts = str((blob or {}).get("generated") or "")[:16].replace("T", " ")
    return (list(s.get("rows") or []), str(s.get("label") or key),
            s.get("error"), ts)


def row_to_fields(row: dict, key: str) -> dict:
    """Vad håven kan förifylla i respektive ark — bara sådant som är
    samma tal i arket (arket räknar allt annat själv)."""
    m = row.get("m") or {}
    out = {"ticker": str(row.get("ticker") or "").upper(),
           "name": str(row.get("name") or "")}
    if row.get("ins_id") is not None:
        out["ins_id"] = row["ins_id"]          # så sifferuppdateringen hittar bolaget
    if key == "rule":
        if m.get("ev_ebitda") is not None:
            out["ev_ebitda"] = float(m["ev_ebitda"])
        if m.get("nd_ebitda") is not None:
            out["nd_ebitda"] = float(m["nd_ebitda"])
    elif key in ("durrett", "tiggre") and row.get("mcap_musd") is not None:
        out["mcap"] = float(row["mcap_musd"])
    elif key == "royalty" and m.get("ev_ebitda") is not None:
        out["ev_now"] = float(m["ev_ebitda"])
    return out


def render_screen_section(key: str, existing: set, on_add: Callable[[dict], None],
                          key_prefix: Optional[str] = None) -> None:
    """Expander med håvens träffar och en knapp per rad."""
    blob = load_screens()
    rows, label, error, ts = screen_rows(blob, key)
    kp = key_prefix or key
    title = f"🕸 Håven — {label}: {len(rows)} träffar" + (f" · {ts}" if ts else "")
    with st.expander(title, expanded=False):
        crit = ((blob.get("screens") or {}).get(key) or {}).get("criteria")
        if crit:
            st.caption(f"Körs schemalagt i Börsdata-API:t (screens_scan.py) med "
                       f"kriterierna: {crit}.")
        if error:
            st.warning(error)
            return
        if not blob:
            st.caption("Ingen körning i Gisten ännu — scheduled-scan-workflowen fyller den.")
            return
        if not rows:
            st.caption("Inga träffar just nu.")
            return
        for r in rows:
            t = str(r.get("ticker") or "").upper()
            m = r.get("m") or {}
            bits = [f"{r.get('mcap_musd'):,.0f} MUSD" if r.get("mcap_musd") else None,
                    f"skuld/EBITDA {m['nd_ebitda']:g}" if m.get("nd_ebitda") is not None else None,
                    f"EV/EBITDA {m['ev_ebitda']:g}" if m.get("ev_ebitda") is not None else None,
                    f"P/B {m['pb']:g}" if m.get("pb") is not None else None,
                    f"P/S {m['ps']:g}" if m.get("ps") is not None else None,
                    f"brutto {m['gross_margin']:g} %" if m.get("gross_margin") is not None else None,
                    f"EBIT {m['ebit_margin']:g} %" if m.get("ebit_margin") is not None else None]
            c1, c2 = st.columns([4, 1])
            c1.markdown(
                f"<span style='color:{TEXT};font-weight:700;'>{t}</span> "
                f"<span style='color:{DIM};'>{r.get('name', '')} · "
                f"{'Norden' if r.get('universe') == 'nordic' else 'Globalt'} · "
                f"{' · '.join(b for b in bits if b)}</span>"
                + (f"<br><span style='color:#d4943a;font-size:0.78rem;'>"
                   f"{' · '.join(r.get('notes') or [])}</span>" if r.get("notes") else ""),
                unsafe_allow_html=True)
            if t in existing:
                c2.caption("I arket")
            elif c2.button("➕ Lägg in", key=f"scr_add_{kp}_{t}"):
                on_add(row_to_fields(r, key))
                st.rerun()
