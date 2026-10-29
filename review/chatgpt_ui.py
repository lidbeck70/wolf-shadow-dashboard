"""
review/chatgpt_ui.py — "📋 Underlag till ChatGPT" i arkens kort, i Scorecardet och i Thors Arken-flik.

Excel (med rullgardiner) och CSV att ladda ner, och prompten att klistra in med kopieringsknappen. Bara läsning:
ingenting skrivs i arken — ChatGPT:s svar granskar du och skriver in i arket, och 💾 Spara sparar som vanligt.
Excel-filen byggs först när knappen trycks (download_button med callable), så ett kort med många rader ritas inte
långsammare av att knappen finns.
"""

from __future__ import annotations

import copy
from typing import Optional

import streamlit as st

_LOAD = object()          # standard: läs sifferuppdateringen (samma cachade blob som arkens förslag)
TITLE = "📋 Underlag till ChatGPT"


def session_stores() -> dict:
    """Arken ur sessionen (samma objekt som flikarna skriver till) — bara läsning."""
    import review_link
    import storage
    defaults = dict(review_link.STORE_DEFAULTS)
    defaults.update({"scorecard": {"cards": {}}, "confidence": {"companies": {}, "commodity_overrides": {}}})
    out = {}
    for name, default in defaults.items():
        try:
            out[name] = storage.session_load(name, copy.deepcopy(default))
        except Exception:
            out[name] = None
    return out


def _blob() -> Optional[dict]:
    try:
        import refresh_ui
        b = refresh_ui.load_refresh()
        return b if isinstance(b, dict) else {}
    except Exception:
        return None


def render_export(ticker: str, stores: Optional[dict] = None, sheets=None, key: str = "", name: str = "",
                  strategy: str = "", dom_text: str = "", stage: Optional[str] = None, blob=_LOAD,
                  rows: Optional[dict] = None, entry: Optional[dict] = None) -> None:
    """Expandern med Excel, CSV och prompten för aktien. sheets=None tar arken aktien finns i; rows={ark: rad}
    låser kortets egen rad (Scorecardet: rows['scorecard'] är kortet och entry dess kandidatrad).

    blob=None hoppar över sifferuppdateringen (Scorecardet och Thor, som inte läser den annars): då står tomma
    Börsdata-fält kvar som Börsdata — förslaget med "Använd" finns i arket.
    """
    from review import export as ex
    try:
        exp = ex.build(ticker, stores if stores is not None else session_stores(), sheets, stage,
                       _blob() if blob is _LOAD else blob, name, rows, entry)
    except Exception as exc:                                # underlaget får aldrig fälla kortet
        st.caption(f"Underlaget till ChatGPT kunde inte byggas: {exc}")
        return
    if not exp.rows:
        return
    out = exp.rows
    xlsx_name = ex.filename(exp.ticker, "xlsx")
    prompt = ex.build_prompt(exp.ticker, exp.name, out, strategy, dom_text, exp.missing, xlsx_name)
    need = sum(1 for r in out if r["required"])
    k = key or exp.ticker
    with st.expander(TITLE, expanded=False):
        st.caption(f"{need} fält saknas för köpbeslut · {exp.filled}/{len(out)} fält ifyllda. Ladda ner arket, "
                   f"bifoga det i ChatGPT och klistra in texten nedan. ChatGPT fyller bara i fakta och förslag med "
                   f"källa — du granskar och skriver in i arket, och 💾 Spara sparar som vanligt.")
        if exp.absent:
            from review import catalog as cat
            st.caption("Finns inte i " + ", ".join(cat.SHEETS[s].label for s in exp.absent)
                       + " än — de raderna är en tom mall. Lägg in bolaget i arket för att kunna skriva in svaren.")
        c1, c2 = st.columns(2)
        if ex.xlsx_available():
            c1.download_button("⬇ Excel (.xlsx)", data=lambda: ex.to_xlsx(out, prompt) or b"", file_name=xlsx_name,
                               mime=ex.XLSX_MIME, key=f"rv_xlsx_{k}", on_click="ignore",
                               help="Blå kolumner fylls i, rullgardiner för Ja/Nej, val och 0–2. Fliken "
                                    "Instruktion har prompten.")
        else:
            c1.caption("Excel kräver paketet openpyxl — CSV fungerar alltid.")
        c2.download_button("⬇ CSV", data=ex.to_csv(out), file_name=ex.filename(exp.ticker, "csv"), mime="text/csv",
                           key=f"rv_csv_{k}", on_click="ignore",
                           help="Semikolonseparerad, samma kolumner som Excel-arket.")
        st.code(prompt, language=None)
