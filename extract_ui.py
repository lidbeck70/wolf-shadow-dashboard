"""
extract_ui.py — "🤖 Läs ur presentationen" i granskningsarken.

Ladda upp en PDF eller klistra in text → ett knappstyrt modellanrop →
förslag per fält med sida och citat → "Använd" skriver in värdet i raden
(och i widgetens session-state via on_click, samma mekanik som
refresh_ui). Textfynd (jurisdiktion, kapitaldisciplin, byggmeriter …) visas
bara som läsning — kryssrutorna och poängen sätter användaren själv.

Anropet sker aldrig i renderingsvägen: Streamlit kör om skriptet vid varje
widget-interaktion, så bara knappen får kosta pengar.
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional

import streamlit as st

from ai import extract_prompt as xp
from ai import document as doc

DIM, TEXT, GREEN, AMBER = "#8a8578", "#e8e4dc", "#2d8a4e", "#d4943a"
_CONF_COLOR = {"high": GREEN, "medium": AMBER, "low": "#c44545"}


def _apply_value(row: dict, field: str, value, widget_key: Optional[str],
                 on_apply: Optional[Callable], nested: Optional[str]) -> None:
    target = row.setdefault(nested, {}) if nested else row
    target[field] = value
    if widget_key:
        st.session_state[widget_key] = value
    if on_apply:
        on_apply()


def _apply_catalysts(row: dict, cats: list, on_apply: Optional[Callable],
                     waiting_status: str) -> None:
    existing = {(c.get("name"), c.get("date")) for c in row.get("catalysts", [])}
    for c in cats:
        if (c["name"], c["date"]) in existing:
            continue
        row.setdefault("catalysts", []).append(
            {"id": uuid.uuid4().hex[:8], "name": c["name"], "date": c["date"],
             "status": waiting_status})
    if on_apply:
        on_apply()


def render_extractor(sheet: str, row: dict, widget_keys: dict,
                     on_apply: Optional[Callable] = None,
                     nested: Optional[dict] = None,
                     catalyst_status: Optional[str] = None) -> None:
    """Expander med uppladdning, knapp och förslag.

    widget_keys: {fältnyckel: widget-key} för de fält som får skrivas in.
    nested: {fältnyckel: undernyckel} när fältet ligger i en dict i raden
    (Tiggres screen-flaggor: row["screen"][key]).
    catalyst_status: Tiggres CAT_WAITING — aktiverar katalysator-knappen.
    """
    from ai import openai_client as oc

    rid = row.get("id", "x")
    skey = f"xt_{sheet}_{rid}"
    with st.expander("🤖 Läs ur presentationen (AI-förslag — inget skrivs in utan Använd)",
                     expanded=False):
        if not oc.configured():
            st.caption("OPENAI_API_KEY saknas i secrets — extraktionen är avstängd. "
                       "Arket fungerar som vanligt.")
            return
        st.caption("Ladda upp bolagspresentationen eller tekniska rapporten (PDF) eller "
                   "klistra in text. Modellen letar upp arkets fält och anger sida + citat. "
                   "Poäng, status och kryssrutor räknar och sätter du som förut.")
        c1, c2 = st.columns([1, 1])
        up = c1.file_uploader("PDF", type=["pdf"], key=f"{skey}_pdf")
        pasted = c2.text_area("… eller klistra in text", height=120, key=f"{skey}_txt",
                              placeholder="Text ur presentationen")
        if st.button("Extrahera", key=f"{skey}_go"):
            try:
                if up is not None:
                    text = doc.pdf_to_text(up.getvalue())
                elif pasted.strip():
                    text = doc.text_with_pages(pasted)
                else:
                    st.warning("Ladda upp en PDF eller klistra in text först.")
                    return
                if not text.strip():
                    st.warning("Ingen text hittades i dokumentet (skannad PDF utan textlager?).")
                    return
                prompt = xp.build_extract_prompt(sheet, str(row.get("ticker") or ""),
                                                 str(row.get("name") or ""), text)
                with st.spinner("Läser dokumentet …"):
                    reply = oc.complete(xp.SYSTEM_EXTRACT, prompt, max_output_tokens=1800,
                                        timeout=90.0, json_mode=True)
                parsed = xp.parse_extraction(reply.text)
                st.session_state[skey] = {
                    "proposals": xp.proposals(sheet, parsed),
                    "catalysts": xp.catalysts(parsed) if sheet == "tiggre" else [],
                    "notes": [str(n) for n in (parsed.get("notes") or [])][:6],
                    "pages": doc.page_count(text), "chars": len(text), "model": reply.model,
                }
            except (oc.AIError, xp.ExtractionError, RuntimeError) as exc:
                st.error(str(exc))
                return

        res = st.session_state.get(skey)
        if not res:
            return
        st.caption(f"{res['pages']} sidor · {res['chars']:,} tecken · {res['model']}")
        if not res["proposals"] and not res["catalysts"]:
            st.info("Modellen hittade inget av arkets fält i dokumentet.")
        for p in res["proposals"]:
            col = _CONF_COLOR.get(p["confidence"], DIM)
            val = (f"{p['value']:g}" if isinstance(p["value"], float)
                   else ("Ja" if p["value"] is True else "Nej" if p["value"] is False
                         else str(p["value"])))
            unit = f" {p['unit']}" if p["unit"] and p["kind"] == "number" else ""
            page = f" · sida {p['page']}" if p["page"] else ""
            a, b = st.columns([4, 1])
            a.markdown(
                f"<span style='color:{TEXT};font-weight:700;'>{p['label']}:</span> "
                f"<span style='color:{TEXT};'>{val}{unit}</span> "
                f"<span style='color:{col};font-size:0.72rem;'>{p['confidence']}</span>"
                f"<span style='color:{DIM};font-size:0.78rem;'>{page}"
                + (f" · ”{p['quote']}”" if p["quote"] else "") + "</span>",
                unsafe_allow_html=True)
            if p["apply"] and p["key"] in widget_keys:
                current = (row.get(nested.get(p["key"]) if nested else None) or {}).get(p["key"]) \
                    if nested and p["key"] in nested else row.get(p["key"])
                same = current == p["value"] or (
                    isinstance(current, (int, float)) and isinstance(p["value"], float)
                    and abs(float(current) - p["value"]) < 1e-9)
                if same:
                    b.caption("I arket")
                else:
                    b.button("Använd", key=f"{skey}_use_{p['key']}", on_click=_apply_value,
                             args=(row, p["key"], p["value"], widget_keys[p["key"]], on_apply,
                                   (nested or {}).get(p["key"])))
            else:
                b.caption("läsning")
        if res["catalysts"]:
            st.markdown("<span style='font-weight:700;color:%s;'>Katalysatorer i dokumentet</span>"
                        % TEXT, unsafe_allow_html=True)
            for c in res["catalysts"]:
                st.caption(f"• {c['name']} — {c['date']}"
                           + (f" (sida {c['page']})" if c.get("page") else ""))
            if catalyst_status:
                st.button("Lägg in katalysatorerna i kalendern", key=f"{skey}_cats",
                          on_click=_apply_catalysts,
                          args=(row, res["catalysts"], on_apply, catalyst_status))
        for n in res["notes"]:
            st.caption(f"📝 {n}")
