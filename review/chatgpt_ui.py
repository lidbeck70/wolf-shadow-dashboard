"""
review/chatgpt_ui.py — "📋 Underlag till ChatGPT" i arkens kort, i Scorecardet och i Thors Arken-flik.

Ut: Excel (med rullgardiner) och CSV att ladda ner, och prompten att klistra in med kopieringsknappen. Filerna och
prompten byggs först när de behövs — Excel och CSV när knappen trycks (download_button med callable), prompten när
"Visa prompten" slås på — så ett stängt kort skickar ingen stor text eller fil till webbläsaren vid varje ändring.

In (render_import, samma expander): det ifyllda arket eller det inklistrade svaret läses (review.answers),
kontrolleras (review.importer) och visas som en skillnadstabell med "Använd" per rad — förkryssad för säkerhet
high/medium med källa, utan varning och utan att skriva över ett värde som redan står i arket. "Skriv in N
ändringar" skriver i arkets rad i sessionen (on_click, och kortets widgets släpps så att de ritas om med de nya
värdena). Ingenting skrivs till GitHub — 💾 Spara som vanligt. Thors Arken-flik är bara läsning och har ingen import.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import streamlit as st

_LOAD = object()          # standard: läs sifferuppdateringen (samma cachade blob som arkens förslag)
TITLE = "📋 Underlag till ChatGPT"
IMPORT_TITLE = "Läs tillbaka svaret"


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
                  rows: Optional[dict] = None, entry: Optional[dict] = None, importable: bool = True,
                  targets: Optional[dict] = None, reset: tuple = (), on_apply: Optional[Callable] = None) -> None:
    """Expandern med Excel, CSV och prompten för aktien — och importen av svaret. sheets=None tar arken aktien
    finns i; rows={ark: rad} låser kortets egen rad (Scorecardet: rows['scorecard'] är kortet och entry dess
    kandidatrad).

    blob=None hoppar över sifferuppdateringen (Scorecardet och Thor, som inte läser den annars): då står tomma
    Börsdata-fält kvar som Börsdata — förslaget med "Använd" finns i arket.

    Importen skriver i targets={ark: rad eller funktion som ger raden} (standard: rows — platta ark skickar in
    radens eget objekt); bara de arken tar emot svaret härifrån. reset=(id:n, prefix) är kortets widgetnycklar
    som släpps efter inskrivningen (standard: radernas id). importable=False (Thor) visar ingen import.
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

    def _prompt() -> str:
        return ex.build_prompt(exp.ticker, exp.name, out, strategy, dom_text, exp.missing, xlsx_name)
    need = sum(1 for r in out if r["required"])
    k = key or exp.ticker
    with st.expander(TITLE, expanded=False):
        st.caption(f"{need} fält saknas för köpbeslut · {exp.filled}/{len(out)} fält ifyllda. Ladda ner arket, "
                   f"bifoga det i ChatGPT och klistra in texten nedan. ChatGPT fyller bara i fakta och förslag med "
                   f"källa — du granskar svaret nedan innan något skrivs in, och 💾 Spara sparar som vanligt.")
        if exp.absent:
            from review import catalog as cat
            st.caption("Finns inte i " + ", ".join(cat.SHEETS[s].label for s in exp.absent)
                       + " än — de raderna är en tom mall. Lägg in bolaget i arket för att kunna skriva in svaren.")
        c1, c2 = st.columns(2)
        if ex.xlsx_available():
            c1.download_button("⬇ Excel (.xlsx)", data=lambda: ex.to_xlsx(out, _prompt()) or b"", file_name=xlsx_name,
                               mime=ex.XLSX_MIME, key=f"rv_xlsx_{k}", on_click="ignore",
                               help="Blå kolumner fylls i, rullgardiner för Ja/Nej, val och 0–2. Fliken "
                                    "Instruktion har prompten.")
        else:
            c1.caption("Excel kräver paketet openpyxl — CSV fungerar alltid.")
        c2.download_button("⬇ CSV", data=lambda: ex.to_csv(out), file_name=ex.filename(exp.ticker, "csv"),
                           mime="text/csv", key=f"rv_csv_{k}", on_click="ignore",
                           help="Semikolonseparerad, samma kolumner som Excel-arket.")
        if st.toggle("Visa prompten", key=f"rv_prompt_{k}",
                     help="Texten att klistra in i ChatGPT tillsammans med filen — kopieringsknappen sitter i rutans "
                          "hörn. Den står också i Excel-arkets flik Instruktion."):
            st.code(_prompt(), language=None)
        if importable:
            live = dict(targets) if targets is not None else dict(rows or {})
            if not reset:
                reset = ([str(r.get("id")) for r in (rows or {}).values() if isinstance(r, dict) and r.get("id")], ())
            render_import(exp, live, k, reset, on_apply)
        else:
            st.caption("Svaret läses tillbaka i arket — knapparna nedan öppnar det på bolaget.")


# ── Importen ─────────────────────────────────────────────────────────────────
def _fmt(v) -> str:
    """Värdet i skillnadstabellen på svenska: '1 250,5', 'Ja'/'Nej', '–' för tomt."""
    if v is None or v == "":
        return "–"
    if isinstance(v, bool):
        return "Ja" if v else "Nej"
    if isinstance(v, float):
        s = f"{v:,.6f}".rstrip("0").rstrip(".")       # högst sex decimaler, aldrig 1.2e+06
        if v and s in ("0", "-0"):
            s = f"{v:.3g}"                              # ett mycket litet tal blir inte 0
        return s.replace(",", " ").replace(".", ",")
    return str(v)


def _apply(changes: list, targets: dict, k: str, gen: int, reset: tuple, on_apply: Optional[Callable]) -> None:
    """on_click: skriv de valda ändringarna i raderna, släpp kortets widgets (de ritas om med nya värden) och töm
    importen. Bara sessionen — 💾 Spara skriver till GitHub."""
    from review import importer as im
    live = {}
    for sheet in {c.sheet for c in changes}:
        t = targets.get(sheet)
        live[sheet] = t() if callable(t) else t
    n = im.apply(changes, live)
    tokens, prefixes = (tuple(reset) + ((), ()))[:2]
    for wk in im.widget_keys(list(st.session_state.keys()), tokens, prefixes):
        st.session_state.pop(wk, None)
    st.session_state[f"rv_imp_gen_{k}"] = gen + 1
    st.session_state[f"rv_imp_msg_{k}"] = (f"{n} ändring inskriven" if n == 1 else f"{n} ändringar inskrivna") + \
        " i arket — granska kortet och spara med 💾."
    if on_apply:
        on_apply()


def _list(title: str, items: list, warn: bool) -> None:
    if not items:
        return
    lines = [f"rad {line} · {key or '–'} — {text}" for line, key, text in items[:12]]
    more = f"\n\n… och {len(items) - 12} till" if len(items) > 12 else ""
    (st.warning if warn else st.caption)(f"{title}\n\n" + "\n\n".join(lines) + more)


def render_import(exp, targets: dict, k: str, reset: tuple = (), on_apply: Optional[Callable] = None) -> None:
    """Uppladdning eller inklistring → kontroll → skillnadstabell med "Använd" → "Skriv in N ändringar"."""
    import pandas as pd
    from review import export as ex
    from review import importer as im
    writable = {s for s, t in (targets or {}).items() if t is not None}
    st.markdown(f"**{IMPORT_TITLE}**")
    msg = st.session_state.pop(f"rv_imp_msg_{k}", None)
    if msg:
        st.success(msg)
    gen = int(st.session_state.get(f"rv_imp_gen_{k}") or 0)
    c1, c2 = st.columns(2)
    up = c1.file_uploader("Det ifyllda arket", type=["xlsx", "csv", "txt", "json"], key=f"rv_imp_file_{k}_{gen}",
                          help="Arket ChatGPT skickade tillbaka (.xlsx eller .csv), eller svaret sparat som text.")
    txt = c2.text_area("… eller klistra in svaret", key=f"rv_imp_txt_{k}_{gen}", height=110,
                       placeholder=ex.ANSWER_HEADER,
                       help="CSV-kodblocket ChatGPT svarar med — enklast på mobilen.")
    if up is None and not str(txt or "").strip():
        st.caption("Inget skrivs in förrän du granskat skillnaden och tryckt Skriv in — och inget sparas till "
                   "GitHub förrän du trycker 💾 Spara.")
        return
    try:
        answers = im.read_file(up.getvalue(), up.name) if up is not None else im.read_text(txt)
    except im.ImportFormatError as exc:
        st.error(str(exc))
        return
    res = im.check(answers, exp, writable)
    _list(f"{len(res.issues)} rader kunde inte läsas in:", res.issues, warn=True)
    _list(f"{len(res.ignored)} rader hoppades över:", res.ignored, warn=False)
    if res.same:
        st.caption(f"{res.same} värden stod redan i arket.")
    if not res.changes:
        st.info("Inga ändringar att skriva in — svaret stämmer med arket eller innehöll inga godkända värden.")
        return
    table = pd.DataFrame([{
        "Använd": c.default_use, "Fält": c.field.label, "Nu": _fmt(c.old), "Nytt": _fmt(c.new),
        "Enhet": c.unit or c.field.unit, "Källa": im.provenance(c).removeprefix("ChatGPT: "),
        "Säkerhet": c.confidence or "–", "Varning": " · ".join(c.warnings), "Fältnyckel": c.field.key,
    } for c in res.changes])
    edited = st.data_editor(
        table, key=f"rv_imp_tab_{k}_{gen}_{res.signature()}", hide_index=True, width="stretch",
        disabled=[c for c in table.columns if c != "Använd"],
        column_config={"Använd": st.column_config.CheckboxColumn(
            "Använd", help="Förkryssad för säkerhet high/medium med källa och utan varning. Ett värde som skriver "
                           "över det som står i arket kryssar du själv.")})
    picked = [c for c, use in zip(res.changes, list(edited["Använd"])) if bool(use)]
    n = len(picked)
    st.button(f"Skriv in {n} ändring" if n == 1 else f"Skriv in {n} ändringar", key=f"rv_imp_go_{k}",
              type="primary", disabled=not picked, on_click=_apply, args=(picked, targets, k, gen, reset, on_apply))
    st.caption("Skriver bara i arket i sessionen — granska kortet och spara med 💾. Varningar om enhet eller valuta "
               "räknas aldrig om: rätta talet i arket om enheten är fel.")
