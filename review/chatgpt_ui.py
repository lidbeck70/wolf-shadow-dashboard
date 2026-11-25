"""
review/chatgpt_ui.py — "📋 Underlag till ChatGPT" i arkens kort, i Scorecardet och i Thors Arken-flik.

Ut: Excel (med rullgardiner) och CSV att ladda ner, och prompten att klistra in med kopieringsknappen. Filerna och
prompten byggs först när de behövs — Excel och CSV när knappen trycks (download_button med callable), prompten när
"Visa prompten" slås på — så ett stängt kort skickar ingen stor text eller fil till webbläsaren vid varje ändring.

In (render_import, samma expander): det ifyllda arket eller det inklistrade svaret läses (review.answers),
kontrolleras (review.importer) och visas som en skillnadstabell med "Använd" per rad — förkryssad för säkerhet
high/medium med källa, utan varning och utan att skriva över ett värde som redan står i arket. Varje rad går till
det ark Fältnyckeln pekar på (review.routing): kortets eget ark och alla andra ark bolaget finns i, så ett svar på
Thors underlag (Rick Rule + Insider + Scorecard …) läses in i ett enda kort. "Skriv in N ändringar" skriver i
arkens rader i sessionen (on_click; arkens widgets släpps så att de ritas om med de nya värdena) och lägger ett
kvitto: inskrivet per ark, det som fortfarande saknas för köpbeslut och 💾 för alla berörda ark — inget sparas
till GitHub förrän du trycker där eller på arkets sida. Thors Arken-flik har samma import, utan egna mål: där går
varje rad till arket den hör till, och kvittot räknar som Thors underlag (köpgrindens position).

Svar för fel bolag (rader med annan ticker): EN ruta per bolag överst i importen i stället för en rad per avvisad
rad — vad som hände och de två vanliga orsakerna (fel kort, eller ChatGPT svarade om fel bolag). Finns bolaget i ett
ark svaret pekar på: "Öppna XXX och läs in där" (on_click, aldrig när sidan öppnas) lägger svaret i CARRY och sätter
arkens fokus på XXX (review.focus, låst på radens former; Wolf Asymmetry byter 'Bolag'). XXX-kortet läser då in
det flyttade svaret med samma granskning och samma "Skriv in" — inget skrivs förrän där, och inget sparas till
GitHub. Syns kortet du läste in i fortfarande, säger rutan var svaret väntar och hur du öppnar kortet (ingen knapp).
Utan registrets tickerkarta flyttas inget till en bar rad som kan vara ett annat bolag ('TEL' för TEL.OL).
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import streamlit as st

_LOAD = object()          # standard: läs sifferuppdateringen (samma cachade blob som arkens förslag)
TITLE = "📋 Underlag till ChatGPT"
IMPORT_TITLE = "Läs tillbaka svaret"
CARRY = "rv_imp_carry"    # svaret som följer med till rätt bolags kort: {ticker, data, file, text, from_ticker, …}
MAX_OTHER = 3             # högst så många rutor för andra bolag — resten på en rad '+N'


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


RETRY_S = 60     # ett misslyckat läsförsök av registret prövas igen tidigast efter så här många sekunder


def registry_sheet_map() -> Optional[dict]:
    """Registrets tickerkarta för arkens bara nordiska former ('BOL' → 'BOL.ST'; 'TEL' är TE Connectivity, inte
    Telenor) — samma som Thor bygger. Bara en karta ur ett HELT läst register gäller: saknas USA-delen tror kartan
    att 'TEL' är Telenor. Registret läses här (Thors karta återanvänds inte — den bär inget besked om att registret
    var helt läst) och sparas i sessionen först när alla filer gick att läsa. Ett misslyckat försök sparas inte — det
    prövas igen efter RETRY_S sekunder. None utan säker karta (då skriver importen inte i bara arkrader för en
    nordisk notering)."""
    import time
    hit = st.session_state.get("rv_sheet_map")
    if isinstance(hit, dict) and hit:
        return dict(hit)
    if time.time() - float(st.session_state.get("rv_sheet_map_failed") or 0) < RETRY_S:
        return None
    found, bad = {}, []
    try:
        import gist_storage as gs
        from universe import registry as ur
        inner = ur.default_loader()                     # Gisten (testerna byter ut den)

        def _load(name, fallback):
            v = inner(name, fallback)
            if name in gs.UNREADABLE_NAMES:             # just den här läsningen fallerade
                bad.append(name)
            return v
        reg_ = ur.load(None, _load)
        found = reg_.sheet_map() if reg_ is not None and not bad else {}
    except Exception:
        found = {}
    if not found:
        st.session_state["rv_sheet_map_failed"] = time.time()
        return None
    st.session_state["rv_sheet_map"] = dict(found)
    return dict(found)


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
                  targets: Optional[dict] = None, reset: tuple = (), on_apply: Optional[Callable] = None,
                  sheet_map: Optional[dict] = None, scan: Optional[dict] = None) -> None:
    """Expandern med Excel, CSV och prompten för aktien — och importen av svaret. sheets=None tar arken aktien
    finns i; rows={ark: rad} låser kortets egen rad (Scorecardet: rows['scorecard'] är kortet och entry dess
    kandidatrad). sheet_map = registrets karta för arkens bara nordiska former (Thor: samma rader som dossiern —
    'TEL' i arket är TE Connectivity, inte Telenor TEL.OL).

    blob=None hoppar över sifferuppdateringen (Scorecardet och Thor, som inte läser den annars): då står tomma
    Börsdata-fält kvar som Börsdata — förslaget med "Använd" finns i arket. scan = Insider-skanningen (Insider-
    sidan skickar den): skanningens fält står som Börsdata när den har talet för raden, annars frågas ChatGPT.

    Importen skriver i targets={ark: rad eller funktion som ger raden} (standard: rows — platta ark skickar in
    radens eget objekt); bara de arken tar emot svaret härifrån. reset=(id:n, prefix) är kortets widgetnycklar
    som släpps efter inskrivningen (standard: radernas id). importable=False (Thor) visar ingen import.
    """
    from review import export as ex
    try:
        exp = ex.build(ticker, stores if stores is not None else session_stores(), sheets, stage,
                       _blob() if blob is _LOAD else blob, name, rows, entry, sheet_map, scan)
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
    # Öppen bara medan ett flyttat svar väntar här — och medan kvittot efter det står kvar, eller ditt eget svar som
    # ersatte det (rv_imp_keep_): en expander som byter expanded=True → False fälls ihop av webbläsaren, och då skulle
    # kvittot efter "Skriv in" — eller svaret du just klistrade in — döljas.
    rec = st.session_state.get(f"rv_imp_receipt_{k}")
    opened = ((importable and _carry_for(exp.ticker) is not None) or (isinstance(rec, dict) and bool(rec.get("moved")))
              or bool(st.session_state.get(f"rv_imp_keep_{k}")))
    with st.expander(TITLE, expanded=bool(opened)):
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
            st.caption("Svaret läses tillbaka i arket — i valfritt ark bolaget finns i: varje rad hamnar i sitt "
                       "ark, så en import räcker. Knapparna nedan öppnar arken på bolaget.")


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


def _sheet_label(sheet: str) -> str:
    from review import catalog as cat
    return cat.SHEETS[sheet].label if sheet in cat.SHEETS else sheet


def _apply(changes: list, plan, k: str, gen: int, on_apply: Optional[Callable], moved: bool = False,
           others: Optional[dict] = None) -> None:
    """on_click: skriv de valda ändringarna i varje arks rad (routning.Plan — kortets egen rad eller raden i
    sessionens lager), släpp arkens widgets (de ritas om med nya värden), töm importen och lägg ett kvitto:
    vad som skrevs in per ark och vad som fortfarande saknas för köpbeslut. Bara sessionen — 💾 i kvittot (eller
    på arkets sida) skriver till GitHub. moved = svaret var flyttat hit (CARRY): det släpps när något skrevs in.
    others = {ticker: rader} för andra bolag i svaret — kvittot säger att de inte lästes in."""
    from review import importer as im
    from review import routing as rt
    sheets = list(dict.fromkeys(c.sheet for c in changes))
    live = rt.resolved_targets(plan, sheets)
    done = [c for c in changes if isinstance(live.get(c.sheet), dict)]
    paths: dict = {}
    n = im.apply(done, live, paths)
    tokens, prefixes = [], []
    for sheet in sheets:
        t, p = plan.reset.get(sheet, ((), ()))
        tokens += list(t or ())
        prefixes += list(p or ())
    for wk in im.widget_keys(list(st.session_state.keys()), tokens, prefixes):
        st.session_state.pop(wk, None)
    st.session_state[f"rv_imp_gen_{k}"] = gen + 1
    by_sheet: dict = {}
    for c in done:
        final = paths.get(id(c), c.path)
        if final is None:                               # en tom katalysatorplats som togs bort
            continue
        by_sheet.setdefault(c.sheet, []).append((_final_label(c, final), _fmt(c.new)))
    stores = None
    try:
        stores = session_stores()
    except Exception:
        stores = None
    missing = rt.missing_now(plan.exp, {s: live.get(s) for s in by_sheet}, stores, plan.own)
    st.session_state[f"rv_imp_receipt_{k}"] = {
        "n": n, "sheets": [(_sheet_label(s), rows) for s, rows in by_sheet.items()],
        "missing": [(_sheet_label(s), names) for s, names in missing.items()],
        "stores": list(dict.fromkeys(plan.store_of(s) for s in by_sheet if plan.store_of(s))),
        "skipped": [_sheet_label(s) for s in sheets if s not in by_sheet],
        "moved": bool(moved or st.session_state.pop(f"rv_imp_keep_{k}", False)),  # underlaget öppet med kvittot
        "others": [(t, int(m)) for t, m in (others or {}).items()],                # rader för andra bolag, ej inlästa
    }
    st.session_state.pop(f"rv_imp_saved_{k}", None)
    if moved and n:
        _drop_carry()                                   # det flyttade svaret är inläst — det släpps
    if on_apply:
        on_apply()


def _final_label(change, final: str) -> str:
    """Fältets namn där värdet hamnade: en Tiggre-katalysator som flyttades upp i listan (svaret skrev plats 2 i en
    tom lista) heter efter sin plats nu, inte svarets."""
    import re
    m = re.match(r"^catalysts\.(\d+)\.", final or "")
    if change.sheet == "tiggre" and m and final != change.path:
        return re.sub(r"Katalysator \d+", f"Katalysator {int(m.group(1)) + 1}", change.field.label)
    return change.field.label


def _save_stores(stores: list, k: str) -> None:
    """on_click: 💾 för arken som fick svar — samma sparning som sparraden på varje arks sida."""
    import storage
    done, failed = [], []
    for name in stores:
        try:
            if storage.is_dirty(name):
                storage.save_session(name)
            done.append(name)
        except Exception as exc:                        # storage.StorageError och nätfel — visas, sväljs inte
            failed.append(f"{name}: {exc}")
    st.session_state[f"rv_imp_saved_{k}"] = (done, failed)


def _close_receipt(k: str) -> None:
    st.session_state.pop(f"rv_imp_receipt_{k}", None)
    st.session_state.pop(f"rv_imp_saved_{k}", None)
    st.session_state.pop(f"rv_imp_keep_{k}", None)


def _receipt(k: str) -> None:
    """Kvittot efter "Skriv in": fälten per ark, det som fortfarande saknas för köpbeslut, och 💾 för alla berörda
    lager på en gång."""
    from review import routing as rt
    r = st.session_state.get(f"rv_imp_receipt_{k}")
    if not isinstance(r, dict):
        return
    import storage
    n = int(r.get("n") or 0)
    names = [lbl for lbl, _rows in r.get("sheets") or []]
    where = names[0] if len(names) == 1 else (", ".join(names[:-1]) + " och " + names[-1] if names else "arket")
    st.success((f"{n} ändring inskriven" if n == 1 else f"{n} ändringar inskrivna") + f" i {where}.")
    for lbl, rows in r.get("sheets") or []:
        st.caption(f"**{lbl}** · {len(rows)} inskrivna: " + " · ".join(f"{f} = {v}" for f, v in rows))
    for lbl in r.get("skipped") or []:
        st.warning(f"{lbl}: raden hittades inte när svaret skulle skrivas in — inget ändrat där.")
    for t, m in r.get("others") or []:
        st.caption(f"{_rows(m)} för {t} lästes inte in här — gäller {'den' if m == 1 else 'de'} {t}: läs in samma "
                   f"svar i {t}-kortet.")
    miss = [(lbl, m) for lbl, m in r.get("missing") or [] if m]
    if miss:
        st.caption("Saknas fortfarande för köpbeslut — " + " · ".join(
            f"**{lbl}** {len(m)}: {', '.join(m[:8])}{' …' if len(m) > 8 else ''}" for lbl, m in miss))
    elif names:
        st.caption("Inget saknas längre för köpbeslut i " + where + ".")
    stores = [s for s in r.get("stores") or [] if s]
    saved = st.session_state.get(f"rv_imp_saved_{k}")
    if saved:
        done, failed = saved
        if done:
            st.success("Sparat: " + ", ".join(rt.STORE_LABEL.get(s, s) for s in done) + ".")
        for line in failed:
            st.error(f"SPARNINGEN MISSLYCKADES — {line}. Ändringarna finns kvar i sessionen men är INTE sparade.")
    dirty = [s for s in stores if storage.is_dirty(s)]
    c1, c2 = st.columns(2)
    if dirty:
        c1.button("💾 Spara " + ", ".join(rt.STORE_LABEL.get(s, s) for s in dirty), key=f"rv_imp_save_{k}",
                  type="primary", on_click=_save_stores, args=(dirty, k),
                  help="Sparar arken som fick svar till GitHub — samma som 💾 Spara på varje arks sida.")
    c2.button("Stäng kvittot", key=f"rv_imp_close_{k}", on_click=_close_receipt, args=(k,))


def _list(title: str, items: list, warn: bool) -> None:
    if not items:
        return
    lines = [f"rad {line} · {key or '–'} — {text}" for line, key, text in items[:12]]
    more = f"\n\n… och {len(items) - 12} till" if len(items) > 12 else ""
    (st.warning if warn else st.caption)(f"{title}\n\n" + "\n\n".join(lines) + more)


def _rows(n: int) -> str:
    return f"{n} rad" if n == 1 else f"{n} rader"


def _summary(res, n_answers: int, n_other: int = 0, issues: Optional[list] = None) -> str:
    """En rad om vart svaret tar vägen: ändringar per ark, det som redan stod, tomma, annan ticker (rutan ovan),
    avvisade och överhoppade. issues = de avvisade utom annan ticker (standard: alla)."""
    issues = res.issues if issues is None else issues
    per: dict = {}
    for c in res.changes:
        per[c.sheet] = per.get(c.sheet, 0) + 1
    parts = [f"{_sheet_label(s)} {n}" for s, n in per.items()]
    head = (f"{len(res.changes)} att skriva in → " + " · ".join(parts)) if parts else "Inget nytt att skriva in"
    tail = [f"{res.same} stod redan i arken" if res.same else "",
            f"{len(res.blank)} fält ChatGPT skulle fylla står tomma" if res.blank else "",
            f"{n_other} med annan ticker" if n_other else "",
            f"{len(issues)} kunde inte läsas in" if issues else "",
            f"{len(res.ignored)} hoppades över" if res.ignored else ""]
    return f"{_rows(n_answers)} i svaret. " + head + "".join(f" · {t}" for t in tail if t) + "."


# ── Svar för ett annat bolag: rutan och flytten ──────────────────────────────
def carried(ticker: str, sheet_map: Optional[dict] = None) -> Optional[dict]:
    """Svaret som följer med till det här bolagets kort (CARRY), eller None. Exakt samma ticker, eller samma notering
    i arkformerna (routing.same_listing): 'BOL.ST' och 'BOL' med registrets karta — utan kartan läser en bar 'TEL'
    aldrig ett svar som flyttats till TEL.OL (det kan vara ett annat bolag)."""
    c = st.session_state.get(CARRY)
    if not isinstance(c, dict) or not str(c.get("ticker") or "").strip():
        return None
    from review import routing as rt
    return c if rt.same_listing(c.get("ticker"), ticker, sheet_map) else None


def _carry_for(ticker: str) -> Optional[dict]:
    """carried med registrets karta — kartan läses bara när ett flyttat svar väntar någonstans."""
    if not isinstance(st.session_state.get(CARRY), dict):
        return None
    return carried(ticker, registry_sheet_map())


def _drop_carry() -> None:
    st.session_state.pop(CARRY, None)


def _typed(wkey: str, ticker: str, k: str = "") -> None:
    """on_change: du laddade upp eller klistrade in något eget i kortet — ett svar som flyttats hit släpps, ditt
    gäller, och underlaget står kvar öppet (rv_imp_keep_) så att ditt svar syns. Att tömma rutan eller ta bort
    filen släpper det inte."""
    v = st.session_state.get(wkey)
    if v is None or (isinstance(v, str) and not v.strip()):
        return
    if _carry_for(ticker) is not None:
        _drop_carry()
        if k:
            st.session_state[f"rv_imp_keep_{k}"] = True


def _move(carry: dict, sheet_map: Optional[dict] = None) -> None:
    """on_click (aldrig när sidan öppnas): svaret följer med till bolagets kort (CARRY) och arken öppnar på bolaget
    (review.focus). Inget skrivs i något ark — det gör bara "Skriv in" i det kortet.

    Fokus låses på radens former, som Thors arkknappar: utan registrets karta bara radens egen form — annars tar
    fokusväljaren 'TEL' (sorteras först) när svaret gäller TEL.OL. Wolf Asymmetry har ingen fokusväljare: där byts
    bolaget i 'Bolag' (asymmetry.ui:s asym_goto) när svaret står i Durrett/Asymmetry-raden."""
    from review import catalog as cat
    from review import focus as fc
    from review import importer as im
    t = str(carry.get("ticker") or "").strip().upper()
    st.session_state[CARRY] = dict(carry)
    fc.set_focus(t)
    st.session_state[fc.FORMS] = {"ticker": t, "forms": sorted(im._forms(t, sheet_map)) if sheet_map else [t]}
    if carry.get("sheet") == "confidence" and str(carry.get("from_key") or "").startswith("asym_"):
        comps = (st.session_state.get(cat.SHEETS["confidence"].store) or {}).get("companies")
        key = next((x for x in comps if str(x).strip().upper() == t), None) if isinstance(comps, dict) else None
        if key:
            st.session_state["asym_goto"] = key         # väljaren 'Bolag' öppnar på bolaget vid nästa körning


def _read_carry(carry: dict) -> list:
    """Det flyttade svaret som svarsrader — filen (samma byte och filnamn) eller den inklistrade texten."""
    from review import importer as im
    if carry.get("data") is not None:
        return im.read_file(bytes(carry["data"]), str(carry.get("file") or ""))
    return im.read_text(str(carry.get("text") or ""))


def _carry_note(carry: dict, k: str, own: bool, here: str = "") -> None:
    """Raden i kortet som läser ett flyttat svar: varifrån det kom, vilket kort som nu är öppet, och ✕ Släpp."""
    src = str(carry.get("from_ticker") or "").strip() or "ett annat"
    st.caption(f"Svaret du läste in i {src}-kortet — inläst här."
               + (" Filen eller texten ovan används inte medan det ligger här." if own else "")
               + (f" Det här är {here}-kortet — Skriv in skriver hit." if here else ""))
    st.button("✕ Släpp", key=f"rv_imp_drop_{k}", on_click=_drop_carry,
              help="Släpper det flyttade svaret härifrån — inget i arken ändras.")


def _base(ticker: str) -> str:
    """Tickern utan börssuffix ('ABC.TO' → 'ABC') — samma symbol på en annan notering."""
    return str(ticker or "").strip().upper().split(".")[0]


def _home_label(sheet: str, own) -> str:
    """Arket bolagets kort står i, som rutan säger det: Wolf Asymmetry delar Durretts rad — står du där heter det så."""
    return _sheet_label("asymmetry" if sheet == "confidence" and "asymmetry" in own else sheet)


def _next_step(row: str, sheet: str, own, where: str) -> str:
    """Vad du gör när svaret väntar i ett annat kort som inte syns: välj bolaget här i arket (Wolf Asymmetry:
    'Bolag', övriga ark: Fokus) eller öppna arket det står i. own = kortets egna ark (tomt i Thors Arken-flik)."""
    from review import catalog as cat
    from review import focus as fc
    same = sheet in own or (sheet == "confidence" and bool(set(own) & set(cat.CONFIDENCE_SHEETS)))
    if same and "asymmetry" in own:
        return f"Välj {row} under Bolag överst på sidan."
    if same:
        return f"Välj {row} i Fokus överst i arket."
    if fc.focus_ticker() == row:
        return f"Öppna {where} i ARKEN — fokus står redan på {row}."
    return f"Öppna {where} i ARKEN och välj {row} i Fokus överst."


def _other_boxes(others: dict, answers: list, plan, stores: Optional[dict], smap: Optional[dict], k: str,
                 here: str, source: dict, own=frozenset()) -> None:
    """En ruta per annat bolag i svaret (högst MAX_OTHER, sedan '+N'): vad som hände och de två vanliga orsakerna.
    Finns bolaget i ett ark svaret pekar på: "Öppna XXX och läs in där" (_move). När svaret redan väntar där syns
    ingen knapp — rutan säger var det väntar och hur du öppnar kortet (_next_step). Kan en rad vara bolaget men inte
    säkert (utan registrets karta): ingen knapp, rutan säger det. Annars: lägg in bolaget först. Ingenting skrivs
    här. own = kortets egna ark (tomt i Thors Arken-flik)."""
    from review import importer as im
    from review import routing as rt
    if not others:
        return
    mixed = sum(others.values()) < len(answers)            # svaret har också rader för det här kortet
    part = mixed or len(others) > 1                         # svaret gäller fler än ett bolag
    waiting = st.session_state.get(CARRY)
    items = list(others.items())
    for i, (t, n) in enumerate(items[:MAX_OTHER]):
        sheets = im.answer_sheets(answers, plan.exp, t) or list(getattr(plan.exp, "sheets", None) or [])
        home = rt.find_company(t, sheets, stores, smap)
        unsure = None if home else rt.unsure_company(t, sheets, stores, smap)
        where = _home_label(home[0], own) if home else ""
        sent = bool(home and isinstance(waiting, dict) and waiting.get("from_key") == k
                    and rt.same_listing(waiting.get("ticker"), home[1], smap))
        head = (f"En del av svaret gäller {t}" if part else f"Svaret gäller {t}") + f" — det här kortet är {here}."
        body = (f"{_rows(n)} har annan ticker ({t}) och lästes inte in — inget skrivs in i {here} från "
                f"{'den' if n == 1 else 'dem'}.")
        if sent:
            fix = (f"**Svaret väntar i {home[1]}-kortet ({where}).** {_next_step(home[1], home[0], own, where)} "
                   f"Inget skrivs in förrän du tryckt Skriv in där.")
        elif home:
            fix = (f"Läste du in i fel kort? Tryck **Öppna {t} och läs in där** — svaret följer med till "
                   f"{home[1]}-kortet i {where}, och inget skrivs in förrän du tryckt Skriv in där.")
        elif unsure:
            fix = (f"Läste du in i fel kort? {_home_label(unsure[0], own)} har raden {unsure[1]}, men registrets "
                   f"tickerkarta kunde inte läsas — det kan vara ett annat bolag än {t}, så svaret flyttas inte dit. "
                   f"Är det samma bolag: läs in svaret i {unsure[1]}-kortet själv.")
        elif _base(t) == _base(here):
            fix = (f"ChatGPT skrev {t} i stället för {here} — en annan notering, så panelen läser det som ett annat "
                   f"bolag. Gäller svaret {here}: ändra Ticker till {here} i svaret (eller be ChatGPT behålla "
                   f"tickern exakt) och läs in igen.")
        else:
            fix = (f"Läste du in i fel kort? {t} finns inte i arken än — lägg in bolaget först och läs sedan in "
                   f"svaret i dess kort.")
        gpt = f"Svarade ChatGPT om fel bolag (t.ex. i en gammal chatt)? Be om ett nytt svar för {here} i en ny chatt."
        st.warning(f"**{head}**\n\n{body}\n\n{fix}\n\n{gpt}")
        if not home or sent:                            # efter trycket: ingen knapp som ser ut att inte göra något
            continue
        st.button(f"Öppna {t} och läs in där", key=f"rv_imp_move_{k}" + (f"_{i}" if i else ""),
                  type="secondary" if mixed else "primary", on_click=_move,
                  args=({"ticker": home[1], "sheet": home[0], "from_ticker": here, "from_key": k, **source}, smap),
                  help=f"Svaret följer med till {home[1]}-kortet och arken öppnar på {home[1]}. Där granskar du "
                       f"skillnaden som vanligt — inget skrivs in förrän du trycker Skriv in.")
    if len(items) > MAX_OTHER:
        more = items[MAX_OTHER:]
        st.caption(f"+{len(more)} bolag till i svaret, inte heller inlästa: "
                   + " · ".join(f"{t} ({_rows(n)})" for t, n in more))


def render_import(exp, targets: dict, k: str, reset: tuple = (), on_apply: Optional[Callable] = None) -> None:
    """Uppladdning eller inklistring → kontroll → skillnadstabell med "Använd" → "Skriv in N ändringar".

    Varje rad går till det ark Fältnyckeln pekar på (review.routing): kortets eget ark och alla andra ark bolaget
    finns i — ett svar på Thors underlag läses in i ett enda kort.

    Rader med ett annat bolags ticker visas som en ruta per bolag (_other_boxes). Ett svar som flyttats hit från ett
    annat kort (CARRY) läses in som om det klistrats in här; laddar du upp eller klistrar in något eget släpps det."""
    import pandas as pd
    import storage
    from review import export as ex
    from review import importer as im
    from review import routing as rt
    st.markdown(f"**{IMPORT_TITLE}**")
    _receipt(k)
    gen = int(st.session_state.get(f"rv_imp_gen_{k}") or 0)
    here = str(getattr(exp, "ticker", "") or "")
    fkey, tkey = f"rv_imp_file_{k}_{gen}", f"rv_imp_txt_{k}_{gen}"
    c1, c2 = st.columns(2)
    up = c1.file_uploader("Det ifyllda arket", type=["xlsx", "csv", "txt", "json"], key=fkey,
                          on_change=_typed, args=(fkey, here, k),
                          help="Arket ChatGPT skickade tillbaka (.xlsx eller .csv), eller svaret sparat som text. "
                               "Raderna läses in i det ark Fältnyckeln pekar på — alla ark bolaget finns i.")
    txt = c2.text_area("… eller klistra in svaret", key=tkey, height=110, on_change=_typed, args=(tkey, here, k),
                       placeholder=ex.ANSWER_HEADER,
                       help="CSV-kodblocket ChatGPT svarar med — enklast på mobilen.")
    carry = _carry_for(here)
    own = up is not None or bool(str(txt or "").strip())
    if carry is None and not own:
        st.caption("Svaret läses in i alla ark bolaget finns i — Fältnyckeln avgör arket. Inget skrivs in förrän "
                   "du granskat skillnaden och tryckt Skriv in, och inget sparas till GitHub förrän du trycker 💾.")
        return
    if carry is not None:
        _carry_note(carry, k, own, here)
    try:
        if carry is not None:
            answers = _read_carry(carry)
            source = {"data": carry.get("data"), "file": carry.get("file"), "text": carry.get("text")}
        elif up is not None:
            source = {"data": up.getvalue(), "file": up.name, "text": None}
            answers = im.read_file(source["data"], up.name)
        else:
            source = {"data": None, "file": None, "text": str(txt)}
            answers = im.read_text(txt)
    except im.ImportFormatError as exc:
        st.error(str(exc))
        return
    try:
        stores = session_stores()
    except Exception:
        stores = None
    smap = registry_sheet_map() if stores is not None else None        # bara en karta ur ett helt läst register
    view = None if targets else set(getattr(exp, "sheets", None) or [])  # Thors Arken-flik: de exporterade arken
    plan = rt.plan(exp, targets, reset, stores, storage.load_failed, smap, view)
    res = im.check(answers, plan.exp, plan.writable, plan.blocked)
    others = im.other_tickers(answers, plan.exp)                       # {ticker: rader} — en ruta per bolag
    rest = im.split_issues(res.issues)[1] if others else list(res.issues)
    _other_boxes(others, answers, plan, stores, smap, k, here, source, set(targets or {}))
    st.caption(_summary(res, len(answers), sum(others.values()), rest))
    _list(f"{_rows(len(rest))} kunde inte läsas in:", rest, warn=True)
    _list(f"{_rows(len(res.ignored))} hoppades över:", res.ignored, warn=False)
    if res.blank:
        per: dict = {}
        for f in res.blank:
            per.setdefault(_sheet_label(f.sheet), []).append(f.label)
        with st.expander(f"{len(res.blank)} fält ChatGPT skulle fylla står fortfarande tomma", expanded=False):
            for lbl, names in per.items():
                st.caption(f"**{lbl}** · {len(names)}: " + " · ".join(names))
    if not res.changes:
        if not others or sum(others.values()) < len(answers):          # bara annat bolag: rutan ovan säger allt
            st.info("Inga ändringar att skriva in — svaret stämmer med arken eller innehöll inga godkända värden.")
        return
    many = len({c.sheet for c in res.changes}) > 1
    table = pd.DataFrame([{
        "Använd": c.default_use, **({"Ark": _sheet_label(c.sheet)} if many else {}), "Fält": c.field.label,
        "Nu": _fmt(c.old), "Nytt": _fmt(c.new), "Enhet": c.unit or c.field.unit,
        "Källa": im.provenance(c).removeprefix("ChatGPT: "), "Säkerhet": c.confidence or "–",
        "Varning": " · ".join(c.warnings), "Fältnyckel": c.field.key,
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
              type="primary", disabled=not picked, on_click=_apply,
              args=(picked, plan, k, gen, on_apply, carry is not None, others))
    st.caption("Skriver i arken i sessionen — kvittot visar vad som hamnade var och har 💾 för alla berörda ark. "
               "Varningar om enhet eller valuta räknas aldrig om: rätta talet i arket om enheten är fel.")
