"""
refresh_ui.py — "🤖 Börsdata: 123,4 · Använd" bredvid ett arkfält.

Läser sheets_refresh.json (sheets_refresh.py i GitHub Actions) och visar
jobbets färska tal som FÖRSLAG. Knappen skriver in värdet i raden och i
widgetens session-state via on_click-callback (det enda tillfället då
Streamlit tillåter att en instansierad widgets state ändras) — inget
skrivs annars, och 💾 Spara sköter persistensen som vanligt.

offer() är samma knapp för förslag av vilken typ som helst (heltal, text, val,
kryss) ur andra källor än bloben — t.ex. Insider-skanningen.
"""
from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, Optional

import streamlit as st

DIM = "#8a8578"

# Bolagsformer och ord som inte skiljer bolag åt: "Boliden AB" liknar "Boliden", men "Gold" gör inte
# "Gold Fields" till "Gold Resource".
_LEGAL = frozenset("ab publ asa as oyj abp a s inc corp corporation co company ltd limited plc sa se nv ag "
                   "the group holding holdings class ser series".split())
_GENERIC = frozenset("gold silver mining mines minerals metals resources royalty royalties energy energi "
                     "international global capital".split())
NAME_RATIO_MIN = 0.8        # teckenlikhet mellan de hopskrivna namnen som räknas som samma namn


_NORDIC = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE"})     # NFKD delar inte ø/æ ('Møller' = 'Moller')


def _name_parts(name) -> list:
    s = unicodedata.normalize("NFKD", str(name or "").translate(_NORDIC))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def _name_tokens(name) -> list:
    return [t for t in _name_parts(name) if len(t) > 1 and t not in _LEGAL]


def name_resembles(own, bd_name) -> bool:
    """Liknar radens eget namn Börsdatas? Bolagsformer, accenter och skiljetecken räknas inte ('Var Energi' =
    'Vår Energi ASA'); ett namn som ryms i det andra ('Kinross' i 'Kinross Gold') räcker, liksom ett
    gemensamt ord som inte är branschord ('Gold', 'Mining'). True när något av namnen saknas — inget att jämföra."""
    a, b = _name_tokens(own), _name_tokens(bd_name)
    if not a or not b:
        return True
    ja, jb = "".join(a), "".join(b)
    if ja in jb or jb in ja:
        return True
    if {t for t in a if len(t) >= 3 and t not in _GENERIC} & {t for t in b if len(t) >= 3 and t not in _GENERIC}:
        return True
    return SequenceMatcher(None, ja, jb).ratio() >= NAME_RATIO_MIN


def matched_note(blob: dict, sheet: str, row: dict, row_id=None) -> str:
    """' (matchat: Genovis AB)' när sifferuppdateringens bolag inte liknar radens eget namn, eller raden saknar
    namn — så syns det när en manuell rad fått fel bolag. Tom sträng annars, och utan Börsdatas namn."""
    s = ((blob or {}).get("rows") or {}).get(f"{sheet}:{row_id or (row or {}).get('id')}") or {}
    bd = str(s.get("bd_name") or "").strip()
    if not bd:
        return ""
    own = str((row or {}).get("name") or "").strip()
    if _name_tokens(own):
        if name_resembles(own, bd):
            return ""
    elif own:
        # bara enbokstavsdelar ('H&M', 'B&O'): bokstäverna mot Börsdata-namnets initialer ('H&M B' → hmb,
        # 'Bang & Olufsen' → bo) — inte var som helst i namnet, då vore 'B&O' samma bolag som Boliden
        ca = "".join(t for t in _name_parts(own) if t not in _LEGAL)
        ib = "".join(t[0] for t in _name_parts(bd) if t not in _LEGAL)
        if ca and (ib.startswith(ca) or (len(ib) >= 2 and ca.startswith(ib))):
            return ""
    return f" (matchat: {html.escape(bd).replace('$', '&#36;')})"


def _registry_member(ticker):
    """Universum-registrets medlem för tickern (marknadens fil ur Gisten, sedan i sessionen — samma uppslag som
    Thor, thor.ui.universe_member). None utan register (ingen Gist) eller okänd ticker."""
    try:
        from thor.ui import universe_member
        return universe_member(st, ticker)
    except Exception:
        return None


def registry_identity(ticker, name: str = "", member=None) -> dict:
    """Fälten en rad du lägger in för hand får ur universum-registret: ins_id och Börsdatas namn (bd_name) när
    registret har bolaget från Börsdata, och namnet när du lämnat det tomt. Anropas bara på ditt klick. Tom dict
    utan register eller utan träff — då blir raden som förut."""
    if member is None:
        member = _registry_member(ticker)
    if member is None:
        return {}
    out: dict = {}
    nm = str(getattr(member, "name", "") or "").strip()
    raw = getattr(member, "ins_id", None)
    try:
        iid = None if isinstance(raw, bool) else int(raw)
    except (TypeError, ValueError):
        iid = None
    if iid is not None and iid > 0:
        out["ins_id"] = iid                       # så att nattjobbet hämtar just det bolaget
        if nm:
            out["bd_name"] = nm
    if nm and not str(name or "").strip():
        out["name"] = nm
    return out


def identity_note(row: dict) -> str:
    """'Börsdata: <namn>' för en rad som fått sitt ins_id ur registret — så syns en fel matchning. Med en varning
    när namnet du skrev inte liknar Börsdatas. Tom sträng utan bd_name. Escapad för st.caption (två '$' blir
    annars matematik)."""
    bd = str((row or {}).get("bd_name") or "").strip()
    if not bd:
        return ""
    own = str((row or {}).get("name") or "").strip()
    note = f"Börsdata: {bd}"
    if own and not name_resembles(own, bd):
        note += f" — liknar inte '{own}'. Kontrollera tickern."
    return note.replace("$", "\\$")


@st.cache_data(ttl=600, show_spinner=False)
def load_refresh() -> dict:
    try:
        from gist_storage import load_blob
        d = load_blob("sheets_refresh.json", None)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def suggestion(blob: dict, sheet: str, row: dict, source_field: str,
               row_id=None, unit_field: Optional[str] = None):
    """(värde, asof, valuta/enhet) ur bloben för raden, eller None.

    row_id när raden som skrivs är en underdict (Lukacs FV ligger i
    row["fv"]) — nyckeln är fortfarande arkets rad. unit_field pekar på ett
    enhetsfält i bloben (commodity_unit) i stället för valutan."""
    s = ((blob or {}).get("rows") or {}).get(f"{sheet}:{row_id or row.get('id')}")
    if not s or s.get(source_field) is None:
        return None
    try:
        unit = s.get(unit_field) if unit_field else s.get("currency")
        return float(s[source_field]), str(s.get("asof") or "")[:10], unit or ""
    except (TypeError, ValueError):
        return None


def _apply(row: dict, field: str, value, widget_key: Optional[str],
           on_apply: Optional[Callable]) -> None:
    row[field] = value
    if widget_key:
        st.session_state[widget_key] = value
    if on_apply:
        on_apply()


def _shown(value, fmt="{:g}") -> str:
    """Förslagets värde som text: kryss som ja/nej, tal med fmt (formatsträng eller funktion), text som den står."""
    if isinstance(value, bool):
        return "ja" if value else "nej"
    if isinstance(value, (int, float)):
        return fmt(value) if callable(fmt) else fmt.format(value)
    return str(value)


def _same(cur, value) -> bool:
    """Har fältet redan förslagets värde? Kryss som kryss, tal som tal, text utan blanksteg i kanterna."""
    if isinstance(value, bool):
        return bool(cur) == value
    if isinstance(value, (int, float)):
        try:
            return cur is not None and cur != "" and abs(float(cur) - float(value)) < 1e-6
        except (TypeError, ValueError):
            return False
    return str(cur if cur is not None else "").strip() == str(value).strip()


def offer(row: dict, field: str, value, label: str, key: str, source: str = "Börsdata", asof: str = "",
          widget_key: Optional[str] = None, on_apply: Optional[Callable] = None, cast: Optional[Callable] = None,
          fmt="{:g}", unit: str = "", note: str = "", text: str = "") -> bool:
    """Ett förslag av vilken typ som helst (tal, text, val, kryss): '🤖 <källa> <datum>: <etikett> <värde> · Använd'.
    Använd skriver fältet i raden och widgetens state via on_click (_apply, samma mekanik som suggest) — inget annat
    skrivs, och att rita förslaget skriver ingenting. cast=int för heltalsrutor (Streamlit tar inte ett flyttal i en
    heltalsruta). Ritas inte när fältet redan har värdet. note = kort tillägg efter värdet (underlaget). text = färdig
    rad i stället för källa/etikett/värde ('Förslag: Olja (BERSERK-listan)'). key gör knappen unik. True när
    förslaget ritades."""
    if value is None:
        return False
    if cast is not None:
        try:
            value = cast(value)
        except (TypeError, ValueError):
            return False
    if _same(row.get(field), value):
        return False
    when = f" {asof}" if asof else ""
    text = text or (f"🤖 {source}{when}: {label} {_shown(value, fmt)}" + (f" {unit}" if unit else "")
                    + (f" — {note}" if note else ""))
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>{html.escape(text).replace('$', '&#36;')}</span>",
                unsafe_allow_html=True)
    c2.button("Använd", key=f"rf_offer_{key}", on_click=_apply, args=(row, field, value, widget_key, on_apply))
    return True


NO_COMMODITY = "—"


def _pick_commodity(row: dict, widget_key: str, on_apply: Optional[Callable]) -> None:
    """on_change för råvaruväljaren: ditt val in i raden innan sidan ritas om — då får nästa ruta nyckeln för det nya
    värdet direkt, och ett andra byte på samma kort går inte förlorat."""
    row["commodity"] = str(st.session_state.get(widget_key) or "")
    if on_apply:
        on_apply()


def commodity_choice(row: dict, names, key: str, on_apply: Optional[Callable] = None, container=None) -> bool:
    """Råvaran på ett arkkort (Rick Rule, Poängmodellen): väljaren ('—' = ingen) och under den förslaget
    'Förslag: Olja (BERSERK-listan) · Använd' (thor.ticker.commodity_hint — listorna och Börsdata-branschen, aldrig
    arket självt). Att rita skriver ingenting: bara ditt byte i rutan (on_change) eller Använd skriver råvaran.
    Rutans nyckel bär det sparade värdet, så en råvara som byts på annat håll (Använd, import) ger en ny ruta i
    stället för att den gamla skriver tillbaka sitt val. Inget förslag när svaret är osäkert — då en rad om varför,
    när råvaran är tom. True när rutan visar något annat än raden (skrivs då in här)."""
    import zlib
    names = [n for n in (names or ()) if n]
    cur = str(row.get("commodity") or "").strip()
    opts = [""] + names + ([cur] if cur and cur not in names else [])
    wk = f"{key}_{format(zlib.crc32(cur.encode('utf-8')), '08x')}"
    picked = (container or st).selectbox(
        "Råvara", opts, index=opts.index(cur), format_func=lambda v: v or NO_COMMODITY, key=wk,
        on_change=_pick_commodity, args=(row, wk, on_apply),
        help="Styr nattens råvarupris och enhetskontrollen. — = ingen vald.")
    changed = picked != cur
    if changed:
        row["commodity"] = picked
    try:
        from thor import ticker as tk
        name, why = tk.commodity_hint(row.get("ticker"), row.get("branch_id"), names)
    except Exception:
        name, why = None, ""
    if name:
        offer(row, "commodity", name, "råvara", key=key, on_apply=on_apply, text=f"Förslag: {name} ({why})")
    elif why and not row.get("commodity"):
        st.caption(f"Inget förslag på råvara: {why} — välj själv.".replace("$", "\\$"))
    return changed


def suggest(sheet: str, row: dict, field: str, source_field: str, label: str,
            widget_key: Optional[str] = None, on_apply: Optional[Callable] = None,
            fmt: str = "{:g}", with_currency: bool = False, row_id=None,
            unit_field: Optional[str] = None, lo: Optional[float] = None) -> None:
    """Rita förslaget om det finns och skiljer sig från fältets värde. lo = fältets golv: ett förslag under det
    (Börsdatas negativa EV/EBITDA vid negativt EBITDA) erbjuds inte — rutan skulle inte kunna visa det."""
    blob = load_refresh()
    sug = suggestion(blob, sheet, row, source_field, row_id=row_id,
                     unit_field=unit_field)
    if sug is None or (lo is not None and sug[0] < lo):
        return
    val, asof, ccy = sug
    with_currency = with_currency or bool(unit_field)
    cur = row.get(field)
    try:
        if cur is not None and abs(float(cur) - val) < 1e-6:
            return
    except (TypeError, ValueError):
        pass
    unit = f" {ccy}" if with_currency and ccy else ""
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>🤖 Börsdata {asof}: "
                f"{label} {fmt.format(val)}{unit}{matched_note(blob, sheet, row, row_id)}</span>",
                unsafe_allow_html=True)
    c2.button("Använd", key=f"rf_{sheet}_{row_id or row.get('id')}_{field}",
              on_click=_apply, args=(row, field, val, widget_key, on_apply))
