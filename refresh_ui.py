"""
refresh_ui.py — "🤖 Börsdata: 123,4 · Använd" bredvid ett arkfält.

Läser sheets_refresh.json (sheets_refresh.py i GitHub Actions) och visar
jobbets färska tal som FÖRSLAG. Knappen skriver in värdet i raden och i
widgetens session-state via on_click-callback (det enda tillfället då
Streamlit tillåter att en instansierad widgets state ändras) — inget
skrivs annars, och 💾 Spara sköter persistensen som vanligt.

offer() är samma knapp för förslag av vilken typ som helst (heltal, text, val,
kryss) ur andra källor än bloben — t.ex. Insider-skanningen. offer_fields() är
samma knapp för tal som bara gäller tillsammans (Royalty C:s GEO-par).

'Hämta nu' (fetch_live, fetch_button): en rad hämtad på ditt klick läggs i
sessionen under radens nyckel ('producers:<id>', 'tiggre:<id>',
'confidence:<TICKER>') och current_blob() lägger den över nattens blob — så
läser kortets förslag den först. Bara i sessionen: inget skrivs till Gisten
eller repot, och raden i arket ändras bara med Använd (💾 sparar).
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
    bd = str(s.get("bd_name") or s.get("yahoo_name") or "").strip()      # Yahoo-reservens namn för 'Hämta nu'
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


# ── 'Hämta nu': raden du hämtat, bara i sessionen ───────────────────────────
LIVE_KEY = "rf_live_rows"          # session: {'producers:<id>': rad} — aldrig till Gisten eller repot
MSG_KEY = "rf_live_msg"            # session: {'producers:<id>': varför senaste 'Hämta nu' inte gav någon rad}
_COMMODITY_FIELDS = ("commodity_price", "commodity_unit", "commodity_asof", "commodity_ticker")
FETCH_HELP = ("Hämtar bolagets tal nu, utan att vänta på nattens körning: Börsdata först, Yahoo om Börsdata inte "
              "känner tickern. Varje klick drar Börsdata-kvot. Utan ins_id kan fel bolag matchas — kontrollera "
              "namnet som visas. Talen blir förslag med Använd; inget sparas förrän du trycker 💾.")


def _session_dict(key: str) -> dict:
    try:
        v = st.session_state.get(key)
    except Exception:
        return {}
    return v if isinstance(v, dict) else {}


def live_rows() -> dict:
    """{radnyckel: rad} som hämtats med 'Hämta nu' i sessionen. Tom utan sådana (eller utan session)."""
    return _session_dict(LIVE_KEY)


def _set_msg(ref: str, text: str) -> None:
    """Varför 'Hämta nu' inte gav någon rad, under radens nyckel (inte knappens) — så att en rad som kommer in på
    annat sätt ('Hämta EV nu') tar bort ett gammalt 'hittade inget'. Tom text tar bort; skriver bara vid ändring."""
    msgs = _session_dict(MSG_KEY)
    if text:
        st.session_state[MSG_KEY] = {**msgs, ref: text}
    elif ref in msgs:
        st.session_state[MSG_KEY] = {k: v for k, v in msgs.items() if k != ref}


def _live_applies(night: Optional[dict], live: Optional[dict]) -> bool:
    """Går den hämtade raden före nattens? Ja, utom när nattens rad är nyare (en körning efter hämtningen, i en session
    som stått öppen) — datumen jämförs som text (ÅÅÅÅ-MM-DD). Saknar någon av dem datum gäller hämtningen."""
    n, f = str((night or {}).get("asof") or "")[:10], str((live or {}).get("asof") or "")[:10]
    return not (n and f and n > f)


def is_live(ref: str) -> bool:
    """Läser kortet raden '<ark>:<id>' ur en hämtning i sessionen (och inte ur nattens blob)?"""
    live = live_rows().get(ref)
    if live is None:
        return False
    try:
        night = ((load_refresh() or {}).get("rows") or {}).get(ref)
    except Exception:
        night = None
    return _live_applies(night, live)


def put_live(ref: str, row: dict) -> None:
    """Lägg en hämtad rad i sessionen under radens nyckel. Ersätter en tidigare hämtning av samma rad och tar bort
    ett gammalt 'hittade inget' för raden."""
    rows = dict(live_rows())
    rows[ref] = dict(row)
    st.session_state[LIVE_KEY] = rows
    _set_msg(ref, "")


def with_live(blob, live: Optional[dict]):
    """Nattens blob med de hämtade raderna ovanpå: en hämtad rad ersätter nattens rad för samma nyckel — utom när
    nattens rad är nyare (en körning efter hämtningen, i en session som stått öppen). Nattens råvarupris
    (commodity_*) följer med när hämtningen inte fick något eget — det har sitt eget datum och gäller bara medan
    kortets råvara är densamma (commodity_prices.night_fits). Bloben som den är utan hämtade rader."""
    if not live:
        return blob
    night = (blob or {}).get("rows") or {}
    rows = dict(night)
    for ref, r in live.items():
        n = night.get(ref) or {}
        if not _live_applies(n, r):
            continue                                    # natten har kört sedan du hämtade
        merged = dict(r)
        if merged.get("commodity_price") is None and n.get("commodity_price") is not None:
            merged.update({k: n[k] for k in _COMMODITY_FIELDS if k in n})
        rows[ref] = merged
    return {**(blob or {}), "rows": rows}


def current_blob():
    """Sifferuppdateringens blob som kortens förslag läser: raderna du hämtat nu (live_rows) före nattens (load_refresh)."""
    return with_live(load_refresh(), live_rows())


def source_name(s: Optional[dict]) -> str:
    """Var radens tal kommer ifrån: 'Börsdata', eller 'Yahoo (FNV.TO)' för Yahoo-reservens rad — suffixsökningen kan
    träffa fel bolag, så symbolen den använde står med."""
    s = s or {}
    if str(s.get("source") or "") == "yfinance" and s.get("yahoo"):
        return f"Yahoo ({s['yahoo']})"
    return "Börsdata"


def fetched_line(s: Optional[dict], row: Optional[dict]) -> str:
    """Raden under 'Hämta nu': varifrån talen kom, datum och vilket bolag som matchades — alltid med namnet, eftersom
    en rad utan ins_id kan få fel bolag. Varning när namnet inte liknar radens, när raden saknar ins_id och när
    kursvalutan saknas i växelkurstabellen (då inga omräknade MUSD-tal). Oescapad text; tom sträng utan rad."""
    if not isinstance(s, dict):
        return ""
    row = row or {}
    asof = str(s.get("asof") or "")[:10]
    yahoo = source_name(s) != "Börsdata"
    name = str((s.get("yahoo_name") if yahoo else s.get("bd_name")) or "").strip()
    sym = "" if yahoo else str(s.get("bd_ticker") or "").strip()
    who = (name + (f" ({sym})" if sym else "")) if name else sym
    px = s.get("price")
    seen = (f"kurs {px:,.4g}{' ' + str(s.get('currency')) if s.get('currency') else ''}"
            if isinstance(px, (int, float)) and not isinstance(px, bool) else "")
    txt = (f"🤖 Hämtat nu ur {source_name(s)}{' ' + asof if asof else ''}: "
           + (f"{who}{', ' + seen if seen else ''}." if who else
              f"svaret har inget bolagsnamn{' (' + seen + ')' if seen else ''} — kontrollera att det är rätt bolag."))
    own = str(row.get("name") or "").strip()
    if name and own and not name_resembles(own, name):
        txt += f" Liknar inte '{own}' — kontrollera tickern."
    if yahoo:
        txt += " Yahoo provar tickern med börssuffix och kan träffa fel bolag."
    elif row.get("ins_id") in (None, ""):
        txt += " Raden saknar ins_id — matchat på tickern."
    ccy = str(s.get("currency") or "").strip().upper()
    try:
        from sheets_refresh import FX_TO_USD
        unknown = bool(ccy) and ccy not in FX_TO_USD
    except Exception:                                   # pragma: no cover
        unknown = False
    if unknown:
        txt += f" Kursen står i {ccy}, som saknas i växelkurstabellen — inga omräknade MUSD-tal."
    return txt


def _global_list_answered(api) -> None:
    """Kastar när Börsdatas globala lista inte svarade. Tickeruppslaget (sheets_refresh.resolve_global) läser den
    genom get_global_instruments_list, som sväljer felet och ger [] — så en timeout, 5xx, 401, 429 eller nätet blev
    'känner inte' och Yahoo på bara tickern, fast NYSE-, TSX- och ASX-tickrar bara finns i den listan. Anropas bara
    när inget hittades: en nordisk ticker laddar inte ner listan. Efter en lyckad hämtning svarar klientens diskcache
    (ingen extra kvot). 400 (ingen global licens, cachad som {}) och 403 ger 'känner inte' och Yahoo som förut."""
    raw_get = getattr(api, "_get", None)
    if not callable(raw_get) or not callable(getattr(api, "get_global_instruments_list", None)):
        return
    try:
        raw_get("/instruments/global")
    except Exception as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) != 403:
            raise


def fetch_live(sheet: str, row: dict, row_id=None, commodity: Optional[str] = None, api=None, info_getter=None,
               yahoo: bool = True) -> tuple:
    """'Hämta nu' för en arkrad: samma rad som nattjobbet bygger (sheets_refresh.fetch_row, Börsdata) — eller Yahoo-
    reservens (asymmetry/fetch.yahoo_row) när Börsdata inte känner tickern och yahoo=True — i nattens form för arket
    (sheets_refresh.as_sheet_row), med dagens terminspris för commodity (kortets råvara; commodity_prices.spot).
    Raden läggs i sessionen (put_live) och ingenting skrivs i arket. Svarar Börsdata inte (kvoten, nätet) provas inte
    Yahoo: den söker på bara tickern och kan träffa ett annat bolag, så nattens rad och en tidigare hämtning står
    kvar. (rad | None, varför Börsdata inte gav någon — tom när den gjorde det)."""
    import sheets_refresh as sr
    t = str((row or {}).get("ticker") or "").strip().upper()
    if not t:
        return None, "Raden saknar ticker."
    try:
        if api is None:
            from borsdata_api import BorsdataAPI
            api = BorsdataAPI()
        # Instrumentlistan först: utan ins_id sväljer tickeruppslaget (sheets_refresh.resolve) felet och svarar 'känner
        # inte', så en saknad nyckel, 401, 429 eller nätet gav Yahoo på bara tickern. fetch_row läser listan ändå, och
        # klienten cachar den — ingen extra kvot.
        instruments = getattr(api, "get_instruments", None)
        if callable(instruments):
            instruments()
        got = sr.fetch_row(api, t, (row or {}).get("ins_id"))
        if got is None:
            _global_list_answered(api)
    except Exception as exc:
        return None, f"Börsdata svarade inte: {exc}"
    why = "" if got is not None else f"Börsdata känner inte {t}."
    if got is None and yahoo:
        try:
            from asymmetry import fetch as af
            got = af.yahoo_row(t, info_getter)
        except Exception:
            got = None
        if got is None:
            return None, why + f" Yahoo har inte heller {t} — vänta på nattens körning eller skriv talen själv."
    if got is None:
        return None, why
    out = sr.as_sheet_row(got, sheet)
    if commodity:
        try:
            import commodity_prices
            sp = commodity_prices.spot(commodity)
        except Exception:
            sp = None
        if sp:
            out.update(commodity_price=sp["price"], commodity_unit=sp["unit"], commodity_asof=sp["asof"],
                       commodity_ticker=sp.get("ticker"))
    put_live(f"{sheet}:{row_id or row.get('id')}", out)
    return out, why


def _fetch_click(sheet: str, row: dict, row_id, commodity: Optional[str]) -> None:
    """on_click för 'Hämta nu': raden in i sessionen (put_live tar då bort ett gammalt meddelande) och, när inget
    hittades, varför (visas under knappen)."""
    got, why = fetch_live(sheet, row, row_id=row_id, commodity=commodity)
    if got is None:
        _set_msg(f"{sheet}:{row_id or row.get('id')}", why)


def fetch_button(sheet: str, row: dict, key: str, row_id=None, commodity: Optional[str] = None,
                 container=None) -> None:
    """'Hämta nu' med raden om vad som hämtades (fetched_line) eller varför inget hittades. Att rita hämtar ingenting —
    bara klicket (on_click), och det skriver bara sessionens överlagring, aldrig raden i arket. container = där
    knappen ritas (Tiggre: under Börsvärde; raden står då under kolumnerna); utan den står knapp och rad bredvid
    varandra."""
    if container is None:
        container, note = st.columns([1, 3], vertical_alignment="center")
    else:
        note = st
    container.button("Hämta nu", key=key, on_click=_fetch_click, args=(sheet, row, row_id, commodity),
                     help=FETCH_HELP)
    ref = f"{sheet}:{row_id or row.get('id')}"
    msg = _session_dict(MSG_KEY).get(ref)
    text = msg or (fetched_line(live_rows().get(ref), row) if is_live(ref) else "")
    if text:
        note.caption(text.replace("$", "\\$"))


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
    _offer_line(text, key, _apply, (row, field, value, widget_key, on_apply))
    return True


def _offer_line(text: str, key: str, on_click: Callable, args: tuple) -> None:
    """Förslagsraden: texten (escapad — två '$' blir annars matematik) och Använd med on_click."""
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>{html.escape(text).replace('$', '&#36;')}</span>",
                unsafe_allow_html=True)
    c2.button("Använd", key=f"rf_offer_{key}", on_click=on_click, args=args)


def _apply_fields(row: dict, values: dict, widget_keys: dict, on_apply: Optional[Callable]) -> None:
    """on_click för offer_fields: alla fälten och deras rutor (_apply), sedan on_apply en gång."""
    for field, value in values.items():
        _apply(row, field, value, widget_keys.get(field), None)
    if on_apply:
        on_apply()


def _same_rel(cur, value) -> bool:
    """Samma tal, relativt — så att små tal (GEO per aktie 0,00004) inte räknas som lika för att de ligger nära noll."""
    try:
        c, v = float(cur), float(value)
    except (TypeError, ValueError):
        return False
    return abs(c - v) <= 1e-9 * max(abs(c), abs(v))


def offer_fields(row: dict, values: dict, text: str, key: str, widget_keys: Optional[dict] = None,
                 on_apply: Optional[Callable] = None) -> bool:
    """Tal som bara gäller tillsammans (GEO per aktie nu och för tre år sedan) med EN Använd. text = hela raden,
    med talen bakom förslaget. Använd skriver alla fälten i raden och deras rutor (widget_keys: fält → nyckel) via
    on_click, som offer — att rita skriver ingenting. Ritas inte när något värde saknas, eller när alla fälten
    redan har sina värden. True när förslaget ritades."""
    if not values or any(v is None for v in values.values()):
        return False
    if all(_same_rel(row.get(f), v) for f, v in values.items()):
        return False
    _offer_line(text, key, _apply_fields, (row, dict(values), dict(widget_keys or {}), on_apply))
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
    (Börsdatas negativa EV/EBITDA vid negativt EBITDA) erbjuds inte — rutan skulle inte kunna visa det. En rad
    hämtad nu (current_blob) går före nattens och märks '(hämtat nu)'; Yahoo-reservens med symbolen."""
    blob = current_blob()
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
    ref = f"{sheet}:{row_id or row.get('id')}"
    src = html.escape(source_name(((blob or {}).get("rows") or {}).get(ref)))
    now = " (hämtat nu)" if is_live(ref) else ""
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>🤖 {src} {asof}{now}: "
                f"{label} {fmt.format(val)}{unit}{matched_note(blob, sheet, row, row_id)}</span>",
                unsafe_allow_html=True)
    c2.button("Använd", key=f"rf_{sheet}_{row_id or row.get('id')}_{field}",
              on_click=_apply, args=(row, field, val, widget_key, on_apply))
