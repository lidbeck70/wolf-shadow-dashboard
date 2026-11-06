"""
review/routing.py — ett svar, alla ark: varje rad i ChatGPT-svaret skrivs i det ark Fältnyckeln pekar på.

Förr tog kortets import bara emot sitt eget arks rader ("hör till Rick Rule — läs in svaret där"), så ett svar på
Thors underlag (Rick Rule + Insider + Scorecard) fick läsas in kort för kort. Nu prövas svaret mot alla ark:

  plan(exp, targets, stores)   Routningen för kortets underlag exp: fälten och raderna för ALLA ark (där bolaget
                               finns: dess rad, annars en tom mall — då säger importen "lägg in det i arket först"),
                               målraden per ark (kortets egna mål, för övriga ark raden i sessionens lager — samma
                               objekt arken själva ritar och 💾 sparar), widgetnycklarna som ska släppas per ark och
                               ark som inte kan ta emot svar härifrån, med skälet
  live_row(sheet, …)           raden i sessionens lager — aldrig en kopia (Tiggres positioner i registret är vyer)
  missing_now(…)               det som fortfarande saknas för köpbeslut per ark, efter inskrivningen

Inget sparas härifrån — kvittot i importen har en 💾-knapp för de ark som fick svar. Inget nätverk utöver det
storage.session_load redan gör när ett lager läses första gången i sessionen.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Optional

from review import catalog as cat

# Arken importen prövar svaret mot. Wolf Asymmetry och Durrett 10-steg delar bolagsraden (confidence.json) —
# Durrett-arket står för båda (catalog.CONFIDENCE_SHEETS; importer.resolve tar asymmetry:-nycklar dit).
ROUTED = tuple(s for s in cat.SHEETS if s != "asymmetry")
# Lagrens namn i klartext — samma som sparraden på sidan
STORE_LABEL = {"producers": "Granskningsarken", "scoring": "Poängmodellen", "tiggre": "Tiggre",
               "insider": "Insiderbevakaren", "scorecard": "Master Scorecard", "confidence": "Durrett / Confidence"}


@dataclass
class Plan:
    """Routningen för ett svar: underlaget att pröva mot och vart varje ark skrivs."""
    exp: object                                        # fält och källrader för alla ark (som review.export.Export)
    targets: dict = field(default_factory=dict)        # {ark: rad eller funktion som ger raden}
    reset: dict = field(default_factory=dict)          # {ark: (id:n, prefix)} — kortets widgets för arket
    blocked: dict = field(default_factory=dict)        # {ark: skäl} — bolaget finns där men går inte att skriva
    own: set = field(default_factory=set)              # kortets egna ark — nycklar utan arkprefix prövas där först

    @property
    def writable(self) -> set:
        return {s for s, t in self.targets.items() if t is not None and s not in self.blocked}

    def store_of(self, sheet: str) -> str:
        return cat.SHEETS[sheet].store if sheet in cat.SHEETS else ""


def _forms(ticker: str, sheet_map: Optional[dict] = None) -> list:
    from review import export as ex
    return ex._forms(ticker, sheet_map)


def live_row(sheet: str, ticker: str, stores: dict, sheet_map: Optional[dict] = None) -> Optional[dict]:
    """Bolagets rad i sessionens lager (samma objekt som arket ritar), eller None. Platta ark: arkets egen lista
    (Tiggre: kandidaterna — en position i registret är en vy och tar inte emot svar). Durrett: bolaget i
    companies. Scorecardet: kortet i lagret, None om det inte finns ännu (scorecard_row skapar det)."""
    if sheet not in cat.SHEETS:
        return None
    data = (stores or {}).get(cat.SHEETS[sheet].store)
    if not isinstance(data, dict):
        return None
    forms = [f.upper() for f in _forms(ticker, sheet_map)]
    if sheet in cat.CONFIDENCE_SHEETS:
        comps = data.get("companies")
        if not isinstance(comps, dict):
            return None
        return next((comps[f] for f in forms if isinstance(comps.get(f), dict)), None)
    if sheet == "scorecard":
        key, _shown, _entry = card_key(ticker, stores, sheet_map)
        cards = data.get("cards")
        return cards.get(key) if key and isinstance(cards, dict) and isinstance(cards.get(key), dict) else None
    rows = data.get(cat.SHEETS[sheet].bucket)
    if not isinstance(rows, list):
        return None
    for f in forms:
        for row in rows:
            if isinstance(row, dict) and str(row.get("ticker") or "").strip().upper() == f:
                return row
    return None


def scorecard_row(ticker: str, stores: dict, sheet_map: Optional[dict] = None) -> Optional[dict]:
    """Scorecardkortet att skriva i — skapas först när ett svar skrivs in (som i Scorecardet: ett orört kort skrivs
    aldrig), med strategin kortet visas med. None utan kandidat eller kort."""
    data = (stores or {}).get(cat.SHEETS["scorecard"].store)
    if not isinstance(data, dict):
        return None
    key, shown, _entry = card_key(ticker, stores, sheet_map)
    if not key:
        return None
    cards = data.get("cards")
    if not isinstance(cards, dict):
        cards = data["cards"] = {}
    if not isinstance(cards.get(key), dict):
        cards[key] = {"strategy": (shown or {}).get("strategy")}
    return cards[key]


def _reset_for(sheet: str, ticker: str, row, stores: dict, sheet_map: Optional[dict]) -> tuple:
    """Widgetnycklarna arkets kort håller värdena i: radens id (platta ark), kortnyckeln (Scorecardet) eller
    Durrett/Wolf Asymmetrys prefix per bolag."""
    if sheet in cat.CONFIDENCE_SHEETS:
        t = str(cat._as_row(row).get("ticker") or ticker).strip()
        return (), (f"cf_{t}_", f"ds_{t}_")
    if sheet == "scorecard":
        key = card_key(ticker, stores, sheet_map)[0]
        return ([key] if key else []), ()
    rid = cat._as_row(row).get("id")
    return ([str(rid)] if rid else []), ()


def _in_store(sheet: str, row, stores: dict) -> bool:
    """Är raden självaste objektet i sessionens lager (inte en kopia eller en vy)?"""
    data = (stores or {}).get(cat.SHEETS[sheet].store)
    if not isinstance(row, dict) or not isinstance(data, dict):
        return False
    if sheet in cat.CONFIDENCE_SHEETS:
        comps = data.get("companies")
        return isinstance(comps, dict) and any(v is row for v in comps.values())
    rows = data.get(cat.SHEETS[sheet].bucket)
    return isinstance(rows, list) and any(r is row for r in rows)


def ambiguous(row_ticker, ticker: str, sheet_map: Optional[dict]) -> bool:
    """Utan registrets tickerkarta är en bar arkrad ('TEL') inte säkert samma bolag som en nordisk notering
    ('TEL.OL' — Telenor, medan 'TEL' i USA är TE Connectivity). Thor visar den med en notering; importen skriver
    inte dit. Formen med mellanslag ('ERIC B') är alltid nordisk."""
    from thor import candidates as tc
    t = str(ticker or "").strip().upper()
    r = str(row_ticker or "").strip().upper()
    return (not sheet_map and "." in t and tc.is_nordic(t) and bool(r) and r != t and "." not in r
            and " " not in r)


def card_key(ticker: str, stores: dict, sheet_map: Optional[dict] = None) -> tuple:
    """(kortnyckel, kortet som det visas, kandidatraden) — som export.card_for (samma kort som Thors underlag), men
    när det kortet står på en bar form som kan vara ett annat bolag (ambiguous) tas kandidaten med exakt ticker
    ('telenor|TEL.OL' före 'TEL', som sorteras först)."""
    import scorecard as sc
    from review import export as ex
    key, shown, found = ex.card_for(ticker, stores, sheet_map)
    if found and ambiguous(found.get("ticker"), ticker, sheet_map):
        data = (stores or {}).get(cat.SHEETS["scorecard"].store)
        cards = data.get("cards") if isinstance(data, dict) and isinstance(data.get("cards"), dict) else {}
        for k, e in sc.entries_for([str(ticker or "").strip().upper()], stores or {}).items():
            c = cards.get(k)
            return k, sc._shown(c if isinstance(c, dict) else {}, e), e
    return key, shown, found


def plan(exp, targets: dict, reset: tuple, stores: Optional[dict], load_failed: Optional[Callable] = None,
         sheet_map: Optional[dict] = None) -> Plan:
    """Routningen för kortets underlag exp (review.export.Export). targets = kortets egna mål {ark: rad/funktion},
    reset = kortets widgetnycklar. stores = sessionens lager (chatgpt_ui.session_stores) — None ger bara kortets
    egna ark, som förr. load_failed(lagernamn) → felet när lagret inte gick att läsa i sessionen: då skrivs inget
    där (en 💾 skulle skriva över det som ligger i GitHub). sheet_map = registrets karta (som Thor läser): 'BOL' i
    ett ark är BOL.ST, 'TEL' är inte Telenor — utan den skrivs inget i en bar arkrad för en nordisk notering.

    Raden jämförelsen läses ur är raden som skrivs: kortets låsta rad när den står i lagret (Scorecardets kandidat
    läser arkens senaste rad), annars bolagets rad i lagret — och källraden sätts till just den."""
    own = {s: t for s, t in (targets or {}).items()}
    tokens, prefixes = (tuple(reset or ()) + ((), ()))[:2]
    own_reset = (tuple(tokens or ()), tuple(prefixes or ()))
    if stores is None:
        return Plan(exp, dict(own), {s: own_reset for s in own}, own=set(own))
    from review import export as ex
    from universe import registry as ur
    card_ticker = getattr(exp, "ticker", "")
    smap = getattr(exp, "sheet_map", None) or sheet_map or None
    ticker = ur.nordic_form(card_ticker, smap) or card_ticker          # 'BOL' i arket → BOL.ST med kartan
    rows = {s: r for s, r in (getattr(exp, "source_rows", None) or {}).items() if r is not None}
    own_sheets = set(getattr(exp, "sheets", None) or []) | set(own)
    name = getattr(exp, "name", "")
    entry = getattr(exp, "entry", None) if "scorecard" in own_sheets else None   # annars hela lagrens kandidat
    if "scorecard" not in own_sheets:
        _k, shown, found = card_key(ticker, stores, smap)
        if found is not None:                           # samma kort som Scorecardet skrivs i (exakt ticker först)
            entry = found
            rows.setdefault("scorecard", shown)
    sc_store = cat.SHEETS["scorecard"].store
    try:
        # Fälten i två synvinklar: köpgrindens position (Scorecardets — som Thors underlag) och arkets egen (som
        # arkets kort). Ett fält som någon av dem visar går att läsa in; kvittot räknar som arket (missing_now).
        full = ex.build(ticker, stores, list(ROUTED), None, None, name, rows, entry, smap)
        bare = {k: v for k, v in stores.items() if k != sc_store}
        own_view = ex.build(ticker, bare, [s for s in ROUTED if s != "scorecard"], None, None, name,
                            {s: r for s, r in rows.items() if s != "scorecard"}, None, smap)
    except Exception:                                   # routningen får aldrig fälla kortets egen import
        return Plan(exp, dict(own), {s: own_reset for s in own}, own=set(own))
    fields = dict(own_view.fields)
    fields.update(full.fields)
    fields.update(getattr(exp, "fields", None) or {})                  # kortets egna fält först
    source_rows = dict(full.source_rows)
    source_rows.update(rows)
    shown_who = {r.get("key"): r.get("who") for r in getattr(exp, "rows", None) or [] if isinstance(r, dict)}
    merged = SimpleNamespace(ticker=ticker, name=name, fields=fields, source_rows=source_rows,
                             entry=full.entry, sheet_map=smap, own_sheets=set(own), shown_who=shown_who,
                             sheets=list(dict.fromkeys(list(getattr(exp, "sheets", None) or []) + list(full.sheets))))
    p = Plan(merged, dict(own), {s: own_reset for s in own}, own=set(own))
    for sheet in ROUTED:
        if sheet in own:
            continue                                    # kortets eget ark — kortets mål, som förr
        label = cat.SHEETS[sheet].label
        err = load_failed(p.store_of(sheet)) if load_failed is not None else None
        if err:
            p.targets[sheet] = None
            p.blocked[sheet] = (f"{label} kunde inte läsas i den här sessionen ({err}) — ladda om sidan innan "
                                f"något skrivs in där")
            continue
        src = source_rows.get(sheet)
        if src is None:
            continue                                    # bolaget finns inte i arket — importen säger det per rad
        if sheet == "scorecard":
            key, _shown, found = card_key(ticker, stores, smap)
            if found and ambiguous(found.get("ticker"), ticker, smap):
                p.targets[sheet] = None
                p.blocked[sheet] = (f"{label}: kortet står på '{found.get('ticker')}' och registrets tickerkarta "
                                    f"kunde inte läsas — det kan vara ett annat bolag; läs in svaret i Scorecardet")
                continue
            p.targets[sheet] = (lambda t=ticker, m=smap: scorecard_row(t, stores, m))
            p.reset[sheet] = ([key] if key else []), ()
            continue
        row = src if _in_store(sheet, src, stores) else live_row(sheet, ticker, stores, smap)
        if row is None:
            p.targets[sheet] = None
            p.blocked[sheet] = (f"{label}: bolaget står bara som position i registret — positionen har inga fält "
                                f"för de här raderna, så de sparas inte. Ska det granskas: lägg in det som kandidat "
                                f"i Tiggre-arket och läs in svaret där" if sheet == "tiggre" else
                                f"{label}: raden hittades inte i arket")
            continue
        if ambiguous(cat._as_row(row).get("ticker"), ticker, smap):
            p.targets[sheet] = None
            p.blocked[sheet] = (f"{label}: raden står på '{cat._as_row(row).get('ticker')}' och registrets "
                                f"tickerkarta kunde inte läsas — det kan vara ett annat bolag; läs in svaret i "
                                f"det arket")
            continue
        p.targets[sheet] = row
        source_rows[sheet] = cat._as_row(row)           # jämförelsen läser raden som skrivs
        p.reset[sheet] = _reset_for(sheet, ticker, row, stores, smap)
    if load_failed is not None:
        for sheet in own:
            err = load_failed(p.store_of(sheet))
            if err:
                p.blocked[sheet] = (f"{cat.SHEETS[sheet].label} kunde inte läsas i den här sessionen ({err}) — "
                                    f"ladda om sidan innan något skrivs in där")
    return p


def resolved_targets(p: Plan, sheets) -> dict:
    """{ark: raden} för arken som fick ändringar — funktionerna anropas nu (Scorecardkortet skapas här)."""
    out = {}
    for sheet in sheets:
        t = p.targets.get(sheet)
        out[sheet] = t() if callable(t) else t
    return out


def _collect(e, live: dict, out: dict) -> None:
    for it in e.missing:
        if it.sheet in live:
            names = out.setdefault(it.sheet, [])
            if it.label not in names:
                names.append(it.label)


def missing_now(exp, rows: dict, stores: Optional[dict], own=()) -> dict:
    """{ark: [fältets namn]} — det som fortfarande saknas för köpbeslut i arken efter inskrivningen, räknat på
    raderna som nu står i arken (rows = {ark: raden}) och som varje ark själv räknar: arkens egna kontroller efter
    kortets position, utom när importen körs i Scorecardet — där följer kontrollerna köpgrindens position, som
    Scorecardets underlag. Scorecardkortet räknas alltid mot hela lagrens kandidat."""
    from review import export as ex
    live = {s: r for s, r in (rows or {}).items() if r is not None}
    if not live:
        return {}
    stores = stores or {}
    ticker, name = getattr(exp, "ticker", ""), getattr(exp, "name", "")
    entry, smap = getattr(exp, "entry", None), getattr(exp, "sheet_map", None) or None
    in_scorecard = "scorecard" in set(own or ())
    sc_store = cat.SHEETS["scorecard"].store
    out: dict = {}
    flat = [s for s in live if s != "scorecard"]
    try:
        if flat:
            base = stores if in_scorecard else {k: v for k, v in stores.items() if k != sc_store}
            frows = {s: copy.deepcopy(cat._as_row(live[s])) for s in flat}
            if in_scorecard:                            # kortet importen kördes i — dess position och strategi
                card = dict((getattr(exp, "source_rows", None) or {}).get("scorecard") or {})
                card.update(cat._as_row(live.get("scorecard")) if live.get("scorecard") is not None else {})
                if card:
                    frows["scorecard"] = copy.deepcopy(card)   # läses bara för position/strategi — inte ett ark här
            e = ex.build(ticker, base, flat, None, None, name, frows, entry if in_scorecard else None, smap)
            _collect(e, live, out)
        if "scorecard" in live:
            e = ex.build(ticker, stores, ["scorecard"], None, None, name,
                         {"scorecard": copy.deepcopy(cat._as_row(live["scorecard"]))}, entry, smap)
            _collect(e, live, out)
    except Exception:
        return out
    return {s: out[s] for s in live if s in out}
