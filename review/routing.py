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
        from review import export as ex
        key, _shown, _entry = ex.card_for(ticker, stores, sheet_map)
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
    from review import export as ex
    data = (stores or {}).get(cat.SHEETS["scorecard"].store)
    if not isinstance(data, dict):
        return None
    key, shown, _entry = ex.card_for(ticker, stores, sheet_map)
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
        from review import export as ex
        key = ex.card_for(ticker, stores, sheet_map)[0]
        return ([key] if key else []), ()
    rid = cat._as_row(row).get("id")
    return ([str(rid)] if rid else []), ()


def plan(exp, targets: dict, reset: tuple, stores: Optional[dict], load_failed: Optional[Callable] = None) -> Plan:
    """Routningen för kortets underlag exp (review.export.Export). targets = kortets egna mål {ark: rad/funktion},
    reset = kortets widgetnycklar. stores = sessionens lager (chatgpt_ui.session_stores) — None ger bara kortets
    egna ark, som förr. load_failed(lagernamn) → felet när lagret inte gick att läsa i sessionen: då skrivs inget
    där (en 💾 skulle skriva över det som ligger i GitHub)."""
    own = {s: t for s, t in (targets or {}).items()}
    tokens, prefixes = (tuple(reset or ()) + ((), ()))[:2]
    own_reset = (tuple(tokens or ()), tuple(prefixes or ()))
    if stores is None:
        p = Plan(exp, dict(own), {s: own_reset for s in own})
        return p
    from review import export as ex
    ticker = getattr(exp, "ticker", "")
    smap = getattr(exp, "sheet_map", None) or None
    rows = {s: r for s, r in (getattr(exp, "source_rows", None) or {}).items() if r is not None}
    try:
        full = ex.build(ticker, stores, list(ROUTED), None, None, getattr(exp, "name", ""), rows,
                        getattr(exp, "entry", None), smap)
    except Exception:                                   # routningen får aldrig fälla kortets egen import
        return Plan(exp, dict(own), {s: own_reset for s in own})
    fields = dict(full.fields)
    fields.update(getattr(exp, "fields", None) or {})                  # kortets egna fält först
    source_rows = dict(full.source_rows)
    source_rows.update(getattr(exp, "source_rows", None) or {})
    merged = SimpleNamespace(ticker=ticker, name=getattr(exp, "name", ""), fields=fields, source_rows=source_rows,
                             entry=getattr(exp, "entry", None) or full.entry,
                             sheet_map=getattr(exp, "sheet_map", None), sheets=list(dict.fromkeys(
                                 list(getattr(exp, "sheets", None) or []) + list(full.sheets))))
    p = Plan(merged, dict(own), {s: own_reset for s in own})
    for sheet in ROUTED:
        if sheet in own or source_rows.get(sheet) is None:
            continue                                    # kortets eget ark, eller bolaget finns inte i arket
        if sheet == "scorecard":
            p.targets[sheet] = (lambda t=ticker, m=smap: scorecard_row(t, stores, m))
            p.reset[sheet] = _reset_for(sheet, ticker, None, stores, smap)
            continue
        row = live_row(sheet, ticker, stores, smap)
        if row is None:
            p.targets[sheet] = None
            p.blocked[sheet] = (f"{cat.SHEETS[sheet].label}: bolaget står bara som position i registret — läs in "
                                f"svaret i arket där kandidaten granskas" if sheet == "tiggre" else
                                f"{cat.SHEETS[sheet].label}: raden hittades inte i arket")
            continue
        p.targets[sheet] = row
        p.reset[sheet] = _reset_for(sheet, ticker, row, stores, smap)
    if load_failed is not None:
        for sheet in list(p.targets):
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


def missing_now(exp, rows: dict, stores: Optional[dict]) -> dict:
    """{ark: [fältets namn]} — det som fortfarande saknas för köpbeslut i arken efter inskrivningen, räknat på
    raderna som nu står i arken (rows = {ark: raden})."""
    from review import export as ex
    live = {s: r for s, r in (rows or {}).items() if r is not None}
    if not live:
        return {}
    try:
        e = ex.build(getattr(exp, "ticker", ""), stores or {}, list(live), None, None, getattr(exp, "name", ""),
                     {s: copy.deepcopy(cat._as_row(r)) for s, r in live.items()}, getattr(exp, "entry", None),
                     getattr(exp, "sheet_map", None) or None)
    except Exception:
        return {}
    out: dict = {}
    for it in e.missing:
        if it.sheet in live:
            names = out.setdefault(it.sheet, [])
            if it.label not in names:
                names.append(it.label)
    return out
