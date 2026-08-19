"""
cycle.py — cykelläget in i Copiloten.

Rotationsfliken betygsätter råvarorna varje månad (Triple Signal: hat, case,
katalysator → AGERA/Bevaka/Vila). Det ÄR panelens cykelläge — men Copiloten
visste inte om det, så köpgrindens första fråga ("är strategin aktiv i rätt
fas?") fick ställas ur minnet.

Modulen kopplar ihop dem: slår upp kandidatens råvara i granskningsarken,
läser rotationsbetyget, och översätter det till PASS/MANUAL/FAIL. Blindspot-
läsningen är samma idé för tickern själv — senaste SPARADE rapporten, aldrig
en ny körning; Blindspot-motorn gör nätverksanrop och hör inte hemma i en
rerun-väg.

Rena funktioner: allt tar data som argument. Ingen Streamlit.
"""

from __future__ import annotations

from typing import Optional

import rotation

# Strategierna vars köpgrind kräver rotationsflikens cykelläge — Masterguidens
# fem råvarustrategier. Contrarian och quality talar också om cykelfas, men
# deras fas är inte nödvändigtvis Triple Signal-betyget, och en felaktigt
# påtvingad FAIL-grind är värre än ingen: hellre tyst än fel källa.
COMMODITY_STRATEGIES = ("rule", "durrett", "sprott", "tiggre", "royalty")

UNGRADED = ("Råvaran är inte betygsatt i rotationsfliken — sätt Triple "
            "Signal-betyget där först. Ett obetygsatt läge är inte ett "
            "godkänt läge.")
NO_COMMODITY = ("Ingen råvara vald. Välj i listan — eller lägg in bolaget i "
                "Rick Rule-arket, så hittas råvaran därifrån.")


def requires_cycle(strategy: str) -> bool:
    return (strategy or "").strip().lower() in COMMODITY_STRATEGIES


def commodity_for_ticker(ticker: str, producers_data: dict) -> Optional[str]:
    """Råvarunamnet ur granskningsarken, om bolaget redan är inlagt där.

    Rick Rule-arket frågar redan vilken råvara varje bolag tillhör — den
    frågan ska inte ställas två gånger. Royalty-arket har ingen råvarukolumn;
    royaltybolag är per definition diversifierade, så där finns inget att
    hämta.
    """
    name = (ticker or "").strip().upper()
    if not name:
        return None
    for row in (producers_data or {}).get("producers", []) or []:
        if isinstance(row, dict) and (row.get("ticker") or "").upper() == name:
            return row.get("commodity") or None
    return None


def commodity_key(commodity_name: str) -> Optional[str]:
    """Rotationsnyckeln (guld, uran, ...) för ett råvarunamn."""
    wanted = (commodity_name or "").strip().lower()
    if not wanted:
        return None
    for c in rotation.COMMODITIES:
        if c.name.lower() == wanted or c.key == wanted:
            return c.key
    return None


def cycle_state(commodity_name: str, rotation_data: dict) -> Optional[dict]:
    """Cykelläget för en råvara, ur rotationsfliken.

    None när råvaran inte är betygsatt — det är INTE samma sak som Vila.
    Obetygsatt betyder att månadens gradering inte är gjord; Vila betyder att
    den är gjord och sa nej.
    """
    key = commodity_key(commodity_name)
    if key is None:
        return None
    grade = ((rotation_data or {}).get("grades") or {}).get(key)
    if not isinstance(grade, dict) or not grade:
        return None
    status, why = rotation.status(grade)
    return {
        "commodity": commodity_name,
        "key": key,
        "status": status,
        "why": why,
        "sum": rotation.signal_sum(grade),
        "max": rotation.SUM_MAX,
        "warnings": rotation.warnings(grade),
        "month": (rotation_data or {}).get("month", ""),
    }


def gate_from_cycle(state: Optional[dict], commodity_name: str = "") -> tuple:
    """(status, notering) för köpgrindens cykelregel.

    AGERA → PASS, Vila → FAIL, Bevaka → MANUAL: rotationsfliken säger själv
    "Bevaka väntar" — det är ett beslut kvar att fatta, inte ett nej.
    Obetygsatt eller utan råvara → MANUAL med instruktion, aldrig ett tyst
    godkännande.
    """
    if not commodity_name:
        return ("MANUAL", NO_COMMODITY)
    if state is None:
        return ("MANUAL", UNGRADED)
    line = (f"{state['commodity']}: {state['status']} "
            f"({state['sum']}/{state['max']}, {state['month'] or 'okänd månad'}) "
            f"— {state['why']}")
    for w in state["warnings"]:
        line += f" · VARNING: {w}"
    if state["status"] == rotation.AGERA:
        return ("PASS", line)
    if state["status"] == rotation.BEVAKA:
        return ("MANUAL", line)
    return ("FAIL", line)


def blindspot_latest(ticker: str) -> Optional[dict]:
    """Senaste sparade Blindspot-raden för tickern. Läser bara — kör aldrig.

    None när ingen rapport finns eller modulen inte går att läsa. Raderna är
    ögonblicksbilder; timestampen följer med så att läsaren ser hur gammal
    bilden är i stället för att tro att den är färsk.
    """
    name = (ticker or "").strip().upper()
    if not name:
        return None
    try:
        from blindspot.history import read_history
        entries = read_history(name)
    except Exception:
        return None
    return entries[-1] if entries else None
