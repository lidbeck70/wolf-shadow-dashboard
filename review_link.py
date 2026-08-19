"""
review_link.py — granskningsarken in i Copilotens regelkontroll.

Ordningen i systemet är screener → granskning → Copilot/scorecard → köp.
Copiloten ska inte göra om granskningen — den ska LÄSA den. Har du satt
Rick Rule-poängen på EQX ska Copilotens regelkontroll visa det utfallet,
inte be dig kontrollera det manuellt en gång till.

Varje ark behåller sin egen bedömningslogik: producer_verdict, royalty_signal,
poängmodellens band, Tiggres hårda grindar, Insiderbevakarens statusflöde.
Den här modulen översätter bara respektive arks eget utfall till
PASS/MANUAL/FAIL — trösklarna bor kvar i arken, så Copiloten och arket kan
per konstruktion inte säga olika saker.

Eftersom alla flikar delar session_state läser Copiloten samma objekt som
granskningsfliken skriver: en ändrad gruvlivslängd syns här direkt, före Spara.

Rena funktioner: allt tar data som argument. Ingen Streamlit.
"""

from __future__ import annotations

from typing import Optional

import controls as ctl
import insider as ins
import producers as prod
import scoring as sco
import tiggre as tig

PASS, MANUAL, FAIL = "PASS", "MANUAL", "FAIL"

# Vilket lager varje strategi granskas i, och vad arket heter i panelen.
SHEET = {
    "rule": ("producers", "Rick Rule"),
    "royalty": ("producers", "Royalty C"),
    "sprott": ("scoring", "Poängmodellen (Sprott)"),
    "durrett": ("scoring", "Poängmodellen (Durrett)"),
    "tiggre": ("tiggre", "Tiggre-arket"),
    "insider": ("insider", "Insiderbevakaren"),
}

STORE_DEFAULTS = {
    "producers": {"producers": [], "royalty": []},
    "scoring": {"sprott": [], "durrett": []},
    "tiggre": {"candidates": [], "positions": [], "closed": [], "parked": []},
    "insider": {"signals": []},
}


def has_review(strategy: str) -> bool:
    return (strategy or "").strip().lower() in SHEET


def sheet_name(strategy: str) -> str:
    return SHEET.get((strategy or "").strip().lower(), ("", ""))[1]


def _rows(strategy: str, stores: dict) -> list:
    """Raderna i det ark strategin granskas i."""
    key = (strategy or "").strip().lower()
    store_name, _label = SHEET.get(key, (None, None))
    data = (stores or {}).get(store_name) or {}
    if key == "rule":
        return data.get("producers", []) or []
    if key == "royalty":
        return data.get("royalty", []) or []
    if key in ("sprott", "durrett"):
        return data.get(key, []) or []
    if key == "tiggre":
        # kandidater först, men ett bolag som redan är position hittas också
        return (data.get("candidates", []) or []) + (data.get("positions", []) or [])
    if key == "insider":
        return data.get("signals", []) or []
    return []


def find_row(strategy: str, ticker: str, stores: dict) -> Optional[dict]:
    name = (ticker or "").strip().upper()
    if not name:
        return None
    for row in _rows(strategy, stores):
        if isinstance(row, dict) and (row.get("ticker") or "").upper() == name:
            return row
    return None


# ── Arkens egna utfall, översatta ────────────────────────────────────────────
def _rule_review(row: dict) -> tuple:
    score = prod.producer_score(row)
    dying = prod.asset_dying(row)
    vd = prod.producer_verdict(score, dying)
    if vd is None:
        return (MANUAL, "Poängen går inte att räkna — fyll i råvarupris och "
                        "kostnad per enhet i arket.")
    note = f"{score}/{prod.PROD_MAX_SCORE} p — {vd.label}. {vd.why}"
    if dying:
        return (FAIL, note)
    if vd.label == prod.P_BUY:
        return (PASS, note)
    return (MANUAL if vd.label == prod.P_WATCH else FAIL, note)


def _royalty_review(row: dict) -> tuple:
    vd = prod.royalty_signal(row)
    note = f"{vd.label}. {vd.why}"
    if vd.label == prod.R_BUY:
        return (PASS, note)
    if vd.label == prod.R_GEO_WARN:
        return (FAIL, note)
    return (MANUAL, note)          # Nära botten eller Neutral — beslut kvar


def _scoring_review(row: dict) -> tuple:
    score = sco.total_score(row.get("factors", {}))
    vd = sco.verdict(score)
    if vd is None:
        return (MANUAL, "Ej poängsatt — sätt de fem faktorerna i "
                        "Poängmodellen.")
    note = f"{score}/{sco.MAX_SCORE} p — {vd}"
    if vd == sco.CORE:
        return (PASS, note)
    return (MANUAL if vd == sco.WATCH else FAIL, note)


def _tiggre_review(row: dict) -> tuple:
    gates = tig.buy_gates(row)
    failed = [f"{label} ({detail})" for label, ok, detail in gates if not ok]
    if not failed:
        return (PASS, "Alla köpgrindar gröna: "
                      + " · ".join(label for label, _ok, _d in gates))
    # Arkets egen regel: köp ENDAST när alla är gröna. En röd grind är
    # därför ett nej här, inte en fråga — oavsett om orsaken är ett värde
    # eller ett tomt fält, för Tiggres grindar är skrivna så att tomt = rött.
    return (FAIL, "Röda köpgrindar: " + " · ".join(failed))


def _insider_review(row: dict) -> tuple:
    stat = ins.status(row)
    pts = ins.score(row)
    if stat is None:
        return (MANUAL, "Ofullständig signal — antal insiders, roll och "
                        "belopp måste alla vara ifyllda.")
    note = f"{pts}/10 p — {stat}"
    if stat == ins.S_BUY:
        return (PASS, note)
    if stat in (ins.S_NOISE, ins.S_GATE_FAIL):
        return (FAIL, note)
    return (MANUAL, note)          # Bevaka / kör grinden / väntar på trigger


_REVIEWERS = {
    "rule": _rule_review,
    "royalty": _royalty_review,
    "sprott": _scoring_review,
    "durrett": _scoring_review,
    "tiggre": _tiggre_review,
    "insider": _insider_review,
}


# ── Kontrollerna (DS/AQS/CSM) ur radens egna fält ────────────────────────────
def control_findings(row: dict, strategy: str) -> list:
    """[(status, etikett, notering)] — samma luck-regel som scorecardet,
    fast mjukare: en obedömd kontroll är MANUAL här, inte ett nej. Det hårda
    nejet hör hemma i Master Scorecard; Copiloten är snabbvyn.

    Hårda flaggor är dock hårda överallt: DS-låset, CSM:s Bear-flagga och
    AQS i PASS-bandet är arkens egna spärrar och blir FAIL.
    """
    r = row or {}
    required = ctl.required_sections(r.get("position_pct"), strategy,
                                     bool(r.get("dilution_risk")))
    out = []

    ds = ctl.ds_total(r)
    if ctl.ds_blocks_buy(r):
        out.append((FAIL, "DS", ctl.ds_note(r)))
    elif ds is not None:
        out.append((PASS, "DS", f"{ds}/{ctl.DS_MAX} — {ctl.ds_band(ds)}"))
    elif ctl.SEC_DS in required:
        out.append((MANUAL, "DS", "Ej bedömd — fylls i på kortet i arket."))

    aqs = ctl.aqs_total(r)
    if aqs is not None:
        band = ctl.aqs_band(aqs)
        out.append((FAIL if band == ctl.AQS_PASS else PASS, "AQS",
                    f"{aqs}/{ctl.AQS_MAX} — {band}"))
    elif ctl.SEC_AQS in required:
        out.append((MANUAL, "AQS", "Ej bedömd — krävs för positioner över "
                                   f"{ctl.FULL_WORK_MIN_PCT:g} % av totalen."))

    if ctl.csm_red_flag(r.get("csm_kind", ctl.PRODUCER), r.get("csm", {}),
                        bool(r.get("secured_cash"))):
        out.append((FAIL, "CSM", ctl.CSM_BEAR_FAIL))
    elif ctl.csm_complete(r.get("csm", {}), bool(r.get("is_core"))):
        out.append((PASS, "CSM", "Alla scenarier ifyllda, Bear överlevs."))
    elif ctl.SEC_CSM in required:
        out.append((MANUAL, "CSM", "Ej ifylld för alla scenarier."))

    return out


# ── Sammanställningen ────────────────────────────────────────────────────────
def review(strategy: str, ticker: str, stores: dict) -> Optional[dict]:
    """Granskningens utfall för kandidaten, eller None när strategin saknar ark.

    Ett bolag som inte finns i arket är MANUAL med en instruktion — Copiloten
    pekar mot granskningen, den ersätter den inte.
    """
    key = (strategy or "").strip().lower()
    if key not in SHEET:
        return None
    label = sheet_name(key)
    row = find_row(key, ticker, stores)
    if row is None:
        return {"found": False, "sheet": label, "row": None,
                "status": MANUAL, "controls": [],
                "note": f"{(ticker or '?').upper()} finns inte i {label} — "
                        f"lägg in och granska bolaget där först. Ordningen är "
                        f"screener → granskning → Copilot."}
    status, note = _REVIEWERS[key](row)
    return {"found": True, "sheet": label, "row": row,
            "status": status, "note": note,
            "controls": control_findings(row, key)}


def prompt_lines(rev: Optional[dict]) -> list:
    """Granskningen som rader till AI-prompten."""
    if rev is None:
        return []
    if not rev["found"]:
        return [f"Granskning ({rev['sheet']}): SAKNAS — {rev['note']}"]
    out = [f"Granskning ({rev['sheet']}): {rev['status']} — {rev['note']}"]
    for status, label, note in rev["controls"]:
        out.append(f"  {label}: {status} — {note}")
    return out
