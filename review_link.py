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


def _ja(value) -> str:
    return "Ja" if value else "Nej"


def _n(value, suffix: str = "", zero_empty: bool = False) -> str:
    """Talformat för promptraderna.

    zero_empty=True för nummerfälten: de ritas med min_value=0.0 och kan inte
    lämnas tomma, så en nolla där betyder EJ IFYLLT — samma semantik som
    storage.differs och producers._years. Faktorpoäng skickar False: där är
    0/2 ett riktigt betyg.
    """
    v = _numf(value)
    if v is None or (zero_empty and v == 0):
        return "–"
    return f"{v:g}{suffix}"


def _numf(value):
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def detail_lines(strategy: str, row: dict) -> list:
    """Granskningens IFYLLDA fält, i klartext.

    Utan de här raderna såg modellen bara summan ("PASS — 5/5 p") och bad
    användaren kontrollera landrisk, kostnadsposition, ledning och
    kapitaldisciplin — de fyra frågor som redan VAR besvarade i arket. Summan
    utan komponenterna är en inbjudan att fråga om komponenterna.
    """
    key = (strategy or "").strip().lower()
    r = row or {}
    out = []

    if key == "rule":
        m = prod.margin_pct(r.get("price"), r.get("unit_cost"))
        out.append(f"  Arkets fält: marginal "
                   f"{_n(round(m, 1) if m is not None else None, ' %')} "
                   f"(pris {_n(r.get('price'), zero_empty=True)} / kostnad "
                   f"{_n(r.get('unit_cost'), zero_empty=True)}) · EV/EBITDA "
                   f"{_n(r.get('ev_ebitda'), zero_empty=True)} · nettoskuld/EBITDA "
                   f"{_n(r.get('nd_ebitda'), zero_empty=True)} · gruvlivslängd "
                   f"{_n(r.get('mine_life'), ' år', zero_empty=True)} · R/P "
                   f"{_n(r.get('rp_ratio'), ' år', zero_empty=True)}")
        out.append(f"  Disciplinfrågorna (redan besvarade i arket): "
                   f"jurisdiktion {_ja(r.get('jurisdiktion'))} · "
                   f"kapitaldisciplin {_ja(r.get('kapitaldisciplin'))} · "
                   f"insynsägande {_ja(r.get('insyn'))} · tänkt position "
                   f"{_n(r.get('position_pct'), ' %', zero_empty=True)} av totalen")
    elif key == "royalty":
        disc = prod.discount_vs_bottom(r.get("pnav_now"), r.get("pnav_bottom"))
        med = prod.vs_median(r.get("ev_now"), r.get("ev_median"))
        geo = prod.geo_growth(r.get("geo_now"), r.get("geo_3y"))
        out.append(f"  Arkets fält: nivå {_n(r.get('level'))} · mot egen "
                   f"P/NAV-botten "
                   f"{_n(round(disc, 1) if disc is not None else None, ' %')} · "
                   f"mot egen EV/EBITDA-median "
                   f"{_n(round(med, 1) if med is not None else None, ' %')} · "
                   f"GEO/aktie-tillväxt 3 år "
                   f"{_n(round(geo, 1) if geo is not None else None, ' %')}")
    elif key in ("sprott", "durrett"):
        factors = r.get("factors", {}) or {}
        parts = [f"{f.label} {_n(factors.get(f.key))}/2"
                 for f in sco.FACTORS if _numf(factors.get(f.key)) is not None]
        if parts:
            out.append("  Faktorpoängen (redan satta i arket): "
                       + " · ".join(parts))
    elif key == "tiggre":
        up = tig.upside_pct(r.get("mcap"), r.get("nav"))
        un = tig.un_ratio(up, r.get("downside"))
        cats = [c for c in r.get("catalysts", [])
                if isinstance(c, dict) and c.get("name") and c.get("date")]
        out.append(f"  Arkets fält: U/N "
                   f"{_n(round(un, 1) if un is not None else None, ':1')} · "
                   f"{len(cats)} namngivna och tidsatta katalysatorer")
    elif key == "insider":
        out.append(f"  Signalens fält: {_n(r.get('insiders'), zero_empty=True)} insiders · "
                   f"roll {r.get('role') or '–'} · belopp "
                   f"{_n(r.get('amount'), ' MSEK', zero_empty=True)} · kurs mot klustersnitt "
                   f"{_n(round(ins.vs_cluster(r), 1) if ins.vs_cluster(r) is not None else None, ' %')}")

    # Kontrollernas komponenter — de svaga punkterna med namn, inte bara summan
    aqs_weak = [f.label for f in ctl.AQS_FIELDS
                if _numf(r.get(f.key)) == 0]
    if ctl.aqs_total(r) is not None and aqs_weak:
        out.append("  AQS svagast (0 p): " + " · ".join(aqs_weak))
    ds_extra = []
    for ikey, ilabel in ctl.DS_INFO_FIELDS:
        if _numf(r.get(ikey)):
            ds_extra.append(f"{ilabel} {_n(r.get(ikey))}")
    if ds_extra:
        out.append("  DS-underlag: " + " · ".join(ds_extra))
    lev = ctl.leverage_ratio(r.get("csm", {}))
    if lev is not None:
        out.append(f"  CSM hävstångskvot (Bull/Bear FCF): {lev:g}×")

    # Lukacs FV — bara när något är ifyllt
    try:
        import lukacs
        if r.get("fv"):
            ev = lukacs.evaluate(r)
            if ev["fv_base"] is not None:
                out.append(
                    f"  Lukacs FV: fair value Base "
                    f"{round(ev['fv_base'], 2):g}/aktie · säkerhetsmarginal "
                    f"{_n(round(ev['mos'], 1) if ev['mos'] is not None else None, ' %')} "
                    f"({ev['mos_band'] or '–'}) · klass "
                    f"{r.get('fcf_kvalitet') or '–'}")
    except Exception:
        pass
    return out


def prompt_lines(rev: Optional[dict], strategy: str = "") -> list:
    """Granskningen som rader till AI-prompten — verdiktet OCH underlaget."""
    if rev is None:
        return []
    if not rev["found"]:
        return [f"Granskning ({rev['sheet']}): SAKNAS — {rev['note']}"]
    out = [f"Granskning ({rev['sheet']}): {rev['status']} — {rev['note']}"]
    for status, label, note in rev["controls"]:
        out.append(f"  {label}: {status} — {note}")
    out += detail_lines(strategy or _strategy_of(rev), rev["row"])
    return out


def _strategy_of(rev: dict) -> str:
    """Strategin ur arknamnet, för anropare som inte skickar den."""
    sheet = (rev or {}).get("sheet", "")
    for key, (_store, label) in SHEET.items():
        if label == sheet:
            return key
    return ""
