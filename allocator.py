"""
allocator.py — Portföljallokeraren (Masterguiden Del 2).

The layer above every strategy. Strategies decide what to buy; this decides how
much may exist at all — and it is where the guide says the hard decisions are
made in advance:

  Fördelningsmodellen  — mål och ram per strategi (larmar utanför ram)
  De två taken         — råvarutaket 55 % och positionstak per bolag
  Kassaregeln          — under 5 % fylls vid nästa försäljning, över 25 % i mer
                         än ett kvartal sänks ribban
  Strömbrytaren        — NORMAL / SKÄRPT / HALVERAD RISK efter fall från topp
  Nytt kapital         — går alltid till strategin längst under sitt mål

One rule from the guide is deliberately encoded as a warning and never as an
action: "ombalanseringen överprövar ALDRIG strategiernas egna säljregler". Stops
and exits execute immediately; rebalancing only distributes what they free up.

Percentages are of the equity portfolio (the buffer sits outside it).
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from datetime import date
from typing import Optional

import csv_export
import storage
import storage_ui

_CACHE_KEY = "allocator_data"
STORE = "allocator"   # data/allocator.json

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF"
BG_CARD, BORDER = "#14141e", "#2a2a38"


@dataclass(frozen=True)
class Sleeve:
    key: str
    name: str
    target: float          # mål i %
    lo: float              # ram, undre
    hi: float              # ram, övre
    role: str
    commodity: bool        # räknas mot råvarutaket
    position_cap: Optional[float] = None   # positionstak per bolag, % av total
    cap_note: str = ""


# ── Fördelningsmodellen (Masterguiden Del 2) ─────────────────────────────────
SLEEVES: tuple[Sleeve, ...] = (
    Sleeve("royalty", "Royaltykärnan (nivå 1)", 20, 15, 30,
           "Stabilisatorn — ägs genom cykeln", True, 10.0,
           "Royaltykärnan får glida till 12 % innan den trimmas"),
    Sleeve("producenter", "Producenter (Rule/rotation)", 15, 10, 25,
           "Kontrariska kassaflödesbolag", True, 4.0),
    Sleeve("optionalitet", "Optionalitet (Sprott + Tiggre)", 7, 0, 12,
           "Lottsedlar & sweet spot — hård gräns", True, 4.0,
           "Sprott 1,5 % · Tiggre 2–4 %"),
    Sleeve("durrett", "Durrett (guld/silver)", 8, 0, 15,
           "Hävstången mot ädelmetallcykeln", True, 3.0),
    Sleeve("swing", "Momentum-swing (Norden)", 20, 10, 30,
           "Okorrelerad motor", False, 6.0),
    Sleeve("insider", "Insider (Norden)", 20, 10, 30,
           "Andra okorrelerade motorn", False, 4.0),
    Sleeve("kassa", "Kassa", 10, 5, 25,
           "Ammunition — växer när inget är billigt", False),
)

SLEEVE_BY_KEY = {s.key: s for s in SLEEVES}

# ── Positionsregeln — två nivåer (Masterguiden 4.0) ──────────────────────────
# "NORMAL POSITION anges inom strategidelen, HÅRT TAK mot hela portföljen."
# De blandas aldrig ihop: 20 % av en optionalitetsdel som är 7 % av portföljen
# är 1,4 % av totalen, och det är totalen taket mäts mot.


@dataclass(frozen=True)
class PositionRule:
    key: str
    name: str
    sleeve: str
    normal_lo: float       # % av strategidelen
    normal_hi: float
    hard_cap: float        # % av hela portföljen — bryts aldrig


POSITION_RULES: tuple[PositionRule, ...] = (
    PositionRule("royalty1", "Royalty nivå 1", "royalty", 5, 10, 10.0),
    PositionRule("rule", "Rule-producent", "producenter", 5, 10, 4.0),
    PositionRule("sprott", "Sprott", "optionalitet", 10, 20, 1.5),
    PositionRule("tiggre", "Tiggre", "optionalitet", 15, 30, 4.0),
    PositionRule("durrett", "Durrett", "durrett", 10, 20, 3.0),
    PositionRule("swing", "Swing", "swing", 15, 30, 6.0),
    PositionRule("insider", "Insider", "insider", 10, 20, 4.0),
)
RULE_BY_KEY = {r.key: r for r in POSITION_RULES}

# Sleeves med exakt en regel kan härledas ur en gammal position; optionaliteten
# kan inte, eftersom Sprott och Tiggre har olika tak (1,5 % mot 4 %).
_RULES_PER_SLEEVE = {}
for _r in POSITION_RULES:
    _RULES_PER_SLEEVE.setdefault(_r.sleeve, []).append(_r)
AMBIGUOUS_SLEEVES = tuple(k for k, v in _RULES_PER_SLEEVE.items() if len(v) > 1)

COMMODITY_CAP = 55.0        # råvarutaket: royalty + producenter + optionalitet + durrett
COMMODITY_WARN = 50.0       # arket varnar redan här — taket ska aldrig nås oplanerat
CASH_LOW, CASH_HIGH = 5.0, 25.0
ROYALTY_DRIFT_CAP = 12.0    # kärnan får glida hit innan trimning
POSITION_WARN_FRAC = 0.9    # "nära taket" när positionen passerat 90 % av det

# Positions- och råvarulägen
POS_OK, POS_NEAR, POS_OVER = "inom tak", "nära taket", "över tak"

# ── Strömbrytaren ────────────────────────────────────────────────────────────
BREAKER_LEVELS = (
    (0.0, 10.0, "NORMAL", GREEN,
     "Alla strategier enligt plan. Normalt brus."),
    (10.0, 20.0, "SKÄRPT", AMBER,
     "Inga nya köp i strategin som driver nedgången. Halv positionsstorlek på "
     "övriga nya köp. Extra journalkoll: marknaden eller regelbrott?"),
    (20.0, float("inf"), "HALVERAD RISK", RED,
     "Halvera antalet positioner i swing + insider (behåll de starkaste). Inga "
     "nya optionalitetsköp. Royaltykärna + kassa röres ej. Hävs när halva fallet "
     "återtagits ELLER genomgång visar följda regler + marknadsbred nedgång."),
)


# ── Rena beräkningar ─────────────────────────────────────────────────────────
def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def sleeve_pct(values: dict, total: Optional[float] = None) -> dict:
    """Andel av portföljen per sleeve. total=None -> summan av värdena."""
    vals = {s.key: max(0.0, _num(values.get(s.key), 0.0) or 0.0) for s in SLEEVES}
    t = _num(total) if total else sum(vals.values())
    if not t or t <= 0:
        return {k: 0.0 for k in vals}
    return {k: v / t * 100 for k, v in vals.items()}


def sleeve_status(key: str, pct: float) -> tuple[str, str]:
    """(status, åtgärd) mot sleevens ram. Utanför ram -> tillbaka till målet."""
    s = SLEEVE_BY_KEY.get(key)
    if s is None:
        return "okänd", ""
    if pct < s.lo:
        return "under ram", f"Höj mot målet {s.target:g} % — nytt kapital hit först"
    if pct > s.hi:
        return "över ram", f"Trimma tillbaka mot målet {s.target:g} %"
    return "inom ram", "Rör inget"


def commodity_exposure(values: dict, total: Optional[float] = None) -> float:
    """Råvarutaket: royalty + producenter + optionalitet + durrett.

    Avrundad av samma skäl som nedgången i drawdown_pct: en portfölj som är
    exakt 55 % råvara summerar till 55.000000000000007 i binär flyttalsform,
    och taket säger "högst 55 %" — inte "under 55 %".
    """
    pcts = sleeve_pct(values, total)
    return round(sum(pcts[s.key] for s in SLEEVES if s.commodity), 6)


def commodity_breach(values: dict, total: Optional[float] = None) -> bool:
    return commodity_exposure(values, total) > COMMODITY_CAP


def commodity_state(values: dict, total: Optional[float] = None) -> tuple[str, str]:
    """(läge, text) för råvarutaket — med förvarning innan taket bryts.

    Arket varnar från 50 % just för att 55 % inte ska nås av misstag: när alla
    kontrariska ben laddas samtidigt går exponeringen fort.
    """
    exp = commodity_exposure(values, total)
    if exp > COMMODITY_CAP:
        return POS_OVER, (f"Råvaruexponeringen är {exp:.1f} % — över taket "
                          f"{COMMODITY_CAP:g} %. Trimma ner.")
    if exp >= COMMODITY_WARN:
        return POS_NEAR, (f"Råvaruexponeringen är {exp:.1f} % — närmar sig taket "
                          f"{COMMODITY_CAP:g} %. Nya råvaruköp kräver att något "
                          f"annat minskar.")
    return POS_OK, f"Råvaruexponering {exp:.1f} % av {COMMODITY_CAP:g} %."


def position_rule(row: dict) -> Optional[PositionRule]:
    """Regeln för en position. None när den inte går att härleda.

    En gammal position lagrar bara sin sleeve. För sex av sju sleeves finns
    bara en regel och den kan härledas; optionaliteten kan inte, och att gissa
    där vore att välja mellan 1,5 % och 4 % åt användaren.
    """
    r = row or {}
    key = (r.get("rule") or "").strip().lower()
    if key in RULE_BY_KEY:
        return RULE_BY_KEY[key]
    sleeve = (r.get("sleeve") or "").strip().lower()
    candidates = _RULES_PER_SLEEVE.get(sleeve, [])
    return candidates[0] if len(candidates) == 1 else None


def normal_pct(value, sleeve_value) -> Optional[float]:
    """Positionens andel av sin egen strategidel."""
    v, sv = _num(value), _num(sleeve_value)
    if v is None or sv is None or sv <= 0:
        return None
    return round(v / sv * 100, 6)


def normal_state(pct_of_sleeve: Optional[float],
                 rule: Optional[PositionRule]) -> tuple[str, str]:
    """NORMAL-intervallet inom strategidelen — vägledning, inte tak."""
    if rule is None or pct_of_sleeve is None:
        return "okänd", ""
    if pct_of_sleeve < rule.normal_lo:
        return "under normal", (f"Under {rule.normal_lo:g} % av "
                                f"{rule.sleeve}-delen — får fyllas på")
    if pct_of_sleeve > rule.normal_hi:
        return "över normal", (f"Över {rule.normal_hi:g} % av "
                               f"{rule.sleeve}-delen — dubbelsignal tillåter "
                               f"övre delen, inte mer")
    return "normal", "Inom normalintervallet"


def position_state(pct: float, cap: Optional[float]) -> str:
    """inom tak / nära taket / över tak för en enskild position."""
    c = _num(cap)
    p = _num(pct, 0.0) or 0.0
    if c is None or c <= 0:
        return POS_OK
    if p > c:
        return POS_OVER
    return POS_NEAR if p >= c * POSITION_WARN_FRAC else POS_OK


def cash_rule(pct_cash: float, quarters_high: int = 0) -> tuple[str, str]:
    """Kassaregeln. Returnerar (status, åtgärd)."""
    p = _num(pct_cash, 0.0) or 0.0
    if p < CASH_LOW:
        return "låg", ("Under 5 % — nästa försäljning fyller kassan")
    if p > CASH_HIGH and quarters_high >= 1:
        return "hög", ("Över 25 % i mer än ett kvartal — sänk medvetet ribban "
                       "en nivå i bevakningslistorna")
    if p > CASH_HIGH:
        return "hög", ("Över 25 % — håll koll. Kvarstår det ett kvartal gäller "
                       "regeln nedan")
    return "ok", "Inom ram"


def drawdown_pct(peak: float, current: float) -> Optional[float]:
    """Fall från topp i procent (positivt tal)."""
    p, c = _num(peak), _num(current)
    if p is None or c is None or p <= 0:
        return None
    # Rounded so the breaker boundaries are stable: an exact 20 % fall computes
    # to 19.999999999999996 in binary floating point, which would leave the
    # portfolio in SKÄRPT when the guide says HALVERAD RISK.
    return round(max(0.0, (1 - c / p) * 100), 6)


def breaker_state(dd: Optional[float]) -> tuple[str, str, str]:
    """(läge, färg, åtgärd) ur fallet från topp."""
    if dd is None:
        return "OKÄND", DIM, "Ange portföljens topp och nuvarande värde."
    for lo, hi, label, color, action in BREAKER_LEVELS:
        if lo <= dd < hi:
            return label, color, action
    return "OKÄND", DIM, ""


def next_capital_target(values: dict, total: Optional[float] = None) -> Optional[Sleeve]:
    """Nytt kapital går alltid till strategin längst under sitt mål."""
    pcts = sleeve_pct(values, total)
    gaps = [(s.target - pcts[s.key], s) for s in SLEEVES if s.key != "kassa"]
    gaps = [g for g in gaps if g[0] > 0]
    if not gaps:
        return None
    return max(gaps, key=lambda g: g[0])[1]


def position_breaches(positions: list, total: Optional[float] = None) -> list:
    """Bolag över sitt positionstak. positions: [{ticker, sleeve, value}]."""
    t = _num(total) or sum(max(0.0, _num(p.get("value"), 0.0) or 0.0) for p in positions)
    out = []
    if not t or t <= 0:
        return out
    for p in positions:
        rule = position_rule(p)
        s = SLEEVE_BY_KEY.get((rule.sleeve if rule else p.get("sleeve", "")))
        cap = rule.hard_cap if rule else (s.position_cap if s else None)
        if cap is None:
            continue
        pct = round((max(0.0, _num(p.get("value"), 0.0) or 0.0)) / t * 100, 6)
        # The royalty core is allowed to drift before it is trimmed.
        effective = ROYALTY_DRIFT_CAP if (s and s.key == "royalty") else cap
        if pct > effective:
            out.append({"ticker": p.get("ticker", "?"),
                        "sleeve": s.name if s else "?",
                        "rule": rule.name if rule else None,
                        "pct": pct, "cap": cap, "effective": effective})
    return out


def unresolved_positions(positions: list) -> list:
    """Positioner vars tak inte går att avgöra — Sprott eller Tiggre?

    De rapporteras hellre än att tilldelas det lösaste taket: skillnaden är
    1,5 % mot 4 %, alltså mer än en faktor två i tillåten storlek.
    """
    return [p for p in positions or []
            if position_rule(p) is None
            and (p.get("sleeve") or "").strip().lower() in AMBIGUOUS_SLEEVES]


# ── Lagring ──────────────────────────────────────────────────────────────────
def _default() -> dict:
    return {"values": {s.key: 0.0 for s in SLEEVES}, "positions": [],
            "peak": 0.0, "current": 0.0, "quarters_cash_high": 0,
            "updated": ""}


def _load() -> dict:
    """Laddas EN gång per session; därefter äger sessionen sanningen."""
    data = storage.session_load(STORE, _default(),
                                legacy_file="allocator_data.json")
    if not isinstance(data, dict):
        data = _default()
        st.session_state[STORE] = data
    for k, v in _default().items():
        data.setdefault(k, v)
    return data


def _save(data: dict) -> None:
    """Skriver till sessionen. Persistensen sker via 💾 Spara.

    Medvetet ingen nätverkstrafik här: en commit per
    tangenttryckning skulle slå i GitHubs rate limits, och en
    tyst misslyckad sådan var precis den bugg det här ersätter.
    """
    st.session_state[STORE] = data


# ── UI ───────────────────────────────────────────────────────────────────────
def render_allocator_page() -> None:
    """Huvud-entry point för Portföljallokeraren."""
    data = _load()
    storage_ui.save_bar(STORE, "Portföljallokeraren")
    try:
        st.markdown(
            f"<h1 style='color:{TEXT};margin:0;letter-spacing:0.06em;'>"
            f"Portföljallokeraren <span style='color:{CYAN};'>· ramarna</span></h1>"
            f"<p style='color:{DIM};font-size:0.8rem;margin:6px 0 14px;'>"
            f"Strategierna bestämmer VAD du köper — det här bestämmer hur mycket som "
            f"får finnas. Procenten avser aktieportföljen; bufferten ligger utanför."
            f"</p>", unsafe_allow_html=True)

        _export(data)
        _breaker(data)
        _allocation(data)
        _caps(data)
        _positions(data)
    except Exception as e:
        st.error(f"Portföljallokeraren kunde inte renderas: {e}")


SLEEVE_CSV = [("_name", "Strategi"), ("_value", "Värde"), ("_pct", "Andel %"),
              ("_target", "Mål %"), ("_lo", "Ram låg"), ("_hi", "Ram hög"),
              ("_status", "Status"), ("_action", "Åtgärd"),
              ("_cap", "Positionstak %")]

POS_CSV = [("ticker", "Ticker"), ("_rule", "Typ"), ("sleeve", "Strategi"),
           ("value", "Värde"), ("_pct", "Andel av total %"),
           ("_of_sleeve", "Andel av delen %"), ("_normal", "Normalintervall"),
           ("_cap", "Tak %"), ("_state", "Läge")]


def _export(data: dict) -> None:
    vals = data.get("values", {})
    pcts = sleeve_pct(vals)
    sleeve_rows = []
    for s in SLEEVES:
        st_, action = sleeve_status(s.key, pcts[s.key])
        sleeve_rows.append({
            "_name": s.name, "_value": vals.get(s.key), "_pct": round(pcts[s.key], 1),
            "_target": s.target, "_lo": s.lo, "_hi": s.hi, "_status": st_,
            "_action": action, "_cap": s.position_cap})

    positions = data.get("positions", [])
    total = sum(max(0.0, _num(v, 0.0) or 0.0) for v in vals.values())
    total = total or sum(max(0.0, _num(p.get("value"), 0.0) or 0.0)
                         for p in positions)
    pos_rows = []
    for p in positions:
        rule = position_rule(p)
        s = SLEEVE_BY_KEY.get((rule.sleeve if rule else p.get("sleeve", "")))
        pct = ((max(0.0, _num(p.get("value"), 0.0) or 0.0) / total * 100)
               if total else 0.0)
        cap = rule.hard_cap if rule else (s.position_cap if s else None)
        eff = (ROYALTY_DRIFT_CAP if s and s.key == "royalty" else cap)
        of_sleeve = normal_pct(p.get("value"),
                               _num(vals.get(s.key), 0.0) if s else None)
        pos_rows.append({**p, "_rule": rule.name if rule else "OKÄND",
                         "_pct": round(pct, 1),
                         "_of_sleeve": None if of_sleeve is None else round(of_sleeve, 1),
                         "_normal": (f"{rule.normal_lo:g}–{rule.normal_hi:g} %"
                                     if rule else ""),
                         "_cap": eff, "_state": position_state(pct, eff)})

    c1, c2 = st.columns(2)
    with c1:
        csv_export.download_button(sleeve_rows, SLEEVE_CSV, "allokering",
                                   label="⬇ Allokering (CSV)", key="csv_alloc")
    with c2:
        csv_export.download_button(pos_rows, POS_CSV, "positioner",
                                   label="⬇ Positioner (CSV)", key="csv_alloc_pos")


def _breaker(data: dict) -> None:
    st.markdown(f"<div style='color:{DIM};font-size:0.7rem;letter-spacing:0.12em;"
                f"margin-bottom:6px;'>STRÖMBRYTAREN — PORTFÖLJENS STOP LOSS</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2.4])
    peak = c1.number_input("Topp (SEK)", min_value=0.0,
                           value=float(_num(data.get("peak"), 0.0) or 0.0),
                           step=10000.0, key="al_peak")
    cur = c2.number_input("Nuvarande (SEK)", min_value=0.0,
                          value=float(_num(data.get("current"), 0.0) or 0.0),
                          step=10000.0, key="al_cur")
    if peak != data.get("peak") or cur != data.get("current"):
        data["peak"], data["current"] = peak, cur
        _save(data)

    dd = drawdown_pct(peak, cur)
    label, color, action = breaker_state(dd)
    with c3:
        st.markdown(
            f"<div style='background:{color}15;border:1px solid {color}66;"
            f"border-radius:8px;padding:10px 14px;'>"
            f"<span style='color:{color};font-weight:800;font-size:1.05rem;'>"
            f"{label}</span>"
            f"<span style='color:{DIM};font-size:0.8rem;margin-left:10px;'>"
            f"{'fall från topp ' + format(dd, '.1f') + ' %' if dd is not None else ''}"
            f"</span>"
            f"<div style='color:{TEXT};font-size:0.8rem;margin-top:4px;'>{action}</div>"
            f"</div>", unsafe_allow_html=True)
    st.caption("Halvera i stället för att sälja allt: strategiernas bästa köplägen "
               "uppstår exakt i breda nedgångar. Strömbrytaren hindrar förblödning "
               "— inte bottenfiske.")


def _allocation(data: dict) -> None:
    st.markdown(f"<div style='color:{DIM};font-size:0.7rem;letter-spacing:0.12em;"
                f"margin:18px 0 6px;'>FÖRDELNINGSMODELLEN</div>",
                unsafe_allow_html=True)

    vals = data["values"]
    changed = False
    cols = st.columns(4)
    for i, s in enumerate(SLEEVES):
        v = cols[i % 4].number_input(s.name, min_value=0.0,
                                     value=float(_num(vals.get(s.key), 0.0) or 0.0),
                                     step=10000.0, key=f"al_v_{s.key}")
        if v != vals.get(s.key):
            vals[s.key] = v
            changed = True
    if changed:
        _save(data)

    total = sum(max(0.0, _num(v, 0.0) or 0.0) for v in vals.values())
    if total <= 0:
        st.info("Fyll i värdet per strategi ovan så räknas ramar, tak och "
                "ombalansering automatiskt.")
        return

    pcts = sleeve_pct(vals)
    rows = ""
    for s in SLEEVES:
        p = pcts[s.key]
        status, action = sleeve_status(s.key, p)
        c = {"under ram": AMBER, "över ram": RED, "inom ram": GREEN}.get(status, DIM)
        width = min(p / max(s.hi, 1) * 100, 100)
        rows += (
            f"<div style='margin-bottom:9px;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.78rem;'>"
            f"<span style='color:{TEXT};'>{s.name} "
            f"<span style='color:{DIM};font-size:0.7rem;'>mål {s.target:g} % · "
            f"ram {s.lo:g}–{s.hi:g} %</span></span>"
            f"<span style='color:{c};font-weight:700;'>{p:.1f} % · {status}</span></div>"
            f"<div style='background:#2a2a38;border-radius:4px;height:7px;margin-top:3px;'>"
            f"<div style='width:{width:.0f}%;background:{c};height:7px;"
            f"border-radius:4px;'></div></div>"
            f"<div style='color:{DIM};font-size:0.68rem;margin-top:2px;'>{action}</div>"
            f"</div>")
    st.markdown(rows, unsafe_allow_html=True)

    nxt = next_capital_target(vals)
    if nxt:
        st.markdown(
            f"<div style='background:{CYAN}11;border:1px solid {CYAN}44;"
            f"border-radius:6px;padding:8px 14px;font-size:0.82rem;color:{TEXT};'>"
            f"💧 <b>Nytt kapital →</b> {nxt.name} "
            f"<span style='color:{DIM};'>(längst under sitt mål — gratis "
            f"ombalansering utan att sälja)</span></div>", unsafe_allow_html=True)


def _caps(data: dict) -> None:
    vals = data["values"]
    total = sum(max(0.0, _num(v, 0.0) or 0.0) for v in vals.values())
    if total <= 0:
        return
    st.markdown(f"<div style='color:{DIM};font-size:0.7rem;letter-spacing:0.12em;"
                f"margin:18px 0 6px;'>DE TVÅ TAKEN — VIKTIGARE ÄN MÅLEN</div>",
                unsafe_allow_html=True)

    exp = commodity_exposure(vals)
    state, note = commodity_state(vals)
    c = {POS_OVER: RED, POS_NEAR: AMBER}.get(state, GREEN)
    detail = ("ÖVER TAKET — nytt kapital går till Norden-delen eller kassan, "
              "oavsett hur billigt något ser ut." if state == POS_OVER else note)
    st.markdown(
        f"<div style='background:{c}11;border:1px solid {c}55;border-radius:8px;"
        f"padding:10px 14px;margin-bottom:8px;'>"
        f"<span style='color:{c};font-weight:700;'>Råvarutaket "
        f"{exp:.1f} % / {COMMODITY_CAP:g} %</span>"
        f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>{detail}"
        f"</div></div>", unsafe_allow_html=True)

    cash_pct = sleeve_pct(vals)["kassa"]
    status, action = cash_rule(cash_pct, int(_num(data.get("quarters_cash_high"), 0) or 0))
    cc = {"låg": AMBER, "hög": AMBER}.get(status, GREEN)
    st.markdown(
        f"<div style='background:{cc}11;border:1px solid {cc}55;border-radius:8px;"
        f"padding:10px 14px;'>"
        f"<span style='color:{cc};font-weight:700;'>Kassaregeln {cash_pct:.1f} %</span>"
        f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>{action}</div>"
        f"</div>", unsafe_allow_html=True)


def _positions(data: dict) -> None:
    st.markdown(f"<div style='color:{DIM};font-size:0.7rem;letter-spacing:0.12em;"
                f"margin:18px 0 6px;'>POSITIONSTAK PER BOLAG</div>",
                unsafe_allow_html=True)

    st.caption("Två nivåer som aldrig blandas ihop: NORMAL anges inom "
               "strategidelen, HÅRT TAK mot hela portföljen. Dubbelsignal "
               "tillåter övre delen av NORMAL — taket bryts aldrig, av "
               "något skäl.")
    caps = " · ".join(f"{r.name}: {r.normal_lo:g}–{r.normal_hi:g} % av delen, "
                      f"tak {r.hard_cap:g} %" for r in POSITION_RULES)
    st.caption(caps)

    a1, a2, a3, a4 = st.columns([1.2, 1.6, 1.2, 0.8])
    tkr = a1.text_input("Ticker", key="al_p_tkr")
    rule_key = a2.selectbox("Typ", [r.key for r in POSITION_RULES],
                            format_func=lambda k: RULE_BY_KEY[k].name,
                            key="al_p_rule")
    val = a3.number_input("Värde (SEK)", min_value=0.0, value=0.0, step=10000.0,
                          key="al_p_val")
    if a4.button("Lägg till", key="al_p_add"):
        if tkr.strip() and val > 0:
            data["positions"].append({"ticker": tkr.upper().strip(),
                                      "rule": rule_key,
                                      "sleeve": RULE_BY_KEY[rule_key].sleeve,
                                      "value": val})
            _save(data)
            st.rerun()

    positions = data.get("positions", [])
    if not positions:
        st.caption("Inga positioner inlagda — lägg in dem för att kontrollera taken.")
        return

    total = sum(max(0.0, _num(v, 0.0) or 0.0) for v in data["values"].values())
    total = total or sum(max(0.0, _num(p.get("value"), 0.0) or 0.0) for p in positions)
    breaches = {b["ticker"] for b in position_breaches(positions, total)}

    for i, p in enumerate(list(positions)):
        pct = (max(0.0, _num(p.get("value"), 0.0) or 0.0) / total * 100) if total else 0
        rule = position_rule(p)
        s = SLEEVE_BY_KEY.get((rule.sleeve if rule else p.get("sleeve", "")))
        over = p.get("ticker") in breaches
        cap = rule.hard_cap if rule else (s.position_cap if s else None)
        eff = (ROYALTY_DRIFT_CAP if s and s.key == "royalty" else cap)
        state = POS_OVER if over else position_state(pct, eff)
        c = RED if over else (AMBER if state == POS_NEAR else TEXT)

        sleeve_value = _num(data["values"].get(s.key), 0.0) if s else None
        of_sleeve = normal_pct(p.get("value"), sleeve_value)
        nstate, nwhy = normal_state(of_sleeve, rule)

        r1, r2, r3, r4 = st.columns([1.2, 1.6, 1.2, 0.8])
        r1.markdown(f"<span style='color:{c};font-weight:700;'>{p.get('ticker','?')}"
                    f"</span>", unsafe_allow_html=True)
        r2.markdown(f"<span style='color:{DIM};font-size:0.8rem;'>"
                    f"{rule.name if rule else (s.name if s else '?')}</span>",
                    unsafe_allow_html=True)
        of_sleeve_txt = (f"<div style='color:{DIM};font-size:0.68rem;'>"
                         f"{of_sleeve:.0f} % av delen · normal "
                         f"{rule.normal_lo:g}–{rule.normal_hi:g} %</div>"
                         if of_sleeve is not None and rule else "")
        r3.markdown(f"<span style='color:{c};font-size:0.82rem;'>{pct:.1f} % "
                    f"<span style='color:{DIM};'>/ tak "
                    f"{cap:g} %</span></span>{of_sleeve_txt}" if cap
                    else f"{pct:.1f} %", unsafe_allow_html=True)
        if nstate == "över normal":
            st.caption(f"↳ {p.get('ticker','?')}: {nwhy}")
        if r4.button("✕", key=f"al_p_del_{i}"):
            data["positions"] = [x for j, x in enumerate(positions) if j != i]
            _save(data)
            st.rerun()

    for b in position_breaches(positions, total):
        st.error(f"{b['ticker']} är {b['pct']:.1f} % — över taket {b['cap']:g} % "
                 f"({b.get('rule') or b['sleeve']}). Trimma ner.")

    # Gamla positioner i optionaliteten: Sprott eller Tiggre? Taken skiljer
    # sig med mer än en faktor två, så de gissas inte.
    for p in unresolved_positions(positions):
        st.warning(f"{p.get('ticker','?')} ligger i optionaliteten utan typ — "
                   f"välj Sprott (tak {RULE_BY_KEY['sprott'].hard_cap:g} %) "
                   f"eller Tiggre (tak {RULE_BY_KEY['tiggre'].hard_cap:g} %). "
                   f"Tills dess kontrolleras den inte mot något tak.")

    # Förvarning: en position som passerat 90 % av taket ska inte fyllas på.
    for p in positions:
        s = SLEEVE_BY_KEY.get(p.get("sleeve", ""))
        if s is None or s.position_cap is None or p.get("ticker") in breaches:
            continue
        pct = (max(0.0, _num(p.get("value"), 0.0) or 0.0) / total * 100) if total else 0
        eff = ROYALTY_DRIFT_CAP if s.key == "royalty" else s.position_cap
        if position_state(pct, eff) == POS_NEAR:
            st.warning(f"{p.get('ticker','?')} är {pct:.1f} % — nära taket "
                       f"{eff:g} %. Fyll inte på.")

    st.caption("Ombalanseringen överprövar ALDRIG strategiernas egna säljregler — "
               "stoppar och exits exekveras omedelbart; ombalanseringen fördelar "
               "bara kapitalet de frigör.")
