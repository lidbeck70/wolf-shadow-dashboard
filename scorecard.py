"""
scorecard.py — Master Scorecard och köpgrinden (Masterguiden 4.0).

Sista steget före ett köp. Fliken skriver inget eget om kandidaterna: den
läser dem ur de flikar som redan gjort jobbet — Poängmodellen, Lobo,
Rick Rule, Royalty C, Insider — och sammanställer per bolag och ticker.
Det enda som matas in här är beslutet: position, värdering, trigger, säljregel
och de sju kryssen i köpgrinden.

Köpgrindens hårda regel: en lucka i tabellen är inte ett neutralt läge.

  "Luckor i tabellen = standardbeslut INGEN AFFÄR."

Därför blockerar obedömda kontroller här, till skillnad från i strategiernas
egna grindar där en tom DS bara läser "ej bedömd". Proportionalitetsregeln
avgör vilka kontroller som ens efterfrågas — en 1 %-lott ska inte kosta ett
kärninnehavs arbete.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import streamlit as st

import controls as ctl
import csv_export
import lukacs
import storage
import storage_ui

_CACHE_KEY = "scorecard_data"
STORE = "scorecard"   # data/scorecard.json

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"

NO_TRADE = "Luckor i tabellen = standardbeslut INGEN AFFÄR"
READY = "KLAR FÖR KÖP"

# Kontroller som inte påverkat ett beslut på ett år tas bort vid årsgenomgången.
STALE_DAYS = 365


# ── Köpgrinden ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Gate:
    key: str
    label: str
    help: str


GATES: tuple[Gate, ...] = (
    Gate("strategi_aktiv", "Strategin är aktiv",
         "Regimen tillåter strategin just nu, och strömbrytaren är inte slagen."),
    Gate("screener_kval", "Screenern kvalar",
         "Bolaget kom ur rätt screener, inte ur ett nyhetsflöde."),
    Gate("granskning_klar", "Granskningen är klar",
         "Strategipoängen är satt och räcker."),
    Gate("inga_roda_flaggor", "Kontrollerna utan permanent-risk-flagga",
         "AQS/DS/CSM enligt proportionalitetsregeln. Flaggan gäller just "
         "permanent kapitalförlust — utspädning, skuld, tillgångsförstörelse."),
    Gate("sakerhetsmarginal", "Värderingen ger säkerhetsmarginal",
         "Priset ger utrymme för att du har fel om en del av caset."),
    Gate("trigger_definierad", "Trigger definierad där strategin kräver det",
         "Vad ska hända, och när? Tiggre kräver det alltid; Rule gör det inte."),
    Gate("position_saljregel", "Position och säljregel satta",
         "Storleken och exitregeln bestämda före köpet, inte efter."),
)

GATE_BY_KEY = {g.key: g for g in GATES}


def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


# ── Sammanställningen ────────────────────────────────────────────────────────
def _key(name: str, ticker: str) -> str:
    """Nyckeln som binder ihop flikarna: bolag + ticker, normaliserad."""
    return f"{(name or '').strip().lower()}|{(ticker or '').strip().upper()}"


def collect(sources: dict) -> list:
    """Kandidater ur de andra flikarnas data, hopslagna på bolag + ticker.

    sources: {"sprott": [...], "durrett": [...], "tiggre": [...],
              "producenter": [...], "royalty": [...], "insider": [...]}
    Rena data in, rena data ut — ingen Streamlit, så den går att testa.
    """
    out = {}
    for strategy, rows in (sources or {}).items():
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            k = _key(row.get("name", ""), ticker)
            entry = out.setdefault(k, {
                "key": k, "ticker": ticker, "name": (row.get("name") or "").strip(),
                "strategies": [], "rows": {},
            })
            if strategy not in entry["strategies"]:
                entry["strategies"].append(strategy)
            entry["rows"][strategy] = row
    return sorted(out.values(), key=lambda e: (e["ticker"], e["name"]))


def source_row(entry: dict, strategy: Optional[str] = None) -> dict:
    """Raden som kontrollerna faktiskt ligger i."""
    rows = (entry or {}).get("rows", {})
    if strategy and strategy in rows:
        return rows[strategy] or {}
    for _s, r in rows.items():
        if r:
            return r
    return {}


def control_state(entry: dict, position_pct, strategy: str) -> dict:
    """DS/AQS/CSM för kandidaten, tillsammans med vad som faktiskt krävs."""
    row = source_row(entry, strategy)
    required = ctl.required_sections(position_pct, strategy,
                                     bool(row.get("dilution_risk")))
    ds = ctl.ds_total(row)
    aqs = ctl.aqs_total(row)
    csm_flag = ctl.csm_red_flag(row.get("csm_kind", ctl.PRODUCER),
                                row.get("csm", {}),
                                bool(row.get("secured_cash")))
    return {
        "required": required,
        "ds": ds, "ds_band": ctl.ds_band(ds), "ds_blocks": ctl.ds_blocks_buy(row),
        "aqs": aqs, "aqs_band": ctl.aqs_band(aqs),
        "csm_flag": csm_flag,
        "csm_complete": ctl.csm_complete(row.get("csm", {}),
                                         bool(row.get("is_core"))),
    }


def control_gaps(entry: dict, position_pct, strategy: str) -> list:
    """Kontroller som krävs men inte är ifyllda, plus röda flaggor.

    Det här är luck-regeln: en obedömd kontroll räknas som ett nej, inte som
    ett tyst ja.
    """
    st_ = control_state(entry, position_pct, strategy)
    req = st_["required"]
    gaps = []
    if ctl.SEC_DS in req:
        if st_["ds"] is None:
            gaps.append("DS är inte bedömd")
        elif st_["ds_blocks"]:
            gaps.append(f"DS {st_['ds']}/{ctl.DS_MAX} låser köpet — "
                        f"finansieringskatalysator saknas")
    if ctl.SEC_AQS in req:
        if st_["aqs"] is None:
            gaps.append("AQS är inte bedömd")
        elif st_["aqs_band"] == ctl.AQS_PASS:
            gaps.append(f"AQS {st_['aqs']}/{ctl.AQS_MAX} — {ctl.AQS_PASS}")
    if ctl.SEC_CSM in req:
        if not st_["csm_complete"]:
            gaps.append("CSM är inte ifylld för alla scenarier")
        if st_["csm_flag"]:
            gaps.append(ctl.CSM_BEAR_FAIL)
    return gaps


def valuation_gaps(entry: Optional[dict], card: dict) -> list:
    """Steg 5, säkerhetsmarginalen — mekaniskt där Lukacs FV krävs.

    Guiden gör modulen obligatorisk för producentköp över 2 % av totalen. Där
    är steg 5 inte längre ett omdöme utan en uträkning: MoS >= 25 % och ett
    ifyllt 'what must go right'. Utanför det området finns ingen FV att räkna
    på, och steget faller tillbaka på det manuella krysset.
    """
    c = card or {}
    pos, strategy = c.get("position_pct_total"), c.get("strategy", "")
    if not lukacs.fv_required(pos, strategy):
        return []
    if entry is None:
        return ["Kandidatdata saknas — fair value går inte att läsa"]
    return lukacs.gate_gaps(source_row(entry, strategy), pos, strategy)


def deleveraging_cap(entry: Optional[dict], card: dict) -> Optional[float]:
    """Positionstaket efter deleveraging-regeln, i % av totalen.

    None när regeln inte slår eller strategin saknar ett känt tak.
    """
    if entry is None:
        return None
    c = card or {}
    strategy = c.get("strategy", "")
    row = source_row(entry, strategy)
    cap = _strategy_cap(strategy)
    state = lukacs.deleveraging_state(row.get("nd_ebitda"),
                                      row.get("ar_till_lag_skuld"))
    if not state["half_position"] or cap is None:
        return None
    return lukacs.max_position_pct(row.get("nd_ebitda"), cap,
                                   row.get("ar_till_lag_skuld"))


def position_gaps(entry: Optional[dict], card: dict) -> list:
    """Att positionen är "satt" räcker inte — den måste rymmas i taket.

    Deleveraging-regeln halverar taket så fort skulden är över 1,0× vid köp.
    Utan den här kontrollen kan man kryssa "position bestämd" med en storlek
    som regeln förbjuder.
    """
    cap = deleveraging_cap(entry, card)
    pos = _num((card or {}).get("position_pct_total"))
    if cap is None or pos is None:
        return []
    if round(pos, 6) > round(cap, 6):
        return [f"Position {pos:g} % av totalen överstiger taket {cap:g} % — "
                f"skulden är över {lukacs.DELEV_ND_MIN:g}× och halverar "
                f"positionen."]
    return []


def _mechanical_gate(gate: Gate, ok: bool, gaps: list, note: str) -> None:
    """Ett grindsteg som räknas fram i stället för att kryssas."""
    st.markdown(
        f"<div style='color:{GREEN if ok else RED};font-size:0.86rem;"
        f"padding:3px 0;'>{'☑' if ok else '☐'} {gate.label} "
        f"<span style='color:{DIM};font-size:0.76rem;'>— {note}</span></div>",
        unsafe_allow_html=True)
    for gap in gaps:
        st.markdown(
            f"<div style='color:{RED};font-size:0.78rem;padding-left:22px;'>"
            f"• {gap}</div>", unsafe_allow_html=True)


def _strategy_cap(strategy: str) -> Optional[float]:
    """Strategins hårda positionstak ur allokeringsreglerna."""
    try:
        import allocator
    except Exception:                                    # pragma: no cover
        return None
    key = (strategy or "").strip().lower()
    caps = [r.hard_cap for r in allocator.POSITION_RULES if r.sleeve == key]
    return min(caps) if caps else None


def gate_state(card: dict, entry: Optional[dict] = None) -> dict:
    """Köpgrindens sju kryss plus kontrollernas luckor.

    Två av stegen kan inte kryssas förbi. Kontrollgrinden
    (inga_roda_flaggor) sätts av control_gaps, för det är precis den rutan man
    annars kryssar av vana. Säkerhetsmarginalen (sakerhetsmarginal) sätts av
    Lukacs FV där modulen är obligatorisk — över 2 % av totalen är steget en
    uträkning, inte ett omdöme.
    """
    c = card or {}
    if entry is None:
        # Ingen kandidatdata = inget att kontrollera mot. Det är en lucka, inte
        # ett godkännande: annars skulle ett kort utan källrad passera grinden
        # just för att det saknar allt.
        gaps = ["Kandidatdata saknas — kontrollerna går inte att läsa"]
    else:
        gaps = control_gaps(entry, c.get("position_pct_total"),
                            c.get("strategy", ""))
    val_gaps = valuation_gaps(entry, c)
    pos_gaps = position_gaps(entry, c)
    mechanical_fv = lukacs.fv_required(c.get("position_pct_total"),
                                       c.get("strategy", ""))

    checks = {}
    for g in GATES:
        if g.key == "inga_roda_flaggor":
            checks[g.key] = not gaps
        elif g.key == "sakerhetsmarginal" and mechanical_fv:
            checks[g.key] = not val_gaps
        elif g.key == "position_saljregel":
            checks[g.key] = bool(c.get(g.key)) and not pos_gaps
        else:
            checks[g.key] = bool(c.get(g.key))
    missing = [GATE_BY_KEY[k].label for k, ok in checks.items() if not ok]
    return {"checks": checks, "missing": missing, "gaps": gaps,
            "valuation_gaps": val_gaps, "position_gaps": pos_gaps,
            "fv_mechanical": mechanical_fv, "ready": not missing}


def is_ready(card: dict, entry: Optional[dict] = None) -> bool:
    return gate_state(card, entry)["ready"]


def stale_controls(cards: list, today: Optional[date] = None) -> list:
    """Årsregeln: kontroller som inte påverkat ett beslut på tolv månader."""
    day = today or date.today()
    out = []
    for c in cards or []:
        last = (c or {}).get("last_decision")
        if not last:
            continue
        try:
            d = date.fromisoformat(str(last)[:10])
        except (ValueError, TypeError):
            continue
        if (day - d).days >= STALE_DAYS:
            out.append({"card": c, "days": (day - d).days})
    return out


# ── Lagring ──────────────────────────────────────────────────────────────────
def _default() -> dict:
    return {"cards": {}}


def _load() -> dict:
    """Laddas EN gång per session; därefter äger sessionen sanningen."""
    data = storage.session_load(STORE, _default(),
                                legacy_file="scorecard_data.json")
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


def _gather() -> dict:
    """Läser de andra flikarnas lagring. Saknas en flik hoppas den över."""
    sources = {}

    def _try(module_name, extract):
        try:
            mod = __import__(module_name)
            sources.update(extract(mod))
        except Exception:
            pass

    _try("scoring", lambda m: {"sprott": m._load().get("sprott", []),
                               "durrett": m._load().get("durrett", [])})
    _try("tiggre", lambda m: {"tiggre": (m._load().get("candidates", [])
                                         + m._load().get("positions", []))})
    _try("producers", lambda m: {"producenter": m._load().get("producers", []),
                                 "royalty": m._load().get("royalty", [])})
    _try("insider", lambda m: {"insider": m._load().get("signals", [])})
    return sources


# ── UI ───────────────────────────────────────────────────────────────────────
def render_scorecard_page() -> None:
    data = _load()
    storage_ui.save_bar(STORE, "Master Scorecard")
    st.markdown(
        f"<div style='text-align:center;padding:10px 0 4px;'>"
        f"<h2 style='color:{GOLD};letter-spacing:0.12em;margin:0;'>"
        f"MASTER SCORECARD</h2>"
        f"<p style='color:{DIM};font-size:0.78rem;margin:6px 0 0;'>"
        f"Sista steget före köp. Kandidaterna läses ur de andra flikarna — "
        f"här fattas bara beslutet.</p></div>", unsafe_allow_html=True)

    entries = collect(_gather())
    if not entries:
        st.info("Inga kandidater ännu. Lägg upp dem i Poängmodellen, Tiggre, "
                "Granskningsarken eller Insider — de dyker upp här automatiskt.")
        return

    cards = data.setdefault("cards", {})
    _export(entries, cards)

    ready = [e for e in entries
             if is_ready(cards.get(e["key"], {}), e)]
    c1, c2 = st.columns(2)
    c1.metric("Kandidater", len(entries))
    c2.metric("Klara för köp", len(ready))

    for entry in entries:
        _card(data, cards, entry)

    _stale(cards)


def _card(data: dict, cards: dict, entry: dict) -> None:
    card = cards.setdefault(entry["key"], {})
    state = gate_state(card, entry)
    head = (f"{'✅' if state['ready'] else '📋'}  {entry['ticker']}"
            f"  ·  {entry['name'] or '—'}"
            f"  ·  {', '.join(entry['strategies'])}")

    with st.expander(head, expanded=False):
        changed = False

        # Vilken strategi beslutet gäller — styr proportionalitetsregeln.
        strategies = entry["strategies"]
        cur = card.get("strategy") if card.get("strategy") in strategies else strategies[0]
        s1, s2, s3 = st.columns(3)
        strategy = s1.selectbox("Strategi", strategies,
                                index=strategies.index(cur),
                                key=f"sc_strat_{entry['key']}")
        pos_total = s2.number_input(
            "Position (% av total)", min_value=0.0, step=0.5,
            value=float(_num(card.get("position_pct_total"), 0.0) or 0.0),
            key=f"sc_post_{entry['key']}")
        pos_strat = s3.number_input(
            "Position (% av strategidelen)", min_value=0.0, step=1.0,
            value=float(_num(card.get("position_pct_strategy"), 0.0) or 0.0),
            key=f"sc_poss_{entry['key']}")
        if (strategy != card.get("strategy")
                or pos_total != card.get("position_pct_total")
                or pos_strat != card.get("position_pct_strategy")):
            card["strategy"] = strategy
            card["position_pct_total"] = pos_total
            card["position_pct_strategy"] = pos_strat
            changed = True

        # Sammanställningen — läses, matas inte in.
        st_ = control_state(entry, pos_total, strategy)
        _summary_row(entry, st_, strategy)

        v1, v2 = st.columns(2)
        val = v1.text_input("Värdering", value=card.get("valuation", ""),
                            key=f"sc_val_{entry['key']}",
                            placeholder="t.ex. 4,2× EV/EBITDA, P/NAV 0,35")
        trig = v2.text_input("Trigger / katalysator",
                             value=card.get("trigger", ""),
                             key=f"sc_trig_{entry['key']}",
                             placeholder="Vad, och när?")
        sell = st.text_input("Vald säljregel", value=card.get("sell_rule", ""),
                             key=f"sc_sell_{entry['key']}",
                             placeholder="Vilken regel tar dig ur — bestäms nu")
        if (val != card.get("valuation", "") or trig != card.get("trigger", "")
                or sell != card.get("sell_rule", "")):
            card["valuation"], card["trigger"], card["sell_rule"] = val, trig, sell
            changed = True

        # Köpgrinden
        st.markdown(f"<div style='height:6px;'></div>"
                    f"<b style='color:{TEXT};'>Köpgrinden</b>",
                    unsafe_allow_html=True)
        for g in GATES:
            if g.key == "inga_roda_flaggor":
                _mechanical_gate(g, not state["gaps"], state["gaps"],
                                 "sätts av kontrollerna, kan inte kryssas förbi")
                continue
            if g.key == "sakerhetsmarginal" and state["fv_mechanical"]:
                _mechanical_gate(g, not state["valuation_gaps"],
                                 state["valuation_gaps"],
                                 "räknas av Lukacs FV — obligatorisk över "
                                 f"{ctl.FULL_WORK_MIN_PCT:g} % av totalen")
                continue
            v = st.checkbox(g.label, value=bool(card.get(g.key)),
                            key=f"sc_g_{entry['key']}_{g.key}", help=g.help)
            if v != bool(card.get(g.key)):
                card[g.key] = v
                changed = True
            if g.key == "position_saljregel":
                for gap in state["position_gaps"]:
                    st.markdown(
                        f"<div style='color:{RED};font-size:0.78rem;"
                        f"padding-left:22px;'>• {gap}</div>",
                        unsafe_allow_html=True)

        state = gate_state(card, entry)
        if state["ready"]:
            st.markdown(
                f"<div style='background:{GREEN}22;border:2px solid {GREEN};"
                f"border-radius:10px;padding:14px;text-align:center;"
                f"margin:10px 0;'><span style='color:{GREEN};font-size:1.3rem;"
                f"font-weight:800;letter-spacing:0.1em;'>{READY}</span></div>",
                unsafe_allow_html=True)
            if st.button("Registrera beslut", key=f"sc_dec_{entry['key']}"):
                card["last_decision"] = date.today().isoformat()
                _save(data)
                st.rerun()
        else:
            st.markdown(
                f"<div style='background:{RED}11;border:1px solid {RED}66;"
                f"border-radius:8px;padding:10px 14px;margin:10px 0;'>"
                f"<b style='color:{RED};'>{NO_TRADE}</b>"
                + "".join(f"<div style='color:{TEXT};font-size:0.82rem;"
                          f"margin-top:4px;'>• {m}</div>"
                          for m in state["missing"])
                + "</div>", unsafe_allow_html=True)

        if changed:
            _save(data)


def _summary_row(entry: dict, state: dict, strategy: str) -> None:
    req = state["required"]
    chips = []

    def _chip(label, value, band, color):
        return (f"<span style='background:{color}18;border:1px solid {color}66;"
                f"border-radius:5px;padding:3px 10px;margin-right:6px;"
                f"display:inline-block;margin-bottom:4px;'>"
                f"<b style='color:{color};'>{label} {value}</b>"
                f"<span style='color:{DIM};font-size:0.72rem;'> {band}</span>"
                f"</span>")

    if ctl.SEC_DS in req:
        band = state["ds_band"] or "ej bedömd"
        chips.append(_chip("DS", state["ds"] if state["ds"] is not None else "–",
                           band, ctl.DS_BAND_COLOR.get(state["ds_band"], DIM)))
    if ctl.SEC_AQS in req:
        band = state["aqs_band"] or "ej bedömd"
        chips.append(_chip("AQS",
                           state["aqs"] if state["aqs"] is not None else "–",
                           band, ctl.AQS_BAND_COLOR.get(state["aqs_band"], DIM)))
    if ctl.SEC_CSM in req:
        ok = not state["csm_flag"] and state["csm_complete"]
        chips.append(_chip("CSM", "✓" if ok else "✕",
                           "Bear överlevs" if ok else "brist eller röd flagga",
                           GREEN if ok else RED))

    st.markdown(f"<div style='margin:8px 0 4px;'>{''.join(chips)}</div>",
                unsafe_allow_html=True)
    hidden = {ctl.SEC_DS, ctl.SEC_AQS, ctl.SEC_CSM} - req
    if hidden:
        st.caption(f"Proportionalitetsregeln döljer {', '.join(sorted(hidden))} "
                   f"för den här positionsstorleken och strategin.")


CSV_COLUMNS = [
    ("ticker", "Ticker"), ("name", "Bolag"), ("strategy", "Strategi"),
    ("position_pct_total", "Position % av total"),
    ("position_pct_strategy", "Position % av strategidel"),
    ("valuation", "Värdering"), ("trigger", "Trigger"),
    ("sell_rule", "Säljregel"), ("_ds", "DS"), ("_aqs", "AQS"),
    ("_csm", "CSM"), ("_missing", "Saknas"), ("_ready", "Klar för köp"),
]


def _export(entries: list, cards: dict) -> None:
    rows = []
    for e in entries:
        card = cards.get(e["key"], {})
        strategy = card.get("strategy") or (e["strategies"][0]
                                            if e["strategies"] else "")
        stt = control_state(e, card.get("position_pct_total"), strategy)
        gs = gate_state(card, e)
        rows.append({
            **card, "ticker": e["ticker"], "name": e["name"],
            "strategy": strategy, "_ds": stt["ds"], "_aqs": stt["aqs"],
            "_csm": "röd flagga" if stt["csm_flag"] else "ok",
            "_missing": " · ".join(gs["missing"]),
            "_ready": gs["ready"],
        })
    csv_export.download_button(rows, CSV_COLUMNS, "master_scorecard",
                               key="csv_scorecard")


def _stale(cards: dict) -> None:
    stale = stale_controls(list(cards.values()))
    if not stale:
        return
    with st.expander(f"🗓 Årsregeln — {len(stale)} kort utan beslut på ett år",
                     expanded=False):
        st.caption("En kontroll som inte påverkat något beslut på tolv månader "
                   "kostar mer än den ger. Ta bort den vid årsgenomgången.")
        for s in stale:
            c = s["card"]
            st.markdown(
                f"<div style='color:{TEXT};font-size:0.82rem;'>"
                f"{c.get('strategy', '?')} — senaste beslut för "
                f"{s['days']} dagar sedan</div>", unsafe_allow_html=True)
