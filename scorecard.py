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

import calendar
import re
from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Optional

import streamlit as st

import controls as ctl
import csv_export
import lukacs
import storage
import storage_ui
from ui.components import confirm_delete, page_header

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


def _strategy_rule(strategy: str):
    """Strategins positionsregel ur allokeringsreglerna (allocator.PositionRule), None utan regel (confidence)."""
    try:
        import allocator
    except Exception:                                    # pragma: no cover
        return None
    key = (strategy or "").strip().lower()
    # Strategins egen regel först (tiggre 4 %, sprott 1,5 % — de delar sleeve
    # men inte tak); därefter regeln med det strängaste taket i sleeven med det namnet.
    rule = allocator.rule_for_strategy(key)
    if rule is not None:
        return rule
    rules = [r for r in allocator.POSITION_RULES if r.sleeve == key]
    return min(rules, key=lambda r: r.hard_cap) if rules else None


def _strategy_cap(strategy: str) -> Optional[float]:
    """Strategins hårda positionstak ur allokeringsreglerna."""
    rule = _strategy_rule(strategy)
    return rule.hard_cap if rule is not None else None


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
    # saknade nycklar fylls i utan att det räknas som en osparad ändring
    return storage.session_normalize(STORE, data, lambda d: storage.fill_defaults(d, _default()))


def _save(data: dict) -> None:
    """Skriver till sessionen. Persistensen sker via 💾 Spara.

    Medvetet ingen nätverkstrafik här: en commit per
    tangenttryckning skulle slå i GitHubs rate limits, och en
    tyst misslyckad sådan var precis den bugg det här ersätter.
    """
    st.session_state[STORE] = data


def _gather() -> dict:
    """Läser de andra flikarnas lagring. Saknas en flik hoppas den över."""
    stores = {}

    def _try(name, load):
        try:
            stores[name] = load()
        except Exception:
            pass

    _try("scoring", lambda: __import__("scoring")._load())
    _try("tiggre", lambda: __import__("tiggre")._load())
    _try("holdings", lambda: __import__("positions").load())
    _try("producers", lambda: __import__("producers")._load())
    _try("insider", lambda: __import__("insider")._load())
    _try("confidence", lambda: __import__("confidence.ui", fromlist=["load_store"]).load_store())
    return sources_from_stores(stores)


def sources_from_stores(stores: dict) -> dict:
    """Källorna till collect() ur redan lästa lager ({arknamn: data}, samma som storage.session_load ger).
    Ren funktion — så att Thor kan räkna grinden på samma kandidatrad som fliken (Arkens dom)."""
    import positions
    s = stores or {}

    def _part(name, key):
        store = s.get(name)
        return list((store.get(key) if isinstance(store, dict) else None) or [])

    return {"sprott": _part("scoring", "sprott"), "durrett": _part("scoring", "durrett"),
            "tiggre": _part("tiggre", "candidates") + positions.view_rows_from(s.get("holdings"), "Tiggre"),
            "producenter": _part("producers", "producers"), "royalty": _part("producers", "royalty"),
            "insider": _part("insider", "signals"),
            "confidence": confidence_rows(s.get("confidence") if isinstance(s.get("confidence"), dict) else {})}


def entries_for(tickers, stores: dict) -> dict:
    """{kortnyckel: kandidatrad} för tickerns former ('BOL.ST', 'BOL') — raden gate_state ska läsa, inte en
    attrapp utan arkrad (då blev DS alltid 'inte bedömd' i Thor)."""
    want = {str(t or "").strip().upper() for t in tickers or [] if t}
    return {e["key"]: e for e in collect(sources_from_stores(stores)) if e["ticker"] in want}


def confidence_rows(store: dict) -> list:
    """Confidence-/Durrett-arkets bolag som scorecard-rader.

    Arket lagrar bolagen som CompanyInput (fält i `fields`), inte som platta
    rader. Här plattas de till {ticker, name, stage, ...} plus de kontrollfält
    scorecardet redan förstår, så att ett bolag som granskats i Durrett-fliken
    dyker upp i köpgrinden i stället för att behöva läggas in en gång till.
    Bedömningen (BUY CANDIDATE/WATCH/… och Durrett-motorns risknivå) räknas
    i kortet — se engine_summary.
    """
    out = []
    companies = (store or {}).get("companies") or {}
    if not isinstance(companies, dict):
        return out
    for ticker, c in companies.items():
        if not isinstance(c, dict):
            continue
        t = str(c.get("ticker") or ticker or "").strip().upper()
        if not t:
            continue
        row = {k: v for k, v in c.items() if k != "fields"}
        row["ticker"], row["name"] = t, str(c.get("name") or "")
        row["_sheet"] = "confidence"
        out.append(row)
    return out


def engine_summary(ticker: str) -> Optional[str]:
    """En rad ur Confidence-scoren och Durrett-motorn för ett bolag i arket.

    Läses, matas inte in. None när bolaget inte finns i arket eller motorerna
    inte kan köras (t.ex. i test utan Streamlit-session).
    """
    try:
        from confidence import store as cs, reports
        from confidence.ui import load_store, _signals_for
        from engines.durrett.engine import analyze as durrett_analyze
    except Exception:                                    # pragma: no cover
        return None
    try:
        data = load_store()
        company = cs.get(data, (ticker or "").strip().upper())
        if company is None:
            return None
        a = reports.analyze(company, cs.overrides(data), _signals_for(company.commodity),
                            date.today())
        d = durrett_analyze(company, today=date.today())
        q = d.quality_score.value
        return (f"Confidence: {a.recommendation} · case {a.case.total:.0f} ({a.case.rating}) · "
                f"Durrett-motorn: kvalitet {q:.0f} · risk {d.risk_level}"
                if q is not None else
                f"Confidence: {a.recommendation} · case {a.case.total:.0f} ({a.case.rating}) · "
                f"Durrett-motorn: risk {d.risk_level}")
    except Exception:
        return None


# ── UI ───────────────────────────────────────────────────────────────────────
def render_scorecard_page() -> None:
    data = _load()
    storage_ui.save_bar(STORE, "Master Scorecard")
    page_header("Scorecard", "Sista steget före köp. Kandidaterna läses ur de andra "
                "flikarna — här fattas bara beslutet.")

    entries = collect(_gather())
    if not entries:
        st.info("Inga kandidater ännu. Lägg upp dem i Poängmodellen, Tiggre, "
                "Granskningsarken, Durrett/Confidence eller Insider — de dyker "
                "upp här automatiskt.")
        return

    cards = data.setdefault("cards", {})
    _export(entries, cards)

    ready = [e for e in entries
             if is_ready(_shown(cards.get(e["key"], {}), e), e)]
    c1, c2 = st.columns(2)
    c1.metric("Kandidater", len(entries))
    c2.metric("Klara för köp", len(ready))

    rc = _rc()
    foc = rc.focus("scorecard", [e["ticker"] for e in entries]) if rc else None
    for entry in entries:
        if rc and not rc.same_ticker(entry["ticker"], foc):
            continue
        _card(data, cards, entry, expanded=bool(foc))

    _stale(cards)


def _rc():
    """Kortstödet (fokus, x/y ifyllda, saknas-raden) — None om review-paketet inte går att läsa."""
    try:
        from review import card
        return card
    except Exception:
        return None


def _shown(card: dict, entry: dict) -> dict:
    """Kortet som det visas: strategin förvald till kandidatens första när ingen giltig är vald.

    Läses bara. Ett orört kort skrivs inte in i lagringen — förr skapade varje öppning ett kort med
    position 0,0 per kandidat, och Scorecardet stod som osparat innan något ändrats."""
    c = dict(card or {})
    strategies = (entry or {}).get("strategies") or []
    if strategies and c.get("strategy") not in strategies:
        c["strategy"] = strategies[0]
    return c


def _card(data: dict, cards: dict, entry: dict, expanded: bool = False) -> None:
    card = cards.get(entry["key"])
    card = card if isinstance(card, dict) else {}       # läggs i cards först när något ändras
    state = gate_state(_shown(card, entry), entry)
    rc = _rc()
    head = (f"{'✅' if state['ready'] else '📋'}  {entry['ticker']}"
            f"  ·  {entry['name'] or '—'}"
            f"  ·  {', '.join(entry['strategies'])}")

    with st.expander(head, expanded=expanded):
        top = rc.slot() if rc else None
        changed = False

        # Vilken strategi beslutet gäller — styr proportionalitetsregeln.
        strategies = entry["strategies"]
        cur = _shown(card, entry)["strategy"]
        s1, s2, s3 = st.columns(3)
        strategy = s1.selectbox("Strategi", strategies,
                                index=strategies.index(cur),
                                key=f"sc_strat_{entry['key']}")
        pos_total = s2.number_input(
            "Position (% av total)", min_value=0.0, step=0.5,
            value=_num(card.get("position_pct_total")),
            key=f"sc_post_{entry['key']}")
        pos_strat = s3.number_input(
            "Position (% av strategidelen)", min_value=0.0, step=1.0,
            value=_num(card.get("position_pct_strategy")),
            key=f"sc_poss_{entry['key']}")
        if (strategy != cur
                or storage.differs(pos_total, card.get("position_pct_total"))
                or storage.differs(pos_strat, card.get("position_pct_strategy"))):
            card["strategy"] = strategy
            card["position_pct_total"] = pos_total
            card["position_pct_strategy"] = pos_strat
            changed = True
        _rule_position(data, cards, entry, strategy, pos_total)
        _position_lines(entry, strategy)

        # Sammanställningen — läses, matas inte in.
        st_ = control_state(entry, pos_total, strategy)
        _summary_row(entry, st_, strategy)
        if "confidence" in entry["strategies"]:
            line = engine_summary(entry["ticker"])
            if line:
                st.caption(f"🐺 {line}")

        v1, v2 = st.columns(2)
        val = v1.text_input("Värdering", value=card.get("valuation", ""),
                            key=f"sc_val_{entry['key']}",
                            placeholder="t.ex. 4,2× EV/EBITDA, P/NAV 0,35")
        trig = v2.text_input("Trigger / katalysator",
                             value=card.get("trigger", ""),
                             key=f"sc_trig_{entry['key']}",
                             placeholder="Vad, och när?")
        _valuation_hint(data, cards, entry, strategy, val)
        _trigger_hint(data, cards, entry, strategy, trig)
        w1, w2 = st.columns([3, 2], vertical_alignment="bottom")
        sell = w1.text_input("Vald säljregel", value=card.get("sell_rule", ""),
                             key=f"sc_sell_{entry['key']}",
                             placeholder="Vilken regel tar dig ur — bestäms nu")
        _sell_rule_pick(w2, data, cards, entry, strategy)
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
            if g.key == "granskning_klar":
                _review_hint(data, cards, entry, strategy, v)
            if g.key == "position_saljregel":
                for gap in state["position_gaps"]:
                    st.markdown(
                        f"<div style='color:{RED};font-size:0.78rem;"
                        f"padding-left:22px;'>• {gap}</div>",
                        unsafe_allow_html=True)

        state = gate_state(_shown(card, entry), entry)
        if state["ready"]:
            st.markdown(
                f"<div style='background:{GREEN}22;border:2px solid {GREEN};"
                f"border-radius:10px;padding:14px;text-align:center;"
                f"margin:10px 0;'><span style='color:{GREEN};font-size:1.3rem;"
                f"font-weight:800;letter-spacing:0.1em;'>{READY}</span></div>",
                unsafe_allow_html=True)
            if st.button("Registrera beslut", key=f"sc_dec_{entry['key']}"):
                card["last_decision"] = date.today().isoformat()
                cards[entry["key"]] = card
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
            cards[entry["key"]] = card
            _save(data)
        _chatgpt(entry, _shown(card, entry), data)
        if rc:
            rc.fill(top, "scorecard", _shown(card, entry), entry=entry)


def _sv(v: float, d: int = 1) -> str:
    """Tal med svenskt decimalkomma och mellanslag som tusentalsavgränsare: 5,5 · 1 250."""
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",").replace("-", "−")


_CMP = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def _edge_digits(v: float, d: int, limits) -> Optional[int]:
    """Kortets decimaler — men fler när avrundningen skulle hamna på andra sidan en gräns som domen bygger på
    (2,96 visas inte som '3,0:1' bredvid 'under kravet 3'). limits = [(gräns, jämförelse)] som arkets egen regel.
    None när inte ens tre decimaler till räcker: talet ligger på gränsen (flyttalsbrus, 2,9999999999999996) — då
    visas kortets decimaler med '≈' framför (se _edge), inte en rad nollor."""
    for dd in range(d, d + 4):
        r = round(v, dd)
        if all(_CMP[op](r, lim) == _CMP[op](v, lim) for lim, op in limits):
            return dd
    return None


def _edge(v: float, d: int, limits, prefix: str = "") -> str:
    """Talet för en dom mot limits: _edge_digits decimaler, eller '≈' + kortets decimaler precis på gränsen."""
    dd = _edge_digits(v, d, limits)
    return f"{prefix}{_sv(v, dd)}" if dd is not None else f"≈{prefix}{_sv(v, d)}"


def _edge_abs(v: float, d: int, limits) -> str:
    """Som _edge, men beloppet (tecknet står i orden 'över/under')."""
    dd = _edge_digits(v, d, limits)
    return _sv(abs(v), dd) if dd is not None else f"≈{_sv(abs(v), d)}"


DURRETT_STRATEGY = "durrett"         # Poängmodellens Durrett-del (snabbpoängen) — inte ARKEN → 🧭 Durrett
DURRETT_SHEET_STRATEGY = "confidence"  # ARKEN → 🧭 Durrett (10-stegsarket, Durrett-motorn) i kandidatraden
ROYALTY_STRATEGY = "royalty"          # Royalty C i kandidatraden (sources_from_stores)
VALUATION_SOURCE = {DURRETT_STRATEGY: "Poängmodellen → Durrett (snabbpoäng)",
                    DURRETT_SHEET_STRATEGY: "Durrett-arket (ARKEN → 🧭 Durrett)",
                    "producenter": "Rick Rule", ROYALTY_STRATEGY: "Royalty C",
                    "tiggre": "Tiggre (Lobo-arket)", "sprott": "Poängmodellen → Sprott (optionalitet)"}


def _ev_ebitda_part(ref: Optional[dict]) -> list:
    ev_eb = _num((ref or {}).get("ev_ebitda")) if isinstance(ref, dict) else None
    return [f"EV/EBITDA {_sv(ev_eb)}× (Börsdata)"] if ev_eb is not None and ev_eb > 0 else []


def _metric(metrics: Optional[dict], name: str) -> Optional[float]:
    m = (metrics or {}).get(name) if isinstance(metrics, dict) else None
    return _num(m.get("value")) if isinstance(m, dict) else None


def valuation_suggestion(entry: Optional[dict], strategy: str, ref: Optional[dict] = None,
                         metrics: Optional[dict] = None) -> Optional[str]:
    """Värderingen ur arkets egna tal — samma uträkning som arket gör, inte en ny. None när inget går att räkna.

    durrett (Poängmodellens snabbpoäng): MCap/framtida vinst mot köpregeln (under 10×) och MCap/uns mot $100/$50
    — scoring-modulens funktioner och gränser på arkradens börsvärde, uns och framtida vinst.
    confidence (ARKEN → 🧭 Durrett): Durrett-motorns mått (metrics = analyze(...).metrics) — MCap/framtida vinst
    efter skatt mot motorns köpregel, EV per resursenhet och EV/NPV — samma tal som arket visar.
    Båda: Börsdatas EV/EBITDA ur sifferuppdateringen (ref) när den finns. Ren funktion — ingen Streamlit."""
    parts: list = []
    if strategy == DURRETT_STRATEGY:
        row = (((entry or {}).get("rows") or {}).get(DURRETT_STRATEGY))
        if not isinstance(row, dict):
            return None
        import scoring
        ratio = scoring.mcap_per_earnings(row.get("mcap"), row.get("profit"))
        if ratio is not None:
            ok = scoring.durrett_buy_ok(ratio)
            parts.append(f"MCap/framtida vinst {_edge(ratio, 1, [(scoring.DURRETT_BUY_MAX, '<')])}× "
                         f"({'under' if ok else 'över'} köpregelns "
                         f"{scoring.DURRETT_BUY_MAX:g}×)")
        per_oz = scoring.mcap_per_oz(row.get("mcap"), row.get("moz"))
        if per_oz is not None:
            band = ("mycket billigt" if per_oz < scoring.MCAP_PER_OZ_VERY
                    else "billigt" if per_oz < scoring.MCAP_PER_OZ_CHEAP
                    else f"över ${scoring.MCAP_PER_OZ_CHEAP:g}")
            lim = [(scoring.MCAP_PER_OZ_VERY, "<"), (scoring.MCAP_PER_OZ_CHEAP, "<")]
            parts.append(f"MCap/uns {_edge(per_oz, 0, lim, prefix='$')} ({band})")
    elif strategy == DURRETT_SHEET_STRATEGY:
        if not isinstance(metrics, dict):
            return None
        from engines.durrett import config as dcfg
        buy_max = float(dcfg.DURRETT_CONFIG["mcap_future_earnings_buy_max"])
        r = _metric(metrics, "mcap_future_earnings")
        if r is not None:
            parts.append(f"MCap/framtida vinst {_edge(r, 1, [(buy_max, '<')])}× efter skatt "
                         f"({'under' if r < buy_max else 'över'} "
                         f"köpregelns {buy_max:g}×)")
        ev_res = _metric(metrics, "ev_per_resource_unit")
        if ev_res is not None:
            unit = str((metrics.get("ev_per_resource_unit") or {}).get("unit") or "USD/enhet")
            parts.append(f"EV/resurs {_sv(ev_res, 0 if ev_res >= 10 else 2)} {unit}")
        ev_npv = _metric(metrics, "ev_npv")
        if ev_npv is not None:
            parts.append(f"EV/NPV {_sv(ev_npv, 2)}×")
    elif strategy == "producenter":
        row = (((entry or {}).get("rows") or {}).get("producenter"))
        if not isinstance(row, dict):
            return None
        parts += _rule_parts(row, ref)
        return " · ".join(parts) or None
    elif strategy == ROYALTY_STRATEGY:
        row = (((entry or {}).get("rows") or {}).get(ROYALTY_STRATEGY))
        if not isinstance(row, dict):
            return None
        parts += _royalty_parts(row, ref)
        return " · ".join(parts) or None
    elif strategy in ("tiggre", "sprott"):
        row = (((entry or {}).get("rows") or {}).get(strategy))
        if not isinstance(row, dict):
            return None
        parts += _tiggre_parts(row, ref) if strategy == "tiggre" else _sprott_parts(row)
        return " · ".join(parts) or None
    else:
        return None
    parts += _ev_ebitda_part(ref)
    return " · ".join(parts) or None


def _sheet_ev_ebitda(row: dict, field: str, ref: Optional[dict]) -> list:
    """Arkets egen EV/EBITDA-ruta; tom ruta → Börsdatas (märkt) ur sifferuppdateringen. Negativ = lucka."""
    v = _num(row.get(field))
    if v is not None and v > 0:
        return [f"EV/EBITDA {_sv(v, 2)}×"]                # två decimaler, som kortets ruta
    return _ev_ebitda_part(ref)


def _rule_parts(row: dict, ref: Optional[dict]) -> list:
    """Rick Rule: EV/NAV med arkets bedömning (och NAV-prisvarningen), Lukacs säkerhetsmarginal mot fair value Base
    med modulens band — med arkets varning om FCF-enhet/kursvaluta när den finns — och EV/EBITDA. Samma funktioner
    och gränser som Rick Rule-kortet (producers, lukacs)."""
    import producers as pr
    parts = []
    r = pr.ev_nav(row.get("ev_musd"), row.get("nav_musd"))
    if r is not None:
        label = (pr.NAV_CHEAP if r < pr.EV_NAV_CHEAP else pr.NAV_FAIR if r <= pr.EV_NAV_FAIR else pr.NAV_RICH)
        lim = [(pr.EV_NAV_CHEAP, "<"), (pr.EV_NAV_FAIR, "<=")]
        txt = f"EV/NAV {_edge(r, 2, lim)}× ({label[:1].lower() + label[1:]})"
        gap = pr.nav_price_gap(row.get("nav_price"), row.get("price"))
        if gap is not None and abs(gap) > pr.NAV_PRICE_GAP_PCT:
            txt += (f", NAV räknad på {_sv(_num(row.get('nav_price')), 0)} mot spot {_sv(_num(row.get('price')), 0)} "
                    f"({gap:+.0f} %)".replace(".", ","))
        parts.append(txt)
    ev = lukacs.evaluate(row)
    if ev.get("not_fcf_valued"):
        parts.append("Lukacs: klass D — värderas inte på FCF")
    elif ev.get("mos") is not None and ev.get("fv_base") is not None:
        band = ev.get("mos_band") or ""
        m6 = round(ev["mos"], 6)                          # mos_band jämför det avrundade till 6 decimaler
        lim = [(lukacs.MOS_VERY_MIN, ">"), (lukacs.MOS_BUY_MIN, ">="), (lukacs.MOS_WATCH_MIN, ">=")]
        txt = (f"Lukacs: säkerhetsmarginal {_edge(m6, 1, lim)} % mot fair value Base {_sv(ev['fv_base'], 2)}"
               + (f" ({band[:1].lower() + band[1:]})" if band else ""))      # en decimal, som kortet
        obs = []
        code = row.get("fcf_kvalitet")                                     # som kortet: ingen vald klass är '—'
        if ev.get("quality") is None:
            obs.append("saknad FCF-klass")
        elif any(lukacs.yield_error(code, sc.get("target_yield")) not in (None, lukacs.NOT_FCF_VALUED)
                 for sc in (row.get("fv") or {}).values() if isinstance(sc, dict)):
            obs.append("target-yield utanför klassens band")
        cur = lukacs.price_currency(row.get("aktuell_kurs"), ref)
        if lukacs.fcf_unit_warnings(row, ref) or (cur is not None and cur.get("stored_is_local")):
            obs.append("FCF-enhet eller kursvaluta")
        if obs:
            txt += " — OBS: arket varnar för " + " och ".join(obs)
        parts.append(txt)
    parts += _sheet_ev_ebitda(row, "ev_ebitda", ref)
    return parts


def _royalty_parts(row: dict, ref: Optional[dict]) -> list:
    """Royalty C: P/NAV mot egen botten (köpläge högst +15 %), EV/EBITDA mot egen median, GEO per aktie på tre år
    och arkets signal — samma funktioner och gränser som Royalty-kortet (producers)."""
    import producers as pr
    parts = []
    disc = pr.discount_vs_bottom(row.get("pnav_now"), row.get("pnav_bottom"))
    if disc is not None:                               # en decimal, som kortets 'Mot botten'
        side = "över" if disc >= 0 else "under"
        amount = _edge_abs(disc, 1, [(pr.DISCOUNT_MAX, "<="), (0.0, "<")])
        parts.append(f"P/NAV {_sv(_num(row.get('pnav_now')), 2)} ({amount} % {side} egen botten "
                     f"{_sv(_num(row.get('pnav_bottom')), 2)} — köpläge högst {pr.DISCOUNT_MAX:g} % över)")
    med = pr.vs_median(row.get("ev_now"), row.get("ev_median"))
    if med is not None:
        rel = ("på" if med == 0 else
               f"{_edge_abs(med, 1, [(0.0, '<'), (0.0, '>')])} % {'över' if med > 0 else 'under'}")
        parts.append(f"EV/EBITDA {_sv(_num(row.get('ev_now')), 2)}× ({rel} egen median "
                     f"{_sv(_num(row.get('ev_median')), 2)}×)")
    else:
        parts += _sheet_ev_ebitda(row, "ev_now", ref)
    geo = pr.geo_growth(row.get("geo_now"), row.get("geo_3y"))
    if geo is not None:
        parts.append(f"GEO/aktie {'+' if geo >= 0 else ''}{_edge(geo, 1, [(0.0, '<'), (0.0, '>')])} % på 3 år")
    if disc is not None or med is not None or geo is not None:   # kortets signal — GEO-varningen står på egen hand
        parts.append(f'signal "{pr.royalty_signal(row).label}"')
    return parts


def _nonneg(v) -> Optional[float]:
    x = _num(v)
    return None if x is None or x < 0 else x


def _tiggre_parts(row: dict, ref: Optional[dict]) -> list:
    """Tiggre (Lobo-arket): P/NAV mot köpzonen 0,2–0,4×, uppsidan till 0,8× NAV och U/N-kvoten mot kravet ≥ 3 —
    samma funktioner som kandidatkortet (tiggre). En öppen position räknas som positionskortet: på dagens börsvärde
    ur sifferuppdateringen när det finns, och med slutsäljregeln (0,8–1,0× NAV) i stället för köpets mått.
    Negativa tal är luckor (rutorna har golvet 0)."""
    import tiggre as tg
    parts = []
    held = str(row.get("strategy") or "") == tg.STRATEGY
    mc, nav = _nonneg(row.get("mcap")), _nonneg(row.get("nav"))
    if held:
        fresh = _nonneg((ref or {}).get("mcap_musd")) if isinstance(ref, dict) else None
        mc = fresh or mc
    pn = tg.p_nav(mc, nav) if mc and nav else None
    if pn is not None and held:
        stage = tg.nav_exit_stage(pn)
        lim = [(tg.NAV_EXIT_LO, ">="), (tg.NAV_EXIT_HI, ">=")]
        note = (f"≥ {tg.NAV_EXIT_HI:g}× NAV — omvärderingen gjord, sälj resten" if stage == "klar" else
                f"{tg.NAV_EXIT_LO:g}–{tg.NAV_EXIT_HI:g}× NAV — slutsälj i etapper" if stage == "etapp" else
                f"under {tg.NAV_EXIT_LO:g}× NAV — slutsäljregeln inte nådd")
        parts.append(f"P/NAV {_edge(pn, 2, lim)}× ({note})".replace(".", ","))
        return parts
    if pn is not None:
        parts.append(f"P/NAV {_sv(pn, 2)}× (köpzon 0,2–0,4×)")
    up = tg.upside_pct(mc, nav) if mc and nav else None
    if up is not None:
        parts.append(f"uppsida till {_sv(tg.NAV_TARGET, 1)}× NAV {'+' if up >= 0 else ''}{_sv(up, 0)} %")
        un = tg.un_ratio(up, row.get("downside"))
        if un is not None:
            ok = un >= tg.UN_MIN
            parts.append(f"U/N {_edge(un, 1, [(tg.UN_MIN, '>=')])}:1 ({'klarar' if ok else 'under'} kravet "
                         f"{tg.UN_MIN:g}:1)")
    return parts


def _sprott_parts(row: dict) -> list:
    """Sprott: runway = kassa ÷ årsburn, i år och månader mot Sprotts hårda grind (18 månader) — samma funktioner
    som Poängmodellens Sprott-kort (scoring)."""
    import scoring
    rw = scoring.runway_years(row.get("cash"), row.get("burn"))
    if rw is None:
        return []
    ok = scoring.sprott_runway_ok(rw)
    lim_m = scoring.SPROTT_RUNWAY_MIN_MONTHS
    months = _edge(rw * 12.0, 0, [(float(lim_m), ">=")])
    years = _edge(rw, 1, [(lim_m / 12.0, ">=")])          # även åren: 1,465 år är inte '1,5 år' bredvid 'under'
    return [f"runway {years} år = {months} mån ({'klarar' if ok else 'under'} Sprotts grind {lim_m} mån"
            f"{'' if ok else ' — avstå oavsett poäng'})"]


def _durrett_metrics(ticker: str) -> Optional[dict]:
    """Durrett-motorns mått för bolaget i ARKEN → 🧭 Durrett — samma körning som engine_summary. None utan bolag."""
    try:
        from confidence import store as cs
        from confidence.ui import load_store
        from engines.durrett.engine import analyze as durrett_analyze
        company = cs.get(load_store(), (ticker or "").strip().upper())
        return None if company is None else durrett_analyze(company, today=date.today()).metrics
    except Exception:
        return None


def _valuation_ref(entry: Optional[dict], strategy: str) -> Optional[dict]:
    """Sifferuppdateringens rad för kandidatens arkrad ('scoring:<id>' / 'confidence:<TICKER>') — None utan rad."""
    rows_e = (entry or {}).get("rows") or {}
    if strategy == DURRETT_STRATEGY:
        row = rows_e.get(DURRETT_STRATEGY)
        key = f"scoring:{row['id']}" if isinstance(row, dict) and row.get("id") else None
    elif strategy == DURRETT_SHEET_STRATEGY:
        key = f"confidence:{str((entry or {}).get('ticker') or '').strip().upper()}"
    elif strategy == "tiggre":                                 # kandidater och positioner: tiggre:<id>
        row = rows_e.get("tiggre")
        key = f"tiggre:{row['id']}" if isinstance(row, dict) and row.get("id") else None
    elif strategy in ("producenter", ROYALTY_STRATEGY):         # båda bor i arket producers, nyckel per rad-id
        row = rows_e.get(strategy)
        if not (isinstance(row, dict) and row.get("id")):
            return None
        try:                    # samma rad som Lukacs-panelen — Yahoo-reservens valuta ur tickerns suffix
            import lukacs_ui
            return lukacs_ui._refresh_row("producers", row["id"])
        except Exception:
            return None
    else:
        return None
    if not key:
        return None
    try:
        import refresh_ui
        rows = (refresh_ui.load_refresh() or {}).get("rows") or {}
        r = rows.get(key)
        if r is None and strategy == DURRETT_SHEET_STRATEGY:     # arkets nyckel kan ha annan skiftläge
            r = next((v for k, v in rows.items() if str(k).upper() == key.upper()), None)
        return r if isinstance(r, dict) else None
    except Exception:
        return None


def _use_card(data: dict, cards: dict, entry: dict, strategy: str, field: str, value, widget_key: str) -> None:
    """on_click: ett förslag in i kortets fält och i widgetens state — kortet skapas först här (ett orört kort skrivs
    aldrig). Strategin följer med: förslaget gällde den. 💾 Spara sköter persistensen som vanligt."""
    card = cards.get(entry["key"]) if isinstance(cards.get(entry["key"]), dict) else {}
    card["strategy"] = strategy
    card[field] = value
    cards[entry["key"]] = card
    st.session_state[widget_key] = value
    _save(data)


def _use_valuation(data: dict, cards: dict, entry: dict, strategy: str, text: str) -> None:
    """on_click: arkets värdering in i kortets Värdering-ruta (samma mönster som _use_position)."""
    _use_card(data, cards, entry, strategy, "valuation", text, f"sc_val_{entry['key']}")


def _valuation_hint(data: dict, cards: dict, entry: dict, strategy: str, current: str) -> None:
    """Förslaget under Värdering: räknat ur arket strategin hör till, visas av sig självt med arkets namn — skrivs in
    först med 'Använd' (att öppna sidan skriver ingenting). Döljs när rutan redan har exakt den texten."""
    if strategy not in VALUATION_SOURCE:
        return
    try:
        metrics = _durrett_metrics(entry.get("ticker")) if strategy == DURRETT_SHEET_STRATEGY else None
        text = valuation_suggestion(entry, strategy, _valuation_ref(entry, strategy), metrics)
    except Exception:
        return
    if not text or (current or "").strip() == text:
        return
    a, b = st.columns([3, 1], vertical_alignment="center")
    shown = text.replace("$", "\\$")                   # två '$' på en rad blir annars en formel i Markdown
    a.caption(f"🤖 Ur {VALUATION_SOURCE[strategy]}: {shown} — förslag, inte inskrivet.")
    b.button("Använd", key=f"sc_useval_{entry['key']}", on_click=_use_valuation,
             args=(data, cards, entry, strategy, text))


def _use_position(data: dict, cards: dict, entry: dict, strategy: str, value: float) -> None:
    """on_click: Rick Rule-kortets tänkta position in i scorecardet (samma tal, en gång)."""
    _use_card(data, cards, entry, strategy, "position_pct_total", value, f"sc_post_{entry['key']}")


RULE_STRATEGY = "producenter"        # Rick Rule-arkets strateginamn i kandidatraden (sources_from_stores)


def _rule_position(data: dict, cards: dict, entry: dict, strategy: str, pos_total) -> None:
    """Förslag: positionen du skrev i Rick Rule-kortet för bolaget — samma tal styr kontrollerna där. "Använd"
    skriver in den här; inget fylls i av sig självt. Bara när beslutet gäller Rick Rule (strategin producenter) och
    kortet ännu saknar position (tomt eller 0 %) — har du valt en storlek står den, och en annan strategi har sin
    egen."""
    if strategy != RULE_STRATEGY or (_num(pos_total) or 0) > 0:
        return
    rule = _num((((entry or {}).get("rows") or {}).get(RULE_STRATEGY) or {}).get("position_pct"))
    if rule is None or not storage.differs(pos_total, rule):
        return
    a, b = st.columns([3, 1], vertical_alignment="center")
    a.caption(f"Rick Rule-kortet: tänkt position {rule:g} % av total — förslag, inte inskrivet.")
    b.button("Använd", key=f"sc_userule_{entry['key']}", on_click=_use_position,
             args=(data, cards, entry, strategy, rule))


# ── Förslag och visning ur arken som redan har svaret (batch 9) ──────────────
# Scorecardets strateginamn → granskningen (review_link._REVIEWERS) och playbooken (strategy_rules.PLAYBOOKS); övriga
# heter lika. confidence (ARKEN → 🧭 Durrett) har ingen granskare i review_link — Poängmodellens Durrett är ett annat
# ark — men Durretts playbook.
REVIEW_KEY = {RULE_STRATEGY: "rule"}
PLAYBOOK_KEY = {**REVIEW_KEY, DURRETT_SHEET_STRATEGY: "durrett"}
SHEET_LABEL = {**VALUATION_SOURCE, "insider": "Insiderbevakaren"}


def _g(v: float) -> str:
    """Tal utan onödiga decimaler, med decimalkomma: 4 · 1,5 · 12,5."""
    return f"{v:g}".replace(".", ",").replace("-", "−")


def review_verdict(entry: Optional[dict], strategy: str) -> Optional[tuple]:
    """(ark, status, notering) ur arkets egen dom (review_link._REVIEWERS) på raden för kortets strategi — samma
    PASS/MANUAL/FAIL som Copiloten och Thor läser. None när strategin saknar granskare eller raden saknas."""
    import review_link
    key = REVIEW_KEY.get(strategy, strategy)
    fn = review_link._REVIEWERS.get(key)
    row = ((entry or {}).get("rows") or {}).get(strategy)
    if fn is None or not isinstance(row, dict):
        return None
    status, note = fn(row)
    return review_link.sheet_name(key), status, note


def _review_hint(data: dict, cards: dict, entry: dict, strategy: str, checked: bool) -> None:
    """Under 'Granskningen är klar': arkets dom. PASS ger ett förslag med Använd (bara när rutan inte är kryssad);
    MANUAL och FAIL står som text som säger vad som saknas — de kryssar ingenting."""
    try:
        verdict = review_verdict(entry, strategy)
    except Exception:
        return
    if verdict is None:
        return
    sheet, status, note = verdict
    if status == "PASS":
        if checked:
            return
        a, b = st.columns([3, 1], vertical_alignment="center")
        shown = f"🤖 Ur {sheet}: PASS — {note}".replace("$", "\\$")
        a.caption(f"{shown} — förslag, inte inskrivet.")
        b.button("Använd", key=f"sc_usegr_{entry['key']}", on_click=_use_card,
                 args=(data, cards, entry, strategy, "granskning_klar", True,
                       f"sc_g_{entry['key']}_granskning_klar"))
        return
    color = RED if status == "FAIL" else AMBER
    text = escape(f"🤖 Ur {sheet}: {status} — {note}").replace("$", "&#36;")
    st.markdown(f"<div style='color:{color};font-size:0.78rem;padding-left:22px;'>{text}</div>",
                unsafe_allow_html=True)


def position_frame(entry: Optional[dict], strategy: str, peak=None, current=None) -> str:
    """Raden under 'Position (% av total)': strategins tak och normalband ur allokeraren, halveringen vid skuld
    (deleveraging_cap, samma som grinden) och strömbrytarens läge (allocator.breaker_state på allokerarens topp och
    nuvarande). Bara visning — taket är inget storleksförslag. Ren funktion."""
    import allocator
    rule = _strategy_rule(strategy)
    if rule is None:
        parts = ["Inget tak i allokeraren"]
    else:
        parts = [f"Tak {_g(rule.hard_cap)} % av totalen",
                 f"normalt {_g(rule.normal_lo)}–{_g(rule.normal_hi)} % av strategidelen"]
        half = deleveraging_cap(entry, {"strategy": strategy})
        if half is not None:
            nd = _num(source_row(entry, strategy).get("nd_ebitda"))
            parts.append(f"nettoskulden {_g(nd)}× EBITDA halverar taket till {_g(half)} %")
    dd = allocator.drawdown_pct(peak, current)
    label, _color, action = allocator.breaker_state(dd)
    if dd is None:
        parts.append("strömbrytaren okänd — fyll i topp och nuvarande i Allokeringen")
    elif label == allocator.BREAKER_LEVELS[0][2]:
        parts.append(f"strömbrytaren {label} ({_sv(dd)} % från topp)")
    else:
        parts.append(f"strömbrytaren {label} ({_sv(dd)} % från topp): {action}")
    return " · ".join(parts)


def held_weight(ticker: str, rows: list, total) -> Optional[str]:
    """'Äger redan …': registrets öppna positioner i bolaget (allocator.derived_positions — positions.valuation, antal
    × kurs i SEK) mot allokerarens total (allocator._cap_total). None när bolaget inte ägs. Bara visning."""
    t = str(ticker or "").strip().upper()
    mine = [r for r in rows or [] if str(r.get("ticker") or "").strip().upper() == t]
    if not t or not mine:
        return None
    vals = [_num(r.get("value")) for r in mine]
    if any(v is None for v in vals):
        return f"Äger redan {t} — antal eller kurs saknas i Positioner, vikten går inte att räkna."
    value, tot = sum(vals), _num(total)
    if not tot or tot <= 0:
        return f"Äger redan {t} för {_sv(value, 0)} kr — totalen saknas i Allokeringen."
    rough = any(str(r.get("_currency") or "SEK").upper() != "SEK" for r in mine)
    return (f"Äger redan: {_sv(value / tot * 100)} % av totalen ({_sv(value, 0)} kr av {_sv(tot, 0)} kr"
            + (", grovt omräknat till SEK" if rough else "") + ").")


def _position_lines(entry: dict, strategy: str) -> None:
    """Raderna under positionen: ramen (position_frame) och nuvarande vikt när bolaget redan ägs. Läses bara.
    Vikten visar innehavets värde och totalen (med kassan) — bara när innehaven är upplåsta (private_lock).
    Låst syns ingen rad alls, så att raden inte själv avslöjar vilka bolag som ägs."""
    try:
        import allocator
        import positions
        from portfolio import private_lock
        alloc = allocator._load()
        st.caption(f"📏 {position_frame(entry, strategy, alloc.get('peak'), alloc.get('current'))}")
        if not private_lock.unlocked():
            return
        t = str(entry.get("ticker") or "").strip().upper()
        if not any(r.get("ticker") == t for r in positions.all_positions()):
            return
        rows = allocator.derived_positions(allocator._fresh_rows())
        line = held_weight(t, rows, allocator._cap_total(alloc.get("values") or {}, rows))
        if line:
            st.caption(line)
    except Exception:
        return


def sell_rule_options(entry: Optional[dict], strategy: str) -> list:
    """Säljreglerna ur strategins playbook (strategy_rules.PLAYBOOKS[pb].exit), i playbookens ordning. För Insider
    läggs stoppen till (insider.stop_price: klustersnittet × STOP_FRAC). Tom lista utan playbook."""
    import strategy_rules
    pb = strategy_rules.PLAYBOOKS.get(PLAYBOOK_KEY.get(strategy, strategy))
    out = [r.text for r in (pb.exit if pb else ())]
    row = ((entry or {}).get("rows") or {}).get(strategy)
    if strategy == "insider" and isinstance(row, dict):
        import insider as ins
        stop = ins.stop_price(row)
        if stop is not None and stop > 0:
            out.append(f"Stopp {_sv(stop, 2)} (klustersnittet {_sv(_num(row.get('cluster_avg')), 2)} × "
                       f"{_sv(ins.STOP_FRAC, 2)})")
    return out


def _pick_sell_rule(data: dict, cards: dict, entry: dict, strategy: str) -> None:
    """on_change: den valda regeln kopieras in i 'Vald säljregel'; listan står sedan tom igen (inget förval)."""
    pick = f"sc_sellpick_{entry['key']}"
    text = st.session_state.get(pick)
    if text:
        _use_card(data, cards, entry, strategy, "sell_rule", text, f"sc_sell_{entry['key']}")
    st.session_state[pick] = None


def _sell_rule_pick(col, data: dict, cards: dict, entry: dict, strategy: str) -> None:
    try:
        options = sell_rule_options(entry, strategy)
    except Exception:
        return
    if not options:
        return
    col.selectbox("Ur playbooken", options, index=None, placeholder="Välj — texten kopieras",
                  key=f"sc_sellpick_{entry['key']}", on_change=_pick_sell_rule,
                  args=(data, cards, entry, strategy))


def _period_end(text) -> Optional[date]:
    """Sista dagen i perioden: ÅÅÅÅ-MM-DD, ÅÅÅÅ-MM, ÅÅÅÅ-Qn/Qn ÅÅÅÅ, ÅÅÅÅ-Hn eller ÅÅÅÅ. None när det inte går att
    läsa — en katalysator utan när är ingen trigger."""
    s = str(text or "").strip()
    try:
        m = re.match(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?$", s)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if m.group(3):
                return date(y, mo, int(m.group(3)))
            return date(y, mo, calendar.monthrange(y, mo)[1])
        m = (re.match(r"^(\d{4})\s*-?\s*([QqHh])([1-4])$", s)
             or re.match(r"^([QqHh])([1-4])\s*-?\s*(\d{4})$", s))
        if m:
            g = m.groups()
            y, kind, n = (g[0], g[1], g[2]) if g[0].isdigit() else (g[2], g[0], g[1])
            months = 3 if kind.upper() == "Q" else 6
            if kind.upper() == "H" and int(n) > 2:
                return None
            mo = int(n) * months
            return date(int(y), mo, calendar.monthrange(int(y), mo)[1])
        if re.match(r"^\d{4}$", s):
            return date(int(s), 12, 31)
    except ValueError:
        return None
    return None


def next_catalyst(entry: Optional[dict], strategy: str, today: Optional[date] = None) -> Optional[tuple]:
    """(källa, text) för nästa väntande katalysator på raden för kortets strategi: Tiggre-radens kalender (status
    Väntar), Durrett-arkets katalysatorer (engines.durrett.thesis.catalysts) och DS-finansieringskatalysatorn
    (fin_catalyst_text/fin_catalyst_date). Försenade, uteblivna, levererade och passerade föreslås aldrig, och inte
    heller en utan datum. Ren funktion."""
    day = today or date.today()
    row = ((entry or {}).get("rows") or {}).get(strategy)
    if not isinstance(row, dict):
        return None
    found = []                                          # (periodens slut, ordning, källa, text)

    def _add(name, when, source):
        name, when = str(name or "").strip(), str(when or "").strip()
        end = _period_end(when)
        if name and end is not None and end >= day:
            found.append((end, len(found), source, f"{name} ({when})"))

    if strategy == "tiggre":
        import tiggre as tg
        done = (tg.CAT_DELIVERED, tg.CAT_LATE_1, tg.CAT_LATE_2, tg.CAT_MISSED)
        for c in row.get("catalysts") or []:
            if isinstance(c, dict) and c.get("status") not in done:
                _add(c.get("name"), c.get("date"), SHEET_LABEL["tiggre"])
    elif strategy == DURRETT_SHEET_STRATEGY:
        from types import SimpleNamespace
        from engines.durrett import thesis
        cats = row.get("catalysts") if isinstance(row.get("catalysts"), list) else []
        for c in thesis.catalysts(SimpleNamespace(catalysts=cats)):
            _add(c.name, c.expected, SHEET_LABEL[DURRETT_SHEET_STRATEGY])
    _add(row.get("fin_catalyst_text"), row.get("fin_catalyst_date"),
         f"{SHEET_LABEL.get(strategy, strategy)} · DS-finansiering")
    if not found:
        return None
    best = min(found)
    return best[2], best[3]


def _trigger_hint(data: dict, cards: dict, entry: dict, strategy: str, current: str) -> None:
    """Förslaget under Trigger / katalysator: nästa väntande katalysator ur arket — skrivs in först med 'Använd'.
    Döljs när rutan redan har exakt den texten."""
    try:
        hit = next_catalyst(entry, strategy)
    except Exception:
        return
    if not hit or (current or "").strip() == hit[1]:
        return
    source, text = hit
    a, b = st.columns([3, 1], vertical_alignment="center")
    shown = f"🤖 Ur {source}: {text}".replace("$", "\\$")
    a.caption(f"{shown} — nästa väntande katalysator, förslag, inte inskrivet.")
    b.button("Använd", key=f"sc_usetrig_{entry['key']}", on_click=_use_card,
             args=(data, cards, entry, strategy, "trigger", text, f"sc_trig_{entry['key']}"))


def _chatgpt(entry: dict, card: dict, data: Optional[dict] = None) -> None:
    """📋 Underlag till ChatGPT för hela köpbeslutet: kandidatens ark plus kortet. Svaret läses in här i kortets
    egna fält (värdering, trigger) och i arkens rader (review.routing) — kvittot har 💾 för alla berörda ark.

    Sifferuppdateringen läses inte här (blob=None) — Börsdata-förslagen med "Använd" finns i respektive ark.
    """
    try:
        from review import catalog as rcat
        from review import chatgpt_ui
    except Exception:
        return
    sheets = [rcat.STRATEGY_SHEET[s] for s in entry.get("strategies") or [] if s in rcat.STRATEGY_SHEET]
    rows = {rcat.STRATEGY_SHEET[s]: r for s, r in (entry.get("rows") or {}).items()
            if rcat.STRATEGY_SHEET.get(s) in rcat.FLAT_SHEETS and isinstance(r, dict)}
    cards = (data or {}).get("cards") if isinstance((data or {}).get("cards"), dict) else None

    def _target():
        """Kortet i lagret — skapas först när svaret skrivs in (ett orört kort skrivs aldrig)."""
        if not isinstance(cards.get(entry["key"]), dict):
            cards[entry["key"]] = {"strategy": card.get("strategy")}
        return cards[entry["key"]]
    chatgpt_ui.render_export(entry.get("ticker", ""), None, sheets + ["scorecard"], key=f"scorecard_{entry['key']}",
                             name=entry.get("name", ""), blob=None, rows={**rows, "scorecard": card}, entry=entry,
                             targets={"scorecard": _target} if cards is not None else {},
                             reset=([entry["key"]], ()), on_apply=lambda: _save(data))


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
        card = _shown(cards.get(e["key"], {}), e)
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
