"""
tabs/copilot.py — AI Trading Copilot
======================================
Interaktiv kandidatanalys med deterministisk regelkontroll och AI-kommentar.

Flöde:
    1. Välj strategi + ange ticker
    2. Marknadsdata hämtas (market_data) — ATR, EMA, volym, swing-nivåer
    3. Fyll i prisdata; levels.py räknar fram alternativa stoppnivåer
    4. Regelkontroll per regel (PASS / MANUAL / FAIL) mot verkliga tal
    5. Kandidatkort med rekommendation
    6. AI-kommentar på knapptryck — kommenterar, räknar inte om
    7. Journal med utfall, statistik och AI-granskning

Arbetsdelningen är CLAUDE.md:s: motorerna äger besluten, AI:n förklarar dem.
Varje siffra modellen ser är redan uträknad här, och prompten förbjuder den att
räkna om R:R, risk eller regelutfall.

Journalen ligger i data/copilot_journal.json via storage.py. Den låg tidigare i
en lokal fil, vilket på Streamlit Cloud betydde att den försvann vid varje
omstart — och omstart sker vid varje commit till deploy-branchen.
"""
from __future__ import annotations

import json
import os
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import streamlit as st

from strategy_rules import PLAYBOOKS, Playbook
from ui.theme import section_title, PALETTE as _P
from ai import copilot_prompt, openai_client

import cycle
import journal_stats
import review_link
import swing_verdict
import levels
import market_data
import storage
import storage_ui

# ── Palette shortcuts ─────────────────────────────────────────────────────────
_GREEN  = _P.get("green",    "#2d8a4e")
_RED    = _P.get("red",      "#c44545")
_AMBER  = _P.get("amber",    "#d4943a")
_CYAN   = _P.get("gold",     "#00E5FF")
_DIM    = _P.get("text_dim", "#6B7280")
_BG     = "#1A1F25"
_BORDER = "rgba(255,255,255,0.06)"

# ── Journal storage ───────────────────────────────────────────────────────────
STORE = "copilot_journal"       # data/copilot_journal.json

# Den gamla lokala filen. Streamlit Cloud har ett flyktigt filsystem och startar
# om appen vid varje commit till deploy-branchen, så allt som skrevs hit
# försvann vid nästa deploy. Den läses fortfarande EN gång, för att rädda det
# som råkar finnas kvar i en levande container.
_LEGACY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".copilot_journal.json",
)


def _legacy_entries() -> list[dict]:
    if not os.path.exists(_LEGACY_PATH):
        return []
    try:
        with open(_LEGACY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_journal() -> list[dict]:
    """Journalen ur sessionen; laddas en gång per session ur repot."""
    data = storage.session_load(STORE, {"entries": []})
    if not isinstance(data, dict):
        data = {"entries": []}
        st.session_state[STORE] = data
    data.setdefault("entries", [])

    if not data["entries"]:
        rescued = _legacy_entries()
        if rescued:
            data["entries"] = rescued
            st.info(f"Hittade {len(rescued)} poster i den gamla lokala "
                    f"journalfilen. De ligger nu i sessionen — tryck 💾 Spara "
                    f"så hamnar de i repot och överlever nästa omstart.")
    return data["entries"]


def _save_journal(entries: list[dict]) -> None:
    """Skriver till sessionen. Persistensen sker via 💾 Spara.

    Medvetet ingen nätverkstrafik här — en commit per tangenttryck slår i
    GitHubs rate limits.
    """
    data = st.session_state.setdefault(STORE, {"entries": []})
    data["entries"] = entries


# ── Rule check logic ──────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    number: int
    text: str
    status: str          # "PASS" | "MANUAL" | "FAIL"
    note: str = ""
    hard: bool = False


def _trend_check(snap) -> Optional[tuple]:
    """(status, notering) för en trendregel — eller None utan data."""
    if snap is None or snap.dist_ema200_pct is None:
        return None
    above200 = snap.above_ema200
    above50 = snap.above_ema50
    if above200 and above50:
        return ("PASS", f"Kurs {snap.dist_ema50_pct:+.1f} % mot EMA50, "
                        f"{snap.dist_ema200_pct:+.1f} % mot EMA200")
    if not above200:
        return ("FAIL", f"Kurs {snap.dist_ema200_pct:+.1f} % mot EMA200 — "
                        f"under den långa trenden")
    return ("MANUAL", f"Över EMA200 men {snap.dist_ema50_pct:+.1f} % mot EMA50 "
                      f"— trenden är inte entydig")


def _volume_check(snap) -> Optional[tuple]:
    if snap is None or snap.vol_ratio is None:
        return None
    if snap.vol_ratio >= 1.2:
        return ("PASS", f"Volym {snap.vol_ratio:.2f}× 20-dagarssnittet")
    if snap.vol_ratio < 0.8:
        return ("FAIL", f"Volym {snap.vol_ratio:.2f}× snittet — ingen "
                        f"bekräftelse i omsättningen")
    return ("MANUAL", f"Volym {snap.vol_ratio:.2f}× snittet — varken "
                      f"bekräftelse eller varning")


def _match_swing(text_lower: str, checks) -> Optional[tuple]:
    """Momentum-regeln som matchar texten, ur swing_verdict.rule_checks."""
    if not checks:
        return None
    if "positionsstorlek" in text_lower:
        return checks.get("positionsstorlek")
    if "marknadsfilt" in text_lower:      # "marknadsfiltret" böjs
        return checks.get("marknadsfilter")
    if "topp 20" in text_lower or "ranking" in text_lower:
        return checks.get("ranking")
    if "setup" in text_lower:
        return checks.get("setup")
    if "köp per vecka" in text_lower:
        return checks.get("köp per vecka")
    if "positioner" in text_lower:
        return checks.get("positioner")
    return None


def _check_rules(pb: Playbook, entry: float, stop: float, target: float,
                 snap=None, swing_checks=None) -> list[RuleResult]:
    """
    Deterministisk regelkontroll.

    Regler som går att räkna räknas. Med en ögonblicksbild (market_data) kan
    även trend- och volymreglerna avgöras mekaniskt i stället för att skickas
    vidare som MANUAL — utan den föll de tidigare igenom obesvarade.

    En regel markeras bara PASS eller FAIL när det finns ett tal bakom
    beslutet. Saknas datan står den kvar som MANUAL: en obesvarad fråga får
    inte se ut som ett godkännande.
    """
    results: list[RuleResult] = []

    risk_pct = levels.risk_pct(entry, stop)
    ratio = levels.rr(entry, stop, target)

    for r in pb.entry:
        text_lower = r.text.lower()

        # Momentum: regim, ranking, setup, veckotak och positionstak läses
        # ur Swing Regime, Swing Screener och Swing-flikens positioner.
        matched = _match_swing(text_lower, swing_checks)
        if matched is not None:
            results.append(RuleResult(r.number, r.text, matched[0],
                                      matched[1], r.hard))
            continue

        # R:R-kontroll
        if "1:2" in r.text or "r:r" in text_lower or "reward" in text_lower:
            if ratio is None:
                results.append(RuleResult(r.number, r.text, "MANUAL",
                                          "Fyll i entry, stop och target", r.hard))
            elif round(ratio, 6) >= levels.RR_MIN:
                results.append(RuleResult(r.number, r.text, "PASS",
                                          f"R:R = {ratio:.1f}x ✓", r.hard))
            else:
                need = levels.target_for_rr(entry, stop, levels.RR_MIN)
                results.append(RuleResult(r.number, r.text, "FAIL",
                                          f"R:R = {ratio:.1f}x — kräver "
                                          f"≥ {levels.RR_MIN:g} (target {need:g})",
                                          r.hard))

        # Stop-nivå
        elif "stop" in text_lower and risk_pct is not None:
            results.append(RuleResult(r.number, r.text, "PASS",
                                      f"Stop = {risk_pct:.1f} % från entry", r.hard))

        # Trend — går att avgöra så fort vi har kursdata
        elif any(w in text_lower for w in ("trend", "ma200", "ma50", "ema")):
            checked = _trend_check(snap)
            results.append(RuleResult(
                r.number, r.text, checked[0] if checked else "MANUAL",
                checked[1] if checked else
                "Kontrollera i panelen: " + r.panel_guide[:80], r.hard))

        # Volym
        elif "volym" in text_lower or "volume" in text_lower:
            checked = _volume_check(snap)
            results.append(RuleResult(
                r.number, r.text, checked[0] if checked else "MANUAL",
                checked[1] if checked else
                "Kontrollera i panelen: " + r.panel_guide[:80], r.hard))

        # Allt annat kräver manuell koll
        else:
            results.append(RuleResult(r.number, r.text, "MANUAL",
                                      "Kontrollera i panelen: " + r.panel_guide[:80],
                                      r.hard))

    return results


# ── Marknadsdata och nivåer ──────────────────────────────────────────────────

def _prefill_entry(ticker: str, snap, entry: float) -> None:
    """Förifyll entry med aktuell kurs — en gång per ticker.

    Bara när fältet står orört på noll: har användaren skrivit ett eget värde,
    eller nollat det med flit, ska panelen inte skriva över det.
    """
    if snap is None or entry:
        return
    if st.session_state.get(_PREFILL_KEY) == ticker:
        return
    st.session_state[_PREFILL_KEY] = ticker
    _queue_prices(entry=snap.price)


def _render_cycle(ticker: str, strategy_key: str):
    """Cykelläget för råvarustrategierna: (state, råvarunamn).

    Råvaran slås upp i Rick Rule-arket där den redan är ifylld; listan är
    fallback. Statusen läses ur rotationsflikens sparade betyg — sätts den
    om där, ändras den här.
    """
    if not cycle.requires_cycle(strategy_key):
        return None, ""

    import producers as prod_mod
    import rotation as rot_mod
    producers_data = storage.session_load(
        prod_mod.STORE, {"producers": [], "royalty": []})
    rotation_data = storage.session_load(
        rot_mod.STORE, {"grades": {}, "history": [], "month": ""})

    looked_up = cycle.commodity_for_ticker(ticker, producers_data)
    names = ["– välj råvara –"] + [c.name for c in rot_mod.COMMODITIES]
    idx = names.index(looked_up) if looked_up in names else 0
    chosen = st.selectbox(
        "Råvara (för cykelläget)", names, index=idx, key="copilot_commodity",
        help="Hämtas från Rick Rule-arket när bolaget finns där. Cykelläget "
             "kommer från rotationsflikens Triple Signal-betyg.")
    name = "" if chosen == names[0] else chosen

    state = cycle.cycle_state(name, rotation_data) if name else None
    if state is not None:
        color = rot_mod.STATUS_COLOR.get(state["status"], _DIM)
        warn = "".join(f"<div style='color:{_AMBER};font-size:11px;'>⚠️ {w}"
                       f"</div>" for w in state["warnings"])
        st.markdown(
            f'<div style="border:1px solid {color}55;background:{color}0d;'
            f'border-radius:8px;padding:8px 12px;margin:4px 0 10px;">'
            f'<span style="color:{color};font-weight:700;">Cykelläge '
            f'{state["commodity"]}: {state["status"]} '
            f'{state["sum"]}/{state["max"]}</span>'
            f'<div style="color:{_DIM};font-size:11px;margin-top:2px;">'
            f'{state["why"]} · betyg {state["month"] or "okänd månad"}</div>'
            f'{warn}</div>', unsafe_allow_html=True)
    elif name:
        st.caption(f"{name} är inte betygsatt i rotationsfliken ännu — "
                   f"cykelregeln står som MANUAL tills det är gjort.")
    return state, name


def _render_review(ticker: str, strategy_key: str):
    """Granskningsboxen: arkets eget utfall plus DS/AQS/CSM, läst ur samma
    session som granskningsfliken skriver till. Returnerar review-dicten."""
    if not review_link.has_review(strategy_key):
        return None
    store_name, _label = review_link.SHEET[strategy_key]
    stores = {store_name: storage.session_load(
        store_name, review_link.STORE_DEFAULTS[store_name])}
    rev = review_link.review(strategy_key, ticker, stores)
    if rev is None:
        return None

    color = {_c: v for _c, v in (("PASS", _GREEN), ("MANUAL", _AMBER),
                                 ("FAIL", _RED))}[rev["status"]]
    controls_line = " · ".join(
        f"{label} {status}" for status, label, _n in rev["controls"]) or ""
    st.markdown(
        f'<div style="border:1px solid {color}55;background:{color}0d;'
        f'border-radius:8px;padding:8px 12px;margin:4px 0 10px;">'
        f'<span style="color:{color};font-weight:700;">Granskningen — '
        f'{rev["sheet"]}: {rev["status"]}</span>'
        f'<div style="color:{_DIM};font-size:11px;margin-top:2px;">'
        f'{rev["note"]}</div>'
        + (f'<div style="color:{_DIM};font-size:11px;">{controls_line}</div>'
           if controls_line else "")
        + '</div>', unsafe_allow_html=True)
    if rev["found"]:
        details = review_link.detail_lines(strategy_key, rev["row"])
        if details:
            with st.expander("Granskningens underlag — det arket redan svarat på",
                             expanded=False):
                for line in details:
                    st.caption(line.strip())
    return rev


def _render_market_phase(ticker: str, strategy_key: str):
    """Fas-boxen för contrarian/quality. Returnerar fas-dicten eller None.

    Motorn är densamma som REGIME → Market Cycle och cachas en timme — samma
    fas där som här.
    """
    if not cycle.requires_market_cycle(strategy_key):
        return None
    state, err = cycle.market_phase(ticker)
    if err:
        st.warning(f"Marknadscykelfasen kunde inte läsas: {err}")
        return None
    sets = cycle.playbook_phases(strategy_key)
    phase = state["phase"]
    color = (_GREEN if phase in sets["buy"] else
             _RED if phase in sets["sell"] else _AMBER)
    st.markdown(
        f'<div style="border:1px solid {color}55;background:{color}0d;'
        f'border-radius:8px;padding:8px 12px;margin:4px 0 10px;">'
        f'<span style="color:{color};font-weight:700;">Marknadscykelfas: '
        f'{phase.replace("_", " ")} · {state["confidence"]:g} % säkerhet</span>'
        f'<div style="color:{_DIM};font-size:11px;margin-top:2px;">'
        f'Market Cycle Engine — samma motor som REGIME-fliken. Playbookens '
        f'köpfaser: {", ".join(sorted(sets["buy"])) or "–"}.</div></div>',
        unsafe_allow_html=True)
    return state


def _swing_rule_checks(ticker: str, strategy_key: str):
    """Momentum-reglernas underlag: regimljus, ranking, setup, veckotak.

    Läser samma sessioncachade data som REGIME- och SCREENING-flikarna, plus
    swing-lagret. Bara för momentum — övriga strategier får None och
    regelkontrollen faller tillbaka på sina vanliga vägar.
    """
    if strategy_key != "momentum":
        return None
    try:
        import wolf_regime_ui
        import wolf_screener_ui
        regime_data = wolf_regime_ui._get_data() or {}
        screener_data = wolf_screener_ui._get_data() or {}
    except Exception:
        regime_data, screener_data = {}, {}
    swing_data = storage.session_load(
        "swing", {"positions": [], "market": {}, "watchlist": [],
                  "closed": [], "checklist": {}})
    checks = swing_verdict.rule_checks(ticker, screener_data, regime_data,
                                       swing_data)

    # Domboxen — samma funktion som Swing Regime-flikens ticker-koll.
    v = swing_verdict.verdict(ticker, screener_data, regime_data, swing_data)
    color = {swing_verdict.BUY: _GREEN, swing_verdict.HOLD: _GREEN,
             swing_verdict.PARTIAL: _AMBER, swing_verdict.WATCH: _AMBER,
             swing_verdict.SELL: _RED, swing_verdict.ABSTAIN: _RED,
             swing_verdict.UNKNOWN: _DIM}.get(v["verdict"], _DIM)
    tag = " · INNEHAV" if v["held"] else ""
    st.markdown(
        f'<div style="border:1px solid {color}55;background:{color}0d;'
        f'border-radius:8px;padding:8px 12px;margin:4px 0 10px;">'
        f'<span style="color:{color};font-weight:700;">Swing-dom{tag}: '
        f'{v["verdict"]}</span>'
        f'<div style="color:{_DIM};font-size:11px;margin-top:2px;">'
        f'{v["reasons"][0] if v["reasons"] else ""}</div></div>',
        unsafe_allow_html=True)
    return checks


def _render_market(snap, error: Optional[str]) -> None:
    """Ögonblicksbilden. Uteblir den syns det — en tyst tom ruta läses som
    att inget var anmärkningsvärt."""
    if error:
        st.warning(f"Ingen marknadsdata: {error}\n\nRegelkontrollen kör vidare, "
                   f"men trend- och volymreglerna kan inte avgöras utan kurs.")
        return
    if snap is None:
        return

    def _m(col, label, value, suffix="", fmt="{:.2f}", color="#E8EDF2"):
        text = "–" if value is None else fmt.format(value) + suffix
        col.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-size:17px;font-weight:700;color:{color};">{text}</div>'
            f'<div style="font-size:10px;color:{_DIM};">{label}</div></div>',
            unsafe_allow_html=True)

    with st.expander(f"📊 Marknadsdata — {snap.ticker} per {snap.as_of}",
                     expanded=True):
        c = st.columns(6)
        _m(c[0], "Kurs", snap.price)
        _m(c[1], "ATR(14)", snap.atr_pct, " %",
           color=_AMBER if (snap.atr_pct or 0) > 5 else "#E8EDF2")
        _m(c[2], "mot EMA50", snap.dist_ema50_pct, " %", "{:+.1f}",
           _GREEN if snap.above_ema50 else _RED)
        _m(c[3], "mot EMA200", snap.dist_ema200_pct, " %", "{:+.1f}",
           _GREEN if snap.above_ema200 else _RED)
        _m(c[4], "RSI(14)", snap.rsi14, "", "{:.0f}")
        _m(c[5], "Volym", snap.vol_ratio, "×",
           color=_GREEN if (snap.vol_ratio or 0) >= 1.2 else _DIM)
        st.caption(f"52 v {snap.low_52w:.2f}–{snap.high_52w:.2f} · "
                   f"{snap.from_high_pct:+.1f} % från toppen · "
                   f"swing-low 20 d {snap.swing_low_20:.2f} · "
                   f"EMA, inte SMA — samma definition som regimmotorn.")


def _render_levels(entry: float, stop: float, target: float, snap,
                   pb: Playbook) -> None:
    """Räknade stoppnivåer och vad de valda innebär.

    Panelen väljer inte åt dig. Att se att en ATR-stop kostar 8 % risk medan
    swing-low kostar 4 % ÄR beslutsunderlaget.
    """
    fixed = _fixed_stop_pct(pb)
    alts = (levels.stop_candidates(entry, snap, fixed, _atr_mult(pb))
            if entry else [])
    if not alts and not entry:
        return

    with st.expander("🎯 Entry- och exitnivåer — räknade", expanded=bool(alts)):
        if not alts:
            st.caption("Fyll i entry, och hämta kursdata, så räknas "
                       "stoppnivåerna fram.")
        for i, s in enumerate(alts):
            chosen = stop and abs(s.price - stop) / s.price < 0.005
            mark = " ← din nivå" if chosen else ""
            c_txt, c_btn = st.columns([5, 1])
            c_txt.markdown(
                f'<div style="border-left:2px solid '
                f'{_CYAN if chosen else _BORDER};padding:6px 0 6px 12px;'
                f'margin-bottom:8px;">'
                f'<span style="font-size:13px;font-weight:700;color:#E8EDF2;">'
                f'{s.name} {s.price:g}</span>'
                f'<span style="font-size:11px;color:{_AMBER};margin-left:10px;">'
                f'{s.risk_pct:.1f} % risk</span>'
                f'<span style="font-size:11px;color:{_CYAN};margin-left:10px;">'
                f'{mark}</span>'
                f'<div style="font-size:11px;color:{_DIM};margin-top:2px;">'
                f'{s.why} Target {s.target_for_min_rr:g} för '
                f'{levels.RR_MIN:g}:1 · {s.target_for_pref_rr:g} för '
                f'{levels.RR_PREFERRED:g}:1.</div></div>',
                unsafe_allow_html=True)
            if c_btn.button("Använd", key=f"copilot_use_stop_{i}",
                            help=f"Sätter stop {s.price:g}"
                                 + ("" if target else
                                    f" och target {s.target_for_min_rr:g} "
                                    f"({levels.RR_MIN:g}:1)")):
                # Motorn räknade nivån; knappen bara flyttar den till fälten.
                # Target fylls bara i när fältet är tomt — ett eget target
                # skrivs aldrig över av en stoppändring.
                _queue_prices(stop=s.price,
                              target=None if target else s.target_for_min_rr)

        if entry and stop:
            t1, t2, t3 = st.columns([1, 1, 3])
            if t1.button(f"Target {levels.RR_MIN:g}:1", key="copilot_t_min"):
                _queue_prices(target=levels.target_for_rr(entry, stop,
                                                          levels.RR_MIN))
            if t2.button(f"Target {levels.RR_PREFERRED:g}:1",
                         key="copilot_t_pref"):
                _queue_prices(target=levels.target_for_rr(entry, stop,
                                                          levels.RR_PREFERRED))
            t3.caption("Räknat från din nuvarande stop — byter du stop, "
                       "tryck igen.")

        assessment = levels.assess(entry, stop, target, snap)
        for note in assessment.notes:
            st.warning(note)
        if not assessment.notes and assessment.rr is not None:
            st.success(f"R:R {assessment.rr:.1f}x — inga invändningar mot "
                       f"nivåerna.")


def _fixed_stop_pct(pb: Playbook) -> Optional[float]:
    """Strategins fasta stoppavstånd, om den har ett.

    Läses ur strategy_rules.py i stället för att hårdkodas här — den filen är
    källan, och en ändrad tröskel ska slå igenom utan att någon minns att
    Copiloten har en egen kopia.

    Bara NEGATIVA procenttal räknas: momentums stop är "−10 % från entry", och
    "+20 %" i samma mening är nivån där stoppen flyttas till break-even. Utan
    tecknet hade den lästs som ett stoppavstånd.
    """
    import re
    m = re.search(r"[-−]\s*(\d+(?:[.,]\d+)?)\s*%", str(pb.risk.stop))
    return float(m.group(1).replace(",", ".")) if m else None


def _atr_mult(pb: Playbook) -> Optional[float]:
    """Strategins ATR-multipel: Viking 1,5×, Wolf 2,5×."""
    import re
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*[×xX]\s*ATR", str(pb.risk.stop))
    return float(m.group(1).replace(",", ".")) if m else None


# ── Candidate card ────────────────────────────────────────────────────────────

def _status_color(status: str) -> str:
    return {"PASS": _GREEN, "MANUAL": _AMBER, "FAIL": _RED}.get(status, _DIM)


def _render_rule_row(res: RuleResult) -> None:
    col_s, col_t, col_n = st.columns([1, 4, 4])
    color = _status_color(res.status)
    badge = (
        f'<span style="background:{color}22;color:{color};font-size:10px;'
        f'font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:1px;">'
        f'{res.status}</span>'
    )
    with col_s:
        st.markdown(badge, unsafe_allow_html=True)
    with col_t:
        hard_marker = " 🔒" if res.hard else ""
        st.markdown(
            f'<span style="font-size:12px;color:#E8EDF2;">{res.text}{hard_marker}</span>',
            unsafe_allow_html=True,
        )
    with col_n:
        st.markdown(
            f'<span style="font-size:11px;color:{_DIM};">{res.note}</span>',
            unsafe_allow_html=True,
        )


def _overall_status(results: list[RuleResult]) -> str:
    if any(r.status == "FAIL" and r.hard for r in results):
        return "REJECT"
    if any(r.status == "FAIL" for r in results):
        return "REJECT"
    if any(r.status == "MANUAL" for r in results):
        return "WATCH"
    return "BUY CANDIDATE"


def _status_to_color(s: str) -> str:
    return {
        "BUY CANDIDATE": _GREEN,
        "WATCH":         _AMBER,
        "REJECT":        _RED,
    }.get(s, _DIM)


# ── AI comment stub ───────────────────────────────────────────────────────────

def _ai_comment(ticker: str, pb: Playbook, results: list[RuleResult],
                entry: float, stop: float, target: float) -> str:
    """
    Stub — returnerar deterministiskt genererad kommentar.
    Ersätt med ett GPT-anrop när du vill ha riktig AI-text.

    Byt ut denna funktion mot:
        import openai
        response = openai.chat.completions.create(...)
        return response.choices[0].message.content
    """
    passed  = [r.text for r in results if r.status == "PASS"]
    failed  = [r.text for r in results if r.status == "FAIL"]
    manual  = [r.text for r in results if r.status == "MANUAL"]
    rr      = abs(target - entry) / abs(entry - stop) if (entry and stop and entry != stop) else 0
    risk_p  = abs(entry - stop) / entry * 100 if entry else 0

    lines = [f"**{ticker}** analyseras mot **{pb.name}**.", ""]

    if passed:
        lines.append(f"✅ Automatiska kontroller godkända: {len(passed)} regler klarade.")
    if manual:
        lines.append(f"⚠️ {len(manual)} regler kräver manuell kontroll i panelen.")
    if failed:
        lines.append(f"❌ {len(failed)} hårda regler misslyckades — affären bör undvikas.")

    lines += [
        "",
        f"Risk per affär: **{risk_p:.1f} %** av entry.",
        f"R:R-förhållande: **{rr:.1f}x** (kräver ≥ 2,0 för PASS).",
    ]

    if rr >= 3.0:
        lines.append("📈 Utmärkt R:R — affären erbjuder bra asymmetri.")
    elif rr >= 2.0:
        lines.append("📊 Godkänt R:R — klara alla MANUAL-regler innan du agerar.")
    else:
        lines.append("🚫 R:R under 2,0 — leta en bättre entry eller ett vidare target.")

    lines += [
        "",
        "_Deterministisk sammanfattning — räknad av panelen, inte skriven av "
        "en modell._",
    ]

    return "\n".join(lines)


# ── AI-kommentar: knappstyrd, aldrig i renderingsvägen ────────────────────────

def _ai_cache_key(ticker: str, strategy_key: str,
                  entry: float, stop: float, target: float,
                  as_of: str = "") -> str:
    """Ändras underlaget blir det gamla svaret ogiltigt.

    Utan detta ligger en kommentar om entry 100 kvar när du ändrat till 120,
    och den ser lika auktoritativ ut som när den skrevs.
    """
    return (f"{strategy_key}|{ticker.upper()}|{entry:g}|{stop:g}|{target:g}"
            f"|{as_of}")


def _render_ai_section(ticker: str, strategy_key: str, pb: Playbook,
                       results: list[RuleResult],
                       entry: float, stop: float, target: float,
                       snap=None, cyc_state=None, bspot=None,
                       rev=None, phase_state=None) -> None:
    """Deterministisk sammanfattning alltid; modellsvar på knapptryck.

    Anropet ligger BAKOM en knapp med flit. Streamlit kör om hela skriptet vid
    varje widget-interaktion — ett anrop i renderingsvägen hade blivit ett
    betalt API-anrop varje gång du rör ett reglage.
    """
    st.markdown(_ai_comment(ticker, pb, results, entry, stop, target))

    cache_key = _ai_cache_key(ticker, strategy_key, entry, stop, target,
                              getattr(snap, 'as_of', '') or '')
    store = st.session_state.setdefault("copilot_ai", {})
    stale = store.get("key") and store["key"] != cache_key

    col_b, col_s = st.columns([1, 3])
    with col_b:
        asked = st.button("🤖 Fråga modellen", key="copilot_ask",
                          type="secondary",
                          disabled=not openai_client.configured())
    with col_s:
        if not openai_client.configured():
            st.caption(f"Modellen är inte påslagen — {openai_client.KEY_NAME} "
                       f"saknas i secrets. Panelen fungerar utan den.")
        elif stale:
            st.caption("Underlaget har ändrats sedan svaret nedan skrevs.")

    if asked:
        with st.spinner("Analyserar…"):
            try:
                reply = openai_client.complete(
                    copilot_prompt.SYSTEM,
                    copilot_prompt.build_prompt(
                        ticker=ticker, strategy=pb.name,
                        status=_overall_status(results),
                        entry=entry, stop=stop, target=target,
                        rr=levels.rr(entry, stop, target),
                        risk_pct=levels.risk_pct(entry, stop),
                        passed=[r.text for r in results if r.status == "PASS"],
                        manual=[r.text for r in results if r.status == "MANUAL"],
                        failed=[r.text for r in results if r.status == "FAIL"],
                        risk_per_trade=pb.risk.risk_per_trade,
                        snapshot=snap,
                        assessment=levels.assess(entry, stop, target, snap),
                        alternatives=levels.stop_candidates(
                            entry, snap, _fixed_stop_pct(pb), _atr_mult(pb)),
                        cycle_state=cyc_state, blindspot=bspot,
                        review_lines=review_link.prompt_lines(rev, strategy_key),
                        market_phase=phase_state))
            except openai_client.AIError as exc:
                # Ingen tyst fallback. Uteblir svaret ska det synas att det
                # uteblev — annars läses stubben ovanför som modelltext.
                st.error(f"Ingen AI-kommentar: {exc}")
                return
            store.update({"key": cache_key, "text": reply.text,
                          "model": reply.model})
            stale = False

    if store.get("text"):
        st.markdown(
            f"<div style='border:1px solid {_BORDER};background:{_BG};"
            f"border-radius:10px;padding:14px 16px;margin-top:12px;"
            f"opacity:{'0.55' if stale else '1'};'>"
            f"<div style='font-size:11px;color:{_DIM};margin-bottom:6px;'>"
            f"{store.get('model', '')}"
            f"{' · gäller ett tidigare underlag' if stale else ''}</div></div>",
            unsafe_allow_html=True)
        st.markdown(store["text"])


# ── Journal section ───────────────────────────────────────────────────────────

def _render_journal_log(ticker: str, strategy_key: str,
                        entry: float, stop: float, target: float) -> None:
    """Snabb loggning av trade-kandidat till journal."""
    section_title("📓 Logga kandidat", "")
    st.caption("Spara kandidaten till din lokala journal för uppföljning.")

    with st.form("copilot_log_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            log_date   = st.date_input("Datum", value=datetime.date.today())
            log_ticker = st.text_input("Ticker", value=ticker.upper())
        with col2:
            log_strat  = st.selectbox(
                "Strategi",
                options=list(PLAYBOOKS.keys()),
                index=list(PLAYBOOKS.keys()).index(strategy_key) if strategy_key in PLAYBOOKS else 0,
                format_func=lambda k: PLAYBOOKS[k].name,
            )
            log_shares = st.number_input(
                "Antal aktier (valfritt)", min_value=0, step=1, value=0,
                help="Krävs för vinstandel och payoff-kvot, som räknas i "
                     "kronor. R-multipeln fungerar utan.")
            log_note   = st.text_input("Anteckning (valfri)")

        submitted = st.form_submit_button("💾 Spara till journal", type="primary")
        if submitted:
            entries = _load_journal()
            entries.append({
                "date":     str(log_date),
                "ticker":   log_ticker.upper(),
                "strategy": log_strat,
                "entry":    entry,
                "stop":     stop,
                "target":   target,
                "shares":   int(log_shares) or None,
                "note":     log_note,
                "logged_at": datetime.datetime.utcnow().isoformat(),
            })
            _save_journal(entries)
            st.success(f"✅ {log_ticker.upper()} loggad!")


def _with_metrics(entries: list[dict]) -> list[dict]:
    """Journalposterna med utfallen uträknade.

    Räknas vid varje anrop i stället för att lagras: R-multipeln mäts mot
    stoppen du faktiskt la, och en lagrad kopia hade blivit fel så fort en
    rad rättades.
    """
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        row = dict(e)
        entry, stop = e.get("entry"), e.get("stop")
        exit_price = e.get("exit_price")
        if exit_price:
            row["pnl_pct"] = journal_stats.pnl_pct(entry, exit_price)
            row["r_multiple"] = journal_stats.r_multiple(entry, stop,
                                                         exit_price)
            row["holding_days"] = journal_stats.holding_days(
                e.get("date"), e.get("exit_date"))
            # Vinstandel och payoff-kvot räknas i KRONOR i journal_stats, så de
            # kräver antal aktier. Utan det förblir de None — och UI:t säger
            # varför i stället för att visa ett streck utan förklaring.
            row["pnl_amount"] = journal_stats.pnl_amount(
                entry, exit_price, e.get("shares"), e.get("fees") or 0)
        out.append(row)
    return out


def _render_journal_stats(entries: list[dict]) -> None:
    """Statistikbladet plus AI-granskningen.

    Det är HÄR avkastningen faktiskt mäts. En kommentar före ett köp är en
    gissning; det här är utfallen.
    """
    rows = _with_metrics(entries)
    stats = journal_stats.summary(rows)

    if not stats["closed"]:
        st.caption("Inga avslutade affärer ännu. Fyll i säljkurs på en post "
                   "nedan, så börjar statistiken räknas.")
        return

    def _m(col, label, value, suffix="", fmt="{:.1f}"):
        text = "–" if value is None else fmt.format(value) + suffix
        col.metric(label, text)

    c = st.columns(5)
    _m(c[0], "Avslutade", stats["closed"], "", "{:.0f}")
    _m(c[1], "Vinstandel", stats["win_rate"], " %")
    _m(c[2], "Payoff", stats["payoff"], "", "{:.2f}")
    _m(c[3], "Snitt-R", stats["avg_r"], "R", "{:.2f}")
    _m(c[4], "Innehav", stats["avg_days"], " d", "{:.0f}")

    if stats["win_rate"] is None:
        st.caption("Vinstandel och payoff-kvot räknas i kronor och kräver "
                   "antal aktier på posten. Fyll i det när du loggar, så "
                   "börjar de räknas — R-multipeln fungerar redan utan.")
    elif not stats["enough"]:
        st.caption(f"Under {journal_stats.MIN_TRADES} avslutade affärer är "
                   f"vinstandel och payoff-kvot brus. Siffrorna visas, men "
                   f"dra inga slutsatser av dem än.")

    closed = [r for r in rows if r.get("exit_price")]
    _render_review_button(stats, closed)


def _render_review_button(stats: dict, closed: list) -> None:
    """AI-granskning av journalen. Knappstyrd, som allt annat som kostar."""
    store = st.session_state.setdefault("copilot_review", {})
    key = f"{stats['closed']}|{stats.get('win_rate')}|{stats.get('avg_r')}"

    col_b, col_s = st.columns([1, 3])
    with col_b:
        asked = st.button("🔎 Granska journalen", key="copilot_review_btn",
                          disabled=not openai_client.configured())
    with col_s:
        if not openai_client.configured():
            st.caption("Kräver OPENAI_API_KEY i secrets.")
        elif store.get("key") and store["key"] != key:
            st.caption("Journalen har ändrats sedan granskningen nedan.")

    if asked:
        with st.spinner("Läser journalen…"):
            try:
                reply = openai_client.complete(
                    copilot_prompt.REVIEW_SYSTEM,
                    copilot_prompt.build_review_prompt(
                        stats, list(reversed(closed))[:30],
                        journal_stats.MIN_TRADES))
            except openai_client.AIError as exc:
                st.error(f"Ingen granskning: {exc}")
                return
            store.update({"key": key, "text": reply.text})

    if store.get("text"):
        st.markdown(store["text"])


def _render_journal_history() -> None:
    """Posterna, med möjlighet att stänga en affär."""
    entries = _load_journal()
    if not entries:
        st.caption("Inga poster ännu.")
        return

    for idx in range(len(entries) - 1, max(len(entries) - 21, -1), -1):
        e = entries[idx]
        ratio = levels.rr(e.get("entry"), e.get("stop"), e.get("target"))
        pb_name = PLAYBOOKS[e["strategy"]].name if e.get("strategy") in PLAYBOOKS \
            else e.get("strategy", "")
        closed = bool(e.get("exit_price"))
        pnl = journal_stats.pnl_pct(e.get("entry"), e.get("exit_price"))
        head = (f"{e.get('ticker', '?')} · {e.get('date', '')} · {pb_name}"
                + (f" · {pnl:+.1f} %" if pnl is not None else " · öppen"))

        with st.expander(head, expanded=False):
            st.caption(f"Entry {e.get('entry', '—')} · Stop {e.get('stop', '—')} "
                       f"· Target {e.get('target', '—')} · "
                       f"R:R {ratio:.1f}x" if ratio is not None else
                       f"Entry {e.get('entry', '—')} · Stop {e.get('stop', '—')}")
            if e.get("note"):
                st.caption(f"Anteckning: {e['note']}")

            c1, c2, c3 = st.columns(3)
            exit_price = c1.number_input(
                "Säljkurs", min_value=0.0, step=0.5, format="%.2f",
                value=float(e.get("exit_price") or 0.0),
                key=f"cj_exit_{idx}")
            exit_date = c2.date_input(
                "Säljdatum", key=f"cj_date_{idx}",
                value=datetime.date.fromisoformat(str(e["exit_date"])[:10])
                if e.get("exit_date") else datetime.date.today())
            sell_rule = c3.selectbox(
                "Säljregel", [""] + list(journal_stats.SELL_RULES),
                index=([""] + list(journal_stats.SELL_RULES)).index(
                    e.get("sell_rule", "")) if e.get("sell_rule", "") in
                journal_stats.SELL_RULES else 0,
                format_func=lambda r: journal_stats.SELL_RULE_LABEL.get(r, "—"),
                key=f"cj_rule_{idx}")

            d1, d2, d3 = st.columns(3)
            shares = d3.number_input(
                "Antal aktier", min_value=0, step=1,
                value=int(e.get("shares") or 0), key=f"cj_shares_{idx}",
                help="Krävs för vinstandel och payoff-kvot.")
            setup = d1.selectbox(
                "Setup", [""] + list(journal_stats.SETUPS),
                index=([""] + list(journal_stats.SETUPS)).index(e.get("setup", ""))
                if e.get("setup", "") in journal_stats.SETUPS else 0,
                format_func=lambda x: journal_stats.SETUP_LABEL.get(x, "—"),
                key=f"cj_setup_{idx}")
            followed = d2.selectbox(
                "Följde du planen?", ["", "Ja", "Nej"],
                index=["", "Ja", "Nej"].index(e.get("followed_plan", ""))
                if e.get("followed_plan", "") in ("Ja", "Nej") else 0,
                key=f"cj_plan_{idx}",
                help="Den enda frågan som mäter dig och inte marknaden.")

            new = {"exit_price": exit_price or None,
                   "exit_date": str(exit_date) if exit_price else None,
                   "sell_rule": sell_rule, "setup": setup,
                   "followed_plan": followed, "shares": int(shares) or None}
            if any(e.get(k) != v for k, v in new.items()):
                e.update(new)
                _save_journal(entries)

            if closed:
                r = journal_stats.r_multiple(e.get("entry"), e.get("stop"),
                                             e.get("exit_price"))
                st.caption(f"Resultat {pnl:+.1f} % · "
                           + (f"{r:+.2f}R" if r is not None else "R saknas"))

            if st.button("Ta bort posten", key=f"cj_del_{idx}"):
                entries.pop(idx)
                _save_journal(entries)
                st.rerun()


# ── Main render function ──────────────────────────────────────────────────────

_APPLY_KEY = "copilot_apply"
_PREFILL_KEY = "copilot_prefilled_for"


def _queue_prices(**values) -> None:
    """Lägg pris i väntkön och rita om.

    Streamlit tillåter inte att ett widgetvärde sätts efter att widgeten
    ritats i samma körning — därför går Använd-knapparna via en kö som töms
    överst i nästa körning, innan fälten skapas.
    """
    pending = st.session_state.setdefault(_APPLY_KEY, {})
    pending.update({k: v for k, v in values.items() if v is not None})
    st.rerun()


def _apply_queued_prices() -> None:
    """Töm väntkön in i prisfälten. Körs FÖRE widgetarna ritas."""
    pending = st.session_state.pop(_APPLY_KEY, None)
    if not pending:
        return
    for name, value in pending.items():
        st.session_state[f"copilot_{name}"] = round(float(value), 2)


def render_copilot_page() -> None:
    """Entry point — anropas från wolf_panel.py."""
    _apply_queued_prices()
    section_title("AI Trading Copilot", "🤖")
    st.markdown(
        f'<p style="color:{_DIM};font-size:0.82rem;margin:-8px 0 24px;">'
        f'Kandidatanalys · Regelkontroll · Riskkontroll · Journal</p>',
        unsafe_allow_html=True,
    )
    _load_journal()
    storage_ui.save_bar(STORE, "Copilot-journalen")

    # ── Inmatning ─────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([2, 3])
    with col_a:
        strategy_key = st.selectbox(
            "Strategi",
            options=list(PLAYBOOKS.keys()),
            format_func=lambda k: PLAYBOOKS[k].name,
            key="copilot_strategy",
        )
    with col_b:
        ticker = st.text_input(
            "Ticker",
            placeholder="t.ex. ERIC B, ABB, VOLV B",
            key="copilot_ticker",
        ).strip().upper()

    pb: Playbook = PLAYBOOKS[strategy_key]

    st.markdown(
        f'<div style="background:{pb.color}11;border-left:3px solid {pb.color};'
        f'border-radius:6px;padding:10px 14px;margin:8px 0 16px;">'
        f'<span style="font-size:12px;color:{pb.color};font-weight:700;">{pb.name}</span>'
        f'<span style="font-size:11px;color:{_DIM};margin-left:10px;">{pb.tagline}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Prisdata ──────────────────────────────────────────────────────────────
    with st.expander("📐 Prisdata & R:R-beräkning", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            entry = st.number_input("Entry-kurs (kr)", min_value=0.0,
                                    step=0.5, format="%.2f", key="copilot_entry")
        with c2:
            stop = st.number_input("Stop-kurs (kr)", min_value=0.0,
                                   step=0.5, format="%.2f", key="copilot_stop")
        with c3:
            target = st.number_input("Target-kurs (kr)", min_value=0.0,
                                     step=0.5, format="%.2f", key="copilot_target")

        if entry and stop and entry != stop:
            rr      = abs(target - entry) / abs(entry - stop) if target else 0
            risk_p  = abs(entry - stop) / entry * 100
            rr_col  = _GREEN if rr >= 2.0 else (_AMBER if rr >= 1.5 else _RED)
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:22px;font-weight:700;color:{rr_col};">'
                    f'{rr:.1f}x</div>'
                    f'<div style="font-size:11px;color:{_DIM};">R:R-förhållande</div></div>',
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:22px;font-weight:700;color:{_AMBER};">'
                    f'{risk_p:.1f} %</div>'
                    f'<div style="font-size:11px;color:{_DIM};">Risk från entry</div></div>',
                    unsafe_allow_html=True,
                )
            with mc3:
                upside = abs(target - entry) / entry * 100 if target else 0
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:22px;font-weight:700;color:{_GREEN};">'
                    f'+{upside:.1f} %</div>'
                    f'<div style="font-size:11px;color:{_DIM};">Potentiell vinst</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Regelkontroll ─────────────────────────────────────────────────────────
    if not ticker:
        st.info("👆 Ange en ticker ovan för att starta analysen.", icon="ℹ️")
        return

    if not pb.entry:
        st.warning("Denna strategi har inga definierade entry-regler ännu.")
        return

    snap, snap_error = market_data.try_snapshot(ticker)
    _prefill_entry(ticker, snap, entry)
    _render_market(snap, snap_error)
    cyc_state, cyc_name = _render_cycle(ticker, strategy_key)
    phase_state = _render_market_phase(ticker, strategy_key)
    bspot = cycle.blindspot_latest(ticker)
    _render_levels(entry, stop, target, snap, pb)

    rev = _render_review(ticker, strategy_key)
    swing_checks = _swing_rule_checks(ticker, strategy_key)

    results = _check_rules(pb, entry, stop, target, snap, swing_checks)
    if rev is not None:
        # Granskningens utfall och kontroller som regelrader — arkets egen
        # bedömning, läst, inte omräknad.
        for status, label, note in reversed(rev["controls"]):
            results.insert(0, RuleResult(0, f"Kontroll: {label}",
                                         status, note,
                                         hard=(status == "FAIL")))
        results.insert(0, RuleResult(
            0, f"Granskningen — {rev['sheet']}", rev["status"], rev["note"],
            hard=(rev["status"] == "FAIL")))
    if cycle.requires_cycle(strategy_key):
        # Köpgrindens första fråga — mekanisk, ur rotationsfliken. Vila fäller
        # kandidaten oavsett hur bra bolaget är: fel fas är fel fas.
        gate_status, gate_note = cycle.gate_from_cycle(cyc_state, cyc_name)
        results.insert(0, RuleResult(
            0, "Cykelläge — rotationsflikens Triple Signal",
            gate_status, gate_note, hard=True))
    if cycle.requires_market_cycle(strategy_key):
        # Contrarian och quality namnger sina faser i playbooken — grinden
        # läser dem därifrån, så en ändrad playbook flyttar grinden med sig.
        gate_status, gate_note = cycle.gate_from_market_phase(phase_state,
                                                              strategy_key)
        results.insert(0, RuleResult(
            0, "Marknadscykelfasen — Market Cycle Engine",
            gate_status, gate_note, hard=True))
    overall = _overall_status(results)
    overall_color = _status_to_color(overall)

    # Kandidatkort header
    st.markdown(
        f'<div style="background:{_BG};border:1px solid {_BORDER};'
        f'border-top:3px solid {overall_color};border-radius:10px;'
        f'padding:18px 20px;margin-bottom:20px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<span style="font-size:22px;font-weight:800;color:#E8EDF2;">{ticker}</span>'
        f'<span style="font-size:13px;color:{_DIM};margin-left:12px;">{pb.name}</span>'
        f'</div>'
        f'<div style="background:{overall_color}22;color:{overall_color};'
        f'font-size:12px;font-weight:700;padding:6px 16px;border-radius:6px;'
        f'letter-spacing:2px;">{overall}</div>'
        f'</div>'
        f'<div style="margin-top:12px;display:flex;gap:24px;flex-wrap:wrap;">'
        f'<span style="font-size:12px;color:{_DIM};">Entry <b style="color:#E8EDF2;">'
        f'{entry:.2f} kr</b></span>'
        f'<span style="font-size:12px;color:{_DIM};">Stop <b style="color:{_RED};">'
        f'{stop:.2f} kr</b></span>'
        f'<span style="font-size:12px;color:{_DIM};">Target <b style="color:{_GREEN};">'
        f'{target:.2f} kr</b></span>'
        f'<span style="font-size:12px;color:{_DIM};">Risk '
        f'<b style="color:{_AMBER};">{pb.risk.risk_per_trade}</b></span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Regelrader
    section_title("Regelkontroll — Entry", "✅")
    st.markdown(
        f'<p style="font-size:11px;color:{_DIM};margin:-6px 0 10px;">'
        f'🔒 = hård regel (aldrig bruten) · MANUAL = kräver kontroll i panelen</p>',
        unsafe_allow_html=True,
    )
    for res in results:
        _render_rule_row(res)

    passed_count = sum(1 for r in results if r.status == "PASS")
    manual_count = sum(1 for r in results if r.status == "MANUAL")
    fail_count   = sum(1 for r in results if r.status == "FAIL")
    st.markdown(
        f'<div style="margin:14px 0 24px;font-size:12px;color:{_DIM};">'
        f'<span style="color:{_GREEN};">✅ {passed_count} PASS</span> &nbsp;·&nbsp; '
        f'<span style="color:{_AMBER};">⚠️ {manual_count} MANUAL</span> &nbsp;·&nbsp; '
        f'<span style="color:{_RED};">❌ {fail_count} FAIL</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── AI-kommentar ──────────────────────────────────────────────────────────
    section_title("AI-kommentar", "💬")
    _render_ai_section(ticker, strategy_key, pb, results, entry, stop, target,
                       snap, cyc_state, bspot, rev, phase_state)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:28px 0;'>",
                unsafe_allow_html=True)

    # ── Journal ───────────────────────────────────────────────────────────────
    _render_journal_log(ticker, strategy_key, entry, stop, target)

    section_title("Journalstatistik", "📊")
    _render_journal_stats(_load_journal())

    with st.expander("📋 Senaste journal-poster", expanded=False):
        _render_journal_history()
    storage_ui.footer()
