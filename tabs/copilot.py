"""
tabs/copilot.py — AI Trading Copilot
======================================
Interaktiv kandidatanalys med deterministisk regelkontroll och AI-kommentar.

Flöde:
    1. Välj strategi + ange ticker
    2. Fyll i prisdata (entry, stop, target)
    3. Visa regelkontroll per regel (PASS / MANUAL / FAIL)
    4. Visa kandidatkort med rekommendation
    5. AI-kommentar (stub → redo för GPT-integration)
    6. Snabb journal-logg

Inga beroenden utanför panelen utöver strategy_rules.py och ui/theme.py.
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

# ── Palette shortcuts ─────────────────────────────────────────────────────────
_GREEN  = _P.get("green",    "#2d8a4e")
_RED    = _P.get("red",      "#c44545")
_AMBER  = _P.get("amber",    "#d4943a")
_CYAN   = _P.get("gold",     "#00E5FF")
_DIM    = _P.get("text_dim", "#6B7280")
_BG     = "#1A1F25"
_BORDER = "rgba(255,255,255,0.06)"

# ── Journal storage ───────────────────────────────────────────────────────────
_JOURNAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".copilot_journal.json",
)


def _load_journal() -> list[dict]:
    if os.path.exists(_JOURNAL_PATH):
        try:
            with open(_JOURNAL_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_journal(entries: list[dict]) -> None:
    try:
        with open(_JOURNAL_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        st.warning(f"Kunde inte spara journal: {exc}")


# ── Rule check logic ──────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    number: int
    text: str
    status: str          # "PASS" | "MANUAL" | "FAIL"
    note: str = ""
    hard: bool = False


def _check_rules(pb: Playbook, entry: float, stop: float, target: float) -> list[RuleResult]:
    """
    Deterministisk regelkontroll.
    Regler som kräver prisdata (R:R, stop-nivå) räknas automatiskt.
    Regler som kräver extern information (regim, ranking) markeras MANUAL.
    """
    results: list[RuleResult] = []

    risk_pct = abs(entry - stop) / entry * 100 if entry else 0
    rr = abs(target - entry) / abs(entry - stop) if (entry and stop and entry != stop) else 0

    for r in pb.entry:
        text_lower = r.text.lower()

        # R:R-kontroll
        if "1:2" in r.text or "r:r" in text_lower or "reward" in text_lower:
            if rr >= 2.0:
                results.append(RuleResult(r.number, r.text, "PASS",
                                           f"R:R = {rr:.1f}x ✓", r.hard))
            else:
                results.append(RuleResult(r.number, r.text, "FAIL",
                                           f"R:R = {rr:.1f}x — kräver ≥ 2,0", r.hard))

        # Stop-nivå
        elif "stop" in text_lower and entry and stop:
            results.append(RuleResult(r.number, r.text, "PASS",
                                       f"Stop = {risk_pct:.1f} % från entry", r.hard))

        # Allt annat kräver manuell koll
        else:
            results.append(RuleResult(r.number, r.text, "MANUAL",
                                       "Kontrollera i panelen: " + r.panel_guide[:80],
                                       r.hard))

    return results


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
                  entry: float, stop: float, target: float) -> str:
    """Ändras underlaget blir det gamla svaret ogiltigt.

    Utan detta ligger en kommentar om entry 100 kvar när du ändrat till 120,
    och den ser lika auktoritativ ut som när den skrevs.
    """
    return f"{strategy_key}|{ticker.upper()}|{entry:g}|{stop:g}|{target:g}"


def _render_ai_section(ticker: str, strategy_key: str, pb: Playbook,
                       results: list[RuleResult],
                       entry: float, stop: float, target: float) -> None:
    """Deterministisk sammanfattning alltid; modellsvar på knapptryck.

    Anropet ligger BAKOM en knapp med flit. Streamlit kör om hela skriptet vid
    varje widget-interaktion — ett anrop i renderingsvägen hade blivit ett
    betalt API-anrop varje gång du rör ett reglage.
    """
    st.markdown(_ai_comment(ticker, pb, results, entry, stop, target))

    cache_key = _ai_cache_key(ticker, strategy_key, entry, stop, target)
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
                        rr=_rr(entry, stop, target),
                        risk_pct=_risk_pct(entry, stop),
                        passed=[r.text for r in results if r.status == "PASS"],
                        manual=[r.text for r in results if r.status == "MANUAL"],
                        failed=[r.text for r in results if r.status == "FAIL"],
                        risk_per_trade=pb.risk.risk_per_trade))
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


def _rr(entry: float, stop: float, target: float) -> float:
    if not entry or entry == stop:
        return 0.0
    return abs(target - entry) / abs(entry - stop)


def _risk_pct(entry: float, stop: float) -> float:
    return abs(entry - stop) / entry * 100 if entry else 0.0


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
                "note":     log_note,
                "logged_at": datetime.datetime.utcnow().isoformat(),
            })
            _save_journal(entries)
            st.success(f"✅ {log_ticker.upper()} loggad!")


def _render_journal_history() -> None:
    """Visa senaste journal-poster."""
    entries = _load_journal()
    if not entries:
        st.caption("Inga poster ännu.")
        return

    recent = list(reversed(entries[-20:]))
    for e in recent:
        rr_val = abs(e.get("target", 0) - e.get("entry", 0)) / abs(
            e.get("entry", 0) - e.get("stop", 1e-9) or 1e-9
        ) if e.get("entry") else 0
        st.markdown(
            f'<div style="background:{_BG};border:1px solid {_BORDER};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:13px;font-weight:700;color:#E8EDF2;">'
            f'{e.get("ticker","?")} '
            f'<span style="font-size:11px;color:{_DIM};font-weight:400;">'
            f'— {PLAYBOOKS.get(e.get("strategy",""), type("",(),{"name":e.get("strategy","")})()).name}'  # type: ignore[attr-defined]
            f'</span></span>'
            f'<span style="font-size:11px;color:{_DIM};">'
            f'Entry {e.get("entry","—")} · Stop {e.get("stop","—")} · '
            f'Target {e.get("target","—")} · R:R {rr_val:.1f}x'
            f'</span>'
            f'<span style="font-size:11px;color:{_DIM};">{e.get("date","")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Main render function ──────────────────────────────────────────────────────

def render_copilot_page() -> None:
    """Entry point — anropas från wolf_panel.py."""
    section_title("AI Trading Copilot", "🤖")
    st.markdown(
        f'<p style="color:{_DIM};font-size:0.82rem;margin:-8px 0 24px;">'
        f'Kandidatanalys · Regelkontroll · Riskkontroll · Journal</p>',
        unsafe_allow_html=True,
    )

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
            entry  = st.number_input("Entry-kurs (kr)", min_value=0.0, value=0.0,
                                     step=0.5, format="%.2f", key="copilot_entry")
        with c2:
            stop   = st.number_input("Stop-kurs (kr)", min_value=0.0, value=0.0,
                                     step=0.5, format="%.2f", key="copilot_stop")
        with c3:
            target = st.number_input("Target-kurs (kr)", min_value=0.0, value=0.0,
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

    results = _check_rules(pb, entry, stop, target)
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
    _render_ai_section(ticker, strategy_key, pb, results, entry, stop, target)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:28px 0;'>",
                unsafe_allow_html=True)

    # ── Journal ───────────────────────────────────────────────────────────────
    _render_journal_log(ticker, strategy_key, entry, stop, target)

    with st.expander("📋 Senaste journal-poster", expanded=False):
        _render_journal_history()
