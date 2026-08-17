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


# ── AI comment (OpenAI GPT) ───────────────────────────────────────────────────

def _ai_comment(ticker: str, pb: Playbook, results: list[RuleResult],
                entry: float, stop: float, target: float) -> str:
    """
    Anropar OpenAI Chat Completions och returnerar en kort analys på svenska.

    Kräver att OPENAI_API_KEY är satt i st.secrets (Streamlit Cloud / secrets.toml)
    eller som miljövariabel. Faller tillbaka till en deterministisk kommentar om
    nyckeln saknas eller anropet misslyckas.
    """
    passed = [r.text for r in results if r.status == "PASS"]
    failed = [r.text for r in results if r.status == "FAIL"]
    manual = [r.text for r in results if r.status == "MANUAL"]
    rr     = abs(target - entry) / abs(entry - stop) if (entry and stop and entry != stop) else 0
    risk_p = abs(entry - stop) / entry * 100 if entry else 0

    # Hämta API-nyckel från Streamlit secrets eller miljövariabel
    api_key: Optional[str] = (
        st.secrets.get("OPENAI_API_KEY")
        if hasattr(st, "secrets")
        else None
    ) or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return _fallback_comment(ticker, pb, passed, failed, manual, rr, risk_p)

    prompt = (
        f"Du är en professionell swing-trading-analytiker.\n\n"
        f"Ticker: {ticker}\n"
        f"Strategi: {pb.name} ({pb.tagline})\n"
        f"Entry: {entry:.2f} · Stop: {stop:.2f} · Target: {target:.2f}\n"
        f"R:R: {rr:.1f}x · Risk från entry: {risk_p:.1f}%\n"
        f"Godkända regler ({len(passed)}): {', '.join(passed) or 'inga'}\n"
        f"Manuella regler ({len(manual)}): {', '.join(manual) or 'inga'}\n"
        f"Misslyckade regler ({len(failed)}): {', '.join(failed) or 'inga'}\n\n"
        f"Ge en kort analys (3–5 meningar) på svenska:\n"
        f"- Är affären värd att gå vidare med?\n"
        f"- Vad bör kontrolleras manuellt?\n"
        f"- Är R:R acceptabelt?\n"
    )

    try:
        import openai  # noqa: PLC0415 — lazy import to keep startup fast
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.4,
        )
        return response.choices[0].message.content or "Inget svar från AI."
    except Exception as exc:  # pragma: no cover
        return (
            f"{_fallback_comment(ticker, pb, passed, failed, manual, rr, risk_p)}\n\n"
            f"_⚠️ OpenAI-anropet misslyckades: {exc}_"
        )


def _fallback_comment(
    ticker: str,
    pb: Playbook,
    passed: list[str],
    failed: list[str],
    manual: list[str],
    rr: float,
    risk_p: float,
) -> str:
    """Deterministisk fallback-kommentar när OpenAI ej är tillgängligt."""
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
        "_💡 Sätt OPENAI_API_KEY i Streamlit Secrets för att aktivera GPT-analys._",
    ]

    return "\n".join(lines)


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
    with st.spinner("Hämtar AI-kommentar…"):
        comment = _ai_comment(ticker, pb, results, entry, stop, target)
    st.markdown(comment)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:28px 0;'>",
                unsafe_allow_html=True)

    # ── Journal ───────────────────────────────────────────────────────────────
    _render_journal_log(ticker, strategy_key, entry, stop, target)

    with st.expander("📋 Senaste journal-poster", expanded=False):
        _render_journal_history()
