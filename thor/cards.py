"""
thor/cards.py — Thors kort som rena HTML-funktioner (testbara utan Streamlit), i husets stil
(ui.tokens, samma chips och dämpade detaljrader som BERSERK Regime och Valhall).

Regeln för varje kort: verdikt överst, bevis under. Mobilen först: en kolumn, inget som kräver bredd.
"""

from __future__ import annotations

from html import escape
from typing import Optional

from thor import candidates as tc
from thor import registry as reg
from ui.tokens import AMBER, BG_ALT, BORDER_FAINT, CYAN, DIM, EMBER, FONT_MONO, GOLD, GREEN, GREY, RED, TEXT, regime_color

CLASS_COLOR = {tc.KOP: GREEN, tc.BEVAKA: AMBER, tc.TRAFF: CYAN, tc.HALL: GREY, tc.SPARRAD: RED, tc.SALJ: RED}
VERDICT_WORD = {tc.KOP: "KÖP", tc.BEVAKA: "BEVAKA", tc.TRAFF: "GRANSKA", tc.HALL: "AVVAKTA", tc.SPARRAD: "SPÄRRAD",
                tc.SALJ: "SÄLJ"}
GATE_COLOR = {"PASS": GREEN, "MANUAL": AMBER, "FAIL": RED}


def _e(x) -> str:
    return escape(str(x if x is not None else ""))


def _num(x, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def _pct(x, nd: int = 1) -> str:
    try:
        return f"{float(x):+.{nd}f} %"
    except (TypeError, ValueError):
        return "—"


# ── Byggstenar ──────────────────────────────────────────────────────────────
def chip(text, color: str = DIM, filled: bool = False, strike: bool = False) -> str:
    """Litet chip: tunn kant i färgen, fyllt för det som ska sticka ut (INNEHAV, NY)."""
    bg = f"background:{color}22;" if filled else ""
    line = "text-decoration:line-through;" if strike else ""
    return (f"<span style='display:inline-block;border:1px solid {color};border-radius:3px;padding:0 5px;color:{color};"
            f"font-size:0.66rem;font-weight:700;white-space:nowrap;line-height:1.5;{bg}{line}'>{_e(text)}</span>")


def status_chip(c: tc.Candidate) -> str:
    return chip(c.status, CLASS_COLOR.get(c.status_class, GREY), filled=c.status_class == tc.KOP,
                strike=c.status_class == tc.SPARRAD)


def family_chip(strategy: str) -> str:
    s = reg.get(strategy)
    return chip(s.label if s else strategy, reg.FAMILY_COLOR.get(s.family, GREY) if s else GREY)


def segment_chip(segment: Optional[str], status: Optional[str] = None, cykel: Optional[str] = None) -> str:
    """Segmentets namn med Valhall-status (och cykelläge) i trafikljusets färg."""
    if not segment:
        return ""
    from valhall import segments as sgm
    s = sgm.get(segment)
    label = s.label if s else segment
    parts = [label] + [x for x in (status, cykel) if x]
    return chip(" · ".join(parts), regime_color(status) if status else DIM)


def kpi_chip(label: str, value, color: str = CYAN, sub: str = "") -> str:
    """Lägets kort: etikett, värde, en dämpad rad — fyra på en rad i telefonen tack vare flex-wrap."""
    return (f"<div style='flex:1 1 140px;min-width:0;background:{BG_ALT};border:1px solid {BORDER_FAINT};"
            f"border-left:3px solid {color};border-radius:6px;padding:6px 10px;'>"
            f"<div style='color:{DIM};font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;'>{_e(label)}</div>"
            f"<div style='color:{color};font-size:0.95rem;font-weight:700;'>{_e(value)}</div>"
            + (f"<div style='color:{DIM};font-size:0.64rem;'>{_e(sub)}</div>" if sub else "") + "</div>")


def kpi_row(items: list) -> str:
    """[(etikett, värde, färg, underrad)] → en flex-rad som bryter snyggt i mobilen."""
    return ("<div style='display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px;'>"
            + "".join(kpi_chip(*it) for it in items) + "</div>")


# ── Kandidatraden ───────────────────────────────────────────────────────────
def levels_text(c: tc.Candidate) -> str:
    parts = []
    if c.entry is not None:
        parts.append(f"entry {_num(c.entry)}")
    if c.stop is not None:
        parts.append(f"stopp {_num(c.stop)}")
    if c.target is not None:
        parts.append(f"mål {_num(c.target)}")
    if c.rr is not None:
        parts.append(f"R {_num(c.rr, 1)}")
    return " · ".join(parts)


def candidate_card(c: tc.Candidate) -> str:
    """En rad i flödet: ticker och namn, Thor-rank stort till höger, chips (strategi, verdikt, segment),
    strategins egen siffra, nivåer och varför."""
    col = CLASS_COLOR.get(c.status_class, GREY)
    score = f"{c.score:.0f}" if c.score is not None else "—"
    chips = [family_chip(c.strategy), status_chip(c), segment_chip(c.segment, c.segment_status, c.cykel)]
    if c.theme:
        from berserk import themes as bth
        chips.append(chip(bth.label(c.theme), EMBER))
    if c.new:
        chips.append(chip("NY", CYAN, filled=True))
    if c.stale:
        chips.append(chip(f"gammal · {c.asof}", GREY))
    detail = " · ".join(x for x in (c.score_text, levels_text(c), c.why) if x)
    return (f"<div style='border:1px solid {col}55;border-left:4px solid {col};border-radius:6px;padding:8px 10px;"
            f"margin:4px 0 2px;background:{BG_ALT};min-width:0;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:baseline;gap:8px;'>"
            f"<div style='min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
            f"<b style='color:{TEXT};font-size:0.95rem;'>{_e(c.ticker)}</b> "
            f"<span style='color:{DIM};font-size:0.74rem;'>{_e(c.name)}</span></div>"
            f"<div style='color:{col};font-family:{FONT_MONO};font-size:1.3rem;font-weight:700;flex:none;'>{score}</div></div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 3px;'>{''.join(ch for ch in chips if ch)}</div>"
            f"<div style='color:{DIM};font-size:0.7rem;line-height:1.5;'>{_e(detail)}</div></div>")


# ── Analysblockets huvud ────────────────────────────────────────────────────
def verdict_card(word: str, sentence: str, detail: str = "", color: str = DIM, label: str = "VAD SKA JAG GÖRA?") -> str:
    """Verdiktet i ett ord, en mening och en dämpad rad — samma mönster som Alpha Regime."""
    return (f"<div style='border:2px solid {color}66;background:{color}0d;border-radius:10px;padding:14px 16px;margin:8px 0;'>"
            f"<div style='color:{DIM};font-size:0.66rem;letter-spacing:0.14em;'>{_e(label)}</div>"
            f"<div style='color:{color};font-size:2rem;font-weight:900;letter-spacing:0.04em;line-height:1.1;'>{_e(word)}</div>"
            f"<div style='color:{TEXT};font-size:0.9rem;margin-top:6px;line-height:1.4;'>{_e(sentence)}</div>"
            + (f"<div style='color:{DIM};font-size:0.74rem;margin-top:4px;'>{_e(detail)}</div>" if detail else "") + "</div>")


def verdict_for(c: Optional[tc.Candidate], strategy: str) -> str:
    """Verdiktkortet ur dagens kandidatrad för vald strategi — eller ett ärligt 'inte i listan'."""
    if c is None:
        return verdict_card("INGEN SIGNAL", f"{reg.label(strategy)} har inte {'den här aktien'} i dagens lista.",
                            "Analysen nedan gäller ändå — regimen per strategi kommer i fas 4.", GREY)
    col = CLASS_COLOR.get(c.status_class, GREY)
    sentence = f"{c.label}: {c.status}" + (f" · {c.score_text}" if c.score_text else "")
    run = (f"körd {c.asof}" if c.asof else "") + (f" på stängning {c.data_asof}" if c.data_asof else "")
    detail = " · ".join(x for x in (levels_text(c), c.why, run.strip()) if x)
    return verdict_card(VERDICT_WORD.get(c.status_class, c.status_class), sentence, detail, col)


# ── Viking Nine i Thor: en beslutsrad, en kompakt 9-ruta, en Execution-rad ──
VN_STATUS_COLOR = {"GOLDEN TICKET": GOLD, "WAIT": AMBER, "NO TRADE": RED}


def vn_decision_row(status: str, nine_text: str, exec_text: str, block: str, block_color: str, levels: str,
                    bars: str) -> str:
    """Beslutet i en rad: verdiktet, Nine och Execution som siffror, den första grinden som stoppar, nivåerna
    och vilka stängningar motorn räknade på. Orsakerna i detalj ligger i expandern under."""
    col = VN_STATUS_COLOR.get(status, GREY)
    return (f"<div style='border:2px solid {col}66;background:{col}0d;border-radius:10px;padding:12px 14px;margin:8px 0;'>"
            f"<div style='display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;'>"
            f"<div style='color:{col};font-size:1.5rem;font-weight:900;letter-spacing:0.06em;'>{_e(status)}</div>"
            f"<div style='color:{TEXT};font-size:0.9rem;'>Nine <b>{_e(nine_text)}</b> · Execution <b>{_e(exec_text)}</b>"
            f"</div></div>"
            f"<div style='color:{block_color};font-size:0.86rem;margin-top:6px;'>Stoppar: {_e(block)}</div>"
            + (f"<div style='color:{TEXT};font-size:0.8rem;margin-top:2px;'>{_e(levels)}</div>" if levels else "")
            + f"<div style='color:{DIM};font-size:0.72rem;margin-top:4px;'>{_e(bars)}</div></div>")


def vn_nine_compact(nine) -> str:
    """OVTLYR Nine i en rad: MARKET SPY 1/3 ✗✗✓ · SECTOR XLV 1/2 · STOCK 4/4 · TOTAL — faktorn som fattas står
    i underraden. nine = ovtlyr_nine.NineResult."""
    cells = []
    for name in ("market", "sector", "stock"):
        fs = nine.layer(name)
        n = nine.layer_passed(name)
        marks = "".join("✓" if f.passed else "✗" if f.status == "FAIL" else "?" for f in fs)
        tag = {"market": nine.market_label, "sector": nine.sector_etf or "—"}.get(name, "")
        col = GREEN if fs and n == len(fs) else AMBER if n else RED
        gaps = " · ".join(f.label for f in fs if not f.passed)
        cells.append((f"{name.upper()} {tag}".strip(), f"{n}/{len(fs)} {marks}", col, gaps or "allt grönt"))
    cells.append(("TOTAL", f"{nine.passed}/9 · viktat {nine.weighted:g}", GOLD if nine.passed == 9 else CYAN, nine.status))
    return kpi_row(cells)


def vn_execution_row(decision) -> str:
    """Viking Executions filter som chips på en rad: ✓ grönt fyllt, ✗ rött (gult när filtret bara är information),
    ? när data saknas; flaggan (WAIT FOR CLOSE, NO CHASE …) står på chipet. decision = viking_execution.EntryDecision."""
    chips = []
    for c in decision.checks:
        if c.status == "PASS":
            col, m = GREEN, "✓"
        elif c.status == "FAIL":
            col, m = (RED if c.required else AMBER), "✗"
        else:
            col, m = DIM, "?"
        text = f"{m} {c.label}" + (f" · {c.flag}" if c.flag else "") + ("" if c.required else " (info)")
        chips.append(chip(text, col, filled=c.status == "PASS"))
    return (f"<div style='margin:4px 0 8px;'><div style='color:{CYAN};font-size:0.66rem;letter-spacing:0.12em;'>"
            f"VIKING EXECUTION {decision.execution_passed}/{decision.execution_total}</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;'>{''.join(chips)}</div></div>")


def snapshot_row(snap: Optional[dict]) -> str:
    if not snap:
        return ""
    items = [("Kurs", _num(snap.get("price")), TEXT, snap.get("as_of") or ""),
             ("1 mån", _pct(snap.get("ret_1m_pct")), GREEN if (snap.get("ret_1m_pct") or 0) >= 0 else RED, ""),
             ("3 mån", _pct(snap.get("ret_3m_pct")), GREEN if (snap.get("ret_3m_pct") or 0) >= 0 else RED, ""),
             ("ATR", _pct(snap.get("atr_pct")).lstrip("+"), AMBER, "per dag"),
             ("EMA50", _pct(snap.get("dist_ema50_pct")), GREEN if (snap.get("dist_ema50_pct") or 0) >= 0 else RED, "avstånd"),
             ("EMA200", _pct(snap.get("dist_ema200_pct")), GREEN if (snap.get("dist_ema200_pct") or 0) >= 0 else RED, "avstånd"),
             ("52 v", _pct(snap.get("from_high_pct")), DIM, "från toppen")]
    return kpi_row(items)


def header_card(t: dict, cands: list, held: bool = False) -> str:
    """Aktiens huvud: ticker och namn, chips (marknad, segment, råvarutema med källa, INNEHAV) och
    'finns i' — dagens kandidatrader per strategi."""
    chips = [chip(f"{t.get('home_index') or '—'} · {t.get('market') or ''}".strip(" ·"), CYAN)]
    seg = t.get("segment")
    first = cands[0] if cands else None
    if seg:
        chips.append(segment_chip(seg, first.segment_status if first else None, first.cykel if first else None))
    if t.get("theme"):
        chips.append(chip(f"{t.get('theme_label') or t['theme']} · {t.get('theme_source') or ''}".strip(" ·"), EMBER))
    if t.get("sector_etf"):
        chips.append(chip(f"sektor-ETF {t['sector_etf']}", DIM))
    if held:
        chips.append(chip("INNEHAV", GREEN, filled=True))
    found = "".join(f"<span style='margin-right:4px;'>{family_chip(c.strategy)}{status_chip(c)}</span>" for c in cands)
    return (f"<div style='margin:4px 0;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap;'>"
            f"<div><b style='color:{TEXT};font-size:1.3rem;'>{_e(t.get('yf'))}</b> "
            f"<span style='color:{DIM};font-size:0.84rem;'>{_e(t.get('name'))}</span></div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:4px;'>{''.join(chips)}</div></div>"
            + (f"<div style='color:{DIM};font-size:0.68rem;margin-top:4px;'>finns i dag i: {found}</div>" if cands else
               f"<div style='color:{DIM};font-size:0.68rem;margin-top:4px;'>inte i någon strategis lista i dag</div>")
            + "</div>")


# ── Cykeln ──────────────────────────────────────────────────────────────────
def phase_strip(phase: Optional[str], buy: list, hold: list, sell: list) -> str:
    """De 14 faserna på en rad: aktiens fas markerad, playbookens köp/håll/sälj-faser färgade."""
    from market_cycle.rules import MARKET_CYCLE_RULES, PHASE_ORDER
    out = []
    for p in PHASE_ORDER:
        col = GREEN if p in buy else AMBER if p in hold else RED if p in sell else GREY
        here = p == phase
        emoji = MARKET_CYCLE_RULES[p].get("emoji", "")
        out.append(f"<span title='{_e(p)}' style='display:inline-block;padding:1px 4px;border-radius:3px;font-size:0.62rem;"
                   f"color:{TEXT if here else col};background:{col + '55' if here else 'transparent'};"
                   f"border:1px solid {col};font-weight:{'700' if here else '400'};'>"
                   f"{emoji} {_e(p.replace('_', ' ').title())}</span>")
    return f"<div style='display:flex;flex-wrap:wrap;gap:3px;margin:6px 0;'>{''.join(out)}</div>"


def gate_card(gate: dict, playbook: str) -> str:
    """Playbookens cykelgrind: PASS/MANUAL/FAIL med noteringen."""
    status = (gate or {}).get("status") or "MANUAL"
    col = GATE_COLOR.get(status, AMBER)
    title = f"Cykelgrinden ({playbook})" if playbook else "Cykelgrinden"
    return (f"<div style='border:1px solid {col}55;background:{col}0d;border-radius:8px;padding:8px 12px;margin:6px 0;'>"
            f"<span style='color:{col};font-weight:700;'>{_e(title)}: {_e(status)}</span>"
            f"<div style='color:{DIM};font-size:0.72rem;margin-top:2px;'>{_e((gate or {}).get('note'))}</div></div>")


def wolf_score_card(total, label: str, color: str, ticker: str, when: str = "") -> str:
    """Wolf Regimes stora poängkort: n / 125 MAX och etiketten — samma som sidan, i en kolumn."""
    return (f"<div style='border:1px solid {color};border-radius:12px;padding:22px 16px;text-align:center;"
            f"background:linear-gradient(135deg, rgba(0,229,255,0.06) 0%, rgba(139,115,64,0.03) 100%);margin:6px 0;'>"
            f"<div style='font-size:11px;letter-spacing:4px;color:rgba(0,229,255,0.5);'>REGIME SCORE</div>"
            f"<div style='font-size:64px;font-weight:900;line-height:1;color:{color};font-family:{FONT_MONO};'>{_e(total)}</div>"
            f"<div style='font-size:11px;letter-spacing:3px;color:rgba(0,229,255,0.4);margin-top:4px;'>/ 125 MAX</div>"
            f"<div style='margin-top:12px;'><span style='background:{color}1a;border:1px solid {color};border-radius:20px;"
            f"color:{color};font-size:11px;font-weight:700;letter-spacing:3px;padding:5px 16px;'>{_e(label)}</span></div>"
            f"<div style='margin-top:14px;font-size:10px;color:rgba(0,229,255,0.3);letter-spacing:2px;'>{_e(ticker)}"
            + (f" &nbsp;·&nbsp; {_e(when)}" if when else "") + "</div></div>")


def gate_chips(gs: list, summary: dict) -> str:
    """De elva grindarna som chips (✓ ✗ ◌) och raden 'n/m MÄTBARA GRINDAR · k manuella'."""
    out = []
    for g in gs:
        if g["passed"] is None:
            c, icon = GREY, "◌"
        else:
            c, icon = (GREEN, "✓") if g["passed"] else (RED, "✗")
        out.append(f"<div style='background:{BG_ALT};border:1px solid {c};border-radius:4px;padding:3px 8px;font-size:0.65rem;'>"
                   f"<span style='color:{c};font-weight:700;'>{icon}</span> <span style='color:{TEXT};'>{_e(g['rule'])}</span> "
                   f"<span style='color:{DIM};'>({_e(g['value'])})</span></div>")
    gc = summary.get("color", DIM)
    return (f"<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px;'>{''.join(out)}</div>"
            f"<div style='color:{gc};font-size:0.8rem;font-weight:700;'>{summary.get('passed')}/{summary.get('measurable')} MÄTBARA GRINDAR"
            f" <span style='color:{DIM};font-weight:400;'>· {summary.get('manual')} manuella (◌) bedömer du själv</span></div>")


def check_rows(checks: dict) -> str:
    """{regel: (PASS/MANUAL/FAIL, notering)} som rader med färgad status (Momentum-reglerna, Long Screeners grindar)."""
    rows = []
    for name, (status, note) in checks.items():
        col = GATE_COLOR.get(str(status).upper(), GREY if str(status).upper() in ("N/A", "—") else AMBER)
        rows.append(f"<div style='display:flex;gap:8px;font-size:0.78rem;line-height:1.5;padding:2px 0;'>"
                    f"<span style='color:{col};font-weight:700;min-width:62px;'>{_e(status)}</span>"
                    f"<span style='color:{TEXT};'>{_e(name)}</span><span style='color:{DIM};'>{_e(note)}</span></div>")
    return f"<div style='background:{BG_ALT};border:1px solid {BORDER_FAINT};border-radius:6px;padding:6px 10px;'>{''.join(rows)}</div>"


def detail_rows(title: str, details: dict, color: str = CYAN, score_text: str = "") -> str:
    """Kriterier med pass/fail (Long Screeners fundamenta och teknik): ✓/✗, namn, värde."""
    rows = []
    for name, d in (details or {}).items():
        ok = (d or {}).get("pass")
        c = GREEN if ok else RED if ok is False else GREY
        val = (d or {}).get("value")
        rows.append(f"<div style='font-size:0.76rem;line-height:1.5;color:{TEXT};'><span style='color:{c};font-weight:700;'>"
                    f"{'✓' if ok else '✗' if ok is False else '?'}</span> {_e(name)} <span style='color:{DIM};'>{_e(val)}</span></div>")
    return (f"<div style='border:1px solid {color}55;border-radius:6px;padding:8px 10px;margin:4px 0;background:{BG_ALT};'>"
            f"<div style='color:{color};font-size:0.7rem;letter-spacing:0.1em;'>{_e(title)}"
            + (f" <span style='color:{TEXT};'>{_e(score_text)}</span>" if score_text else "") + "</div>"
            + "".join(rows) + "</div>")


def review_card(key: str, rev: dict) -> str:
    """Granskningens utfall ur ett ark: PASS/MANUAL/FAIL, noteringen och kontrollerna."""
    status = (rev or {}).get("status") or "MANUAL"
    col = GATE_COLOR.get(status, AMBER)
    controls = " · ".join(f"{lbl} {st}" for st, lbl, _n in (rev or {}).get("controls") or [])
    return (f"<div style='border:1px solid {col}55;background:{col}0d;border-radius:8px;padding:8px 12px;margin:4px 0;'>"
            f"<span style='color:{col};font-weight:700;'>{_e((rev or {}).get('sheet') or key)}: {_e(status)}</span>"
            f"<div style='color:{DIM};font-size:0.72rem;margin-top:2px;'>{_e((rev or {}).get('note'))}</div>"
            + (f"<div style='color:{DIM};font-size:0.7rem;'>{_e(controls)}</div>" if controls else "") + "</div>")
