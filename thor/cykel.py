"""
thor/cykel.py — Cykel-fliken i Thors analysblock.

Aktiens egen psykologifas (Market Cycle Engine, 14 faser DISBELIEF → DISBELIEF_NEW) med mätaren, fasens
beskrivning, regelutfallet (vilka villkor som slog in), nyckelindikatorerna och tidslinjen; hemmamarknadens
fas; Valhall-segmentets (eller temats) status/fas/cykel med ⚡ cykelstart; och playbookens köp/håll/sälj-faser
som grinden PASS/MANUAL/FAIL överst. Samma motor och samma grafer som ODINS VISDOM → Marknad → Market Cycle
(market_cycle.cache, tabs/market_cycle) — inga regler ändras.
"""

from __future__ import annotations

from html import escape
from typing import Callable, Optional

from thor import cards
from thor import dossier as td
from thor.ticker import Ticker
from ui.tokens import AMBER, CYAN, DIM, TEXT, regime_color

PERIOD = "1y"


def _engine() -> tuple:
    """(analys, historik) ur Market Cycle Engine — båda cachade en timme i Streamlit."""
    from market_cycle.cache import cached_market_cycle_analysis, cached_market_cycle_history
    return cached_market_cycle_analysis, cached_market_cycle_history


def cycle_start(board_blob: Optional[dict], segment: Optional[str]) -> Optional[str]:
    """Var i kartan segmentet står i ⚡ cykelstart (BAISSE · VÄNDER) i dag — 'Norden', 'globalt' … — eller None."""
    if not segment:
        return None
    for s in (board_blob or {}).get("cycle_starts") or []:
        if isinstance(s, dict) and s.get("segment") == segment:
            return str(s.get("where") or "ja")
    return None


def phase_config(phase: Optional[str]) -> dict:
    """Fasens emoji, färg och beskrivning ur motorns regelbok (tom när fasen är okänd)."""
    if not phase:
        return {}
    try:
        from market_cycle.rules import MARKET_CYCLE_RULES
        return MARKET_CYCLE_RULES.get(phase) or {}
    except Exception:
        return {}


def render(st, data: dict, ident: Ticker, strategy: str, phase_fn: Optional[Callable] = None,
           analysis_fn: Optional[Callable] = None, history_fn: Optional[Callable] = None) -> None:
    """Rita fliken. phase_fn(ticker) → ({phase, confidence}, fel); analysis_fn/history_fn(ticker, period) är
    Market Cycle Engines cachade funktioner (bytbara i test)."""
    from ui.components import note
    board = data.get("board")
    cyc = td.cycle(ident, strategy, board, phase_fn)
    st.markdown(cards.gate_card(cyc["gate"], cyc["playbook"]), unsafe_allow_html=True)
    own = cyc.get("phase") or {}
    st.markdown(cards.phase_strip(own.get("phase"), cyc["phases"]["buy"], cyc["phases"]["hold"], cyc["phases"]["sell"]),
                unsafe_allow_html=True)
    chips = []
    bench = cyc.get("benchmark") or {}
    if (bench.get("phase") or {}).get("phase"):
        chips.append(cards.chip(f"{bench['index']}: {bench['phase']['phase'].replace('_', ' ')} "
                                f"{bench['phase'].get('confidence', 0):.0f} %", CYAN))
    seg = cyc.get("segment") or {}
    if seg.get("status"):
        parts = (f"Valhall {seg.get('view')}", seg.get("status"), seg.get("phase"), seg.get("cykel"))
        chips.append(cards.chip(" · ".join(x for x in parts if x), regime_color(seg.get("status"))))
    start = cycle_start(board, ident.segment)
    if start:
        chips.append(cards.chip(f"⚡ Cykelstart ({start})", AMBER, filled=True))
    if chips:
        st.markdown("<div style='display:flex;flex-wrap:wrap;gap:4px;margin:4px 0;'>" + "".join(chips) + "</div>",
                    unsafe_allow_html=True)
    if not own.get("phase"):
        if cyc.get("error"):
            note(f"Aktiens fas kunde inte läsas: {cyc['error']}")
        _footnote(note)
        return
    cfg = phase_config(own["phase"])
    try:
        from tabs.market_cycle import _build_gauge
        st.plotly_chart(_build_gauge(own["phase"], float(own.get("confidence") or 0)), use_container_width=True,
                        key="thor_cycle_gauge")
    except Exception as exc:                                     # pragma: no cover - ritningen är inte regeln
        note(f"Mätaren kunde inte ritas: {exc}")
    if cfg.get("description"):
        st.markdown(f"<div style='color:{TEXT};font-size:0.82rem;line-height:1.5;margin:2px 0 8px;'>"
                    f"{escape(str(cfg.get('emoji') or ''))} <b style='color:{cfg.get('color', CYAN)};'>"
                    f"{escape(own['phase'].replace('_', ' '))}</b> — {escape(str(cfg['description']))}</div>",
                    unsafe_allow_html=True)
    _details(st, ident.yf, own["phase"], cfg, analysis_fn, history_fn, note)
    _footnote(note)


def _details(st, ticker: str, phase: str, cfg: dict, analysis_fn, history_fn, note) -> None:
    """Regelutfallet, nyckelindikatorerna och tidslinjen — var del för sig, ett fel blir en notering."""
    try:
        an, hi = analysis_fn, history_fn
        if an is None or hi is None:
            d_an, d_hi = _engine()
            an, hi = an or d_an, hi or d_hi
        analysis = an(ticker, PERIOD) or {}
    except Exception as exc:
        note(f"Regelutfallet kunde inte läsas: {exc}")
        return
    result, indicators = analysis.get("result") or {}, analysis.get("indicators") or {}
    if result.get("matched_rules") and cfg:
        try:
            from tabs.market_cycle import _render_rules_breakdown
            m = result["matched_rules"]
            st.markdown(f"<div style='color:{DIM};font-size:0.74rem;margin-top:6px;'>Fasens villkor: "
                        f"{len(m.get('matched') or [])} av {len(m.get('matched') or []) + len(m.get('unmatched') or [])} "
                        f"uppfyllda</div>", unsafe_allow_html=True)
            _render_rules_breakdown(m, cfg)
        except Exception as exc:
            note(f"Regelutfallet kunde inte ritas: {exc}")
    if indicators:
        try:
            from tabs.market_cycle import _render_indicator_summary
            _render_indicator_summary(indicators)
        except Exception as exc:
            note(f"Indikatorerna kunde inte ritas: {exc}")
    try:
        history = hi(ticker, PERIOD) or []
        if history:
            from tabs.market_cycle import _build_timeline
            st.plotly_chart(_build_timeline(history), use_container_width=True, key="thor_cycle_timeline")
    except Exception as exc:
        note(f"Tidslinjen kunde inte ritas: {exc}")


def _footnote(note) -> None:
    note("Aktiens egen psykologifas (Market Cycle Engine, 14 faser) och hemmamarknadens. Grinden läser playbookens "
         "köp-, håll- och säljfaser ordagrant — grön = köpfas, gul = håll eller fasen nämns inte, röd = säljfas. "
         "Valhall-chipet är segmentets (eller temats) läge i dagens karta; ⚡ = cykelstart (BAISSE · VÄNDER).")
