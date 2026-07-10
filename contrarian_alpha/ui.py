"""
ui.py — Contrarian Alpha Screener · Streamlit UI (Fas 7).

Tre zoner:
  Zon 1  Kontrollpanel   — marknadsväljare, scan-knapp, senaste tidpunkt
  Zon 2  Resultattabell  — sorterad på Composite Score, alla delpoäng + flaggor
  Zon 3  Detaljkort      — klick expanderar score-breakdown, nyckeltal, Viking-badge

Anrop från wolf_panel.py:
    from contrarian_alpha.ui import render_contrarian_alpha_page
    with tab_contrarian_alpha:
        render_contrarian_alpha_page()

Självständig modul — importerar inte från ui/ i wolf-shadow-dashboard.
Paletten matchar ui/theme.py exakt (PALETTE nedan).
"""
from __future__ import annotations

import sys
import os
import math
import logging
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

# ─── Optional annotations (discipline fields) ────────────────────────────────
_ANN_OK = False
try:
    from contrarian_alpha.annotations import get_annotation, save_annotation
    _ANN_OK = True
except ImportError:
    pass

# ─── Palette (speglar ui/theme.py PALETTE) ───────────────────────────────────

P = {
    "bg":          "#0c0c12",
    "bg2":         "#14141e",
    "bg3":         "#1a1a28",
    "surface":     "#10101a",
    "gold":        "#00E5FF",
    "gold_dim":    "#00A8BF",
    "gold_muted":  "rgba(0,229,255,0.40)",
    "gold_faint":  "rgba(0,229,255,0.12)",
    "silver":      "#b8c4d0",
    "ice_blue":    "#a0b4c8",
    "green":       "#2d8a4e",
    "red":         "#c44545",
    "amber":       "#d4943a",
    "text":        "#e8e4dc",
    "text_dim":    "#8a8578",
    "border":      "rgba(0,229,255,0.15)",
    "border_hi":   "rgba(0,229,255,0.35)",
}

_REGIME_COLOR = {"green": "#2d8a4e", "orange": "#d4943a", "red": "#c44545",
                 "unknown": "#8a8578"}

# ─── Market presets ───────────────────────────────────────────────────────────

try:
    from borsdata_api import ALL_NORDIC_MARKETS as _ALL_NORDIC
except ImportError:
    _ALL_NORDIC = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 18, 19]

_MARKETS = {
    "Norden":   {"market_ids": list(_ALL_NORDIC), "include_global": False},
    "Global":   {"market_ids": list(_ALL_NORDIC), "include_global": True},
    "US":       {"market_ids": [], "include_global": False,
                 "manual_tickers": [
                     "FCX","NEM","GOLD","WPM","UUUU","CCJ","XOM","CVX",
                     "RIO","BHP","VALE","AA","CLF","MP","LTHM",
                 ]},
    # PR1 foundation: static US/CA resource universe (Rick Rule / Eric Sprott style).
    # Loads config/universes/us_ca_resource.csv. Scoring is NOT yet stage-aware,
    # so the Nordic fundamental grinds still apply (see warning in control panel).
    "US/CA Resource": {"market_ids": [], "include_global": False,
                       "universe": "us_ca_resource"},
    "Custom":   {"market_ids": [], "include_global": False},
}

# ─── CSS ──────────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
/* ── Base ── */
.ca-section-title {{
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: {P['gold_muted']};
    border-bottom: 1px solid {P['border']};
    padding-bottom: 6px;
    margin: 0 0 16px 0;
    font-family: 'Courier New', monospace;
}}

/* ── Control Panel ── */
.ca-stat-pill {{
    display: inline-block;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: {P['text_dim']};
    background: {P['bg2']};
    border: 1px solid {P['border']};
    border-radius: 4px;
    padding: 2px 8px;
    margin-right: 6px;
}}
.ca-stat-pill b {{ color: {P['gold']}; }}

/* ── Table ── */
.ca-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}}
.ca-table th {{
    text-align: left;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {P['gold_muted']};
    border-bottom: 1px solid {P['border_hi']};
    padding: 6px 10px 8px;
}}
.ca-table td {{
    padding: 8px 10px;
    border-bottom: 1px solid {P['border']};
    color: {P['text']};
    vertical-align: middle;
}}
.ca-table tr:hover td {{ background: {P['bg3']}; }}

/* Score cells */
.ca-score {{
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.05em;
}}
.ca-score-bar {{
    height: 3px;
    border-radius: 2px;
    margin-top: 3px;
    background: {P['border']};
}}
.ca-score-fill {{
    height: 3px;
    border-radius: 2px;
}}

/* Rank badge */
.ca-rank {{
    display: inline-block;
    width: 22px;
    height: 22px;
    line-height: 22px;
    text-align: center;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    background: {P['gold_faint']};
    color: {P['gold']};
    border: 1px solid {P['border']};
}}

/* Ticker cell */
.ca-ticker {{
    font-weight: 700;
    font-size: 14px;
    color: {P['gold']};
    letter-spacing: 0.08em;
}}
.ca-name {{
    font-size: 11px;
    color: {P['text_dim']};
    margin-top: 1px;
}}
.ca-sector {{
    font-size: 10px;
    color: {P['text_dim']};
    letter-spacing: 0.06em;
}}

/* Regime badge */
.ca-regime {{
    display: inline-block;
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 3px;
    font-weight: 700;
    border: 1px solid;
}}

/* Empty state */
.ca-empty {{
    text-align: center;
    padding: 60px 24px;
    background: {P['bg2']};
    border: 1px solid {P['border']};
    border-radius: 10px;
    margin: 24px 0;
}}
.ca-empty-icon {{ font-size: 2.4rem; margin-bottom: 12px; }}
.ca-empty-title {{
    font-family: 'Courier New', monospace;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {P['gold']};
    margin-bottom: 6px;
}}
.ca-empty-sub {{
    font-family: 'Courier New', monospace;
    font-size: 0.75rem;
    color: {P['text_dim']};
    letter-spacing: 0.06em;
}}
.ca-empty-hint {{
    margin-top: 20px;
    font-size: 11px;
    color: {P['text_dim']};
    border-top: 1px solid {P['border']};
    padding-top: 16px;
}}

/* Expander row header */
.ca-row-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    font-family: 'Courier New', monospace;
}}

/* Detail card */
.ca-detail-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 12px 0;
}}
.ca-metric {{
    background: {P['bg3']};
    border: 1px solid {P['border']};
    border-radius: 6px;
    padding: 8px 12px;
}}
.ca-metric-label {{
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {P['text_dim']};
    font-family: 'Courier New', monospace;
    margin-bottom: 4px;
}}
.ca-metric-value {{
    font-size: 15px;
    font-weight: 700;
    font-family: 'Courier New', monospace;
    color: {P['text']};
}}
.ca-metric-sub {{
    font-size: 10px;
    color: {P['text_dim']};
    margin-top: 2px;
}}
</style>
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _section_title(text: str, icon: str = "") -> None:
    prefix = f"{icon}&nbsp;&nbsp;" if icon else ""
    st.markdown(
        f'<p class="ca-section-title">{prefix}{text}</p>',
        unsafe_allow_html=True,
    )


def _score_color(score: float) -> str:
    if score >= 70:   return P["green"]
    if score >= 55:   return P["gold"]
    if score >= 40:   return P["amber"]
    return P["red"]


def _score_cell(score: float, max_score: float = 100.0) -> str:
    pct  = min(100, max(0, score / max_score * 100))
    col  = _score_color(score)
    return (
        f'<div class="ca-score" style="color:{col}">{score:.1f}</div>'
        f'<div class="ca-score-bar">'
        f'  <div class="ca-score-fill" style="width:{pct:.0f}%;background:{col}"></div>'
        f'</div>'
    )


def _regime_badge(color: str) -> str:
    c = _REGIME_COLOR.get(color, P["text_dim"])
    labels = {"green": "● GRÖN", "orange": "● ORANGE", "red": "● RÖD", "unknown": "● ?"}
    label = labels.get(color, color.upper())
    return (
        f'<span class="ca-regime" style="color:{c};border-color:{c}44;'
        f'background:{c}11">{label}</span>'
    )


def _fmt(val, fmt=".1f", fallback="—") -> str:
    if val is None:
        return fallback
    try:
        return f"{val:{fmt}}"
    except (TypeError, ValueError):
        return fallback


# ─── Zone 1: Kontrollpanel ───────────────────────────────────────────────────

def _render_control_panel() -> tuple[dict, bool]:
    """
    Renderar kontrollpanelen.
    Returnerar (config_kwargs: dict, run_now: bool).
    """
    _section_title("KONTROLLPANEL", "⚙️")

    c1, c2, c3, c4, c5 = st.columns([2, 1.6, 2, 2, 1])

    with c1:
        market = st.selectbox(
            "Marknad",
            options=list(_MARKETS.keys()),
            index=0,
            key="ca_market",
            help="Väljer vilka börser som scannas.",
        )

    with c2:
        mode = st.selectbox(
            "Modus",
            options=["quality", "deep_contrarian"],
            format_func=lambda x: "Quality" if x == "quality" else "Deep Contrarian",
            key="ca_mode",
            help=(
                "Quality — ROIC>15%, Quality-vikt 30%, Hat-vikt 20%.\n"
                "Deep Contrarian — ROIC>10%, Hat-vikt 30%, Quality-vikt 20%."
            ),
        )

    # ── US/CA Resource: guardrails + composite v1 + enrichment v1 (PR2/PR3/PR4) ──
    if market == "US/CA Resource":
        st.caption(
            "🛡️ **US/CA Resource** — statiskt resurs-universum · resurs-scoring v1 "
            "(watchlist/ranking, **ej köpsignal**). Tomma fält = saknas/"
            "needs_validation, **ej** negativ signal."
        )
        with st.expander("Om resurs-scoring (PR2–PR6)", expanded=False):
            st.markdown(
                "Statiskt US/Kanada-universum (Rick Rule / Eric Sprott-stil) från "
                "`config/universes/us_ca_resource.csv`.\n\n"
                "- **Stage-guardrails (PR2):** explorers/developers och rader utan "
                "Börsdata-fundamenta elimineras inte — necessity/balansräknings-"
                "grindarna avväpnas för det kurerade råvaru-universumet.\n"
                "- **Resurs-composite (PR3):** blandar survival/cash-runway, dilution, "
                "jurisdiktion och commodity-necessity med stage-vikter, separat från "
                "det nordiska composite-scoret.\n"
                "- **Enrichment (PR4):** datakvalitet (HIGH/MEDIUM/LOW), `data_as_of`-"
                "flaggor och commodity-proxy-metadata (t.ex. uranium→URNM/URA).\n"
                "- **Existing-source overlay (PR5/PR6):** likviditet, drawdown, "
                "commodity-RS, short/analytiker och FRED-makro från befintliga källor.\n\n"
                "Riktig finansdata (FMP/GoldStockData), NAV/resurs-ounces, insider-"
                "ägande och commodity/regim-triggers kommer i senare PR."
            )

    custom_tickers: list[str] = []
    if market == "Custom":
        with c3:
            raw = st.text_area(
                "Tickers (kommaseparerade)",
                height=68,
                key="ca_custom_tickers",
                placeholder="FCX, NEM, UUUU, ...",
            )
            custom_tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    with c4:
        top_n = st.slider("Visa topp-N", 5, 50, 20, key="ca_top_n")

    with c5:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_now = st.button("🔍 Kör scan", type="primary", width='stretch',
                            key="ca_run_btn")

    # Tidpunkt och stats
    last_ts = st.session_state.get("ca_last_run_ts", "")
    last_stats = st.session_state.get("ca_last_stats", {})
    pills = ""
    if last_ts:
        pills += f'<span class="ca-stat-pill">Senaste scan: <b>{last_ts}</b></span>'
    for k, v in last_stats.items():
        pills += f'<span class="ca-stat-pill">{k}: <b>{v}</b></span>'
    if pills:
        st.markdown(f'<div style="margin-top:6px">{pills}</div>', unsafe_allow_html=True)

    # Cache + Gist status (collapsed by default)
    with st.expander("Cache & lagring", expanded=False):
        _render_cache_status()

    preset = _MARKETS[market].copy()
    if market == "Custom":
        preset["manual_tickers"] = custom_tickers
    preset["top_n"] = top_n
    preset["mode"] = mode
    return preset, run_now


def _render_cache_status() -> None:
    """Visar TTL-cache storlek och Gist-status."""
    try:
        from contrarian_alpha.cache import cache_stats, gist_storage_status, TTL_FUNDAMENTALS, TTL_PRICE, TTL_SENTIMENT, TTL_REGIME
        stats  = cache_stats()
        status = gist_storage_status()

        ttls = {"fundamentals": TTL_FUNDAMENTALS, "price": TTL_PRICE,
                "sentiment": TTL_SENTIMENT, "regime": TTL_REGIME,
                "insider": TTL_PRICE}
        labels = {
            "fundamentals": "Fundamentals (24h)",
            "price":        "Prisdata (1h)",
            "sentiment":    "Sentiment (6h)",
            "regime":       "Viking Regime (1h)",
            "insider":      "Insider (1h)",
        }
        cols = st.columns(len(stats))
        for col, (name, stat) in zip(cols, stats.items()):
            ttl_h = ttls.get(name, 1) // 3600
            col.metric(labels[name], f"{stat['size']} poster", f"TTL {ttl_h}h")

        gist_icon = "☁️ Gist OK" if status == "cloud_ok" else (
                    "💾 Lokal" if status == "local_only" else f"⚠️ {status}")
        gist_col = "green" if status == "cloud_ok" else ("orange" if status == "local_only" else "red")
        st.markdown(
            f'<span style="font-family:monospace;font-size:11px;color:{P[gist_col]}">'
            f'Lagring: {gist_icon}</span>',
            unsafe_allow_html=True,
        )
        if st.button("Rensa cache", key="ca_clear_cache"):
            from contrarian_alpha.cache import clear_all
            clear_all()
            st.success("Cache rensad.")
    except Exception as e:
        st.caption(f"Cache-status ej tillgänglig: {e}")


# ─── Auto-scan check (07:30 CEST daily) ──────────────────────────────────────

def _check_auto_scan() -> bool:
    """
    Returns True if an automatic morning scan should run.
    Marks the scan as done in both the marker file and session_state.
    """
    if st.session_state.get("ca_auto_scanned_today"):
        return False
    try:
        from contrarian_alpha.cache import should_auto_scan
        return should_auto_scan()
    except Exception:
        return False


# ─── Pipeline runner (cached in session_state + Gist) ────────────────────────

def _get_or_run_pipeline(config_kwargs: dict, run_now: bool):
    """
    Kör pipeline om run_now == True, auto-scan triggar, eller inget cachat
    resultat finns i session_state.
    Sparar resultatet i session_state och i GitHub Gist (via cache.save_screener_results).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))

    mode        = config_kwargs.get("mode", "quality")
    cache_key   = f"ca_result_{mode}"
    auto_scan   = _check_auto_scan()
    run         = run_now or auto_scan   # cold start no longer forces a 5-10 min live scan

    # If not running, try to restore from Gist when session is cold
    if not run:
        if cache_key not in st.session_state:
            try:
                from contrarian_alpha.cache import load_screener_results
                saved = load_screener_results()
                if saved.get("timestamp"):
                    st.session_state["ca_last_run_ts"] = saved["timestamp"][:16] + " UTC (Gist)"
                    st.session_state["ca_last_stats"] = {
                        "Universum": saved.get("universe_count", "?"),
                        "Gist-resultat": len(saved.get("results", [])),
                    }
            except Exception:
                pass
        return st.session_state.get(cache_key)

    from contrarian_alpha.engine import PipelineConfig, run_pipeline
    from contrarian_alpha.flags  import attach_flags

    cfg = PipelineConfig(**{k: v for k, v in config_kwargs.items()
                            if k in PipelineConfig.__dataclass_fields__})

    label = "🌅 Morgonscan 07:30…" if auto_scan else "🔍 Scannar universum…"
    with st.spinner(label):
        try:
            result = run_pipeline(cfg)
        except Exception as e:
            st.error(f"Pipeline-fel: {e}")
            return None

    attach_flags(result.results)
    st.session_state[cache_key] = result

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.session_state["ca_last_run_ts"] = ts
    _stats = {
        "Universum":   result.universe_count,
        "Necessity ✓": result.necessity_passed,
        "Hate ✓":      result.hate_passed,
        "BS ✓":        result.bs_passed,
        "Rankade":     result.composite_ranked,
        "Tid":         f"{result.run_duration_s:.1f}s",
    }
    if getattr(result, "delisted_count", 0) > 0:
        _stats["Delistade skip"] = result.delisted_count
    st.session_state["ca_last_stats"] = _stats

    # Persist to Gist (non-blocking best-effort)
    try:
        from contrarian_alpha.cache import save_screener_results, mark_auto_scanned
        save_screener_results(result)
        if auto_scan:
            mark_auto_scanned()
            st.session_state["ca_auto_scanned_today"] = True
            try:
                from contrarian_alpha.cache import send_morning_alert
                send_morning_alert(result)
            except Exception:
                pass
    except Exception:
        pass

    return result


# ─── Zone 2: Resultattabell ───────────────────────────────────────────────────

def _render_results_table(pipeline_result) -> str | None:
    """
    Renderar resultattabellen som HTML + st.expander per rad (klickbar).
    Returnerar vald ticker (None om ingen vald).
    """
    _section_title("RESULTATTABELL", "📋")

    results = pipeline_result.results
    if not results:
        _m = st.session_state.get("ca_mode", "quality")
        _hint = (
            'Alla bolag klarar inte Quality-kraven: bevisad ROIC &gt;= 15%, '
            'sund skuldsattning (Net Debt/EBITDA &lt;= 3.5), tillracklig '
            'nodvandighet och balansrakning. Prova Deep Contrarian for hatade '
            'cykelbottnar, eller ett bredare universum.'
        ) if _m == "quality" else (
            'Alla bolag ar antingen for alskade (Hat &lt; 45), for lag '
            'nodvandighet (Necessity &lt; 60), svag balansrakning eller '
            'klarar inte ROIC-gransen. Prova Quality-modus eller ett bredare universum.'
        )
        st.markdown(
            '<div class="ca-empty">'
            '  <div class="ca-empty-icon">🔍</div>'
            '  <div class="ca-empty-title">Inga bolag klarar filtren just nu</div>'
            '  <div class="ca-empty-sub">— det är ett korrekt svar, inte ett fel.</div>'
            '  <div class="ca-empty-hint">' + _hint + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        _render_elimination_breakdown(pipeline_result)
        return None

    # ── Tabell-header ──
    st.markdown(
        '<table class="ca-table">'
        '<thead><tr>'
        '<th>#</th>'
        '<th>Ticker / Namn</th>'
        '<th>Sektor</th>'
        '<th>Composite</th>'
        '<th>Necessity</th>'
        '<th>Hat</th>'
        '<th>Quality</th>'
        '<th>Value</th>'
        '<th>Catalyst</th>'
        '<th>Viking</th>'
        '<th>Flaggor</th>'
        '</tr></thead>'
        '</table>',
        unsafe_allow_html=True,
    )

    # ── Rader som expanderbara kort ──
    selected = None
    for r in results:
        flags = getattr(r, "ui_flags", [])
        flag_str = "  ".join(f.emoji for f in flags) if flags else "—"
        regime_color = (r.catalyst_result.viking_regime_color
                        if r.catalyst_result else "unknown")
        v_col = _REGIME_COLOR.get(regime_color, P["text_dim"])

        # Expander-titel = komprimerad tabellrad
        expander_label = (
            f"**#{r.rank}**  &nbsp; `{r.ticker}` &nbsp; — &nbsp; {r.name}  "
            f"&nbsp;&nbsp; **{r.composite_score:.1f}**  "
            f"| N {r.necessity_score:.0f} · H {r.hat_score:.0f} "
            f"· Q {r.quality_score:.0f} · V {r.value_score:.0f} "
            f"· C {r.catalyst_score:.0f}  "
            f"&nbsp; {flag_str}"
        )

        # Ren text-fallback för expander (Streamlit stödjer inte HTML i label)
        kap_str = "  ★ KAP" if getattr(r, "kap_badge", False) else ""
        expander_plain = (
            f"#{r.rank}  {r.ticker} — {r.name}  │  "
            f"Score {r.composite_score:.1f}  │  "
            f"N {r.necessity_score:.0f} · H {r.hat_score:.0f} "
            f"· Q {r.quality_score:.0f} · V {r.value_score:.0f} "
            f"· C {r.catalyst_score:.0f}  "
            f"{flag_str}{kap_str}"
        )

        with st.expander(expander_plain, expanded=False):
            selected = r.ticker
            _render_detail_card(r)

    return selected


# ─── Discipline fields ────────────────────────────────────────────────────────

def _render_discipline_fields(r) -> None:
    """
    Render Ogiltigförklaras om + Trolig trigger fields for a CA result card.
    Shows INVALIDATED badge if price breaches invalidation_price.
    Includes an edit expander for saving new values.
    """
    if not _ANN_OK:
        return

    ann = get_annotation(r.ticker)
    inv_text  = ann.get("invalidation_text", "")
    inv_price = ann.get("invalidation_price")
    catalyst  = ann.get("probable_catalyst", "")

    # Decide whether to show INVALIDATED badge
    close_now = getattr(r, "close", None) or 0.0
    invalidated = (
        inv_price is not None
        and inv_price > 0
        and close_now > 0
        and close_now < inv_price
    )

    # Only render the info block when at least one field is filled
    has_data = bool(inv_text or inv_price or catalyst)

    if has_data or invalidated:
        inv_badge_html = ""
        if invalidated:
            inv_badge_html = (
                f'<span style="background:{P["red"]};color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-size:0.7rem;font-weight:700;'
                f'letter-spacing:0.1em;margin-left:10px;">⛔ INVALIDATED</span>'
            )

        rows_html = ""
        if inv_text or inv_price:
            price_str = f"  (nivå: {inv_price:.2f})" if inv_price else ""
            rows_html += (
                f'<div style="margin-bottom:6px;">'
                f'<span style="font-size:0.65rem;color:{P["text_dim"]};'
                f'text-transform:uppercase;letter-spacing:0.1em;">Ogiltigförklaras om</span>'
                f'<div style="font-size:0.82rem;color:{P["text"]};margin-top:2px;">'
                f'{inv_text or "—"}{price_str}</div>'
                f'</div>'
            )
        if catalyst:
            rows_html += (
                f'<div>'
                f'<span style="font-size:0.65rem;color:{P["text_dim"]};'
                f'text-transform:uppercase;letter-spacing:0.1em;">Trolig trigger</span>'
                f'<div style="font-size:0.82rem;color:{P["text"]};margin-top:2px;">'
                f'{catalyst}</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="border:1px solid {P["border"]};border-radius:6px;'
            f'padding:10px 14px;margin-top:12px;background:{P["bg3"]};">'
            f'<div style="font-family:\'Courier New\',monospace;font-size:9px;'
            f'letter-spacing:0.18em;text-transform:uppercase;color:{P["gold_muted"]};'
            f'margin-bottom:6px;">Disciplinfält{inv_badge_html}</div>'
            f'{rows_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Edit expander ─────────────────────────────────────────────────────
    with st.expander("✏️ Redigera disciplinfält", expanded=False):
        ef1, ef2 = st.columns([3, 1])
        with ef1:
            new_inv_text = st.text_input(
                "Ogiltigförklaras om (fritext)",
                value=inv_text,
                key=f"ann_inv_text_{r.ticker}",
                placeholder="t.ex. Pris bryter under stöd vid 180, ny lägsta lämnar trend",
            )
            new_catalyst = st.text_input(
                "Trolig trigger (fritext)",
                value=catalyst,
                key=f"ann_cat_{r.ticker}",
                placeholder="t.ex. Q2-rapport, Fed-beslut, teknikgenombrott",
            )
        with ef2:
            new_inv_price = st.number_input(
                "Prisnivå (0 = ej satt)",
                value=float(inv_price) if inv_price else 0.0,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key=f"ann_inv_price_{r.ticker}",
            )

        if st.button("Spara", key=f"ann_save_{r.ticker}"):
            save_annotation(
                r.ticker,
                invalidation_text=new_inv_text.strip(),
                invalidation_price=new_inv_price if new_inv_price > 0 else None,
                probable_catalyst=new_catalyst.strip(),
            )
            st.success(f"Disciplinfält sparade för {r.ticker}.")
            st.rerun()


# ─── Zone 3: Detaljkort ───────────────────────────────────────────────────────

def _render_detail_card(r) -> None:
    """Fullständigt detaljkort för ett ContrairianAlphaResult."""
    from contrarian_alpha.flags import format_flags_html, evaluate_flags

    flags = getattr(r, "ui_flags", None)
    if flags is None:
        flags = evaluate_flags(r)

    regime_color = (r.catalyst_result.viking_regime_color
                    if r.catalyst_result else "unknown")
    regime_hex   = _REGIME_COLOR.get(regime_color, P["text_dim"])
    v_label = {"green": "GRÖN", "orange": "ORANGE", "red": "RÖD"}.get(regime_color, "OKÄND")

    col_chart, col_meta = st.columns([3, 2])

    # ── Score-breakdown stapeldiagram (vänster) ──
    with col_chart:
        _section_title("SCORE BREAKDOWN")
        _render_breakdown_chart(r)

    # ── Metadata (höger) ──
    with col_meta:
        # KAP badge (quality mode only)
        if getattr(r, "kap_badge", False):
            st.markdown(
                f'<div style="background:{P["gold_faint"]};border:1px solid {P["gold"]};'
                f'border-radius:6px;padding:6px 14px;margin-bottom:12px;'
                f'font-family:\'Courier New\',monospace;font-size:11px;font-weight:700;'
                f'letter-spacing:0.18em;color:{P["gold"]};text-align:center">'
                f'★ KAP SCREENED</div>',
                unsafe_allow_html=True,
            )

        # Resource stage guardrail + composite transparency (us_ca_resource only)
        _stage = getattr(r, "stage", "")
        if _stage:
            _commodity = getattr(r, "primary_commodity", "") or "—"
            _gate_mode = getattr(r, "resource_gate_mode", "") or "—"
            _res_flags = getattr(r, "resource_flags", []) or []
            _flags_disp = ", ".join(_res_flags) if _res_flags else "inga"
            # Resource-composite v1 (PR3) — separate from the Nordic composite.
            _rc = float(getattr(r, "resource_composite", 0.0) or 0.0)
            _surv = float(getattr(r, "survival_score", 0.0) or 0.0)
            _dil = float(getattr(r, "dilution_score", 0.0) or 0.0)
            _jur = float(getattr(r, "jurisdiction_score", 0.0) or 0.0)
            _comm = float(getattr(r, "commodity_score", 0.0) or 0.0)
            _conf = float(getattr(r, "resource_confidence", 0.0) or 0.0)
            _profile = getattr(r, "resource_stage_profile", "") or _stage
            # PR4 enrichment transparency.
            _dq = getattr(r, "resource_data_quality", "") or "—"
            _proxy = getattr(r, "commodity_proxy", "") or "—"
            _as_of = getattr(r, "resource_data_as_of", "") or "—"
            _dq_hex = {"HIGH": P["green"], "MEDIUM": P["gold"]}.get(_dq, P["text_dim"])
            st.markdown(
                f'<div style="background:{P["surface"]};border:1px solid {P["border"]};'
                f'border-radius:6px;padding:8px 12px;margin-bottom:12px;'
                f'font-family:\'Courier New\',monospace;font-size:10px;'
                f'color:{P["text_dim"]}">'
                f'<b style="color:{P["text"]}">RESURS-COMPOSITE v1</b> '
                f'<span style="color:{P["text_dim"]}">(ej köpsignal · watchlist/ranking)</span><br>'
                f'Composite: <b style="color:{P["gold"]}">{_rc:.1f}</b>/100 '
                f'· Konfidens: <b>{_conf:.0%}</b> · Profil: <b>{_profile}</b><br>'
                f'Survival: <b>{_surv:.0f}</b> · Dilution: <b>{_dil:.0f}</b> · '
                f'Jurisdiktion: <b>{_jur:.0f}</b> · Commodity: <b>{_comm:.0f}</b><br>'
                f'Stage: <b>{_stage}</b> · Commodity: <b>{_commodity}</b> · '
                f'GateMode: <b>{_gate_mode}</b><br>'
                f'Datakvalitet: <b style="color:{_dq_hex}">{_dq}</b> · '
                f'Data as-of: <b>{_as_of}</b> · Proxy: <b>{_proxy}</b> '
                f'<span style="color:{P["text_dim"]}">(metadata, ej signal)</span><br>'
                f'ResourceFlags: {_flags_disp}<br>'
                f'<span style="color:{P["text_dim"]}">Tomma fält = saknas/'
                f'needs_validation, ej negativ signal.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Existing-source overlay (PR5) — context/watchlist only, NOT a buy
            # trigger and separate from the resource composite above.
            _ov = float(getattr(r, "resource_overlay_score", 0.0) or 0.0)
            _mcap = getattr(r, "market_cap_bucket", "") or "—"
            _liq = getattr(r, "liquidity_flag", "") or "—"
            _dd = getattr(r, "drawdown_52w_pct", None)
            _dd_disp = f"{_dd:.1f}%" if _dd is not None else "—"
            _crs = getattr(r, "commodity_relative_strength", None)
            _crs_flag = getattr(r, "commodity_rs_flag", "") or "—"
            _crs_disp = (
                f"{_crs:+.1f}pp · {_crs_flag}" if _crs is not None else _crs_flag
            )
            _si = getattr(r, "short_interest_flag", "") or "—"
            _ar = getattr(r, "analyst_revision_flag", "") or "—"
            _sa = getattr(r, "sentiment_attention_flag", "") or "—"
            _mac = getattr(r, "macro_context_flag", "") or "—"
            _es_flags = getattr(r, "existing_source_flags", []) or []
            _es_disp = ", ".join(_es_flags) if _es_flags else "inga"
            st.markdown(
                f'<div style="background:{P["surface"]};border:1px solid {P["border"]};'
                f'border-radius:6px;padding:8px 12px;margin-bottom:12px;'
                f'font-family:\'Courier New\',monospace;font-size:10px;'
                f'color:{P["text_dim"]}">'
                f'<b style="color:{P["text"]}">EXISTING-SOURCE OVERLAY</b> '
                f'<span style="color:{P["text_dim"]}">(kontext/watchlist · ej köpsignal)</span><br>'
                f'Overlay-score: <b style="color:{P["gold"]}">{_ov:.1f}</b>/100 '
                f'· Market cap: <b>{_mcap}</b> · Likviditet: <b>{_liq}</b><br>'
                f'52v drawdown: <b>{_dd_disp}</b> · Commodity RS: <b>{_crs_disp}</b><br>'
                f'Short interest: <b>{_si}</b> · Analytiker: <b>{_ar}</b><br>'
                f'Makro: <b>{_mac}</b> '
                f'<span style="color:{P["text_dim"]}">(FRED räntekurva)</span><br>'
                f'Sentiment: <b>{_sa}</b> '
                f'<span style="color:{P["text_dim"]}">(adapter klar · ej live-hämtad)</span><br>'
                f'OverlayFlags: {_es_disp}<br>'
                f'<span style="color:{P["text_dim"]}">Befintliga källor (yfinance/'
                f'EODHD/CSV). Saknad data = flagga, ej negativ signal.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        _section_title("VIKING REGIME")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
            f'  <span class="ca-regime" style="color:{regime_hex};border-color:{regime_hex}44;'
            f'  background:{regime_hex}11;font-size:13px;padding:4px 14px">'
            f'  ● {v_label}</span>'
            f'  <span style="font-family:\'Courier New\',monospace;font-size:10px;'
            f'  color:{P["text_dim"]}">OVTLYR Viking Regime</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        nine = r.catalyst_result.ovtlyr_nine if r.catalyst_result else None
        if nine is not None:
            st.markdown(
                f'<div style="font-family:\'Courier New\',monospace;font-size:10px;'
                f'color:{P["text_dim"]};margin-bottom:12px">'
                f'OVTLYR NINE: <b style="color:{P["gold"]}">{nine}/100</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        _section_title("NYCKELTAL")
        _render_metrics_grid(r)

        _section_title("AKTIVA FLAGGOR")
        if flags:
            st.markdown(format_flags_html(flags), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for f in flags:
                st.markdown(
                    f'<div style="font-family:\'Courier New\',monospace;font-size:10px;'
                    f'color:{P["text_dim"]};margin:4px 0;padding:4px 8px;'
                    f'border-left:2px solid {regime_hex}44">'
                    f'{f.emoji} <b>{f.label}</b>: {f.detail[:120]}…</div>'
                    if len(f.detail) > 120 else
                    f'<div style="font-family:\'Courier New\',monospace;font-size:10px;'
                    f'color:{P["text_dim"]};margin:4px 0;padding:4px 8px;'
                    f'border-left:2px solid {regime_hex}44">'
                    f'{f.emoji} <b>{f.label}</b>: {f.detail}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<p style="font-family:\'Courier New\',monospace;font-size:11px;'
                f'color:{P["green"]}">✓ Inga varningar</p>',
                unsafe_allow_html=True,
            )

    # ── Discipline fields (full width below the two-column layout) ────────
    _render_discipline_fields(r)


def _finite(value, default: float | None = 0.0) -> float | None:
    """Coerce ``value`` to a finite float.

    Returns ``default`` for None, NaN, inf, or anything non-numeric. Keeps the
    breakdown chart safe for US/CA resource rows whose sub-scores may be missing
    or NaN when fundamentals were unavailable. Pass ``default=None`` to detect
    missing values (rather than silently substituting a zero).
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# Per-pillar colors for the breakdown chart, keyed by label so the color list
# stays aligned even when pillars with missing data are skipped. Colors live in
# the render layer (not the pure data helper) so the helper stays palette-free
# and unit-testable in isolation.
_BREAKDOWN_COLOR_BY_LABEL = {
    "Necessity": "gold",
    "Hat":       "amber",
    "Quality":   "green",
    "Value":     "ice_blue",
    "Catalyst":  "silver",
    "Viking":    "gold_dim",
}


def build_breakdown_data(r, mode: str = "quality") -> dict:
    """Build equal-length, finite-numeric arrays for the score-breakdown chart.

    Pure (no Streamlit / Plotly) so it can be unit-tested with malformed rows.
    Each sub-score is coerced via :func:`_finite`; pillars whose raw score is
    missing (None / NaN / non-numeric / absent attribute) are **skipped
    entirely** rather than drawn as a misleading zero bar. As long as at least
    one pillar has data the chart renders — a single missing pillar never blanks
    the whole breakdown. Returns matching-length lists
    ``labels``/``scores_raw``/``contributions``/``max_contribs`` plus scalar
    ``total`` and bool ``has_data``.
    """
    if mode == "deep_contrarian":
        wmap = {"necessity": 0.15, "hate": 0.30, "quality": 0.20, "value": 0.20, "catalyst": 0.15}
    else:
        wmap = {"necessity": 0.15, "hate": 0.20, "quality": 0.30, "value": 0.20, "catalyst": 0.15}

    # (label, raw_score, weight). Use None as the getattr default so a genuinely
    # absent attribute is treated as missing (skipped), not as a real 0.
    specs = [
        ("Necessity", getattr(r, "necessity_score",  None), wmap["necessity"]),
        ("Hat",       getattr(r, "hat_score",        None), wmap["hate"]),
        ("Quality",   getattr(r, "quality_score",    None), wmap["quality"]),
        ("Value",     getattr(r, "value_score",      None), wmap["value"]),
        ("Catalyst",  getattr(r, "catalyst_score",   None), wmap["catalyst"]),
        ("Viking",    getattr(r, "viking_bonus_raw", None), 0.05),
    ]

    labels: list[str] = []
    scores_raw: list[float] = []
    contributions: list[float] = []
    max_contribs: list[float] = []
    for label, raw, weight in specs:
        raw_f = _finite(raw, None)
        if raw_f is None:
            continue  # skip pillar with missing/non-finite data
        labels.append(label)
        scores_raw.append(raw_f)
        contributions.append(round(raw_f * weight, 1))
        max_contribs.append(round(weight * 100, 1))

    return {
        "labels":        labels,
        "scores_raw":    scores_raw,
        "contributions": contributions,
        "max_contribs":  max_contribs,
        "total":         _finite(getattr(r, "composite_score", 0.0), 0.0),
        # At least one present pillar actually contributes. All-zero (or fully
        # missing) rows fall back to the neutral "nothing to show" caption rather
        # than the scary "kunde inte ritas" error.
        "has_data":      any(c > 0 for c in contributions),
    }


def _render_breakdown_chart(r) -> None:
    """Plotly horisontellt stapeldiagram med sub-score bidrag (5-pelare)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Plotly krävs för breakdown-diagram (pip install plotly).")
        return

    mode = st.session_state.get("ca_mode", "quality")
    data = build_breakdown_data(r, mode)

    if not data["has_data"]:
        st.caption("Ingen score-breakdown att visa (saknade/otillräckliga delpoäng).")
        return

    labels        = data["labels"]
    scores_raw    = data["scores_raw"]
    contributions = data["contributions"]
    max_contribs  = data["max_contribs"]
    total         = data["total"]
    # Color per kept pillar — aligned with ``labels`` even when missing pillars
    # were skipped upstream.
    colors        = [P[_BREAKDOWN_COLOR_BY_LABEL.get(lbl, "gold")] for lbl in labels]

    # Wrap all Plotly work so a malformed row can never crash the detail card.
    try:
        fig = go.Figure()

        # Max-bar (background)
        fig.add_trace(go.Bar(
            y=labels, x=max_contribs, orientation="h",
            name="Max möjlig",
            marker_color=[f"{c}22" for c in colors],
            marker_line_color=[f"{c}44" for c in colors],
            marker_line_width=1,
            hovertemplate="%{y}: max %{x:.1f}p<extra></extra>",
        ))

        # Actual-bar
        fig.add_trace(go.Bar(
            y=labels, x=contributions, orientation="h",
            name="Faktiskt bidrag",
            marker_color=colors,
            text=[f"{c:.1f}p  (raw {s:.0f})"
                  for c, s in zip(contributions, scores_raw)],
            textposition="inside",
            textfont=dict(size=10, color=P["bg"], family="Courier New"),
            hovertemplate="%{y}: %{x:.1f}p<extra></extra>",
        ))

        # Total linje
        fig.add_vline(
            x=total,
            line_dash="dash",
            line_color=P["gold"],
            annotation_text=f"  Total: {total:.1f}",
            annotation_font_color=P["gold"],
            annotation_font_size=11,
        )

        fig.update_layout(
            barmode="overlay",
            plot_bgcolor=P["bg2"],
            paper_bgcolor=P["bg"],
            font=dict(family="Courier New", color=P["text"], size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            showlegend=False,
            xaxis=dict(
                range=[0, 105],
                gridcolor=P["border"],
                tickfont=dict(size=9),
                title=dict(text="Bidrag till Composite Score (max 100p)", font=dict(size=9)),
            ),
            yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        )

        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    except Exception as e:  # pragma: no cover - defensive Plotly guardrail
        # We only reach here with has_data == True (≥1 pillar). A Plotly failure
        # is a render problem, not missing data, so don't show the misleading
        # "dataproblem" message — degrade to a compact text breakdown instead.
        logger.warning("Breakdown chart render failed for %s: %s",
                       getattr(r, "ticker", "?"), e)
        _txt = "  ·  ".join(
            f"{lbl} {c:.1f}p" for lbl, c in zip(labels, contributions)
        )
        st.caption(f"Score-breakdown (förenklad): {_txt}  →  Total {total:.1f}")

    # Gate-sammanfattning under diagrammet.
    # Status is three-state: "pass" (grön ✓), "fail" (röd ✗ — värdet FINNS och
    # bryter gränsen) eller "missing" (grå "– saknas" — underliggande Börsdata-
    # fält är None, så rött ✗ vore missvisande).
    gate_checks: list[tuple[str, str]] = []
    _mode = st.session_state.get("ca_mode", "quality")

    if r.strength_result:
        # Prefer the three-state gate_status; fall back to bool gate_results for
        # older StrengthResult objects that predate the status field.
        gate_status = getattr(r.strength_result, "gate_status", None) or {}
        gates = r.strength_result.gate_results

        def _strength_status(key: str) -> str:
            if key in gate_status:
                return gate_status[key]
            return "pass" if gates.get(key, False) else "fail"

        gate_checks += [
            ("FCF > 0",        _strength_status("fcf_positive")),
            ("EBITDA > 0%",    _strength_status("ebitda_margin_positive")),
            ("Equity > 0",     _strength_status("equity_positive")),
            ("Altman Z > 1.8", _strength_status("altman_z_ok")),
        ]
        # Leverage gate label depends on mode
        if _mode == "quality":
            nd_e = getattr(r, "net_debt_ebitda", None)
            _nd_status = "missing" if nd_e is None else ("pass" if float(nd_e) <= 3.5 else "fail")
            gate_checks.append(("ND/EBITDA ≤ 3.5", _nd_status))
        else:
            gate_checks.append(("D/E < 0.6", _strength_status("debt_equity_low")))

    # ROIC gate
    qr = getattr(r, "quality_result", None)
    if qr is not None and qr.roic is not None:
        _gate_pct = "15%" if _mode == "quality" else "10%"
        _roic_pass = qr.passes_gate_quality if _mode == "quality" else qr.passes_gate_deep
        gate_checks.append((f"ROIC > {_gate_pct}", "pass" if _roic_pass else "fail"))

    # Valuation bands gate (quality mode only). NO_DATA → missing (grå), not a
    # broken band; TOO_CHEAP/OK → pass.
    if _mode == "quality":
        vb = getattr(r, "valuation_bands", None)
        if vb is not None:
            def _band_status(band_status: str) -> str:
                if band_status == "NO_DATA":
                    return "missing"
                return "pass" if band_status in ("OK", "TOO_CHEAP") else "fail"
            gate_checks.append(("P/E band [7–25]",    _band_status(vb.pe_status)))
            gate_checks.append(("EV/EBIT band [4–20]", _band_status(vb.ev_ebit_status)))

    if gate_checks:
        _gate_icons = {
            "pass":    f'<span style="color:{P["green"]}">✓</span>',
            "fail":    f'<span style="color:{P["red"]}">✗</span>',
            "missing": f'<span style="color:{P["text_dim"]}">–</span>',
        }
        parts = []
        for label, gstatus in gate_checks:
            ico = _gate_icons.get(gstatus, _gate_icons["missing"])
            _lbl = f"{label} · saknas" if gstatus == "missing" else label
            parts.append(f'{ico} <span style="color:{P["text_dim"]}">{_lbl}</span>')
        st.markdown(
            f'<div style="font-family:\'Courier New\',monospace;font-size:9px;'
            f'letter-spacing:0.08em;display:flex;flex-wrap:wrap;gap:12px;'
            f'padding:6px 0">' + "".join(parts) + "</div>",
            unsafe_allow_html=True,
        )


def _render_metrics_grid(r) -> None:
    """3-kolumns nyckeltalsgrid under Viking-badge (Quality-Contrarian 5-pelare)."""

    mode = st.session_state.get("ca_mode", "quality")

    # SMA200-avvikelse
    sma200_dev = None
    if r.sma200 and r.close and r.sma200 > 0:
        sma200_dev = (r.close - r.sma200) / r.sma200 * 100

    # P/FCF discount vs own history (from value_result)
    vr = getattr(r, "value_result", None)
    pfcf_disc = getattr(vr, "p_fcf_discount", None) if vr else None
    pfcf_sub = f"disc {pfcf_disc:+.0f}% vs hist" if pfcf_disc is not None else "multipel"

    roic_val = getattr(r, "roic", None)

    # Base metrics (both modes)
    metrics = [
        ("ROIC %",      _fmt(roic_val, ".1f"),        "Avkastning på inv.kap"),
        ("P/FCF",       _fmt(r.p_fcf, ".1f"),         pfcf_sub),
        ("EV/EBITDA",   _fmt(r.ev_ebitda, ".1f"),     "Multipel"),
        ("FCF",         _fmt(r.fcf_m, ".0f"),         "MSEK (TTM)"),
        ("EBITDA %",    _fmt(r.ebitda_pct, ".1f"),    "Marginal %"),
        ("SMA200 dev",  _fmt(sma200_dev, "+.1f"),     "% från SMA200"),
        ("Altman Z",    _fmt(r.altman_z, ".2f"),      ">2.99 safe"),
        ("D/E",         _fmt(r.debt_equity, ".2f"),   "Skuld/Eget kapital"),
    ]

    # KAP metrics (quality mode only — shown when data is present)
    if mode == "quality":
        # Net Debt / EBITDA (replaces D/E gate in quality mode)
        nd_e = getattr(r, "net_debt_ebitda", None)
        nd_sub = "≤ 3.5 OK  |  ≤ 0 nettokassa"
        metrics.append(("ND/EBITDA", _fmt(nd_e, ".1f"), nd_sub))

        # P/E band status
        pe_val = getattr(r, "pe_ratio", None)
        vb = getattr(r, "valuation_bands", None)
        pe_band_str = ""
        if vb is not None and vb.pe_status != "NO_DATA":
            pe_band_str = f"[{vb.pe_status}]"
        pe_sub = f"band 7–25  {pe_band_str}".strip()
        metrics.append(("P/E", _fmt(pe_val, ".1f"), pe_sub))

        # EV/EBIT band
        ev_ebit_val = getattr(r, "ev_ebit_ratio", None)
        eb_band_str = ""
        if vb is not None and vb.ev_ebit_status != "NO_DATA":
            eb_band_str = f"[{vb.ev_ebit_status}]"
        metrics.append(("EV/EBIT", _fmt(ev_ebit_val, ".1f"), f"band 4–20  {eb_band_str}".strip()))

        # Revenue growth CAGRs
        rev5  = getattr(r, "revenue_cagr_5y",  None)
        rev10 = getattr(r, "revenue_cagr_10y", None)
        eps10 = getattr(r, "eps_cagr_10y",     None)
        metrics.append(("Rev CAGR 5y",  _fmt(rev5,  ".1f") + ("%" if rev5  is not None else ""), "≥5% KAP"))
        metrics.append(("Rev CAGR 10y", _fmt(rev10, ".1f") + ("%" if rev10 is not None else ""), "≥5% KAP"))
        metrics.append(("EPS CAGR 10y", _fmt(eps10, ".1f") + ("%" if eps10 is not None else ""), "≥5% KAP"))

        # Dividend yield
        div_pct = getattr(r, "dividend_yield_pct", None)
        metrics.append(("Utdelning %",  _fmt(div_pct, ".1f"), "≥1% KAP bonus"))

    # Färgkoda
    def _metric_val_color(label: str, val_str: str) -> str:
        if val_str in ("—", ""):
            return P["text_dim"]
        raw = val_str.replace("+", "").replace("%", "").strip()
        if label == "ROIC %":
            try:
                v = float(raw)
                return P["green"] if v >= 15 else P["amber"] if v >= 10 else P["red"]
            except ValueError:
                return P["text"]
        if label == "SMA200 dev":
            try:
                v = float(raw)
                return P["green"] if v > 0 else P["red"]
            except ValueError:
                return P["text"]
        if label == "Altman Z":
            try:
                v = float(raw)
                return P["green"] if v > 2.99 else P["amber"] if v > 1.8 else P["red"]
            except ValueError:
                return P["text"]
        if label == "D/E":
            try:
                v = float(raw)
                return P["green"] if v < 0.3 else P["amber"] if v < 0.6 else P["red"]
            except ValueError:
                return P["text"]
        if label == "ND/EBITDA":
            try:
                v = float(raw)
                return P["green"] if v <= 1.5 else P["amber"] if v <= 3.5 else P["red"]
            except ValueError:
                return P["text"]
        if label in ("P/E", "EV/EBIT"):
            # Color by band status
            if "[OK]" in val_str or "[OK]" in (
                val_str if isinstance(val_str, str) else ""
            ):
                return P["green"]
            if "[EXPENSIVE]" in str(val_str):
                return P["red"]
            if "[TOO_CHEAP]" in str(val_str):
                return P["amber"]
        if label in ("Rev CAGR 5y", "Rev CAGR 10y", "EPS CAGR 10y"):
            try:
                v = float(raw)
                return P["green"] if v >= 5.0 else P["amber"] if v >= 3.0 else P["red"]
            except ValueError:
                return P["text"]
        if label == "Utdelning %":
            try:
                v = float(raw)
                return P["green"] if v >= 1.0 else P["text_dim"]
            except ValueError:
                return P["text"]
        return P["text"]

    html = '<div class="ca-detail-grid">'
    for label, val_str, sub in metrics:
        vc = _metric_val_color(label, val_str)
        html += (
            f'<div class="ca-metric">'
            f'  <div class="ca-metric-label">{label}</div>'
            f'  <div class="ca-metric-value" style="color:{vc}">{val_str}</div>'
            f'  <div class="ca-metric-sub">{sub}</div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─── Resource watchlists (us_ca_resource only) ───────────────────────────────

_WATCHLIST_COL_LABELS = {
    "ticker":                       "Ticker",
    "name":                         "Namn",
    "stage":                        "Stage",
    "commodity":                    "Commodity",
    "resource_composite":           "Composite",
    "resource_overlay_score":       "Overlay",
    "drawdown_52w_pct":             "52v DD %",
    "commodity_rs_flag":            "Commodity RS",
    "resource_data_quality":        "Datakvalitet",
    "liquidity_flag":               "Likviditet",
    "flags":                        "Flaggor",
}


def _render_resource_watchlists(pipeline_result) -> None:
    """Practical watchlist slices for the US/CA Resource universe (PR15).

    Watchlist views only — reuses existing output fields, changes no scoring
    math, and is rendered only for config.universe == 'us_ca_resource'. The
    main results table above is untouched.
    """
    cfg = getattr(pipeline_result, "config", None)
    if getattr(cfg, "universe", "") != "us_ca_resource":
        return
    results = getattr(pipeline_result, "results", None)
    if not results:
        return

    try:
        from contrarian_alpha.resource_watchlists import (
            build_resource_watchlists, WATCHLIST_ORDER, WATCHLIST_META,
            WATCHLIST_COLUMNS,
        )
        views = build_resource_watchlists(results)
    except Exception as e:  # pragma: no cover - defensive UI guardrail
        logger.warning("Resource watchlists failed: %s", e)
        return

    import pandas as pd

    with st.expander("🗂️ Resurs-watchlists (slices, ej köpsignal)", expanded=False):
        st.caption(
            "Praktiska bevakningsvyer byggda på befintliga resurs-fält. "
            "Detta är **watchlist-slices för manuell granskning, inte köpsignaler**. "
            "Tomma fält = saknas/needs_validation, ej negativ signal."
        )
        tab_labels = [f"{WATCHLIST_META[k]['title']} ({len(views[k])})"
                      for k in WATCHLIST_ORDER]
        tabs = st.tabs(tab_labels)
        for tab, key in zip(tabs, WATCHLIST_ORDER):
            with tab:
                meta = WATCHLIST_META[key]
                st.caption(meta["caption"])
                rows = views[key]
                if not rows:
                    st.caption("— Inga bolag i denna vy just nu.")
                    continue
                df = pd.DataFrame(rows)
                # Keep only known columns that are present, in the recommended order.
                cols = [c for c in WATCHLIST_COLUMNS if c in df.columns]
                df = df[cols].rename(columns=_WATCHLIST_COL_LABELS)
                st.dataframe(df, use_container_width=True, hide_index=True)


# ─── Elimineringsinformation ─────────────────────────────────────────────────

def _render_elimination_breakdown(pipeline_result) -> None:
    """Visar varför listan är tom — fördelning per gate."""
    elim = pipeline_result.eliminated
    if not elim:
        return

    from collections import Counter
    stage_counts = Counter(r.elimination_stage for r in elim)

    _section_title("ELIMINERINGSANALYS")
    cols = st.columns(len(stage_counts) + 1)
    for i, (stage, count) in enumerate(sorted(stage_counts.items())):
        pct = count / pipeline_result.universe_count * 100 if pipeline_result.universe_count else 0
        with cols[i]:
            st.metric(f"❌ {stage}", count, f"{pct:.0f}% av universum")
    with cols[-1]:
        st.metric("✅ Passerade alla", pipeline_result.composite_ranked)


# ─── Huvud-entry point ───────────────────────────────────────────────────────


def _render_cached_preview(mode: str) -> None:
    """Show pre-computed scheduled scan results (from Gist) as an instant preview."""
    try:
        from contrarian_alpha.cache import load_screener_results
        saved = load_screener_results(mode=mode)
    except Exception:
        saved = {}
    results = saved.get("results", []) if saved else []
    ts = saved.get("timestamp", "") if saved else ""
    if not results:
        st.info("Klicka **Kor scan** for att starta screenern. "
                "(Inga schemalagda resultat hittades an - de uppdateras vardagar 08/12/18.)")
        return

    ts_disp = str(ts)[:16].replace("T", " ")
    st.markdown(
        f"<div style='background:#1A1F25;border-left:3px solid #00E5FF;border-radius:8px;"
        f"padding:10px 14px;margin-bottom:12px;'>"
        f"<b style='color:#E8EDF2;'>Schemalagt resultat</b> "
        f"<span style='color:#6B7280;'>&middot; senast uppdaterad {ts_disp} "
        f"&middot; {len(results)} bolag &middot; lage: {mode}</span><br>"
        f"<span style='font-size:0.8rem;color:#9aa4b0;'>Klicka <b>Kor scan</b> for live-analys "
        f"med fullstandiga kort.</span></div>",
        unsafe_allow_html=True,
    )
    import pandas as pd
    rows = []
    for r in results[:40]:
        rows.append({
            "#": r.get("rank", ""),
            "Ticker": r.get("ticker", ""),
            "Namn": (r.get("name") or "")[:24],
            "Score": round(r.get("composite_score", 0), 1),
            "Necessity": round(r.get("necessity_score", 0), 0),
            "Hat": round(r.get("hat_score", 0), 0),
            "Quality": round(r.get("strength_score", 0), 0),
            "Catalyst": round(r.get("catalyst_score", 0), 0),
            "Pris": r.get("close", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_contrarian_alpha_page() -> None:
    """
    Huvud-entry point för Contrarian Alpha Screener-fliken.
    Anropas från wolf_panel.py:

        from contrarian_alpha.ui import render_contrarian_alpha_page
        with tab_contrarian_alpha:
            render_contrarian_alpha_page()
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Header ──
    st.markdown(
        f'<div style="border-top:2px solid {P["gold"]};padding-top:14px;margin-bottom:4px">'
        f'  <span style="font-family:\'Courier New\',monospace;font-size:18px;'
        f'  font-weight:900;letter-spacing:0.2em;text-transform:uppercase;'
        f'  color:{P["gold"]}">🎯 CONTRARIAN ALPHA</span>'
        f'  <span style="font-family:\'Courier New\',monospace;font-size:10px;'
        f'  letter-spacing:0.25em;text-transform:uppercase;color:{P["gold_muted"]};'
        f'  margin-left:16px">SCREENER</span>'
        f'</div>'
        f'<p style="font-family:\'Courier New\',monospace;font-size:10px;'
        f'letter-spacing:0.12em;color:{P["text_dim"]};margin:4px 0 20px 0">'
        f'Necessity · Hate · Quality · Value · Catalyst · Viking Regime</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Zon 1: Kontrollpanel ──
    config_kwargs, run_now = _render_control_panel()

    st.divider()

    # ── Pipeline ──
    pipeline_result = _get_or_run_pipeline(config_kwargs, run_now)

    if pipeline_result is None:
        _mode_now = config_kwargs.get("mode", "quality")
        _render_cached_preview(_mode_now)
        return

    # ── Pipeline-stats-bar ──
    stats = st.session_state.get("ca_last_stats", {})
    if stats:
        pr = pipeline_result.pass_rates
        st.markdown(
            f'<div style="background:{P["bg2"]};border:1px solid {P["border"]};'
            f'border-radius:6px;padding:8px 16px;margin-bottom:16px;'
            f'font-family:\'Courier New\',monospace;font-size:10px;'
            f'letter-spacing:0.08em;display:flex;flex-wrap:wrap;gap:16px">'
            f'<span style="color:{P["text_dim"]}">Necessity: '
            f'<b style="color:{P["gold"]}">{pr.get("necessity","—")}</b></span>'
            f'<span style="color:{P["text_dim"]}">Hate: '
            f'<b style="color:{P["gold"]}">{pr.get("hate","—")}</b></span>'
            f'<span style="color:{P["text_dim"]}">Balance Sheet: '
            f'<b style="color:{P["gold"]}">{pr.get("balance_sheet","—")}</b></span>'
            f'<span style="color:{P["text_dim"]}">Rankade: '
            f'<b style="color:{P["green"]}">{pr.get("ranked","—")}</b></span>'
            f'<span style="color:{P["text_dim"]}">Tid: '
            f'<b style="color:{P["silver"]}">{pipeline_result.run_duration_s:.1f}s</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Zon 2 + 3: Tabell + Detaljkort ──
    _render_results_table(pipeline_result)

    # ── Resurs-watchlists (endast us_ca_resource) ──
    _render_resource_watchlists(pipeline_result)

    # ── Elimineringsanalys (alltid synlig) ──
    if pipeline_result.eliminated:
        with st.expander(
            f"📊 Elimineringsanalys — {len(pipeline_result.eliminated)} bolag eliminerade",
            expanded=False,
        ):
            _render_elimination_breakdown(pipeline_result)
            # Top 5 per stage
            from collections import defaultdict
            by_stage: dict = defaultdict(list)
            for r in pipeline_result.eliminated:
                by_stage[r.elimination_stage].append(r)
            for stage, items in sorted(by_stage.items()):
                st.markdown(
                    f'<p style="font-family:\'Courier New\',monospace;font-size:10px;'
                    f'color:{P["amber"]};letter-spacing:0.1em;margin:10px 0 4px">▶ {stage}</p>',
                    unsafe_allow_html=True,
                )
                for item in items[:8]:
                    st.markdown(
                        f'<span style="font-family:\'Courier New\',monospace;font-size:10px;'
                        f'color:{P["text_dim"]}">'
                        f'<b style="color:{P["text"]}">{item.ticker}</b>  '
                        f'— {item.elimination_reason}</span><br>',
                        unsafe_allow_html=True,
                    )
