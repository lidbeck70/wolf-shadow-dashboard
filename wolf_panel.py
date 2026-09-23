#!/usr/bin/env python3
"""
Nordic Arc Systems — Trading & Investing
=========================================
Mission Control dashboard for Nordic swing trading intelligence.

Run:
    streamlit run wolf_panel.py
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATH SETUP — find screener/backtester modules
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(DASHBOARD_DIR)

WOLFPANEL_DIR = os.path.join(WORKSPACE_DIR, "wolfpanel")

for p in [
    DASHBOARD_DIR,
    os.path.join(WORKSPACE_DIR, "screener"),
    os.path.join(WORKSPACE_DIR, "backtester"),
    WORKSPACE_DIR,
    WOLFPANEL_DIR,
]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Nordic Arc Systems",
    page_icon="🔱",
)

# PWA / Mobile meta tags for iPad home screen
st.markdown(
    """
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Nordic Arc">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#05070A">
    <link rel="apple-touch-icon" href="https://em-content.zobj.net/source/apple/391/trident_1f531.png">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LOCAL MODULE IMPORTS
# ---------------------------------------------------------------------------
from ui.css import tab_not_found
from ui import nav
from ui.theme import inject_css, render_header, render_footer
from auth import render_login_gate
from tabs.home import tab_home
from tabs.screener import tab_screener, render_viking_screener
from tabs.backtest import tab_backtest_consolidated
from tabs.regime import tab_regime
from tabs.alerts import tab_alerts
from tabs.strategy_overview import tab_strategy_overview

# Sector & Global Regime
try:
    from sector_cycle.sector_cycle_streamlit import render_sector_cycle_page
    SECTOR_CYCLE_AVAILABLE = True
except ImportError:
    SECTOR_CYCLE_AVAILABLE = False

# Sentiment & Flow
try:
    from sentiment.sentiment_streamlit import render_sentiment_page
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Heatmap
try:
    from heatmap.heatmap_streamlit import render_heatmap_page
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False

# OVTLYR / Viking Regime
try:
    from ovtlyr.ui.layout import render_ovtlyr_page
    OVTLYR_AVAILABLE = True
except ImportError:
    OVTLYR_AVAILABLE = False

# Rules page
try:
    from ovtlyr.ui.rules_page import render_rules_page
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False

# Inline rules helper for regime tabs
try:
    from ovtlyr.ui.rules_page import render_inline_rules
except ImportError:
    try:
        from rules_page import render_inline_rules
    except ImportError:
        render_inline_rules = None

# Retail Sentiment
try:
    from retail_sentiment import render_retail_sentiment_page
    RETAIL_SENTIMENT_AVAILABLE = True
except ImportError:
    RETAIL_SENTIMENT_AVAILABLE = False

# Odin's Blindspot Index
try:
    from blindspot import render_blindspot_page
    BLINDSPOT_AVAILABLE = True
except ImportError:
    BLINDSPOT_AVAILABLE = False

# Alpha Regime — dual-mode confirmation system (Quality / Deep Contrarian)
try:
    from alpha_regime.ui import render_alpha_regime
    ALPHA_REGIME_AVAILABLE = True
except ImportError:
    ALPHA_REGIME_AVAILABLE = False

# Legacy long-trend monitor (fallback)
try:
    from long_regime_monitor import render_long_regime_monitor
    LONG_REGIME_AVAILABLE = True
except ImportError:
    LONG_REGIME_AVAILABLE = False

# Holdings
try:
    from holdings import render_holdings_page
    HOLDINGS_AVAILABLE = True
except ImportError:
    HOLDINGS_AVAILABLE = False

# Swing (momentum veckorutin)
try:
    from swing import render_swing_page
    SWING_AVAILABLE = True
except ImportError:
    SWING_AVAILABLE = False

# Tiggre (Lobo-arket — Masterguiden Del 4)
try:
    from tiggre import render_tiggre_page
    TIGGRE_AVAILABLE = True
except ImportError:
    TIGGRE_AVAILABLE = False

# Persistens — data/<flik>.json via GitHub Contents API
import storage_ui

# Master Scorecard + köpgrinden (Masterguiden 4.0)
try:
    from scorecard import render_scorecard_page
    SCORECARD_AVAILABLE = True
except ImportError:
    SCORECARD_AVAILABLE = False

# Granskningsarken — Rick Rule + Royalty C (ur ravarurotation.xlsx)
try:
    from producers import render_producers_page
    PRODUCERS_AVAILABLE = True
except ImportError:
    PRODUCERS_AVAILABLE = False

# Poängmodellen (ersätter poangmodell_sprott_durrett.xlsx)
try:
    from scoring import render_scoring_page
    SCORING_AVAILABLE = True
except ImportError:
    SCORING_AVAILABLE = False

# Insiderbevakaren (ersätter insiderbevakaren.xlsx)
try:
    from insider import render_insider_page
    INSIDER_AVAILABLE = True
except ImportError:
    INSIDER_AVAILABLE = False

# Confidence score — Case Score + Confidence Score för gruv-/råvarubolag
try:
    from confidence import render_confidence_page
    CONFIDENCE_AVAILABLE = True
except ImportError:
    CONFIDENCE_AVAILABLE = False

# Durrett — Don Durretts 10-stegsmetod (engines/durrett), delar lagret med Confidence score
try:
    from engines.durrett.ui import render_durrett_page
    DURRETT_AVAILABLE = True
except ImportError:
    DURRETT_AVAILABLE = False

# Swing momentum-screener + regim (data från wolf_data.py)
try:
    from wolf_screener_ui import render_wolf_screener_page
    WOLF_SCREENER_AVAILABLE = True
except ImportError:
    WOLF_SCREENER_AVAILABLE = False

try:
    from wolf_regime_ui import render_wolf_regime_page
    WOLF_REGIME_AVAILABLE = True
except ImportError:
    WOLF_REGIME_AVAILABLE = False

# Råvarurotationen (Masterguiden Del 3)
try:
    from rotation import render_rotation_page
    ROTATION_AVAILABLE = True
except ImportError:
    ROTATION_AVAILABLE = False

# Portföljallokeraren (Masterguiden Del 2)
try:
    from allocator import render_allocator_page
    ALLOCATOR_AVAILABLE = True
except ImportError:
    ALLOCATOR_AVAILABLE = False

# Trade Journal
try:
    from trade_journal import render_trade_journal_page
    JOURNAL_AVAILABLE = True
except ImportError:
    JOURNAL_AVAILABLE = False

# Contrarian Alpha Screener
try:
    from contrarian_alpha.ui import render_contrarian_alpha_page
    CONTRARIAN_ALPHA_AVAILABLE = True
except ImportError:
    CONTRARIAN_ALPHA_AVAILABLE = False

# CAGR / Long Screener
try:
    from cagr.cagr_streamlit import render_cagr_page
    CAGR_AVAILABLE = True
except ImportError:
    CAGR_AVAILABLE = False

# Market Cycle Engine
try:
    from tabs.market_cycle import render_market_cycle_page
    MARKET_CYCLE_AVAILABLE = True
except ImportError:
    MARKET_CYCLE_AVAILABLE = False

# EMBER — commodity swing strategy + regime gauge
try:
    from ember.ui import render_ember_page
    from ember.regime import render_ember_regime_page
    EMBER_AVAILABLE = True
except ImportError:
    EMBER_AVAILABLE = False
    render_ember_regime_page = None


# =============================================================================
# MAIN APP
# =============================================================================

def _is_open(tab) -> bool:
    """Bara den öppna toppfliken renderas (st.tabs on_change="rerun" sätter
    tab.open). None = spårning av; då renderas allt som förr."""
    return tab.open is not False


def _sub(path: str) -> str:
    """Underflikarna ur navigationsträdet (ui/nav.py) — samma lista som
    FLIKGUIDEN och Home ritar, med widget-nyckeln ur nav.STATE_KEY så att
    djuplänken kan seeda den. Etiketten finns (skärmläsare) men döljs."""
    return st.radio(f"Val i {path.replace('/', ' → ')}", nav.options(path),
                    label_visibility="collapsed", horizontal=True,
                    key=nav.STATE_KEY[path])


def _apply_deep_link() -> None:
    """?p=regime/marknad/arc-regime/wolf-regime i adressfältet, eller
    nav_goto från ett Home-kort, väljer flik och underflikar INNAN
    widgetarna ritas. Bara en gång per mål — annars skulle varje omritning
    dra användaren tillbaka till länken när hen klickat vidare."""
    target = st.session_state.pop("nav_goto", None) or st.query_params.get("p") or ""
    if not target or target == st.session_state.get("_nav_applied"):
        return
    st.session_state["_nav_applied"] = target
    segs = nav.resolve(target)
    if segs:
        nav.seed(st.session_state, segs)


def _sync_url() -> None:
    """Adressfältet följer fliken, så URL:en alltid går att kopiera som
    djuplänk. Ingen omritning — bara URL:en byts."""
    slug = nav.slug(nav.current(st.session_state))
    st.session_state["_nav_applied"] = slug
    if st.query_params.get("p") != slug:
        st.query_params["p"] = slug


def _page(available: bool, render, name: str, module: str, rules_key: str = "") -> None:
    if not available or render is None:
        tab_not_found(name, module)
        return
    render()
    if rules_key and render_inline_rules:
        try:
            render_inline_rules(rules_key)
        except Exception:
            pass


def main():
    if not render_login_gate():
        return

    inject_css()
    render_header()
    _apply_deep_link()

    tab_labels = [nav.tab_label(k) for k, _label in nav.TOP]
    tabs = dict(zip([k for k, _l in nav.TOP],
                    st.tabs(tab_labels, on_change="rerun", key="main_tabs")))

    # ── HOME ─────────────────────────────────────────────────────────────────
    if _is_open(tabs["home"]):
        with tabs["home"]:
            tab_home()

    # ── SCREENING ─────────────────────────────────────────────────────────────
    if _is_open(tabs["screening"]):
        with tabs["screening"]:
            sub = _sub("screening")
            st.markdown("---")
            if sub == "Arc Screener":
                inner = _sub("screening/Arc Screener")
                if inner == "Wolf":
                    tab_screener()
                elif inner == "Viking":
                    _page(OVTLYR_AVAILABLE, render_viking_screener, "Viking Screener", "screener_ovtlyr")
                elif inner == "🔥 EMBER":
                    _page(EMBER_AVAILABLE, render_ember_page, "EMBER", "ember")
            elif sub == "Contrarian Alpha":
                inner = _sub("screening/Contrarian Alpha")
                if inner == "Screener":
                    _page(CONTRARIAN_ALPHA_AVAILABLE, render_contrarian_alpha_page,
                          "Contrarian Alpha", "contrarian_alpha")
                elif inner == "Long Screener":
                    _page(CAGR_AVAILABLE, render_cagr_page, "Long Screener", "cagr")
            elif sub == "Swing Screener":
                _page(WOLF_SCREENER_AVAILABLE, render_wolf_screener_page, "Swing Screener", "wolf_screener_ui")

    # ── GRANSKNING ───────────────────────────────────────────────────────────
    # Beslutsunderlaget efter screeningen. Inget här screenar — screeningen
    # sker i Börsdata; det här är arken som avgör vilket bolag som köps.
    if _is_open(tabs["review"]):
        with tabs["review"]:
            sub = _sub("review")
            st.markdown("---")
            if sub in ("Rick Rule", "Royalty C"):
                if PRODUCERS_AVAILABLE:
                    render_producers_page(sheet=sub)
                else:
                    tab_not_found("Granskningsarken", "producers")
            elif sub == "Poängmodell":
                _page(SCORING_AVAILABLE, render_scoring_page, "Poängmodellen", "scoring")
            elif sub == "Tiggre":
                _page(TIGGRE_AVAILABLE, render_tiggre_page, "Tiggre", "tiggre")
            elif sub == "Insider":
                _page(INSIDER_AVAILABLE, render_insider_page, "Insiderbevakaren", "insider")
            elif sub == "🧭 Durrett & Confidence":
                # Två vyer över samma ark (data/confidence.json) — en flik.
                inner = _sub("review/🧭 Durrett & Confidence")
                if inner == "Durrett 10-steg":
                    _page(DURRETT_AVAILABLE, render_durrett_page, "Durrett", "engines/durrett/ui")
                else:
                    _page(CONFIDENCE_AVAILABLE, render_confidence_page, "Confidence score", "confidence")
            elif sub == "🎯 Scorecard":
                _page(SCORECARD_AVAILABLE, render_scorecard_page, "Master Scorecard", "scorecard")

    # ── REGIME ───────────────────────────────────────────────────────────────
    # Delat i två: marknaden (index, sektorer, cykel) och råvarorna.
    if _is_open(tabs["regime"]):
        with tabs["regime"]:
            group = _sub("regime")
            if group == "Marknad":
                sub = _sub("regime/Marknad")
                st.markdown("---")
                if sub == "Arc Regime":
                    inner = _sub("regime/Marknad/Arc Regime")
                    if inner == "Wolf Regime":
                        _page(True, tab_regime, "Wolf Regime", "tabs/regime", rules_key="wolf")
                    elif inner == "Viking Regime":
                        _page(OVTLYR_AVAILABLE, render_ovtlyr_page, "OVTLYR", "ovtlyr", rules_key="viking")
                elif sub == "Alpha Regime":
                    inner = _sub("regime/Marknad/Alpha Regime")
                    if inner == "Quality & Contrarian":
                        _page(ALPHA_REGIME_AVAILABLE, render_alpha_regime, "Alpha Regime Monitor",
                              "alpha_regime", rules_key="alpha")
                    elif inner == "Long Trend":
                        _page(LONG_REGIME_AVAILABLE, render_long_regime_monitor, "Long Trend Monitor",
                              "long_regime_monitor")
                elif sub == "Swing Regime":
                    _page(WOLF_REGIME_AVAILABLE, render_wolf_regime_page, "Swing Regime", "wolf_regime_ui")
                elif sub == "Flow Divergence":
                    _page(SECTOR_CYCLE_AVAILABLE, render_sector_cycle_page, "Sector & Global Regime",
                          "sector_cycle")
                elif sub == "Market Cycle":
                    _page(MARKET_CYCLE_AVAILABLE, render_market_cycle_page, "Market Cycle Engine",
                          "tabs/market_cycle")
            else:
                sub = _sub("regime/Råvaror")
                st.markdown("---")
                if sub == "🌍 EMBER Regime":
                    _page(EMBER_AVAILABLE, render_ember_regime_page, "EMBER Regime", "ember")
                elif sub == "Råvarurotation":
                    _page(ROTATION_AVAILABLE, render_rotation_page, "Råvarurotationen", "rotation")

    # ── INTELLIGENCE ─────────────────────────────────────────────────────────
    if _is_open(tabs["intel"]):
        with tabs["intel"]:
            sub = _sub("intel")
            st.markdown("---")
            if sub == "Odin's Blindspot":
                _page(BLINDSPOT_AVAILABLE, render_blindspot_page, "Odin's Blindspot Index", "blindspot")
            elif sub == "Sentiment":
                _page(SENTIMENT_AVAILABLE, render_sentiment_page, "Sentiment & Flow", "sentiment")
            elif sub == "Retail Pulse":
                _page(RETAIL_SENTIMENT_AVAILABLE, render_retail_sentiment_page, "Retail Sentiment",
                      "retail_sentiment")
            elif sub == "Heatmap":
                _page(HEATMAP_AVAILABLE, render_heatmap_page, "Heatmap", "heatmap")

    # ── PORTFOLIO ─────────────────────────────────────────────────────────────
    if _is_open(tabs["portfolio"]):
        with tabs["portfolio"]:
            sub = _sub("portfolio")
            st.markdown("---")
            if sub == "📓 Trade Journal":
                _page(JOURNAL_AVAILABLE, render_trade_journal_page, "Trade Journal", "trade_journal")
            elif sub == "Holdings":
                _page(HOLDINGS_AVAILABLE, render_holdings_page, "Holdings", "holdings")
            elif sub == "Swing":
                _page(SWING_AVAILABLE, render_swing_page, "Swing", "swing")
            elif sub == "Allokering":
                _page(ALLOCATOR_AVAILABLE, render_allocator_page, "Portföljallokeraren", "allocator")
            elif sub == "Backtest":
                tab_backtest_consolidated()

    # ── ALERTS ───────────────────────────────────────────────────────────────
    if _is_open(tabs["alerts"]):
        with tabs["alerts"]:
            tab_alerts()

    # ── RULES ────────────────────────────────────────────────────────────────
    if _is_open(tabs["rules"]):
        with tabs["rules"]:
            sub_rules = _sub("rules")
            st.markdown("---")
            if sub_rules == "Regler & Guider":
                _page(RULES_AVAILABLE, render_rules_page, "Rules", "ovtlyr/ui")
            elif sub_rules == "Position Sizing":
                try:
                    from position_sizing import render_position_sizing
                    render_position_sizing()
                except Exception as _ps_e:
                    st.error(f"Position Sizing kunde inte laddas: {_ps_e}")
            else:
                try:
                    from data_health import render_data_health
                    render_data_health()
                except Exception as _dh_e:
                    st.error(f"Data Health kunde inte laddas: {_dh_e}")

    # ── STRATEGIES ───────────────────────────────────────────────────────────
    if _is_open(tabs["strategies"]):
        with tabs["strategies"]:
            tab_strategy_overview()

    # ── COPILOT ──────────────────────────────────────────────────────────────
    if _is_open(tabs["copilot"]):
        with tabs["copilot"]:
            try:
                from tabs.copilot import render_copilot_page
            except ImportError as _cp_e:
                tab_not_found("AI Trading Copilot", f"tabs/copilot ({_cp_e})")
            else:
                render_copilot_page()

    # Senast sparad, och vad som ligger osparat i sessionen.
    storage_ui.footer()
    render_footer()
    _sync_url()


if __name__ == "__main__":
    main()
