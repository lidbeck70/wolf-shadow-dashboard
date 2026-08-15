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
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <meta name="theme-color" content="#05070A">
    <link rel="apple-touch-icon" href="https://em-content.zobj.net/source/apple/391/trident_1f531.png">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LOCAL MODULE IMPORTS
# ---------------------------------------------------------------------------
from ui.css import tab_not_found
from ui.theme import inject_css, render_header, render_footer
from auth import render_login_gate
from tabs.home import tab_home
from tabs.screener import tab_screener_consolidated, tab_screener, render_viking_screener
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

def main():
    if not render_login_gate():
        return

    inject_css()
    render_header()

    tab_labels = [
        "  🏠 HOME  ",
        "  🔱 SCREENING  ",
        "  📡 REGIME  ",
        "  👁 INTELLIGENCE  ",
        "  💼 PORTFOLIO  ",
        "  🔔 ALERTS  ",
        "  📋 RULES  ",
        "  🧬 STRATEGIES  ",
    ]
    (tab_home_page, tab_screening, tab_regime_main,
     tab_intelligence, tab_portfolio,
     tab_alerts_page, tab_rules,
     tab_strat_overview) = st.tabs(tab_labels)

    # ── HOME ─────────────────────────────────────────────────────────────────
    with tab_home_page:
        tab_home()

    # ── SCREENING ─────────────────────────────────────────────────────────────
    with tab_screening:
        sub = st.radio(
            "", ["Arc Screener", "Contrarian Alpha", "Market Cycle", "Swing",
                 "Swing Screener", "Tiggre", "Insider", "Poängmodell"],
            horizontal=True, key="sub_screening",
        )
        st.markdown("---")
        if sub == "Arc Screener":
            inner = st.radio(
                "", ["Wolf", "Viking", "🔥 EMBER"],
                horizontal=True, key="sub_screening_arc",
            )
            if inner == "Wolf":
                tab_screener()
            elif inner == "Viking":
                if OVTLYR_AVAILABLE:
                    render_viking_screener()
                else:
                    tab_not_found("Viking Screener", "screener_ovtlyr")
            elif inner == "🔥 EMBER":
                if EMBER_AVAILABLE:
                    render_ember_page()
                else:
                    tab_not_found("EMBER", "ember")

        elif sub == "Contrarian Alpha":
            inner = st.radio(
                "", ["Contrarian Alpha", "Long Screener"],
                horizontal=True, key="sub_screening_contrarian",
            )
            if inner == "Contrarian Alpha":
                if CONTRARIAN_ALPHA_AVAILABLE:
                    render_contrarian_alpha_page()
                else:
                    tab_not_found("Contrarian Alpha", "contrarian_alpha")
            elif inner == "Long Screener":
                if CAGR_AVAILABLE:
                    render_cagr_page()
                else:
                    tab_not_found("Long Screener", "cagr")

        elif sub == "Market Cycle":
            if MARKET_CYCLE_AVAILABLE:
                render_market_cycle_page()
            else:
                tab_not_found("Market Cycle Engine", "tabs/market_cycle")

        elif sub == "Swing":
            if SWING_AVAILABLE:
                render_swing_page()
            else:
                tab_not_found("Swing", "swing")

        elif sub == "Swing Screener":
            if WOLF_SCREENER_AVAILABLE:
                render_wolf_screener_page()
            else:
                tab_not_found("Swing Screener", "wolf_screener_ui")

        elif sub == "Tiggre":
            if TIGGRE_AVAILABLE:
                render_tiggre_page()
            else:
                tab_not_found("Tiggre", "tiggre")

        elif sub == "Insider":
            if INSIDER_AVAILABLE:
                render_insider_page()
            else:
                tab_not_found("Insiderbevakaren", "insider")

        elif sub == "Poängmodell":
            if SCORING_AVAILABLE:
                render_scoring_page()
            else:
                tab_not_found("Poängmodellen", "scoring")

    # ── REGIME ───────────────────────────────────────────────────────────────
    with tab_regime_main:
        sub = st.radio(
            "", ["Arc Regime", "Alpha Regime", "Flow Divergence", "Swing Regime",
                 "Råvarurotation"],
            horizontal=True, key="sub_regime",
        )
        st.markdown("---")
        if sub == "Råvarurotation":
            if ROTATION_AVAILABLE:
                render_rotation_page()
            else:
                tab_not_found("Råvarurotationen", "rotation")
        elif sub == "Swing Regime":
            if WOLF_REGIME_AVAILABLE:
                render_wolf_regime_page()
            else:
                tab_not_found("Swing Regime", "wolf_regime_ui")
        elif sub == "Arc Regime":
            inner = st.radio(
                "", ["Wolf Regime", "Viking Regime", "🌍 EMBER Regime"],
                horizontal=True, key="sub_regime_arc",
            )
            if inner == "Wolf Regime":
                tab_regime()
                try:
                    if render_inline_rules:
                        render_inline_rules("wolf")
                except Exception:
                    pass
            elif inner == "Viking Regime":
                if OVTLYR_AVAILABLE:
                    render_ovtlyr_page()
                else:
                    tab_not_found("OVTLYR", "ovtlyr")
                try:
                    if render_inline_rules:
                        render_inline_rules("viking")
                except Exception:
                    pass
            elif inner == "🌍 EMBER Regime":
                if EMBER_AVAILABLE:
                    render_ember_regime_page()
                else:
                    tab_not_found("EMBER Regime", "ember")

        elif sub == "Alpha Regime":
            inner = st.radio(
                "", ["Quality & Contrarian", "Long Trend"],
                horizontal=True, key="sub_regime_alpha",
            )
            if inner == "Quality & Contrarian":
                if ALPHA_REGIME_AVAILABLE:
                    render_alpha_regime()
                else:
                    tab_not_found("Alpha Regime Monitor", "alpha_regime")
                try:
                    if render_inline_rules:
                        render_inline_rules("alpha")
                except Exception:
                    pass
            elif inner == "Long Trend":
                if LONG_REGIME_AVAILABLE:
                    render_long_regime_monitor()
                else:
                    tab_not_found("Long Trend Monitor", "long_regime_monitor")

        elif sub == "Flow Divergence":
            if SECTOR_CYCLE_AVAILABLE:
                render_sector_cycle_page()
            else:
                tab_not_found("Sector & Global Regime", "sector_cycle")

    # ── INTELLIGENCE ─────────────────────────────────────────────────────────
    with tab_intelligence:
        sub = st.radio(
            "", ["Odin's Blindspot", "Sentiment", "Retail Pulse", "Heatmap"],
            horizontal=True, key="sub_intel",
        )
        st.markdown("---")
        if sub == "Odin's Blindspot":
            if BLINDSPOT_AVAILABLE:
                render_blindspot_page()
            else:
                tab_not_found("Odin's Blindspot Index", "blindspot")
        elif sub == "Sentiment":
            if SENTIMENT_AVAILABLE:
                render_sentiment_page()
            else:
                tab_not_found("Sentiment & Flow", "sentiment")
        elif sub == "Retail Pulse":
            if RETAIL_SENTIMENT_AVAILABLE:
                render_retail_sentiment_page()
            else:
                tab_not_found("Retail Sentiment", "retail_sentiment")
        elif sub == "Heatmap":
            if HEATMAP_AVAILABLE:
                render_heatmap_page()
            else:
                tab_not_found("Heatmap", "heatmap")

    # ── PORTFOLIO ─────────────────────────────────────────────────────────────
    with tab_portfolio:
        sub = st.radio(
            "", ["Holdings", "Allokering", "Trade Journal", "Backtest"],
            horizontal=True, key="sub_portfolio",
        )
        st.markdown("---")
        if sub == "Holdings":
            if HOLDINGS_AVAILABLE:
                render_holdings_page()
            else:
                tab_not_found("Holdings", "holdings")
        elif sub == "Allokering":
            if ALLOCATOR_AVAILABLE:
                render_allocator_page()
            else:
                tab_not_found("Portföljallokeraren", "allocator")
        elif sub == "Trade Journal":
            if JOURNAL_AVAILABLE:
                render_trade_journal_page()
            else:
                tab_not_found("Trade Journal", "trade_journal")
        elif sub == "Backtest":
            tab_backtest_consolidated()

    # ── ALERTS ───────────────────────────────────────────────────────────────
    with tab_alerts_page:
        tab_alerts()

    # ── RULES ────────────────────────────────────────────────────────────────
    with tab_rules:
        sub_rules = st.radio(
            "", ["Regler & Guider", "Position Sizing", "Data Health"],
            horizontal=True, key="sub_rules",
        )
        st.markdown("---")
        if sub_rules == "Regler & Guider":
            if RULES_AVAILABLE:
                render_rules_page()
            else:
                tab_not_found("Rules", "ovtlyr/ui")
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
    with tab_strat_overview:
        tab_strategy_overview()

    render_footer()


if __name__ == "__main__":
    main()
