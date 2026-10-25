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
# ── LEGACY (designförslaget PR 12, beslut 11) ───────────────────────────────
# Importeras men ritas inte i något flikträd — Thor, Läget och Motorsidorna gör jobbet: tabs.screener
# (tab_screener, render_viking_screener), tabs.strategy_overview (STRATEGIES), contrarian_alpha.ui, cagr,
# ember.ui, berserk.screen_ui (skannersidan), ovtlyr.ui.viking_screens (render_viking_nine_page),
# tech_analysis och wolf_screener_ui. Inget raderas förrän ägaren säger till (rekommendation: om tre månader).
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

# Durrett — Don Durretts 10-stegsmetod (engines/durrett) med eget ark (data/confidence.json).
# Confidence-caset (confidence/) ritas inbyggt i Wolf Asymmetry sedan steg F.
try:
    from engines.durrett.ui import render_durrett_page
    DURRETT_AVAILABLE = True
except ImportError:
    DURRETT_AVAILABLE = False

# Wolf Asymmetry — hävstång, margin of safety, stressmatris (asymmetry/), samma ark som Durrett
try:
    from asymmetry.ui import render_asymmetry_page
    ASYMMETRY_AVAILABLE = True
except ImportError:
    ASYMMETRY_AVAILABLE = False

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

# Råvaruhävstång (GRANSKNING) — resultat- och kurshävstång mot råvaran
try:
    from commodity_leverage.ui import render_commodity_leverage_page
    COMMODITY_LEVERAGE_AVAILABLE = True
except ImportError:
    COMMODITY_LEVERAGE_AVAILABLE = False

# Guld/silver-kvoten (ODIN → Råvaror)
try:
    from gold_silver.ui import render_gold_silver_page
    GOLD_SILVER_AVAILABLE = True
except ImportError:
    GOLD_SILVER_AVAILABLE = False

# Viking Nine (THOR) — OVTLYR Nine + Viking Execution över en lista
try:
    from ovtlyr.ui.viking_screens import render_viking_nine_page
    VIKING_NINE_AVAILABLE = True
except ImportError:
    VIKING_NINE_AVAILABLE = False

# Viking Nine Regime (MER → Motorsidor) — beslutet för en aktie i Viking Nine
try:
    from ovtlyr.ui.viking_nine_regime import render_viking_nine_regime_page
    VIKING_NINE_REGIME_AVAILABLE = True
except ImportError:
    VIKING_NINE_REGIME_AVAILABLE = False

# ⚡ VALHALL — cykelkartan över hela marknaden (läser de schemalagda körningarna ur Gisten)
try:
    from valhall.ui import render_valhall_page
    VALHALL_AVAILABLE = True
except Exception:
    render_valhall_page = None
    VALHALL_AVAILABLE = False

# 🔨 THOR — alla strategiers kandidater i ett flöde och analysen per aktie (läser körningarna ur Gisten)
try:
    from thor.ui import render_thor_page
    THOR_AVAILABLE = True
except Exception:
    render_thor_page = None
    THOR_AVAILABLE = False

# 🪓 BERSERK (THOR, ODIN → Råvaror) — contrarian swing i råvarumarknaden
try:
    from berserk.screen_ui import render_berserk_screen_page
    from berserk.regime_ui import render_berserk_regime_page
    BERSERK_AVAILABLE = True
except ImportError:
    BERSERK_AVAILABLE = False

# Marknadsrisk (ODIN → Marknad) — riskmodellen för SPY och OMXS30
try:
    from market_risk_ui import render_market_risk_page
    MARKET_RISK_AVAILABLE = True
except ImportError:
    MARKET_RISK_AVAILABLE = False

# Teknisk analys (INTELLIGENCE) — pris, order blocks, mönster, risk för valfri ticker
try:
    from tech_analysis import render_tech_analysis_page
    TECH_ANALYSIS_AVAILABLE = True
except ImportError:
    TECH_ANALYSIS_AVAILABLE = False

# Guldkvoter (ODIN → Råvaror) — råvaror och index mätta i guld
try:
    from gold_ratios.ui import render_gold_ratios_page
    GOLD_RATIOS_AVAILABLE = True
except ImportError:
    GOLD_RATIOS_AVAILABLE = False

# 🐺 Fiat Debasement (ODIN → Makro) — penningmängd, inflation, köpkraft och fiat mot guld
try:
    from fiat_debasement.ui import render_fiat_debasement_page
    FIAT_AVAILABLE = True
except ImportError:
    FIAT_AVAILABLE = False

try:
    from fiat_debasement.environment import render_badge as render_fiat_environment
    FIAT_ENV_AVAILABLE = True
except ImportError:
    FIAT_ENV_AVAILABLE = False

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
    """?p=odin/marknad/arc-regime/wolf-regime i adressfältet (gamla regime/…-länkar fungerar), eller
    nav_goto från ett Home-kort, väljer flik och underflikar INNAN
    widgetarna ritas. Bara en gång per mål — annars skulle varje omritning
    dra användaren tillbaka till länken när hen klickat vidare."""
    target = st.session_state.pop("nav_goto", None) or st.query_params.get("p") or ""
    if not target or target == st.session_state.get("_nav_applied"):
        # Streamlit släpper toppflikens värde ur sessionen efter två körningar när det sattes av en länk och inte
        # av ett klick — återställ det ur senaste sökvägen, annars hoppar panelen till I DAG (granskning 2, D1).
        applied = st.session_state.get("_nav_applied")
        if applied and nav.TABS_KEY not in st.session_state:
            segs = nav.resolve(applied)
            if segs:
                st.session_state[nav.TABS_KEY] = nav.tab_label(segs[0])
        return
    st.session_state["_nav_applied"] = target
    segs = nav.resolve(target)
    if segs:
        nav.seed(st.session_state, segs)
        inner = nav.inner_seed(target)                     # gamla Guldkvoter/Retail-länkar öppnar sin inre flik
        if inner:
            st.session_state[inner[0]] = inner[1]
    if str(st.query_params.get("risk") or "") == "1":      # Marknadsrisk-larmets länk visar risken på Läget
        st.session_state["laget_risk"] = True


def _sync_url() -> None:
    """Adressfältet följer fliken, så URL:en alltid går att kopiera som
    djuplänk. Ingen omritning — bara URL:en byts."""
    segs = nav.current(st.session_state)
    slug = nav.slug(segs)
    st.session_state["_nav_applied"] = slug
    if st.query_params.get("p") != slug:
        st.query_params["p"] = slug
    if segs[0] != "thor":                                   # aktien i en länk hör bara till Thor (D5)
        for k in ("t", "s"):
            if k in st.query_params:
                del st.query_params[k]
    if segs[:2] != ["odin", "Läget"] and "risk" in st.query_params:
        del st.query_params["risk"]


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


def _inner(labels: list, key: str) -> str:
    """Inre flikar på en sida (designförslaget PR 10) — bara den öppna ritas, som toppflikarna."""
    tabs = st.tabs(labels, on_change="rerun", key=key)
    for label, tab in zip(labels, tabs):
        if getattr(tab, "open", None):
            return label
    return labels[0]


def _background_note() -> None:
    st.markdown("<div style='color:#8a8578;font-size:0.76rem;border-left:2px solid #8a8578;padding-left:8px;"
                "margin:4px 0 10px;'>Bakgrund — styr inga beslut. Läget och strategierna avgör.</div>",
                unsafe_allow_html=True)


def _tab_odin() -> None:
    group = _sub("odin")
    st.markdown("---")
    if group == "Läget":
        from odin.laget import render_laget_page
        _page(True, render_laget_page, "Läget", "odin/laget")
    elif group == "Kartan":
        if VALHALL_AVAILABLE:
            render_valhall_page("Kartan")
        else:
            tab_not_found("Valhall", "valhall")
    elif group == "Sektorrotation":
        from odin.sektorrotation import render_sektorrotation_page
        _page(True, render_sektorrotation_page, "Sektorrotation", "odin/sektorrotation")
    elif group == "Marknad":
        sub = _sub("odin/Marknad")
        st.markdown("---")
        _background_note()
        if sub == "Sektorflöde":
            _page(SECTOR_CYCLE_AVAILABLE, render_sector_cycle_page, "Sector & Global Regime", "sector_cycle")
        elif sub == "Värmekarta":
            _page(HEATMAP_AVAILABLE, render_heatmap_page, "Heatmap", "heatmap")
    elif group == "Makro":
        sub = _sub("odin/Makro")
        st.markdown("---")
        if sub == "🐺 Fiat Debasement":
            _page(FIAT_AVAILABLE, render_fiat_debasement_page, "Fiat Debasement", "fiat_debasement")
        elif sub == "Sentiment":
            if _inner(["Fear & Greed", "Retail"], "odin_sentiment_tabs") == "Retail":
                _page(RETAIL_SENTIMENT_AVAILABLE, render_retail_sentiment_page, "Retail Sentiment", "retail_sentiment")
            else:
                _page(SENTIMENT_AVAILABLE, render_sentiment_page, "Sentiment & Flow", "sentiment")
    else:
        sub = _sub("odin/Råvaror")
        st.markdown("---")
        if sub == "Odin's Blindspot":
            _page(BLINDSPOT_AVAILABLE, render_blindspot_page, "Odin's Blindspot Index", "blindspot")
        elif sub == "🔥 EMBER":
            _page(EMBER_AVAILABLE, render_ember_regime_page, "EMBER Regime", "ember")
        elif sub == "🪓 BERSERK Regime":
            _page(BERSERK_AVAILABLE, render_berserk_regime_page, "BERSERK Regime", "berserk")
        elif sub == "Råvarurotation":
            _page(ROTATION_AVAILABLE, render_rotation_page, "Råvarurotationen", "rotation")
        elif sub == "🥇 Guld":
            if _inner(["🥇🥈 Guld/Silver", "🥇 Guldkvoter"], "odin_gold_tabs") == "🥇 Guldkvoter":
                _page(GOLD_RATIOS_AVAILABLE, render_gold_ratios_page, "Guldkvoter", "gold_ratios")
            else:
                _page(GOLD_SILVER_AVAILABLE, render_gold_silver_page, "Guld/Silver", "gold_silver")


def _tab_review() -> None:
    sub = _sub("review")
    st.markdown("---")
    if sub in ("Rick Rule", "Royalty C", "🐺 Wolf Asymmetry", "🚀 Råvaruhävstång") and FIAT_ENV_AVAILABLE:
        render_fiat_environment()             # bakgrund, ingen signal
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
        _page(INSIDER_AVAILABLE, render_insider_page, "Insider", "insider")
    elif sub == "🧭 Durrett":
        _page(DURRETT_AVAILABLE, render_durrett_page, "Durrett", "engines/durrett/ui")
    elif sub == "🐺 Wolf Asymmetry":
        _page(ASYMMETRY_AVAILABLE, render_asymmetry_page, "Wolf Asymmetry", "asymmetry/ui")
    elif sub == "🚀 Råvaruhävstång":
        _page(COMMODITY_LEVERAGE_AVAILABLE, render_commodity_leverage_page, "Råvaruhävstång",
              "commodity_leverage/ui")
    elif sub == "🎯 Scorecard":
        _page(SCORECARD_AVAILABLE, render_scorecard_page, "Scorecard", "scorecard")


def _tab_portfolio() -> None:
    sub = _sub("portfolio")
    st.markdown("---")
    if sub == "Journal":
        _page(JOURNAL_AVAILABLE, render_trade_journal_page, "Trade Journal", "trade_journal")
    elif sub == "🎯 Mot målet":
        from portfolio.mot_malet import render_mot_malet_page
        _page(True, render_mot_malet_page, "Mot målet", "portfolio/mot_malet")
    elif sub == "📊 Papperskonton":
        from valhall.paper_ui import render_paper_page
        _page(True, render_paper_page, "Papperskonton", "valhall/paper_ui")
    elif sub == "Positioner":
        _page(HOLDINGS_AVAILABLE, render_holdings_page, "Holdings", "holdings")
    elif sub == "Veckorutin":
        _page(SWING_AVAILABLE, render_swing_page, "Swing", "swing")
    elif sub == "Allokering":
        _page(ALLOCATOR_AVAILABLE, render_allocator_page, "Portföljallokeraren", "allocator")


def _tab_more() -> None:
    sub = _sub("more")
    st.markdown("---")
    if sub == "Larm":
        tab_alerts()
    elif sub == "⚙️ Inställningar":
        from ui.settings import render_settings_page
        _page(True, render_settings_page, "Inställningar", "ui/settings")
    elif sub == "Regler & Guider":
        _page(RULES_AVAILABLE, render_rules_page, "Rules", "ovtlyr/ui")
    elif sub == "Copilot":
        try:
            from tabs.copilot import render_copilot_page
        except ImportError as _cp_e:
            tab_not_found("AI Trading Copilot", f"tabs/copilot ({_cp_e})")
        else:
            render_copilot_page()
    elif sub == "Motorsidor":
        inner = _sub("more/Motorsidor")
        st.markdown("---")
        if inner == "Wolf Regime":
            _page(True, tab_regime, "Wolf Regime", "tabs/regime", rules_key="wolf")
        elif inner == "Viking Regime":
            _page(OVTLYR_AVAILABLE, render_ovtlyr_page, "OVTLYR", "ovtlyr", rules_key="viking")
        elif inner == "⚔️ Viking Nine Regime":
            _page(VIKING_NINE_REGIME_AVAILABLE, render_viking_nine_regime_page, "Viking Nine Regime",
                  "viking_nine_regime")
        elif inner == "Quality & Contrarian":
            _page(ALPHA_REGIME_AVAILABLE, render_alpha_regime, "Alpha Regime Monitor", "alpha_regime",
                  rules_key="alpha")
        elif inner == "Long Trend":
            _page(LONG_REGIME_AVAILABLE, render_long_regime_monitor, "Long Trend Monitor", "long_regime_monitor")
        elif inner == "Market Cycle":
            _page(MARKET_CYCLE_AVAILABLE, render_market_cycle_page, "Market Cycle Engine", "tabs/market_cycle")
    elif sub == "Egen backtest":
        tab_backtest_consolidated()


def main():
    if not render_login_gate():
        return

    inject_css()
    render_header()
    _apply_deep_link()

    tab_labels = [nav.tab_label(k) for k, _label in nav.TOP]
    tabs = dict(zip([k for k, _l in nav.TOP],
                    st.tabs(tab_labels, on_change="rerun", key="main_tabs")))

    # ── 🏠 I DAG ──────────────────────────────────────────────────────────────
    if _is_open(tabs["home"]):
        with tabs["home"]:
            tab_home()

    # ── 👁 ODIN — var kapitalet ska ligga ───────────────────────────────────
    if _is_open(tabs["odin"]):
        with tabs["odin"]:
            _tab_odin()

    # ── 🔨 THOR ───────────────────────────────────────────────────────────────
    if _is_open(tabs["thor"]):
        with tabs["thor"]:
            if THOR_AVAILABLE:
                render_thor_page()
            else:
                tab_not_found("Thor", "thor")

    # ── 📒 ARKEN ──────────────────────────────────────────────────────────────
    # Beslutsunderlaget efter screeningen. Inget här screenar — screeningen
    # sker i Thor och Börsdata; det här är arken som avgör vilket bolag som köps.
    if _is_open(tabs["review"]):
        with tabs["review"]:
            _tab_review()

    # ── 💼 PORTFÖLJ ───────────────────────────────────────────────────────────
    if _is_open(tabs["portfolio"]):
        with tabs["portfolio"]:
            _tab_portfolio()

    # ── ⋯ MER ─────────────────────────────────────────────────────────────────
    if _is_open(tabs["more"]):
        with tabs["more"]:
            _tab_more()

    # Senast sparad, och vad som ligger osparat i sessionen.
    storage_ui.footer()
    render_footer()
    _sync_url()


if __name__ == "__main__":
    main()
