"""
ui/links.py — "Öppna →" till ett mål i panelen (ui/nav.TARGETS), för knappar på startsidan och i larmloggen.

Används som on_click-callback: den körs före omritningen, så Thors widgetnycklar får sättas och
wolf_panel._apply_deep_link öppnar fliken (nav_goto).
"""
from __future__ import annotations

from ui import nav


def go(target: str, ticker: str = "", strategy: str = "", state=None) -> None:
    """Öppna målet. Thor får aktien och strategin ifyllda; övriga mål öppnar sidan."""
    if state is None:
        import streamlit as st
        state = st.session_state
    state["nav_goto"] = nav.target_slug(target)
    if target == "market_risk":                       # Läget visar Marknadsrisken på begäran — slå på den
        state["laget_risk"] = True
    if target == "thor" and ticker:
        state["thor_ticker"] = ticker
        from thor import registry as reg
        if reg.get(strategy):
            state["thor_strategy"] = reg.label(strategy)
            state["_thor_strategy_set"] = True
