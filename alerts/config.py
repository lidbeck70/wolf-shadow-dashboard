"""
alerts/config.py — en nyckel, två källor.

Kanalerna läste bara os.environ. Det fungerar i GitHub Actions (workflowen
sätter miljön) men är skört på Streamlit Cloud: där ligger nycklarna i
st.secrets, och att de OCKSÅ dyker upp som miljövariabler är en bieffekt
som kräver att någon råkat röra st.secrets först — panelens testknapp
felade på exakt det. Nu frågar varje kanal secret() i stället: Streamlit-
secrets först (rotnivån), sedan miljön. Headless (Actions, tester) finns
ingen secrets-fil — då blir det miljövägen, precis som förut.
"""

from __future__ import annotations

import os


def secret(name: str, default: str = "") -> str:
    """Värdet för *name* ur st.secrets (rotnivån) eller miljön, rensat."""
    try:
        import streamlit as st
        value = st.secrets.get(name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(name, default) or default).strip()


def source(name: str) -> str:
    """Var *name* hittades: "secrets", "env" eller "" — för statuskorten,
    som ska kunna säga VAR nyckeln bor utan att visa värdet."""
    try:
        import streamlit as st
        value = st.secrets.get(name, None)
        if value is not None and str(value).strip():
            return "secrets"
    except Exception:
        pass
    return "env" if os.environ.get(name, "").strip() else ""
