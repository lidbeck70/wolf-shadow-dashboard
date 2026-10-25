"""
portfolio/private_lock.py — lösenord på de egna innehaven (PORTFÖLJ → Positioner, fysiska metaller och
startsidans säljrader).

Lösenordet ligger i Streamlit Secrets som HOLDINGS_PASSWORD och visas aldrig. Saknas det är innehaven låsta
tills det är satt — ett lösenord i koden skulle synas i det publika repot. Efter fem fel i rad väntar grinden en
minut. Upplåsningen gäller sessionen; "🔒 Lås" stänger igen.

Grinden skyddar vyn. Uppgifterna i repot skyddas av krypteringen (private_data.py).
"""
from __future__ import annotations

import hmac
import time
from typing import Optional

SECRET = "HOLDINGS_PASSWORD"
STATE, FAILS, UNTIL = "holdings_unlocked", "holdings_fails", "holdings_block_until"
MAX_FAILS, BLOCK_S = 5, 60


def _secret() -> Optional[str]:
    try:
        import streamlit as st
        v = str(st.secrets.get(SECRET, "") or "")
    except Exception:
        v = ""
    return v or None


def configured() -> bool:
    return _secret() is not None


def unlocked(state=None) -> bool:
    if state is None:
        import streamlit as st
        state = st.session_state
    return bool(state.get(STATE))


def check(attempt: str, state, secret: Optional[str], now: Optional[float] = None) -> str:
    """'ok' | 'fel' | 'spärrad' | 'saknas' — och uppdaterar sessionen."""
    now = time.time() if now is None else now
    if not secret:
        return "saknas"
    if now < float(state.get(UNTIL) or 0):
        return "spärrad"
    if hmac.compare_digest(str(attempt or "").encode("utf-8"), secret.encode("utf-8")):
        state[STATE], state[FAILS] = True, 0
        return "ok"
    state[FAILS] = int(state.get(FAILS) or 0) + 1
    if state[FAILS] >= MAX_FAILS:
        state[UNTIL], state[FAILS] = now + BLOCK_S, 0
        return "spärrad"
    return "fel"


def lock(state) -> None:
    state[STATE] = False


def gate(key: str = "holdings_pw") -> bool:
    """Ritar låset. True när innehaven får visas."""
    import streamlit as st

    from ui.components import note
    if unlocked():
        if st.button("🔒 Lås innehaven", key=f"{key}_lock"):
            lock(st.session_state)
            st.rerun()
        return True
    st.markdown("#### 🔒 Dina innehav är låsta")
    if not configured():
        note(f"Lägg in {SECRET} i Streamlit (Settings → Secrets) för att kunna låsa upp. Innehaven visas inte förrän "
             f"lösenordet är satt.")
        return False
    pw = st.text_input("Lösenord för innehaven", type="password", key=key)
    if st.button("Lås upp", key=f"{key}_btn"):
        res = check(pw, st.session_state, _secret())
        if res == "ok":
            st.rerun()
        elif res == "spärrad":
            st.error(f"För många försök — vänta {BLOCK_S} sekunder.")
        else:
            st.error("Fel lösenord.")
    return False
