import os
import base64
import hashlib
import hmac
import time
from typing import Optional

import streamlit as st

# ── Kom ihåg inloggningen ────────────────────────────────────────────────────
# Inloggningen bodde bara i session_state. När mobilen lägger fliken i bakgrunden
# bryts anslutningen, Streamlit startar en ny session och lösenordet krävdes igen.
# Nu får webbläsaren en signerad kaka vid inloggning. En ny session släpps in om
# kakan är giltig. Kakan innehåller bara utgångstid + HMAC, aldrig lösenordet.
# Byts lösenordet blir alla gamla kakor ogiltiga. Kakan förnyas vid varje ny
# session, så den löper ut först efter REMEMBER_DAYS utan besök.
#
# Bryggan: på Streamlit Cloud når kakan inte fram till appen (st.context.cookies är tom bakom Clouds proxy), så
# inloggningen glömdes ändå. Därför sparas samma signerade token också i webbläsarens localStorage, och
# inloggningssidan har ett litet skript som läser den i webbläsaren och laddar om sidan med ?wolf_t=<token>.
# Appen prövar token, släpper in och tar bort parametern ur adressen direkt. Ett försök per token och flik
# (sessionStorage), så en ogiltig eller utgången token ger aldrig en omladdningsslinga.
COOKIE_NAME = "wolf_auth"
QUERY_KEY = "wolf_t"
REMEMBER_DAYS = 30


def _password() -> str:
    try:
        return st.secrets.get("SWEWOLF_PASSWORD", "wolf2026")
    except Exception:
        return "wolf2026"


def _sign_key(password: str) -> bytes:
    return hashlib.sha256(("wolf-login-v1:" + password).encode("utf-8")).digest()


def make_token(password: str, now: Optional[float] = None, days: int = REMEMBER_DAYS) -> str:
    """'utgång.signatur' — utgången i Unix-sekunder, signaturen HMAC-SHA256 med en nyckel ur lösenordet."""
    exp = int((time.time() if now is None else now) + days * 86400)
    sig = hmac.new(_sign_key(password), str(exp).encode("ascii"), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def token_valid(token, password: str, now: Optional[float] = None) -> bool:
    """Sant om kakan är signerad med det här lösenordet och inte har gått ut. Allt annat (tomt, trasigt,
    fel lösenord, utgånget) är falskt, aldrig ett fel."""
    try:
        exp_s, sig = str(token or "").split(".", 1)
        exp = int(exp_s)
    except (ValueError, TypeError):
        return False
    if exp < (time.time() if now is None else now):
        return False
    want = hmac.new(_sign_key(password), exp_s.encode("ascii"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, want)


def _cookie_token() -> str:
    try:
        return str(st.context.cookies.get(COOKIE_NAME) or "")
    except Exception:
        return ""


def _query_token() -> str:
    try:
        return str(st.query_params.get(QUERY_KEY) or "")
    except Exception:
        return ""


def _drop_query_token() -> None:
    """Token ska aldrig ligga kvar i adressraden (historik, en delad länk)."""
    try:
        if QUERY_KEY in st.query_params:
            del st.query_params[QUERY_KEY]
    except Exception:
        pass


def _set_cookie(token: str) -> None:
    """Skriver kakan och localStorage i webbläsaren (JavaScript i sidan; Streamlit kan bara läsa kakor). Secure på
    https. localStorage är det bryggan läser när kakan inte når fram till appen."""
    js = (
        "<script>(function(){var s=location.protocol==='https:'?'; Secure':'';"
        f"document.cookie='{COOKIE_NAME}={token}; Max-Age={REMEMBER_DAYS * 86400}; Path=/; SameSite=Lax'+s;"
        f"try{{localStorage.setItem('{COOKIE_NAME}','{token}');}}catch(e){{}}"
        "})();</script>"
    )
    try:
        st.html(js, unsafe_allow_javascript=True)
    except Exception:
        pass


# Körs på inloggningssidan. Sparad token (localStorage, annars kakan) → ladda om sidan med den i adressen, en gång
# per token och flik. Ingen token, fel form eller redan provad: ingenting händer och inloggningsrutan står kvar.
_BRIDGE_JS = (
    "<script>(function(){try{"
    "var k='" + COOKIE_NAME + "',q='" + QUERY_KEY + "';"
    "var t=localStorage.getItem(k)||'';"
    "if(!t){var m=document.cookie.match(new RegExp('(?:^|; )'+k+'=([^;]+)'));t=m?m[1]:'';}"
    "if(!/^[0-9]{1,12}[.][0-9a-f]{64}$/.test(t))return;"
    "if(sessionStorage.getItem('wolf_bridge')===t)return;"
    "sessionStorage.setItem('wolf_bridge',t);"
    "var u=new URL(location.href);u.searchParams.set(q,t);location.replace(u.toString());"
    "}catch(e){}})();</script>"
)


def _bridge() -> None:
    try:
        st.html(_BRIDGE_JS, unsafe_allow_javascript=True)
    except Exception:
        pass


def render_login_gate() -> bool:
    """Render the login screen. Returns True if authenticated, False otherwise."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated and token_valid(_cookie_token(), _password()):
        st.session_state.authenticated = True             # sparad inloggning från en tidigare session
        st.session_state["_auth_cookie_pending"] = True    # förnya kakan, så att 30 dagar räknas från nu

    qtok = _query_token()
    if qtok:
        if not st.session_state.authenticated and token_valid(qtok, _password()):
            st.session_state.authenticated = True          # bryggan: token ur webbläsarens localStorage
            st.session_state["_auth_cookie_pending"] = True
        _drop_query_token()                                # aldrig kvar i adressraden, giltig eller inte

    if st.session_state.authenticated:
        if st.session_state.pop("_auth_cookie_pending", False):
            _set_cookie(make_token(_password()))
        return True

    # Arc Systems login page styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700;900&family=Inter+Tight:wght@400;500&display=swap');
    .stApp { background-color: #05070A !important; }
    .arc-login-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 0 20px 0;
    }
    .arc-login-wrapper img {
        max-width: 340px;
        width: 100%;
        border-radius: 16px;
        filter: drop-shadow(0 0 32px rgba(0,229,255,0.3));
    }
    .arc-login-title {
        font-family: 'Space Grotesk', monospace;
        font-size: 1.6rem;
        font-weight: 900;
        letter-spacing: 0.2em;
        color: #E8EDF2;
        text-align: center;
        margin-top: 20px;
    }
    .arc-login-title span { color: #00E5FF; text-shadow: 0 0 12px rgba(0,229,255,0.6); }
    .arc-login-slogan {
        font-family: 'Inter Tight', sans-serif;
        font-size: 0.85rem;
        font-style: italic;
        color: #6B7280;
        text-align: center;
        margin-top: 6px;
        letter-spacing: 0.05em;
    }
    </style>
    """, unsafe_allow_html=True)

    _logo_col1, _logo_col2, _logo_col3 = st.columns([1, 2, 1])
    with _logo_col2:
        try:
            # Try new Arc logo first, then fall back to old logo
            for _logo_name in ["logo_arc.png", "logo_arc.jpg", "logo.jpeg"]:
                _logo_path = os.path.join(os.path.dirname(__file__), "assets", _logo_name)
                if os.path.exists(_logo_path):
                    _ext = "png" if _logo_name.endswith(".png") else "jpeg"
                    with open(_logo_path, "rb") as _lf:
                        _logo_b64 = base64.b64encode(_lf.read()).decode()
                    st.markdown(
                        f"<div class='arc-login-wrapper'>"
                        f"<img src='data:image/{_ext};base64,{_logo_b64}'/>"
                        f"<div class='arc-login-title'>NORDIC <span>ARC</span> SYSTEMS</div>"
                        f"<div class='arc-login-slogan'>See What the Market Can't.</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    break
            else:
                st.markdown(
                    "<div class='arc-login-wrapper'>"
                    "<div class='arc-login-title'>NORDIC <span>ARC</span> SYSTEMS</div>"
                    "<div class='arc-login-slogan'>See What the Market Can't.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                "<div class='arc-login-wrapper'>"
                "<div class='arc-login-title'>NORDIC <span>ARC</span> SYSTEMS</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    col_l, col_m, col_r = st.columns([1, 1, 1])
    with col_m:
        password = st.text_input("Access code", type="password", key="login_pw", label_visibility="collapsed",
                                 placeholder="Enter access code...")
        if st.button("ACCESS SYSTEM", use_container_width=True, key="login_btn"):
            if password == _password():
                st.session_state.authenticated = True
                st.session_state["_auth_cookie_pending"] = True   # kakan skrivs i nästa körning
                st.rerun()
            else:
                st.error("Access denied")

    _bridge()
    return False
