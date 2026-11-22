import os
import base64
import functools
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
        if exp < (time.time() if now is None else now):
            return False
        want = hmac.new(_sign_key(password), exp_s.encode("ascii"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig.encode("utf-8"), want.encode("ascii"))
    except (ValueError, TypeError, UnicodeError):   # även en påhittad ?wolf_t= med andra tecken: falskt, aldrig fel
        return False


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


# ── Inloggningssidan (LIDSKJALV §4.1) ────────────────────────────────────────
# Skalets stilblock (ui/shell.py) ritas före inloggningen: panelens bakgrund, typsnitten, fältet (52 px, 16 px —
# ingen iOS-zoom), knappen, kortet på datorn och det dolda huvudet kommer därifrån, valda på nycklarna nedan.
# Här står bara sidans egna klasser (.arc-login-*): loggan, devisen och raden om kakan.
LOGIN_KEY = "login_card"          # kolumnen — ett kort på datorn
WRONG_CODE = "Fel kod — försök igen."
TAGLINE = "See What the Market Can't."

# Loggan beskärs till en kvadrat runt triangeln och ordmärket (logo_arc är 1536 × 1024 med svart runt om):
# bildens kvadrat x 390–1110, y 89–809 → bredd 1536/720, vänster −390/720, topp −89/720 av rutan.
LOGIN_CSS = """
.arc-login-logo { position: relative; width: 148px; height: 148px; margin: 0 auto; overflow: hidden;
    border-radius: 16px; background: #000; filter: drop-shadow(0 0 32px rgba(0,229,255,.3));
    animation: arc-login-breathe 4s ease-in-out infinite; }
.arc-login-logo img { display: block; width: 100%; height: 100%; max-width: none; object-fit: cover; }
.arc-login-logo img.arc-login-crop { position: absolute; width: 213.33%; height: auto; left: -54.17%;
    top: -12.36%; object-fit: fill; }
@keyframes arc-login-breathe {
    0%, 100% { filter: drop-shadow(0 0 22px rgba(0,229,255,.22)); }
    50% { filter: drop-shadow(0 0 40px rgba(0,229,255,.42)); }
}
.arc-login-word { font: 700 20px/1.2 'Space Grotesk', sans-serif; letter-spacing: .2em; color: #e8e4dc;
    text-align: center; padding: 24px 0 0; }
.arc-login-word b { font-weight: 700; color: #00E5FF; text-shadow: 0 0 12px rgba(0,229,255,.6); }
.arc-login-tag { font: italic 400 14px/20px 'Space Grotesk', sans-serif; color: #a8a397; text-align: center;
    margin: 12px 0 0; }
.arc-login-note { font: 400 12px/1.4 'Space Grotesk', sans-serif; color: #a8a397; text-align: center; }
@media (min-width: 641px) {
    .arc-login-logo { width: 180px; height: 180px; }
}
@media (prefers-reduced-motion: reduce) {
    .arc-login-logo { animation: none; }
}
"""

# (fil, bildtyp, beskär) — Arc-loggan först, den gamla som reserv
_LOGOS = (("logo_arc.png", "png", True), ("logo_arc.jpg", "jpeg", True), ("logo.jpeg", "jpeg", False))


@functools.lru_cache(maxsize=1)
def _logo_img() -> str:
    """Loggan som <img> med base64 (läses en gång per process). Ingen fil: tom sträng."""
    for name, kind, crop in _LOGOS:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", name)
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
        except OSError:
            continue
        cls = " class='arc-login-crop'" if crop else ""
        return f"<img{cls} alt='Nordic Arc Systems' src='data:image/{kind};base64,{b64}'/>"
    return ""


def _brand_html() -> str:
    """Stilen, loggan och devisen i EN markdown. Konsten bär redan NORDIC ARC SYSTEMS — ordmärket i text står bara
    där ingen bild finns."""
    img = _logo_img()
    mark = (f"<div class='arc-login-logo'>{img}</div>" if img
            else "<div class='arc-login-word'>NORDIC <b>ARC</b> SYSTEMS</div>")
    return f"<style>{LOGIN_CSS}</style>{mark}<div class='arc-login-tag'>{TAGLINE}</div>"


def _login_form() -> None:
    """Loggan, formuläret och raden om kakan i en kolumn. Ett formulär: tangentbordets "Gå" (Enter) loggar in,
    och sidan körs inte om för varje tecken."""
    with st.container(key=LOGIN_KEY):
        st.markdown(_brand_html(), unsafe_allow_html=True)
        with st.form("login_form", border=False, enter_to_submit=True):
            password = st.text_input("Åtkomstkod", type="password", key="login_pw", placeholder="Ange åtkomstkod",
                                     autocomplete="current-password")
            sent = st.form_submit_button("LOGGA IN", key="login_btn", type="primary", width="stretch")
        if sent:
            if password == _password():
                st.session_state.authenticated = True
                st.session_state["_auth_cookie_pending"] = True   # kakan skrivs i nästa körning
                st.rerun()
            else:
                st.error(WRONG_CODE)
        st.markdown(f"<div class='arc-login-note'>Inloggningen sparas i {REMEMBER_DAYS} dagar på den här enheten."
                    f"</div>", unsafe_allow_html=True)


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

    _login_form()
    _bridge()
    return False
