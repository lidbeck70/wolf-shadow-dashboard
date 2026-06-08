import os
import base64
import streamlit as st


def render_login_gate() -> bool:
    """Render the login screen. Returns True if authenticated, False otherwise."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
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
        password = st.text_input("", type="password", key="login_pw",
                                  placeholder="Enter access code...")
        if st.button("ACCESS SYSTEM", use_container_width=True, key="login_btn"):
            try:
                correct_pw = st.secrets.get("SWEWOLF_PASSWORD", "wolf2026")
            except Exception:
                correct_pw = "wolf2026"

            if password == correct_pw:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Access denied")

    return False
