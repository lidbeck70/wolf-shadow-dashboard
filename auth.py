import os
import base64
import streamlit as st


def render_login_gate() -> bool:
    """Render the login screen. Returns True if authenticated, False otherwise."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    _logo_col1, _logo_col2, _logo_col3 = st.columns([1, 2, 1])
    with _logo_col2:
        try:
            _logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.jpeg")
            if os.path.exists(_logo_path):
                with open(_logo_path, "rb") as _lf:
                    _logo_b64 = base64.b64encode(_lf.read()).decode()
                st.markdown(
                    f"<div style='text-align:center;padding:30px 0 10px 0;'>"
                    f"<img src='data:image/jpeg;base64,{_logo_b64}' "
                    f"style='max-width:380px;width:100%;border-radius:12px;'/>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='text-align:center;padding:40px 0;'>"
                    "<h1 style='color:#c9a84c;letter-spacing:0.15em;'>NORDIC ALPHA SYSTEMS</h1>"
                    "<p style='color:#c9a84c;'>Born of Wolves, Made for Markets</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                "<div style='text-align:center;padding:40px 0;'>"
                "<h1 style='color:#c9a84c;letter-spacing:0.15em;'>NORDIC ALPHA SYSTEMS</h1>"
                "</div>",
                unsafe_allow_html=True,
            )

    col_l, col_m, col_r = st.columns([1, 1, 1])
    with col_m:
        password = st.text_input("Lösenord", type="password", key="login_pw")
        if st.button("LOGGA IN", use_container_width=True, key="login_btn"):
            try:
                correct_pw = st.secrets.get("SWEWOLF_PASSWORD", "wolf2026")
            except Exception:
                correct_pw = "wolf2026"

            if password == correct_pw:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Fel lösenord")

    return False
