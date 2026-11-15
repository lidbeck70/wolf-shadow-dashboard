"""Kom-ihåg-inloggningen: signerad kaka som överlever en ny Streamlit-session."""
import pytest
from streamlit.testing.v1 import AppTest

import auth


def test_token_roundtrip():
    tok = auth.make_token("hemligt", now=1_000_000)
    assert auth.token_valid(tok, "hemligt", now=1_000_000 + 29 * 86400)


def test_token_expires():
    tok = auth.make_token("hemligt", now=1_000_000)
    assert not auth.token_valid(tok, "hemligt", now=1_000_000 + 31 * 86400)


def test_token_wrong_password_rejected():
    tok = auth.make_token("hemligt", now=1_000_000)
    assert not auth.token_valid(tok, "annat", now=1_000_000)


def test_token_tampered_rejected():
    exp, sig = auth.make_token("hemligt", now=1_000_000).split(".")
    assert not auth.token_valid(f"{int(exp) + 10**8}.{sig}", "hemligt", now=1_000_000)   # förlängd utgång
    assert not auth.token_valid(f"{exp}.{'0' * len(sig)}", "hemligt", now=1_000_000)


def test_token_garbage_is_false_not_error():
    for bad in (None, "", "x", "abc.def", ".", "123", "1.2.3", "9999999999.é", "9999999999." + "é" * 64,
                "١٢٣٤٥٦٧٨٩٩.abc"):
        assert auth.token_valid(bad, "hemligt") is False


def test_token_holds_no_password():
    assert "hemligt" not in auth.make_token("hemligt")


# ── Bryggan: token ur webbläsarens localStorage via ?wolf_t= (Streamlit Cloud skickar inte kakan till appen) ──
_GATE = """
import auth
import streamlit as st
if auth.render_login_gate():
    st.markdown("INNE")
"""


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(auth, "_password", lambda: "hemligt")
    monkeypatch.setattr(auth, "_cookie_token", lambda: "")       # som på Cloud: ingen kaka når appen
    return AppTest.from_string(_GATE, default_timeout=30)


def _inne(at):
    return any(m.value == "INNE" for m in at.markdown)


def _bridge_on_page(at):
    return any("wolf_bridge" in str(getattr(h, "proto", "")) for h in at.get("html"))


def test_bridge_valid_token_logs_in_and_leaves_address(gate):
    gate.query_params[auth.QUERY_KEY] = auth.make_token("hemligt")
    gate.query_params["p"] = "holdings"
    at = gate.run()
    assert not at.exception
    assert _inne(at) and at.session_state["authenticated"] is True
    assert auth.QUERY_KEY not in at.query_params              # token borta ur adressraden
    assert at.query_params.get("p") == ["holdings"]           # andra parametrar orörda
    assert not _bridge_on_page(at)                            # inloggad → inget omladdningsskript


@pytest.mark.parametrize("bad", ["", "x", "123.abc", None])
def test_bridge_bad_token_shows_login(gate, bad):
    if bad is not None:
        gate.query_params[auth.QUERY_KEY] = bad
    at = gate.run()
    assert not at.exception
    assert not _inne(at) and at.session_state["authenticated"] is False
    assert auth.QUERY_KEY not in at.query_params
    assert at.button(key="login_btn")


def test_bridge_expired_or_other_password_rejected(gate):
    for tok in (auth.make_token("hemligt", now=1_000_000), auth.make_token("annat")):
        at = AppTest.from_string(_GATE, default_timeout=30)
        at.query_params[auth.QUERY_KEY] = tok
        at.run()
        assert not _inne(at)
        assert auth.QUERY_KEY not in at.query_params


def test_login_screen_carries_bridge_script(gate):
    at = gate.run()
    assert _bridge_on_page(at)


def test_bridge_script_shape():
    js = auth._BRIDGE_JS
    assert "k='wolf_auth'" in js and "localStorage.getItem(k)" in js
    assert "sessionStorage" in js and "location.replace" in js     # ett försök per token och flik
    assert "searchParams.set(q,t)" in js and "q='wolf_t'" in js


def test_set_cookie_also_writes_local_storage(monkeypatch):
    seen = []
    monkeypatch.setattr(auth.st, "html", lambda body, **kw: seen.append((body, kw)))
    auth._set_cookie("123.abc")
    body, kw = seen[0]
    assert "document.cookie='wolf_auth=123.abc;" in body
    assert "localStorage.setItem('wolf_auth','123.abc')" in body
    assert kw.get("unsafe_allow_javascript") is True


def test_manual_login_still_works(gate):
    at = gate.run()
    at.text_input(key="login_pw").input("hemligt")
    at.button(key="login_btn").click().run()
    assert _inne(at)


def test_already_logged_in_drops_stray_token(gate):
    gate.session_state["authenticated"] = True
    gate.query_params[auth.QUERY_KEY] = "skräp"
    at = gate.run()
    assert _inne(at) and auth.QUERY_KEY not in at.query_params


def test_usage_stats_are_off_so_the_bridge_token_never_leaves_for_telemetry():
    """Streamlits statistik skickar sidans adress (med ?wolf_t=) innan appen hunnit ta bort den."""
    import os
    import tomllib
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".streamlit", "config.toml"), "rb") as f:
        assert tomllib.load(f)["browser"]["gatherUsageStats"] is False
