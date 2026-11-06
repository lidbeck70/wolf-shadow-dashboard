"""Kom-ihåg-inloggningen: signerad kaka som överlever en ny Streamlit-session."""
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
    for bad in (None, "", "x", "abc.def", ".", "123", "1.2.3"):
        assert auth.token_valid(bad, "hemligt") is False


def test_token_holds_no_password():
    assert "hemligt" not in auth.make_token("hemligt")
