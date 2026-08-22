"""
Tester för ai/openai_client.py och ai/copilot_prompt.py.

Tyngdpunkten ligger på FELVÄGARNA. Ett misslyckat anrop som ser ut som ett
lyckat är exakt den bugg som gjorde att sparningen tappade data i veckor —
här ska varje fel bli synligt och peka på sin egen orsak.

Inget test rör nätverket.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai

from ai import copilot_prompt as cp
from ai import openai_client as oc


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(oc.KEY_NAME, raising=False)
    monkeypatch.delenv(oc.MODEL_NAME, raising=False)


# ── Konfiguration ────────────────────────────────────────────────────────────
def test_a_missing_key_names_itself_and_shows_the_template():
    with pytest.raises(oc.AINotConfigured) as e:
        oc.get_key()
    assert oc.KEY_NAME in str(e.value)
    assert "sk-proj-" in str(e.value)
    # och den påminner om TOML-fällan som redan kostat en felsökning
    assert "[github]" in str(e.value)


def test_a_placeholder_key_is_rejected_before_the_network(monkeypatch):
    """github_pat_.... igen, fast för OpenAI. Bättre att fånga det här."""
    monkeypatch.setenv(oc.KEY_NAME, "sk-...")
    with pytest.raises(oc.AINotConfigured) as e:
        oc.get_key()
    assert "platshållare" in str(e.value)


def test_a_real_looking_key_passes(monkeypatch):
    monkeypatch.setenv(oc.KEY_NAME, "sk-proj-" + "a" * 40)
    assert oc.get_key().startswith("sk-proj-")


def test_configured_is_false_without_a_key():
    assert oc.configured() is False


def test_configured_is_true_with_a_key(monkeypatch):
    monkeypatch.setenv(oc.KEY_NAME, "sk-proj-" + "a" * 40)
    assert oc.configured() is True


def test_configured_never_touches_the_network(monkeypatch):
    """Den anropas vid varje rerun — den får inte kosta något."""
    monkeypatch.setenv(oc.KEY_NAME, "sk-proj-" + "a" * 40)

    def _boom(*a, **k):
        raise AssertionError("configured() rörde nätverket")

    monkeypatch.setattr(oc, "_client", _boom)
    assert oc.configured() is True


def test_the_model_comes_from_secrets_before_the_constant(monkeypatch):
    assert oc.get_model() == oc.MODEL
    monkeypatch.setenv(oc.MODEL_NAME, "annan-modell")
    assert oc.get_model() == "annan-modell"


def test_the_default_model_is_the_one_that_was_asked_for():
    assert oc.MODEL == "gpt-5.6-luna"


# ── Felöversättningen ────────────────────────────────────────────────────────
class _FakeResponses:
    def __init__(self, exc=None, response=None):
        self._exc, self._response = exc, response
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if self._exc:
            raise self._exc
        return self._response


def _client_with(monkeypatch, exc=None, response=None):
    responses = _FakeResponses(exc, response)
    client = types.SimpleNamespace(responses=responses)
    monkeypatch.setattr(oc, "_client", lambda timeout: client)
    monkeypatch.setenv(oc.KEY_NAME, "sk-proj-" + "a" * 40)
    return responses


def _api_error(cls, status):
    """Bygger en openai-status-exception utan att gå via nätverket."""
    request = types.SimpleNamespace(method="POST", url="/responses")
    response = types.SimpleNamespace(status_code=status, request=request,
                                     headers={})
    return cls("fel", response=response, body=None)


def test_a_bad_key_blames_the_key_not_the_model(monkeypatch):
    _client_with(monkeypatch, _api_error(openai.AuthenticationError, 401))
    with pytest.raises(oc.AIError) as e:
        oc.complete("i", "p")
    msg = str(e.value)
    assert "401" in msg and oc.KEY_NAME in msg
    assert "modell" not in msg.lower()


def test_an_unknown_model_blames_the_model_and_says_where_to_look(monkeypatch):
    """Det troligaste felet med ett modellnamn som aldrig verifierats."""
    _client_with(monkeypatch, _api_error(openai.NotFoundError, 404))
    with pytest.raises(oc.AIError) as e:
        oc.complete("i", "p")
    msg = str(e.value)
    assert oc.MODEL in msg
    assert oc.MODEL_NAME in msg and "openai_client.py" in msg
    assert "docs/models" in msg


def test_rate_limit_separates_quota_from_speed(monkeypatch):
    _client_with(monkeypatch, _api_error(openai.RateLimitError, 429))
    with pytest.raises(oc.AIError) as e:
        oc.complete("i", "p")
    assert "429" in str(e.value) and "kvot" in str(e.value).lower()


def test_a_timeout_says_nothing_was_saved(monkeypatch):
    request = types.SimpleNamespace(method="POST", url="/responses")
    _client_with(monkeypatch, openai.APITimeoutError(request=request))
    with pytest.raises(oc.AIError) as e:
        oc.complete("i", "p", timeout=12)
    assert "12" in str(e.value)


def test_an_unexpected_status_still_raises(monkeypatch):
    _client_with(monkeypatch, _api_error(openai.InternalServerError, 500))
    with pytest.raises(oc.AIError):
        oc.complete("i", "p")


def test_no_error_is_swallowed_into_an_empty_string(monkeypatch):
    """Poängen med hela modulen: complete() returnerar text eller kastar."""
    for exc in (_api_error(openai.AuthenticationError, 401),
                _api_error(openai.NotFoundError, 404),
                _api_error(openai.RateLimitError, 429),
                _api_error(openai.InternalServerError, 500)):
        _client_with(monkeypatch, exc)
        with pytest.raises(oc.AIError):
            oc.complete("i", "p")


# ── Svaret ───────────────────────────────────────────────────────────────────
def test_a_normal_reply_comes_back_with_its_model(monkeypatch):
    resp = types.SimpleNamespace(output_text="  Ser tunt ut på volym.  ")
    calls = _client_with(monkeypatch, response=resp)
    reply = oc.complete("instruktion", "underlag")
    assert reply.text == "Ser tunt ut på volym."
    assert reply.model == oc.MODEL
    assert calls.calls[0]["model"] == oc.MODEL
    assert calls.calls[0]["instructions"] == "instruktion"
    assert calls.calls[0]["input"] == "underlag"


def test_the_text_is_recovered_when_output_text_is_missing(monkeypatch):
    resp = types.SimpleNamespace(
        output_text="",
        output=[types.SimpleNamespace(
            content=[types.SimpleNamespace(text="del ett"),
                     types.SimpleNamespace(text="del två")])])
    _client_with(monkeypatch, response=resp)
    assert oc.complete("i", "p").text == "del ett\ndel två"


def test_an_empty_reply_is_an_error_not_an_empty_comment(monkeypatch):
    """En tom ruta läses som 'AI:n hade inget att invända'. Det är fel."""
    _client_with(monkeypatch, response=types.SimpleNamespace(output_text=""))
    with pytest.raises(oc.AIError) as e:
        oc.complete("i", "p", max_output_tokens=50)
    assert "50" in str(e.value)


def test_the_model_can_be_overridden_per_call(monkeypatch):
    calls = _client_with(monkeypatch,
                         response=types.SimpleNamespace(output_text="ok"))
    reply = oc.complete("i", "p", model="särskild-modell")
    assert calls.calls[0]["model"] == "särskild-modell"
    assert reply.model == "särskild-modell"


# ── Prompten ─────────────────────────────────────────────────────────────────
def test_the_instruction_forbids_recomputing_the_engine():
    """CLAUDE.md: motorerna äger besluten, AI:n förklarar dem."""
    s = cp.SYSTEM.lower()
    assert "räkna aldrig om" in s
    assert "positionsstorlek" in s
    assert "motsäg aldrig statusen" in s


def test_the_prompt_carries_the_engines_numbers_verbatim():
    p = cp.build_prompt(
        ticker="ABB", strategy="Momentum Swing", status="WATCH",
        entry=100.0, stop=90.0, target=130.0, rr=3.0, risk_pct=10.0,
        passed=["Trend uppåt"], manual=["Volymbekräftelse"],
        failed=[], risk_per_trade="1,2–2 %")
    assert "ABB" in p and "Momentum Swing" in p and "WATCH" in p
    assert "3.0x" in p and "10.0 %" in p
    assert "Trend uppåt" in p and "Volymbekräftelse" in p
    assert "1,2–2 %" in p


def test_empty_rule_groups_read_as_none_not_as_a_blank():
    p = cp.build_prompt("X", "S", "REJECT", 1, 2, 3, 0.5, 5.0, [], [], [])
    assert p.count("(inga)") == 3
    assert "FÖLL (0)" in p


# ── Marknadsdata i prompten ──────────────────────────────────────────────────
class _Snap:
    ticker = "ABB"
    as_of = "2026-08-18"
    bars = 250
    price = 100.0
    atr14 = 3.0
    atr_pct = 3.0
    ema50 = 96.0
    ema200 = 90.0
    dist_ema50_pct = 4.17
    dist_ema200_pct = 11.1
    rsi14 = 58.0
    vol_ratio = 1.4
    high_52w = 110.0
    low_52w = 70.0
    from_high_pct = -9.1
    swing_low_20 = 93.0
    swing_high_20 = 105.0
    ret_1m_pct = 6.0
    ret_3m_pct = 12.0


def test_the_prompt_carries_the_market_numbers():
    p = cp.build_prompt("ABB", "Momentum Swing", "WATCH", 100, 95, 115,
                        3.0, 5.0, [], [], [], snapshot=_Snap())
    assert "ATR(14) 3.00" in p and "EMA200 90.00" in p
    assert "1.40×" in p and "RSI(14) 58" in p
    assert "swing-low 93.00" in p


def test_a_missing_snapshot_tells_the_model_not_to_guess():
    """Utan kursdata får modellen inte uttala sig om trend eller volym."""
    p = cp.build_prompt("ABB", "S", "WATCH", 100, 95, 115, 3.0, 5.0, [], [], [])
    assert "Marknadsdata: SAKNAS" in p
    assert "Kommentera\ninte trend" in p or "Kommentera inte trend" in p


def test_the_prompt_survives_an_unfillable_rr():
    """None ska bli ett streck, inte krascha formateringen."""
    p = cp.build_prompt("ABB", "S", "WATCH", 100, 100, 0, None, None,
                        [], [], [])
    assert "R:R –" in p


def test_the_level_alternatives_reach_the_model():
    import levels as lv
    alts = lv.stop_candidates(100, _Snap(), fixed_pct=10)
    p = cp.build_prompt("ABB", "S", "WATCH", 100, 95, 115, 3.0, 5.0,
                        [], [], [], snapshot=_Snap(),
                        assessment=lv.assess(100, 95, 115, _Snap()),
                        alternatives=alts)
    assert "Alternativa stoppnivåer" in p
    assert "Swing-low" in p and "% risk" in p


def test_the_review_instruction_refuses_to_read_noise_as_signal():
    s = cp.REVIEW_SYSTEM.lower()
    assert "räkna aldrig om" in s
    assert "20 avslutade affärer" in s and "brus" in s


def test_the_cycle_state_reaches_the_model_with_source_and_month():
    state = {"commodity": "Guld", "status": "AGERA", "sum": 14, "max": 15,
             "why": "Summa 14/15", "month": "2026-08",
             "warnings": ["Värdefälla?"]}
    bspot = {"timestamp": "2026-08-10T12:13:56", "opportunity": 62.0,
             "hat": 80.0, "strength": 55.0, "catalyst": 40.0,
             "sector": "Uranium"}
    p = cp.build_prompt("CCJ", "Rule", "WATCH", 100, 90, 130, 3.0, 10.0,
                        [], [], [], cycle_state=state, blindspot=bspot)
    assert "Guld = AGERA 14/15" in p and "2026-08" in p
    assert "VARNING: Värdefälla?" in p
    assert "Blindspot" in p and "2026-08-10" in p and "Uranium" in p


def test_no_cycle_data_adds_nothing_to_the_prompt():
    p = cp.build_prompt("ABB", "S", "WATCH", 100, 90, 130, 3.0, 10.0,
                        [], [], [])
    assert "Cykelläge" not in p and "Blindspot" not in p
