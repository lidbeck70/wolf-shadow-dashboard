"""
ai/openai_client.py — det enda stället som pratar med OpenAI.

Skrivet efter samma princip som storage.py: **ett misslyckat anrop får aldrig
se ut som ett lyckat.** Varje felväg kastar AIError med en text som säger vad
som är fel och vad du gör åt det. Ingen tyst fallback till stubben — den som
inte vet att AI-texten uteblev läser en deterministisk sammanfattning som om
en modell hade skrivit den.

Konfiguration i Streamlit secrets (toppnivå, ÖVER eventuella [sektioner] —
allt efter en sektionsrubrik hamnar inuti den sektionen):

    OPENAI_API_KEY = "sk-proj-..."
    OPENAI_MODEL   = "gpt-5.6-luna"    # valfri, se MODEL nedan

Anropet är alltid knappstyrt i UI:t. Streamlit kör om hela skriptet vid varje
widget-interaktion, så ett anrop i renderingsvägen hade betytt ett betalt
API-anrop per reglagedrag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Modellen. Byt här — eller sätt OPENAI_MODEL i secrets, vilket vinner över
# konstanten. Namnet verifieras inte förrän första anropet: stämmer det inte
# svarar OpenAI med "model not found", och complete() översätter det till ett
# meddelande som pekar på just det här och inte på nyckeln.
MODEL = "gpt-5.6-luna"

KEY_NAME = "OPENAI_API_KEY"
MODEL_NAME = "OPENAI_MODEL"

DEFAULT_TIMEOUT = 45.0
DEFAULT_MAX_OUTPUT_TOKENS = 900

SECRETS_TEMPLATE = (
    f'{KEY_NAME} = "sk-proj-..."\n'
    f'{MODEL_NAME}   = "{MODEL}"   # valfri\n\n'
    "Lägg raderna ÖVER [github]-rubriken — allt under en sektionsrubrik "
    "hamnar inuti sektionen och hittas inte."
)


class AIError(RuntimeError):
    """Allt som hindrade ett svar. Texten är till för att visas i panelen."""


class AINotConfigured(AIError):
    """Nyckel eller paket saknas — skiljs ut för att UI:t ska kunna
    säga 'inte påslagen' i stället för 'gick sönder'."""


@dataclass(frozen=True)
class Reply:
    text: str
    model: str


def _secret(name: str) -> str:
    """st.secrets först, sedan miljövariabel. Samma ordning som EODHD-läsaren."""
    try:
        import streamlit as st
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(name) or "").strip()


def get_key() -> str:
    key = _secret(KEY_NAME)
    if not key:
        raise AINotConfigured(
            f"{KEY_NAME} saknas i secrets. Lägg till:\n\n{SECRETS_TEMPLATE}")
    if key.startswith("sk-") and len(key) < 20:
        raise AINotConfigured(
            f"{KEY_NAME} ser ut som en platshållare ({len(key)} tecken). "
            f"Klistra in hela nyckeln från platform.openai.com → API keys.")
    return key


def get_model() -> str:
    return _secret(MODEL_NAME) or MODEL


def configured() -> bool:
    """Om modulen ÖVER HUVUD TAGET kan anropas. Rör inte nätverket.

    Säger inget om nyckeln är giltig eller modellnamnet finns — det vet man
    först efter ett anrop, och complete() rapporterar det då.
    """
    try:
        get_key()
    except AIError:
        return False
    try:
        import openai                                    # noqa: F401
    except ImportError:
        return False
    return True


def _client(timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:                           # pragma: no cover
        raise AINotConfigured(
            "Paketet openai är inte installerat. Det ska ligga pinnat i "
            "requirements.txt — utan det startar appen men fliken kan inte "
            "anropa någon modell.") from exc
    return OpenAI(api_key=get_key(), timeout=timeout)


def _extract(response) -> str:
    """Texten ur svaret.

    output_text är bekvämlighetsfältet; faller det bort i en framtida version
    plockar vi ihop texten ur output-listan i stället för att returnera tomt.
    """
    text = (getattr(response, "output_text", "") or "").strip()
    if text:
        return text
    parts = []
    for item in getattr(response, "output", None) or []:
        for chunk in getattr(item, "content", None) or []:
            piece = getattr(chunk, "text", None)
            if piece:
                parts.append(str(piece))
    return "\n".join(parts).strip()


def complete(instructions: str, prompt: str,
             max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
             timeout: float = DEFAULT_TIMEOUT,
             model: Optional[str] = None) -> Reply:
    """Ett anrop, ett svar. Kastar AIError på allt annat än ett svar.

    Felen översätts till vad DU ska göra åt dem — ett rått
    "AuthenticationError" mitt i en tradingpanel säger ingenting om huruvida
    det är nyckeln, kvoten eller modellnamnet som är fel.
    """
    import openai

    name = model or get_model()
    client = _client(timeout)
    try:
        response = client.responses.create(
            model=name, instructions=instructions, input=prompt,
            max_output_tokens=max_output_tokens)
    except openai.AuthenticationError as exc:
        raise AIError(f"OpenAI avvisade nyckeln (401). Kontrollera "
                      f"{KEY_NAME} i secrets — den kan vara återkallad eller "
                      f"tillhöra fel projekt.") from exc
    except openai.NotFoundError as exc:
        raise AIError(f"Modellen '{name}' finns inte, eller är inte tillgänglig "
                      f"för ditt konto. Namnet sätts av {MODEL_NAME} i secrets, "
                      f"annars av MODEL i ai/openai_client.py. Aktuella namn "
                      f"står på platform.openai.com/docs/models.") from exc
    except openai.PermissionDeniedError as exc:
        raise AIError(f"Nyckeln får inte använda '{name}' (403). Projektet kan "
                      f"sakna tillgång till modellen.") from exc
    except openai.RateLimitError as exc:
        raise AIError("OpenAI svarade 429 — kvot eller hastighetsgräns nådd. "
                      "Är det kvoten behöver kontot fyllas på; är det "
                      "hastigheten räcker det att vänta.") from exc
    except openai.APITimeoutError as exc:
        raise AIError(f"Ingen svarstid inom {timeout:g} s. Ingenting sparades "
                      f"och inget debiterades säkert — försök igen.") from exc
    except openai.APIConnectionError as exc:
        raise AIError("Nådde inte OpenAI. Kontrollera nätverket och att "
                      "utgående trafik tillåts.") from exc
    except openai.APIStatusError as exc:                 # allt annat med status
        raise AIError(f"OpenAI svarade {exc.status_code}: "
                      f"{getattr(exc, 'message', '')}") from exc
    except openai.OpenAIError as exc:                    # sista skyddsnätet
        raise AIError(f"OpenAI-anropet misslyckades: {exc}") from exc

    text = _extract(response)
    if not text:
        raise AIError(f"Modellen '{name}' svarade utan text. Det händer när "
                      f"svaret klipptes av token-taket "
                      f"({max_output_tokens}) — höj det eller korta prompten.")
    return Reply(text=text, model=name)
