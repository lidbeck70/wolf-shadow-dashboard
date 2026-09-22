"""
confidence — Case Score + Confidence Score för gruv-, råvaru- och
strategiska mineralbolag (GRANSKNING → 🧭 Confidence score).

Två poäng som aldrig blandas:
  Case Score        0–100  hur attraktivt caset är (åtta pelare)
  Confidence Score  0–100  hur säkra vi är på att analysen bygger på
                           verifierbara, färska och robusta fakta

Regeln som bär hela paketet: hittat värde eller null. Ingen datapunkt
hittas på; saknad data ger DATA_MISSING, noll poäng i pelaren och lägre
Confidence. Varje poäng ska gå att förklara i klartext.

Rena moduler (config, data, scoring, scenarios, reports) har inga
Streamlit-beroenden. UI:t ligger i confidence/ui.py.
"""
from confidence.ui import render_confidence_page

__all__ = ["render_confidence_page"]
