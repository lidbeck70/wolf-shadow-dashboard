"""
ai/copilot_prompt.py — vad Copiloten får se, och vad den får säga.

Prompten byggs av det regelmotorn REDAN kommit fram till. Modellen får inte
räkna om R:R, risk eller regelutfall — de siffrorna skickas färdiga och
instruktionen säger uttryckligen att de ska citeras, inte räknas om. Det är
CLAUDE.md:s regel i praktiken: motorerna äger besluten, AI:n förklarar dem.

Rena funktioner, ingen Streamlit och inget nätverk — så prompten går att
testa utan API-nyckel.
"""

from __future__ import annotations

SYSTEM = """Du är en analytisk andrahandsläsare i en svensk tradingpanel.

Regelmotorn har redan gjort bedömningen. Ditt jobb är att förklara den och
peka på det som är svagt — inte att göra om den.

Absoluta krav:
- Räkna ALDRIG om R:R, risk i procent eller regelutfall. Siffrorna du får är
  panelens; citera dem exakt som de står.
- Föreslå ALDRIG en positionsstorlek, och ändra aldrig stop eller target.
- Motsäg aldrig statusen. Står det REJECT förklarar du varför den föll, du
  argumenterar inte för affären.
- Hittar du inget att invända mot, skriv det. Fyll inte ut med brasklappar.

Skriv svenska, 120–180 ord, i löpande text utan rubriker. Var konkret:
vilken regel är svagast, vad skulle få dig att ändra dig, vad ska kollas
manuellt innan avslut. Avsluta med den enskilt viktigaste frågan att besvara
före ett köp."""


def build_prompt(ticker: str, strategy: str, status: str,
                 entry: float, stop: float, target: float,
                 rr: float, risk_pct: float,
                 passed: list, manual: list, failed: list,
                 risk_per_trade: str = "") -> str:
    """Underlaget, i klartext. Allt är redan räknat av motorn."""
    def _bullets(items) -> str:
        return "\n".join(f"  - {t}" for t in items) if items else "  (inga)"

    return f"""Kandidat: {ticker}
Strategi: {strategy}
Panelens status: {status}

Nivåer (satta av användaren, redan validerade av motorn):
  Entry {entry:g} · Stop {stop:g} · Target {target:g}
  R:R {rr:.1f}x · risk mot entry {risk_pct:.1f} %{
    chr(10) + '  Strategins riskram: ' + risk_per_trade if risk_per_trade else ''}

Regler som PASSERADE ({len(passed)}):
{_bullets(passed)}

Regler som kräver MANUELL kontroll ({len(manual)}):
{_bullets(manual)}

Regler som FÖLL ({len(failed)}):
{_bullets(failed)}

Kommentera underlaget enligt instruktionen."""
