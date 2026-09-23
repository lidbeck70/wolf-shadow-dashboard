"""
ui/nav.py — panelens navigationsträd, EN gång.

wolf_panel.py ritar flikarna ur det här, FLIKGUIDEN i RULES och Home-sidans
zoner genereras ur samma träd, och testerna läser det i stället för att
regex-tolka wolf_panel.py. En flytt eller ett namnbyte görs på ett ställe
och syns överallt — förut var flikguiden och Home-zonerna handskrivna och
saknade elva flikar.

Struktur: TOP är toppflikarna (nyckel, etikett). SUBS mappar en sökväg
("screening", "regime/Marknad") till listan av alternativ på nästa nivå.
Etiketterna är exakt de strängar som visas i radioknapparna.
"""

from __future__ import annotations

TOP: tuple = (
    ("home", "🏠 HOME"),
    ("screening", "🔱 SCREENING"),
    ("review", "🔬 GRANSKNING"),
    ("regime", "📡 REGIME"),
    ("intel", "👁 INTELLIGENCE"),
    ("portfolio", "💼 PORTFOLIO"),
    ("alerts", "🔔 ALERTS"),
    ("rules", "📋 RULES"),
    ("strategies", "🧬 STRATEGIES"),
    ("copilot", "🤖 COPILOT"),
)
TOP_LABEL = dict(TOP)

# Guidens tre steg i ordning: var kapitalet ska (REGIME), vilka bolag som
# kvalar (SCREENING), och vilket av dem som köps (GRANSKNING).
SUBS: dict = {
    "screening": ["Arc Screener", "Contrarian Alpha", "Swing Screener"],
    "screening/Arc Screener": ["Wolf", "Viking", "🔥 EMBER"],
    "screening/Contrarian Alpha": ["Screener", "Long Screener"],
    # Beslutsunderlaget efter screeningen. Durrett och Confidence är två vyer
    # över samma ark (data/confidence.json) och ligger därför under en flik.
    "review": ["Rick Rule", "Royalty C", "Poängmodell", "Tiggre", "Insider",
               "🧭 Durrett & Confidence", "🎯 Scorecard"],
    "review/🧭 Durrett & Confidence": ["Durrett 10-steg", "Confidence-case"],
    # REGIME delat i två: marknaden (index, sektorer, cykel) och råvarorna.
    "regime": ["Marknad", "Råvaror"],
    "regime/Marknad": ["Arc Regime", "Alpha Regime", "Swing Regime",
                       "Flow Divergence", "Market Cycle"],
    "regime/Marknad/Arc Regime": ["Wolf Regime", "Viking Regime"],
    "regime/Marknad/Alpha Regime": ["Quality & Contrarian", "Long Trend"],
    "regime/Råvaror": ["🌍 EMBER Regime", "Råvarurotation"],
    "intel": ["Odin's Blindspot", "Sentiment", "Retail Pulse", "Heatmap"],
    "portfolio": ["📓 Trade Journal", "Holdings", "Swing", "Allokering", "Backtest"],
    "rules": ["Regler & Guider", "Position Sizing", "Data Health"],
    "rules/Regler & Guider": ["🚀 KOM IGÅNG", "📋 HANDELSREGLER", "⚡ FUSKLAPP",
                              "🗺 FLIKGUIDE", "📖 STRATEGIGUIDER", "🗓 ÅRSHJULET",
                              "📚 SNABBREFERENS"],
}

# Vad Home-sidan säger om varje zon: (toppnyckel, rubrik, [(flik, beskrivning)]).
HOME_ZONES: tuple = (
    ("regime", "REGIME — VAR KAPITALET SKA", [
        ("Marknad", "Wolf · Viking · Alpha · Swing · Flow Divergence · Market Cycle"),
        ("Råvaror", "🌍 EMBER Regime · Råvarurotation"),
    ]),
    ("screening", "SCREENING — VILKA BOLAG SOM KVALAR", [
        ("Arc Screener", "Wolf (EMA/swing) · Viking (OVTLYR) · EMBER (råvaror)"),
        ("Contrarian Alpha", "Hatade bolag med kvalitet · Long Screener (CAGR)"),
        ("Swing Screener", "Momentum-ranking topp 40 med setup A/B"),
    ]),
    ("review", "GRANSKNING — VILKET BOLAG SOM KÖPS", [
        ("Rick Rule · Royalty C", "Producenter och royaltybolag mot kostnadskurvan"),
        ("Poängmodell · Tiggre · Insider", "Sprott, Durrett-snabbpoäng, Lobo-arket, insynsflödet"),
        ("🧭 Durrett & Confidence", "Durretts 10 steg och Confidence-caset — samma ark"),
        ("🎯 Scorecard", "Köpgrinden: sju kryss före ordern"),
    ]),
    ("intel", "INTELLIGENCE — TOLKA SIGNALERNA", [
        ("Odin's Blindspot", "Contrarian sektorintelligens"),
        ("Sentiment", "Fear, greed och kapitalflöden"),
        ("Retail Pulse", "Reddit, StockTwits, retail-flöde"),
        ("Heatmap", "Visuell marknadsvy"),
    ]),
    ("portfolio", "PORTFOLIO — HANTERA POSITIONER", [
        ("Holdings", "Positioner och riskexponering"),
        ("📓 Trade Journal", "Logga affärer och granska R"),
        ("Allokering", "Ramar, tak och strömbrytaren"),
        ("Backtest", "Historisk signalvalidering"),
    ]),
)


def options(path: str) -> list:
    """Alternativen på nästa nivå under sökvägen, eller [] om det är ett löv."""
    return list(SUBS.get(path, []))


def leaves() -> set:
    """Alla etiketter som går att klicka på i panelen (toppflikar + underflikar)."""
    out = {label for _k, label in TOP}
    for opts in SUBS.values():
        out |= set(opts)
    return out


def paths() -> list:
    """Alla fullständiga sökvägar 'REGIME → Marknad → Arc Regime → Wolf Regime'."""
    out = []

    def _walk(key: str, prefix: list) -> None:
        opts = SUBS.get(key)
        if not opts:
            out.append(" → ".join(prefix))
            return
        for o in opts:
            _walk(f"{key}/{o}", prefix + [o])

    for key, label in TOP:
        _walk(key, [label.split(" ", 1)[1] if " " in label else label])
    return out
