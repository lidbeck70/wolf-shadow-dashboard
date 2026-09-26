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

import re

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
               "🧭 Durrett & Confidence", "🐺 Wolf Asymmetry", "🎯 Scorecard"],
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
        ("🐺 Wolf Asymmetry", "Eget ark för alla strategier: hävstång, margin of safety, stressmatris, justerad uppsida"),
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


# Widget-nyckeln i session_state för radion på varje nivå. wolf_panel ritar
# _sub(path) ur den här, och djuplänken seedar samma nycklar.
TABS_KEY = "main_tabs"
STATE_KEY: dict = {
    "screening": "sub_screening",
    "screening/Arc Screener": "sub_screening_arc",
    "screening/Contrarian Alpha": "sub_screening_contrarian",
    "review": "sub_review",
    "review/🧭 Durrett & Confidence": "sub_review_durrett",
    "regime": "sub_regime_group",
    "regime/Marknad": "sub_regime",
    "regime/Marknad/Arc Regime": "sub_regime_arc",
    "regime/Marknad/Alpha Regime": "sub_regime_alpha",
    "regime/Råvaror": "sub_regime_commodities",
    "intel": "sub_intel",
    "portfolio": "sub_portfolio",
    "rules": "sub_rules",
    "rules/Regler & Guider": "rules_sub",          # ritas inne i rules_page
}


def tab_label(key: str) -> str:
    """Toppflikens etikett som st.tabs får den (med luft runt)."""
    return f"  {TOP_LABEL[key]}  "


def options(path: str) -> list:
    """Alternativen på nästa nivå under sökvägen, eller [] om det är ett löv."""
    return list(SUBS.get(path, []))


# ── Djuplänkar: ?p=regime/marknad/arc-regime/wolf-regime ─────────────────────
_TRANS = str.maketrans({"å": "a", "ä": "a", "ö": "o", "é": "e"})


def slugify(label: str) -> str:
    """'🧭 Durrett & Confidence' → 'durrett-confidence', 'Råvaror' → 'ravaror'."""
    s = str(label or "").lower().translate(_TRANS).replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def slug(segments) -> str:
    """Sökvägens segment [toppnyckel, etikett, ...] → URL-slug."""
    return "/".join(slugify(s) for s in segments)


def _top_key(part: str):
    for key, label in TOP:
        if part in (slugify(key), slugify(label.split(" ", 1)[1])):
            return key
    return None


def resolve(link: str):
    """Slug → segment [toppnyckel, etikett, ...], eller None om toppfliken
    är okänd. Okända segment längre ned ignoreras (länken landar så långt
    trädet känner igen den). 'granskning/tiggre' och 'review/tiggre' är samma."""
    parts = [p for p in str(link or "").split("/") if p]
    if not parts:
        return None
    top = _top_key(parts[0])
    if top is None:
        return None
    segs, path = [top], top
    for p in parts[1:]:
        hit = next((o for o in SUBS.get(path, []) if slugify(o) == p), None)
        if hit is None:
            break
        segs.append(hit)
        path = f"{path}/{hit}"
    return segs


def seed(state, segments) -> None:
    """Skriv segmenten till widget-nycklarna INNAN widgetarna ritas, så att
    flik och underflikar öppnar sig på länken."""
    if not segments:
        return
    state[TABS_KEY] = tab_label(segments[0])
    path = segments[0]
    for seg in segments[1:]:
        key = STATE_KEY.get(path)
        if key:
            state[key] = seg
        path = f"{path}/{seg}"


def current(state) -> list:
    """Sökvägen panelen visar just nu, ur widget-nycklarna (första
    alternativet där inget är valt)."""
    lbl = str(state.get(TABS_KEY) or "").strip()
    top = next((k for k, l in TOP if l == lbl), TOP[0][0])
    segs, path = [top], top
    while path in SUBS:
        opts = SUBS[path]
        val = state.get(STATE_KEY.get(path, ""))
        if val not in opts:
            val = opts[0]
        segs.append(val)
        path = f"{path}/{val}"
    return segs


def link_for(path_text: str) -> str:
    """'REGIME → Marknad → Swing Regime' (FLIKGUIDENS form) → '?p=…', eller
    '' om sökvägen inte finns i trädet."""
    parts = [p.strip() for p in str(path_text or "").split("→") if p.strip()]
    if not parts:
        return ""
    segs = resolve("/".join(slugify(p) for p in parts))
    if not segs or len(segs) != len(parts):
        return ""
    return "?p=" + slug(segs)


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
