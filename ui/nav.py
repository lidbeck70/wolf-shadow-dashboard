"""
ui/nav.py — panelens navigationsträd, EN gång.

wolf_panel.py ritar flikarna ur det här, FLIKGUIDEN i RULES och Home-sidans
zoner genereras ur samma träd, och testerna läser det i stället för att
regex-tolka wolf_panel.py. En flytt eller ett namnbyte görs på ett ställe
och syns överallt — förut var flikguiden och Home-zonerna handskrivna och
saknade elva flikar.

Struktur: TOP är toppflikarna (nyckel, etikett). SUBS mappar en sökväg
("odin", "odin/Marknad") till listan av alternativ på nästa nivå.
Etiketterna är exakt de strängar som visas i radioknapparna.

Fas 8: 🔨 THOR ersätter 🔱 SCREENING (alla strategiers kandidater med ⚡ Skanna nu),
👁 ODINS VISDOM samlar 📡 REGIME och 👁 INTELLIGENCE (Råvaror · Makro · Marknad) och
📒 ARKEN är 🔬 GRANSKNING med samma innehåll. Gamla länkar landar rätt via LEGACY_PATHS.
"""

from __future__ import annotations

import re

TOP: tuple = (
    ("home", "🏠 HOME"),
    ("valhall", "⚡ VALHALL"),
    ("thor", "🔨 THOR"),
    ("odin", "👁 ODINS VISDOM"),
    ("review", "📒 ARKEN"),
    ("portfolio", "💼 PORTFOLIO"),
    ("alerts", "🔔 ALERTS"),
    ("rules", "📋 RULES"),
    ("strategies", "🧬 STRATEGIES"),
    ("copilot", "🤖 COPILOT"),
)
TOP_LABEL = dict(TOP)

# Guidens steg i ordning: var kapitalet ska (ODINS VISDOM), vilka bolag som
# kvalar och analysen före köp (THOR), och arken som avgör (ARKEN).
SUBS: dict = {
    # Valhall: cykelkartan över hela marknaden med korten, papperskontona och backtesterna.
    "valhall": ["Kartan", "Papperskonton", "Backtester"],
    # Odins visdom: råvarorna, makro och marknaden. Arc och Alpha Regime ligger kvar under Marknad —
    # Thors bryggknappar öppnar dem för en aktie, och Viking-sidorna rörs inte.
    "odin": ["Råvaror", "Makro", "Marknad"],
    "odin/Råvaror": ["Odin's Blindspot", "🌍 EMBER Regime", "🪓 BERSERK Regime", "Råvarurotation",
                     "🥇🥈 Guld/Silver", "🥇 Guldkvoter"],
    "odin/Makro": ["🐺 Fiat Debasement", "Sentiment"],
    "odin/Marknad": ["Läget", "Swing Regime", "Flow Divergence", "Market Cycle", "🌩️ Marknadsrisk",
                     "Heatmap", "Retail Pulse", "Arc Regime", "Alpha Regime"],
    "odin/Marknad/Arc Regime": ["Wolf Regime", "Viking Regime", "⚔️ Viking Nine Regime"],
    "odin/Marknad/Alpha Regime": ["Quality & Contrarian", "Long Trend"],
    # Beslutsunderlaget efter screeningen. Durrett har sitt ark
    # (data/confidence.json); Wolf Asymmetry har sitt eget med Confidence inbyggt.
    "review": ["Rick Rule", "Royalty C", "Poängmodell", "Tiggre", "Insider",
               "🧭 Durrett", "🐺 Wolf Asymmetry", "🚀 Råvaruhävstång", "🎯 Scorecard", "📊 Papperskonton"],
    "portfolio": ["📓 Trade Journal", "🎯 Mot målet", "Holdings", "Swing", "Allokering", "Backtest"],
    "rules": ["Regler & Guider", "⚙️ Inställningar", "Position Sizing", "Data Health"],
    "rules/Regler & Guider": ["🚀 KOM IGÅNG", "📋 HANDELSREGLER", "⚡ FUSKLAPP",
                              "🗺 FLIKGUIDE", "📖 STRATEGIGUIDER", "🗓 ÅRSHJULET",
                              "📚 SNABBREFERENS"],
}

# Vad Home-sidan säger om varje zon: (toppnyckel, rubrik, [(flik, beskrivning)]).
HOME_ZONES: tuple = (
    ("odin", "ODINS VISDOM — VAR KAPITALET SKA", [
        ("Råvaror", "Odin's Blindspot · 🌍 EMBER Regime · 🪓 BERSERK Regime · Råvarurotation · Guld/Silver · Guldkvoter"),
        ("Makro", "🐺 Fiat Debasement och Sentiment — penningvärde, fear & greed"),
        ("Marknad", "Swing Regime · Flow Divergence · Market Cycle · Marknadsrisk · Heatmap · Retail Pulse · "
                    "Arc och Alpha Regime"),
    ]),
    ("thor", "THOR — KANDIDATER OCH ANALYS", [
        ("🔨 THOR", "Alla strategiers kandidater, ⚡ Skanna nu per strategi och analysen före köp: regim, "
                   "teknik, cykel, råvara, asymmetri, arken, backtest och regler"),
    ]),
    ("review", "ARKEN — VILKET BOLAG SOM KÖPS", [
        ("Rick Rule · Royalty C", "Producenter och royaltybolag mot kostnadskurvan"),
        ("Poängmodell · Tiggre · Insider", "Sprott, Durrett-snabbpoäng, Lobo-arket, insynsflödet"),
        ("🧭 Durrett", "Durretts 10 steg med Håven och eget ark"),
        ("🐺 Wolf Asymmetry", "En sida per bolag: hävstång, margin of safety, confidence, stressmatris, justerad uppsida"),
        ("🚀 Råvaruhävstång", "Bolagens hävstång mot sin råvara: resultat, kursbeta upp/ned och 5× — jämför upp till åtta"),
        ("🎯 Scorecard", "Köpgrinden: sju kryss före ordern"),
    ]),
    ("valhall", "VALHALL — HELA SYSTEMET", [
        ("Kartan", "Cykeln per sektor och råvara med swing, Wolf, kvalitet, hävstång och Long Screener per ruta"),
        ("Papperskonton", "Varje strategi på papper — per segment, status och cykel vid köp"),
        ("Backtester", "Besluten och varför: swing, kvalitet, råvaror och Long Screener 2008–idag"),
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
    "valhall": "sub_valhall",
    "odin": "sub_odin",
    "odin/Råvaror": "sub_odin_ravaror",
    "odin/Makro": "sub_odin_makro",
    "odin/Marknad": "sub_odin_marknad",
    "odin/Marknad/Arc Regime": "sub_regime_arc",
    "odin/Marknad/Alpha Regime": "sub_regime_alpha",
    "review": "sub_review",
    "portfolio": "sub_portfolio",
    "rules": "sub_rules",
    "rules/Regler & Guider": "rules_sub",          # ritas inne i rules_page
}

# Gamla djuplänkar som fortfarande ska landa rätt: slug → dagens slug.
# 'granskning/durrett-confidence/...' öppnade "🧭 Durrett & Confidence" innan
# Confidence-caset byggdes in i Wolf Asymmetry (steg F).
LEGACY_SLUGS: dict = {"durrett-confidence": "durrett"}

# Fas 8: gamla sökvägar (slug-prefix) → dagens. Längsta prefixet vinner. SCREENING och Teknisk analys
# landar i Thor; REGIME och INTELLIGENCE i Odins visdom; GRANSKNING heter ARKEN.
LEGACY_PATHS: dict = {
    "screening": "thor",
    "regime/marknad": "odin/marknad",
    "regime/ravaror": "odin/ravaror",
    "regime/makro": "odin/makro",
    "regime": "odin",
    "intel/odins-blindspot": "odin/ravaror/odins-blindspot",
    "intel/sentiment": "odin/makro/sentiment",
    "intel/retail-pulse": "odin/marknad/retail-pulse",
    "intel/heatmap": "odin/marknad/heatmap",
    "intel/teknisk-analys": "thor",
    "intel": "odin",
    "intelligence": "intel",
    "granskning": "review",
}


# Målnycklar (designförslaget PR 1): larmen, Home och bryggorna pekar på ett MÅL, inte på en sökväg. Flyttar en
# sida ändras bara raden här — länkarna i Discord och knapparna i panelen leder rätt före och efter flytten.
TARGETS: dict = {
    "home": ("home",),
    "thor": ("thor",),
    "turn": ("valhall", "Kartan"),
    "valhall_map": ("valhall", "Kartan"),
    "laget": ("odin", "Marknad", "Läget"),
    "swing_regime": ("odin", "Marknad", "Läget"),
    "market_risk": ("odin", "Marknad", "🌩️ Marknadsrisk"),
    "rotation": ("odin", "Råvaror", "Råvarurotation"),
    "fiat": ("odin", "Makro", "🐺 Fiat Debasement"),
    "paper": ("review", "📊 Papperskonton"),
    "insider": ("review", "Insider"),
    "rule": ("review", "Rick Rule"),
    "royalty": ("review", "Royalty C"),
    "scoring": ("review", "Poängmodell"),
    "tiggre": ("review", "Tiggre"),
    "durrett": ("review", "🧭 Durrett"),
    "scorecard": ("review", "🎯 Scorecard"),
    "holdings": ("portfolio", "Holdings"),
    "journal": ("portfolio", "📓 Trade Journal"),
    "alerts": ("alerts",),
    "settings": ("rules", "⚙️ Inställningar"),
    "goal": ("portfolio", "🎯 Mot målet"),
    "backtests": ("valhall", "Backtester"),
}


def target_slug(target: str) -> str:
    """Målets sökväg som slug ('market_risk' → 'odin/marknad/marknadsrisk'); okänt mål → Home."""
    return slug(TARGETS.get(target) or TARGETS["home"])


def target_link(target: str, base: str = "", ticker: str = "", strategy: str = "") -> str:
    """Hela länken till ett mål: '<base>/?p=thor&t=BOL.ST&s=wolf'. Utan base bara '?p=…' (panelens egna knappar).
    t = aktien som öppnas, s = Thors strategi (thor.registry-nyckel)."""
    from urllib.parse import quote
    q = "?p=" + target_slug(target)
    if ticker:
        q += "&t=" + quote(str(ticker).strip(), safe=".-_")
    if strategy:
        q += "&s=" + quote(str(strategy).strip(), safe="_")
    base = str(base or "").strip().rstrip("/")
    return f"{base}/{q}" if base else q


def _legacy(link: str) -> str:
    """'regime/marknad/swing-regime' → 'odin/marknad/swing-regime', 'screening/…' → 'thor'."""
    s = "/".join(p for p in str(link or "").lower().split("/") if p)
    for _ in range(3):                                   # intelligence → intel → odin/…
        hit = next((old for old in sorted(LEGACY_PATHS, key=len, reverse=True)
                    if s == old or s.startswith(old + "/")), None)
        if hit is None:
            break
        new = LEGACY_PATHS[hit]
        s = new if new == "thor" else new + s[len(hit):]
    return s


def tab_label(key: str) -> str:
    """Toppflikens etikett som st.tabs får den (med luft runt)."""
    return f"  {TOP_LABEL[key]}  "


def options(path: str) -> list:
    """Alternativen på nästa nivå under sökvägen, eller [] om det är ett löv."""
    return list(SUBS.get(path, []))


# ── Djuplänkar: ?p=odin/marknad/arc-regime/wolf-regime ───────────────────────
_TRANS = str.maketrans({"å": "a", "ä": "a", "ö": "o", "é": "e"})


def slugify(label: str) -> str:
    """'🐺 Wolf Asymmetry' → 'wolf-asymmetry', 'Råvaror' → 'ravaror'."""
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
    if parts and _top_key(parts[0]) is None:
        parts = [p for p in _legacy(link).split("/") if p]
    if not parts:
        return None
    top = _top_key(parts[0])
    if top is None:
        return None
    segs, path = [top], top
    for p in parts[1:]:
        p = LEGACY_SLUGS.get(p, p)
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
    """'ODINS VISDOM → Marknad → Swing Regime' (FLIKGUIDENS form) → '?p=…', eller
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
    """Alla fullständiga sökvägar 'ODINS VISDOM → Marknad → Arc Regime → Wolf Regime'."""
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
