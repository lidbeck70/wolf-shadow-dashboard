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
👁 ODIN samlar 📡 REGIME och 👁 INTELLIGENCE (Råvaror · Makro · Marknad) och
📒 ARKEN är 🔬 GRANSKNING med samma innehåll. Gamla länkar landar rätt via LEGACY_PATHS.
"""

from __future__ import annotations

import re

TOP: tuple = (
    ("home", "🏠 I DAG"),
    ("odin", "👁 ODIN"),
    ("thor", "🔨 THOR"),
    ("review", "📒 ARKEN"),
    ("portfolio", "💼 PORTFÖLJ"),
    ("more", "⋯ MER"),
)
TOP_LABEL = dict(TOP)

# Designförslaget PR 9: sex toppflikar i beslutskedjans ordning — I DAG (morgonbeskedet), ODIN (var kapitalet
# ska), THOR (vilka bolag och hur stort), ARKEN (vilket som köps), PORTFÖLJ (positioner och målet), MER (larm,
# inställningar, regler, Copilot och motorsidorna). VALHALL-fliken är borta (beslut 1): Kartan ligger i ODIN,
# papperskontona i PORTFÖLJ och backtesterna under Mot målet. Nycklarna i koden är oförändrade (beslut 8).
SUBS: dict = {
    "odin": ["Läget", "Kartan", "Råvaror", "Makro", "Marknad"],
    # Design PR 10: Guld/Silver och Guldkvoter är en sida (🥇 Guld) med två inre flikar; Sentiment och Retail Pulse
    # likaså. Marknad är bakgrund och märks så på sidorna.
    "odin/Råvaror": ["Odin's Blindspot", "🔥 EMBER", "🪓 BERSERK Regime", "Råvarurotation", "🥇 Guld"],
    "odin/Makro": ["🐺 Fiat Debasement", "Sentiment"],
    "odin/Marknad": ["Sektorflöde", "Värmekarta"],
    # Beslutsunderlaget efter screeningen — köpgrinden först. Durrett har sitt ark (data/confidence.json);
    # Wolf Asymmetry har sitt eget med Confidence inbyggt.
    "review": ["🎯 Scorecard", "Rick Rule", "Royalty C", "Poängmodell", "Tiggre", "Insider",
               "🧭 Durrett", "🐺 Wolf Asymmetry", "🚀 Råvaruhävstång"],
    # Design PR 11: fliken heter som sidan och som larmet — Positioner, Journal, Veckorutin.
    "portfolio": ["Positioner", "🎯 Mot målet", "📊 Papperskonton", "Journal", "Veckorutin", "Allokering"],
    # Motorsidorna: Thors flikar gör samma jobb och bryggorna leder hit. Gamla Vikings kod rörs inte.
    "more": ["Larm", "⚙️ Inställningar", "Regler & Guider", "Copilot", "Motorsidor", "Egen backtest"],
    "more/Motorsidor": ["Wolf Regime", "Viking Regime", "⚔️ Viking Nine Regime", "Quality & Contrarian",
                        "Long Trend", "Market Cycle"],
    "more/Regler & Guider": ["🚀 KOM IGÅNG", "📋 HANDELSREGLER", "⚡ FUSKLAPP",
                             "🗺 FLIKGUIDE", "📖 STRATEGIGUIDER", "🗓 ÅRSHJULET",
                             "📚 SNABBREFERENS"],
}

# Flikguiden (startsidans expander): (toppnyckel, rubrik, [(flik, beskrivning)]).
HOME_ZONES: tuple = (
    ("odin", "ODIN — VAR KAPITALET SKA", [
        ("Läget", "Allokering · Swing · Marknadsrisk · Norden — fyra ljus, och Swing-läget"),
        ("Kartan", "Valhalls cykelkarta per sektor, råvarukomplex och tema med vändningarna"),
        ("Råvaror", "Odin's Blindspot · 🔥 EMBER · BERSERK · Råvarurotation · Guld"),
        ("Makro", "🐺 Fiat Debasement och Sentiment"),
        ("Marknad", "Sektorflöde och värmekarta — bakgrund"),
    ]),
    ("thor", "THOR — KANDIDATER, ANALYS OCH STORLEK", [
        ("🔨 THOR", "Alla strategiers kandidater, ⚡ Skanna nu, analysen före köp och Storlek med Logga köp"),
    ]),
    ("review", "ARKEN — VILKET BOLAG SOM KÖPS", [
        ("🎯 Scorecard", "Köpgrinden: sju kryss före ordern"),
        ("Rick Rule · Royalty C", "Producenter och royaltybolag mot kostnadskurvan"),
        ("Poängmodell · Tiggre · Insider", "Sprott, Durrett-snabbpoäng, Lobo-arket, insynsflödet"),
        ("🧭 Durrett", "Durretts 10 steg med Håven och eget ark"),
        ("🐺 Wolf Asymmetry", "Hävstång, margin of safety, confidence, stressmatris"),
    ]),
    ("portfolio", "PORTFÖLJ — POSITIONER OCH MÅLET", [
        ("Positioner", "Positionerna och säljreglerna"),
        ("🎯 Mot målet", "Årstakten mot 20/35 %, per strategi, och vad reglerna gett historiskt"),
        ("📊 Papperskonton", "Varje strategi på papper"),
        ("Journal", "Stängda affärer och R"),
        ("Allokering", "Ramar, tak och strömbrytaren"),
    ]),
    ("more", "MER — LARM, INSTÄLLNINGAR OCH REGLER", [
        ("Larm", "Larmloggen och kanalerna"),
        ("⚙️ Inställningar", "Kapital, låsta regler, positionskalkylator och datakällor"),
        ("Regler & Guider", "Handelsregler, fusklapp, flikguide, strategiguider, årshjulet"),
        ("Motorsidor", "Wolf, Viking, Viking Nine, Quality & Contrarian, Long Trend, Market Cycle"),
    ]),
)


# Widget-nyckeln i session_state för radion på varje nivå. wolf_panel ritar
# _sub(path) ur den här, och djuplänken seedar samma nycklar.
TABS_KEY = "main_tabs"
STATE_KEY: dict = {
    "odin": "sub_odin",
    "odin/Råvaror": "sub_odin_ravaror",
    "odin/Makro": "sub_odin_makro",
    "odin/Marknad": "sub_odin_marknad",
    "review": "sub_review",
    "portfolio": "sub_portfolio",
    "more": "sub_more",
    "more/Motorsidor": "sub_motor",
    "more/Regler & Guider": "rules_sub",          # ritas inne i rules_page
}

# Gamla djuplänkar som fortfarande ska landa rätt: slug → dagens slug.
# 'granskning/durrett-confidence/...' öppnade "🧭 Durrett & Confidence" innan
# Confidence-caset byggdes in i Wolf Asymmetry (steg F).
LEGACY_SLUGS: dict = {"durrett-confidence": "durrett"}

# Gamla sökvägar (slug-prefix) → dagens. Längsta prefixet vinner, och flera steg följs (regime → odin → …).
# Fas 8: SCREENING och Teknisk analys landar i Thor; REGIME och INTELLIGENCE i ODIN; GRANSKNING heter ARKEN.
# Design PR 9: VALHALL, ALERTS, RULES, STRATEGIES och COPILOT har flyttat (MER och PORTFÖLJ), Swing Regime och
# Marknadsrisk är Läget, Arc/Alpha Regime och Market Cycle är Motorsidor.
LEGACY_PATHS: dict = {
    "screening": "thor",
    "regime/marknad": "odin/marknad",
    "regime/ravaror": "odin/ravaror",
    "regime/makro": "odin/makro",
    "regime": "odin",
    "intel/odins-blindspot": "odin/ravaror/odins-blindspot",
    "intel/sentiment": "odin/makro/sentiment",
    "intel/retail-pulse": "odin/makro/sentiment",
    "intel/heatmap": "odin/marknad/heatmap",
    "intel/teknisk-analys": "thor",
    "intel": "odin",
    "intelligence": "intel",
    "granskning": "review",
    "odins-visdom": "odin",
    "valhall/kartan": "odin/kartan",
    "valhall/papperskonton": "portfolio/papperskonton",
    "valhall/backtester": "portfolio/mot-malet",
    "valhall": "odin/kartan",
    "review/papperskonton": "portfolio/papperskonton",
    "odin/marknad/laget": "odin/laget",
    "odin/marknad/swing-regime": "odin/laget",
    "odin/marknad/marknadsrisk": "odin/laget",
    "odin/marknad/market-cycle": "more/motorsidor/market-cycle",
    "odin/marknad/arc-regime": "more/motorsidor",
    "odin/marknad/alpha-regime": "more/motorsidor",
    "odin/marknad/retail-pulse": "odin/makro/sentiment",
    "odin/makro/retail-pulse": "odin/makro/sentiment",
    "odin/ravaror/guld-silver": "odin/ravaror/guld",
    "odin/ravaror/guldkvoter": "odin/ravaror/guld",
    "odin/ravaror/ember-regime": "odin/ravaror/ember",
    "odin/marknad/flow-divergence": "odin/marknad/sektorflode",
    "odin/marknad/heatmap": "odin/marknad/varmekarta",
    "portfolio/holdings": "portfolio/positioner",
    "portfolio/trade-journal": "portfolio/journal",
    "portfolio/swing": "portfolio/veckorutin",
    "alerts": "more/larm",
    "rules/regler-guider": "more/regler-guider",
    "rules/installningar": "more/installningar",
    "rules/position-sizing": "more/installningar",
    "rules/data-health": "more/installningar",
    "rules": "more/regler-guider",
    "strategies": "more/regler-guider/strategiguider",
    "copilot": "more/copilot",
    "portfolio/backtest": "more/egen-backtest",
}


# Målnycklar (designförslaget PR 1): larmen, startsidan och bryggorna pekar på ett MÅL, inte på en sökväg. Flyttar en
# sida ändras bara raden här — länkarna i Discord och knapparna i panelen leder rätt före och efter flytten.
TARGETS: dict = {
    "home": ("home",),
    "thor": ("thor",),
    "turn": ("home",),
    "valhall_map": ("odin", "Kartan"),
    "laget": ("odin", "Läget"),
    "swing_regime": ("odin", "Läget"),
    "market_risk": ("odin", "Läget"),
    "rotation": ("odin", "Råvaror", "Råvarurotation"),
    "fiat": ("odin", "Makro", "🐺 Fiat Debasement"),
    "paper": ("portfolio", "📊 Papperskonton"),
    "insider": ("review", "Insider"),
    "rule": ("review", "Rick Rule"),
    "royalty": ("review", "Royalty C"),
    "scoring": ("review", "Poängmodell"),
    "tiggre": ("review", "Tiggre"),
    "durrett": ("review", "🧭 Durrett"),
    "scorecard": ("review", "🎯 Scorecard"),
    "holdings": ("portfolio", "Positioner"),
    "journal": ("portfolio", "Journal"),
    "alerts": ("more", "Larm"),
    "settings": ("more", "⚙️ Inställningar"),
    "goal": ("portfolio", "🎯 Mot målet"),
    "backtests": ("portfolio", "🎯 Mot målet"),
}


# Gamla länkar till sidor som nu är en inre flik: slug → (widgetnyckel för st.tabs, flikens etikett).
INNER_SEEDS: dict = {
    "odin/ravaror/guldkvoter": ("odin_gold_tabs", "🥇 Guldkvoter"),
    "odin/makro/retail-pulse": ("odin_sentiment_tabs", "Retail"),
    "odin/marknad/retail-pulse": ("odin_sentiment_tabs", "Retail"),
    "intel/retail-pulse": ("odin_sentiment_tabs", "Retail"),
}


def inner_seed(link: str):
    """(nyckel, etikett) för den inre flik en gammal länk pekade på, längs hela kedjan av gamla sökvägar; annars None."""
    s = "/".join(p for p in str(link or "").lower().split("/") if p)
    for _ in range(6):
        if s in INNER_SEEDS:
            return INNER_SEEDS[s]
        hit = next((old for old in sorted(LEGACY_PATHS, key=len, reverse=True)
                    if s == old or s.startswith(old + "/")), None)
        if hit is None:
            return None
        new = LEGACY_PATHS[hit]
        s = new if new == "thor" else new + s[len(hit):]
    return None


def target_slug(target: str) -> str:
    """Målets sökväg som slug ('market_risk' → 'odin/marknad/marknadsrisk'); okänt mål → Home."""
    return slug(TARGETS.get(target) or TARGETS["home"])


def target_link(target: str, base: str = "", ticker: str = "", strategy: str = "") -> str:
    """Hela länken till ett mål: '<base>/?p=thor&t=BOL.ST&s=wolf'. Utan base bara '?p=…' (panelens egna knappar).
    t = aktien som öppnas, s = Thors strategi (thor.registry-nyckel)."""
    from urllib.parse import quote
    q = "?p=" + target_slug(target)
    if target == "thor" and ticker:                  # bara Thor läser aktien ur länken (granskning 2, D5)
        q += "&t=" + quote(str(ticker).strip(), safe=".-_")
    if target == "thor" and strategy:
        q += "&s=" + quote(str(strategy).strip(), safe="_")
    if target == "market_risk":                      # Läget slår på Marknadsrisken (D6)
        q += "&risk=1"
    base = str(base or "").strip().rstrip("/")
    return f"{base}/{q}" if base else q


def _legacy(link: str) -> str:
    """'regime/marknad/swing-regime' → 'odin/marknad/swing-regime', 'screening/…' → 'thor'."""
    s = "/".join(p for p in str(link or "").lower().split("/") if p)
    for _ in range(6):                                   # intelligence → intel → odin/marknad/… → odin/makro/…
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
    parts = [p for p in _legacy(link).split("/") if p]       # gamla sökvägar, även inne i dagens flikar
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
    """'ODIN → Läget' (FLIKGUIDENS form) → '?p=…', eller
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
    """Alla fullständiga sökvägar 'MER → Motorsidor → Wolf Regime'."""
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
