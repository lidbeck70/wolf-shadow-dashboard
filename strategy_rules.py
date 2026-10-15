"""
strategy_rules.py — Single source of truth for every strategy's playbook.

Both the RULES tab (ovtlyr/ui/rules_page.py) and the STRATEGIES tab
(tabs/strategy_overview.py) read from here, so a number can never drift between
"what the panel says" and "what the panel does" again. Before this module the
Wolf risk-per-trade was stated as 1 %, 2 % and 5 % in three different places.

Each Playbook carries everything needed to *learn* the strategy, not just run it:
  idea      — what it is, in plain Swedish
  level     — Nybörjare / Medel / Avancerad
  risk      — the numbers, written out explicitly
  entry/exit— the rules, each with a panel_guide ("where do I look?")
  workflow  — the routine, step by step
  cheatsheet— one-screen quick reference
  pitfalls  — the mistakes that actually cost money
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Palette (shared with the UI modules) ─────────────────────────────────────
CYAN, GREEN, AMBER, RED, PURPLE, EMBER = (
    "#00E5FF", "#2d8a4e", "#d4943a", "#c44545", "#B400FF", "#FF6B3D")

LEVEL_BEGINNER = "Nybörjare"
LEVEL_MEDIUM = "Medel"
LEVEL_ADVANCED = "Avancerad"


@dataclass(frozen=True)
class Rule:
    """One rule, with the panel location that answers 'where do I look?'."""
    number: int
    text: str
    explanation: str
    panel_guide: str
    hard: bool = False        # hard rule — never broken, no judgement call

    def as_dict(self) -> dict:
        """Backwards-compatible shape for the existing card renderer."""
        return {"number": self.number, "text": self.text,
                "explanation": self.explanation, "panel_guide": self.panel_guide}


@dataclass(frozen=True)
class RiskModel:
    risk_per_trade: str
    position_size: str
    max_positions: str
    stop: str
    targets: str = "—"


# How much of a strategy the panel actually implements. Stated per playbook so
# the gaps are named rather than discovered mid-trade.
SUPPORT_FULL = "I panelen"          # screener/gating implemented
SUPPORT_PARTIAL = "Delvis"          # some support, manual steps remain
SUPPORT_MANUAL = "Utanför panelen"  # documented here, executed in Excel/Börsdata

SUPPORT_COLOR = {SUPPORT_FULL: GREEN, SUPPORT_PARTIAL: AMBER, SUPPORT_MANUAL: "#6B7280"}


@dataclass(frozen=True)
class Playbook:
    key: str
    name: str
    tagline: str
    color: str
    level: str
    horizon: str
    universe: str
    where: str                       # where in the panel it lives
    idea: str                        # plain-Swedish explanation
    risk: RiskModel
    entry: tuple[Rule, ...] = ()
    exit: tuple[Rule, ...] = ()
    mindset: tuple[Rule, ...] = ()
    workflow: tuple[str, ...] = ()
    cheatsheet: tuple[tuple[str, str], ...] = ()
    pitfalls: tuple[str, ...] = ()
    note: str = ""
    support: str = SUPPORT_FULL      # what the panel actually does for you
    support_note: str = ""           # exactly what is and isn't implemented
    source: str = "Panelen"          # where the rules come from


def _rules(items: list[tuple]) -> tuple[Rule, ...]:
    """(text, explanation, panel_guide[, hard]) -> numbered Rule tuple."""
    out = []
    for i, it in enumerate(items, start=1):
        text, expl, guide = it[0], it[1], it[2]
        hard = it[3] if len(it) > 3 else False
        out.append(Rule(i, text, expl, guide, hard))
    return tuple(out)


# ═════════════════════════════════════════════════════════════════════════════
#  1. MOMENTUM SWING  (veckorutinen — Swing / Swing Screener / Swing Regime)
# ═════════════════════════════════════════════════════════════════════════════

MOMENTUM = Playbook(
    key="momentum",
    name="Momentum Swing",
    tagline="Köp det som redan stiger — sälj mekaniskt när det slutar",
    color=GREEN,
    level=LEVEL_BEGINNER,
    horizon="Swing — veckor till månader",
    universe="Svenska bolag som klarat Börsdata-screenerns kvalitetsfilter",
    where="PORTFOLIO → Swing · Swing Screener  ·  REGIME → Swing Regime",
    idea=(
        "Den enklaste strategin i panelen och den bästa att börja med. Du gör tre "
        "saker: (1) kollar om marknaden överhuvudtaget tillåter köp, (2) tar de "
        "starkaste bolagen från en färdig momentum-ranking, (3) säljer enligt tre "
        "mekaniska regler. Inga diagram att tolka, inga mönster att lära sig — "
        "panelen räknar allt. Rutinen görs en gång i veckan och tar ~15 minuter."
    ),
    risk=RiskModel(
        risk_per_trade="≈1,2–2 % av kapitalet (12–20 % position × 10 % stop)",
        position_size="12–20 % av swing-kapitalet vid GRÖN regim · halv storlek vid GUL "
                      "· aldrig över 6 % av totalportföljen (allocator.py: hard_cap = 6.0)",
        max_positions="6–8 st (hårt tak: 8)",
        stop="−10 % från entry · flyttas till entry vid +20 %",
        targets="Inget vinstmål — säljreglerna avgör. Halva säljs vid +20 %",
    ),
    entry=_rules([
        ("Marknadsfiltret måste vara grönt",
         "OMXSPI över MA200. Är index under MA200 tas INGA nya köp — bara exits.",
         "REGIME → Swing Regime: trafikljuset. RÖD = köpknapparna låses automatiskt "
         "i Swing-fliken. Du kan alltså inte råka bryta regeln.", True),
        ("Bolaget ska ligga i topp 20 på rankingen",
         "Momentum-score = 50/50 på 3- och 6-månadersavkastning. Topp 20 = köpbara. "
         "För att kvala krävs dessutom pris över MA200, positiv 3-månaders och "
         "minst +10 % på 6 månader (wolf_data.py: MOM_LONG_MIN).",
         "SCREENING → Swing Screener: rad 1–20 är köpbara. Rad 21–40 visas dämpade "
         "— de är rank-exit-gränsen för innehav, inte köpkandidater."),
        ("Det krävs en setup — A eller B",
         "Setup A = pullback till MA20/50 med RSI 35–55. Setup B = utbrott, inom 3 % "
         "av 52-veckorshögsta. Ingen setup = ingen affär, hur bra bolaget än ser ut.",
         "SCREENING → Swing Screener: grön 'A'-flagga eller blå 'B?'-flagga. "
         "Köpknappen i Swing-fliken är låst tills du valt setup A eller B.", True),
        ("Max 1–2 nya köp per vecka",
         "Skyddar mot att du fyller portföljen på en enda vecka och därmed satsar "
         "allt på ett enda marknadsläge.",
         "PORTFOLIO → Swing: veckochecklistan punkt 4 påminner dig. Räkna dina köp "
         "i journalen innan du lägger nästa order."),
        ("Max 8 positioner — mål 6–8",
         "Färre positioner = du hinner följa dem. Fler = du tappar överblick.",
         "PORTFOLIO → Swing: Positioner visar 'x/6–8'. Panelen vägrar lägga till "
         "en nionde position.", True),
        ("Positionsstorlek 12–20 % — halv vid GUL regim",
         "GUL regim (index över MA200 men bredden < 45 % eller marginalen < 2 %) "
         "betyder att marknaden smalnar. Då halveras storleken.",
         "REGIME → Swing Regime: regelverket under trafikljuset visar exakt vilken "
         "storlek som gäller just nu."),
    ]),
    exit=_rules([
        ("Säljregel 1 — stängning under MA50",
         "Bolaget har tappat sin trend. Sälj hela positionen.",
         "PORTFOLIO → Swing: kryssa 'Stängt under MA50' på positionskortet — kortet "
         "blir rött med 'SÄLJ — regel utlöst'.", True),
        ("Säljregel 2 — stop på −10 %",
         "Hård stop från entry. Räknas automatiskt, ingen tolkning.",
         "PORTFOLIO → Swing: 'Stop (−10 %)' på varje positionskort. Kursen når stop "
         "→ kortet blir rött automatiskt. Lägg alltid stopen hos mäklaren också.", True),
        ("Säljregel 3 — ur topp 40 på rankingen",
         "Momentum har lämnat bolaget även om priset inte fallit än. Den regel som "
         "oftast glöms — och den som räddar dig från långsamma förlorare.",
         "SCREENING → Swing Screener: syns bolaget inte längre i listan (rad 1–40)? "
         "Kryssa då 'Ur topp 40' på positionskortet i Swing-fliken.", True),
        ("Vid +20 % — sälj halva, flytta stop till entry",
         "Du säkrar vinst och gör resten riskfri. Resten får löpa utan vinstmål.",
         "PORTFOLIO → Swing: en gul '+20 % — sälj halva'-flagga dyker upp på kortet. "
         "Kryssa 'Halva såld' när du gjort det."),
        ("Röd regim — inga nya köp, hantera bara exits",
         "Befintliga innehav säljs enligt regel 1–3 som vanligt. Du slutar bara köpa.",
         "REGIME → Swing Regime: rött trafikljus. Köpknapparna låses i Swing-fliken."),
    ]),
    workflow=(
        "Datan uppdateras automatiskt varje handelsdag kl 08:00 (GitHub Action) — "
        "du behöver inte köra något själv.",
        "REGIME → Swing Regime: får jag köpa den här veckan? Grön = full storlek, "
        "gul = halv storlek, röd = inga köp.",
        "SCREENING → Swing Screener: vad ska jag köpa? Titta på topp 20 och leta "
        "A/B-flaggor. Klicka '→ Bevakning' på kandidaterna.",
        "PORTFOLIO → Swing: gå igenom ALLA befintliga positioner mot säljregel 1–3 "
        "innan du köper något nytt.",
        "PORTFOLIO → Swing: välj setup (A/B) på bevakningen, sätt entry-kurs och "
        "klicka KÖP. Max 1–2 st.",
        "Lägg stopen hos mäklaren direkt, och logga affären i journalen. "
        "Bocka av veckochecklistan.",
    ),
    cheatsheet=(
        ("Marknadsfilter", "OMXSPI > MA200 — annars inga nya köp"),
        ("Setup A", "Pullback till MA20/50, RSI 35–55"),
        ("Setup B", "Utbrott — inom 3 % av 52v-högsta"),
        ("Positionsstorlek", "12–20 % (GRÖN) · halv (GUL)"),
        ("Stop", "−10 % från entry"),
        ("Halvsälj", "+20 % → sälj halva, stop till entry"),
        ("Säljregler", "MA50-brott · stop −10 % · ur topp 40"),
        ("Antal positioner", "6–8 (max 8)"),
        ("Nya köp", "Max 1–2 per vecka"),
        ("Normal vinstandel", "40–55 % — lägre än du tror är helt normalt"),
        ("Payoff-kvot", "Mål > 2,0 (vinsterna ska vara större än förlusterna)"),
        ("Utvärdering", "Under 15–20 affärer: dra INGA slutsatser"),
    ),
    pitfalls=(
        "Köpa i röd regim för att 'det här bolaget är ju så bra' — regeln finns just "
        "för att den känslan är som starkast precis när den är som farligast.",
        "Hoppa över säljregel 3 (ur topp 40). Priset har inte fallit, så det känns "
        "onödigt — men det är så du fastnar i döda innehav i månader.",
        "Fler än 1–2 köp på en vecka när allt ser bra ut. Då har du råkat satsa hela "
        "portföljen på ett enda marknadsläge.",
        "Flytta stopen nedåt för att 'ge det lite mer utrymme'. Stopen flyttas bara "
        "uppåt (till entry vid +20 %), aldrig nedåt.",
        "Dra slutsatser efter 5 affärer. Med 40–55 % vinstandel är fyra förluster i "
        "rad helt väntat och säger ingenting om strategin.",
    ),
    support=SUPPORT_FULL,
    support_note="Screener, regim och positionshantering körs helt i panelen.",
    source="Masterguiden Del 4, Strategi 6 (Momentum-swing)",
    note="Rankingen förutsätter att universumet redan klarat Börsdata-screenerns "
         "kvalitetsfilter (börsvärde, F-score). F-score går inte att räkna ur prisdata "
         "— kör Börsdata-screenern som vanligt och använd panelen som ranking-lager.",
)


# ═════════════════════════════════════════════════════════════════════════════
#  2. WOLF  (EMA + Ichimoku swing)
# ═════════════════════════════════════════════════════════════════════════════

WOLF = Playbook(
    key="wolf",
    name="Wolf x Shadow",
    tagline="Taktisk swing med EMA-stack, Ichimoku-exit och strukturbaserad stop",
    color=CYAN,
    level=LEVEL_ADVANCED,
    horizon="Swing — dagar till veckor",
    universe="Nordiska + amerikanska aktier",
    where="REGIME → Arc Regime → Wolf Regime  ·  SCREENING → Arc Screener → Wolf",
    idea=(
        "Den mest handpåläggande strategin: du bedömer trend, key levels, pullback "
        "och candlestick-trigger själv — panelen räknar indikatorerna men du fattar "
        "beslutet. Kräver att du kan läsa en graf. Börja inte här; börja med "
        "Momentum Swing och gå hit när du vill ha fler och snabbare affärer."
    ),
    risk=RiskModel(
        risk_per_trade="2 % av kapitalet (strategies/wolf.py: risk_pct = 0.02)",
        position_size="Risk ÷ stopavstånd (2,5 × ATR14)",
        max_positions="Max 2 förluster per dag — sedan stängs dagen",
        stop="2,5 × ATR14 från entry · SL till BE först efter ny HH/LL",
        targets="TP1 @ 2,6R (13 % av core) · TP2 @ 5,2R (17 %) · min R:R 1:2, helst 1:3",
    ),
    entry=_rules([
        ("Handla endast i trendens riktning",
         "Upptrend = long. Nedtrend = short. Aldrig mot trenden.",
         "REGIME → Arc Regime → Wolf Regime: Regime Score visar trendriktning. Grön badge = long. Röd = stå utanför.", True),
        ("Ta inga trades i konsolidering",
         "Range = förbjudet område. Vänta på breakout.",
         "REGIME → Arc Regime → Wolf Regime: Entry Checklist: RSI + ATR ratio visar konsolidering. Gate #2 = auto-check."),
        ("En trade kräver en key level",
         "Supply/demand eller tydligt stöd/motstånd.",
         "REGIME → Arc Regime → Wolf Regime: Entry Checklist: Order Blocks-kortet visar bullish/bearish OBs + närmaste nivå. Gate #3 = OB inom 3%."),
        ("Entry endast efter pullback",
         "Inga impulsiva entries i rakt fall eller rally.",
         "REGIME → Arc Regime → Wolf Regime: Entry Checklist: Trend-kortet visar EMA10/20 nivåer. Gate #4 = pris inom 2% av EMA10/20."),
        ("Candlestick-trigger krävs",
         "Pinbar, engulfing eller break-and-retest.",
         "REGIME → Arc Regime → Wolf Regime: Entry Checklist: Candlestick-kortet visar detekterade mönster (Hammer, Engulfing, Doji). Gate #5 = auto-check."),
        ("Volym måste bekräfta rörelsen",
         "Ingen volym = ingen trade.",
         "REGIME → Arc Regime → Wolf Regime: Entry Checklist: Momentum-kortet visar Vol ratio. Gate #6 = ≥ 1.0x."),
        ("Minsta R:R är 1:2",
         "Helst 1:3. Aldrig under 1:2.",
         "REGIME → Arc Regime → Wolf Regime: Entry Checklist: Volatilitet-kortet visar ATR. OB-kortet visar target. Gate #7 = R:R auto-beräknad.", True),
    ]),
    exit=_rules([
        ("Max 2 % risk per trade",
         "Stop sätts på struktur (2,5 × ATR14) — positionsstorleken anpassas så att "
         "förlusten blir 2 % av kapitalet om stopen träffas.",
         "REGIME → Arc Regime → Wolf Regime: SL/TP Calculator + Entry Checklist: ATR-värde i Volatilitet-kortet. "
         "Gate #8 visar SL-avstånd i %.", True),
        ("Flytta SL till BE först efter ny HH/LL",
         "Inte tidigare, inte senare.",
         "REGIME → Arc Regime → Wolf Regime: Ichimoku-gaugen visar prisnivåer. När pris gör ny swing high ovanför din entry → flytta SL till entry."),
        ("Max två förluster per dag",
         "Stoppa dagen direkt efter två minus.",
         "Egen disciplin. Två förluster = stäng plattformen. Nordic Alpha analyserar — du handlar nästa dag.", True),
        ("Exit: Kijun-sen trail + 2,5 × ATR hård stop",
         "Kijun-sen (26p) som dynamiskt trailing stop. Stäng om pris stänger under Kijun OCH under EMA 10. "
         "Den hårda stopen är 2,5 × ATR14 från entry (strategies/wolf.py: atr_stop_mult = 2.5).",
         "REGIME → Arc Regime → Wolf Regime: Ichimoku-gaugen visar Kijun-nivå. Entry Checklist: Trend-kortet visar EMA10. Gate #11 visar båda nivåer."),
    ]),
    workflow=(
        "REGIME → Wolf Regime: kolla Regime Score. Röd = handla inte alls idag.",
        "Gå igenom Entry Checklist uppifrån och ner — alla gates ska vara gröna.",
        "Hitta key level + vänta in pullback och candlestick-trigger.",
        "Räkna R:R i SL/TP Calculator. Under 1:2 → skippa affären.",
        "Lägg order + stop samtidigt. Logga i journalen.",
        "Efter ny HH/LL: flytta stop till break-even. Två förluster på en dag = sluta.",
    ),
    cheatsheet=(
        ("Trend", "EMA10 > 21 > 50 > 200 (full bull stack)"),
        ("Filter", "RSI(14) 45–70 · ADX(14) ≥ 19"),
        ("Stop", "2,5 × ATR14"),
        ("Risk/trade", "2 % av kapitalet"),
        ("R:R", "Minst 1:2, helst 1:3"),
        ("Trail", "Kijun-sen + EMA10"),
        ("Dagsstopp", "2 förluster → stäng dagen"),
    ),
    pitfalls=(
        "Ta en trade utan candlestick-trigger för att 'nivån är så tydlig'.",
        "Handla i konsolidering — de flesta förluster kommer därifrån.",
        "Fortsätta efter två förluster samma dag. Regeln finns för att du är som "
        "sämst på beslut just då.",
        "Flytta stop till break-even för tidigt (innan ny HH/LL) och bli utstoppad "
        "i normalt brus.",
    ),
    support=SUPPORT_FULL,
    support_note="Wolf Regime kör Entry Checklist med auto-pass/fail per gate.",
    source="Panelen"
)


# ═════════════════════════════════════════════════════════════════════════════
#  3. ALPHA  (regim/trend — långsiktig)
# ═════════════════════════════════════════════════════════════════════════════

ALPHA = Playbook(
    key="alpha",
    name="Alpha Trend / Regim",
    tagline="Långsiktiga innehav som ägs så länge regimen är grön",
    color=GREEN,
    level=LEVEL_MEDIUM,
    horizon="Position — 0,5–3 år",
    universe="Nordiska + amerikanska large-caps",
    where="REGIME → Alpha Regime  ·  SCREENING → Contrarian Alpha → Long Screener",
    idea=(
        "Den lugnaste strategin. Du äger 8–10 bolag så länge marknadsregimen är "
        "grön och trenden intakt, och minskar eller säljer när regimen vänder. "
        "Få beslut, långa innehav — passar dig som inte vill titta på panelen varje "
        "dag men vill ha ett tydligt regelverk för när man kliver av."
    ),
    risk=RiskModel(
        risk_per_trade="1,5 % av kapitalet (strategies/alpha.py: risk_pct = 0.015)",
        position_size="Max 10 % per aktie · max 20–25 % per sektor",
        max_positions="8–10 innehav (koncentrerad portfölj)",
        stop="Pris − EMA200 · EMA200-brott = minska 50 %, röd regim = sälj resten",
        targets="TP1 @ 3R (30 % av core) · TP2 @ 6R (30 %)",
    ),
    entry=_rules([
        ("Köp endast i grön regim",
         "Regimindikatorn måste vara grön.",
         "REGIME → Alpha Regime → Quality & Contrarian: Regime-badge visar GRÖN/ORANGE/RÖD. Gates 1-7 måste passera. Alla gröna = OK att köpa.", True),
        ("Pris måste ligga över 200 EMA",
         "Bekräftar långsiktig upptrend.",
         "REGIME → Alpha Regime → Quality & Contrarian: Gate #2 visar 'Pris vs EMA200' med exakt avstånd i %. REGIME → Arc Regime → Viking Regime: Magenta-linjen (EMA 200) i grafen.", True),
        ("50 EMA måste ligga över 200 EMA",
         "Golden cross = positivt momentum.",
         "REGIME → Alpha Regime → Quality & Contrarian: Gate #3 visar 'Golden Cross' eller 'Death Cross'. REGIME → Arc Regime → Viking Regime: Gul (EMA 50) ovanför magenta (EMA 200)."),
        ("Sektorn måste vara grön",
         "Ingen exponering i svaga sektorer.",
         "REGIME → Alpha Regime → Quality & Contrarian: Gate #4 visar sektorstatus (0-3). REGIME → Flow Divergence: Sektorhjulet visar grön/gul/röd per sektor."),
        ("Fear & Greed under 60 vid köp",
         "Undvik eufori och toppjakt.",
         "REGIME → Alpha Regime → Quality & Contrarian: Gate #5 visar F&G-score + OK/EJ OK. INTELLIGENCE → Sentiment: Stor F&G-gauge. Under 60 = OK. Över 60 = vänta med köp."),
        ("Max 10 % per aktie",
         "Ingen enskild position får dominera.",
         "SCREENING → Contrarian Alpha → Long Screener: högsta ranking = allokera max 10%. Lägre ranking = allokera max 7%. Aldrig mer oavsett hur bra det ser ut.", True),
        ("Max 20–25 % per sektor",
         "Riskkontroll på portföljnivå.",
         "SCREENING → Contrarian Alpha → Long Screener: sortera efter sektor. Räkna: hur mycket äger du redan i samma sektor? Max 25%.", True),
    ]),
    exit=_rules([
        ("Minska vid EMA200-brott, sälj vid regimskifte",
         "EMA200-brott = reducera 50%. Regim röd = sälj resten.",
         "REGIME → Alpha Regime → Quality & Contrarian: Gate #6 visar dagar under EMA200. Gate #7 visar regime-färg. Orange = reducera halvt. Röd = sälj allt.", True),
        ("Sälj vid sektor + breadth crossover",
         "Sektor OCH marknadsbreadd vänder ner = sälj sektorpositioner.",
         "REGIME → Flow Divergence: Sektorhjulet: om sektor byter från grön till röd. REGIME → Arc Regime → Viking Regime: VIKING NINE sektorpoäng under 30."),
        ("Analysera alltid historiska nedgångar",
         "Avgör om fallet är brus eller strukturellt.",
         "REGIME → Arc Regime → Viking Regime: Drawdowns sub-tab klassificerar nedgångar. PORTFOLIO → Backtest: Alpha mode: historisk prestation med drawdown-analys."),
    ]),
    workflow=(
        "REGIME → Alpha Regime: är regimen grön? Röd/orange = inga nya köp.",
        "SCREENING → Long Screener: leta kandidater med hög composite-score.",
        "Kontrollera gates 1–7 för bolaget innan köp.",
        "Räkna position: max 10 % per aktie, max 25 % per sektor.",
        "Kvartalsvis: gå igenom innehaven mot exit-reglerna. Röd regim = sälj.",
    ),
    cheatsheet=(
        ("Regim", "Måste vara GRÖN för nya köp"),
        ("Trend", "Pris > EMA200 · EMA50 > EMA200"),
        ("Sentiment", "F&G under 60 vid köp"),
        ("Risk/trade", "1,5 % av kapitalet"),
        ("Max per aktie", "10 %"),
        ("Max per sektor", "20–25 %"),
        ("Antal innehav", "8–10"),
        ("Exit", "EMA200-brott = −50 % · röd regim = sälj allt"),
    ),
    pitfalls=(
        "Köpa när Fear & Greed är över 60 — du köper då toppen av eufori.",
        "Låta ett vinnande innehav växa förbi 10 % av portföljen utan att trimma.",
        "Sitta kvar genom en röd regim med hoppet att 'det vänder snart'.",
        "Samla flera innehav i samma sektor utan att räkna sektorexponeringen.",
    ),
    support=SUPPORT_FULL,
    support_note="Alpha Regime kör alla gates; Long Screener ger kandidater.",
    source="Panelen"
)


# ═════════════════════════════════════════════════════════════════════════════
#  4. VIKING  (OVTLYR Golden Ticket)
# ═════════════════════════════════════════════════════════════════════════════

VIKING = Playbook(
    key="viking",
    name="Viking (OVTLYR Golden Ticket)",
    tagline="10 entry-krav, 10 exit-triggers — allt eller inget",
    color=PURPLE,
    level=LEVEL_ADVANCED,
    horizon="Swing — dagar till veckor",
    universe="Nordiska + amerikanska aktier",
    where="REGIME → Arc Regime → Viking Regime  ·  SCREENING → Arc Screener → Viking",
    idea=(
        "Den mest regelstyrda strategin: tio krav måste vara uppfyllda samtidigt "
        "för entry — marknad, sektor och aktie ska alla peka åt samma håll. Det gör "
        "att du tar få affärer, men de du tar har hela marknaden i ryggen. Exit är "
        "lika mekanisk: tio triggers, varav den första ($SPY under EMA20) stänger allt."
    ),
    risk=RiskModel(
        risk_per_trade="1,5 % av kapitalet (strategies/viking.py: risk_pct = 0.015)",
        position_size="Risk ÷ stopavstånd (1,5 × ATR14)",
        max_positions="Följ portföljens allmänna tak",
        stop="1,5 × ATR14 från entry · EMA10 som trailing stop",
        targets="TP1 @ 2R (25 % av core) · TP2 @ 4R (25 %) · F&G-target enligt exit-regel 9",
    ),
    entry=_rules([
        ("Market Trend: SPY 10EMA > 20EMA, Price > 50EMA", "Bullish = buy zone. Bearish = inga trades.",
         "REGIME → Arc Regime → Viking Regime: Trend-kort visar 'Direction: Bullish/Bearish'. Regime-badge = GRÖN krävs.", True),
        ("Market Signal: Köpsignal på $SPY", "Viking overlay måste vara grön.",
         "REGIME → Arc Regime → Viking Regime: Välj SPY som ticker. Long-term signal = 'BUY'. Regime = GRÖN."),
        ("Market Breadth: Bull List bullish crossover", "Måste matcha market trend.",
         "REGIME → Arc Regime → Viking Regime: Bull List % gauge (Advanced Analysis). Under 25 + vänder upp = bästa entry. Över 75 + vänder ner = stopp."),
        ("Sector Breadth: Stigande", "Bullish 10EMA-kors krävs.",
         "REGIME → Flow Divergence: Sektorhjulet: sektorn måste vara grön. Trend Distribution: sektorn i 'Uptrend'."),
        ("Sector Fear & Greed: Stigande", "Sektorsentiment måste förbättras.",
         "INTELLIGENCE → Sentiment: F&G-gauge stigande. REGIME → Arc Regime → Viking Regime: Sentiment-kort: score stigande (jämför med förra veckan)."),
        ("Stock Signal: Köp", "Viking signal måste visa Buy.",
         "REGIME → Arc Regime → Viking Regime: Long-term signal badge visar 'BUY' (grön). Score > 60 krävs."),
        ("Stock Trend: 10EMA/20EMA, Price > 50EMA", "Alla EMA:er alignade.",
         "REGIME → Arc Regime → Viking Regime: Grafen: vit (10) > orange (20) > gul (50). Alla stigande. Pris ovanför alla tre."),
        ("Stock Fear & Greed: Stigande", "Aktiens sentiment förbättras.",
         "REGIME → Arc Regime → Viking Regime: Oscillator Direction visar 'Rising' + timing 'Early' eller 'Mid'. Inte 'Exhausted'."),
        ("Order Blocks: Inga restriktiva OBs", "Inga bearish OBs blockerar vägen uppåt.",
         "REGIME → Arc Regime → Viking Regime: Grafen: inga röda OB-zoner ovanför nuvarande pris. Order Blocks-tab: inga aktiva bearish OBs nära."),
        ("Momentum: Pris ovanför gårdagens lägsta", "Bekräftar positivt momentum.",
         "REGIME → Arc Regime → Viking Regime: Grafen: dagens candle stänger ovanför gårdagens lägsta nivå. Momentum-kort: RSI > 50."),
    ]),
    exit=_rules([
        ("$SPY stänger under 20 EMA → STÄNG ALLT", "Hård exit. Inga undantag.",
         "REGIME → Arc Regime → Viking Regime: Välj SPY. Om pris under orange linje (EMA 20) = stäng alla positioner omedelbart.", True),
        ("1,5 × ATR Stop Loss från entry", "Strukturbaserad stop, aldrig %. Höjd från ½ ATR för att ge priset utrymme (strategies/viking.py: atr_stop_mult = 1.5).",
         "REGIME → Arc Regime → Viking Regime: Volatility-kort: ATR 14 värde. SL = entry-pris minus (1,5 × ATR).", True),
        ("10 EMA Trailing Stop", "Pris stänger under 10 EMA = exit.",
         "REGIME → Arc Regime → Viking Regime: Grafen: vit linje = EMA 10. Om candle stänger under den vita linjen = exit."),
        ("Order Block hit", "Pris springer in i restriktivt OB = exit.",
         "REGIME → Arc Regime → Viking Regime: Grafen: om pris rör sig in i en röd OB-zon (bearish) = stäng positionen."),
        ("Gap & Crap", "Gap up följt av reversal = omedelbar exit.",
         "REGIME → Arc Regime → Viking Regime: Grafen: om dagens öppning gappar upp men sedan faller tillbaka under gårdagens stängning = exit direkt."),
        ("Stängning under gårdagens lägsta", "Efter att du redan rullat (moved SL) = exit.",
         "REGIME → Arc Regime → Viking Regime: Grafen: jämför dagens stängning med gårdagens lägsta. Stänger under = exit."),
        ("Sektor + Market breadth crossover", "Sälj alla trades i den sektorn.",
         "REGIME → Flow Divergence: Om sektorn byter från grön till röd, OCH Bull List % vänder ner = sälj alla positioner i sektorn."),
        ("Stock Sell signal", "Viking signal flippar till Sell.",
         "REGIME → Arc Regime → Viking Regime: Long-term signal badge byter till 'SELL' (röd) eller 'REDUCE' (magenta)."),
        ("Fear & Greed target hit", "Beror på var du köpte: 0-50 = exit vid 63. 50-75 = 10p spread. 75+ = 5p spread.",
         "INTELLIGENCE → Sentiment: F&G-gauge. Notera ditt entry-F&G-värde. Räkna target: entry + spread. Exit när target nås."),
        ("Earnings risk", "Stäng position minst 1 vecka före rapportdag.",
         "Kolla rapportdatum externt (t.ex. Börsdata, Yahoo Finance). Stäng senast 5 handelsdagar innan rapport.", True),
    ]),
    mindset=_rules([
        ("Det finns INGA FÖRVÄNTNINGAR på utfallet", "Handla planen, inte prediktionen.",
         "Alla flikar i Nordic Alpha visar DATA, inte åsikter. Följ signalerna — känn ingenting."),
        ("Det finns INGA VINSTMÅL", "Låt exit-signalerna göra sitt jobb.",
         "Sätt aldrig en TP-order baserat på känsla. Använd trailing stop (EMA 10) eller exit-signal."),
        ("Jag har bara en plan att ta mig ur", "Planen ÄR din edge. Exekveringen är allt.",
         "RULES-fliken (denna sida) = din plan. Läs igenom före varje handelsdag. Inga avvikelser."),
    ]),
    workflow=(
        "REGIME → Viking Regime: välj SPY. Är market trend bullish och signalen BUY?",
        "Kolla marknadsbredden (Bull List %) och sektorn — båda ska stiga.",
        "Först därefter: analysera aktien mot krav 6–10.",
        "Alla tio kraven gröna? Först då är det en affär.",
        "Sätt 1,5 × ATR-stop direkt. Trail med EMA10.",
        "Kolla rapportdatum — stäng senast 5 handelsdagar före.",
    ),
    cheatsheet=(
        ("Entry", "Alla 10 krav måste vara gröna — inga undantag"),
        ("Marknad först", "SPY bullish + BUY-signal + breadth upp"),
        ("Stop", "1,5 × ATR14"),
        ("Risk/trade", "1,5 % av kapitalet"),
        ("Trail", "EMA10 — stängning under = exit"),
        ("Hård exit", "$SPY under EMA20 = stäng ALLT"),
        ("Rapport", "Stäng 5 handelsdagar före"),
    ),
    pitfalls=(
        "Ta affären när 8 av 10 krav är uppfyllda. Strategin bygger på att alla tio är det.",
        "Ignorera $SPY-exiten för att den enskilda aktien 'ser stark ut'.",
        "Sätta ett eget vinstmål istället för att låta trailing-stopen jobba.",
        "Glömma rapportdatum och sitta med positionen över rapporten.",
    ),
    support=SUPPORT_FULL,
    support_note="Viking Regime + Arc Screener kör hela regelverket.",
    source="Panelen"
)


# ═════════════════════════════════════════════════════════════════════════════
#  5. QUALITY  (Buffett / Kvalitetsaktiepodden)
# ═════════════════════════════════════════════════════════════════════════════

QUALITY = Playbook(
    key="quality",
    name="Quality",
    tagline="Välskötta bolag till rimliga priser — ägda länge",
    color=CYAN,
    level=LEVEL_MEDIUM,
    horizon="Position — 0,5–3 år",
    universe="Nordiska + amerikanska kvalitetsbolag",
    where="SCREENING → Contrarian Alpha (quality-läge)  ·  REGIME → Alpha Regime → Quality",
    idea=(
        "Inspirerad av Warren Buffett och Kvalitetsaktiepodden: hitta välskötta bolag "
        "med stark ekonomi, köp dem till rimliga priser när marknadsläget är gynnsamt "
        "och håll dem länge. 'Wonderful companies at fair prices' kombinerat med krav "
        "på hög lönsamhet, stabila marginaler och låg skuldsättning. Du fattar få "
        "beslut — men de kräver att du förstår både värdering och marknadscykel."
    ),
    risk=RiskModel(
        risk_per_trade="Positionsbaserad — ingen fast stop-%. Risken styrs av "
                       "positionsstorlek och att du bara köper i rätt cykelfas",
        position_size="Max 10 % per aktie · hela positionen byggs i en omgång",
        max_positions="8–10 innehav · max 20–25 % per sektor",
        stop="Kurs under EMA200 = halvera · regimskifte = sälj resten",
        targets="Inget vinstmål — säljs på värdering, cykel eller kvalitetsfall",
    ),
    entry=_rules([
        ("Trend: pris över EMA200 och EMA50 över EMA200",
         "Gyllene kors. Bolaget ska redan vara i långsiktig upptrend när du köper.",
         "REGIME → Alpha Regime → Quality & Contrarian (Quality-läge) → ANALYSERA. Trend-villkoret visas "
         "som en av fyra gröna gates.", True),
        ("Värdering: P/E 7–25 och EV/EBIT 4–20",
         "Varken för dyrt eller suspekt billigt. Utanför spannet = avstå.",
         "REGIME → Alpha Regime → Quality & Contrarian: värderingsgaten visar P/E och EV/EBIT "
         "med spannet utsatt.", True),
        ("Marknadscykel: DISBELIEF, HOPE, OPTIMISM, BELIEF eller DISBELIEF_NEW",
         "Tidigt till mitt i en uppgång — aldrig i eufori.",
         "REGIME → Marknad → Market Cycle: aktuell fas. Alpha Regime visar samma fas som "
         "cykelgate."),
        ("Bolagskvalitet ≥ 55/100 eller KAP-badge",
         "ROIC, marginaler och tillväxt vägs till en kvalitetspoäng.",
         "SCREENING → Contrarian Alpha: kvalitetspoäng per bolag, ★ KAP SCREENED-"
         "badge på detaljkortet."),
        ("Max 10 % per aktie",
         "Ingen enskild position får dominera portföljen.",
         "PORTFOLIO → Holdings: Quality-sektionen visar '⚠ >10% position' när en "
         "position vuxit förbi taket.", True),
        ("Max 20–25 % per sektor",
         "Riskkontroll på portföljnivå — flera kvalitetsbolag i samma sektor är "
         "fortfarande en enda vadslagning.",
         "PORTFOLIO → Holdings: sektorexponering + korrelationsmatris.", True),
    ]),
    exit=_rules([
        ("Trenden bryts — kurs under EMA200",
         "Halvera positionen vid brottet, sälj resten vid regimskifte.",
         "REGIME → Alpha Regime: gate visar dagar under EMA200. PORTFOLIO → "
         "Holdings: signalkolumnen slår om till REDUCE/EXIT.", True),
        ("Värderingen spricker — P/E > 25 eller EV/EBIT > 20",
         "Cykeln kan vara på topp. Du betalar för optimism, inte för bolaget.",
         "SCREENING → Contrarian Alpha: värderingsband på detaljkortet."),
        ("Cykeln skiftar — THRILL, EUPHORIA, COMPLACENCY, ANXIETY eller DENIAL",
         "Sencykliska faser. Kvalitet skyddar inte mot en hel marknad som vänder.",
         "REGIME → Marknad → Market Cycle: fasindikatorn."),
        ("Bolagskvaliteten faller",
         "Kvalitetspoängen rasar eller marginalerna försämras konsekvent.",
         "SCREENING → Contrarian Alpha: kvalitetspoäng över tid + gate-checklistan "
         "på detaljkortet."),
    ]),
    workflow=(
        "SCREENING → Contrarian Alpha: sortera på kvalitetspoäng, leta KAP-badges.",
        "REGIME → Alpha Regime → Quality & Contrarian (Quality-läge): kör ANALYSERA på kandidaten.",
        "Alla fyra gates gröna = BUY. Tre gröna = WATCH. Två eller färre = WAIT.",
        "Larm: 💎 Quality-köpsignal (SCHEMALAGDA) säger till när ett av de 25 högst "
        "rankade i listan NYTT får alla fyra gröna.",
        "Räkna positionen: max 10 % av portföljen, kolla sektorexponeringen först.",
        "PORTFOLIO → Holdings: lägg in innehavet så regim- och signalbevakning körs.",
        "Kvartalsvis: gå igenom innehaven mot de fyra exit-reglerna.",
    ),
    cheatsheet=(
        ("Trend", "Pris > EMA200 · EMA50 > EMA200"),
        ("Värdering", "P/E 7–25 · EV/EBIT 4–20"),
        ("Cykel", "DISBELIEF · HOPE · OPTIMISM · BELIEF"),
        ("Kvalitet", "≥ 55/100 eller KAP-badge"),
        ("Signal", "4 gröna = BUY · 3 = WATCH · ≤2 = WAIT"),
        ("Max per aktie", "10 %"),
        ("Max per sektor", "20–25 %"),
        ("Antal innehav", "8–10"),
        ("Exit", "EMA200-brott = halvera · regimskifte = sälj"),
    ),
    pitfalls=(
        "Köper för sent — när nyheterna är positiva är kursen ofta redan i THRILL-fasen.",
        "Betalar för mycket för ett bra bolag. Kvalitet till fel pris är fortfarande "
        "en dålig affär — värderingsgaten finns just därför.",
        "Låter en vinnare växa förbi 10 % utan att trimma.",
        "Sitter kvar när kvalitetspoängen faller, för att bolaget 'brukade vara bra'.",
    ),
    support=SUPPORT_FULL,
    support_note="Contrarian Alpha-screenern och Alpha Regime-gaten kör Quality-läget.",
    source="Panelen + Masterguiden"
)


# ═════════════════════════════════════════════════════════════════════════════
#  6. DEEP CONTRARIAN  (Rick Rule / Eric Sprott)
# ═════════════════════════════════════════════════════════════════════════════

DEEP_CONTRARIAN = Playbook(
    key="contrarian",
    name="Deep Contrarian (panelens)",
    tagline="Köp i maximalt hat, sälj i eufori — i tredjedelar, styrt av cykelfasen",
    color=EMBER,
    level=LEVEL_ADVANCED,
    horizon="Cykel — 1–3 år per position",
    universe="Råvarurelaterat: gruvbolag, guld, silver, olja, gas",
    where="SCREENING → Contrarian Alpha (deep_contrarian)  ·  REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian)",
    idea=(
        "Rick Rule: 'du är antingen kontrarian eller så är du ett offer.' Du köper "
        "råvarurelaterade tillgångar i faser av maximal förtvivlan och säljer när alla "
        "är euforiska. Eric Sprotts princip: bygg alltid i tredjedelar, aldrig allt på "
        "en gång. Den svåraste strategin psykologiskt — du köper när det känns värst "
        "och måste kunna sitta genom 50 % nedgång utan att sälja."
    ),
    risk=RiskModel(
        risk_per_trade="Ingen fast procentsats — storleken bestäms av total portfölj "
                       "och conviction. Räkna med att kunna hålla genom 50 % nedgång",
        position_size="Byggs i tre lika delar (1/3 → 2/3 → 3/3)",
        max_positions="Max ~25 % per råvarukategori",
        stop="Ingen mekanisk stop — cykelfasen styr. Sälj i distributionsfaserna",
        targets="Distribueras i tre steg: THRILL → EUPHORIA → ANXIETY/DENIAL",
    ),
    entry=_rules([
        ("ACCUMULATE 1/3 — fas CAPITULATION eller DEPRESSION",
         "Maximalt pessimism, alla ger upp. Köp första tredjedelen.",
         "REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian) → ANALYSERA: fasindikatorn visar "
         "CAPITULATION/DEPRESSION och stage-etiketten ACCUMULATE 1/3."),
        ("ACCUMULATE 2/3 — fas DISBELIEF, ANGER eller PANIC",
         "Tvivel och ilska. Kursen fortfarande under 200-dagars MA. Andra tredjedelen.",
         "REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian): stage ACCUMULATE 2/3."),
        ("ACCUMULATE 3/3 — fas HOPE, kurs nära MA200 (−5 % till +5 %)",
         "Sista chansen att ackumulera billigt innan trenden är uppenbar.",
         "REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian): stage ACCUMULATE 3/3."),
        ("Bygg ALLTID i tre lika delar",
         "Aldrig allt på en gång. Det är så du får ner snittkursen när det faller mer.",
         "PORTFOLIO → Holdings: Deep Contrarian-innehav har fältet "
         "'Tranches deployed' (0/3–3/3) som visar var du är.", True),
        ("Bekräftelse: gummisnodde ≥ 90:e percentilen",
         "T.ex. Gold/Silver-ratio historiskt sträckt = råvaran max-billig vs motvikten.",
         "REGIME → Alpha Regime: välj 'Commodity exposure' för att se om ratios är "
         "historiskt sträckta."),
        ("Sentiment-overlay: retail sentiment < 30/100",
         "Extrem rädsla ger HIGH confidence på köpsignalen.",
         "INTELLIGENCE → Retail Pulse: sentiment-score."),
        ("Max ~25 % per råvarukategori",
         "T.ex. max 25 % i guldgruvbolag totalt — de rör sig som en enda position.",
         "PORTFOLIO → Holdings: Deep Contrarian-sektionen varnar '⚠ Sektor >25%'.", True),
    ]),
    exit=_rules([
        ("DISTRIBUTE 1/3 — fas THRILL",
         "Entusiasm och FOMO driver priserna. Sälj 25–33 %.",
         "REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian): stage DISTRIBUTE 1/3."),
        ("DISTRIBUTE 2/3 — fas EUPHORIA eller COMPLACENCY",
         "Alla är euforiska. Sälj 50–75 %.",
         "REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian): stage DISTRIBUTE 2/3."),
        ("DISTRIBUTE 3/3 — fas ANXIETY eller DENIAL",
         "Trenden bryter ner. Sälj resterande.",
         "REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian): stage DISTRIBUTE 3/3.", True),
        ("HOLD i OPTIMISM och BELIEF — sälj inte, köp inte mer",
         "Trenden har precis börjat. Säljer du här missar du det mesta av rörelsen.",
         "REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian): stage HOLD.", True),
        ("Sentiment-overlay: retail sentiment > 70/100",
         "Extrem girighet stärker säljsignalen.",
         "INTELLIGENCE → Retail Pulse: sentiment-score."),
    ]),
    workflow=(
        "REGIME → Alpha Regime → Quality & Contrarian (Deep Contrarian) → ANALYSERA: vilken cykelfas?",
        "Kolla gummisnodden (Commodity exposure) — är råvaran historiskt sträckt?",
        "INTELLIGENCE → Retail Pulse: bekräfta med sentiment (<30 = rädsla).",
        "SCREENING → Contrarian Alpha: hitta kandidater via Hat Score + Necessity.",
        "Köp EN tredjedel. Notera 'Tranches deployed' i Holdings.",
        "Vänta på nästa fas innan nästa tredjedel — månader, inte dagar.",
    ),
    cheatsheet=(
        ("Köpfas 1/3", "CAPITULATION · DEPRESSION"),
        ("Köpfas 2/3", "DISBELIEF · ANGER · PANIC"),
        ("Köpfas 3/3", "HOPE (kurs −5 % till +5 % vs MA200)"),
        ("Håll", "OPTIMISM · BELIEF — rör ingenting"),
        ("Sälj 1/3", "THRILL"),
        ("Sälj 2/3", "EUPHORIA · COMPLACENCY"),
        ("Sälj 3/3", "ANXIETY · DENIAL"),
        ("Bekräftelse", "Gummisnodde ≥ 90:e percentilen"),
        ("Sentiment", "<30 = köp · >70 = sälj"),
        ("Max per kategori", "~25 %"),
        ("Uthållighet", "Måste klara 50 % nedgång utan panik"),
    ),
    pitfalls=(
        "Köper allt på en gång istället för i tredjedelar — då kan du inte pressa ner "
        "snittkursen när det faller vidare.",
        "Säljer i HOLD-fasen för tidigt. Trenden har precis börjat — det är där de "
        "flesta hoppar av och missar hela uppgången.",
        "Ignorerar gummisnodde-signalerna, som ger objektiv bekräftelse när magkänslan "
        "sviker (och den sviker alltid i de här faserna).",
        "Köper i DENIAL/ANXIETY och tror det är billigt — det är ett SÄLJ-läge.",
    ),
    support=SUPPORT_PARTIAL,
    support_note="Panelen ger cykelfas, ackumulerings-/distributionssteg och hat-rankning. Detta är panelens EGEN kontrariska modell — den motsvarar inte Masterguidens fem råvarustrategier (Rule, Sprott, Durrett, Tiggre, Royalty), som har egna screeners och poängmodeller och ligger utanför panelen.",
    source="Panelen"
)


# ═════════════════════════════════════════════════════════════════════════════
#  7. EMBER  (råvarucykel — Rick Rule-inspirerad, kodstyrd)
# ═════════════════════════════════════════════════════════════════════════════

# Live thresholds from ember/config.py so this page can never drift from the
# engine. Fallbacks match the committed config if ember isn't importable.
try:  # pragma: no cover - trivial import guard
    from ember.config import (
        RISK_PCT as _EMBER_RISK_PCT,
        PULLBACK_EMA_PCT as _EMBER_PULLBACK_PCT,
        RSI_ENTRY_MAX as _EMBER_RSI_MAX,
        VERDICT_BUY_MIN as _EMBER_BUY, VERDICT_WATCH_MIN as _EMBER_WATCH,
        PULLBACK_MAX_PCT as _EMBER_PB_MAX, RSI_FULL as _EMBER_RSI_FULL, RSI_ZERO as _EMBER_RSI_ZERO,
        RS_FULL_PCT as _EMBER_RS_FULL, LATE_CYCLE_PCT as _EMBER_LATE,
        REGIME_PA_MIN as _EMBER_REG_PA, REGIME_AV_RED_MIN as _EMBER_REG_RED,
    )
except Exception:  # pragma: no cover
    _EMBER_RISK_PCT, _EMBER_PULLBACK_PCT, _EMBER_RSI_MAX = 0.02, 3.0, 45
    _EMBER_BUY, _EMBER_WATCH, _EMBER_PB_MAX, _EMBER_RSI_FULL, _EMBER_RSI_ZERO = 70.0, 50.0, 6.0, 40.0, 55.0
    _EMBER_RS_FULL, _EMBER_LATE = 5.0, 85.0
    _EMBER_REG_PA, _EMBER_REG_RED = 3, 2

EMBER_PB = Playbook(
    key="ember",
    name="🔥 Ember",
    tagline="Råvarucykel med tre hårda grindar och graderad setup-poäng — kontrariansk timing",
    color=EMBER,
    level=LEVEL_ADVANCED,
    horizon="Swing — dagar till veckor",
    universe="Börsdata råvarufilter + GDX/GDXJ/SIL/COPX/URA/XLE-konstituenter",
    where="SCREENING → Arc Screener → 🔥 EMBER  ·  REGIME → Råvaror → 🌍 EMBER Regime",
    idea=(
        "Ember tar Deep Contrarians cykeltänk och lägger teknisk precision ovanpå: "
        "makro/cykelfilter avgör OM du får handla, tre hårda grindar (50V EMA, inte sen "
        "cykel, regimen inte AV) avgör VAD, och en setup-poäng 0–100 avgör NÄR: "
        f"≥ {_EMBER_BUY:.0f} = KÖPLÄGE, {_EMBER_WATCH:.0f}–{_EMBER_BUY:.0f} = BEVAKA. Allt är kodstyrt — men "
        "det är många villkor att hålla reda på, så lär dig Deep Contrarian först. "
        "Det fullständiga regelverket (13 sektioner med exakta trösklar) finns under "
        "STRATEGIGUIDER → Ember."
    ),
    risk=RiskModel(
        risk_per_trade=f"{_EMBER_RISK_PCT * 100:.0f} % av kapitalet "
                       f"(ember/config.py: RISK_PCT = {_EMBER_RISK_PCT})",
        position_size="Risk ÷ stopavstånd — alltid strukturbaserat, aldrig fast antal",
        max_positions="Justeras efter regim (GRÖN/GUL/RÖD)",
        stop="Strukturbaserad (ATR/swing-low) — aldrig en godtycklig procentsats",
        targets="T1 och T2 per setup-kort, med R:R utsatt",
    ),
    entry=_rules([
        ("Komplex-regimen måste tillåta handel",
         "Varje råvarukomplex (Energi, Ädelmetaller, Basmetaller, Agri) har en egen "
         "regim med 5 pelare. Netto = gröna − röda (gul och DATA_GAP räknas 0): "
         f"netto ≥ {_EMBER_REG_PA} = PÅ (full storlek), 0–{_EMBER_REG_PA - 1} = SELEKTIV (halverad, endast "
         f"topp-1 och topp-2), under 0 eller ≥ {_EMBER_REG_RED} röda = AV (inga nya trades "
         "enligt regeln, korten visas ändå). Gul straffas inte längre som röd.",
         "REGIME → Råvaror → 🌍 EMBER Regime: ett verdikt per komplex. Screenern "
         "grindar varje setup mot sitt eget komplex — ett guldcase mot ÄDELMETALLER, "
         "inte mot ENERGI.", True),
        (f"Hård grind: pris över 50-veckors EMA och inte sen cykel (10å-percentil ≤ {_EMBER_LATE:.0f})",
         "Rätt komplex i rätt fas. Under 50V EMA eller på cykeltoppen övervägs ingen entry, "
         "oavsett setup-poäng.",
         "SCREENING → Arc Screener → 🔥 EMBER: grindarna märkta HÅRD på setup-kortet.", True),
        (f"Setup-poäng ≥ {_EMBER_BUY:.0f} = KÖPLÄGE, {_EMBER_WATCH:.0f}–{_EMBER_BUY:.0f} = BEVAKA",
         "Trend och entry är poäng, inte grindar: 20D > 50D EMA (15), relativ styrka mot "
         f"sektor-ETF graderad −{_EMBER_RS_FULL:.0f}…+{_EMBER_RS_FULL:.0f} % (20), stigande bottnar "
         f"2 = halva / 3 = full (15), pullback till 20D EMA 0…{_EMBER_PB_MAX:.0f} % (20), RSI(14) "
         f"{_EMBER_RSI_FULL:.0f}…{_EMBER_RSI_ZERO:.0f} (15), MACD, volym, candle, fallande ATR (15). "
         "Summa 100. Ett bolag som gått 1 % sämre än GDX stoppas inte längre, det tappar några poäng.",
         "SCREENING → Arc Screener → 🔥 EMBER: varje rad på kortet visar poäng/max.", True),
        ("Flaggorna ATR-surge och DXY-rally är avdrag, inte stopp",
         "Volatilitetsspik −15 p, dollarrally −10 p. Chop-zonen mellan 20D och 50D EMA är borttagen: "
         "den mätte samma sak som pullback-grinden med motsatt tecken.",
         "SCREENING → Arc Screener → 🔥 EMBER: flaggorna på kortet med avdraget utsatt."),
    ]),
    exit=_rules([
        ("Stop enligt setup-kortet — strukturbaserad",
         "Stopen sätts av struktur (ATR/swing-low), inte av en procentsats.",
         "SCREENING → Arc Screener → 🔥 EMBER: fältet Stop på setup-kortet.", True),
        ("T1 och T2 enligt kortet",
         "Delavyttring vid T1, resten mot T2. R:R står utskrivet innan du går in.",
         "SCREENING → Arc Screener → 🔥 EMBER: fälten T1, T2 och R:R."),
        ("Ogiltigförklaras om — casets premiss brister",
         "Varje kort listar vad som gör caset ogiltigt. Inträffar det: ut, oavsett kurs.",
         "SCREENING → Arc Screener → 🔥 EMBER: fältet 'Ogiltigförklaras om' på setup-kortet.", True),
        ("Komplexets regim slår om till AV",
         "Inga nya entries i det komplexet; befintliga hanteras enligt stop och "
         "invalidering. Övriga komplex påverkas inte.",
         "REGIME → Råvaror → 🌍 EMBER Regime: verdikt per komplex."),
    ]),
    workflow=(
        "REGIME → Råvaror → 🌍 EMBER Regime: vilket verdikt har komplexet du är "
        "intresserad av? AV = läs bara, handla inte.",
        "SCREENING → Arc Screener → 🔥 EMBER: gå igenom topp-korten uppifrån.",
        "Kontrollera T1–T4 (alla PASS) och att inga F1–F4-flaggor är aktiva.",
        "Kontrollera E1 (pullback) + E2 (RSI) — båda måste passera.",
        "Läs 'Ogiltigförklaras om' innan du köper — det är din exit-premiss.",
        "Position = risk ÷ stopavstånd. Lägg stop direkt, logga i journalen.",
    ),
    cheatsheet=(
        ("Regim", f"Per komplex, netto gröna − röda: ≥ {_EMBER_REG_PA} = PÅ · 0–{_EMBER_REG_PA - 1} = SELEKTIV · < 0 eller ≥ {_EMBER_REG_RED} röda = AV"),
        ("Trendgrindar", "T1–T4 — alla måste passera"),
        ("Entry E1", f"Pullback inom ±{_EMBER_PULLBACK_PCT:.0f} % av 20D EMA"),
        ("Entry E2", f"RSI(14) < {_EMBER_RSI_MAX}"),
        ("No-trade", "F1–F4 — någon aktiv = blockerad"),
        ("Risk/affär", f"{_EMBER_RISK_PCT * 100:.0f} % av kapitalet"),
        ("Stop", "Strukturbaserad (ATR/swing-low)"),
        ("Rangordning", "asymmetry_score + cykelbonus"),
        ("Elitcase", "Alla T + båda E + inga F"),
    ),
    pitfalls=(
        "Ta ett case där 3 av 4 trendgrindar passerar. Grindarna är hårda av en "
        "anledning — 'nästan' finns inte.",
        "Ignorera en no-trade-flagga för att caset annars ser perfekt ut.",
        "Köpa utan att ha läst 'Ogiltigförklaras om' — då vet du inte när du ska ut.",
        "Sätta en procentbaserad stop istället för den strukturbaserade på kortet.",
    ),
    note="Det fullständiga regelverket med alla 13 sektioner och exakta trösklar "
         "(hämtade live ur koden) finns under STRATEGIGUIDER → 🔥 Ember.",
    support=SUPPORT_FULL,
    support_note="Ember-screenern och per-komplex-regimerna körs i panelen.",
    source="Panelen"
)


# ═════════════════════════════════════════════════════════════════════════════
#  8. VIKING NINE  (OVTLYR Nine + Viking Execution + exitmotorn — panelens egen)
# ═════════════════════════════════════════════════════════════════════════════
# Talen hämtas ur motorerna (viking_execution, ovtlyr_nine) så att playbooken
# inte kan glida från det som faktiskt handlas — tests/test_strategy_rules_fas7.py
# låser dem. Reservvärdena är dagens tal.
try:  # pragma: no cover - trivial import guard
    from viking_execution import (
        MAX_RISK_PCT as _VN_RISK, ATR_STOP_MULT as _VN_ATR, RELATIVE_VOLUME_MIN as _VN_RVOL,
        MAX_CHASE_PCT as _VN_CHASE, MINIMUM_RR as _VN_RR, EARNINGS_BUFFER_DAYS as _VN_EARN,
        MAX_DAILY_LOSSES as _VN_LOSSES, MAX_POSITION_PCT as _VN_POS, NINE_DEVELOPING as _VN_DEV,
    )
    from ovtlyr_nine import BREADTH_LOW as _VN_BR_LO, BREADTH_HIGH as _VN_BR_HI, FG_MAX as _VN_FG, STALE_BDAYS as _VN_STALE
except Exception:  # pragma: no cover
    _VN_RISK, _VN_ATR, _VN_RVOL, _VN_CHASE, _VN_RR = 1.5, 1.5, 1.20, 2.0, 2.0
    _VN_EARN, _VN_LOSSES, _VN_POS, _VN_DEV = 5, 2, 25.0, 7
    _VN_BR_LO, _VN_BR_HI, _VN_FG, _VN_STALE = 25.0, 75.0, 90.0, 3


def _sv(x: float) -> str:
    """1.5 → '1,5' — svenskt decimaltecken i regeltext."""
    return f"{x:g}".replace(".", ",")


_VNR = "REGIME → Marknad → Arc Regime → ⚔️ Viking Nine Regime"
_VNS = "SCREENING → Arc Screener → ⚔️ Viking Nine"

VIKING_NINE = Playbook(
    key="viking_nine",
    name="⚔️ Viking Nine",
    tagline="OVTLYR Nine 9/9 avgör om setupen finns — Viking Execution avgör om entryn finns",
    color=PURPLE,
    level=LEVEL_ADVANCED,
    horizon="Swing — dagar till veckor",
    universe="Nordiska aktier (Börsdata) mot OMXS30 och svensk Large Cap-bredd · amerikanska mot SPY och SPDR-bredden",
    where=f"{_VNS}  ·  {_VNR}  ·  Thor (Regim, Backtest)  ·  VALHALL (Kartan, Papperskonton)",
    idea=(
        "Panelens egen version av OVTLYR Golden Ticket, räknad på riktiga priser (WOLF APPROXIMATION — aldrig "
        "OVTLYR:s egen data). Nio faktorer i tre lager måste alla vara gröna: marknaden (trend, signal, bredd), "
        "sektorn (Fear & Greed, bredd) och aktien (trend, signal, Fear & Greed, order blocks). Nio av nio är en "
        "setup, inte en affär — Viking Execution kräver dessutom momentum, trendstruktur, en stängd grön "
        "triggercandle med volym, att du inte jagar, R/R minst 2 och ingen rapport i närheten. Riskmotorn sätter "
        "storleken. Exitmotorn sköter resten: marknaden under EMA20 stänger allt. Saknad eller gammal data "
        "räknas aldrig som grönt."
    ),
    risk=RiskModel(
        risk_per_trade=f"{_sv(_VN_RISK)} % av kapitalet (strategies/viking.py: risk_pct = 0.015 — samma tal i Viking Execution)",
        position_size=f"Risk ÷ stoppavstånd ({_sv(_VN_ATR)} × ATR14), men högst {_sv(_VN_POS)} % av kapitalet i en position",
        max_positions=f"Portföljens tak (allokeraren: swing-delen, 6 % av totalen per position) · max {_VN_LOSSES} "
                      f"förlustaffärer per dag, sedan ingen ny entry",
        stop=f"{_sv(_VN_ATR)} × ATR14 från entry · breakeven efter ny högre topp · EMA10 som trail först när stoppen "
             f"ligger på breakeven",
        targets="Inget fast mål — exitmotorn: F&G-target (entry-F&G 0–50 → 63 · 50–75 → +10 · 75+ → +5), trail och marknadsexit",
    ),
    entry=_rules([
        ("Marknadslagret 3/3 — trend, signal och bredd på SPY (OMXS30 för nordiska aktier)",
         "Trend = EMA10 över EMA20 och kurs över EMA50. Signal = kurs över EMA20 (under = OVTLYR:s säljsignal för "
         "marknaden, då tas inga affärer). Bredd = andelen av de elva SPDR-sektorfonderna över EMA50 (Norden: svensk "
         f"Large Cap) enligt OVTLYR:s regler — bredden över sitt EMA10, under {_VN_BR_LO:.0f} bara efter en "
         f"uppvändning, över {_VN_BR_HI:.0f} och nedvänd = inga nya affärer.",
         f"{_VNR}: marknadsbannern MARKET SPY n/3 (OMXS30 för nordiska) överst — tre bockar krävs.", True),
        ("Sektorlagret 2/2 — sektor-ETF:ens Fear & Greed stigande och ETF:en över stigande EMA20",
         f"Sektorn hämtas ur Börsdata och mappas till en av de elva SPDR-fonderna. Syntetiskt F&G under {_VN_FG:.0f} "
         "och högre än för fem dagar sedan; ETF:en över EMA20 med EMA20 stigande.",
         f"{_VNR}: OVTLYR Nine-kortet, raden SECTOR med sektor-ETF:en utskriven."),
        ("Aktielagret 4/4 — trend, signal, Fear & Greed och inga restriktiva order blocks",
         "Samma trend- och signalregler på aktien, aktiens syntetiska F&G stigande, och inget bearish order block "
         "som blockerar vägen uppåt (ovtlyr.indicators.orderblocks).",
         f"{_VNR}: OVTLYR Nine-kortet, raden STOCK, och grafen med order blocks."),
        (f"Nine 9/9 är setupen — {_VN_DEV}–8/9 är DEVELOPING och ger WAIT, 6/9 eller lägre ger NO TRADE",
         f"Setup är inte entry. Saknad data och data äldre än {_VN_STALE} handelsdagar räknas aldrig som PASS — "
         "en faktor utan data gör att nio av nio inte går att nå.",
         f"{_VNS}: kategorin GOLDEN TICKET / READY / DEVELOPING / REJECTED per rad, och Nine n/9.", True),
        ("Viking Execution — momentum och trendstruktur",
         "RSI14 över 50 och stigande, kursen över gårdagens low, och kurs över EMA10 över EMA20 över EMA50.",
         f"{_VNR}: VIKING EXECUTION-kortet, raderna Momentum och Trend."),
        (f"Stängd grön triggercandle med relativ volym ≥ {_VN_RVOL:.2f}×",
         "Candlen måste ha stängt (Stockholm 17:30, Oslo 16:20, Köpenhamn 17:00, Helsingfors 18:30, New York 16:00) "
         "och vara grön — stängning över öppning och minst föregående stängning. Volymen den dagen minst "
         f"{_VN_RVOL:.2f} × snittet; kravet är backtestat i båda marknaderna.",
         f"{_VNR}: raderna Entry candle och Volym — WAIT FOR CLOSE tills candlen är stängd.", True),
        (f"Jaga inte — högst {_sv(_VN_CHASE)} % över triggerns stängning, och R/R ≥ {_VN_RR:g} till närmaste motstånd",
         "Motståndet är närmaste bearish order block, annars swing-högsta inom ett år. Under 2 i R/R = INSUFFICIENT "
         "R/R, över 2 % ovanför = NO CHASE — båda ger WAIT.",
         f"{_VNR}: raderna No chase och R/R, med nivåerna utskrivna."),
        (f"Ingen rapport inom {_VN_EARN} handelsdagar — okänt rapportdatum ger WAIT",
         "Saknad data är aldrig grönt: finns inget rapportdatum stannar beslutet på WAIT tills du kontrollerat det.",
         f"{_VNR}: raden Earnings. Rapportdatum ur Börsdata.", True),
        (f"Riskmotorn — {_sv(_VN_RISK)} % risk, stopp {_sv(_VN_ATR)} × ATR14, högst {_sv(_VN_POS)} % av kapitalet, "
         f"max {_VN_LOSSES} förlustaffärer per dag",
         f"Aktier = riskbudget ÷ stoppavstånd, men positionen får aldrig överstiga {_sv(_VN_POS)} % av kapitalet. "
         f"Efter {_VN_LOSSES} förlustaffärer samma dag tas ingen ny entry (journalen räknar).",
         f"{_VNR}: RISK ENGINE-kortet med kapital, riskbudget, stopp, antal aktier och exponering.", True),
        ("Marknadsrisken (🌩️) spärrar — OMXS30 från FÖRHÖJD, SPY vid HÖG",
         "Samma spärr i skannern, regimsidan och papperskontot (market_risk_gate). Backtestets snabbläge i Thor "
         "kör utan spärren och säger det.",
         "REGIME → Marknad → 🌩️ Marknadsrisk: nivån per marknad; bannern på Viking Nine Regime."),
    ]),
    exit=_rules([
        ("Marknaden under EMA20 (SPY, OMXS30 för nordiska) — stäng allt",
         "Hård exit, inga undantag. Samma regel som entryns signalfaktor, fast omvänd.",
         f"{_VNR}: exitmotorn, raden HARD MARKET EXIT — CLOSE ALL.", True),
        (f"Initial stopp {_sv(_VN_ATR)} × ATR14 från entry",
         "ATR på entrydagen. Stoppen ligger intradag: low under stoppen = ut på stoppen, gap under = öppningen.",
         f"{_VNR}: exitmotorn, raden INITIAL STOP.", True),
        ("Breakeven efter ny högre topp — sedan är stängning under gårdagens low en exit",
         "Ny högre topp = high över högsta high de tio dagarna före entry. Då flyttas stoppen till entry.",
         f"{_VNR}: exitmotorn, raden BE EXIT och fältet breakeven armed."),
        ("EMA10 som trailing stop — först när stoppen ligger på breakeven",
         "Backtestet: +0,21 R per affär mot +0,04 R med EMA10 från dag ett, som sålde affärer innan de hunnit gå.",
         f"{_VNR}: exitmotorn, raden TRAILING STOP med EMA10-nivån."),
        ("Bearish order block — kursen går in i ett restriktivt block",
         "Blocket som entryn räknade R/R mot är nu nått: motståndet håller.",
         f"{_VNR}: exitmotorn, raden BEARISH BLOCK, och grafen."),
        ("Gap & crap — gap upp över gårdagens high som stänger under gårdagens stängning",
         "Falsk styrka. Exit på nästa öppning.",
         f"{_VNR}: exitmotorn, raden GAP & CRAP."),
        ("Sektorn bearish och marknadsbredden faller — ut ur sektorn",
         "Samma två faktorer som i OVTLYR Nine, fast som exit: sektor-ETF:en under EMA20 och bredden under sitt EMA10.",
         f"{_VNR}: exitmotorn, raden SECTOR/BREADTH."),
        ("Aktiens signal SELL — stängning under EMA20",
         "Signalfaktorn på aktien bryts. Exit på nästa öppning.",
         f"{_VNR}: exitmotorn, raden STOCK SIGNAL."),
        ("F&G-target — entry-F&G 0–50 → exit vid 63 · 50–75 → +10 · 75+ → +5",
         "Målet sätts av var sentimentet stod när du köpte, inte av en kursnivå.",
         f"{_VNR}: exitmotorn, raden F&G EXIT med entry-F&G och target."),
        (f"Rapport inom {_VN_EARN} handelsdagar — EXIT eller REDUCE BEFORE EARNINGS",
         "Samma fönster som entryspärren. Ingen position över rapport.",
         f"{_VNR}: exitmotorn, raden EARNINGS.", True),
    ]),
    mindset=_rules([
        ("Nio av nio eller ingenting — 'nästan' finns inte",
         "Strategins kant är att alla lager pekar åt samma håll. Åtta av nio är en annan strategi.",
         f"{_VNS}: DEVELOPING-raderna är bevakning, inte köp."),
        ("Saknad data är aldrig grönt",
         "DATA UNAVAILABLE och STALE DATA är FAIL i praktiken. Då väntar du eller letar data — du gissar inte.",
         f"{_VNR}: statusen per faktor i Nine-kortet."),
        ("Exitmotorn äger utgången",
         "Inga egna vinstmål. Marknadsexiten, stoppen, breakeven, trailen och F&G-target bestämmer.",
         f"{_VNR}: exitmotorn för en öppen position — fyll i entry och datum."),
    ]),
    workflow=(
        f"{_VNS}: kör skanningen (två steg: trend + signal på hela universumet, sedan full Nine på de bästa). "
        "Läs kategorin — GOLDEN TICKET och READY är kandidater.",
        f"{_VNR}: öppna aktien. Marknadsbannern n/3, Nine-kortet lager för lager, VIKING EXECUTION-kortet.",
        "GOLDEN TICKET = alla krav klara. WAIT = setupen finns men entryn är inte klar — vänta på stängd candle, "
        "volym, rätt pris eller rapportdatum. NO TRADE = Nine ≤ 6/9, marknadens signal SELL, rapport nära eller "
        "dagsgränsen nådd.",
        f"Riskmotorn ger antal aktier: {_sv(_VN_RISK)} % risk, {_sv(_VN_ATR)} × ATR14, högst {_sv(_VN_POS)} % av kapitalet.",
        "Lägg stoppen hos mäklaren direkt och logga affären i PORTFOLIO → 📓 Trade Journal.",
        f"Varje dag: exitmotorn på {_VNR} — marknaden under EMA20 stänger allt, oavsett aktie.",
        "Thor: samma motor i fliken Regim, backtestet i fliken Backtest (signal på stängning, entry nästa öppning, "
        "stopp intradag).",
    ),
    cheatsheet=(
        ("Setup", "OVTLYR Nine 9/9 — marknad 3 · sektor 2 · aktie 4"),
        ("Entry", f"stängd grön candle · rvol ≥ {_VN_RVOL:.2f}× · ≤ {_sv(_VN_CHASE)} % över trigger · R/R ≥ {_VN_RR:g}"),
        ("Risk/trade", f"{_sv(_VN_RISK)} % av kapitalet"),
        ("Stop", f"{_sv(_VN_ATR)} × ATR14 · breakeven efter ny topp · trail EMA10 därefter"),
        ("Position", f"högst {_sv(_VN_POS)} % av kapitalet · 6 % av totalen (allokeraren)"),
        ("Hård exit", "marknaden under EMA20 = stäng allt"),
        ("Rapport", f"ingen entry inom {_VN_EARN} handelsdagar · okänt datum = WAIT"),
        ("Dagsgräns", f"max {_VN_LOSSES} förlustaffärer per dag"),
    ),
    pitfalls=(
        "Ta affären på 8/9 för att den nionde faktorn 'bara saknar data'. Saknad data är ett nej.",
        "Köpa på en candle som inte stängt — nordisk stängning 17:30, inte lunch.",
        "Jaga en GOLDEN TICKET som redan gått mer än 2 % från triggern.",
        "Flytta EMA10-trailen före breakeven — backtestet sålde då vinnarna för tidigt.",
        "Sitta kvar när SPY eller OMXS30 stängt under EMA20 för att aktien 'ser stark ut'.",
    ),
    note="Viking (OVTLYR Golden Ticket) är den äldre tio-kravs-formuleringen av samma idé; Viking Nine är motorn "
         "som skannern, regimsidan, papperskontot och backtestet faktiskt kör.",
    support=SUPPORT_FULL,
    support_note="Skannern (⚔️ Viking Nine), regimsidan, exitmotorn, Valhalls papperskonto och backtestet kör samma "
                 "funktioner (ovtlyr_nine, viking_execution, viking_exit, viking_backtest).",
    source="Panelen (ovtlyr_nine.py, viking_execution.py, viking_exit.py)",
)


# ═════════════════════════════════════════════════════════════════════════════
#  9. BERSERK  (råvaran först — tre setups mot råvarucykeln)
# ═════════════════════════════════════════════════════════════════════════════
try:  # pragma: no cover - trivial import guard
    from berserk import signals as _bsg
    from berserk.backtest import (
        MIN_TURNOVER_M as _BK_TURN, RISK_BY_SETUP as _BK_RISK, STOP_ATR as _BK_STOP,
        live_config as _bk_live_config, live_portfolio as _bk_live_portfolio,
    )
    _BK_CFG, _BK_PORT = _bk_live_config(), _bk_live_portfolio()
    _BK_S1, _BK_S2, _BK_S3 = _bsg.S1, _bsg.S2, _bsg.S3
    _BK_S3_LIVE = _BK_S3 in _BK_CFG.setups
    _BK_GAP = float(_BK_CFG.max_gap_atr or 0)
    _BK_FREE = ", ".join(_BK_CFG.commodity_free)
    _BK_NPOS, _BK_POSPCT, _BK_HEAT = _BK_PORT.max_positions, _BK_PORT.max_position_pct, _BK_PORT.max_heat_pct
    _BK_LAG, _BK_RV1, _BK_RV2 = abs(_bsg.DIV_LAG) * 100, _bsg.RVOL_S1, _bsg.RVOL_S2
    _BK_BEAR, _BK_HATED, _BK_HATED_ETF = abs(_bsg.BEAR_DD) * 100, abs(_bsg.HATED_DD) * 100, abs(_bsg.HATED_DD_ETF) * 100
    _BK_RANGE, _BK_LOOK, _BK_RSI2 = _bsg.RANGE_LOW * 100, _bsg.LOOKBACK_BEAR, _bsg.RSI2_MAX
    _BK_JUMP, _BK_JUMP_DAYS = _bsg.JUMP_MAX * 100, _bsg.JUMP_BLOCK_DAYS
except Exception:  # pragma: no cover
    _BK_S1, _BK_S2, _BK_S3 = "S1 Divergens", "S2 Cykelvändning", "S3 Snapback"
    _BK_RISK, _BK_STOP = {_BK_S1: 1.25, _BK_S2: 1.25, _BK_S3: 1.0}, {_BK_S1: 2.0, _BK_S2: 2.0, _BK_S3: 3.0}
    _BK_TURN, _BK_S3_LIVE, _BK_GAP, _BK_FREE = 5.0, False, 1.0, "Norden"
    _BK_NPOS, _BK_POSPCT, _BK_HEAT = 8, 20.0, 8.0
    _BK_LAG, _BK_RV1, _BK_RV2, _BK_BEAR, _BK_HATED, _BK_HATED_ETF = 10.0, 1.2, 1.5, 30.0, 40.0, 30.0
    _BK_RANGE, _BK_LOOK, _BK_RSI2, _BK_JUMP, _BK_JUMP_DAYS = 20.0, 126, 10.0, 60.0, 20

_BKR = "REGIME → Råvaror → 🪓 BERSERK Regime"
_BKS = "SCREENING → Arc Screener → 🪓 BERSERK"

BERSERK_PB = Playbook(
    key="berserk",
    name="🪓 BERSERK",
    tagline="Råvaran först — tre setups för när råvarucykeln och aktien pekar åt olika håll",
    color=AMBER,
    level=LEVEL_ADVANCED,
    horizon="Swing — dagar till veckor",
    universe="berserk.universe: nordiska råvaruproducenter (kärnan) samt USA, Kanada, London och råvaru-ETF:er (satelliter)",
    where=f"{_BKS}  ·  {_BKR}  ·  Thor (Regim, Råvara, Backtest)  ·  VALHALL (Kartan, Papperskonton)",
    idea=(
        "Varje råvaruaktie mäts mot sin egen råvara — drivaren (koppar mot HG=F, guld mot GC=F, uran mot URA …). "
        f"{_BK_S1}: råvaran är stark men aktien har halkat efter och vänder nu upp. {_BK_S2}: råvaran har varit i "
        "baisse och vänder, aktien är hatad och bryter upp — det sällsynta, stora läget. "
        f"{_BK_S3}: en snabb återhämtning i en stigande råvara (av i live, kvar i backtestet). Marknadsgrinden, "
        "råvarugrinden, likviditeten och datavakten måste alla vara öppna, annars är raden BEVAKA, inte KÖP. "
        "Signalen bedöms på stängd dag, entry nästa öppning, stoppar intradag — samma motor i skannern, "
        "regimsidan, Valhalls papperskonto och backtestet."
    ),
    risk=RiskModel(
        risk_per_trade=f"{_sv(_BK_RISK[_BK_S1])} % av kapitalet för {_BK_S1} och {_BK_S2}, {_sv(_BK_RISK[_BK_S3])} % för "
                       f"{_BK_S3} (berserk/backtest.py: RISK_BY_SETUP)",
        position_size=f"Risk ÷ stoppavstånd, men högst {_sv(_BK_POSPCT)} % av BERSERK-boken per position",
        max_positions=f"{_BK_NPOS} positioner · högst 2 per tema · 4 per komplex · samlad värme {_sv(_BK_HEAT)} % (live_portfolio)",
        stop=f"{_BK_S1} och {_BK_S2} {_sv(_BK_STOP[_BK_S1])} × ATR14 · {_BK_S3} {_sv(_BK_STOP[_BK_S3])} × ATR14 "
             "(katastrofstopp) — intradag, gap under ger öppningen",
        targets="Inga fasta mål — exitreglerna per setup: trail på EMA20 (S1) eller EMA50 (S2), tesen bruten, tidsstopp",
    ),
    entry=_rules([
        ("Råvaran är drivaren — temat har en prisserie och signalen bedöms mot den, på stängd dag",
         "Drivaren per tema väljs i preferensordning (den första med data från periodens start). Råvarans värden "
         "läggs på aktiens dagar med senast kända värde — aldrig ett senare. Teman utan drivare kan bara få S3.",
         f"{_BKR}: temakortet med drivaren, fasen (BAISSE · VÄNDER, BAISSE, STARK, UPPTREND, SVAG) och producenterna "
         "mot råvaran.", True),
        (f"{_BK_S1} — råvaran stark, aktien efter och vänder",
         f"Råvaran över SMA50 och positiv på 63 dagar; aktien minst {_BK_LAG:.0f} procentenheter efter råvaran på 63 "
         f"dagar; vändningen = stängning över gårdagens high, grön dag och relativ volym ≥ {_sv(_BK_RV1)}×.",
         f"{_BKS}: raden med setup {_BK_S1}, divergensen i procentenheter och rvol."),
        (f"{_BK_S2} — råvaran i baisse och vänder, aktien hatad och bryter upp",
         f"Råvaran minst {_BK_BEAR:.0f} % under 52-veckorshögsta eller i nedre {_BK_RANGE:.0f} % av sitt femårsintervall "
         f"någon gång de senaste {_BK_LOOK} dagarna, och vänder (över EMA50, EMA20 stigande). Aktien minst "
         f"{_BK_HATED:.0f} % under sin topp de senaste {_BK_LOOK} dagarna (ETF {_BK_HATED_ETF:.0f} %) och bryter "
         f"20-dagarshögsta med relativ volym ≥ {_sv(_BK_RV2)}×. Flera setups samma dag — S2 före S1 före S3.",
         f"{_BKS}: raden med setup {_BK_S2} och dd252. Thor-listan ger S2 högst rang.", True),
        (f"{_BK_S3} — {'på' if _BK_S3_LIVE else 'av'} i live, kvar i backtestet",
         f"Råvaran över SMA200 (utan drivare: aktien över stigande SMA200), RSI(2) under {_BK_RSI2:.0f} och stängning "
         "under nedre Bollinger (20, 2). Portföljvarianterna 2026-10 valde S3 av, värme 8 %, råvarugrind och "
         "gapregel — bättre än basen i båda perioderna.",
         f"{_BKS}: skannern följer live_config — S3-rader visas bara om setupen är på."),
        ("Marknadsgrinden — regionens index över SMA200 (OMXS30, SPY, TSX, FTSE)",
         "Index under SMA200 gör raden BEVAKA även om setupen finns. Saknas regionens index gäller SPY.",
         f"{_BKR}: marknadschipen per region, och raden 'under SMA200' i skälen.", True),
        (f"Råvarugrinden — råvarukorgen DBC över SMA200 för satelliter; {_BK_FREE} är fri",
         "Utanför kärnan krävs att råvarorna som helhet stiger. Nordiska producenter handlas utan den grinden.",
         f"{_BKS}: skälet 'råvarugrind' på raden; {_BKR}: DBC-chipet."),
        (f"Likviditet ≥ {_sv(_BK_TURN)} miljoner per dag (snitt 20 dagar, lokal valuta) och datavakten",
         f"Omsättning under gränsen = BEVAKA. Datavakten spärrar {_BK_JUMP_DAYS} dagar efter ett dagshopp över "
         f"{_BK_JUMP:.0f} % (ojusterad split, extrautdelning eller datafel — indikatorerna är då opålitliga).",
         f"{_BKS}: skälen 'omsättning under' och 'datavakt' på raden."),
        (f"Entry nästa öppning — men inte om öppningen gappar mer än {_sv(_BK_GAP)} × ATR över signalstängningen",
         "Gapregeln ur portföljvarianterna: ett stort gap äter risken innan affären börjat. Då hoppas signalen över.",
         f"{_BKS}: planen för nästa öppning på raden; VALHALL (Papperskonton) fyller på samma regel."),
        ("KÖP kräver setup i dag och alla grindar öppna — annars BEVAKA",
         "Setup utan grindar = BEVAKA med skälet utskrivet. Nästan-setup (watch_reason) = BEVAKA. Inget av det = tom rad.",
         f"{_BKS}: statuskolumnen KÖP / BEVAKA med skälen; Thor-listan översätter rakt av.", True),
    ]),
    exit=_rules([
        (f"{_BK_S1} — stopp {_sv(_BK_STOP[_BK_S1])} × ATR14; +1 ATR flyttar stoppen till entry − 0,25 ATR; "
         "+2 ATR ger trail på stängning under EMA20",
         "ATR på signaldagen. Stängningsreglerna ger exit nästa öppning; stoppen ligger intradag.",
         f"{_BKR}: exitplanen per setup; VALHALL (Papperskonton): samma simulate_exit.", True),
        (f"{_BK_S1} — tesen bruten när råvaran stänger under SMA50; tidsstopp 15 dagar utan stängning ≥ +0,5 ATR",
         "Divergensen ska stängas av att aktien kommer ikapp — gör den inte det inom 15 dagar är caset fel.",
         f"{_BKR}: drivarens SMA50 på temakortet."),
        (f"{_BK_S2} — stopp {_sv(_BK_STOP[_BK_S2])} × ATR14; +2 ATR ger trail på stängning under EMA50; "
         "tesen bruten när råvaran stänger under EMA50",
         "Cykelvändningen ska hålla i råvaran — faller drivaren tillbaka under EMA50 är vändningen falsk.",
         f"{_BKR}: drivarens EMA50 på temakortet.", True),
        (f"{_BK_S3} — katastrofstopp {_sv(_BK_STOP[_BK_S3])} × ATR14, exit vid stängning över SMA5 eller efter 5 dagar",
         "Snapbacken är en kort affär: målet är medelvärdet, inte trenden.",
         f"{_BKS}: bara när S3 är på i live_config."),
        ("Stoppar intradag — low under stoppen ger exit på stoppen, gap under ger öppningen",
         "Samma fyllnadsregel i backtestet och papperskontot. Lägg stoppen hos mäklaren.",
         "VALHALL (Papperskonton): kontot wolf_arc/berserk visar fyllnaderna; Thor fliken Backtest märker regeln."),
        (f"Max {_BK_PORT.max_daily_losses} förlustaffärer per dag och rapportspärren sköter du själv",
         "Portföljregeln finns i portfolio_config; rapportdatum finns inte i backtestet (historiska datum saknas) — "
         "kontrollera dem i Börsdata innan entry.",
         "PORTFOLIO → 📓 Trade Journal: räkna dagens förluster. Rapportdatum ur Börsdata."),
    ]),
    mindset=_rules([
        ("Råvaran bestämmer — aktien är bara hävstången",
         "En BERSERK-affär är ett bet på råvarans läge. Går drivaren fel är affären fel, oavsett aktiens graf.",
         f"{_BKR}: temakortet före aktieraden."),
        ("S2 är det stora läget och det sällsynta — vänta på det",
         "Cykelvändningar i hatade råvaror kommer få gånger per decennium. S1 fyller tiden däremellan, S3 är av.",
         f"{_BKS}: sortera på setup; VALHALL (Kartan): ⚡ cykelstart på temat."),
    ]),
    workflow=(
        f"{_BKR}: vilka teman är i BAISSE · VÄNDER eller STARK? Vilka marknadsgrindar är öppna? DBC över SMA200?",
        f"{_BKS}: raderna med KÖP — setup i dag och alla grindar. BEVAKA-rader har skälet utskrivet.",
        "Kontrollera drivaren och aktien i Thor (Regim: setupen och planen; Råvara: temats cykel, hävstång och kvoter).",
        f"Storlek = risk ÷ stoppavstånd ({_sv(_BK_RISK[_BK_S1])} % för S1/S2), högst {_sv(_BK_POSPCT)} % av boken, "
        f"högst 2 per tema och 4 per komplex, värme under {_sv(_BK_HEAT)} %.",
        "Entry nästa öppning om gapet är under 1 × ATR. Lägg stoppen hos mäklaren direkt.",
        "Följ exitplanen per setup varje dag — trail, tesen bruten, tidsstopp. Logga i PORTFOLIO → 📓 Trade Journal.",
    ),
    cheatsheet=(
        ("Setups", f"{_BK_S1} (råvara stark, aktie efter) · {_BK_S2} (råvara vänder, aktie hatad) · S3 {'på' if _BK_S3_LIVE else 'av'}"),
        ("Grindar", f"index över SMA200 · DBC över SMA200 (ej {_BK_FREE}) · ≥ {_sv(_BK_TURN)} milj/dag · datavakt"),
        ("Risk/affär", f"{_sv(_BK_RISK[_BK_S1])} % (S1/S2) · {_sv(_BK_RISK[_BK_S3])} % (S3)"),
        ("Stop", f"{_sv(_BK_STOP[_BK_S1])} × ATR14 (S1/S2) · {_sv(_BK_STOP[_BK_S3])} × ATR14 (S3) · intradag"),
        ("Trail", "S1 under EMA20 efter +2 ATR · S2 under EMA50 efter +2 ATR"),
        ("Tesen bruten", "S1 råvaran under SMA50 · S2 råvaran under EMA50"),
        ("Bok", f"{_BK_NPOS} positioner · {_sv(_BK_POSPCT)} % per position · 2 per tema · 4 per komplex · värme {_sv(_BK_HEAT)} %"),
        ("Fyllnad", f"signal på stängning · entry nästa öppning (gap ≤ {_sv(_BK_GAP)} × ATR) · stopp intradag"),
    ),
    pitfalls=(
        "Köpa en BEVAKA-rad för att setupen 'nästan' är där — grinden som stängde den finns av en anledning.",
        "Handla S1 i ett tema där råvaran står i TOPP (tioårspercentil ≥ 90) utan att läsa Blindspot-chipet.",
        "Ignorera tesen-bruten-regeln för att aktien fortfarande stiger — drivaren under SMA50/EMA50 är exit.",
        "Fylla boken med fyra kopparaktier: högst två per tema, fyra per komplex.",
        "Köpa på öppningen efter ett gap över 1 × ATR — risken är redan uppäten.",
    ),
    note="Backtestet (berserk.backtest) rapporterar i R och säger själv vad som inte ingår: rapportspärr, "
         "överlevnadsbias i universumet, rullning i terminsbaserade ETF:er.",
    support=SUPPORT_FULL,
    support_note="Skannern (berserk_scan), 🪓 BERSERK Regime, Valhalls papperskonto och backtestet kör samma regler "
                 "(berserk.live.evaluate, berserk.backtest.simulate_exit, live_config).",
    source="Panelen (berserk/signals.py, berserk/backtest.py, berserk/live.py)",
)


# ═════════════════════════════════════════════════════════════════════════════
#  Registry + helpers
# ═════════════════════════════════════════════════════════════════════════════

PLAYBOOKS: dict[str, Playbook] = {
    "momentum": MOMENTUM,
    "quality": QUALITY,
    "alpha": ALPHA,
    "viking": VIKING,
    "viking_nine": VIKING_NINE,
    "wolf": WOLF,
    "contrarian": DEEP_CONTRARIAN,
    "ember": EMBER_PB,
    "berserk": BERSERK_PB,
}

# Masterguiden's non-native strategies (Rule · Sprott · Durrett · Tiggre ·
# Royalty · Insider). Defined in a separate module for readability; imported here
# so there is still exactly one registry. Import is at the bottom on purpose —
# that module imports Playbook/RiskModel from this one.
try:  # pragma: no cover - import guard
    from strategy_rules_masterguide import MASTERGUIDE_PLAYBOOKS as _COMMODITY
    PLAYBOOKS.update(_COMMODITY)
except Exception:  # pragma: no cover
    _COMMODITY = {}

# Suggested learning order — easiest and most mechanical first, hardest last.
# Panel-native strategies come first; the Masterguide commodity family follows,
# ordered from the safest (Royalty) to the most speculative (Sprott/Tiggre).
LEARNING_ORDER: tuple[str, ...] = (
    "momentum", "quality", "alpha", "viking", "viking_nine", "wolf", "contrarian", "ember", "berserk",
    "royalty", "insider", "rule", "durrett", "tiggre", "sprott",
)


def by_support(support: str) -> list[Playbook]:
    """Playbooks grouped by how much of them the panel actually implements."""
    return [p for p in PLAYBOOKS.values() if p.support == support]

LEVEL_COLOR = {LEVEL_BEGINNER: GREEN, LEVEL_MEDIUM: AMBER, LEVEL_ADVANCED: RED}


def get(key: str) -> Playbook | None:
    return PLAYBOOKS.get(key)


def by_level(level: str) -> list[Playbook]:
    return [p for p in PLAYBOOKS.values() if p.level == level]


def as_rule_dicts(rules: tuple[Rule, ...]) -> list[dict]:
    """Legacy shape for the existing rule-card renderer."""
    return [r.as_dict() for r in rules]


# Backwards-compatible aliases (rules_page.py historically owned these lists).
SWING_RULES = as_rule_dicts(WOLF.entry + WOLF.exit)
LONGTERM_RULES = as_rule_dicts(ALPHA.entry + ALPHA.exit)
OVTLYR_ENTRY_RULES = as_rule_dicts(VIKING.entry)
OVTLYR_EXIT_RULES = as_rule_dicts(VIKING.exit)
OVTLYR_MINDSET = as_rule_dicts(VIKING.mindset)
MOMENTUM_ENTRY_RULES = as_rule_dicts(MOMENTUM.entry)
MOMENTUM_EXIT_RULES = as_rule_dicts(MOMENTUM.exit)
