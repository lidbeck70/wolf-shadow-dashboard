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
    where="SCREENING → Swing · Swing Screener  ·  REGIME → Swing Regime",
    idea=(
        "Den enklaste strategin i panelen och den bästa att börja med. Du gör tre "
        "saker: (1) kollar om marknaden överhuvudtaget tillåter köp, (2) tar de "
        "starkaste bolagen från en färdig momentum-ranking, (3) säljer enligt tre "
        "mekaniska regler. Inga diagram att tolka, inga mönster att lära sig — "
        "panelen räknar allt. Rutinen görs en gång i veckan och tar ~15 minuter."
    ),
    risk=RiskModel(
        risk_per_trade="≈1,2–2 % av kapitalet (12–20 % position × 10 % stop)",
        position_size="12–20 % vid GRÖN regim · halv storlek vid GUL",
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
         "Momentum-score = 50/50 på 3- och 6-månadersavkastning. Topp 20 = köpbara.",
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
         "SCREENING → Swing: veckochecklistan punkt 4 påminner dig. Räkna dina köp "
         "i journalen innan du lägger nästa order."),
        ("Max 8 positioner — mål 6–8",
         "Färre positioner = du hinner följa dem. Fler = du tappar överblick.",
         "SCREENING → Swing: Positioner visar 'x/6–8'. Panelen vägrar lägga till "
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
         "SCREENING → Swing: kryssa 'Stängt under MA50' på positionskortet — kortet "
         "blir rött med 'SÄLJ — regel utlöst'.", True),
        ("Säljregel 2 — stop på −10 %",
         "Hård stop från entry. Räknas automatiskt, ingen tolkning.",
         "SCREENING → Swing: 'Stop (−10 %)' på varje positionskort. Kursen når stop "
         "→ kortet blir rött automatiskt. Lägg alltid stopen hos mäklaren också.", True),
        ("Säljregel 3 — ur topp 40 på rankingen",
         "Momentum har lämnat bolaget även om priset inte fallit än. Den regel som "
         "oftast glöms — och den som räddar dig från långsamma förlorare.",
         "SCREENING → Swing Screener: syns bolaget inte längre i listan (rad 1–40)? "
         "Kryssa då 'Ur topp 40' på positionskortet i Swing-fliken.", True),
        ("Vid +20 % — sälj halva, flytta stop till entry",
         "Du säkrar vinst och gör resten riskfri. Resten får löpa utan vinstmål.",
         "SCREENING → Swing: en gul '+20 % — sälj halva'-flagga dyker upp på kortet. "
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
        "SCREENING → Swing: gå igenom ALLA befintliga positioner mot säljregel 1–3 "
        "innan du köper något nytt.",
        "SCREENING → Swing: välj setup (A/B) på bevakningen, sätt entry-kurs och "
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
         "WOLF REGIME → Regime Score visar trendriktning. Grön badge = long. Röd = stå utanför.", True),
        ("Ta inga trades i konsolidering",
         "Range = förbjudet område. Vänta på breakout.",
         "WOLF REGIME → Entry Checklist: RSI + ATR ratio visar konsolidering. Gate #2 = auto-check."),
        ("En trade kräver en key level",
         "Supply/demand eller tydligt stöd/motstånd.",
         "WOLF REGIME → Entry Checklist: Order Blocks-kortet visar bullish/bearish OBs + närmaste nivå. Gate #3 = OB inom 3%."),
        ("Entry endast efter pullback",
         "Inga impulsiva entries i rakt fall eller rally.",
         "WOLF REGIME → Entry Checklist: Trend-kortet visar EMA10/20 nivåer. Gate #4 = pris inom 2% av EMA10/20."),
        ("Candlestick-trigger krävs",
         "Pinbar, engulfing eller break-and-retest.",
         "WOLF REGIME → Entry Checklist: Candlestick-kortet visar detekterade mönster (Hammer, Engulfing, Doji). Gate #5 = auto-check."),
        ("Volym måste bekräfta rörelsen",
         "Ingen volym = ingen trade.",
         "WOLF REGIME → Entry Checklist: Momentum-kortet visar Vol ratio. Gate #6 = ≥ 1.0x."),
        ("Minsta R:R är 1:2",
         "Helst 1:3. Aldrig under 1:2.",
         "WOLF REGIME → Entry Checklist: Volatilitet-kortet visar ATR. OB-kortet visar target. Gate #7 = R:R auto-beräknad.", True),
    ]),
    exit=_rules([
        ("Max 2 % risk per trade",
         "Stop sätts på struktur (2,5 × ATR14) — positionsstorleken anpassas så att "
         "förlusten blir 2 % av kapitalet om stopen träffas.",
         "WOLF REGIME → SL/TP Calculator + Entry Checklist: ATR-värde i Volatilitet-kortet. "
         "Gate #8 visar SL-avstånd i %.", True),
        ("Flytta SL till BE först efter ny HH/LL",
         "Inte tidigare, inte senare.",
         "WOLF REGIME → Ichimoku-gaugen visar prisnivåer. När pris gör ny swing high ovanför din entry → flytta SL till entry."),
        ("Max två förluster per dag",
         "Stoppa dagen direkt efter två minus.",
         "Egen disciplin. Två förluster = stäng plattformen. Nordic Alpha analyserar — du handlar nästa dag.", True),
        ("Exit: Kijun-sen trail + ½ ATR hård stop",
         "Kijun-sen (26p) som dynamiskt trailing stop. Stäng om pris stänger under Kijun OCH under EMA 10. ½ ATR som nödstopp.",
         "WOLF REGIME → Ichimoku-gaugen visar Kijun-nivå. Entry Checklist: Trend-kortet visar EMA10. Gate #11 visar båda nivåer."),
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
         "ALPHA REGIME → Regime-badge visar GRÖN/ORANGE/RÖD. Gates 1-7 måste passera. Alla gröna = OK att köpa.", True),
        ("Pris måste ligga över 200 EMA",
         "Bekräftar långsiktig upptrend.",
         "ALPHA REGIME → Gate #2 visar 'Pris vs EMA200' med exakt avstånd i %. VIKING REGIME → Magenta-linjen (EMA 200) i grafen.", True),
        ("50 EMA måste ligga över 200 EMA",
         "Golden cross = positivt momentum.",
         "ALPHA REGIME → Gate #3 visar 'Golden Cross' eller 'Death Cross'. VIKING REGIME → Gul (EMA 50) ovanför magenta (EMA 200)."),
        ("Sektorn måste vara grön",
         "Ingen exponering i svaga sektorer.",
         "ALPHA REGIME → Gate #4 visar sektorstatus (0-3). SECTOR & REGIME → Sektorhjulet visar grön/gul/röd per sektor."),
        ("Fear & Greed under 60 vid köp",
         "Undvik eufori och toppjakt.",
         "ALPHA REGIME → Gate #5 visar F&G-score + OK/EJ OK. SENTIMENT → Stor F&G-gauge. Under 60 = OK. Över 60 = vänta med köp."),
        ("Max 10 % per aktie",
         "Ingen enskild position får dominera.",
         "SCREENER → Alpha Screener: högsta ranking = allokera max 10%. Lägre ranking = allokera max 7%. Aldrig mer oavsett hur bra det ser ut.", True),
        ("Max 20–25 % per sektor",
         "Riskkontroll på portföljnivå.",
         "SCREENER → Alpha Screener: sortera efter sektor. Räkna: hur mycket äger du redan i samma sektor? Max 25%.", True),
    ]),
    exit=_rules([
        ("Minska vid EMA200-brott, sälj vid regimskifte",
         "EMA200-brott = reducera 50%. Regim röd = sälj resten.",
         "ALPHA REGIME → Gate #6 visar dagar under EMA200. Gate #7 visar regime-färg. Orange = reducera halvt. Röd = sälj allt.", True),
        ("Sälj vid sektor + breadth crossover",
         "Sektor OCH marknadsbreadd vänder ner = sälj sektorpositioner.",
         "SECTOR & REGIME → Sektorhjulet: om sektor byter från grön till röd. VIKING REGIME → VIKING NINE sektorpoäng under 30."),
        ("Analysera alltid historiska nedgångar",
         "Avgör om fallet är brus eller strukturellt.",
         "VIKING REGIME → Drawdowns sub-tab klassificerar nedgångar. BACKTEST → Alpha mode: historisk prestation med drawdown-analys."),
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
        position_size="Risk ÷ stopavstånd (½ × ATR14)",
        max_positions="Följ portföljens allmänna tak",
        stop="½ × ATR14 från entry · EMA10 som trailing stop",
        targets="TP1 @ 2R (25 % av core) · TP2 @ 4R (25 %) · F&G-target enligt exit-regel 9",
    ),
    entry=_rules([
        ("Market Trend: SPY 10EMA > 20EMA, Price > 50EMA", "Bullish = buy zone. Bearish = inga trades.",
         "VIKING REGIME → Trend-kort visar 'Direction: Bullish/Bearish'. Regime-badge = GRÖN krävs.", True),
        ("Market Signal: Köpsignal på $SPY", "Viking overlay måste vara grön.",
         "VIKING REGIME → Välj SPY som ticker. Long-term signal = 'BUY'. Regime = GRÖN."),
        ("Market Breadth: Bull List bullish crossover", "Måste matcha market trend.",
         "VIKING REGIME → Bull List % gauge (Advanced Analysis). Under 25 + vänder upp = bästa entry. Över 75 + vänder ner = stopp."),
        ("Sector Breadth: Stigande", "Bullish 10EMA-kors krävs.",
         "SECTOR & REGIME → Sektorhjulet: sektorn måste vara grön. Trend Distribution: sektorn i 'Uptrend'."),
        ("Sector Fear & Greed: Stigande", "Sektorsentiment måste förbättras.",
         "SENTIMENT → F&G-gauge stigande. VIKING REGIME → Sentiment-kort: score stigande (jämför med förra veckan)."),
        ("Stock Signal: Köp", "Viking signal måste visa Buy.",
         "VIKING REGIME → Long-term signal badge visar 'BUY' (grön). Score > 60 krävs."),
        ("Stock Trend: 10EMA/20EMA, Price > 50EMA", "Alla EMA:er alignade.",
         "VIKING REGIME → Grafen: vit (10) > orange (20) > gul (50). Alla stigande. Pris ovanför alla tre."),
        ("Stock Fear & Greed: Stigande", "Aktiens sentiment förbättras.",
         "VIKING REGIME → Oscillator Direction visar 'Rising' + timing 'Early' eller 'Mid'. Inte 'Exhausted'."),
        ("Order Blocks: Inga restriktiva OBs", "Inga bearish OBs blockerar vägen uppåt.",
         "VIKING REGIME → Grafen: inga röda OB-zoner ovanför nuvarande pris. Order Blocks-tab: inga aktiva bearish OBs nära."),
        ("Momentum: Pris ovanför gårdagens lägsta", "Bekräftar positivt momentum.",
         "VIKING REGIME → Grafen: dagens candle stänger ovanför gårdagens lägsta nivå. Momentum-kort: RSI > 50."),
    ]),
    exit=_rules([
        ("$SPY stänger under 20 EMA → STÄNG ALLT", "Hård exit. Inga undantag.",
         "VIKING REGIME → Välj SPY. Om pris under orange linje (EMA 20) = stäng alla positioner omedelbart.", True),
        ("½ ATR Stop Loss från entry", "Strukturbaserad stop, aldrig %.",
         "VIKING REGIME → Volatility-kort: ATR 14 värde. SL = entry-pris minus (ATR ÷ 2).", True),
        ("10 EMA Trailing Stop", "Pris stänger under 10 EMA = exit.",
         "VIKING REGIME → Grafen: vit linje = EMA 10. Om candle stänger under den vita linjen = exit."),
        ("Order Block hit", "Pris springer in i restriktivt OB = exit.",
         "VIKING REGIME → Grafen: om pris rör sig in i en röd OB-zon (bearish) = stäng positionen."),
        ("Gap & Crap", "Gap up följt av reversal = omedelbar exit.",
         "VIKING REGIME → Grafen: om dagens öppning gappar upp men sedan faller tillbaka under gårdagens stängning = exit direkt."),
        ("Stängning under gårdagens lägsta", "Efter att du redan rullat (moved SL) = exit.",
         "VIKING REGIME → Grafen: jämför dagens stängning med gårdagens lägsta. Stänger under = exit."),
        ("Sektor + Market breadth crossover", "Sälj alla trades i den sektorn.",
         "SECTOR & REGIME → Om sektorn byter från grön till röd, OCH Bull List % vänder ner = sälj alla positioner i sektorn."),
        ("Stock Sell signal", "Viking signal flippar till Sell.",
         "VIKING REGIME → Long-term signal badge byter till 'SELL' (röd) eller 'REDUCE' (magenta)."),
        ("Fear & Greed target hit", "Beror på var du köpte: 0-50 = exit vid 63. 50-75 = 10p spread. 75+ = 5p spread.",
         "SENTIMENT → F&G-gauge. Notera ditt entry-F&G-värde. Räkna target: entry + spread. Exit när target nås."),
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
        "Sätt ½ ATR-stop direkt. Trail med EMA10.",
        "Kolla rapportdatum — stäng senast 5 handelsdagar före.",
    ),
    cheatsheet=(
        ("Entry", "Alla 10 krav måste vara gröna — inga undantag"),
        ("Marknad först", "SPY bullish + BUY-signal + breadth upp"),
        ("Stop", "½ × ATR14"),
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
)


# ═════════════════════════════════════════════════════════════════════════════
#  Registry + helpers
# ═════════════════════════════════════════════════════════════════════════════

PLAYBOOKS: dict[str, Playbook] = {
    "momentum": MOMENTUM,
    "wolf": WOLF,
    "alpha": ALPHA,
    "viking": VIKING,
}

# Suggested learning order — easiest and most mechanical first.
LEARNING_ORDER: tuple[str, ...] = ("momentum", "alpha", "viking", "wolf")

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
