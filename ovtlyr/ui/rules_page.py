"""
Rules page — clean cyberpunk-styled display of all trading rules.
Every rule has a PANEL guide showing exactly which Nordic Alpha tab to use.
Updated for the consolidated 9-tab layout.
"""

import streamlit as st

_BG2     = "#14141e"
_CYAN    = "#c9a84c"
_GREEN   = "#2d8a4e"
_MAGENTA = "#8b7340"
_RED     = "#c44545"
_YELLOW  = "#d4943a"
_BLUE    = "#c9a84c"
_TEXT    = "#e8e4dc"
_DIM     = "#8a8578"

# ------------------------------------------------------------------ #
#  Rule data — updated panel guides for 9-tab layout
# ------------------------------------------------------------------ #

SWING_RULES: list[dict] = [
    {
        "number": 1,
        "text": "Handla endast i trendens riktning",
        "explanation": "Upptrend = long. Nedtrend = short. Aldrig mot trenden.",
        "panel_guide": "WOLF REGIME → Regime Score visar trendriktning. Grön badge = long. Röd = stå utanför.",
    },
    {
        "number": 2,
        "text": "Ta inga trades i konsolidering",
        "explanation": "Range = förbjudet område. Vänta på breakout.",
        "panel_guide": "WOLF REGIME → Entry Checklist: RSI + ATR ratio visar konsolidering. Gate #2 = auto-check.",
    },
    {
        "number": 3,
        "text": "En trade kräver en key level",
        "explanation": "Supply/demand eller tydligt stöd/motstånd.",
        "panel_guide": "WOLF REGIME → Entry Checklist: Order Blocks-kortet visar bullish/bearish OBs + närmaste nivå. Gate #3 = OB inom 3%.",
    },
    {
        "number": 4,
        "text": "Entry endast efter pullback",
        "explanation": "Inga impulsiva entries i rakt fall eller rally.",
        "panel_guide": "WOLF REGIME → Entry Checklist: Trend-kortet visar EMA10/20 nivåer. Gate #4 = pris inom 2% av EMA10/20.",
    },
    {
        "number": 5,
        "text": "Candlestick-trigger krävs",
        "explanation": "Pinbar, engulfing eller break-and-retest.",
        "panel_guide": "WOLF REGIME → Entry Checklist: Candlestick-kortet visar detekterade mönster (Hammer, Engulfing, Doji). Gate #5 = auto-check.",
    },
    {
        "number": 6,
        "text": "Volym måste bekräfta rörelsen",
        "explanation": "Ingen volym = ingen trade.",
        "panel_guide": "WOLF REGIME → Entry Checklist: Momentum-kortet visar Vol ratio. Gate #6 = ≥ 1.0x.",
    },
    {
        "number": 7,
        "text": "Minsta R:R är 1:2",
        "explanation": "Helst 1:3. Aldrig under 1:2.",
        "panel_guide": "WOLF REGIME → Entry Checklist: Volatilitet-kortet visar ATR. OB-kortet visar target. Gate #7 = R:R auto-beräknad.",
    },
    {
        "number": 8,
        "text": "Max 5% risk per trade",
        "explanation": "SL baseras på struktur. Max 5% av kapital per trade.",
        "panel_guide": "WOLF REGIME → SL/TP Calculator + Entry Checklist: ATR-värde i Volatilitet-kortet. Gate #8 visar SL-avstånd i %.",
    },
    {
        "number": 9,
        "text": "Flytta SL till BE först efter ny HH/LL",
        "explanation": "Inte tidigare, inte senare.",
        "panel_guide": "WOLF REGIME → Ichimoku-gaugen visar prisnivåer. När pris gör ny swing high ovanför din entry → flytta SL till entry.",
    },
    {
        "number": 10,
        "text": "Max två förluster per dag",
        "explanation": "Stoppa dagen direkt efter två minus.",
        "panel_guide": "Egen disciplin. Två förluster = stäng plattformen. Nordic Alpha analyserar — du handlar nästa dag.",
    },
    {
        "number": 11,
        "text": "Exit: Kijun-sen trail + ½ ATR hård stop",
        "explanation": "Kijun-sen (26p) som dynamiskt trailing stop. Stäng om pris stänger under Kijun OCH under EMA 10. ½ ATR som nödstopp.",
        "panel_guide": "WOLF REGIME → Ichimoku-gaugen visar Kijun-nivå. Entry Checklist: Trend-kortet visar EMA10. Gate #11 visar båda nivåer.",
    },
]

LONGTERM_RULES: list[dict] = [
    {
        "number": 1,
        "text": "Köp endast i grön regim",
        "explanation": "Regimindikatorn måste vara grön.",
        "panel_guide": "ALPHA REGIME → Regime-badge visar GRÖN/ORANGE/RÖD. Gates 1-7 måste passera. Alla gröna = OK att köpa.",
    },
    {
        "number": 2,
        "text": "Pris måste ligga över 200 EMA",
        "explanation": "Bekräftar långsiktig upptrend.",
        "panel_guide": "ALPHA REGIME → Gate #2 visar 'Pris vs EMA200' med exakt avstånd i %. VIKING REGIME → Magenta-linjen (EMA 200) i grafen.",
    },
    {
        "number": 3,
        "text": "50 EMA måste ligga över 200 EMA",
        "explanation": "Golden cross = positivt momentum.",
        "panel_guide": "ALPHA REGIME → Gate #3 visar 'Golden Cross' eller 'Death Cross'. VIKING REGIME → Gul (EMA 50) ovanför magenta (EMA 200).",
    },
    {
        "number": 4,
        "text": "Sektorn måste vara grön",
        "explanation": "Ingen exponering i svaga sektorer.",
        "panel_guide": "ALPHA REGIME → Gate #4 visar sektorstatus (0-3). SECTOR & REGIME → Sektorhjulet visar grön/gul/röd per sektor.",
    },
    {
        "number": 5,
        "text": "Fear & Greed under 60 vid köp",
        "explanation": "Undvik eufori och toppjakt.",
        "panel_guide": "ALPHA REGIME → Gate #5 visar F&G-score + OK/EJ OK. SENTIMENT → Stor F&G-gauge. Under 60 = OK. Över 60 = vänta med köp.",
    },
    {
        "number": 6,
        "text": "Minska vid EMA200-brott, sälj vid regimskifte",
        "explanation": "EMA200-brott = reducera 50%. Regim röd = sälj resten.",
        "panel_guide": "ALPHA REGIME → Gate #6 visar dagar under EMA200. Gate #7 visar regime-färg. Orange = reducera halvt. Röd = sälj allt.",
    },
    {
        "number": 7,
        "text": "Sälj vid sektor + breadth crossover",
        "explanation": "Sektor OCH marknadsbreadd vänder ner = sälj sektorpositioner.",
        "panel_guide": "SECTOR & REGIME → Sektorhjulet: om sektor byter från grön till röd. VIKING REGIME → VIKING NINE sektorpoäng under 30.",
    },
    {
        "number": 8,
        "text": "Max 20-25% per sektor",
        "explanation": "Riskkontroll på portföljnivå.",
        "panel_guide": "SCREENER → Alpha Screener: sortera efter sektor. Räkna: hur mycket äger du redan i samma sektor? Max 25%.",
    },
    {
        "number": 9,
        "text": "Max 10% per aktie",
        "explanation": "Ingen enskild position får dominera.",
        "panel_guide": "SCREENER → Alpha Screener: STRONG BUY = allokera 10%. BUY = allokera 7%. Aldrig mer oavsett hur bra det ser ut.",
    },
    {
        "number": 10,
        "text": "Analysera alltid historiska nedgångar",
        "explanation": "Avgör om fallet är brus eller strukturellt.",
        "panel_guide": "VIKING REGIME → Drawdowns sub-tab klassificerar nedgångar. BACKTEST → Alpha mode: historisk prestation med drawdown-analys.",
    },
]

OVTLYR_ENTRY_RULES: list[dict] = [
    {"number": 1, "text": "Market Trend: SPY 10EMA > 20EMA, Price > 50EMA", "explanation": "Bullish = buy zone. Bearish = inga trades.",
     "panel_guide": "VIKING REGIME → Trend-kort visar 'Direction: Bullish/Bearish'. Regime-badge = GRÖN krävs."},
    {"number": 2, "text": "Market Signal: Köpsignal på $SPY", "explanation": "Viking overlay måste vara grön.",
     "panel_guide": "VIKING REGIME → Välj SPY som ticker. Long-term signal = 'BUY'. Regime = GRÖN."},
    {"number": 3, "text": "Market Breadth: Bull List bullish crossover", "explanation": "Måste matcha market trend.",
     "panel_guide": "VIKING REGIME → Bull List % gauge (Advanced Analysis). Under 25 + vänder upp = bästa entry. Över 75 + vänder ner = stopp."},
    {"number": 4, "text": "Sector Breadth: Stigande", "explanation": "Bullish 10EMA-kors krävs.",
     "panel_guide": "SECTOR & REGIME → Sektorhjulet: sektorn måste vara grön. Trend Distribution: sektorn i 'Uptrend'."},
    {"number": 5, "text": "Sector Fear & Greed: Stigande", "explanation": "Sektorsentiment måste förbättras.",
     "panel_guide": "SENTIMENT → F&G-gauge stigande. VIKING REGIME → Sentiment-kort: score stigande (jämför med förra veckan)."},
    {"number": 6, "text": "Stock Signal: Köp", "explanation": "Viking signal måste visa Buy.",
     "panel_guide": "VIKING REGIME → Long-term signal badge visar 'BUY' (grön). Score > 60 krävs."},
    {"number": 7, "text": "Stock Trend: 10EMA/20EMA, Price > 50EMA", "explanation": "Alla EMA:er alignade.",
     "panel_guide": "VIKING REGIME → Grafen: vit (10) > orange (20) > gul (50). Alla stigande. Pris ovanför alla tre."},
    {"number": 8, "text": "Stock Fear & Greed: Stigande", "explanation": "Aktiens sentiment förbättras.",
     "panel_guide": "VIKING REGIME → Oscillator Direction visar 'Rising' + timing 'Early' eller 'Mid'. Inte 'Exhausted'."},
    {"number": 9, "text": "Order Blocks: Inga restriktiva OBs", "explanation": "Inga bearish OBs blockerar vägen uppåt.",
     "panel_guide": "VIKING REGIME → Grafen: inga röda OB-zoner ovanför nuvarande pris. Order Blocks-tab: inga aktiva bearish OBs nära."},
    {"number": 10, "text": "Momentum: Pris ovanför gårdagens lägsta", "explanation": "Bekräftar positivt momentum.",
     "panel_guide": "VIKING REGIME → Grafen: dagens candle stänger ovanför gårdagens lägsta nivå. Momentum-kort: RSI > 50."},
]

OVTLYR_EXIT_RULES: list[dict] = [
    {"number": 1, "text": "$SPY stänger under 20 EMA → STÄNG ALLT", "explanation": "Hård exit. Inga undantag.",
     "panel_guide": "VIKING REGIME → Välj SPY. Om pris under orange linje (EMA 20) = stäng alla positioner omedelbart."},
    {"number": 2, "text": "½ ATR Stop Loss från entry", "explanation": "Strukturbaserad stop, aldrig %. ",
     "panel_guide": "VIKING REGIME → Volatility-kort: ATR 14 värde. SL = entry-pris minus (ATR ÷ 2)."},
    {"number": 3, "text": "10 EMA Trailing Stop", "explanation": "Pris stänger under 10 EMA = exit.",
     "panel_guide": "VIKING REGIME → Grafen: vit linje = EMA 10. Om candle stänger under den vita linjen = exit."},
    {"number": 4, "text": "Order Block hit", "explanation": "Pris springer in i restriktivt OB = exit.",
     "panel_guide": "VIKING REGIME → Grafen: om pris rör sig in i en röd OB-zon (bearish) = stäng positionen."},
    {"number": 5, "text": "Gap & Crap", "explanation": "Gap up följt av reversal = omedelbar exit.",
     "panel_guide": "VIKING REGIME → Grafen: om dagens öppning gappar upp men sedan faller tillbaka under gårdagens stängning = exit direkt."},
    {"number": 6, "text": "Stängning under gårdagens lägsta", "explanation": "Efter att du redan rullat (moved SL) = exit.",
     "panel_guide": "VIKING REGIME → Grafen: jämför dagens stängning med gårdagens lägsta. Stänger under = exit."},
    {"number": 7, "text": "Sektor + Market breadth crossover", "explanation": "Sälj alla trades i den sektorn.",
     "panel_guide": "SECTOR & REGIME → Om sektorn byter från grön till röd, OCH Bull List % vänder ner = sälj alla positioner i sektorn."},
    {"number": 8, "text": "Stock Sell signal", "explanation": "Viking signal flippar till Sell.",
     "panel_guide": "VIKING REGIME → Long-term signal badge byter till 'SELL' (röd) eller 'REDUCE' (magenta)."},
    {"number": 9, "text": "Fear & Greed target hit", "explanation": "Beror på var du köpte: 0-50 = exit vid 63. 50-75 = 10p spread. 75+ = 5p spread.",
     "panel_guide": "SENTIMENT → F&G-gauge. Notera ditt entry-F&G-värde. Räkna target: entry + spread. Exit när target nås."},
    {"number": 10, "text": "Earnings risk", "explanation": "Stäng position minst 1 vecka före rapportdag.",
     "panel_guide": "Kolla rapportdatum externt (t.ex. Börsdata, Yahoo Finance). Stäng senast 5 handelsdagar innan rapport."},
]

OVTLYR_MINDSET: list[dict] = [
    {"number": 1, "text": "Det finns INGA FÖRVÄNTNINGAR på utfallet", "explanation": "Handla planen, inte prediktionen.",
     "panel_guide": "Alla flikar i Nordic Alpha visar DATA, inte åsikter. Följ signalerna — känn ingenting."},
    {"number": 2, "text": "Det finns INGA VINSTMÅL", "explanation": "Låt exit-signalerna göra sitt jobb.",
     "panel_guide": "Sätt aldrig en TP-order baserat på känsla. Använd trailing stop (EMA 10) eller exit-signal."},
    {"number": 3, "text": "Jag har bara en plan att ta mig ur", "explanation": "Planen ÄR din edge. Exekveringen är allt.",
     "panel_guide": "RULES-fliken (denna sida) = din plan. Läs igenom före varje handelsdag. Inga avvikelser."},
]


# ------------------------------------------------------------------ #
#  HTML helpers
# ------------------------------------------------------------------ #

def _rule_card_html(rule: dict, color: str) -> str:
    num = rule["number"]
    text = rule["text"]
    expl = rule["explanation"]
    guide = rule.get("panel_guide", "")

    guide_html = ""
    if guide:
        guide_html = (
            f"<div style='color:{_CYAN};font-size:0.62rem;margin-top:5px;"
            f"padding:5px 8px;background:rgba(201,168,76,0.05);border-radius:3px;"
            f"border-left:2px solid rgba(201,168,76,0.2);'>"
            f"<b>NORDIC ALPHA:</b> {guide}</div>"
        )

    return (
        f"<div style='background:{_BG2};border-left:3px solid {color};"
        f"padding:10px 14px;margin:6px 0;border-radius:4px;'>"
        f"<span style='color:{color};font-size:1.1rem;font-weight:700;'>#{num}</span>"
        f"<span style='color:{_TEXT};font-size:0.88rem;margin-left:10px;'>{text}</span>"
        f"<div style='color:{_DIM};font-size:0.7rem;margin-top:3px;'>{expl}</div>"
        f"{guide_html}"
        f"</div>"
    )


def _section_header_html(title: str, subtitle: str, color: str) -> str:
    return (
        f"<div style='border-bottom:2px solid {color};padding-bottom:8px;margin-bottom:12px;'>"
        f"<h3 style='color:{color};margin:0;letter-spacing:0.1em;'>{title}</h3>"
        f"<span style='color:{_DIM};font-size:0.7rem;letter-spacing:0.08em;'>{subtitle}</span>"
        f"</div>"
    )


# ------------------------------------------------------------------ #
#  Main render
# ------------------------------------------------------------------ #

def render_rules_page() -> None:
    st.markdown(
        f"<div style='text-align:center;padding:20px 0 10px 0;'>"
        f"<h1 style='color:{_CYAN};letter-spacing:0.15em;margin:0;'>TRADING RULES</h1>"
        f"<p style='color:{_DIM};font-size:0.75rem;letter-spacing:0.12em;'>"
        f"Varje regel har en <span style='color:{_CYAN};'>NORDIC ALPHA</span>-guide "
        f"som visar exakt vilken flik och vad du ska titta på.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("Reglerna visas även direkt i varje regime-flik under 'Regler'-sektionen.")

    # ── Wolf + Alpha side by side ─────────────────────────────────────
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown(
            _section_header_html("Wolf Trading", "11 regler — kortsiktig taktik + Ichimoku exit", _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown("".join(_rule_card_html(r, _CYAN) for r in SWING_RULES), unsafe_allow_html=True)

    with right_col:
        st.markdown(
            _section_header_html("Alpha Trend / Regim", "10 regler — strategisk position", _GREEN),
            unsafe_allow_html=True,
        )
        st.markdown("".join(_rule_card_html(r, _GREEN) for r in LONGTERM_RULES), unsafe_allow_html=True)

    # ── Viking Golden Ticket ──────────────────────────────────────────
    st.markdown("<hr style='border-color:rgba(201,168,76,0.13);margin:30px 0;'/>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;margin-bottom:20px;'>"
        f"<h2 style='color:{_BLUE};letter-spacing:0.15em;'>VIKING GOLDEN TICKET</h2>"
        f"<p style='color:{_DIM};font-size:0.75rem;'>WHERE OUTLIERS WIN</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    oc1, oc2, oc3 = st.columns([1, 1, 0.6])
    with oc1:
        st.markdown(_section_header_html("Open (Alpha)", "10 entry-regler", _BLUE), unsafe_allow_html=True)
        st.markdown("".join(_rule_card_html(r, _BLUE) for r in OVTLYR_ENTRY_RULES), unsafe_allow_html=True)
    with oc2:
        st.markdown(_section_header_html("Close (Alpha)", "10 exit-regler", _RED), unsafe_allow_html=True)
        st.markdown("".join(_rule_card_html(r, _RED) for r in OVTLYR_EXIT_RULES), unsafe_allow_html=True)
    with oc3:
        st.markdown(_section_header_html("Mindset", "3 gyllene regler", _YELLOW), unsafe_allow_html=True)
        st.markdown("".join(_rule_card_html(r, _YELLOW) for r in OVTLYR_MINDSET), unsafe_allow_html=True)

    st.markdown(
        f"<div style='text-align:center;padding:30px 0 10px 0;'>"
        f"<span style='color:{_BLUE};font-size:0.7rem;letter-spacing:0.15em;'>"
        f"SAVE TIME &middot; MAKE MONEY &middot; START WINNING WITH LESS RISK</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Panel guide table ─────────────────────────────────────────────
    st.markdown("<hr style='border-color:rgba(201,168,76,0.13);margin:20px 0;'/>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;margin-bottom:16px;'>"
        f"<h2 style='color:{_MAGENTA};letter-spacing:0.12em;'>NORDIC ALPHA — FLIKGUIDE</h2>"
        f"<p style='color:{_DIM};font-size:0.7rem;'>Vilken flik kontrollerar vilken regel?</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    guide_data = [
        ("SCREENER", "Wolf / Alpha / Viking Screener",
         "Wolf: Regime Score + volymbekräftelse. Alpha: 20-poängs fundamental + globala marknader (US/CA/UK/EU). Viking: Z-score composite. Alla tre har marknadsväljare."),
        ("BACKTEST", "Wolf / Alpha / Viking / RS Sector",
         "Verifiera strategi historiskt. Test Top N: skicka screener-resultat till backtest. Alpha Backtest har marknadsväljare."),
        ("HOLDINGS", "Wolf / Viking / Alpha Portfolio",
         "Tre portföljer med live-signaler. Korrelationsmatris visar överlappande risk. Cloud-lagring — innehav överlever omstart."),
        ("WOLF REGIME", "Wolf #1-11 (alla — self-contained)",
         "4-lagers regime + Entry Checklist (Trend/Volatilitet/Momentum/Candlestick/OB). Alla 11 gates med auto-pass/fail. SL/TP-kalkylator. Benchmark RS. Regler i flikbotten."),
        ("ALPHA REGIME", "Alpha #1-10 + globala marknader",
         "Alla 10 regler som live gates. GRÖN/ORANGE/RÖD badge. EMA-nivåer + F&G + drawdown. Marknadsväljare för alla regioner. Regler i flikbotten."),
        ("VIKING REGIME", "Viking alla + Alpha #1-5",
         "Prisgraf + EMA 10/20/50/200 + Order Blocks. Trend/Volatility/Sentiment/Momentum-kort. Oscillator Direction. SL/TP-kalkylator. Benchmark RS. Regler i flikbotten."),
        ("SECTOR & REGIME", "Wolf #1 | Alpha #4, #7 | Viking #4",
         "Sektorhjul grön/gul/röd. Global index-regime. Risk-On/Off."),
        ("SENTIMENT", "Alpha #5 | Viking #5, #9",
         "Fear & Greed gauge 0-100. Under 60 = OK köpa. Över 60 = vänta."),
        ("HEATMAP", "Alpha #4, #8",
         "Performance per sektor/land. 1D/5D/1M. Identifiera starka/svaga sektorer."),
        ("RETAIL SENTIMENT", "Retail hype + StockTwits",
         "Composite Score per ticker (confidence-viktad). Reddit ApeWisdom top 25. StockTwits bull/bear-ratio. Volume anomalies. Hype Alert: multi-source overlap."),
        ("ODIN'S BLINDSPOT", "Konträr opportunism",
         "Hatade men nödvändiga aktier med stark kassa. Hat Score + Necessity + Strength + Catalyst = Opportunity Score. Value Trap-varningar. Potential Reversal-flaggor."),
        ("RULES", "Alla regelverk",
         "Denna sida. Läs före varje handelsdag. Inga avvikelser. Regler finns även inline i varje regime-flik."),
    ]

    guide_html = "<table style='width:100%;border-collapse:collapse;'>"
    guide_html += (
        f"<tr style='border-bottom:1px solid rgba(201,168,76,0.15);'>"
        f"<th style='text-align:left;color:{_CYAN};font-size:0.75rem;padding:8px;'>FLIK</th>"
        f"<th style='text-align:left;color:{_CYAN};font-size:0.75rem;padding:8px;'>KONTROLLERAR</th>"
        f"<th style='text-align:left;color:{_CYAN};font-size:0.75rem;padding:8px;'>HUR DU ANVÄNDER DEN</th>"
        f"</tr>"
    )
    for tab, rules, usage in guide_data:
        guide_html += (
            f"<tr style='border-bottom:1px solid rgba(138,133,120,0.2);'>"
            f"<td style='color:{_TEXT};font-size:0.8rem;padding:6px 8px;font-weight:700;'>{tab}</td>"
            f"<td style='color:{_YELLOW};font-size:0.72rem;padding:6px 8px;'>{rules}</td>"
            f"<td style='color:{_DIM};font-size:0.72rem;padding:6px 8px;'>{usage}</td>"
            f"</tr>"
        )
    guide_html += "</table>"
    st.markdown(guide_html, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Inline rules helper — renders a collapsible rules section
# ------------------------------------------------------------------ #

def render_inline_rules(strategy: str) -> None:
    """Render a collapsible rules section for a given strategy.

    strategy: 'wolf', 'viking', or 'alpha'
    """
    try:
        _STRATEGY_MAP = {
            "wolf": ("Wolf Trading", SWING_RULES, _CYAN),
            "alpha": ("Alpha Trend", LONGTERM_RULES, _GREEN),
            "viking": ("Viking Golden Ticket", OVTLYR_ENTRY_RULES + OVTLYR_EXIT_RULES, _BLUE),
        }
        config = _STRATEGY_MAP.get(strategy.lower())
        if not config:
            return
        label, rules, color = config
        with st.expander(f"Regler — {label}", expanded=False):
            st.markdown(
                "".join(_rule_card_html(r, color) for r in rules),
                unsafe_allow_html=True,
            )
    except Exception:
        pass
