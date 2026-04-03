"""
Rules page — clean cyberpunk-styled display of all trading rules
plus a practical guide on how to use each rule with the SweWolf Panel.
No external data dependencies. Pure display.
"""

import streamlit as st

# Cyberpunk palette (inline — no circular import)
_BG2     = "#0a0a1e"
_CYAN    = "#00ffff"
_GREEN   = "#00ff88"
_MAGENTA = "#ff00ff"
_RED     = "#ff3355"
_YELLOW  = "#ffdd00"
_BLUE    = "#00aaff"
_TEXT    = "#e0e0ff"
_DIM     = "#4a4a6a"

# ------------------------------------------------------------------ #
#  Rule data
# ------------------------------------------------------------------ #

SWING_RULES: list[dict] = [
    {
        "number": 1,
        "text": "Handla endast i trendens riktning",
        "explanation": "Upptrend = long. Nedtrend = short. Aldrig mot trenden.",
        "panel_guide": "TRADE SCREENER → Kontrollera Regime Score. Grön = long. Röd = short. OVTLYR → Trend-kort visar Bullish/Bearish.",
    },
    {
        "number": 2,
        "text": "Ta inga trades i konsolidering",
        "explanation": "Range = förbjudet område. Vänta på breakout.",
        "panel_guide": "OVTLYR → Om priset ligger mellan EMA 10 och EMA 50 utan riktning, och RSI pendlar 40-60 = konsolidering. Vänta.",
    },
    {
        "number": 3,
        "text": "En trade kräver en key level",
        "explanation": "Supply/demand eller tydligt stöd/motstånd.",
        "panel_guide": "OVTLYR → Order Blocks i grafen markerar supply/demand-zoner. Grön zon = stöd. Röd zon = motstånd.",
    },
    {
        "number": 4,
        "text": "Entry endast efter pullback",
        "explanation": "Inga impulsiva entries i rakt fall eller rally.",
        "panel_guide": "OVTLYR → Vänta tills pris drar tillbaka till EMA 10/20 eller till en bullish OB-zon innan entry.",
    },
    {
        "number": 5,
        "text": "Candlestick-trigger krävs",
        "explanation": "Pinbar, engulfing eller break-and-retest.",
        "panel_guide": "Använd TradingView med Deepthought-skriptet för candlestick-analys. SweWolf visar signalen, TV bekräftar mönstret.",
    },
    {
        "number": 6,
        "text": "Volym måste bekräfta rörelsen",
        "explanation": "Ingen volym = ingen trade.",
        "panel_guide": "OVTLYR → Momentum-kort visar Vol ratio. Över 1.3x = bekräftad. Under 0.8x = svag, undvik.",
    },
    {
        "number": 7,
        "text": "Minsta R/R är 1:2",
        "explanation": "Helst 1:3. Aldrig under 1:2.",
        "panel_guide": "Beräkna manuellt: SL-avstånd (½ ATR) vs närmaste motstånd/OB-zon. Om motståndet är inom 2× SL = skippa.",
    },
    {
        "number": 8,
        "text": "Max 1% risk per trade",
        "explanation": "SL baseras på struktur, aldrig procent.",
        "panel_guide": "OVTLYR → Volatility-kort visar ATR 14. Position size = (Kapital × 1%) / (½ ATR). Aldrig mer.",
    },
    {
        "number": 9,
        "text": "Flytta SL till BE först efter ny HH/LL",
        "explanation": "Inte tidigare, inte senare.",
        "panel_guide": "Bevaka i TradingView. När ny swing high skapas ovanför din entry = flytta SL till entry.",
    },
    {
        "number": 10,
        "text": "Max två förluster per dag",
        "explanation": "Stoppa dagen direkt efter två minus.",
        "panel_guide": "Håll daglig logg. Två förluster = stäng av datorn. SweWolf Panelen analyserar — du handlar nästa dag.",
    },
    {
        "number": 11,
        "text": "Exit: Kijun-sen trail + ½ ATR hård stop",
        "explanation": "Ichimoku Kijun-sen (26p) som dynamiskt trailing stop. Stäng om pris stänger under Kijun OCH under EMA 10. ½ ATR som absolut nödstopp.",
        "panel_guide": "REGIME MONITOR → Ichimoku-gaugen visar Kijun-nivå. OVTLYR → EMA 10 i grafen. Exit när BÅDA bryts. Wolf v4 Pine Script hanterar detta automatiskt med Kijun trail + BE efter swing high.",
    },
]

LONGTERM_RULES: list[dict] = [
    {
        "number": 1,
        "text": "Köp endast i grön regim",
        "explanation": "Regimindikatorn måste vara grön.",
        "panel_guide": "REGIME MONITOR → Total score > 70 = grön. OVTLYR → Regime-badge visar GREEN/ORANGE/RED.",
    },
    {
        "number": 2,
        "text": "Pris måste ligga över 200 EMA",
        "explanation": "Bekräftar långsiktig upptrend.",
        "panel_guide": "OVTLYR → Trend-kort visar EMA 200-värde. LONG SCREENER → Tech-poäng inkluderar 'Price > EMA200'.",
    },
    {
        "number": 3,
        "text": "50 EMA måste ligga över 200 EMA",
        "explanation": "Golden cross = positivt momentum.",
        "panel_guide": "OVTLYR → Trend-kort: EMA 50 > EMA 200. LONG SCREENER → Tech-poäng inkluderar 'EMA50 > EMA200'.",
    },
    {
        "number": 4,
        "text": "Sektorn måste vara grön",
        "explanation": "Ingen exponering i svaga sektorer.",
        "panel_guide": "SECTOR & REGIME → Sektorhjulet visar grön/gul/röd per sektor. LONG SCREENER → Cycle Score visar sektorstatus.",
    },
    {
        "number": 5,
        "text": "Fear & Greed under 60 vid köp",
        "explanation": "Undvik eufori och toppjakt.",
        "panel_guide": "SENTIMENT → Fear & Greed-gaugen. OVTLYR → Sentiment-kort. Under 60 = OK att köpa. Över 60 = vänta.",
    },
    {
        "number": 6,
        "text": "Minska vid EMA200-brott, sälj vid regimskifte",
        "explanation": "Stängning under EMA200 = reducera 50%. Om regim byter till röd = sälj resten. Ger andrum vid korta dippar.",
        "panel_guide": "OVTLYR → Bevaka EMA 200 i Trend-kortet. REGIME MONITOR → Om regim går från grön till orange = reducera. Orange till röd = sälj allt.",
    },
    {
        "number": 7,
        "text": "Sälj vid sektor + breadth crossover",
        "explanation": "Om sektorn OCH marknadsbredden vänder nedåt samtidigt = sälj positioner i den sektorn.",
        "panel_guide": "SECTOR & REGIME → Sektorhjul + OVTLYR NINE → Om sector score faller under 30% = sälj sektorpositioner.",
    },
    {
        "number": 8,
        "text": "Max 20-25% per sektor",
        "explanation": "Riskkontroll på portföljnivå.",
        "panel_guide": "LONG SCREENER → Sortera efter sektor. Räkna din totala allokering per sektor manuellt.",
    },
    {
        "number": 9,
        "text": "Max 10% per aktie",
        "explanation": "Ingen enskild position får dominera.",
        "panel_guide": "LONG SCREENER → STRONG BUY = 10% allokering, BUY = 7%. Aldrig mer per position.",
    },
    {
        "number": 10,
        "text": "Analysera alltid historiska nedgångar",
        "explanation": "Avgör om fallet är brus eller strukturellt.",
        "panel_guide": "LONG-TERM TREND → Drawdown-tabell klassificerar nedgångar som Noise/Fundamental/Macro/Sector. OVTLYR → Drawdowns sub-tab.",
    },
]

OVTLYR_ENTRY_RULES: list[dict] = [
    {"number": 1, "text": "Market Trend: SPY 10 DEMA > 20 DEMA, Price > 50 DEMA", "explanation": "Bullish = buy zone. Bearish = inga trades."},
    {"number": 2, "text": "Market Signal: Buy signal on $SPY", "explanation": "OVTLYR overlay måste vara grön."},
    {"number": 3, "text": "Market Breadth: Bull list 10EMA bullish crossover", "explanation": "Måste matcha market trend."},
    {"number": 4, "text": "Sector Breadth: Advancing", "explanation": "Bullish 10EMA cross krävs."},
    {"number": 5, "text": "Sector Fear & Greed: Advancing", "explanation": "Sektorsentiment måste förbättras."},
    {"number": 6, "text": "Stock Signal: Buy", "explanation": "OVTLYR signal måste visa Buy."},
    {"number": 7, "text": "Stock Trend: 10EMA/20EMA, Price > 50EMA", "explanation": "Alla EMA:er alignade för entry."},
    {"number": 8, "text": "Stock Fear & Greed: Advancing", "explanation": "Aktiens sentiment förbättras."},
    {"number": 9, "text": "Order Blocks: Inga restriktiva OBs", "explanation": "Inga bearish OBs blockerar vägen."},
    {"number": 10, "text": "Momentum: Pris ovanför gårdagens lägsta", "explanation": "Bekräftar positivt momentum."},
]

OVTLYR_EXIT_RULES: list[dict] = [
    {"number": 1, "text": "$SPY stänger under 20 EMA → STÄNG ALLT", "explanation": "Hård exit. Inga undantag."},
    {"number": 2, "text": "½ ATR Stop Loss från entry", "explanation": "Strukturbaserad stop."},
    {"number": 3, "text": "10 EMA Trailing Stop", "explanation": "Pris stänger under 10 EMA = exit."},
    {"number": 4, "text": "Order Block hit", "explanation": "Pris springer in i restriktivt OB = exit."},
    {"number": 5, "text": "Gap & Crap", "explanation": "Gap up följt av reversal = omedelbar exit."},
    {"number": 6, "text": "Stängning under gårdagens lägsta", "explanation": "Efter rolling: exit."},
    {"number": 7, "text": "Sektor + Market breadth crossover", "explanation": "Sälj alla trades i sektorn."},
    {"number": 8, "text": "Stock Sell signal", "explanation": "OVTLYR signal flippar till Sell."},
    {"number": 9, "text": "Fear & Greed target hit", "explanation": "0-50: exit vid 63. 50-75: 10pt spread. 75+: 5pt spread."},
    {"number": 10, "text": "Earnings risk", "explanation": "Stäng position före rapportdag."},
]

OVTLYR_MINDSET: list[dict] = [
    {"number": 1, "text": "Det finns INGA FÖRVÄNTNINGAR på utfallet", "explanation": "Handla planen, inte prediktionen."},
    {"number": 2, "text": "Det finns INGA VINSTMÅL", "explanation": "Låt exit-signalerna göra sitt jobb."},
    {"number": 3, "text": "Jag har bara en plan att ta mig ur när en exit-signal triggar", "explanation": "Planen ÄR din edge."},
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
            f"<div style='color:{_CYAN};font-size:0.65rem;margin-top:4px;"
            f"padding:4px 8px;background:rgba(0,255,255,0.05);border-radius:3px;'>"
            f"PANEL: {guide}</div>"
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
#  Main render function
# ------------------------------------------------------------------ #

def render_rules_page() -> None:
    """Display all trading rules with SweWolf Panel usage guides."""

    st.markdown(
        f"<div style='text-align:center;padding:20px 0 10px 0;'>"
        f"<h1 style='color:{_CYAN};letter-spacing:0.15em;margin:0;'>TRADING RULES</h1>"
        f"<p style='color:{_DIM};font-size:0.75rem;letter-spacing:0.12em;'>"
        f"The rules that govern every trade. No exceptions. "
        f"<span style='color:{_CYAN};'>PANEL</span>-guider visar hur varje regel kontrolleras i SweWolf.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Two-column layout: Swing + Long-term ──────────────────────────
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown(
            _section_header_html(
                "Swing Trading",
                "11 regler — kortsiktig taktik + Ichimoku exit",
                _CYAN,
            ),
            unsafe_allow_html=True,
        )
        cards_html = "".join(_rule_card_html(r, _CYAN) for r in SWING_RULES)
        st.markdown(cards_html, unsafe_allow_html=True)

    with right_col:
        st.markdown(
            _section_header_html(
                "Långsiktig Trend / Regim",
                "10 regler — strategisk position",
                _GREEN,
            ),
            unsafe_allow_html=True,
        )
        cards_html = "".join(_rule_card_html(r, _GREEN) for r in LONGTERM_RULES)
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Divider ───────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:rgba(0,255,255,0.13);margin:30px 0;'/>",
        unsafe_allow_html=True,
    )

    # ── OVTLYR Golden Ticket ──────────────────────────────────────────
    st.markdown(
        f"<div style='text-align:center;margin-bottom:20px;'>"
        f"<h2 style='color:{_BLUE};letter-spacing:0.15em;'>OVTLYR GOLDEN TICKET</h2>"
        f"<p style='color:{_DIM};font-size:0.75rem;letter-spacing:0.1em;'>"
        f"GOLDEN TICKET TRADING STRATEGY — WHERE OUTLIERS WIN</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    oc1, oc2, oc3 = st.columns([1, 1, 0.6])

    with oc1:
        st.markdown(
            _section_header_html("Open (Long)", "10 entry-regler", _BLUE),
            unsafe_allow_html=True,
        )
        cards_html = "".join(_rule_card_html(r, _BLUE) for r in OVTLYR_ENTRY_RULES)
        st.markdown(cards_html, unsafe_allow_html=True)

    with oc2:
        st.markdown(
            _section_header_html("Close (Long)", "10 exit-regler", _RED),
            unsafe_allow_html=True,
        )
        cards_html = "".join(_rule_card_html(r, _RED) for r in OVTLYR_EXIT_RULES)
        st.markdown(cards_html, unsafe_allow_html=True)

    with oc3:
        st.markdown(
            _section_header_html("Mindset", "3 gyllene regler", _YELLOW),
            unsafe_allow_html=True,
        )
        cards_html = "".join(_rule_card_html(r, _YELLOW) for r in OVTLYR_MINDSET)
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='text-align:center;padding:30px 0 10px 0;'>"
        f"<span style='color:{_BLUE};font-size:0.7rem;letter-spacing:0.15em;'>"
        f"SAVE TIME &middot; MAKE MONEY &middot; START WINNING WITH LESS RISK</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Panel usage guide ─────────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:rgba(0,255,255,0.13);margin:20px 0;'/>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='text-align:center;margin-bottom:16px;'>"
        f"<h2 style='color:{_MAGENTA};letter-spacing:0.12em;'>SWEWOLF PANEL — REGELGUIDE</h2>"
        f"<p style='color:{_DIM};font-size:0.7rem;'>Så här kontrollerar du varje regel med panelens flikar</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    guide_data = [
        ("TRADE SCREENER", "Swing #1, #6", "Regime Score visar trendriktning. Volymbekräftelse i scan-resultat."),
        ("BACKTEST", "Swing #7", "Verifiera R:R-kvot historiskt. Kör backtest med dina parametrar."),
        ("RS BACKTEST", "Swing #1", "Momentum-filtrerade sektorer. Bara starkaste tickers."),
        ("OVTLYR", "Swing #2-4, #6, #8, #11 | Lång #1-5", "Order Blocks = key levels. EMA 10/20/50/200 i grafen. Regime-badge. F&G. ATR för position sizing."),
        ("REGIME MONITOR", "Swing #1, #11 | Lång #1", "4-lagers regime (Market + Sector + Stock + Ichimoku). Kijun-nivå för trailing stop."),
        ("LONG SCREENER", "Lång #1-5, #8-9", "20-poängs fundamental + 7-poängs teknisk. STRONG BUY = alla gates passerar."),
        ("LONG-TERM TREND", "Lång #2-3, #6, #10", "EMA200/50 trend. Rick Rule-signaler. Drawdown-klassificering."),
        ("SECTOR & REGIME", "Swing #1 | Lång #4, #7", "Sektorhjul grön/röd. Global regime Risk-On/Off."),
        ("SENTIMENT", "Lång #5", "Fear & Greed gauge. Under 60 = OK köpa. Över 60 = vänta."),
        ("HEATMAP", "Lång #4, #8", "Performance per sektor/land. Identifiera starka/svaga sektorer."),
    ]

    guide_html = "<table style='width:100%;border-collapse:collapse;'>"
    guide_html += (
        f"<tr style='border-bottom:1px solid rgba(0,255,255,0.15);'>"
        f"<th style='text-align:left;color:{_CYAN};font-size:0.75rem;padding:8px;'>Flik</th>"
        f"<th style='text-align:left;color:{_CYAN};font-size:0.75rem;padding:8px;'>Regler</th>"
        f"<th style='text-align:left;color:{_CYAN};font-size:0.75rem;padding:8px;'>Användning</th>"
        f"</tr>"
    )
    for tab, rules, usage in guide_data:
        guide_html += (
            f"<tr style='border-bottom:1px solid rgba(74,74,106,0.2);'>"
            f"<td style='color:{_TEXT};font-size:0.8rem;padding:6px 8px;font-weight:700;'>{tab}</td>"
            f"<td style='color:{_YELLOW};font-size:0.72rem;padding:6px 8px;'>{rules}</td>"
            f"<td style='color:{_DIM};font-size:0.72rem;padding:6px 8px;'>{usage}</td>"
            f"</tr>"
        )
    guide_html += "</table>"
    st.markdown(guide_html, unsafe_allow_html=True)
