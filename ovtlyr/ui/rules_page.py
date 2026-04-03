"""
Rules page — clean cyberpunk-styled display of all 20 trading rules.
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
    },
    {
        "number": 2,
        "text": "Ta inga trades i konsolidering",
        "explanation": "Range = förbjudet område. Vänta på breakout.",
    },
    {
        "number": 3,
        "text": "En trade kräver en key level",
        "explanation": "Supply/demand eller tydligt stöd/motstånd.",
    },
    {
        "number": 4,
        "text": "Entry endast efter pullback",
        "explanation": "Inga impulsiva entries i rakt fall eller rally.",
    },
    {
        "number": 5,
        "text": "Candlestick-trigger krävs",
        "explanation": "Pinbar, engulfing eller break-and-retest.",
    },
    {
        "number": 6,
        "text": "Volym måste bekräfta rörelsen",
        "explanation": "Ingen volym = ingen trade.",
    },
    {
        "number": 7,
        "text": "Minsta R/R är 1:2",
        "explanation": "Helst 1:3. Aldrig under 1:2.",
    },
    {
        "number": 8,
        "text": "Max 1% risk per trade",
        "explanation": "SL baseras på struktur, aldrig procent.",
    },
    {
        "number": 9,
        "text": "Flytta SL till BE först efter ny HH/LL",
        "explanation": "Inte tidigare, inte senare.",
    },
    {
        "number": 10,
        "text": "Max två förluster per dag",
        "explanation": "Stoppa dagen direkt efter två minus.",
    },
]

LONGTERM_RULES: list[dict] = [
    {
        "number": 1,
        "text": "Köp endast i grön regim",
        "explanation": "Regimindikatorn måste vara grön.",
    },
    {
        "number": 2,
        "text": "Pris måste ligga över 200 EMA",
        "explanation": "Bekräftar långsiktig upptrend.",
    },
    {
        "number": 3,
        "text": "50 EMA måste ligga över 200 EMA",
        "explanation": "Golden cross = positivt momentum.",
    },
    {
        "number": 4,
        "text": "Sektorn måste vara grön",
        "explanation": "Ingen exponering i svaga sektorer.",
    },
    {
        "number": 5,
        "text": "Fear & Greed under 60 vid köp",
        "explanation": "Undvik eufori och toppjakt.",
    },
    {
        "number": 6,
        "text": "Sälj när pris stänger under 200 EMA",
        "explanation": "Hård exitregel. Ingen diskussion.",
    },
    {
        "number": 7,
        "text": "Sälj när regim blir röd",
        "explanation": "Regimskifte = minska eller lämna.",
    },
    {
        "number": 8,
        "text": "Max 20-25% per sektor",
        "explanation": "Riskkontroll på portföljnivå.",
    },
    {
        "number": 9,
        "text": "Max 10% per aktie",
        "explanation": "Ingen enskild position får dominera.",
    },
    {
        "number": 10,
        "text": "Analysera alltid historiska nedgångar",
        "explanation": "Avgör om fallet är brus eller strukturellt.",
    },
]


# ------------------------------------------------------------------ #
#  Card renderer
# ------------------------------------------------------------------ #

def _rule_card_html(rule: dict, color: str) -> str:
    """Return an HTML string for a single rule card."""
    return (
        f'<div style="'
        f'background:{_BG2}; '
        f'border-left:3px solid {color}; '
        f'padding:12px 16px; '
        f'margin:8px 0; '
        f'border-radius:4px;">'
        f'<span style="color:{color}; font-size:1.1rem; font-weight:700;">#{rule["number"]}</span>'
        f'<span style="color:{_TEXT}; font-size:0.9rem; margin-left:12px;">{rule["text"]}</span>'
        f'<div style="color:{_DIM}; font-size:0.72rem; margin-top:4px;">{rule["explanation"]}</div>'
        f'</div>'
    )


def _section_header_html(title: str, subtitle: str, color: str) -> str:
    """Return an HTML section header."""
    return (
        f'<div style="'
        f'border-bottom:1px solid {color}44; '
        f'padding-bottom:8px; '
        f'margin-bottom:4px; '
        f'margin-top:8px;">'
        f'<span style="'
        f'color:{color}; '
        f'font-size:1.0rem; '
        f'font-weight:700; '
        f'letter-spacing:0.12em; '
        f'text-transform:uppercase;">'
        f'{title}</span>'
        f'<span style="'
        f'color:{_DIM}; '
        f'font-size:0.75rem; '
        f'margin-left:12px;">'
        f'{subtitle}</span>'
        f'</div>'
    )


# ------------------------------------------------------------------ #
#  Main render function
# ------------------------------------------------------------------ #

def render_rules_page() -> None:
    """
    Display all 20 trading rules in a clean cyberpunk-styled layout.

    Left column  — Swing Trading (10 rules, cyan accents)
    Right column — Long-term Trend/Regime (10 rules, green accents)
    Footer       — trading discipline motto
    """

    # Page header
    st.markdown(
        f'<h2 style="'
        f'color:{_CYAN}; '
        f'letter-spacing:0.15em; '
        f'font-size:1.4rem; '
        f'margin-bottom:4px;">'
        f'TRADING RULES</h2>'
        f'<p style="color:{_DIM}; font-size:0.8rem; margin-top:0; margin-bottom:20px;">'
        f'The 20 rules that govern every trade. No exceptions.</p>',
        unsafe_allow_html=True,
    )

    # Two-column layout
    left_col, right_col = st.columns([1, 1])

    # ── Left: Swing Rules ─────────────────────────────────────────────
    with left_col:
        st.markdown(
            _section_header_html(
                "Swing Trading",
                "10 regler — kortsiktig taktik",
                _CYAN,
            ),
            unsafe_allow_html=True,
        )
        cards_html = "".join(_rule_card_html(r, _CYAN) for r in SWING_RULES)
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Right: Long-term Rules ────────────────────────────────────────
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

    # ── Footer motto ──────────────────────────────────────────────────
    st.markdown(
        f'<div style="'
        f'text-align:center; '
        f'margin-top:32px; '
        f'padding:16px; '
        f'border-top:1px solid {_MAGENTA}33;">'
        f'<span style="'
        f'color:{_MAGENTA}88; '
        f'font-size:0.78rem; '
        f'letter-spacing:0.18em; '
        f'text-transform:uppercase; '
        f'font-weight:600;">'
        f'TRADE WITH DISCIPLINE — PROTECT CAPITAL — LET WINNERS RUN'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ================================================================== #
    #  OVTLYR GOLDEN TICKET SECTION
    # ================================================================== #

    # Visual separator
    st.markdown(
        "<hr style='border-color:rgba(0,255,255,0.13);margin:30px 0;'/>",
        unsafe_allow_html=True,
    )

    # OVTLYR header
    st.markdown(
        f"<div style='text-align:center;margin-bottom:20px;'>"
        f"<h2 style='color:{_BLUE};letter-spacing:0.15em;'>OVTLYR GOLDEN TICKET</h2>"
        f"<p style='color:{_DIM};font-size:0.75rem;letter-spacing:0.1em;'>GOLDEN TICKET TRADING STRATEGY — WHERE OUTLIERS WIN</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Three sub-sections: OPEN, CLOSE, MINDSET
    ovtlyr_col1, ovtlyr_col2, ovtlyr_col3 = st.columns([1, 1, 1])

    # ── OPEN (LONG) — ENTRY RULES  [blue accent] ─────────────────────
    with ovtlyr_col1:
        st.markdown(
            _section_header_html(
                "Open (Long)",
                "Entry rules — all 9 must pass",
                _BLUE,
            ),
            unsafe_allow_html=True,
        )
        _OPEN_RULES = [
            {"number": 1,  "text": "Market Trend: SPY 10 DEMA > 20 DEMA, Price > 50 DEMA",
             "explanation": "Bullish = buy zone. Bearish = no trades."},
            {"number": 2,  "text": "Market Signal: Buy signal on $SPY",
             "explanation": "OVTLYR overlay must be green."},
            {"number": 3,  "text": "Market Breadth: Bull list 10EMA bullish crossover",
             "explanation": "Must match market trend to continue."},
            {"number": 4,  "text": "Sector Breadth: Advancing",
             "explanation": "Bullish 10EMA cross required."},
            {"number": 5,  "text": "Sector Fear & Greed: Advancing",
             "explanation": "Sector sentiment must be improving."},
            {"number": 6,  "text": "Stock Signal: Buy",
             "explanation": "OVTLYR signal must show Buy."},
            {"number": 7,  "text": "Stock Trend: 10EMA/20EMA, Price > 50EMA",
             "explanation": "All EMAs aligned for entry."},
            {"number": 8,  "text": "Stock Fear & Greed: Advancing",
             "explanation": "Stock sentiment improving."},
            {"number": 9,  "text": "Order Blocks: No restrictive OBs",
             "explanation": "No bearish OBs blocking the path."},
            {"number": 10, "text": "Momentum: Price above previous day low",
             "explanation": "Confirms positive momentum."},
        ]
        cards_html = "".join(_rule_card_html(r, _BLUE) for r in _OPEN_RULES)
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── CLOSE (LONG) — EXIT RULES  [red accent] ──────────────────────
    with ovtlyr_col2:
        st.markdown(
            _section_header_html(
                "Close (Long)",
                "Exit rules — any one triggers close",
                _RED,
            ),
            unsafe_allow_html=True,
        )
        _CLOSE_RULES = [
            {"number": 1,  "text": "$SPY closes under 20 EMA → CLOSE ALL",
             "explanation": "Hard exit. No exceptions."},
            {"number": 2,  "text": "\u00bd ATR Stop Loss from entry",
             "explanation": "Structure-based stop, not percentage."},
            {"number": 3,  "text": "10 EMA Trailing Stop",
             "explanation": "Price closes below 10EMA = exit."},
            {"number": 4,  "text": "Order Block hit",
             "explanation": "Price runs into restrictive OB = exit."},
            {"number": 5,  "text": "Gap & Crap",
             "explanation": "Gap up followed by reversal = immediate exit."},
            {"number": 6,  "text": "Close below prior day's low",
             "explanation": "After rolling once, this triggers close."},
            {"number": 7,  "text": "Sector + Market breadth crossover",
             "explanation": "Sell all trades in that sector."},
            {"number": 8,  "text": "Stock Sell signal",
             "explanation": "OVTLYR signal flips to Sell."},
            {"number": 9,  "text": "Fear & Greed target hit",
             "explanation": "0-50: exit at 63. 50-75: 10pt spread. 75+: 5pt spread."},
            {"number": 10, "text": "Earnings risk",
             "explanation": "Close position before earnings release."},
        ]
        cards_html = "".join(_rule_card_html(r, _RED) for r in _CLOSE_RULES)
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── MINDSET — 3 GOLDEN RULES  [gold/yellow accent] ───────────────
    with ovtlyr_col3:
        st.markdown(
            _section_header_html(
                "Mindset",
                "3 golden rules — no exceptions",
                _YELLOW,
            ),
            unsafe_allow_html=True,
        )
        _MINDSET_RULES = [
            {"number": 1, "text": "There are NO EXPECTATIONS for the outcome",
             "explanation": "Trade the plan, not the prediction."},
            {"number": 2, "text": "There are NO PROFIT TARGETS",
             "explanation": "Let the exit signals do their job."},
            {"number": 3, "text": "I just have a plan to get out when one of my exit signals happens",
             "explanation": "The plan IS the edge."},
        ]
        cards_html = "".join(_rule_card_html(r, _YELLOW) for r in _MINDSET_RULES)
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── OVTLYR footer ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="'
        f'text-align:center; '
        f'margin-top:28px; '
        f'padding:14px; '
        f'border-top:1px solid rgba(0,170,255,0.18);">'
        f'<span style="'
        f'color:{_BLUE}; '
        f'font-size:0.78rem; '
        f'letter-spacing:0.18em; '
        f'text-transform:uppercase; '
        f'font-weight:600;">'
        f'SAVE TIME \u00b7 MAKE MONEY \u00b7 START WINNING WITH LESS RISK'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
