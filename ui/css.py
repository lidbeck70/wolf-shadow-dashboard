import os
import base64
import streamlit as st

CYBERPUNK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter+Tight:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset ────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter Tight', sans-serif;
}

h1, h2, h3,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif;
}

code, pre, .stCode, [class*="monospace"] {
    font-family: 'JetBrains Mono', monospace;
}

/* ── Background ──────────────────────────────────────────────── */
.stApp {
    background-color: #05070A;
    background-image:
        radial-gradient(ellipse at 15% 40%, rgba(0,229,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 40%, rgba(180,0,255,0.04) 0%, transparent 50%),
        repeating-linear-gradient(
            0deg, transparent, transparent 39px, rgba(26,31,37,0.8) 40px
        ),
        repeating-linear-gradient(
            90deg, transparent, transparent 39px, rgba(26,31,37,0.8) 40px
        );
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid rgba(0,229,255,0.15);
}

/* ── Flikar (st.tabs) ────────────────────────────────────────── */
/* Streamlit 1.59 ritar st.tabs med React Aria, inte BaseWeb — de gamla
   [data-baseweb="tab"]-reglerna matchade ingenting, så flikarna fick
   Streamlits standard: 14px och en röd list under den aktiva fliken.
   DOM:en i 1.59:
     div.stTabs[data-testid=stTabs] > div > div[role=tablist]
       > div[role=tab][data-testid=stTab][aria-selected]
         > (div[data-testid=stMarkdownContainer] > p) + div.react-aria-SelectionIndicator
   plus knapparna stTabsScrollLeft/stTabsScrollRight när raden inte får plats.
   [role="tablist"] i selektorn ger (0,3,0): slår sidornas egna
   ".stTabs [role=tab]"-regler (Viking-sidan), så flikraden ser likadan ut överallt. */
.stTabs [role="tablist"] {
    gap: 4px;
    overflow-x: auto;                    /* rullar i sidled i stället för att radbryta */
    flex-wrap: nowrap;
    overscroll-behavior-x: contain;
    -webkit-overflow-scrolling: touch;
    /* Baslinjen under raden. En inre skugga på den rullande raden står still
       och täcker hela den synliga bredden, hur långt raden än är rullad. */
    box-shadow: inset 0 -2px 0 rgba(0,229,255,0.2);
}
/* Streamlits egen baslinje är ett ::after lika brett som den synliga raden.
   Det rullar med flikarna och tar slut mitt i raden på telefonen — bort med det. */
.stTabs [role="tablist"]::after {
    content: none;
}

.stTabs [role="tablist"] > [role="tab"] {
    height: 44px;                        /* tryckyta 44px (Apple HIG) */
    padding: 0 14px;
    flex-shrink: 0;
    white-space: nowrap;
    border: none;
    border-radius: 6px 6px 0 0;
    background-color: transparent;
    color: #e8e4dc;                      /* TEXT — 16:1 mot #05070A */
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 600;
    transition: color 0.15s ease, background-color 0.15s ease;
}
/* Etiketten är en markdown-ruta (14px Source Sans) med ett p (vikt 400) —
   båda måste ärva från fliken, annars når storleken och vikten aldrig texten */
.stTabs [role="tablist"] > [role="tab"] [data-testid="stMarkdownContainer"],
.stTabs [role="tablist"] > [role="tab"] p {
    font-family: inherit;
    font-size: inherit;
    font-weight: inherit;
    color: inherit;
}
.stTabs [role="tablist"] > [role="tab"]:hover,
.stTabs [role="tablist"] > [role="tab"][data-focus-visible] {
    color: #00E5FF;
    background-color: rgba(0,229,255,0.06);
}
/* Tangentbordsfokus: cyan ring i stället för Streamlits röda. Ringen ligger
   innanför fliken — raden klipper allt som sticker ut ovanför och under. */
.stTabs [role="tablist"] > [role="tab"][data-focus-visible] {
    box-shadow: inset 0 0 0 2px rgba(0,229,255,0.5);
}

/* Aktiv flik: cyan text på cyan ton och en tjock cyan list — i stället för
   Streamlits röda primärfärg. Listen är React Arias SelectionIndicator;
   klassnamnet är inte hashat (Streamlits egen CSS väljer på det). */
.stTabs [role="tablist"] > [role="tab"][aria-selected="true"] {
    color: #00E5FF;
    background-color: rgba(0,229,255,0.12);
    font-weight: 700;
}
.stTabs [role="tablist"] > [role="tab"] .react-aria-SelectionIndicator {
    height: 3px;
}
.stTabs [role="tablist"] > [role="tab"][aria-selected="true"] .react-aria-SelectionIndicator {
    background-color: #00E5FF;
}

/* Rullpilarna vid radens kanter: lika höga som flikarna, cyan, och tonar
   mot panelens bakgrund (#05070A) i stället för Streamlits #0e1117 */
.stTabs [data-testid="stTabsScrollLeft"],
.stTabs [data-testid="stTabsScrollRight"] {
    height: 44px;
    width: 32px;
    color: #00E5FF;
}
.stTabs [data-testid="stTabsScrollLeft"] {
    background-image: linear-gradient(to left, rgba(5,7,10,0), #05070A 40%);
}
.stTabs [data-testid="stTabsScrollRight"] {
    background-image: linear-gradient(to right, rgba(5,7,10,0), #05070A 40%);
}
.stTabs [data-testid="stTabsScrollLeft"]:focus-visible,
.stTabs [data-testid="stTabsScrollRight"]:focus-visible {
    box-shadow: inset 0 0 0 2px rgba(0,229,255,0.5);
}

/* ── Metric cards ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.06) 0%, rgba(0,168,191,0.03) 100%);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 8px;
    padding: 16px;
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00E5FF, #00A8BF);
}

[data-testid="stMetricLabel"] {
    color: rgba(0,229,255,0.6) !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #00E5FF !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

[data-testid="stMetricDelta"] > div {
    font-size: 13px !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
/* Bara sekundära knappar får panelens cyanram. type="primary" lämnas i fred,
   så sparradens "osparat = primär" faktiskt syns och en bekräfta-radera-knapp
   inte ser ut som "Lägg till". */
.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.15) 0%, rgba(0,168,191,0.1) 100%);
    border: 1px solid #00E5FF;
    border-radius: 4px;
    color: #00E5FF;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 12px 32px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, rgba(0,229,255,0.25) 0%, rgba(0,168,191,0.2) 100%);
    border-color: #00E5FF;
    color: #00E5FF;
    box-shadow: 0 0 12px rgba(0,229,255,0.4);
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.stButton > button:active {
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}

/* ── Inputs ──────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #1A1F25 !important;
    border: 1px solid rgba(0,229,255,0.25) !important;
    border-radius: 4px !important;
    color: #00E5FF !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Streamlit 1.59: rullgardinens ruta får aldrig :focus — React Aria sätter
   data-focus-within på div[role=group] i stället. */
.stSelectbox [role="group"][data-focus-within],
.stTextInput > div > div > input:focus {
    border-color: #00E5FF !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
}

/* ── Läsbar text i fält, rullgardiner och utfällbara rutor ───── */
/* Bakgrunderna ovan målas mörka oavsett Streamlit-tema, så texten måste
   också sättas här — annars ärvs ljusa temats #31333F (1,3:1 mot #1A1F25).
   .streamlit/config.toml tvingar mörkt tema; det här håller även utan det.
   Streamlit 1.59: st.selectbox = React Aria
   (div.stSelectbox > div.react-aria-ComboBox > div[role=group] > input). */

/* Värdet: valt värde i rullgardinen, inskrivet värde, tal, datum, textyta */
.stSelectbox input,
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTextArea textarea,
.stMultiSelect input {
    color: #00E5FF !important;
    -webkit-text-fill-color: #00E5FF !important;   /* iOS Safari */
    caret-color: #00E5FF !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

/* Fältens ytor som Streamlit annars ger secondaryBg (#f0f2f6 i ljust tema) */
[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputContainer"],
[data-testid="stTextAreaRootElement"],
.stDateInput [data-baseweb="input"],
.stMultiSelect [data-baseweb="select"] > div {
    background-color: #1A1F25 !important;
    border-color: rgba(0,229,255,0.25) !important;
}

/* Platshållaren ("Skriv en ticker …", "Choose an option") — ljusare än
   DIM (#8a8578 ger bara 4,5:1 mot #1A1F25), #a8a397 ger 6,6:1 */
.stSelectbox input::placeholder,
.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stDateInput input::placeholder,
.stTextArea textarea::placeholder,
.stMultiSelect input::placeholder {
    color: #a8a397 !important;
    -webkit-text-fill-color: #a8a397 !important;
    opacity: 1 !important;
    font-weight: 400 !important;
}

/* Pilen i rullgardinen och +/- i talfältet */
.stSelectbox [role="group"] button,
[data-testid="stNumberInputContainer"] button {
    color: #00E5FF !important;
}

/* Låsta fält: dämpat men läsbart (DIM) */
.stSelectbox input:disabled,
.stTextInput input:disabled,
.stNumberInput input:disabled,
.stDateInput input:disabled,
.stTextArea textarea:disabled {
    color: #8a8578 !important;
    -webkit-text-fill-color: #8a8578 !important;
}

/* Den utfällda listan (portal i <body>, gäller selectbox och multiselect) */
[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #0D1117 !important;
    border: 1px solid rgba(0,229,255,0.25) !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"] {
    color: #e8e4dc !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused] [data-item-hl],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered] [data-item-hl],
[data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover {
    background-color: rgba(0,229,255,0.12) !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {
    color: #00E5FF !important;
}

/* Utfällbara rutor (st.expander) — bakgrunden tvingas, annars blir den
   öppna rubriken en ljus list (#f8f9fb) där TEXT inte syns */
[data-testid="stExpander"] details {
    border-color: rgba(0,229,255,0.2) !important;
}
[data-testid="stExpander"] summary {
    background-color: transparent !important;
}
[data-testid="stExpander"] details[open] > summary,
[data-testid="stExpander"] summary:hover {
    background-color: rgba(0,229,255,0.06) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    color: #e8e4dc !important;
}
[data-testid="stExpander"] summary p {
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover p,
[data-testid="stExpander"] summary:hover [data-testid="stIconMaterial"] {
    color: #00E5FF !important;
}

/* Etiketter och radio-/kryssval — på label-elementet (p ärver), så sidornas
   egna ".stSelectbox label { color: CYAN !important }" (heatmap, cagr) fortfarande vinner */
[data-testid="stWidgetLabel"] {
    color: #e8e4dc !important;
}
.stRadio [role="radiogroup"] label p,
.stCheckbox label p {
    color: #e8e4dc !important;      /* p har egen färg i 1.59 — ärver inte */
}

/* Telefon: 16px i fälten — större text och ingen iOS-zoom vid fokus */
@media (max-width: 768px) {
    .stSelectbox input,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTextArea textarea {
        font-size: 16px !important;
    }
}

/* ── Slider ──────────────────────────────────────────────────── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00E5FF, #00A8BF) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #00E5FF !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
}

/* ── DataFrames / tables ─────────────────────────────────────── */
.stDataFrame {
    border: 1px solid rgba(0,229,255,0.15) !important;
    border-radius: 6px !important;
}

/* ── Headers ─────────────────────────────────────────────────── */
/* Ingen !important: sidorna sätter sin egen rubrikfärg (guld i arken, EMBER-
   orange, magenta i RULES) och den ska få synas. Cyan är bara standard. */
h1, h2, h3 {
    color: #00E5FF;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    letter-spacing: 3px;
}

/* ── Progress bar ────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00E5FF, #00A8BF);
}

/* (st.error/st.warning/st.success behåller sina egna färger — en cyan
   !important på .stAlert gjorde fel, varning och OK omöjliga att skilja.) */

/* ── Dividers ────────────────────────────────────────────────── */
hr {
    border-color: rgba(0,229,255,0.15) !important;
}

/* ── Custom banner ───────────────────────────────────────────── */
.wolf-banner {
    background: linear-gradient(135deg, #0D1117 0%, #080D14 50%, #0D1117 100%);
    border: 1px solid rgba(0,229,255,0.2);
    border-top: 3px solid #00E5FF;
    border-radius: 8px;
    padding: 20px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.wolf-banner::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00A8BF, transparent);
}

.wolf-banner h1 {
    font-size: 36px !important;
    font-weight: 900 !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
    background: linear-gradient(90deg, #00E5FF, #00A8BF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none !important;
    letter-spacing: 6px !important;
}

.wolf-banner p {
    color: rgba(0,229,255,0.5);
    font-size: 11px;
    letter-spacing: 4px;
    margin: 0;
    text-transform: uppercase;
}

/* ── Score badges ────────────────────────────────────────────── */
.score-green  { color: #2d8a4e; font-weight: 700; }
.score-yellow { color: #d4943a; font-weight: 700; }
.score-red    { color: #c44545; font-weight: 700; }
.entry-yes    { color: #2d8a4e; font-weight: 700; letter-spacing: 1px; }
.entry-no     { color: rgba(232,228,220,0.25); }

/* ── Regime gauge label ──────────────────────────────────────── */
.regime-score {
    font-size: 72px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(180deg, #00E5FF, #00A8BF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    text-shadow: none;
    font-family: 'JetBrains Mono', monospace;
}

.regime-label {
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    text-align: center;
    color: rgba(0,229,255,0.5);
    margin-top: 4px;
}

/* ── Section titles ──────────────────────────────────────────── */
.section-title {
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(0,229,255,0.4);
    border-bottom: 1px solid rgba(0,229,255,0.1);
    padding-bottom: 6px;
    margin-bottom: 16px;
}

/* ── Status pill ─────────────────────────────────────────────── */
.status-active {
    background: rgba(45,138,78,0.15);
    border: 1px solid #2d8a4e;
    border-radius: 20px;
    color: #2d8a4e;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 4px 14px;
    text-transform: uppercase;
    display: inline-block;
}

.status-inactive {
    background: rgba(232,228,220,0.05);
    border: 1px solid rgba(232,228,220,0.15);
    border-radius: 20px;
    color: rgba(232,228,220,0.3);
    font-size: 11px;
    letter-spacing: 2px;
    padding: 4px 14px;
    text-transform: uppercase;
    display: inline-block;
}

/* ── Cards / Aurora borders ──────────────────────────────────── */
.arc-card {
    border: 1px solid rgba(180,0,255,0.25);
    box-shadow: 0 0 8px rgba(180,0,255,0.1);
    border-radius: 8px;
}

/* ── iPad & Mobile Responsive ─────────────────────────────────── */
@media (max-width: 1024px) {
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 9px !important;
    }
}

@media (max-width: 768px) {
    /* Telefon: flikarna behåller 44px tryckyta, lite mindre text och
       luft så att fler syns innan raden rullar */
    .stTabs [role="tablist"] > [role="tab"] {
        font-size: 15px;
        padding: 0 12px;
    }
    [data-testid="stMetricValue"] {
        font-size: 16px !important;
    }
}

/* Touch-friendly targets (44px minimum per Apple HIG) */
.stButton > button,
.stSelectbox > div {
    min-height: 44px;
}

/* Smooth scrolling iOS */
[data-testid="stAppViewContainer"] {
    -webkit-overflow-scrolling: touch;
}

</style>
"""


def inject_css():
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


def wolf_banner():
    try:
        from ui.theme import _banner_b64 as _cached_banner   # kodas en gång per process
        _banner_b64 = _cached_banner()
        if _banner_b64:
            st.markdown(
                f"<div style='text-align:center;margin:-1rem -1rem 1rem -1rem;padding:0;'>"
                f"<img src='data:image/jpeg;base64,{_banner_b64}' "
                f"style='width:100%;height:auto;border-radius:0 0 8px 8px;'/>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div class="wolf-banner">
                <h1>Nordic Arc Systems</h1>
                <p>See What the Market Can't. &nbsp;|&nbsp; Trading & Investing</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div class="wolf-banner">
            <h1>Nordic Arc Systems</h1>
            <p>See What the Market Can't.</p>
        </div>
        """, unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)


def color_score(val):
    if val >= 70:
        return f'<span class="score-green">{val}</span>'
    elif val >= 50:
        return f'<span class="score-yellow">{val}</span>'
    else:
        return f'<span class="score-red">{val}</span>'


def color_entry(val):
    if val == "YES":
        return '<span class="entry-yes">✦ YES</span>'
    return '<span class="entry-no">·no·</span>'


def tab_not_found(module_name: str, folder: str):
    st.warning(f"{module_name} module not found. Make sure the `{folder}/` folder is in the dashboard directory.")
    st.code(f"Expected: dashboard/{folder}/")
