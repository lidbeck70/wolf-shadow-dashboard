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

/* ── Tab styling ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0D1117;
    border-bottom: 2px solid rgba(0,229,255,0.2);
    gap: 4px;
    padding: 0 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border: 1px solid rgba(0,229,255,0.15);
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    color: rgba(0,229,255,0.5);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 20px;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(0,229,255,0.12) 0%, transparent 100%);
    border: 1px solid #00E5FF;
    border-bottom: none;
    color: #00E5FF;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.stTabs [data-baseweb="tab"]:hover {
    color: #00E5FF;
    border-color: rgba(0,229,255,0.4);
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
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

[data-testid="stMetricDelta"] > div {
    font-size: 13px !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
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

.stButton > button:hover {
    border-color: #00E5FF !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.4) !important;
    color: #00E5FF !important;
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

.stSelectbox > div > div:focus,
.stTextInput > div > div > input:focus {
    border-color: #00E5FF !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
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
    border: 1px solid rgba(180,0,255,0.25) !important;
    border-radius: 6px !important;
    box-shadow: 0 0 8px rgba(180,0,255,0.1) !important;
}

/* ── Headers ─────────────────────────────────────────────────── */
h1, h2, h3 {
    color: #00E5FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    letter-spacing: 3px;
}

/* ── Progress bar ────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00E5FF, #00A8BF);
}

/* ── Alerts/info ─────────────────────────────────────────────── */
.stAlert {
    background-color: rgba(0,229,255,0.07) !important;
    border: 1px solid rgba(0,229,255,0.25) !important;
    color: #00E5FF !important;
}

/* ── Dividers ────────────────────────────────────────────────── */
hr {
    border-color: rgba(0,229,255,0.15) !important;
}

/* ── Custom banner ───────────────────────────────────────────── */
.wolf-banner {
    background: linear-gradient(135deg, #0D1117 0%, #0D1117 50%, #0D1117 100%);
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
.score-yellow { color: #FF6B3D; font-weight: 700; }
.score-red    { color: #FF6B3D; font-weight: 700; }
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

/* ── iPad & Mobile Responsive ─────────────────────────────────── */
@media (max-width: 1024px) {
    .stTabs [data-baseweb="tab"] {
        font-size: 10px !important;
        letter-spacing: 1px !important;
        padding: 8px 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        flex-wrap: nowrap;
    }
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 9px !important;
    }
}

@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] {
        font-size: 9px !important;
        padding: 6px 8px !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 16px !important;
    }
    [data-testid="column"] {
        min-width: 100% !important;
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

/* Scrollable tab bar on iPad */
.stTabs [data-baseweb="tab-list"] {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
.stTabs [data-baseweb="tab"] {
    white-space: nowrap;
    flex-shrink: 0;
}

</style>
"""


def inject_css():
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


def wolf_banner():
    try:
        _banner_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "banner.jpg")
        if os.path.exists(_banner_path):
            with open(_banner_path, "rb") as _bf:
                _banner_b64 = base64.b64encode(_bf.read()).decode()
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
        return f'<span class="entry-yes">✦ YES</span>'
    return f'<span class="entry-no">·no·</span>'


def tab_not_found(module_name: str, folder: str):
    st.warning(f"{module_name} module not found. Make sure the `{folder}/` folder is in the dashboard directory.")
    st.code(f"Expected: dashboard/{folder}/")
