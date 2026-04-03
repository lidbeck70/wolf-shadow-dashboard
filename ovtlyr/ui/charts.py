"""
Plotly chart functions for the OVTLYR dashboard.

Design rules:
- template="plotly_dark" on all figures
- paper_bgcolor="#0a0a1e", plot_bgcolor="#050510"
- All transparent fills use rgba() — NEVER 8-digit hex
- Order block zones use fig.add_shape(type="rect")
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from typing import Optional

# Cyberpunk palette (inline to avoid circular import)
_BG       = "#050510"
_BG2      = "#0a0a1e"
_CYAN     = "#00ffff"
_MAGENTA  = "#ff00ff"
_GREEN    = "#00ff88"
_RED      = "#ff3355"
_YELLOW   = "#ffdd00"
_TEXT     = "#e0e0ff"
_DIM      = "#4a4a6a"

_PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=_BG2,
    plot_bgcolor=_BG,
)


# ------------------------------------------------------------------ #
#  build_price_chart
# ------------------------------------------------------------------ #

def build_price_chart(
    df: pd.DataFrame,
    trend: dict,
    orderblocks: list[dict],
    volume_data: Optional[dict] = None,
) -> go.Figure:
    """
    Main price chart with:
    - Candlestick OHLC
    - EMA 50 (yellow dotted) and EMA 200 (magenta solid)
    - Order block zones as semi-transparent rectangles
    - Volume subplot (20% height) with up/down color bars
    - Current price horizontal dashed line

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Date (or index), Open, High, Low, Close, Volume
        Date should be datetime-parseable.
    trend : dict
        Keys: ema50 (list or float), ema200 (list or float), price (float)
    orderblocks : list[dict]
        Each dict: {type: "bullish"|"bearish", date_start, date_end,
                    high, low, volume, status: "active"|"tested"|"invalidated"}
    volume_data : dict, optional
        Unused — volume is taken from df["Volume"] directly.

    Returns
    -------
    go.Figure — two-row subplot (80/20 price/volume)
    """

    # Normalise index to a column
    if "Date" not in df.columns and df.index.name in ("Date", "date", None):
        df = df.reset_index()
        if df.columns[0] not in ("Date",):
            df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.02,
    )

    # ── Candlestick ──────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color=_GREEN,
            decreasing_line_color=_RED,
            increasing_fillcolor=_GREEN,
            decreasing_fillcolor=_RED,
        ),
        row=1, col=1,
    )

    # ── EMA 200 (magenta solid) ───────────────────────────────────────
    ema200 = trend.get("ema200")
    if ema200 is not None:
        y200 = ema200 if hasattr(ema200, "__iter__") else [ema200] * len(df)
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=list(y200)[-len(df):] if len(list(y200)) > len(df) else y200,
                name="EMA 200",
                line=dict(color=_MAGENTA, width=1.8, dash="solid"),
                opacity=0.85,
            ),
            row=1, col=1,
        )

    # ── EMA 50 (yellow dotted) ────────────────────────────────────────
    ema50 = trend.get("ema50")
    if ema50 is not None:
        y50 = ema50 if hasattr(ema50, "__iter__") else [ema50] * len(df)
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=list(y50)[-len(df):] if len(list(y50)) > len(df) else y50,
                name="EMA 50",
                line=dict(color=_YELLOW, width=1.4, dash="dot"),
                opacity=0.85,
            ),
            row=1, col=1,
        )

    # ── Current price line ────────────────────────────────────────────
    current_price = float(trend.get("price", df["Close"].iloc[-1]))
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color=_CYAN,
        line_width=1,
        opacity=0.6,
        row=1, col=1,
        annotation_text=f"  {current_price:.2f}",
        annotation_font_color=_CYAN,
        annotation_font_size=11,
    )

    # ── Order Block zones ─────────────────────────────────────────────
    for ob in (orderblocks or []):
        ob_type   = ob.get("type", "bullish").lower()
        status    = ob.get("status", "active").lower()
        date_start = ob.get("date_start")
        date_end   = ob.get("date_end")
        ob_high   = ob.get("high", 0)
        ob_low    = ob.get("low", 0)
        ob_vol    = ob.get("volume", 0)

        if not date_start or not date_end:
            continue

        # Alpha depends on status
        if status == "invalidated":
            alpha_fill = 0.05
            alpha_line = 0.2
        elif status == "tested":
            alpha_fill = 0.10
            alpha_line = 0.30
        else:  # active
            alpha_fill = 0.15
            alpha_line = 0.40

        if ob_type == "bullish":
            fill_color = f"rgba(0,255,136,{alpha_fill})"
            line_color = f"rgba(0,255,136,{alpha_line})"
            label_color = _GREEN
        else:
            fill_color = f"rgba(255,51,85,{alpha_fill})"
            line_color = f"rgba(255,51,85,{alpha_line})"
            label_color = _RED

        fig.add_shape(
            type="rect",
            x0=str(date_start),
            x1=str(date_end),
            y0=ob_low,
            y1=ob_high,
            fillcolor=fill_color,
            line=dict(color=line_color, width=1),
            row=1, col=1,
        )

        # Annotation for hover-like label
        fig.add_annotation(
            x=str(date_end),
            y=ob_high,
            text=(
                f"{ob_type.upper()} OB<br>"
                f"H:{ob_high:.2f} L:{ob_low:.2f}<br>"
                f"Vol:{ob_vol:,.0f} [{status}]"
            ),
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color=label_color),
            bgcolor=_BG2,
            bordercolor=line_color,
            borderwidth=1,
            opacity=0.8,
            row=1, col=1,
        )

    # ── Volume bars ───────────────────────────────────────────────────
    if "Volume" in df.columns:
        up_mask   = df["Close"] >= df["Open"]
        bar_colors = [
            f"rgba(0,255,136,0.5)" if up else f"rgba(255,51,85,0.5)"
            for up in up_mask
        ]
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                name="Volume",
                marker_color=bar_colors,
                showlegend=False,
            ),
            row=2, col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        **_PLOTLY_BASE,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=24, b=0),
        legend=dict(
            orientation="h",
            x=0, y=1.01,
            font=dict(color=_TEXT, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=480,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(74,74,106,0.3)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(74,74,106,0.3)",
        zeroline=False,
    )

    return fig


# ------------------------------------------------------------------ #
#  build_sentiment_gauge
# ------------------------------------------------------------------ #

def build_sentiment_gauge(score: int, label: str = "Fear & Greed") -> go.Figure:
    """
    Gauge chart 0–100 with colored zones:
      0–25 red (extreme fear), 26–45 yellow (fear),
      46–55 text/neutral, 56–75 yellow (greed), 76–100 red (extreme greed)

    Parameters
    ----------
    score : int  — 0–100
    label : str  — title shown below gauge
    """

    if score <= 25:
        needle_color = _RED
        zone_label   = "Extreme Fear"
    elif score <= 45:
        needle_color = _YELLOW
        zone_label   = "Fear"
    elif score <= 55:
        needle_color = _TEXT
        zone_label   = "Neutral"
    elif score <= 75:
        needle_color = _YELLOW
        zone_label   = "Greed"
    else:
        needle_color = _RED
        zone_label   = "Extreme Greed"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(
            text=f"<b>{label}</b><br><span style='font-size:0.8em;color:{needle_color}'>{zone_label}</span>",
            font=dict(color=_TEXT, size=14),
        ),
        number=dict(font=dict(color=needle_color, size=28)),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor=_DIM,
                tickfont=dict(color=_DIM, size=10),
            ),
            bar=dict(color=needle_color, thickness=0.25),
            bgcolor=_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, 25],   color="rgba(255,51,85,0.25)"),
                dict(range=[25, 45],  color="rgba(255,221,0,0.15)"),
                dict(range=[45, 55],  color="rgba(224,224,255,0.08)"),
                dict(range=[55, 75],  color="rgba(255,221,0,0.15)"),
                dict(range=[75, 100], color="rgba(255,51,85,0.25)"),
            ],
            threshold=dict(
                line=dict(color=_CYAN, width=2),
                thickness=0.75,
                value=score,
            ),
        ),
    ))

    fig.update_layout(
        **_PLOTLY_BASE,
        height=220,
        margin=dict(l=16, r=16, t=40, b=8),
    )
    return fig


# ------------------------------------------------------------------ #
#  build_sector_pie
# ------------------------------------------------------------------ #

def build_sector_pie(breadth_data: dict) -> go.Figure:
    """
    Sector pie chart colored by trend state.

    Parameters
    ----------
    breadth_data : dict
        Keys are sector names, values are dicts with:
          state: "bullish" | "neutral" | "bearish"
          weight: float (0–100, portfolio %)
    """

    labels  = []
    values  = []
    colors  = []

    for sector, info in breadth_data.items():
        labels.append(sector)
        values.append(info.get("weight", 5.0))
        state = info.get("state", "neutral").lower()
        if state == "bullish":
            colors.append(_GREEN)
        elif state == "bearish":
            colors.append(_RED)
        else:
            colors.append(_YELLOW)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=colors,
            line=dict(color=_BG2, width=2),
        ),
        textfont=dict(color=_TEXT, size=11),
        hole=0.45,
    ))

    fig.update_layout(
        **_PLOTLY_BASE,
        height=300,
        margin=dict(l=0, r=0, t=24, b=0),
        legend=dict(
            font=dict(color=_TEXT, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ------------------------------------------------------------------ #
#  build_heatmap
# ------------------------------------------------------------------ #

def build_heatmap(breadth_data: dict) -> go.Figure:
    """
    Sector heatmap grid colored green/yellow/red by state.

    Parameters
    ----------
    breadth_data : dict
        Keys are sector names, values are dicts:
          state: "bullish"|"neutral"|"bearish"
          change: float (% change for the period)
    """

    sectors = list(breadth_data.keys())
    if not sectors:
        fig = go.Figure()
        fig.update_layout(**_PLOTLY_BASE, height=200)
        return fig

    changes = [breadth_data[s].get("change", 0.0) for s in sectors]
    states  = [breadth_data[s].get("state", "neutral") for s in sectors]

    # Numeric encoding for color scale: bullish=1, neutral=0, bearish=-1
    z_vals = [
        1.0 if s == "bullish" else (-1.0 if s == "bearish" else 0.0)
        for s in states
    ]

    # Text inside each cell
    text_vals = [
        f"{sec}<br>{chg:+.1f}%"
        for sec, chg in zip(sectors, changes)
    ]

    # Arrange into grid (try to make roughly square)
    import math
    n = len(sectors)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    # Pad to fill grid
    pad = nrows * ncols - n
    z_vals    += [0.0] * pad
    text_vals += [""] * pad
    sectors   += [""] * pad

    z_grid    = [z_vals[i*ncols:(i+1)*ncols] for i in range(nrows)]
    text_grid = [text_vals[i*ncols:(i+1)*ncols] for i in range(nrows)]

    fig = go.Figure(go.Heatmap(
        z=z_grid,
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=11, color=_TEXT),
        colorscale=[
            [0.0,  "rgba(255,51,85,0.6)"],
            [0.5,  "rgba(255,221,0,0.4)"],
            [1.0,  "rgba(0,255,136,0.6)"],
        ],
        zmin=-1, zmax=1,
        showscale=False,
        xgap=3,
        ygap=3,
    ))

    fig.update_layout(
        **_PLOTLY_BASE,
        height=max(160, nrows * 80),
        margin=dict(l=0, r=0, t=24, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig


# ------------------------------------------------------------------ #
#  build_risk_gauge
# ------------------------------------------------------------------ #

def build_risk_gauge(risk_score: int) -> go.Figure:
    """
    Compact gauge for risk score 0–100.

    0–33  green (low), 34–66 yellow (medium), 67–100 red (high).
    """

    if risk_score <= 33:
        needle_color = _GREEN
        risk_label   = "LOW"
    elif risk_score <= 66:
        needle_color = _YELLOW
        risk_label   = "MEDIUM"
    else:
        needle_color = _RED
        risk_label   = "HIGH"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title=dict(
            text=f"<b>Risk</b> <span style='color:{needle_color}'>{risk_label}</span>",
            font=dict(color=_TEXT, size=13),
        ),
        number=dict(font=dict(color=needle_color, size=26)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=9, color=_DIM)),
            bar=dict(color=needle_color, thickness=0.3),
            bgcolor=_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, 33],   color="rgba(0,255,136,0.15)"),
                dict(range=[33, 66],  color="rgba(255,221,0,0.15)"),
                dict(range=[66, 100], color="rgba(255,51,85,0.20)"),
            ],
        ),
    ))

    fig.update_layout(
        **_PLOTLY_BASE,
        height=180,
        margin=dict(l=12, r=12, t=36, b=8),
    )
    return fig


# ------------------------------------------------------------------ #
#  build_momentum_chart
# ------------------------------------------------------------------ #

def build_momentum_chart(
    df: pd.DataFrame,
    momentum_data: dict,
) -> go.Figure:
    """
    RSI chart with overbought / oversold zones marked.

    Parameters
    ----------
    df : pd.DataFrame
        Must have Date column and Close column (for computing RSI if not provided).
    momentum_data : dict
        Keys:
          rsi_series  – list/array of RSI values matching df length
          rsi         – current RSI float (for annotation)
          ob_level    – overbought threshold (default 70)
          os_level    – oversold threshold (default 30)
    """

    if "Date" not in df.columns:
        df = df.reset_index()

    df["Date"] = pd.to_datetime(df["Date"])

    rsi_series = momentum_data.get("rsi_series")
    rsi_current = float(momentum_data.get("rsi", 50))
    ob_level = float(momentum_data.get("ob_level", 70))
    os_level = float(momentum_data.get("os_level", 30))

    # Fallback: compute RSI from Close if not provided
    if rsi_series is None or len(rsi_series) == 0:
        close = df["Close"].astype(float)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("nan"))
        rsi_series = (100 - (100 / (1 + rs))).fillna(50).tolist()

    rsi_arr = list(rsi_series)[-len(df):]

    fig = go.Figure()

    # OB zone fill
    fig.add_hrect(
        y0=ob_level, y1=100,
        fillcolor="rgba(255,51,85,0.07)",
        line_width=0,
        annotation_text="OB",
        annotation_position="top right",
        annotation_font=dict(color=_RED, size=10),
    )

    # OS zone fill
    fig.add_hrect(
        y0=0, y1=os_level,
        fillcolor="rgba(0,255,136,0.07)",
        line_width=0,
        annotation_text="OS",
        annotation_position="bottom right",
        annotation_font=dict(color=_GREEN, size=10),
    )

    # OB / OS threshold lines
    fig.add_hline(y=ob_level, line_color=f"rgba(255,51,85,0.4)",  line_dash="dot", line_width=1)
    fig.add_hline(y=os_level, line_color=f"rgba(0,255,136,0.4)", line_dash="dot", line_width=1)
    fig.add_hline(y=50,       line_color=f"rgba(74,74,106,0.5)",  line_dash="dot", line_width=1)

    # RSI line colored by zone
    rsi_color = _RED if rsi_current >= ob_level else (_GREEN if rsi_current <= os_level else _CYAN)

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=rsi_arr,
        name="RSI 14",
        line=dict(color=rsi_color, width=1.6),
        fill="tozeroy",
        fillcolor=f"rgba(0,255,255,0.04)",
    ))

    fig.update_layout(
        **_PLOTLY_BASE,
        height=220,
        margin=dict(l=0, r=0, t=24, b=0),
        yaxis=dict(range=[0, 100], tickfont=dict(size=10, color=_DIM)),
        xaxis=dict(tickfont=dict(size=10, color=_DIM)),
        legend=dict(font=dict(color=_TEXT, size=10), bgcolor="rgba(0,0,0,0)"),
    )

    return fig
