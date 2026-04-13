"""
Plotly chart functions for the OVTLYR dashboard.

Design rules:
- template="plotly_dark" on all figures
- paper_bgcolor="#14141e", plot_bgcolor="#0c0c12"
- All transparent fills use rgba() — NEVER 8-digit hex
- Order block zones use fig.add_shape(type="rect")
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List

# Nordic Gold palette
_BG       = "#0c0c12"
_BG2      = "#14141e"
_CYAN     = "#c9a84c"
_MAGENTA  = "#8b7340"
_GREEN    = "#2d8a4e"
_RED      = "#c44545"
_YELLOW   = "#d4943a"
_WHITE    = "#e8e4dc"
_ORANGE   = "#d4943a"
_TEXT     = "#e8e4dc"
_DIM      = "#8a8578"

_PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=_BG2,
    plot_bgcolor=_BG,
)


# ------------------------------------------------------------------ #
#  OrderBlock → chart-ready dict converter
# ------------------------------------------------------------------ #

def _ob_to_chart_dict(ob, df_dates=None) -> dict:
    """Convert an OrderBlock dataclass to the dict format charts expect.
    
    Handles both OrderBlock objects and raw dicts.
    Extends the OB zone forward in time to chart end or mitigation date.
    """
    if isinstance(ob, dict):
        return ob  # already a dict

    # Extract date from OrderBlock object
    ob_date = getattr(ob, "date", "")
    ob_type = getattr(ob, "type", "bullish")
    ob_high = getattr(ob, "high", 0)
    ob_low = getattr(ob, "low", 0)
    ob_vol = getattr(ob, "volume", 0)
    ob_status = getattr(ob, "status", "Active")
    mit_date = getattr(ob, "mitigation_date", "")

    # date_end: extend to mitigation date, or chart end
    if mit_date and ob_status in ("Mitigated", "Invalidated"):
        date_end = mit_date
    elif df_dates is not None and len(df_dates) > 0:
        date_end = str(df_dates[-1].date()) if hasattr(df_dates[-1], "date") else str(df_dates[-1])
    else:
        date_end = ob_date

    return {
        "type": ob_type,
        "date_start": ob_date,
        "date_end": date_end,
        "high": ob_high,
        "low": ob_low,
        "volume": ob_vol,
        "status": ob_status.lower(),
    }


# ------------------------------------------------------------------ #
#  build_price_chart — with EMA 10/20/50/200 + Order Blocks
# ------------------------------------------------------------------ #

def build_price_chart(
    df: pd.DataFrame,
    trend: dict,
    orderblocks: list,
    volume_data: Optional[dict] = None,
) -> go.Figure:
    """
    Main OVTLYR price chart:
    - Candlestick OHLC
    - EMA 10 (white thin), EMA 20 (orange thin), EMA 50 (yellow dotted), EMA 200 (magenta solid)
    - Order block zones as semi-transparent rectangles behind price
    - Volume subplot (20% height) with up/down colored bars
    - Current price horizontal dashed line
    """

    # Normalise index
    if "Date" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "Date":
            df = df.rename(columns={df.columns[0]: "Date"})
    df = df.copy()
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
            increasing_fillcolor="rgba(45,138,78,0.7)",
            decreasing_fillcolor="rgba(196,69,69,0.7)",
        ),
        row=1, col=1,
    )

    # ── Helper to align series to df ─────────────────────────────────
    def _align_series(val, label, color, width=1.4, dash="solid", opacity=0.85):
        if val is None:
            return
        if isinstance(val, (pd.Series, np.ndarray)):
            y = list(val)[-len(df):]
        elif isinstance(val, (list, tuple)):
            y = list(val)[-len(df):]
        else:
            y = [float(val)] * len(df)
        if len(y) < len(df):
            y = [None] * (len(df) - len(y)) + y
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=y,
                name=label,
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
            ),
            row=1, col=1,
        )

    # ── EMA 10 (white, thin) ─────────────────────────────────────────
    ema10 = trend.get("ema10")
    if ema10 is None and "Close" in df.columns:
        ema10 = df["Close"].ewm(span=10, adjust=False).mean()
    _align_series(ema10, "EMA 10", _WHITE, width=0.9, dash="solid", opacity=0.6)

    # ── EMA 20 (orange, thin) ────────────────────────────────────────
    ema20 = trend.get("ema20")
    if ema20 is None and "Close" in df.columns:
        ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    _align_series(ema20, "EMA 20", _ORANGE, width=1.0, dash="solid", opacity=0.65)

    # ── EMA 50 (yellow dotted) ───────────────────────────────────────
    _align_series(trend.get("ema50"), "EMA 50", _YELLOW, width=1.4, dash="dot", opacity=0.85)

    # ── EMA 200 (magenta solid) ──────────────────────────────────────
    _align_series(trend.get("ema200"), "EMA 200", _MAGENTA, width=1.8, dash="solid", opacity=0.85)

    # ── Current price line ───────────────────────────────────────────
    last_close = float(df["Close"].iloc[-1]) if len(df) > 0 else 0
    current_price = trend.get("price", last_close)
    if isinstance(current_price, (pd.Series, np.ndarray)):
        current_price = float(current_price.iloc[-1]) if len(current_price) > 0 else last_close
    current_price = float(current_price)

    fig.add_hline(
        y=current_price,
        line_dash="dash", line_color=_CYAN, line_width=1, opacity=0.6,
        row=1, col=1,
        annotation_text=f"  {current_price:.2f}",
        annotation_font_color=_CYAN,
        annotation_font_size=11,
    )

    # ── Order Block zones ────────────────────────────────────────────
    df_dates = df["Date"].tolist()
    for ob_raw in (orderblocks or []):
        ob = _ob_to_chart_dict(ob_raw, df_dates)
        ob_type = ob.get("type", "bullish").lower()
        status = ob.get("status", "active").lower()
        date_start = ob.get("date_start") or ob.get("date")
        date_end = ob.get("date_end") or ob.get("date")
        ob_high = ob.get("high", 0)
        ob_low = ob.get("low", 0)
        ob_vol = ob.get("volume", 0)

        if not date_start:
            continue

        # Alpha depends on status
        if status == "invalidated":
            alpha_fill, alpha_line = 0.05, 0.15
        elif status in ("mitigated", "tested"):
            alpha_fill, alpha_line = 0.10, 0.25
        else:  # active
            alpha_fill, alpha_line = 0.18, 0.45

        if ob_type == "bullish":
            fill_color = f"rgba(45,138,78,{alpha_fill})"
            line_color = f"rgba(45,138,78,{alpha_line})"
            label_color = _GREEN
        else:
            fill_color = f"rgba(196,69,69,{alpha_fill})"
            line_color = f"rgba(196,69,69,{alpha_line})"
            label_color = _RED

        fig.add_shape(
            type="rect",
            x0=str(date_start), x1=str(date_end),
            y0=ob_low, y1=ob_high,
            fillcolor=fill_color,
            line=dict(color=line_color, width=1),
            layer="below",
            row=1, col=1,
        )

        # Clean OB zones — no text labels, just colored rectangles

    # ── Volume bars ──────────────────────────────────────────────────
    if "Volume" in df.columns:
        up_mask = df["Close"] >= df["Open"]
        bar_colors = [
            "rgba(45,138,78,0.45)" if up else "rgba(196,69,69,0.45)"
            for up in up_mask
        ]
        fig.add_trace(
            go.Bar(
                x=df["Date"], y=df["Volume"],
                name="Volume", marker_color=bar_colors, showlegend=False,
            ),
            row=2, col=1,
        )

    # ── Layout ───────────────────────────────────────────────────────
    fig.update_layout(
        **_PLOTLY_BASE,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=24, b=0),
        legend=dict(
            orientation="h", x=0, y=1.02,
            font=dict(color=_TEXT, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(138,133,120,0.2)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(138,133,120,0.2)", zeroline=False)

    return fig


# ------------------------------------------------------------------ #
#  build_sentiment_gauge
# ------------------------------------------------------------------ #

def build_sentiment_gauge(score: int, label: str = "Fear & Greed") -> go.Figure:
    """Gauge chart 0–100 with colored zones."""

    if score <= 25:
        needle_color, zone_label = _RED, "Extreme Fear"
    elif score <= 45:
        needle_color, zone_label = _YELLOW, "Fear"
    elif score <= 55:
        needle_color, zone_label = _TEXT, "Neutral"
    elif score <= 75:
        needle_color, zone_label = _YELLOW, "Greed"
    else:
        needle_color, zone_label = _RED, "Extreme Greed"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title=dict(
            text=f"<b>{label}</b><br><span style='font-size:0.8em;color:{needle_color}'>{zone_label}</span>",
            font=dict(color=_TEXT, size=14),
        ),
        number=dict(font=dict(color=needle_color, size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=_DIM, tickfont=dict(color=_DIM, size=10)),
            bar=dict(color=needle_color, thickness=0.25),
            bgcolor=_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, 25],  color="rgba(196,69,69,0.25)"),
                dict(range=[25, 45], color="rgba(212,148,58,0.15)"),
                dict(range=[45, 55], color="rgba(232,228,220,0.08)"),
                dict(range=[55, 75], color="rgba(212,148,58,0.15)"),
                dict(range=[75, 100],color="rgba(196,69,69,0.25)"),
            ],
            threshold=dict(line=dict(color=_CYAN, width=2), thickness=0.75, value=score),
        ),
    ))
    fig.update_layout(**_PLOTLY_BASE, height=220, margin=dict(l=16, r=16, t=40, b=8))
    return fig


# ------------------------------------------------------------------ #
#  build_sector_pie
# ------------------------------------------------------------------ #

def build_sector_pie(breadth_data: dict) -> go.Figure:
    """Sector donut chart colored by trend state."""
    labels, values, colors = [], [], []
    for sector, info in breadth_data.items():
        labels.append(sector)
        values.append(info.get("weight", 5.0))
        state = info.get("state", "neutral").lower()
        colors.append(_GREEN if state == "bullish" else (_RED if state == "bearish" else _YELLOW))

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color=_BG2, width=2)),
        textfont=dict(color=_TEXT, size=11), hole=0.45,
    ))
    fig.update_layout(**_PLOTLY_BASE, height=300, margin=dict(l=0, r=0, t=24, b=0),
                      legend=dict(font=dict(color=_TEXT, size=10), bgcolor="rgba(0,0,0,0)"))
    return fig


# ------------------------------------------------------------------ #
#  build_heatmap
# ------------------------------------------------------------------ #

def build_heatmap(breadth_data: dict) -> go.Figure:
    """Sector heatmap grid colored green/yellow/red by state."""
    import math

    sectors = list(breadth_data.keys())
    if not sectors:
        fig = go.Figure()
        fig.update_layout(**_PLOTLY_BASE, height=200)
        return fig

    changes = [breadth_data[s].get("change", 0.0) for s in sectors]
    states = [breadth_data[s].get("state", "neutral") for s in sectors]
    z_vals = [1.0 if s == "bullish" else (-1.0 if s == "bearish" else 0.0) for s in states]
    text_vals = [f"{sec}<br>{chg:+.1f}%" for sec, chg in zip(sectors, changes)]

    ncols = min(4, len(sectors))
    nrows = math.ceil(len(sectors) / ncols)
    pad = nrows * ncols - len(sectors)
    z_vals += [0.0] * pad
    text_vals += [""] * pad

    z_grid = [z_vals[i*ncols:(i+1)*ncols] for i in range(nrows)]
    text_grid = [text_vals[i*ncols:(i+1)*ncols] for i in range(nrows)]

    fig = go.Figure(go.Heatmap(
        z=z_grid, text=text_grid, texttemplate="%{text}",
        textfont=dict(size=11, color=_TEXT),
        colorscale=[[0.0, "rgba(196,69,69,0.6)"], [0.5, "rgba(212,148,58,0.4)"], [1.0, "rgba(45,138,78,0.6)"]],
        zmin=-1, zmax=1, showscale=False, xgap=3, ygap=3,
    ))
    fig.update_layout(**_PLOTLY_BASE, height=max(160, nrows * 80),
                      margin=dict(l=0, r=0, t=24, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"))
    return fig


# ------------------------------------------------------------------ #
#  build_risk_gauge
# ------------------------------------------------------------------ #

def build_risk_gauge(risk_score: int) -> go.Figure:
    """Compact gauge for risk score 0–100."""
    if risk_score <= 33:
        needle_color, risk_label = _GREEN, "LOW"
    elif risk_score <= 66:
        needle_color, risk_label = _YELLOW, "MEDIUM"
    else:
        needle_color, risk_label = _RED, "HIGH"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk_score,
        title=dict(text=f"<b>Risk</b> <span style='color:{needle_color}'>{risk_label}</span>",
                   font=dict(color=_TEXT, size=13)),
        number=dict(font=dict(color=needle_color, size=26)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=9, color=_DIM)),
            bar=dict(color=needle_color, thickness=0.3), bgcolor=_BG, borderwidth=0,
            steps=[
                dict(range=[0, 33],  color="rgba(45,138,78,0.15)"),
                dict(range=[33, 66], color="rgba(212,148,58,0.15)"),
                dict(range=[66, 100],color="rgba(196,69,69,0.20)"),
            ],
        ),
    ))
    fig.update_layout(**_PLOTLY_BASE, height=180, margin=dict(l=12, r=12, t=36, b=8))
    return fig


# ------------------------------------------------------------------ #
#  build_momentum_chart
# ------------------------------------------------------------------ #

def build_momentum_chart(df: pd.DataFrame, momentum_data: dict) -> go.Figure:
    """RSI chart with overbought / oversold zones."""
    if "Date" not in df.columns:
        df = df.reset_index()
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    rsi_series = momentum_data.get("rsi_series")
    rsi_current = float(momentum_data.get("rsi", 50))
    ob_level = float(momentum_data.get("ob_level", 70))
    os_level = float(momentum_data.get("os_level", 30))

    if rsi_series is None or (hasattr(rsi_series, "__len__") and len(rsi_series) == 0):
        close = df["Close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi_series = (100 - (100 / (1 + rs))).fillna(50).tolist()

    rsi_arr = list(rsi_series)[-len(df):]
    rsi_color = _RED if rsi_current >= ob_level else (_GREEN if rsi_current <= os_level else _CYAN)

    fig = go.Figure()
    fig.add_hrect(y0=ob_level, y1=100, fillcolor="rgba(196,69,69,0.07)", line_width=0)
    fig.add_hrect(y0=0, y1=os_level, fillcolor="rgba(45,138,78,0.07)", line_width=0)
    fig.add_hline(y=ob_level, line_color="rgba(196,69,69,0.4)", line_dash="dot", line_width=1)
    fig.add_hline(y=os_level, line_color="rgba(45,138,78,0.4)", line_dash="dot", line_width=1)
    fig.add_hline(y=50, line_color="rgba(138,133,120,0.5)", line_dash="dot", line_width=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=rsi_arr, name="RSI 14",
        line=dict(color=rsi_color, width=1.6),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.04)",
    ))
    fig.update_layout(
        **_PLOTLY_BASE, height=220,
        margin=dict(l=0, r=0, t=24, b=0),
        yaxis=dict(range=[0, 100], tickfont=dict(size=10, color=_DIM)),
        xaxis=dict(tickfont=dict(size=10, color=_DIM)),
        legend=dict(font=dict(color=_TEXT, size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ------------------------------------------------------------------ #
#  build_volatility_histogram
# ------------------------------------------------------------------ #

def build_volatility_histogram(vol_hist: dict) -> go.Figure:
    """
    OVTLYR-style directional volatility histogram.
    Green bars for positive returns, red for negative.
    """
    bins = vol_hist.get("bins", [])
    counts = vol_hist.get("counts", [])
    classification = vol_hist.get("classification", "")
    up_pct = vol_hist.get("up_pct", 50)
    skew = vol_hist.get("skew", 0)

    colors = []
    for b in bins:
        if b > 0:
            colors.append("rgba(45,138,78,0.7)")
        elif b < 0:
            colors.append("rgba(196,69,69,0.7)")
        else:
            colors.append("rgba(138,133,120,0.7)")

    fig = go.Figure(go.Bar(
        x=bins, y=counts,
        marker_color=colors,
        hovertemplate="Return: %{x:.1f}%<br>Days: %{y}<extra></extra>",
    ))

    # Add classification annotation
    class_color = _GREEN if "Bullish" in classification else (_RED if "Bearish" in classification else _YELLOW)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_BG2, plot_bgcolor=_BG,
        title=dict(
            text=f"Volatility Distribution — {classification}",
            font=dict(color=class_color, size=13),
        ),
        xaxis=dict(title="Daily Return %", tickfont=dict(size=10, color=_DIM), gridcolor="rgba(138,133,120,0.2)"),
        yaxis=dict(title="Days", tickfont=dict(size=10, color=_DIM), gridcolor="rgba(138,133,120,0.2)"),
        height=220,
        margin=dict(l=40, r=16, t=36, b=32),
        showlegend=False,
        # Add annotation with stats
        annotations=[dict(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=f"Up: {up_pct:.0f}% | Skew: {skew:+.2f}",
            showarrow=False, font=dict(size=10, color=_DIM),
            xanchor="right",
        )],
    )

    # Vertical line at zero
    fig.add_vline(x=0, line_color="rgba(232,228,220,0.3)", line_dash="dot", line_width=1)

    return fig


# ------------------------------------------------------------------ #
#  build_oscillator_direction
# ------------------------------------------------------------------ #

def build_oscillator_direction(osc: dict) -> go.Figure:
    """
    RSI oscillator with direction arrows and timing indicator.
    """
    rsi_series = osc.get("rsi_series", [])
    rsi_now = osc.get("rsi", 50)
    direction = osc.get("direction", "Flat")
    days = osc.get("days_in_direction", 0)
    timing = osc.get("timing", "Flat")
    timing_color = osc.get("timing_color", "rgba(138,133,120,0.9)")
    signal = osc.get("signal", "WAIT")

    fig = go.Figure()

    # OB/OS zones
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(196,69,69,0.07)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(45,138,78,0.07)", line_width=0)
    fig.add_hline(y=70, line_color="rgba(196,69,69,0.3)", line_dash="dot", line_width=1)
    fig.add_hline(y=30, line_color="rgba(45,138,78,0.3)", line_dash="dot", line_width=1)
    fig.add_hline(y=50, line_color="rgba(138,133,120,0.4)", line_dash="dot", line_width=1)

    # RSI line
    line_color = _GREEN if direction == "Rising" else (_RED if direction == "Falling" else _DIM)
    fig.add_trace(go.Scatter(
        y=rsi_series, mode="lines", name="RSI",
        line=dict(color=line_color, width=2),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.03)",
    ))

    # Direction arrow annotation
    arrow = "▲" if direction == "Rising" else ("▼" if direction == "Falling" else "◆")
    signal_color = _GREEN if signal == "ENTER" else (_RED if signal in ("EXIT", "LATE") else _YELLOW)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_BG2, plot_bgcolor=_BG,
        title=dict(
            text=(
                f"Oscillator: {arrow} {direction} ({days}d) — "
                f"<span style='color:{timing_color}'>{timing}</span> — "
                f"<span style='color:{signal_color}'>{signal}</span>"
            ),
            font=dict(color=_TEXT, size=12),
        ),
        yaxis=dict(range=[0, 100], tickfont=dict(size=10, color=_DIM)),
        xaxis=dict(showticklabels=False),
        height=200,
        margin=dict(l=32, r=16, t=36, b=8),
        showlegend=False,
    )

    return fig


# ------------------------------------------------------------------ #
#  build_bull_list_gauge
# ------------------------------------------------------------------ #

def build_bull_list_gauge(bl: dict) -> go.Figure:
    """
    Bull List % gauge with zone indicators and EMA5 crossover.
    """
    bull_pct = bl.get("bull_pct", 50)
    zone = bl.get("zone", "Bullish")
    signal = bl.get("signal", "CAUTION")
    bull_count = bl.get("bull_count", 0)
    total = bl.get("total_count", 1)

    # Colors based on zone
    if "Extreme Bullish" in zone:
        bar_color = _RED  # Extreme greed = danger
        zone_label = "EXTREME GREED — STOP NEW BUYS"
    elif "Bullish" in zone:
        bar_color = _GREEN
        zone_label = "BULLISH ZONE"
    elif "Extreme Bearish" in zone:
        bar_color = _GREEN  # Extreme fear = opportunity
        zone_label = "EXTREME FEAR — BEST ENTRY"
    else:
        bar_color = _YELLOW
        zone_label = "BEARISH ZONE — CAUTION"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bull_pct,
        title=dict(
            text=f"<b>Bull List %</b><br><span style='font-size:0.7em;color:{bar_color}'>{zone_label}</span>",
            font=dict(color=_TEXT, size=13),
        ),
        number=dict(suffix="%", font=dict(color=bar_color, size=26)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=9, color=_DIM)),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor=_BG, borderwidth=0,
            steps=[
                dict(range=[0, 25], color="rgba(45,138,78,0.15)"),   # Extreme fear = green (opportunity)
                dict(range=[25, 50], color="rgba(212,148,58,0.10)"),
                dict(range=[50, 75], color="rgba(212,148,58,0.10)"),
                dict(range=[75, 100], color="rgba(196,69,69,0.15)"),  # Extreme greed = red (danger)
            ],
            threshold=dict(line=dict(color=_CYAN, width=2), thickness=0.75, value=bull_pct),
        ),
    ))

    fig.add_annotation(
        x=0.5, y=-0.15, xref="paper", yref="paper",
        text=f"{bull_count}/{total} stocks bullish | Signal: {signal}",
        showarrow=False, font=dict(size=10, color=_DIM),
    )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_BG2, plot_bgcolor=_BG,
        height=220, margin=dict(l=16, r=16, t=40, b=24),
    )
    return fig
