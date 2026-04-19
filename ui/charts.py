import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(12,12,18,0)",
    plot_bgcolor="rgba(12,12,18,0)",
    font=dict(family="Courier New, monospace", color="#c9a84c", size=11),
    xaxis=dict(
        gridcolor="rgba(201,168,76,0.08)",
        zerolinecolor="rgba(201,168,76,0.15)",
        tickfont=dict(color="rgba(201,168,76,0.6)"),
    ),
    yaxis=dict(
        gridcolor="rgba(201,168,76,0.08)",
        zerolinecolor="rgba(201,168,76,0.15)",
        tickfont=dict(color="rgba(201,168,76,0.6)"),
    ),
    margin=dict(l=50, r=20, t=50, b=40),
)


def build_equity_chart(eq_df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df["equity"],
        fill="tozeroy",
        fillcolor="rgba(201,168,76,0.05)",
        line=dict(color="#c9a84c", width=1.5),
        name="Equity",
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    fig.add_hline(
        y=100000,
        line=dict(color="rgba(139,115,64,0.4)", width=1, dash="dot"),
        annotation_text="Initial $100k",
        annotation_font=dict(color="rgba(139,115,64,0.6)", size=10),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"EQUITY CURVE — {ticker}", font=dict(size=13, color="#c9a84c")),
        height=320,
        showlegend=False,
    )
    return fig


def build_drawdown_chart(dd_series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_series.index,
        y=dd_series.values * 100,
        fill="tozeroy",
        fillcolor="rgba(139,115,64,0.12)",
        line=dict(color="#8b7340", width=1.5),
        name="Drawdown %",
        hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
    ))
    layout_copy = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
    yaxis_base = PLOTLY_LAYOUT.get("yaxis", {})
    fig.update_layout(
        **layout_copy,
        title=dict(text="DRAWDOWN %", font=dict(size=13, color="#8b7340")),
        height=220,
        yaxis=dict(**yaxis_base, ticksuffix="%"),
        showlegend=False,
    )
    return fig


def build_monthly_heatmap(returns_series, eq_df):
    dates = eq_df.index
    if len(dates) > len(returns_series):
        dates = dates[:len(returns_series)]
    elif len(returns_series) > len(dates):
        returns_series = returns_series[:len(dates)]

    monthly = pd.DataFrame({"date": dates, "return": returns_series.values})
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly_ret = monthly.groupby(["year", "month"])["return"].sum().unstack() * 100

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    existing_months = [m for m in range(1, 13) if m in monthly_ret.columns]
    display_labels = [month_labels[m-1] for m in existing_months]
    z_data = monthly_ret[existing_months].values
    y_labels = [str(y) for y in monthly_ret.index.tolist()]

    text_data = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z_data]

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=display_labels,
        y=y_labels,
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=10, family="Courier New"),
        colorscale=[
            [0.0,  "#ff0044"],
            [0.35, "#4d0022"],
            [0.5,  "#14141e"],
            [0.65, "#003322"],
            [1.0,  "#2d8a4e"],
        ],
        zmid=0,
        zmin=-10,
        zmax=10,
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="rgba(201,168,76,0.6)", size=10),
            ticksuffix="%",
            outlinewidth=0,
        ),
        hovertemplate="<b>%{x} %{y}</b><br>%{z:.2f}%<extra></extra>",
    ))

    layout_copy2 = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis", "xaxis")}
    fig.update_layout(
        **layout_copy2,
        title=dict(text="MONTHLY RETURNS HEATMAP", font=dict(size=13, color="#c9a84c")),
        height=max(180, len(y_labels) * 35 + 80),
        xaxis=dict(**PLOTLY_LAYOUT.get("xaxis", {}), side="bottom"),
        yaxis=dict(**PLOTLY_LAYOUT.get("yaxis", {}), autorange="reversed"),
    )
    return fig


def build_gauge(value, max_val, label, color_cyan=True):
    bar_color = "rgba(201,168,76,0.9)" if color_cyan else "rgba(139,115,64,0.9)"
    bg_color = "rgba(201,168,76,0.08)" if color_cyan else "rgba(139,115,64,0.08)"

    green_thresh = 0.67 * max_val
    yellow_thresh = 0.4 * max_val

    if value >= green_thresh:
        bar_color = "rgba(45,138,78,0.9)"
    elif value >= yellow_thresh:
        bar_color = "rgba(212,148,58,0.9)"
    else:
        bar_color = "rgba(196,69,69,0.9)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(
            font=dict(family="Courier New", color="#c9a84c", size=28),
            suffix=f"/{max_val}",
        ),
        title=dict(
            text=label,
            font=dict(family="Courier New", color="rgba(201,168,76,0.6)", size=11),
        ),
        gauge=dict(
            axis=dict(
                range=[0, max_val],
                tickfont=dict(color="rgba(201,168,76,0.5)", size=9),
                tickcolor="rgba(201,168,76,0.3)",
            ),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor=bg_color,
            borderwidth=1,
            bordercolor="rgba(201,168,76,0.2)",
            steps=[
                dict(range=[0, yellow_thresh], color="rgba(196,69,69,0.07)"),
                dict(range=[yellow_thresh, green_thresh], color="rgba(212,148,58,0.07)"),
                dict(range=[green_thresh, max_val], color="rgba(45,138,78,0.07)"),
            ],
            threshold=dict(
                line=dict(color="#8b7340", width=2),
                thickness=0.6,
                value=green_thresh,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Courier New"),
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig
