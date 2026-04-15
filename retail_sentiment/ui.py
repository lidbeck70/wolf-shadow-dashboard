"""
ui.py — Streamlit rendering for the Retail Sentiment Engine.
Nordic Gold palette throughout. All tables use st.dataframe().
"""
import streamlit as st
import pandas as pd

from retail_sentiment.config import BG, BG2, GOLD, BRONZE, GREEN, RED, TEXT, DIM, detect_market
from retail_sentiment.engine import build_all_reports
from retail_sentiment.history import read_history


def _section_header(text: str, color: str = BRONZE) -> None:
    """Render a styled section header."""
    st.markdown(
        f"<div style='color:{color};font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;font-weight:700;margin:20px 0 8px 0;'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )


def _score_color(score: float) -> str:
    """Map a 0-100 score to a color."""
    if score >= 70:
        return GREEN
    elif score >= 40:
        return GOLD
    else:
        return RED


def _confidence_label(conf: float) -> str:
    """Map confidence to a human-readable label."""
    if conf >= 0.8:
        return "HIGH"
    elif conf >= 0.4:
        return "MED"
    elif conf > 0:
        return "LOW"
    return "N/A"


def _style_composite_df(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply Nordic Gold styling to the composite score dataframe."""

    def _color_score(val):
        try:
            v = float(val)
            if v >= 70:
                return f"color: {GREEN}; font-weight: 700"
            elif v >= 40:
                return f"color: {GOLD}; font-weight: 700"
            else:
                return f"color: {RED}; font-weight: 700"
        except (ValueError, TypeError):
            return f"color: {TEXT}"

    def _color_change(val):
        try:
            v = float(val)
            if v > 0:
                return f"color: {GREEN}"
            elif v < 0:
                return f"color: {RED}"
            return f"color: {DIM}"
        except (ValueError, TypeError):
            return f"color: {DIM}"

    def _color_confidence(val):
        if val == "HIGH":
            return f"color: {GREEN}"
        elif val == "MED":
            return f"color: {GOLD}"
        elif val == "LOW":
            return f"color: {RED}"
        return f"color: {DIM}"

    styler = df.style
    if "Composite" in df.columns:
        styler = styler.map(_color_score, subset=["Composite"])
    if "Change %" in df.columns:
        styler = styler.map(_color_change, subset=["Change %"])
    if "Confidence" in df.columns:
        styler = styler.map(_color_confidence, subset=["Confidence"])
    return styler


@st.cache_data(ttl=900, show_spinner=False, max_entries=3)
def _run_engine():
    """Run the sentiment engine (cached for 15 min)."""
    return build_all_reports()


def render_retail_sentiment_page() -> None:
    """Main retail sentiment dashboard — full engine-powered version."""
    st.markdown(
        f"<h2 style='color:{GOLD};text-align:center;letter-spacing:0.15em;'>"
        f"RETAIL SENTIMENT ENGINE</h2>"
        f"<p style='color:{DIM};text-align:center;font-size:0.8rem;'>"
        f"Multi-Source Confidence-Weighted Scoring</p>",
        unsafe_allow_html=True,
    )

    # Run engine
    try:
        result = _run_engine()
    except Exception as e:
        st.error(f"Engine error: {e}")
        return

    reports = result.get("reports", [])
    sources = result.get("sources", {})

    if not reports:
        st.warning("No sentiment data available. Check data source connectivity.")
        return

    # ── Section 1: COMPOSITE SCORE OVERVIEW ─────────────────────────────
    _section_header("COMPOSITE SCORE OVERVIEW", GOLD)

    rows = []
    for r in reports:
        avg_conf = 0.0
        conf_vals = [v for v in r.confidences.values() if v > 0]
        if conf_vals:
            avg_conf = sum(conf_vals) / len(conf_vals)

        rows.append({
            "Ticker": r.ticker,
            "Market": r.market,
            "Composite": r.scores.get("composite", 0.0),
            "Reddit": r.scores.get("reddit", 0.0),
            "Flow": r.scores.get("retail_flow", 0.0),
            "Yahoo": r.scores.get("yahoo", 0.0),
            "Hype": r.scores.get("hype_overlap", 0.0),
            "Change %": r.metadata.get("price_change_pct", 0.0),
            "Vol Ratio": r.metadata.get("volume_ratio", 0.0),
            "Confidence": _confidence_label(avg_conf),
            "Sources": len(r.data_sources_available),
        })

    comp_df = pd.DataFrame(rows)
    styled = _style_composite_df(comp_df)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=450)

    # ── Section 2: REDDIT TRENDING ──────────────────────────────────────
    _section_header("REDDIT — MEST NAEMNDA AKTIER (24h)", BRONZE)

    reddit_src = sources.get("reddit")
    if reddit_src and reddit_src.confidence > 0 and reddit_src.data:
        raw_results = reddit_src.data.get("raw_results", [])
        if raw_results:
            reddit_df = pd.DataFrame(raw_results[:25])
            cols = ["ticker", "name", "mentions", "upvotes", "rank", "rank_24h_ago"]
            reddit_df = reddit_df[[c for c in cols if c in reddit_df.columns]]

            if "rank_24h_ago" in reddit_df.columns and "rank" in reddit_df.columns:
                reddit_df["rank_change"] = reddit_df["rank_24h_ago"] - reddit_df["rank"]
                reddit_df["rank_change"] = reddit_df["rank_change"].apply(
                    lambda x: f"+{int(x)}" if x > 0 else str(int(x)) if pd.notna(x) else "-"
                )

            rename_map = {
                "ticker": "Ticker",
                "name": "Namn",
                "mentions": "Naemningar",
                "upvotes": "Upvotes",
                "rank": "Rank",
                "rank_change": "24h Rank",
            }
            reddit_df = reddit_df.rename(columns={k: v for k, v in rename_map.items() if k in reddit_df.columns})
            if "rank_24h_ago" in reddit_df.columns:
                reddit_df = reddit_df.drop(columns=["rank_24h_ago"])
            st.dataframe(reddit_df, use_container_width=True, hide_index=True)
        else:
            st.info("Kunde inte hamta Reddit-data just nu.")
    else:
        st.info("Kunde inte hamta Reddit-data just nu.")

    # ── Section 3: RETAIL MOVERS ────────────────────────────────────────
    _section_header("RETAIL MOVERS — DAGENS ROERELSER", GOLD)

    volume_src = sources.get("volume")
    gainers = []
    losers = []
    if volume_src and volume_src.data:
        ticker_vols = volume_src.data.get("tickers", {})
        for t, vd in ticker_vols.items():
            change = vd.get("price_change_pct", 0.0)
            price = vd.get("price", 0.0)
            if change > 0:
                gainers.append({"Ticker": t, "Pris": price, "Forandring %": change})
            elif change < 0:
                losers.append({"Ticker": t, "Pris": price, "Forandring %": change})

    gainers.sort(key=lambda x: x["Forandring %"], reverse=True)
    losers.sort(key=lambda x: x["Forandring %"])

    g_col, l_col = st.columns(2)
    with g_col:
        st.markdown(
            f"<div style='color:{GREEN};font-weight:700;font-size:0.8rem;margin-bottom:6px;'>"
            f"TOPP GAINERS</div>",
            unsafe_allow_html=True,
        )
        if gainers:
            gdf = pd.DataFrame(gainers[:10])
            st.dataframe(gdf, use_container_width=True, hide_index=True)
        else:
            st.caption("Ingen data")

    with l_col:
        st.markdown(
            f"<div style='color:{RED};font-weight:700;font-size:0.8rem;margin-bottom:6px;'>"
            f"TOPP LOSERS</div>",
            unsafe_allow_html=True,
        )
        if losers:
            ldf = pd.DataFrame(losers[:10])
            st.dataframe(ldf, use_container_width=True, hide_index=True)
        else:
            st.caption("Ingen data")

    # ── Section 4: VOLUME ANOMALIES ─────────────────────────────────────
    _section_header("VOLUME ANOMALIES", GOLD)

    anomalies = []
    if volume_src and volume_src.data:
        for t, vd in volume_src.data.get("tickers", {}).items():
            vol_ratio = vd.get("volume_ratio", 0.0)
            if vol_ratio > 2.0:
                anomalies.append({
                    "Ticker": t,
                    "Vol Ratio": vol_ratio,
                    "Volume Z": vd.get("volume_z", 0.0),
                    "Price": vd.get("price", 0.0),
                    "Change %": vd.get("price_change_pct", 0.0),
                })

    if anomalies:
        anomalies.sort(key=lambda x: x["Vol Ratio"], reverse=True)
        adf = pd.DataFrame(anomalies)
        st.dataframe(adf, use_container_width=True, hide_index=True)
    else:
        st.caption("Inga volymanomolier detekterade (volym ratio > 2.0x)")

    # ── Section 5: HYPE ALERT ───────────────────────────────────────────
    _section_header("HYPE ALERT", GOLD)

    hype_tickers = [r for r in reports if r.scores.get("hype_overlap", 0) >= 50]
    if hype_tickers:
        for r in hype_tickers:
            mentions = r.metadata.get("reddit_mentions", 0)
            vol_ratio = r.metadata.get("volume_ratio", 0.0)
            hype_score = r.scores.get("hype_overlap", 0)

            sources_list = []
            if r.metadata.get("reddit_rank", 0) > 0:
                sources_list.append("Reddit")
            yahoo_src = sources.get("yahoo")
            if yahoo_src and yahoo_src.data:
                if r.ticker in yahoo_src.data.get("trending_tickers", []):
                    sources_list.append("Yahoo")
            if vol_ratio > 1.5:
                sources_list.append("Volume")

            source_text = " + ".join(sources_list) if sources_list else "Multi-source"

            st.markdown(
                f"<div style='background:{BG2};border:2px solid rgba(139,115,64,0.3);"
                f"border-radius:8px;padding:10px;margin:6px 0;'>"
                f"<span style='color:{BRONZE};font-weight:700;font-size:1rem;'>{r.ticker}</span>"
                f" — <span style='color:{TEXT};'>{source_text}</span>"
                f" · <span style='color:{DIM};'>Hype: {hype_score}</span>"
                f" · <span style='color:{DIM};'>{mentions} naemningar</span>"
                f" · <span style='color:{DIM};'>Vol: {vol_ratio}x</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Inga tickers med hype-overlap just nu.")

    # ── Section 6: DATA SOURCES STATUS ──────────────────────────────────
    _section_header("DATA SOURCES STATUS", BRONZE)

    source_status = []
    source_names = {
        "reddit": "Reddit (ApeWisdom)",
        "twitter": "StockTwits",
        "yahoo": "Yahoo Trending",
        "volume": "Volume (yfinance)",
        "options": "Options (EODHD)",
    }
    for key, label in source_names.items():
        src = sources.get(key)
        if src:
            conf = src.confidence
            status = "ACTIVE" if conf >= 0.8 else "DEGRADED" if conf > 0 else "OFFLINE"
            error = src.error or ""
        else:
            conf = 0.0
            status = "OFFLINE"
            error = "Not fetched"

        source_status.append({
            "Source": label,
            "Status": status,
            "Confidence": f"{conf:.0%}",
            "Error": error[:60] if error else "",
        })

    status_df = pd.DataFrame(source_status)

    def _color_status(val):
        if val == "ACTIVE":
            return f"color: {GREEN}; font-weight: 700"
        elif val == "DEGRADED":
            return f"color: {GOLD}; font-weight: 700"
        return f"color: {RED}; font-weight: 700"

    styled_status = status_df.style.map(_color_status, subset=["Status"])
    st.dataframe(styled_status, use_container_width=True, hide_index=True)

    # ── Section 7: HISTORY ──────────────────────────────────────────────
    _section_header("COMPOSITE SCORE HISTORY", BRONZE)

    history = read_history(limit=500)
    if history and len(history) > 5:
        try:
            import plotly.graph_objects as go

            # Group by ticker, plot top 5 by latest composite
            hist_df = pd.DataFrame(history)
            if "timestamp" in hist_df.columns and "composite" in hist_df.columns:
                hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

                # Get top 5 tickers by most recent composite
                latest = hist_df.groupby("ticker")["composite"].last().nlargest(5)
                top_tickers = latest.index.tolist()

                colors = [GOLD, BRONZE, GREEN, RED, DIM]
                fig = go.Figure()
                for i, ticker in enumerate(top_tickers):
                    td = hist_df[hist_df["ticker"] == ticker].sort_values("timestamp")
                    fig.add_trace(go.Scatter(
                        x=td["timestamp"],
                        y=td["composite"],
                        name=ticker,
                        mode="lines+markers",
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=4),
                    ))

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(12,12,18,0)",
                    plot_bgcolor="rgba(12,12,18,0)",
                    font=dict(family="Courier New, monospace", color=GOLD, size=11),
                    xaxis=dict(
                        gridcolor="rgba(201,168,76,0.08)",
                        zerolinecolor="rgba(201,168,76,0.15)",
                    ),
                    yaxis=dict(
                        title="Composite Score",
                        gridcolor="rgba(201,168,76,0.08)",
                        zerolinecolor="rgba(201,168,76,0.15)",
                        range=[0, 100],
                    ),
                    margin=dict(l=50, r=20, t=50, b=40),
                    legend=dict(font=dict(color=TEXT, size=10)),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Historikdata saknar ratt kolumner.")
        except ImportError:
            st.caption("Plotly kravs for historik-diagram.")
    else:
        st.caption("Otillracklig historik for trenddiagram. Data byggs upp over tid.")

    # ── Section 8: YAHOO TRENDING ───────────────────────────────────────
    yahoo_src = sources.get("yahoo")
    if yahoo_src and yahoo_src.confidence > 0 and yahoo_src.data:
        trending = yahoo_src.data.get("trending_tickers", [])
        if trending:
            _section_header("YAHOO TRENDING TICKERS", GOLD)
            tickers_str = " · ".join(trending[:20])
            st.markdown(
                f"<div style='color:{TEXT};font-size:0.9rem;line-height:2;'>{tickers_str}</div>",
                unsafe_allow_html=True,
            )

    # Footer
    ts = result.get("timestamp", "")
    st.markdown(
        f"<div style='color:{DIM};font-size:0.65rem;text-align:center;margin-top:24px;'>"
        f"Uppdateras var 15:e minut · Kallor: ApeWisdom, StockTwits, Yahoo Finance, yfinance, EODHD"
        f" · Senast: {ts[:19] if ts else 'N/A'}"
        f"</div>",
        unsafe_allow_html=True,
    )
