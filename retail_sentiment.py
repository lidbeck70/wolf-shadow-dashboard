"""
retail_sentiment.py — Retail Sentiment Dashboard
Tracks trending stocks, Reddit mentions, and retail hype signals.
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ── Constants ────────────────────────────────────────────────────────────
CYAN = "#00ffff"
MAGENTA = "#ff00ff"
BG = "#050510"
BG2 = "#0a0a1e"
TEXT = "#ccddff"
DIM = "#4a4a6a"
GREEN = "#00ff88"
RED = "#ff3355"
GOLD = "#c9a84c"


@st.cache_data(ttl=1800, show_spinner=False, max_entries=3)
def _fetch_wsb_trending() -> pd.DataFrame:
    """Fetch trending tickers from ApeWisdom (Reddit WSB + other subs)."""
    try:
        r = requests.get(
            "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1",
            timeout=15,
            headers={"User-Agent": "NordicAlpha/1.0"},
        )
        if r.status_code == 200:
            data = r.json().get("results", [])
            df = pd.DataFrame(data)
            if not df.empty:
                # Keep relevant columns
                cols = ["ticker", "name", "mentions", "upvotes", "rank", "rank_24h_ago"]
                df = df[[c for c in cols if c in df.columns]].head(25)
                if "rank_24h_ago" in df.columns and "rank" in df.columns:
                    df["rank_change"] = df["rank_24h_ago"] - df["rank"]
                return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False, max_entries=3)
def _fetch_yahoo_trending() -> pd.DataFrame:
    """Fetch trending tickers from Yahoo Finance."""
    try:
        # Try the Yahoo Finance trending API
        r = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/trending/US",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code == 200:
            quotes = r.json().get("finance", {}).get("result", [])
            if quotes:
                tickers = [q["symbol"] for q in quotes[0].get("quotes", [])]
                return pd.DataFrame({"ticker": tickers[:20]})
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def _fetch_market_movers() -> dict:
    """Fetch top gainers and losers via yfinance."""
    try:
        import yfinance as yf
        # Get S&P 500 most active
        gainers = []
        losers = []
        # Use a simple approach: fetch a basket of popular retail tickers
        # and compute today's change
        retail_favorites = [
            "GME", "AMC", "BBBY", "PLTR", "NIO", "SOFI", "RIVN",
            "LCID", "MARA", "RIOT", "COIN", "HOOD", "WISH", "CLOV",
            "BB", "NOK", "TLRY", "SNDL", "ASTS", "IONQ", "RKLB",
            "SMCI", "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "INTC",
            "CCJ", "UEC", "UUUU", "GDX", "SLV", "XLE", "XOP",
        ]
        raw = yf.download(retail_favorites, period="5d", auto_adjust=True, progress=False)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in retail_favorites:
                    try:
                        close = raw.xs(ticker, level=1, axis=1)["Close"].dropna()
                        if len(close) >= 2:
                            change = (close.iloc[-1] / close.iloc[-2] - 1) * 100
                            price = close.iloc[-1]
                            if change > 0:
                                gainers.append({"ticker": ticker, "price": round(price, 2), "change_pct": round(change, 2)})
                            else:
                                losers.append({"ticker": ticker, "price": round(price, 2), "change_pct": round(change, 2)})
                    except (KeyError, ValueError):
                        pass
        gainers.sort(key=lambda x: x["change_pct"], reverse=True)
        losers.sort(key=lambda x: x["change_pct"])
        return {"gainers": gainers[:10], "losers": losers[:10]}
    except Exception:
        return {"gainers": [], "losers": []}


def render_retail_sentiment_page() -> None:
    """Main retail sentiment dashboard."""
    st.markdown(
        f"<h2 style='color:{CYAN};text-align:center;letter-spacing:0.15em;'>"
        f"RETAIL SENTIMENT</h2>"
        f"<p style='color:{DIM};text-align:center;font-size:0.8rem;'>"
        f"Reddit Hype · Trending Tickers · Retail Movers</p>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Reddit WSB Trending ────────────────────────────────────
    st.markdown(
        f"<div style='color:{MAGENTA};font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;font-weight:700;margin:20px 0 8px 0;'>"
        f"REDDIT — MEST NAEMNDA AKTIER (24h)</div>",
        unsafe_allow_html=True,
    )

    wsb_df = _fetch_wsb_trending()
    if not wsb_df.empty:
        # Style the dataframe
        display_df = wsb_df.copy()
        if "rank_change" in display_df.columns:
            display_df["rank_change"] = display_df["rank_change"].apply(
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
        display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Kunde inte hamta Reddit-data just nu.")

    # ── Section 2: Market Movers (Retail Favorites) ───────────────────────
    st.markdown(
        f"<div style='color:{CYAN};font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;font-weight:700;margin:20px 0 8px 0;'>"
        f"RETAIL MOVERS — DAGENS ROERELSER</div>",
        unsafe_allow_html=True,
    )

    movers = _fetch_market_movers()
    g_col, l_col = st.columns(2)
    with g_col:
        st.markdown(
            f"<div style='color:{GREEN};font-weight:700;font-size:0.8rem;margin-bottom:6px;'>"
            f"TOPP GAINERS</div>",
            unsafe_allow_html=True,
        )
        if movers["gainers"]:
            gdf = pd.DataFrame(movers["gainers"])
            gdf.columns = ["Ticker", "Pris", "Forandring %"]
            st.dataframe(gdf, use_container_width=True, hide_index=True)
        else:
            st.caption("Ingen data")

    with l_col:
        st.markdown(
            f"<div style='color:{RED};font-weight:700;font-size:0.8rem;margin-bottom:6px;'>"
            f"TOPP LOSERS</div>",
            unsafe_allow_html=True,
        )
        if movers["losers"]:
            ldf = pd.DataFrame(movers["losers"])
            ldf.columns = ["Ticker", "Pris", "Forandring %"]
            st.dataframe(ldf, use_container_width=True, hide_index=True)
        else:
            st.caption("Ingen data")

    # ── Section 3: Yahoo Trending ─────────────────────────────────────────
    st.markdown(
        f"<div style='color:{GOLD};font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;font-weight:700;margin:20px 0 8px 0;'>"
        f"YAHOO TRENDING TICKERS</div>",
        unsafe_allow_html=True,
    )

    yahoo_df = _fetch_yahoo_trending()
    if not yahoo_df.empty:
        # Display as horizontal chips/tags
        tickers_str = " · ".join(yahoo_df["ticker"].tolist())
        st.markdown(
            f"<div style='color:{TEXT};font-size:0.9rem;line-height:2;'>{tickers_str}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Yahoo trending inte tillganglig just nu.")

    # ── Section 4: Hype Score ─────────────────────────────────────────────
    st.markdown(
        f"<div style='color:{CYAN};font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;font-weight:700;margin:20px 0 8px 0;'>"
        f"HYPE ALERT</div>",
        unsafe_allow_html=True,
    )

    # Cross-reference: tickers appearing in BOTH Reddit top 10 AND Yahoo trending
    if not wsb_df.empty and not yahoo_df.empty:
        wsb_top = set(wsb_df["ticker"].head(10).tolist()) if "ticker" in wsb_df.columns else set()
        yahoo_set = set(yahoo_df["ticker"].tolist()) if "ticker" in yahoo_df.columns else set()
        overlap = wsb_top & yahoo_set
        if overlap:
            for ticker in sorted(overlap):
                mentions = wsb_df.loc[wsb_df["ticker"] == ticker, "mentions"].iloc[0] if "mentions" in wsb_df.columns else "?"
                st.markdown(
                    f"<div style='background:{BG2};border:2px solid rgba(255,0,255,0.3);"
                    f"border-radius:8px;padding:10px;margin:6px 0;'>"
                    f"<span style='color:{MAGENTA};font-weight:700;font-size:1rem;'>{ticker}</span>"
                    f" — <span style='color:{TEXT};'>Reddit Top 10 + Yahoo Trending</span>"
                    f" · <span style='color:{DIM};'>{mentions} naemningar</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Inga tickers i bade Reddit Top 10 och Yahoo Trending just nu.")
    else:
        st.caption("Otillracklig data for hype-analys.")

    # Footer
    st.markdown(
        f"<div style='color:{DIM};font-size:0.65rem;text-align:center;margin-top:24px;'>"
        f"Uppdateras var 30:e minut · Kallor: ApeWisdom, Yahoo Finance, yfinance"
        f"</div>",
        unsafe_allow_html=True,
    )
