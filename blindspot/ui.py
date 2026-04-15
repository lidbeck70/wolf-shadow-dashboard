"""
ui.py — Streamlit dashboard for Odin's Blindspot Index.
Nordic Gold palette. All tables use st.dataframe().
"""
import streamlit as st
import pandas as pd

from blindspot.config import BG, BG2, GOLD, BRONZE, GREEN, RED, TEXT, DIM
from blindspot.engine import build_all_reports
from blindspot.history import read_history


def _section_header(text: str, color: str = BRONZE) -> None:
    """Render a styled section header."""
    st.markdown(
        f"<div style='color:{color};font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;font-weight:700;margin:20px 0 8px 0;'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )


def _score_color(score: float) -> str:
    if score >= 70:
        return GREEN
    elif score >= 40:
        return GOLD
    return RED


def _flag_badges(flags: dict) -> str:
    """Render flag badges as styled text."""
    badges = []
    if flags.get("potential_reversal"):
        badges.append(f"<span style='color:{GREEN};font-weight:700;'>REVERSAL</span>")
    if flags.get("value_trap_risk"):
        badges.append(f"<span style='color:{RED};font-weight:700;'>TRAP RISK</span>")
    if flags.get("low_confidence"):
        badges.append(f"<span style='color:{DIM};'>LOW CONF</span>")
    if flags.get("unmapped_sector"):
        badges.append(f"<span style='color:{DIM};'>UNMAPPED</span>")
    return " ".join(badges) if badges else ""


@st.cache_data(ttl=900, show_spinner=False, max_entries=3)
def _run_engine():
    """Run the blindspot engine (cached for 15 min)."""
    return build_all_reports()


def render_blindspot_page() -> None:
    """Main Odin's Blindspot Index dashboard."""
    st.markdown(
        f"<h2 style='color:{GOLD};text-align:center;letter-spacing:0.15em;'>"
        f"ODIN'S BLINDSPOT INDEX</h2>"
        f"<p style='color:{DIM};text-align:center;font-size:0.8rem;'>"
        f"Contrarian Value Screener — Hated but Necessary</p>",
        unsafe_allow_html=True,
    )

    try:
        result = _run_engine()
    except Exception as e:
        st.error(f"Engine error: {e}")
        return

    reports = result.get("reports", [])
    if not reports:
        st.warning("No blindspot data available. Check data source connectivity.")
        return

    # ── Section 1: TOP OPPORTUNITIES ──────────────────────────────────
    _section_header("TOP OPPORTUNITIES", GOLD)

    top5 = reports[:5]
    cols = st.columns(min(len(top5), 5))
    for i, report in enumerate(top5):
        with cols[i]:
            opp_color = _score_color(report.opportunity_score)
            flag_html = _flag_badges(report.flags)

            st.markdown(
                f"<div style='background:{BG2};border:1px solid rgba(201,168,76,0.2);"
                f"border-radius:8px;padding:12px;text-align:center;'>"
                f"<div style='color:{GOLD};font-weight:700;font-size:1.1rem;'>"
                f"{report.ticker}</div>"
                f"<div style='color:{DIM};font-size:0.7rem;'>{report.sector}</div>"
                f"<div style='color:{opp_color};font-size:1.8rem;font-weight:700;"
                f"margin:6px 0;'>{report.opportunity_score}</div>"
                f"<div style='font-size:0.65rem;color:{DIM};'>"
                f"Hat {report.hat_score} · Str {report.strength_score} · Cat {report.catalyst_score}"
                f"</div>"
                f"<div style='font-size:0.65rem;margin-top:4px;'>{flag_html}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Section 2: FULL RANKING TABLE ─────────────────────────────────
    _section_header("FULL RANKING", GOLD)

    rows = []
    for r in reports:
        rows.append({
            "Ticker": r.ticker,
            "Sector": r.sector,
            "Opportunity": r.opportunity_score,
            "Hat": r.hat_score,
            "Necessity": r.necessity_score,
            "Strength": r.strength_score,
            "Catalyst": r.catalyst_score,
            "Close": r.close,
            "Perf 6m %": r.perf_6m,
            "Perf 12m %": r.perf_12m,
            "Confidence": round(r.overall_confidence, 2),
        })

    df = pd.DataFrame(rows)

    def _color_opportunity(val):
        try:
            v = float(val)
            if v >= 30:
                return f"color: {GREEN}; font-weight: 700"
            elif v >= 15:
                return f"color: {GOLD}; font-weight: 700"
            return f"color: {RED}; font-weight: 700"
        except (ValueError, TypeError):
            return f"color: {TEXT}"

    def _color_hat(val):
        try:
            v = float(val)
            if v >= 60:
                return f"color: {RED}; font-weight: 700"
            elif v >= 35:
                return f"color: {GOLD}"
            return f"color: {GREEN}"
        except (ValueError, TypeError):
            return f"color: {TEXT}"

    def _color_perf(val):
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
        try:
            v = float(val)
            if v >= 0.7:
                return f"color: {GREEN}"
            elif v >= 0.4:
                return f"color: {GOLD}"
            return f"color: {RED}"
        except (ValueError, TypeError):
            return f"color: {DIM}"

    styler = df.style
    if "Opportunity" in df.columns:
        styler = styler.map(_color_opportunity, subset=["Opportunity"])
    if "Hat" in df.columns:
        styler = styler.map(_color_hat, subset=["Hat"])
    for col in ["Perf 6m %", "Perf 12m %"]:
        if col in df.columns:
            styler = styler.map(_color_perf, subset=[col])
    if "Confidence" in df.columns:
        styler = styler.map(_color_confidence, subset=["Confidence"])

    st.dataframe(styler, use_container_width=True, hide_index=True, height=500)

    # ── Section 3: SECTOR VIEW ────────────────────────────────────────
    _section_header("SECTOR BREAKDOWN", BRONZE)

    sector_data = {}
    for r in reports:
        s = r.sector or "Unknown"
        if s not in sector_data:
            sector_data[s] = {"scores": [], "count": 0}
        sector_data[s]["scores"].append(r.opportunity_score)
        sector_data[s]["count"] += 1

    sector_rows = []
    for s, d in sorted(sector_data.items(), key=lambda x: -sum(x[1]["scores"]) / len(x[1]["scores"])):
        avg_opp = sum(d["scores"]) / len(d["scores"])
        sector_rows.append({
            "Sector": s,
            "Avg Opportunity": round(avg_opp, 1),
            "Tickers": d["count"],
            "Best": round(max(d["scores"]), 1),
        })

    if sector_rows:
        sector_df = pd.DataFrame(sector_rows)
        styler_s = sector_df.style.map(_color_opportunity, subset=["Avg Opportunity", "Best"])
        st.dataframe(styler_s, use_container_width=True, hide_index=True)

    # ── Section 4: VALUE TRAP WARNINGS ────────────────────────────────
    _section_header("VALUE TRAP WARNINGS", RED)

    traps = [r for r in reports if r.flags.get("value_trap_risk")]
    if traps:
        trap_rows = []
        for r in traps:
            trap_rows.append({
                "Ticker": r.ticker,
                "Sector": r.sector,
                "Hat": r.hat_score,
                "Strength": r.strength_score,
                "Catalyst": r.catalyst_score,
                "Opportunity": r.opportunity_score,
                "Perf 12m %": r.perf_12m,
            })
        trap_df = pd.DataFrame(trap_rows)
        st.dataframe(trap_df, use_container_width=True, hide_index=True)
        st.caption("High hat score but zero catalyst — potential value traps. Hated for a reason?")
    else:
        st.caption("No value trap warnings detected.")

    # ── Section 5: POTENTIAL REVERSALS ────────────────────────────────
    _section_header("POTENTIAL REVERSALS", GREEN)

    reversals = [r for r in reports if r.flags.get("potential_reversal")]
    if reversals:
        rev_rows = []
        for r in reversals:
            rev_rows.append({
                "Ticker": r.ticker,
                "Sector": r.sector,
                "Hat": r.hat_score,
                "Catalyst": r.catalyst_score,
                "Opportunity": r.opportunity_score,
                "Price > SMA50": r.catalyst_breakdown.get("price_above_sma50", 0) > 0,
                "SMA50 Rising": r.catalyst_breakdown.get("sma50_slope_positive", 0) > 0,
                "Vol Surge": r.catalyst_breakdown.get("vol_surge", 0) > 0,
            })
        rev_df = pd.DataFrame(rev_rows)
        st.dataframe(rev_df, use_container_width=True, hide_index=True)
        st.caption("Hated stocks showing early signs of turnaround.")
    else:
        st.caption("No potential reversals detected.")

    # ── Section 6: CONFIDENCE STATUS ──────────────────────────────────
    _section_header("CONFIDENCE STATUS", BRONZE)

    conf_rows = []
    price_available = sum(1 for r in reports if r.price_confidence > 0)
    fund_available = sum(1 for r in reports if r.fundamentals_confidence > 0)
    fund_eodhd = sum(1 for r in reports if r.fundamentals_confidence >= 1.0)
    necessity_mapped = sum(1 for r in reports if r.necessity_confidence >= 0.8)

    conf_rows.append({
        "Source": "Price Data (yfinance)",
        "Status": "ACTIVE" if price_available > 0 else "OFFLINE",
        "Coverage": f"{price_available}/{len(reports)}",
    })
    conf_rows.append({
        "Source": "Fundamentals (EODHD)",
        "Status": "ACTIVE" if fund_eodhd > 0 else "OFFLINE",
        "Coverage": f"{fund_eodhd}/{len(reports)}",
    })
    conf_rows.append({
        "Source": "Fundamentals (yfinance fallback)",
        "Status": "ACTIVE" if fund_available > fund_eodhd else "OFFLINE",
        "Coverage": f"{fund_available - fund_eodhd}/{len(reports)}",
    })
    conf_rows.append({
        "Source": "Necessity Classification",
        "Status": "MAPPED" if necessity_mapped > 0 else "PARTIAL",
        "Coverage": f"{necessity_mapped}/{len(reports)}",
    })

    conf_df = pd.DataFrame(conf_rows)

    def _color_status(val):
        if val in ("ACTIVE", "MAPPED"):
            return f"color: {GREEN}; font-weight: 700"
        elif val == "PARTIAL":
            return f"color: {GOLD}; font-weight: 700"
        return f"color: {RED}; font-weight: 700"

    styled_conf = conf_df.style.map(_color_status, subset=["Status"])
    st.dataframe(styled_conf, use_container_width=True, hide_index=True)

    # ── Section 7: HISTORY ────────────────────────────────────────────
    _section_header("OPPORTUNITY SCORE HISTORY", BRONZE)

    history = read_history(limit=500)
    if history and len(history) > 5:
        try:
            import plotly.graph_objects as go

            hist_df = pd.DataFrame(history)
            if "timestamp" in hist_df.columns and "opportunity" in hist_df.columns:
                hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

                latest = hist_df.groupby("ticker")["opportunity"].last().nlargest(5)
                top_tickers = latest.index.tolist()

                colors = [GOLD, BRONZE, GREEN, RED, DIM]
                fig = go.Figure()
                for i, ticker in enumerate(top_tickers):
                    td = hist_df[hist_df["ticker"] == ticker].sort_values("timestamp")
                    fig.add_trace(go.Scatter(
                        x=td["timestamp"],
                        y=td["opportunity"],
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
                        title="Opportunity Score",
                        gridcolor="rgba(201,168,76,0.08)",
                        zerolinecolor="rgba(201,168,76,0.15)",
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

    # Footer
    ts = result.get("timestamp", "")
    st.markdown(
        f"<div style='color:{DIM};font-size:0.65rem;text-align:center;margin-top:24px;'>"
        f"Uppdateras var 15:e minut · Kallor: yfinance, EODHD, ApeWisdom"
        f" · Senast: {ts[:19] if ts else 'N/A'}"
        f"</div>",
        unsafe_allow_html=True,
    )
