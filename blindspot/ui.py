"""
ui.py — Streamlit dashboard for Odin's Blindspot Index.
Nordic Gold palette. All tables use st.dataframe().
"""
import streamlit as st
import pandas as pd

from blindspot.config import BG, BG2, GOLD, BRONZE, GREEN, RED, TEXT, DIM
from blindspot.engine import build_all_reports
from blindspot.history import read_history

# ── Optional commodity theme board ───────────────────────────────────────────
_THEME_OK = False
try:
    from blindspot.theme_board import build_theme_board, theme_verdict_text, THEME_RATIO_KEYS
    _THEME_OK = True
except ImportError:
    pass


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


# ── Commodity Theme Board ─────────────────────────────────────────────────────

_CYKEL_COLORS = {
    "TIDIG":   GREEN,
    "MITTEN":  GOLD,
    "SEN":     "#d4943a",
    "TOPP":    RED,
    "DATA_GAP": DIM,
}


def _render_theme_board() -> None:
    """Render the Commodity Theme Board section in Odin's Blindspot."""
    if not _THEME_OK:
        st.caption("Commodity theme board module not available.")
        return

    _section_header("COMMODITY THEME BOARD", GOLD)
    st.caption(
        "Nio råvaruteman rankade efter Blindspot-poäng "
        "(Nödvändighet × Hat × Billigt). "
        "Cachad 12h. Klicka på ett tema för detaljer."
    )

    try:
        themes = build_theme_board()
    except Exception as exc:
        st.error(f"Theme board error: {exc}")
        return

    if not themes:
        st.warning("Ingen temadata tillgänglig.")
        return

    # ── Ranked summary table ──────────────────────────────────────────────
    rows = []
    for t in themes:
        cykel_str = t.cykel_label if not t.error else "DATA_GAP"
        rows.append({
            "Tema":          t.label + (" ⚠PROXY" if t.proxy_flag else ""),
            "Blindspot":     round(t.blindspot_score, 1),
            "Cykel":         cykel_str,
            "Hat":           round(t.hat_score, 1) if not t.error else None,
            "Nödvändighet":  t.necessity,
            "10y Pct":       round(t.percentile_10y, 1) if t.percentile_10y is not None else None,
        })

    tdf = pd.DataFrame(rows)

    def _color_blindspot(val):
        try:
            v = float(val)
            if v >= 25: return f"color: {GREEN}; font-weight: 700"
            if v >= 12: return f"color: {GOLD}; font-weight: 700"
            return f"color: {RED}"
        except (ValueError, TypeError):
            return f"color: {DIM}"

    def _color_cykel(val):
        c = _CYKEL_COLORS.get(str(val), DIM)
        return f"color: {c}; font-weight: 700"

    def _color_hat(val):
        try:
            v = float(val)
            if v >= 60: return f"color: {RED}; font-weight: 700"
            if v >= 35: return f"color: {GOLD}"
            return f"color: {GREEN}"
        except (ValueError, TypeError):
            return f"color: {DIM}"

    fmt = {"Blindspot": "{:.1f}", "Hat": "{:.1f}", "10y Pct": "{:.1f}"}
    styler = tdf.style.format(fmt, na_rep="—")
    if "Blindspot" in tdf.columns:
        styler = styler.map(_color_blindspot, subset=["Blindspot"])
    if "Cykel" in tdf.columns:
        styler = styler.map(_color_cykel, subset=["Cykel"])
    if "Hat" in tdf.columns:
        styler = styler.map(_color_hat, subset=["Hat"])

    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Per-theme expanders ───────────────────────────────────────────────
    for t in themes:
        cykel_color = _CYKEL_COLORS.get(t.cykel_label, DIM)
        proxy_badge = "  ⚠ PROXY" if t.proxy_flag else ""
        score_str   = f"  Blindspot {t.blindspot_score:.1f}" if not t.error else "  DATA SAKNAS"

        with st.expander(
            f"{t.label}{proxy_badge}{score_str}  ·  {t.cykel_label}",
            expanded=False,
        ):
            if t.error:
                st.warning(f"Data ej tillgänglig: {t.error}")
                if t.proxy_note:
                    st.caption(t.proxy_note)
                continue

            if t.proxy_note:
                st.caption(f"Proxy: {t.proxy_note}")

            col_scores, col_spark = st.columns([1, 2])

            with col_scores:
                # Sub-scores
                st.markdown(
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'

                    f'<div style="background:{BG2};border:1px solid rgba(0,229,255,0.15);'
                    f'border-radius:6px;padding:10px 12px;">'
                    f'<div style="font-size:0.65rem;color:{DIM};text-transform:uppercase;'
                    f'letter-spacing:0.1em;">Cykelposition</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{cykel_color};'
                    f'margin-top:4px;">{t.cykel_label}</div>'
                    f'<div style="font-size:0.7rem;color:{DIM};margin-top:2px;">'
                    f'10y pct: {t.percentile_10y:.0f}th' if t.percentile_10y is not None else "10y pct: —"
                    f'</div></div>'

                    f'<div style="background:{BG2};border:1px solid rgba(0,229,255,0.15);'
                    f'border-radius:6px;padding:10px 12px;">'
                    f'<div style="font-size:0.65rem;color:{DIM};text-transform:uppercase;'
                    f'letter-spacing:0.1em;">Hat Score</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;'
                    f'color:{"#c44545" if t.hat_score >= 60 else GOLD};margin-top:4px;">'
                    f'{t.hat_score:.0f}/100</div>'
                    f'<div style="font-size:0.7rem;color:{DIM};margin-top:2px;">'
                    f'5y high dist: -{t.hat_from_5y_high_pct:.0f}%' if t.hat_from_5y_high_pct is not None else ""
                    f'</div></div>'

                    f'<div style="background:{BG2};border:1px solid rgba(0,229,255,0.15);'
                    f'border-radius:6px;padding:10px 12px;">'
                    f'<div style="font-size:0.65rem;color:{DIM};text-transform:uppercase;'
                    f'letter-spacing:0.1em;">Nödvändighet</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{GOLD};margin-top:4px;">'
                    f'{t.necessity}/100</div>'
                    f'<div style="font-size:0.7rem;color:{DIM};margin-top:2px;">statisk vikt</div>'
                    f'</div>'

                    f'<div style="background:{BG2};border:1px solid rgba(0,229,255,0.15);'
                    f'border-radius:6px;padding:10px 12px;">'
                    f'<div style="font-size:0.65rem;color:{DIM};text-transform:uppercase;'
                    f'letter-spacing:0.1em;">Blindspot Score</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;'
                    f'color:{"#2d8a4e" if t.blindspot_score >= 25 else GOLD};margin-top:4px;">'
                    f'{t.blindspot_score:.1f}/100</div>'
                    f'</div>'

                    f'</div>',
                    unsafe_allow_html=True,
                )

                # RS vs SPY
                if t.hat_rs_vs_spy is not None:
                    rs_color = "#2d8a4e" if t.hat_rs_vs_spy >= 0 else "#c44545"
                    st.markdown(
                        f'<div style="font-size:0.75rem;color:{DIM};margin-top:8px;">'
                        f'12m RS vs SPY: <span style="color:{rs_color};font-weight:700;">'
                        f'{t.hat_rs_vs_spy:+.1f}%</span></div>',
                        unsafe_allow_html=True,
                    )

            with col_spark:
                # Sparkline (2y weekly)
                if t.sparkline_dates and t.sparkline_values:
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=t.sparkline_dates,
                            y=t.sparkline_values,
                            mode="lines",
                            line=dict(color=cykel_color, width=1.5),
                            fill="tozeroy",
                            fillcolor=f"rgba(0,229,255,0.04)",
                            name=t.label,
                        ))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor=BG,
                            font=dict(color=TEXT, size=9),
                            xaxis=dict(gridcolor="rgba(138,133,120,0.08)", title=""),
                            yaxis=dict(gridcolor="rgba(138,133,120,0.08)", title=""),
                            margin=dict(l=36, r=8, t=8, b=24),
                            height=130,
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True,
                                        key=f"theme_spark_{t.key}")
                    except Exception:
                        st.caption("Sparkline ej tillgänglig.")

            # Rubber-band ratio links
            ratio_keys = THEME_RATIO_KEYS.get(t.key, [])
            if ratio_keys:
                st.markdown(
                    f'<div style="font-size:0.75rem;color:{DIM};margin-top:6px;">'
                    f'Relaterade gummisnoddar: '
                    f'<span style="color:{GOLD};">{", ".join(ratio_keys)}</span>'
                    f' — se Alpha Regime → Deep Contrarian</div>',
                    unsafe_allow_html=True,
                )

            # Swedish verdict
            verdict = theme_verdict_text(t)
            st.markdown(
                f'<div style="font-size:0.82rem;color:{TEXT};margin-top:10px;'
                f'padding:10px 14px;background:{BG2};border-radius:6px;'
                f'border-left:3px solid {cykel_color};">{verdict}</div>',
                unsafe_allow_html=True,
            )


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
                f"<div style='background:{BG2};border:1px solid rgba(0,229,255,0.2);"
                f"border-radius:8px;padding:12px;text-align:center;'>"
                f"<div style='color:{GOLD};font-weight:700;font-size:1.1rem;'>"
                f"{report.ticker}</div>"
                f"<div style='color:{DIM};font-size:0.7rem;'>{report.sector}</div>"
                f"<div style='color:{opp_color};font-size:1.8rem;font-weight:700;"
                f"margin:6px 0;'>{report.opportunity_score:.1f}</div>"
                f"<div style='font-size:0.65rem;color:{DIM};'>"
                f"Hat {report.hat_score:.0f} · Str {report.strength_score:.0f} · Cat {report.catalyst_score:.0f}"
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
            "Opportunity": round(r.opportunity_score, 1),
            "Hat": round(r.hat_score, 1),
            "Necessity": int(r.necessity_score),
            "Strength": round(r.strength_score, 1),
            "Catalyst": int(r.catalyst_score),
            "Close": round(r.close, 2),
            "Perf 6m %": round(r.perf_6m, 1),
            "Perf 12m %": round(r.perf_12m, 1),
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

    # Number formatting — clean decimals
    fmt = {}
    for col in df.columns:
        if col in ("Opportunity", "Hat", "Strength"):
            fmt[col] = "{:.1f}"
        elif col in ("Perf 6m %", "Perf 12m %"):
            fmt[col] = "{:.1f}"
        elif col == "Close":
            fmt[col] = "{:.2f}"
        elif col == "Confidence":
            fmt[col] = "{:.2f}"
        elif col in ("Necessity", "Catalyst"):
            fmt[col] = "{:.0f}"

    styler = df.style.format(fmt, na_rep="-")
    if "Opportunity" in df.columns:
        styler = styler.map(_color_opportunity, subset=["Opportunity"])
    if "Hat" in df.columns:
        styler = styler.map(_color_hat, subset=["Hat"])
    for col in ["Perf 6m %", "Perf 12m %"]:
        if col in df.columns:
            styler = styler.map(_color_perf, subset=[col])
    if "Confidence" in df.columns:
        styler = styler.map(_color_confidence, subset=["Confidence"])

    st.dataframe(styler, width='stretch', hide_index=True, height=500)

    # ── Section 2b: COMMODITY THEME BOARD ────────────────────────────
    _render_theme_board()

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
        fmt_s = {"Avg Opportunity": "{:.1f}", "Best": "{:.1f}"}
        styler_s = sector_df.style.format(fmt_s, na_rep="-").map(_color_opportunity, subset=["Avg Opportunity", "Best"])
        st.dataframe(styler_s, width='stretch', hide_index=True)

    # ── Section 4: VALUE TRAP WARNINGS ────────────────────────────────
    _section_header("VALUE TRAP WARNINGS", RED)

    traps = [r for r in reports if r.flags.get("value_trap_risk")]
    if traps:
        trap_rows = []
        for r in traps:
            trap_rows.append({
                "Ticker": r.ticker,
                "Sector": r.sector,
                "Hat": round(r.hat_score, 1),
                "Strength": round(r.strength_score, 1),
                "Catalyst": int(r.catalyst_score),
                "Opportunity": round(r.opportunity_score, 1),
                "Perf 12m %": round(r.perf_12m, 1),
            })
        trap_df = pd.DataFrame(trap_rows)
        st.dataframe(trap_df, width='stretch', hide_index=True)
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
                "Hat": round(r.hat_score, 1),
                "Catalyst": int(r.catalyst_score),
                "Opportunity": round(r.opportunity_score, 1),
                "Price > SMA50": r.catalyst_breakdown.get("price_above_sma50", 0) > 0,
                "SMA50 Rising": r.catalyst_breakdown.get("sma50_slope_positive", 0) > 0,
                "Vol Surge": r.catalyst_breakdown.get("vol_surge", 0) > 0,
            })
        rev_df = pd.DataFrame(rev_rows)
        st.dataframe(rev_df, width='stretch', hide_index=True)
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
    st.dataframe(styled_conf, width='stretch', hide_index=True)

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
                        gridcolor="rgba(0,229,255,0.08)",
                        zerolinecolor="rgba(0,229,255,0.15)",
                    ),
                    yaxis=dict(
                        title="Opportunity Score",
                        gridcolor="rgba(0,229,255,0.08)",
                        zerolinecolor="rgba(0,229,255,0.15)",
                    ),
                    margin=dict(l=50, r=20, t=50, b=40),
                    legend=dict(font=dict(color=TEXT, size=10)),
                    height=350,
                )
                st.plotly_chart(fig, width='stretch')
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

    # ── How it works (collapsible guide) ────────────────────────────────────
    _render_blindspot_guide()


def _render_blindspot_guide():
    """Collapsible guide explaining how Odin's Blindspot works."""
    try:
        with st.expander("Hur Odin's Blindspot fungerar", expanded=False):
            st.markdown(
                f"""
<div style="color:{TEXT};line-height:1.7;font-size:0.85rem;">

<h3 style="color:{GOLD};margin-top:0;">Koncept</h3>
<p>Odin's Blindspot hittar aktier som marknaden hatar eller ignorerar, men som ar
nodvandiga for civilisationens infrastruktur och har stark ekonomi. Nar dessa bolag
borjar visa tidiga tecken pa vanding — da uppstar en asymmetrisk mojlighet.</p>

<h3 style="color:{GOLD};">De fyra dimensionerna</h3>

<p><b style="color:{RED};">1. Hat Score (0-100)</b><br/>
Mater hur hatad/ignorerad aktien ar. Hog score = mer hatad. Bygger pa 9 komponenter:</p>
<ul>
<li><b>SMA200-avstand</b> (max 25p) — Ju langre under 200-dagars snitt, desto mer hatad</li>
<li><b>SMA50-avstand</b> (max 10p) — Under 50-dagars snitt</li>
<li><b>52-veckors low</b> (max 15p) — Narhet till arslaget</li>
<li><b>6/12 manaders performance</b> (max 10p) — Negativ prisutveckling</li>
<li><b>Volym-torka</b> (max 10p) — Ingen bryr sig (lag volym vs snitt)</li>
<li><b>Reddit-tystnad</b> (max 10p) — Inga mentions pa WallStreetBets</li>
<li><b>StockTwits-negativitet</b> (max 10p) — Negativt sentiment</li>
<li><b>Retail-apati</b> (max 10p) — Lag retail flow</li>
</ul>

<p><b style="color:{GOLD};">2. Necessity Score (0-100)</b><br/>
Hur kritisk ar sektorn for samhallet? Fem nivaer:</p>
<ul>
<li><b style="color:{GOLD};">100 — Civilisationskritiskt:</b> Uran, olja/gas, koppar, stal, godsel, shipping, kraft, vatten, forsvar</li>
<li><b>80 — Kritisk infrastruktur:</b> Halvledare, sjukvard, adelmetaller, jarnvag</li>
<li><b>60 — Viktig samhallsnytta:</b> Telekom, energilagring, forsakring</li>
<li><b>30 — Bekvamlighet:</b> Fastigheter, fintech, detaljhandel</li>
<li><b style="color:{DIM};">0 — Icke-nodvandigt:</b> SaaS, gaming, sociala medier, lyx</li>
</ul>
<p>Bolag kan ha manuella overrides (t.ex. CCJ = Uran = 100).</p>

<p><b style="color:{GREEN};">3. Strength Score (0-100)</b><br/>
Fundamental ekonomisk styrka. Sex komponenter:</p>
<ul>
<li><b>Free Cash Flow 3 ar</b> (max 30p) — Positivt alla 3 ar = 30p</li>
<li><b>EBITDA-marginal</b> (max 15p) — Hog marginal = stark lonsamhet</li>
<li><b>FCF Yield</b> (max 15p) — Kassaflode relativt foretagsvarde</li>
<li><b>Debt/Equity</b> (max 20p) — Lag skuldsattning = starkare</li>
<li><b>Omsattningstillvaxt</b> (max 15p) — 3-ars CAGR</li>
<li><b>EV/EBITDA bonus</b> (max 5p) — Under 10x = undervarderad</li>
</ul>

<p><b style="color:#d4943a;">4. Catalyst Score (0-20 bonus)</b><br/>
Tidiga tecken pa vanding — skyddar mot value traps:</p>
<ul>
<li><b>Pris over SMA50</b> (+8p) — Aktien har borjat vanda</li>
<li><b>SMA50 lutar uppat</b> (+6p) — Trenden vander</li>
<li><b>Volym okar</b> (+6p) — Intresse atervander (z-score > 1.0)</li>
</ul>

<h3 style="color:{GOLD};">Opportunity Score — Formeln</h3>
<p style="font-family:monospace;background:{BG2};padding:10px;border-radius:6px;border:1px solid rgba(0,229,255,0.15);">
Opportunity = Hat x (Necessity / 100) x (Strength / 100) + Catalyst-bonus
</p>
<p>Catalyst-bonus skalas med Hat Score — catalyst spelar bara roll om aktien faktiskt ar hatad.</p>
<p>Necessity har ett golv pa 20 — ingen aktie nollstalls helt, men civilisationskritiska dominerar.</p>

<h3 style="color:{GOLD};">Flaggor</h3>
<ul>
<li><b style="color:{RED};">Value Trap Risk</b> — Hat > 60 men Catalyst = 0. Aktien sjunker utan tecken pa botten.</li>
<li><b style="color:{GREEN};">Potential Reversal</b> — Hat > 60 och Catalyst >= 10. Marknaden borjar vakna — timing-signal.</li>
<li><b style="color:#d4943a;">Low Confidence</b> — Under 50% datatakning. Behover manuell granskning.</li>
</ul>

<h3 style="color:{GOLD};">Confidence-system</h3>
<p>Varje datakalla returnerar ett confidence-varde (0-1). Kallor som saknar data (t.ex. EODHD utan nyckel)
far confidence = 0 och viktas bort automatiskt. Nordiska aktier utan optionsdata far lagre confidence
men ett meningsfullt score baserat pa tillgangliga kallor.</p>

<h3 style="color:{GOLD};">Sa anvander du det</h3>
<ol>
<li>Sortera pa <b>Opportunity Score</b> — hogst forst</li>
<li>Kontrollera <b>flaggor</b> — undvik Value Trap Risk utan egen analys</li>
<li>Fokusera pa <b>Potential Reversal</b> — dar ar timing-signalen</li>
<li>Verifiera i <b>Wolf/Viking/Alpha Regime</b> innan du tar en position</li>
<li>Anvand <b>SL/TP-kalkylatorn</b> i regime-fliken for position sizing</li>
</ol>

</div>
""",
                unsafe_allow_html=True,
            )
    except Exception:
        pass
