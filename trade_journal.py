"""
trade_journal.py — Trade logging, P&L tracking, and performance analytics.

Uses GitHub Gist for persistence (same pattern as gist_storage.py).
Falls back to session_state if gist is unavailable.
"""
import json
import uuid
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Storage — reuse gist_storage.py token/auth helpers, store journal as
# a separate file in the SAME gist used by holdings.
# ---------------------------------------------------------------------------
GIST_ID = "50348cb5b9e325c8ae91439763d5f144"
JOURNAL_FILENAME = "journal_data.json"
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}"
LOCAL_FALLBACK = ".journal_data.json"

# Nordic Gold palette
_BG_DARK = "#0c0c12"
_BG_CARD = "#14141e"
_GOLD = "#c9a84c"
_GOLD_DIM = "#8b7340"
_GREEN = "#2d8a4e"
_RED = "#c44545"
_TEXT = "#e8e4dc"
_TEXT_DIM = "#8a8578"


def _get_github_token() -> Optional[str]:
    try:
        token = st.secrets.get("GITHUB_TOKEN", None)
        if token:
            return str(token).strip()
    except Exception:
        pass
    try:
        token = st.secrets["GITHUB_TOKEN"]
        if token:
            return str(token).strip()
    except Exception:
        pass
    return None


def _auth_header(token: str) -> dict:
    prefix = "Bearer" if token.startswith("github_pat_") else "token"
    return {
        "Authorization": f"{prefix} {token}",
        "Accept": "application/vnd.github.v3+json",
    }


# ---------------------------------------------------------------------------
# Load / Save journal
# ---------------------------------------------------------------------------
def load_journal() -> list:
    """Load journal trades. Priority: session_state > Gist > empty list."""
    if "journal_trades" in st.session_state:
        return st.session_state["journal_trades"]

    data = None

    try:
        r = requests.get(GIST_API_URL, timeout=10)
        if r.status_code == 200:
            gist = r.json()
            content = gist.get("files", {}).get(JOURNAL_FILENAME, {}).get("content", "")
            if content:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    data = parsed
    except Exception:
        pass

    if data is None:
        try:
            import os
            if os.path.exists(LOCAL_FALLBACK):
                with open(LOCAL_FALLBACK) as f:
                    data = json.load(f)
        except Exception:
            pass

    if data is None:
        data = []

    st.session_state["journal_trades"] = data
    return data


def save_journal(trades: list) -> bool:
    """Save journal to Gist + session_state. Returns True if gist write succeeded."""
    st.session_state["journal_trades"] = trades

    try:
        with open(LOCAL_FALLBACK, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    except Exception:
        pass

    token = _get_github_token()
    if not token:
        return False

    try:
        headers = _auth_header(token)
        payload = {
            "files": {
                JOURNAL_FILENAME: {
                    "content": json.dumps(trades, indent=2, default=str)
                }
            }
        }
        r = requests.patch(GIST_API_URL, headers=headers, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------
def _compute_kpis(df: pd.DataFrame) -> dict:
    """Compute summary KPIs from trades DataFrame."""
    if df.empty:
        return {
            "total": 0, "win_rate": 0.0, "avg_rr": 0.0,
            "profit_factor": 0.0, "best": 0.0, "worst": 0.0,
            "streak": "0", "total_pnl": 0.0,
        }

    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]
    total = len(df)
    win_rate = (len(wins) / total * 100) if total > 0 else 0.0
    avg_rr = df["r_multiple"].mean() if "r_multiple" in df.columns else 0.0
    gross_wins = wins["pnl_amount"].sum() if not wins.empty else 0.0
    gross_losses = abs(losses["pnl_amount"].sum()) if not losses.empty else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else gross_wins
    best = df["pnl_pct"].max()
    worst = df["pnl_pct"].min()
    total_pnl = df["pnl_amount"].sum()

    # Current streak
    sorted_trades = df.sort_values("exit_date", ascending=False)
    streak_count = 0
    streak_type = None
    for _, row in sorted_trades.iterrows():
        is_win = row["pnl_pct"] > 0
        if streak_type is None:
            streak_type = is_win
        if is_win == streak_type:
            streak_count += 1
        else:
            break
    streak_label = f"{streak_count}W" if streak_type else f"{streak_count}L"

    return {
        "total": total,
        "win_rate": round(win_rate, 1),
        "avg_rr": round(avg_rr, 2),
        "profit_factor": round(profit_factor, 2),
        "best": round(best, 2),
        "worst": round(worst, 2),
        "streak": streak_label,
        "total_pnl": round(total_pnl, 2),
    }


def _strategy_stats(df: pd.DataFrame, strategy: str) -> dict:
    """Stats for a single strategy."""
    sub = df[df["strategy"].str.lower() == strategy.lower()]
    if sub.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "total_pnl": 0.0}
    wins = sub[sub["pnl_pct"] > 0]
    return {
        "trades": len(sub),
        "win_rate": round(len(wins) / len(sub) * 100, 1),
        "avg_r": round(sub["r_multiple"].mean(), 2),
        "total_pnl": round(sub["pnl_amount"].sum(), 2),
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _build_equity_curve(df: pd.DataFrame) -> go.Figure:
    """Cumulative P&L line chart — gold on dark."""
    sorted_df = df.sort_values("exit_date").copy()
    sorted_df["cum_pnl"] = sorted_df["pnl_amount"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_df["exit_date"],
        y=sorted_df["cum_pnl"],
        mode="lines+markers",
        line=dict(color=_GOLD, width=2),
        marker=dict(size=5, color=_GOLD),
        fill="tozeroy",
        fillcolor="rgba(201,168,76,0.1)",
        name="Cumulative P&L",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_BG_DARK,
        font=dict(color=_TEXT),
        xaxis=dict(gridcolor="rgba(138,133,120,0.15)", title=""),
        yaxis=dict(gridcolor="rgba(138,133,120,0.15)", title="P&L (SEK)"),
        margin=dict(l=40, r=20, t=30, b=30),
        height=320,
        showlegend=False,
    )
    return fig


def _build_monthly_bar(df: pd.DataFrame) -> go.Figure:
    """Monthly P&L bar chart."""
    df_copy = df.copy()
    df_copy["month"] = pd.to_datetime(df_copy["exit_date"]).dt.to_period("M").astype(str)
    monthly = df_copy.groupby("month")["pnl_amount"].sum().reset_index()

    colors = [_GREEN if v >= 0 else _RED for v in monthly["pnl_amount"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month"],
        y=monthly["pnl_amount"],
        marker_color=colors,
        name="Monthly P&L",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_BG_DARK,
        font=dict(color=_TEXT),
        xaxis=dict(gridcolor="rgba(138,133,120,0.15)", title=""),
        yaxis=dict(gridcolor="rgba(138,133,120,0.15)", title="P&L (SEK)"),
        margin=dict(l=40, r=20, t=30, b=30),
        height=300,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------------------------
def _kpi_card(label: str, value: str, color: str = _GOLD):
    """Render a styled KPI metric card."""
    st.markdown(
        f"""<div style="background:{_BG_CARD}; border:1px solid {_GOLD_DIM};
        border-radius:8px; padding:12px 16px; text-align:center;">
        <div style="color:{_TEXT_DIM}; font-size:0.75rem; text-transform:uppercase;
        letter-spacing:1px;">{label}</div>
        <div style="color:{color}; font-size:1.5rem; font-weight:700;
        margin-top:4px;">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Strategy breakdown card
# ---------------------------------------------------------------------------
def _strategy_card(name: str, stats: dict):
    """Render per-strategy stats card."""
    pnl_color = _GREEN if stats["total_pnl"] >= 0 else _RED
    st.markdown(
        f"""<div style="background:{_BG_CARD}; border:1px solid {_GOLD_DIM};
        border-radius:8px; padding:16px;">
        <div style="color:{_GOLD}; font-size:1rem; font-weight:700;
        text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">{name}</div>
        <div style="color:{_TEXT}; font-size:0.85rem;">
        Trades: <b>{stats['trades']}</b><br/>
        Win Rate: <b>{stats['win_rate']}%</b><br/>
        Avg R: <b>{stats['avg_r']}</b><br/>
        Total P&L: <b style="color:{pnl_color}">{stats['total_pnl']:,.0f} SEK</b>
        </div></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------
def render_trade_journal_page():
    """Render the full Trade Journal tab."""
    try:
        trades = load_journal()
        df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # Ensure column types
        if not df.empty:
            for col in ["pnl_pct", "pnl_amount", "r_multiple", "entry_price",
                         "exit_price", "shares"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if "exit_date" in df.columns:
                df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
            if "entry_date" in df.columns:
                df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

        st.markdown(
            f'<h2 style="color:{_GOLD}; margin-bottom:0;">Trade Journal</h2>'
            f'<p style="color:{_TEXT_DIM}; margin-top:2px;">Closed-trade analytics & P&L tracking</p>',
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # 1. SUMMARY KPIs
        # ------------------------------------------------------------------
        kpis = _compute_kpis(df)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1:
            _kpi_card("Total Trades", str(kpis["total"]))
        with c2:
            _kpi_card("Win Rate", f"{kpis['win_rate']}%",
                       _GREEN if kpis["win_rate"] >= 50 else _RED)
        with c3:
            _kpi_card("Avg R:R", f"{kpis['avg_rr']}")
        with c4:
            _kpi_card("Profit Factor", f"{kpis['profit_factor']}")
        with c5:
            _kpi_card("Best Trade", f"{kpis['best']}%", _GREEN)
        with c6:
            _kpi_card("Worst Trade", f"{kpis['worst']}%", _RED)
        with c7:
            _kpi_card("Streak", kpis["streak"])

        st.markdown("<br/>", unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # 2. EQUITY CURVE + MONTHLY PERFORMANCE
        # ------------------------------------------------------------------
        if not df.empty and "exit_date" in df.columns:
            col_eq, col_mo = st.columns(2)
            with col_eq:
                st.markdown(
                    f'<p style="color:{_GOLD}; font-weight:600; margin-bottom:4px;">'
                    f'Equity Curve — Cumulative P&L</p>',
                    unsafe_allow_html=True,
                )
                try:
                    fig_eq = _build_equity_curve(df)
                    st.plotly_chart(fig_eq, use_container_width=True)
                except Exception:
                    st.info("Not enough data for equity curve.")

            with col_mo:
                st.markdown(
                    f'<p style="color:{_GOLD}; font-weight:600; margin-bottom:4px;">'
                    f'Monthly Performance</p>',
                    unsafe_allow_html=True,
                )
                try:
                    fig_mo = _build_monthly_bar(df)
                    st.plotly_chart(fig_mo, use_container_width=True)
                except Exception:
                    st.info("Not enough data for monthly chart.")
        else:
            st.info("No closed trades yet — add trades below to see equity curve and monthly performance.")

        # ------------------------------------------------------------------
        # 3. PER-STRATEGY BREAKDOWN
        # ------------------------------------------------------------------
        st.markdown(
            f'<p style="color:{_GOLD}; font-weight:600; margin-top:16px; margin-bottom:4px;">'
            f'Per-Strategy Breakdown</p>',
            unsafe_allow_html=True,
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            _strategy_card("Wolf", _strategy_stats(df, "wolf"))
        with s2:
            _strategy_card("Viking", _strategy_stats(df, "viking"))
        with s3:
            _strategy_card("Alpha", _strategy_stats(df, "alpha"))

        st.markdown("<br/>", unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # 4. TRADE LOG TABLE
        # ------------------------------------------------------------------
        st.markdown(
            f'<p style="color:{_GOLD}; font-weight:600; margin-bottom:4px;">'
            f'Trade Log</p>',
            unsafe_allow_html=True,
        )

        if not df.empty:
            display_cols = [
                "exit_date", "ticker", "strategy", "direction",
                "entry_price", "exit_price", "shares",
                "pnl_pct", "pnl_amount", "r_multiple", "exit_reason", "notes",
            ]
            existing_cols = [c for c in display_cols if c in df.columns]
            display_df = df[existing_cols].copy()

            # Sort by exit_date descending
            if "exit_date" in display_df.columns:
                display_df = display_df.sort_values("exit_date", ascending=False)

            col_rename = {
                "exit_date": "Date", "ticker": "Ticker", "strategy": "Strategy",
                "direction": "Dir", "entry_price": "Entry", "exit_price": "Exit",
                "shares": "Shares", "pnl_pct": "P&L %", "pnl_amount": "P&L SEK",
                "r_multiple": "R", "exit_reason": "Reason", "notes": "Notes",
            }
            display_df = display_df.rename(columns=col_rename)

            fmt = {}
            for col in ["Entry", "Exit"]:
                if col in display_df.columns:
                    fmt[col] = "{:.2f}"
            for col in ["P&L %", "R"]:
                if col in display_df.columns:
                    fmt[col] = "{:.2f}"
            if "P&L SEK" in display_df.columns:
                fmt["P&L SEK"] = "{:,.0f}"
            if "Shares" in display_df.columns:
                fmt["Shares"] = "{:.0f}"
            if "Date" in display_df.columns:
                display_df["Date"] = pd.to_datetime(display_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

            def _color_pnl(val):
                try:
                    v = float(str(val).replace(",", ""))
                    if v > 0:
                        return f"color: {_GREEN}"
                    elif v < 0:
                        return f"color: {_RED}"
                except (ValueError, TypeError):
                    pass
                return ""

            styled = display_df.style.format(fmt, na_rep="—")
            for col in ["P&L %", "P&L SEK", "R"]:
                if col in display_df.columns:
                    styled = styled.map(_color_pnl, subset=[col])

            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No trades logged yet.")

        # ------------------------------------------------------------------
        # 5. ADD TRADE FORM
        # ------------------------------------------------------------------
        st.markdown(
            f'<p style="color:{_GOLD}; font-weight:600; margin-top:16px; margin-bottom:4px;">'
            f'Add Trade</p>',
            unsafe_allow_html=True,
        )

        with st.form("add_trade_form", clear_on_submit=True):
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                ticker = st.text_input("Ticker", placeholder="e.g. EQNR.OL")
                strategy = st.selectbox("Strategy", ["Wolf", "Viking", "Alpha"])
                direction = st.selectbox("Direction", ["Long", "Short"])
            with fc2:
                entry_date = st.date_input("Entry Date")
                exit_date = st.date_input("Exit Date")
                sector = st.text_input("Sector", placeholder="e.g. Oil & Gas")
            with fc3:
                entry_price = st.number_input("Entry Price", min_value=0.0,
                                               step=0.01, format="%.2f")
                exit_price = st.number_input("Exit Price", min_value=0.0,
                                              step=0.01, format="%.2f")
                shares = st.number_input("Shares", min_value=0, step=1, value=0)
            with fc4:
                r_multiple = st.number_input("R-Multiple", step=0.1, format="%.2f",
                                              value=0.0)
                exit_reason = st.selectbox(
                    "Exit Reason", ["Target", "Stop Loss", "Manual", "Regime Change"]
                )
                notes = st.text_area("Notes", height=80)

            submitted = st.form_submit_button(
                "Log Trade",
                use_container_width=True,
            )

            if submitted:
                try:
                    if not ticker.strip():
                        st.warning("Ticker is required.")
                    else:
                        pnl_pct = 0.0
                        pnl_amount = 0.0
                        if entry_price > 0 and shares > 0:
                            if direction == "Long":
                                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                                pnl_amount = (exit_price - entry_price) * shares
                            else:
                                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                                pnl_amount = (entry_price - exit_price) * shares

                        trade = {
                            "id": str(uuid.uuid4()),
                            "ticker": ticker.strip().upper(),
                            "strategy": strategy.lower(),
                            "entry_date": str(entry_date),
                            "exit_date": str(exit_date),
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exit_price, 2),
                            "shares": int(shares),
                            "direction": direction.lower(),
                            "pnl_pct": round(pnl_pct, 2),
                            "pnl_amount": round(pnl_amount, 2),
                            "r_multiple": round(r_multiple, 2),
                            "exit_reason": exit_reason.lower().replace(" ", "_"),
                            "notes": notes.strip(),
                            "sector": sector.strip(),
                        }

                        trades = load_journal()
                        trades.append(trade)
                        success = save_journal(trades)

                        if success:
                            st.success(f"Trade logged: {trade['ticker']} — "
                                       f"P&L {pnl_pct:+.2f}% (saved to cloud)")
                        else:
                            st.success(f"Trade logged: {trade['ticker']} — "
                                       f"P&L {pnl_pct:+.2f}% (local only)")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error saving trade: {e}")

        # ------------------------------------------------------------------
        # 6. DELETE TRADE (expander)
        # ------------------------------------------------------------------
        if not df.empty:
            with st.expander("Manage Trades"):
                trade_options = []
                for t in trades:
                    label = (f"{t.get('exit_date', '?')} | {t.get('ticker', '?')} | "
                             f"{t.get('strategy', '?')} | P&L {t.get('pnl_pct', 0):+.2f}%")
                    trade_options.append(label)

                if trade_options:
                    selected = st.selectbox("Select trade to delete", trade_options)
                    if st.button("Delete Selected Trade", type="secondary"):
                        try:
                            idx = trade_options.index(selected)
                            trades = load_journal()
                            if 0 <= idx < len(trades):
                                removed = trades.pop(idx)
                                save_journal(trades)
                                st.success(f"Deleted trade: {removed.get('ticker', '?')}")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting trade: {e}")

    except Exception as e:
        st.error(f"Trade Journal error: {e}")
