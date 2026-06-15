"""
position_sizing.py
Position Sizing Calculator for Nordic Arc Systems.

Three modes, each matching a strategy family:
  TRADING (Wolf/Viking/Ember swing) : fixed % risk, ATR-based stop
  QUALITY (Buffett/KAP)             : conviction-based, max 10%, 8-10 holdings
  DEEP_CONTRARIAN (Rule/Sprott)     : staged thirds, max 25%/sector

Pure calculation layer (dataclasses) + a Streamlit UI renderer.
ATR can be fetched automatically (yfinance) or entered manually.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# Named constants (tunable)
DEFAULT_TRADING_ACCOUNT = 75_000.0   # SEK
DEFAULT_ASK_ACCOUNT     = 37_000.0   # SEK
DEFAULT_RISK_PCT        = 2.0        # % of account per swing trade
DEFAULT_ATR_MULT        = 2.5        # stop = entry - mult * ATR
QUALITY_MAX_POS_PCT     = 10.0       # max single position (Quality)
QUALITY_MIN_HOLDINGS    = 8
QUALITY_MAX_HOLDINGS    = 10
DEEP_MAX_POS_PCT        = 10.0       # full target before tranching
DEEP_MAX_SECTOR_PCT     = 25.0       # sector concentration cap
DEEP_TRANCHES           = 3


@dataclass
class TradingResult:
    account: float
    risk_pct: float
    entry: float
    stop: float
    atr: Optional[float]
    atr_mult: float
    risk_per_share: float = 0.0
    risk_amount: float = 0.0
    shares: int = 0
    position_value: float = 0.0
    position_pct: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    warnings: list = field(default_factory=list)


def calc_trading(account: float, risk_pct: float, entry: float,
                 stop: Optional[float] = None, atr: Optional[float] = None,
                 atr_mult: float = DEFAULT_ATR_MULT) -> TradingResult:
    """Swing sizing: shares = (account * risk%) / (entry - stop).
    If stop not given, derive from ATR: stop = entry - atr_mult * atr."""
    warnings = []
    if stop is None:
        if atr and atr > 0:
            stop = entry - atr_mult * atr
        else:
            warnings.append("Ingen stop och ingen ATR - kan ej rakna.")
            return TradingResult(account, risk_pct, entry, 0.0, atr, atr_mult, warnings=warnings)

    risk_per_share = entry - stop
    res = TradingResult(account, risk_pct, entry, stop, atr, atr_mult)
    if risk_per_share <= 0:
        warnings.append("Stop maste ligga under entry (lang position).")
        res.warnings = warnings
        return res

    risk_amount = account * (risk_pct / 100.0)
    shares = int(risk_amount // risk_per_share)
    position_value = shares * entry
    position_pct = (position_value / account * 100.0) if account else 0.0

    res.risk_per_share = round(risk_per_share, 4)
    res.risk_amount    = round(risk_amount, 2)
    res.shares         = shares
    res.position_value = round(position_value, 2)
    res.position_pct   = round(position_pct, 1)
    res.target1        = round(entry + 2 * risk_per_share, 4)   # 1:2
    res.target2        = round(entry + 3 * risk_per_share, 4)   # 1:3

    if position_pct > 100:
        warnings.append(f"Positionen ({position_pct:.0f}%) overstiger kontot - sank risk% eller vidga stop.")
    elif position_pct > 50:
        warnings.append(f"Stor position ({position_pct:.0f}% av kontot) - tight stop ger stor exponering.")
    res.warnings = warnings
    return res


@dataclass
class LongResult:
    account: float
    mode: str               # "quality" | "deep_contrarian"
    conviction_pct: float   # target position as % of account
    target_value: float = 0.0
    target_pct: float = 0.0
    tranche_value: float = 0.0      # deep_contrarian: 1/3 of target
    tranche_pct: float = 0.0
    tranches_total: int = 1
    next_tranche_n: int = 1
    entry: Optional[float] = None
    shares_full: int = 0
    shares_tranche: int = 0
    warnings: list = field(default_factory=list)


def calc_long(account: float, mode: str, conviction_pct: float,
              entry: Optional[float] = None,
              tranches_deployed: int = 0) -> LongResult:
    """Conviction sizing for long-term positions.
    Quality: single target up to 10%.
    Deep Contrarian: full target tranched into thirds; shows next tranche."""
    warnings = []
    cap = QUALITY_MAX_POS_PCT if mode == "quality" else DEEP_MAX_POS_PCT
    target_pct = min(conviction_pct, cap)
    if conviction_pct > cap:
        warnings.append(f"Conviction {conviction_pct:.0f}% kapad till takets {cap:.0f}% for {mode}.")

    target_value = account * (target_pct / 100.0)
    res = LongResult(account, mode, conviction_pct, entry=entry)
    res.target_value = round(target_value, 2)
    res.target_pct = round(target_pct, 1)

    if mode == "deep_contrarian":
        res.tranches_total = DEEP_TRANCHES
        res.tranche_value = round(target_value / DEEP_TRANCHES, 2)
        res.tranche_pct = round(target_pct / DEEP_TRANCHES, 2)
        res.next_tranche_n = min(tranches_deployed + 1, DEEP_TRANCHES)
        if tranches_deployed >= DEEP_TRANCHES:
            warnings.append("Alla 3 trancher redan utplacerade - full position uppnadd.")
    else:
        res.tranche_value = res.target_value
        res.tranche_pct = res.target_pct

    if entry and entry > 0:
        res.shares_full = int(target_value // entry)
        res.shares_tranche = int(res.tranche_value // entry)

    res.warnings = warnings
    return res


def fetch_atr(ticker: str, period_days: int = 14) -> Optional[float]:
    """Fetch ATR(14) from daily data via yfinance. Returns None on failure."""
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download(ticker, period="3mo", progress=False, multi_level_index=False)
        if df is None or df.empty or len(df) < period_days + 1:
            return None
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low),
                        (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period_days).mean().iloc[-1]
        return float(atr) if atr == atr else None   # NaN check
    except Exception:
        return None


# UI helpers
_P = {
    "cyan": "#00E5FF", "purple": "#B400FF", "ember": "#FF6B3D",
    "green": "#2d8a4e", "text": "#E8EDF2", "dim": "#6B7280", "card": "#1A1F25",
}


def _metric_card(label, value, color="#E8EDF2", sub=""):
    import streamlit as st
    subhtml = f"<div style='font-size:0.72rem;color:#6B7280;margin-top:2px;'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div style='background:#1A1F25;border:1px solid rgba(0,229,255,0.12);"
        f"border-radius:8px;padding:12px 14px;'>"
        f"<div style='font-size:0.72rem;letter-spacing:1.5px;text-transform:uppercase;"
        f"color:#6B7280;'>{label}</div>"
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:1.25rem;"
        f"font-weight:700;color:{color};margin-top:3px;'>{value}</div>{subhtml}</div>",
        unsafe_allow_html=True,
    )


def render_position_sizing() -> None:
    """Render the Position Sizing calculator. Self-contained Streamlit UI."""
    import streamlit as st

    st.markdown(
        "<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.4rem;"
        "font-weight:700;color:#E8EDF2;margin-bottom:2px;'>Position Sizing</div>"
        "<div style='font-size:0.85rem;color:#6B7280;margin-bottom:16px;'>"
        "Rakna ut exakt positionsstorlek per strategi. Disciplin > magkansla.</div>",
        unsafe_allow_html=True,
    )

    strat = st.radio("Strategi", ["Trading (Wolf/Viking/Ember)",
                                  "Quality (langsiktig)",
                                  "Deep Contrarian (stegad)"],
                     horizontal=True, key="ps_strat")
    st.markdown("---")

    if strat.startswith("Trading"):
        _render_trading_ui()
    elif strat.startswith("Quality"):
        _render_long_ui("quality")
    else:
        _render_long_ui("deep_contrarian")


def _render_trading_ui() -> None:
    import streamlit as st
    c1, c2, c3 = st.columns(3)
    with c1:
        account = st.number_input("Kontovarde (kr)", min_value=0.0,
                                  value=DEFAULT_TRADING_ACCOUNT, step=1000.0, key="ps_t_acc")
        risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0,
                                   value=DEFAULT_RISK_PCT, step=0.5, key="ps_t_risk")
    with c2:
        ticker = st.text_input("Ticker (for auto-ATR)", value="", key="ps_t_ticker",
                               placeholder="t.ex. BOL.ST")
        entry = st.number_input("Entry-pris", min_value=0.0, value=100.0, step=1.0, key="ps_t_entry")
    with c3:
        atr_mult = st.number_input("ATR-multipel (stop)", min_value=0.5, max_value=6.0,
                                   value=DEFAULT_ATR_MULT, step=0.5, key="ps_t_mult")
        manual_stop = st.number_input("Manuell stop (0 = anvand ATR)", min_value=0.0,
                                      value=0.0, step=1.0, key="ps_t_stop")

    atr_val = None
    if ticker.strip():
        if st.button("Hamta ATR(14)", key="ps_t_fetchatr"):
            with st.spinner("Hamtar ATR..."):
                atr_val = fetch_atr(ticker.strip().upper())
                st.session_state["ps_t_atr_val"] = atr_val
        atr_val = st.session_state.get("ps_t_atr_val", atr_val)
    atr_override = st.number_input("ATR (override, 0 = auto)", min_value=0.0,
                                   value=float(atr_val) if atr_val else 0.0, step=0.1, key="ps_t_atroverride")
    use_atr = atr_override if atr_override > 0 else atr_val
    use_stop = manual_stop if manual_stop > 0 else None

    res = calc_trading(account, risk_pct, entry, stop=use_stop, atr=use_atr, atr_mult=atr_mult)
    st.markdown("---")
    if res.shares > 0:
        cc = st.columns(4)
        with cc[0]: _metric_card("Antal aktier", f"{res.shares}", _P["cyan"])
        with cc[1]: _metric_card("Positionsvarde", f"{res.position_value:,.0f} kr".replace(",", " "), _P["text"], f"{res.position_pct:.0f}% av konto")
        with cc[2]: _metric_card("Stop", f"{res.stop:.2f}", _P["ember"], f"risk {res.risk_per_share:.2f}/aktie")
        with cc[3]: _metric_card("Riskbelopp", f"{res.risk_amount:,.0f} kr".replace(",", " "), _P["ember"], f"{res.risk_pct:.1f}% av konto")
        cc2 = st.columns(2)
        with cc2[0]: _metric_card("Target 1 (1:2)", f"{res.target1:.2f}", _P["green"])
        with cc2[1]: _metric_card("Target 2 (1:3)", f"{res.target2:.2f}", _P["green"])
    for w in res.warnings:
        st.warning(w)


def _render_long_ui(mode: str) -> None:
    import streamlit as st
    is_deep = mode == "deep_contrarian"
    accent = _P["ember"] if is_deep else _P["cyan"]
    cyan = _P["cyan"]

    c1, c2, c3 = st.columns(3)
    with c1:
        account = st.number_input("Kontovarde ASK (kr)", min_value=0.0,
                                  value=DEFAULT_ASK_ACCOUNT, step=1000.0, key=f"ps_l_acc_{mode}")
    with c2:
        cap = DEEP_MAX_POS_PCT if is_deep else QUALITY_MAX_POS_PCT
        conviction = st.number_input("Conviction / malstorlek (%)", min_value=0.5, max_value=30.0,
                                     value=float(cap), step=0.5, key=f"ps_l_conv_{mode}")
    with c3:
        entry = st.number_input("Entry-pris (valfritt)", min_value=0.0, value=0.0,
                                step=1.0, key=f"ps_l_entry_{mode}")

    deployed = 0
    if is_deep:
        deployed = st.slider("Trancher redan utplacerade", 0, DEEP_TRANCHES, 0, key="ps_l_deployed")

    res = calc_long(account, mode, conviction, entry=(entry if entry > 0 else None),
                    tranches_deployed=deployed)
    st.markdown("---")

    cc = st.columns(3)
    with cc[0]:
        _metric_card("Full malposition", f"{res.target_value:,.0f} kr".replace(",", " "),
                     accent, f"{res.target_pct:.1f}% av konto")
    if entry > 0:
        with cc[1]:
            _metric_card("Aktier (full)", f"{res.shares_full}", _P["text"], f"@ {entry:.2f}")
    if is_deep:
        with cc[2]:
            _metric_card(f"Nasta tranche ({res.next_tranche_n}/3)",
                         f"{res.tranche_value:,.0f} kr".replace(",", " "),
                         cyan, f"{res.tranche_pct:.1f}% av konto")
        box = (
            "<div style='background:#1A1F25;border-left:3px solid " + _P["ember"] +
            ";border-radius:8px;padding:12px 16px;margin-top:10px;'>"
            "<b style='color:#E8EDF2;'>Stegad plan (Rule/Sprott):</b> "
            "<span style='color:#9aa4b0;'>Placera 1/3 nu, 1/3 vid basing/hogre botten, "
            "1/3 vid MA200-atertag. Detta ar tranche " + str(res.next_tranche_n) + " av 3.</span></div>"
        )
        st.markdown(box, unsafe_allow_html=True)
    else:
        box = (
            "<div style='background:#1A1F25;border-left:3px solid " + _P["cyan"] +
            ";border-radius:8px;padding:12px 16px;margin-top:10px;'>"
            "<b style='color:#E8EDF2;'>Quality-regel:</b> "
            "<span style='color:#9aa4b0;'>" + str(QUALITY_MIN_HOLDINGS) + "-" + str(QUALITY_MAX_HOLDINGS) +
            " innehav, max " + str(int(QUALITY_MAX_POS_PCT)) + "% per position. "
            "Kop hela positionen nar alla 4 villkor ar grona.</span></div>"
        )
        st.markdown(box, unsafe_allow_html=True)
    for w in res.warnings:
        st.warning(w)
