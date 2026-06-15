"""
data_health.py
Live status of every data source the panel depends on.
Each probe returns (status, detail): status in {"OK","FALLBACK","OFF","ERROR"}.
Read-only — never mutates app state. Cached 5 min in the UI layer.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass


@dataclass
class SourceStatus:
    name: str
    tier: str            # "Karna" | "Tillval" | "Gratis" | "Makro" | "Sentiment"
    status: str          # OK | FALLBACK | OFF | ERROR
    detail: str
    fallback: str = ""   # what is used instead when OFF


def _key(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "")
        if v:
            return v
    try:
        import streamlit as st
        for n in names:
            v = st.secrets.get(n, "")
            if v:
                return v
    except Exception:
        pass
    return ""


def probe_borsdata() -> SourceStatus:
    """Core Nordic fundamentals. Without it, Contrarian Alpha + EMBER Nordic lose depth."""
    try:
        from borsdata_api import get_api
        api = get_api()
        if not api.is_configured:
            return SourceStatus("Borsdata", "Karna", "OFF",
                                "Ingen API-nyckel - nordisk fundamentaldata saknas",
                                fallback="yfinance (begransad)")
        ins = api.resolve_instrument_id("VOLV-B.ST")
        if ins:
            return SourceStatus("Borsdata", "Karna", "OK",
                                "Ansluten - nordisk fundamentaldata (ROIC, marginaler, tillvaxt, insynshandel)")
        return SourceStatus("Borsdata", "Karna", "ERROR",
                            "Nyckel finns men instrument kunde ej slas upp")
    except Exception as e:
        return SourceStatus("Borsdata", "Karna", "ERROR", f"Fel: {type(e).__name__}: {e}"[:90])


def probe_eodhd() -> SourceStatus:
    """Optional US/global enrichment: short interest, analyst up/downgrades, options."""
    if not _key("EODHD_API_KEY"):
        return SourceStatus("EODHD", "Tillval", "OFF",
                            "Ingen nyckel - paverkar EJ karnan. Short interest/analytiker/options tunnare.",
                            fallback="yfinance (analytiker) + Borsdata (insynshandel)")
    return SourceStatus("EODHD", "Tillval", "OK",
                        "Ansluten - US short interest, analytiker-upp/nedgraderingar, options")


def probe_yfinance() -> SourceStatus:
    """Free EOD prices + basic fundamentals globally. The universal fallback."""
    try:
        import yfinance as yf
        df = yf.download("SPY", period="5d", progress=False, multi_level_index=False)
        if df is not None and not df.empty:
            return SourceStatus("yfinance", "Gratis", "OK",
                                "Gratis EOD-priser + grundlaggande fundamenta (global)")
        return SourceStatus("yfinance", "Gratis", "ERROR", "Tomt svar for SPY")
    except Exception as e:
        return SourceStatus("yfinance", "Gratis", "ERROR", f"Fel: {type(e).__name__}"[:90])


def probe_fred() -> SourceStatus:
    """Yield curve (T10Y2Y) for EMBER macro pillar. Free, no key."""
    try:
        from ember.fred_cache import fetch_t10y2y_values
        try:
            from ember.config import FRED_T10Y2Y_URL as _FURL
        except Exception:
            _FURL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
        vals, stale = fetch_t10y2y_values(_FURL)
        if vals:
            tag = " (cachad)" if stale else ""
            return SourceStatus("FRED (rantekurva)", "Makro", "OK",
                                f"T10Y2Y hamtad{tag} - {len(vals)} punkter")
        return SourceStatus("FRED (rantekurva)", "Makro", "FALLBACK", "Tomt svar - rantekurve-pelaren hoppas over")
    except RuntimeError:
        return SourceStatus("FRED (rantekurva)", "Makro", "FALLBACK",
                            "FRED ej naabar + ingen cache - EMBER-makro tappar rantekurvan (ej kritiskt)",
                            fallback="Ovriga makropelare (DXY, koppar/guld)")
    except Exception as e:
        return SourceStatus("FRED (rantekurva)", "Makro", "FALLBACK", f"Degraderad: {type(e).__name__}"[:90])


def probe_sentiment() -> SourceStatus:
    """Retail sentiment aggregators (ApeWisdom/StockTwits) - free, best-effort."""
    try:
        import requests
        r = requests.get("https://apewisdom.io/api/v1.0/filter/all-stocks/page/1",
                         timeout=8)
        if r.status_code == 200:
            return SourceStatus("Sentiment (ApeWisdom/StockTwits)", "Sentiment", "OK",
                                "Retail-flode tillgangligt")
        return SourceStatus("Sentiment (ApeWisdom/StockTwits)", "Sentiment", "FALLBACK",
                            f"HTTP {r.status_code} - degraderad")
    except Exception:
        return SourceStatus("Sentiment (ApeWisdom/StockTwits)", "Sentiment", "FALLBACK",
                            "Ej naabar just nu - sentiment hoppar over")


def run_all_probes() -> list[SourceStatus]:
    """Run every probe. Order: core first, then optional/free/macro/sentiment."""
    return [
        probe_borsdata(),
        probe_yfinance(),
        probe_eodhd(),
        probe_fred(),
        probe_sentiment(),
    ]


# ── UI ──────────────────────────────────────────────────────────────────────
_STATUS_STYLE = {
    "OK":       ("#2d8a4e", "AKTIV"),
    "FALLBACK": ("#00E5FF", "FALLBACK"),
    "OFF":      ("#6B7280", "AV"),
    "ERROR":    ("#FF6B3D", "FEL"),
}


def render_data_health() -> None:
    """Render the Data Health panel. Self-contained; safe to call from any tab."""
    import streamlit as st

    st.markdown(
        "<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.4rem;"
        "font-weight:700;color:#E8EDF2;margin-bottom:2px;'>Data Health</div>"
        "<div style='font-size:0.85rem;color:#6B7280;margin-bottom:18px;'>"
        "Live-status for varje datakalla panelen anvander. Karnan kraver bara "
        "Borsdata + yfinance - allt annat ar forfining.</div>",
        unsafe_allow_html=True,
    )

    @st.cache_data(ttl=300, show_spinner="Testar datakallor...")
    def _cached_probes():
        return [(s.name, s.tier, s.status, s.detail, s.fallback) for s in run_all_probes()]

    rows = _cached_probes()

    for name, tier, status, detail, fallback in rows:
        color, label = _STATUS_STYLE.get(status, ("#6B7280", status))
        fb = (f"<div style='font-size:0.78rem;color:#6B7280;margin-top:4px;'>"
              f"&#8627; Fallback: {fallback}</div>") if (fallback and status in ("OFF","FALLBACK")) else ""
        st.markdown(
            f"<div style='background:#1A1F25;border:1px solid rgba(0,229,255,0.12);"
            f"border-left:3px solid {color};border-radius:8px;padding:12px 16px;margin-bottom:8px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-weight:600;color:#E8EDF2;'>{name}"
            f"<span style='font-size:0.72rem;color:#6B7280;font-weight:400;'> &middot; {tier}</span></span>"
            f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.75rem;"
            f"font-weight:700;color:{color};border:1px solid {color};border-radius:4px;"
            f"padding:2px 8px;'>{label}</span></div>"
            f"<div style='font-size:0.82rem;color:#9aa4b0;margin-top:5px;'>{detail}</div>"
            f"{fb}</div>",
            unsafe_allow_html=True,
        )

    if st.button("Uppdatera status", key="dh_refresh"):
        _cached_probes.clear()
        st.rerun()

    st.markdown(
        "<div style='margin-top:14px;font-size:0.8rem;color:#6B7280;'>"
        "<b style='color:#9aa4b0;'>Tolkning:</b> Karna AV = panelen tappar viktig funktion. "
        "Tillval/Makro/Sentiment AV eller FALLBACK = panelen fungerar fullt ut, bara med "
        "lite grovre forfining. EODHD kan pausas helt under byggfasen.</div>",
        unsafe_allow_html=True,
    )
