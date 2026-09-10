"""
alpha_regime/screener_link.py
Koppling mellan Alpha Regime-fliken och Deep Contrarian-screenern.

Screenern svarar på VAD (hatade bolag med god ekonomi), regimfliken på NÄR.
Här läses senaste sparade Deep Contrarian-körning så att fliken kan erbjuda
tickers ur listan och visa bolagets screener-status i kortet. Rena funktioner
utan Streamlit — testbara.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SCREENER_MODE = "deep_contrarian"


@dataclass
class ScreenerRow:
    ticker: str            # screenerns form, t.ex. "EKTA B.ST"
    yf_ticker: str         # yfinance-form, t.ex. "EKTA-B.ST"
    name: str = ""
    rank: int = 0
    composite: Optional[float] = None
    hat: Optional[float] = None
    necessity: Optional[float] = None
    altman_z: Optional[float] = None
    net_debt_ebitda: Optional[float] = None
    roic: Optional[float] = None
    close: Optional[float] = None
    sma200: Optional[float] = None
    branch: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"#{self.rank} {self.yf_ticker} — {self.name}" if self.name else f"#{self.rank} {self.yf_ticker}"

    @property
    def pct_vs_sma200(self) -> Optional[float]:
        if self.close and self.sma200:
            return round((self.close / self.sma200 - 1) * 100, 1)
        return None


def to_yf_ticker(ticker: str) -> str:
    """Börsdata/screener-form 'EKTA B.ST' → yfinance-form 'EKTA-B.ST'."""
    return str(ticker or "").strip().upper().replace(" ", "-")


def _norm(ticker: str) -> str:
    return to_yf_ticker(ticker)


def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def rows_from_payload(payload: dict) -> list[ScreenerRow]:
    """Bygg ScreenerRow-lista ur load_screener_results()-payload, rankordnad."""
    rows: list[ScreenerRow] = []
    for d in (payload or {}).get("results") or []:
        t = str(d.get("ticker") or "").strip()
        if not t:
            continue
        rows.append(ScreenerRow(
            ticker=t,
            yf_ticker=to_yf_ticker(t),
            name=str(d.get("name") or ""),
            rank=int(d.get("rank") or 0),
            composite=_f(d.get("composite_score")),
            hat=_f(d.get("hat_score")),
            necessity=_f(d.get("necessity_score")),
            altman_z=_f(d.get("altman_z")),
            net_debt_ebitda=_f(d.get("net_debt_ebitda")),
            roic=_f(d.get("roic")),
            close=_f(d.get("close")),
            sma200=_f(d.get("sma200")),
            branch=str(d.get("branch") or d.get("sector") or ""),
            flags=[str(x) for x in (d.get("all_flags") or [])],
        ))
    rows.sort(key=lambda r: (r.rank if r.rank > 0 else 10_000, r.yf_ticker))
    return rows


def find_row(rows: list[ScreenerRow], ticker: str) -> Optional[ScreenerRow]:
    """Hitta raden för en ticker oavsett mellanslag/bindestreck och skiftläge."""
    key = _norm(ticker)
    if not key:
        return None
    for r in rows:
        if r.yf_ticker == key:
            return r
    return None


def load_deep_contrarian_rows() -> tuple[list[ScreenerRow], str]:
    """Senaste Deep Contrarian-körningen (Gist → lokal fil). Returnerar (rader, tidsstämpel)."""
    try:
        from contrarian_alpha.cache import load_screener_results
        payload = load_screener_results(mode=SCREENER_MODE)
    except Exception as exc:
        logger.debug("load_screener_results failed: %s", exc)
        return [], ""
    ts = str((payload or {}).get("timestamp") or "")
    return rows_from_payload(payload), ts
