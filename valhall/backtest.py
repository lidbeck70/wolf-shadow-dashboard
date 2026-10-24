"""
valhall/backtest.py — fas 3: förbättrar cykelstatusen swingstrategierna?

Affärerna kommer ur de befintliga backtesterna (viking_backtest, berserk.backtest — oförändrade). Varje
affär märks med Valhalls läge PÅ SIGNALDAGEN (senast kända rad i cykelmotorns dagliga serie — inget
facit): segmentets status, fas och cykelläge i Norden (sektorindexet) och globalt (SPDR), och för
BERSERK råvarutemats. Sedan jämförs grupperna (alla / GRÖN / GUL / RÖD, per segment och per period) och
portföljen körs med filter (bara GRÖN, GRÖN + GUL …) i viking_portfolio — samma konto som tidigare tester.

Filtret appliceras på färdiga affärer (en spärrad affär lämnar plats åt nästa signal i portföljen, men
inte åt en senare signal i samma aktie). Det är en nära approximation av en regel i själva backtestet.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

import viking_portfolio as vp
from valhall import cycle

PERIODS = (("2008–2020", None, "2020-12-31"), ("2021–", "2021-01-01", None))


def _turn_cols(frame: pd.DataFrame) -> pd.DataFrame:
    """Lägger till vander_days (dagar i den pågående VÄNDER-perioden, 0 utanför) en gång per ram (fas 3b)."""
    if "vander_days" not in frame:
        v = frame["phase"] == cycle.VANDER
        runs = v.ne(v.shift()).cumsum()
        frame["vander_days"] = v.groupby(runs).cumsum().where(v, 0).astype(int)
    return frame


def asof(frame: Optional[pd.DataFrame], date) -> dict:
    """Cykelmotorns rad för signaldagen (sista raden på eller före datumet) — med vändningen (fas 3b):
    confirmed (bekräftad: VÄNDER och stark) och vander_days."""
    if frame is None or frame.empty:
        return {}
    k = int(frame.index.searchsorted(pd.Timestamp(date), side="right")) - 1
    if k < 0:
        return {}
    frame = _turn_cols(frame)
    row = frame.iloc[k]
    return {"status": row["status"], "phase": row["phase"], "cykel": row["cykel"],
            "confirmed": bool(row["confirmed"]) if "confirmed" in frame else False,
            "vander_days": int(row["vander_days"])}


def tag(trades: list, segment_of: Callable, frames: dict, theme_of: Optional[Callable] = None) -> list:
    """[{trade, segment, theme, n_status, n_phase, n_cykel, g_status, t_status, period}] per affär.
    frames = {"nordic": {seg: frame}, "global": {seg: frame}, "themes": {tema: frame}}."""
    out = []
    for t in trades:
        seg = segment_of(t.ticker)
        theme = theme_of(t.ticker) if theme_of else None
        n = asof((frames.get("nordic") or {}).get(seg), t.signal_date)
        g = asof((frames.get("global") or {}).get(seg), t.signal_date)
        th = asof((frames.get("themes") or {}).get(theme), t.signal_date) if theme else {}
        out.append({"trade": t, "segment": seg, "theme": theme,
                    "n_status": n.get("status", cycle.NODATA), "n_phase": n.get("phase"), "n_cykel": n.get("cykel"),
                    "g_status": g.get("status", cycle.NODATA), "t_status": th.get("status", cycle.NODATA),
                    "n_confirmed": n.get("confirmed", False), "n_vdays": n.get("vander_days", 0),
                    "t_phase": th.get("phase"), "t_confirmed": th.get("confirmed", False),
                    "t_vdays": th.get("vander_days", 0),
                    "period": "2008–2020" if t.signal_date <= "2020-12-31" else "2021–"})
    return out


def stats(rows: list) -> dict:
    rs = [r["trade"].r for r in rows if not r["trade"].open and r["trade"].r is not None]
    if not rs:
        return {"n": 0, "exp": None, "win": None, "sum_r": 0.0, "pf": None}
    wins, losses = sum(x for x in rs if x > 0), -sum(x for x in rs if x < 0)
    return {"n": len(rs), "exp": round(float(np.mean(rs)), 3), "win": round(sum(1 for x in rs if x > 0) / len(rs) * 100, 1),
            "sum_r": round(float(np.sum(rs)), 1), "pf": round(wins / losses, 2) if losses > 0 else None}


def table(tagged: list, by: str, key: str = "n_status") -> list:
    """[{group, status, period, n, exp, win, sum_r, pf}] — per grupp (segment/tema/alla) × status × period."""
    out = []
    groups = sorted({r[by] for r in tagged if r[by]}, key=str) if by != "all" else ["alla"]
    for g in groups:
        mine = tagged if by == "all" else [r for r in tagged if r[by] == g]
        for st in ("alla", cycle.GRON, cycle.GUL, cycle.ROD, cycle.NODATA):
            sel = mine if st == "alla" else [r for r in mine if r[key] == st]
            for per, _a, _b in (("alla", None, None),) + PERIODS:
                rows = sel if per == "alla" else [r for r in sel if r["period"] == per]
                s = stats(rows)
                if s["n"]:
                    out.append({"group": g, "status": st, "period": per, **s})
    return out


FILTERS = {
    "Alla affärer": lambda r: True,
    "Norden GRÖN": lambda r: r["n_status"] == cycle.GRON,
    "Norden GRÖN + GUL": lambda r: r["n_status"] in (cycle.GRON, cycle.GUL),
    "Norden ej RÖD": lambda r: r["n_status"] != cycle.ROD,
    "Globalt GRÖN": lambda r: r["g_status"] == cycle.GRON,
    "Norden och globalt GRÖN": lambda r: r["n_status"] == cycle.GRON and r["g_status"] == cycle.GRON,
    "Tema GRÖN": lambda r: r["t_status"] == cycle.GRON,
    "Tema ej RÖD": lambda r: r["t_status"] != cycle.ROD,
}


# Fas 3b (analysen 2026-10-24, beslut 4 och K9): VÄNDER-fasen och den bekräftade vändningen som swinggrind —
# fas 3a prövade bara status (GRÖN/GUL/RÖD). Mäts på färdiga affärer; ingen regel ändras.
FILTERS_3B = {
    "Alla affärer": lambda r: True,
    "Norden VÄNDER": lambda r: r["n_phase"] == cycle.VANDER,
    "Norden VÄNDER bekräftad": lambda r: r["n_phase"] == cycle.VANDER and r["n_confirmed"],
    "Norden VÄNDER ≤ 20 d": lambda r: 0 < r["n_vdays"] <= 20,
    "Norden VÄNDER ≤ 60 d": lambda r: 0 < r["n_vdays"] <= 60,
    "Norden GUL": lambda r: r["n_status"] == cycle.GUL,
}
FILTERS_3B_THEME = {
    "Alla affärer": lambda r: True,
    "Tema VÄNDER": lambda r: r["t_phase"] == cycle.VANDER,
    "Tema VÄNDER bekräftad": lambda r: r["t_phase"] == cycle.VANDER and r["t_confirmed"],
    "Tema VÄNDER ≤ 20 d": lambda r: 0 < r["t_vdays"] <= 20,
    "Tema VÄNDER ≤ 60 d": lambda r: 0 < r["t_vdays"] <= 60,
    "Tema GUL": lambda r: r["t_status"] == cycle.GUL,
}


def portfolios(tagged: list, years: float, pc: vp.PortfolioConfig = vp.PortfolioConfig(),
               filters: Optional[dict] = None, period: Optional[str] = None) -> list:
    """Portföljen (viking_portfolio.simulate) per filter: affärer, CAGR, DD, expectancy och MC."""
    import viking_robustness as rb
    out = []
    for name, f in (filters or FILTERS).items():
        rows = [r for r in tagged if f(r) and (period is None or r["period"] == period)]
        trades = [r["trade"] for r in rows]
        p = vp.simulate(trades, years=years, pc=pc)
        mc = next((m for m in rb.monte_carlo(p.get("rows") or [], years) if m["Metod"].startswith("Bootstrap +")), {})
        out.append({"filter": name, "trades": len(trades), "taken": p["taken"], "exp": stats(rows)["exp"],
                    "port_exp": (p.get("metrics") or {}).get("expectancy"), "cagr": p.get("cagr_pct"),
                    "dd": p["max_dd_pct"], "mc_p5": mc.get("CAGR p5 %"), "mc_p50": mc.get("CAGR p50 %"),
                    "mc_dd95": mc.get("DD p95 %")})
    return out
