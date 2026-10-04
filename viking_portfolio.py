"""
viking_portfolio.py — portföljläget för Viking Nine-backtestet.

viking_backtest testar varje ticker för sig, som om kapitalet räckte till
alla signaler samtidigt. Här körs samma affärer genom ETT konto med Vikings
egna gränser (viking_execution):

  * positionsstorlek: risk 1,5 % av kapitalet på stoppen, men aldrig mer än
    25 % av kapitalet i en position (då blir risken mindre)
  * ingen belåning: summan av öppna positioner ≤ 100 % av kapitalet
  * max två förlustaffärer per dag — fler förluster på signaldagen → ingen
    ny entry nästa morgon
  * fler signaler än plats samma dag → högst Nine först, sedan starkast
    63-dagarsavkastning (samma ordning som screenern)

  * en aktie per sektor (sektor-ETF:en) — som live, där skannern visar
    SEKTOR UPPTAGEN (backtest: lägre drawdown, högre avkastning i Norden)

OVTLYR-test (av som förval): 'bäst historik först' — prioritet efter
aktiens tidigare stängda affärer i R (walk-forward, OVTLYR: "start with
the highest Signal Return").

Avkastningen räknas på kontot (ränta på ränta, stängda affärer). Drawdown
räknas DAGSVÄRDERAT (mtm_curve): öppna positioner värderas till dagens
stängning, så att förluster under en affärs löptid syns — drawdown på bara
stängda affärer underskattar risken (visas som jämförelse). Förenkling: när en affär hoppas över tas inte en
senare signal i samma aktie under den affärens löptid.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import viking_execution as vx


@dataclass
class PortfolioConfig:
    risk_pct: float = vx.MAX_RISK_PCT              # 1,5 % av kapitalet på stoppen
    max_position_pct: float = vx.MAX_POSITION_PCT  # 25 % per position
    max_exposure_pct: float = 100.0                # ingen belåning
    max_daily_losses: int = vx.MAX_DAILY_LOSSES    # 2
    one_per_sector: bool = True                    # högst en öppen position per sektor (som live)
    history_first: bool = False                    # OVTLYR: bäst egen historik (hist_r) först, sedan Nine


PORTFOLIO_RULES = {
    "history_first": ("Bäst historik först", "fler signaler än plats → aktien med högst summa R i tidigare "
                                             "stängda affärer först (walk-forward), sedan Nine och momentum"),
}


NOTES = (
    "Portföljläget kör affärerna genom ett konto: risk {risk:g} % per affär, max {pos:g} % per position, "
    "max {exp:g} % investerat och max {loss} förluster per dag. Hoppas en affär över tas ingen senare "
    "signal i samma aktie under dess löptid (förenkling)."
)


def position_pct(t, pc: PortfolioConfig = PortfolioConfig()) -> float:
    """Positionens andel av kapitalet (%): risken på stoppen, taket per position."""
    stop_pct = t.risk / t.entry * 100 if t.entry and t.risk else 0.0
    if stop_pct <= 0:
        return 0.0
    return min(pc.risk_pct / stop_pct * 100, pc.max_position_pct)


def _priority(t, history_first: bool = False) -> tuple:
    mom = getattr(t, "mom63", None)
    hist = getattr(t, "hist_r", None) if history_first else None
    return (t.entry_date, -(hist if hist is not None else 0.0), -int(t.nine),
            -(mom if mom is not None else float("-inf")), t.ticker)


def simulate(trades: list, years: Optional[float] = None, pc: PortfolioConfig = PortfolioConfig()) -> dict:
    """Kör affärerna (viking_backtest.Trade) i datumordning genom ett konto."""
    from viking_backtest import metrics

    open_, taken, rows = [], [], []
    losses = Counter()
    skipped_full = skipped_losses = skipped_sector = max_open = 0
    for t in sorted(trades, key=lambda t: _priority(t, pc.history_first)):
        open_ = [(o, p) for o, p in open_ if o.open or o.exit_date >= t.entry_date]
        if losses[t.signal_date] >= pc.max_daily_losses:
            skipped_losses += 1
            continue
        sector = getattr(t, "sector", None)
        if pc.one_per_sector and sector and any(getattr(o, "sector", None) == sector for o, _p in open_):
            skipped_sector += 1
            continue
        pp = position_pct(t, pc)
        if pp <= 0 or sum(p for _o, p in open_) + pp > pc.max_exposure_pct + 1e-9:
            skipped_full += 1
            continue
        open_.append((t, pp))
        max_open = max(max_open, len(open_))
        taken.append(t)
        risk_taken = pp * (t.risk / t.entry)
        ret = None if t.open or t.r is None else t.r * risk_taken
        rows.append({"trade": t, "position_pct": round(pp, 1), "risk_pct": round(risk_taken, 2),
                     "return_pct": None if ret is None else round(ret, 2)})
        if not t.open and t.r is not None and t.r < 0:
            losses[t.exit_date] += 1

    mtm = mtm_curve(rows, pc)
    closed = sorted((r for r in rows if r["return_pct"] is not None),
                    key=lambda r: (r["trade"].exit_date, r["trade"].ticker))
    eq, peak, max_dd, curve = 1.0, 1.0, 0.0, []
    for r in closed:
        eq *= 1 + r["return_pct"] / 100
        peak = max(peak, eq)
        max_dd = max(max_dd, 1 - eq / peak)
        curve.append((r["trade"].exit_date, round((eq - 1) * 100, 2)))
    cagr = None
    if years and years > 0 and closed and eq > 0:
        cagr = round((eq ** (1 / float(years)) - 1) * 100, 1)
    return {
        "metrics": metrics(taken), "rows": rows, "curve": curve, "candidates": len(trades), "taken": len(taken),
        "skipped_full": skipped_full, "skipped_losses": skipped_losses, "skipped_sector": skipped_sector,
        "max_open": max_open,
        "return_pct": round((eq - 1) * 100, 1), "cagr_pct": cagr,
        "max_dd_pct": mtm["max_dd_pct"] if mtm else round(max_dd * 100, 1),     # dagsvärderad när kurser finns
        "closed_dd_pct": round(max_dd * 100, 1), "mtm": mtm,
        "cap_share": round(sum(1 for r in rows if r["position_pct"] >= pc.max_position_pct - 1e-9)
                           / len(rows) * 100, 1) if rows else None,
        "avg_position_pct": round(sum(r["position_pct"] for r in rows) / len(rows), 1) if rows else None,
        "avg_risk_pct": round(sum(r["risk_pct"] for r in rows) / len(rows), 2) if rows else None,
        "note": NOTES.format(risk=pc.risk_pct, pos=pc.max_position_pct, exp=pc.max_exposure_pct,
                             loss=pc.max_daily_losses)
        + (" En aktie per sektor." if pc.one_per_sector else " Flera aktier per sektor tillåtna.")
        + "".join(f" OVTLYR: {PORTFOLIO_RULES[k][0].lower()}." for k in PORTFOLIO_RULES if getattr(pc, k, False)),
    }


def mtm_curve(rows: list, pc: PortfolioConfig = PortfolioConfig()) -> Optional[dict]:
    """Kontot dag för dag med öppna positioner värderade till dagens stängning.

    Realiserat kapital E växer med ränta på ränta vid varje exit; en dag är kontot
    E × (1 + Σ position % × (stängning / entry − 1)) för de öppna positionerna.
    Exitdagen räknas till exitkursen. Kräver Trade.dates/path — annars None."""
    trades = [r for r in rows if getattr(r["trade"], "dates", None) and getattr(r["trade"], "path", None)]
    if not trades or len(trades) < len(rows):
        return None
    days = sorted({d for r in trades for d in r["trade"].dates})
    by_day = [{} for _ in days]
    pos = {d: k for k, d in enumerate(days)}
    for n, r in enumerate(trades):
        for d, c in zip(r["trade"].dates, r["trade"].path):
            by_day[pos[d]][n] = c
    ends = {}
    for n, r in enumerate(trades):
        t = r["trade"]
        if not t.open:
            ends.setdefault(t.exit_date, []).append(n)
    first = {n: r["trade"].dates[0] for n, r in enumerate(trades)}
    last_close, active = {}, set()
    eq_real, peak, max_dd, curve, expo = 1.0, 1.0, 0.0, [], []
    for k, d in enumerate(days):
        for n, c in by_day[k].items():
            last_close[n] = c
            if first[n] == d:
                active.add(n)
        for n in ends.get(d, []):                       # exit i dag: realisera till exitkursen
            t, w = trades[n]["trade"], trades[n]["position_pct"] / 100
            eq_real *= 1 + w * (t.exit / t.entry - 1)
            active.discard(n)
        unreal = sum(trades[n]["position_pct"] / 100 * (last_close[n] / trades[n]["trade"].entry - 1)
                     for n in active)
        eq = eq_real * (1 + unreal)
        peak = max(peak, eq)
        max_dd = max(max_dd, 1 - eq / peak)
        curve.append((d, round((eq - 1) * 100, 2)))
        expo.append(sum(trades[n]["position_pct"] for n in active))
    return {"max_dd_pct": round(max_dd * 100, 1), "curve": curve,
            "avg_exposure_pct": round(sum(expo) / len(expo), 1) if expo else 0.0,
            "max_exposure_pct": round(max(expo), 1) if expo else 0.0}
