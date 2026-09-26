"""
asymmetry/leverage.py — Commodity Leverage Score 0–10.

Hur kraftigt reagerar bolagets FCF (EBITDA i reserv) på råvarupriset?
Griden −30 … +50 % visar intäkt, EBITDA, FCF och marginal per steg;
poängen sätts på svaret vid +20 % enligt tabellen i config. Hög känslighet
är inte automatiskt bra: vid −20 % kontrolleras om FCF eller EBITDA blir
negativa och om kassan täcker underskottet — det ger flaggor, inte
poängavdrag, så att båda sidorna syns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from asymmetry.config import ASYMMETRY_CONFIG as CFG
from asymmetry.model import Inputs, Point, pct_change, step_table


@dataclass
class LeverageResult:
    score: Optional[int]                 # 0–10, None = DATA_MISSING
    max: int
    metric: str                          # "FCF" | "EBITDA" | ""
    response_pct: Optional[float]        # % ändring i metriken vid probe
    probe_pct: float
    grid: list                           # [Point] i prisordning
    flags: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    missing: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return "DATA_MISSING" if self.score is None else f"{self.score}/{self.max}"


def grid(inp: Inputs) -> list:
    return [inp.point(pc) for pc in CFG["price_steps_pct"]]


def _at(points: list, pct: float) -> Optional[Point]:
    return next((p for p in points if p.price_pct == pct), None)


def commodity_leverage(inp: Inputs) -> LeverageResult:
    c = CFG["commodity_leverage"]
    pts = grid(inp)
    base, up, down = _at(pts, 0.0), _at(pts, c["probe_pct"]), _at(pts, c["downside_probe_pct"])
    res = LeverageResult(None, c["max"], "", None, c["probe_pct"], pts, missing=inp.missing)
    if base is None or up is None or base.ebitda_musd is None:
        res.steps.append("DATA_MISSING: prisgriden kan inte räknas — " + "; ".join(
            s for s in (base.steps if base else []) if s.startswith("DATA_MISSING")))
        return res

    metric, resp = "", None
    if base.fcf_musd is not None and base.fcf_musd > 0:
        metric, resp = "FCF", pct_change(up.fcf_musd, base.fcf_musd)
    if resp is None and base.ebitda_musd > 0:
        metric, resp = "EBITDA", pct_change(up.ebitda_musd, base.ebitda_musd)
    if resp is None:
        res.steps.append(f"Bas-EBITDA {base.ebitda_musd:,.0f} MUSD ≤ 0 — hävstången går inte att mäta "
                         f"i procent (ett negativt tal har ingen bas). Se griden i absoluta tal.")
        res.flags.append("Negativ EBITDA vid dagens pris")
        return res
    res.metric, res.response_pct = metric, resp
    res.score = int(step_table(resp, c["table"], default=0))
    res.steps.append(f"+{c['probe_pct']:g} % råvarupris → {metric} {getattr(base, _attr(metric)):,.0f} → "
                     f"{getattr(up, _attr(metric)):,.0f} MUSD ({resp:+.0f} %)")
    res.steps.append("tabell: " + " · ".join(f"≥ {f:g} % → {p}" for f, p in c["table"]) + " · annars 0")
    res.steps.append(f"Commodity Leverage = {res.score}/{c['max']}")

    if down is not None and down.ebitda_musd is not None:
        res.steps.append(f"{c['downside_probe_pct']:+g} % råvarupris → EBITDA {down.ebitda_musd:,.0f}, "
                         f"FCF {down.fcf_musd:,.0f} MUSD")
        if down.ebitda_musd < 0:
            res.flags.append(f"Negativ EBITDA vid {c['downside_probe_pct']:+g} % pris")
        elif down.fcf_musd is not None and down.fcf_musd < 0:
            res.flags.append(f"Negativt FCF vid {c['downside_probe_pct']:+g} % pris")
        if down.fcf_musd is not None and down.fcf_musd < 0:
            need = -down.fcf_musd * c["cash_cover_years"]
            if inp.cash < need:
                res.flags.append(f"Akut finansieringsbehov: kassa {inp.cash:,.0f} < {need:,.0f} MUSD "
                                 f"({c['cash_cover_years']:g} års underskott)")
        if res.score >= 8 and res.flags:
            res.flags.insert(0, "Hög hävstång OCH hög nedsideskänslighet")
    return res


def _attr(metric: str) -> str:
    return "fcf_musd" if metric == "FCF" else "ebitda_musd"
