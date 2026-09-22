"""
engines/durrett/classifier.py — PRODUCER / DEVELOPER / EXPLORER /
ROYALTY_STREAMER / HYBRID / UNKNOWN (SPEC §4), med skäl.

Regler (i ordning):
  stage royalty, eller royaltyintäkt ≥ 40 % utan egen produktion → ROYALTY_STREAMER
  royaltyintäkt ≥ 40 % OCH egen produktion                         → HYBRID
  omsättning ≥ tröskel eller stage producer/produktion nu          → PRODUCER
  PEA eller senare (maturity/lassonde) utan produktion             → DEVELOPER
  exploration/discovery/resource_definition                        → EXPLORER
  inget av detta känt                                              → UNKNOWN
Bolagets eget stage-fält räknas som ett skäl, inte som facit.
"""

from __future__ import annotations

from confidence import config as ccfg
from engines.durrett import config as dc
from engines.durrett._base import Ctx
from engines.durrett.models import Classification

_DEV_STAGES = ("pea", "pfs", "dfs", "fid", "construction")
_DEV_LASSONDE = ("pea", "pfs", "fs", "construction")
_EXP_LASSONDE = ("exploration", "discovery", "resource_definition")


def classify(ctx: Ctx) -> Classification:
    c = ctx.c
    cfg = ctx.cfg["classification"]
    reasons: list = []
    revenue = ctx.num("revenue_musd", required=False)
    prod_now = ctx.num("production_current", required=False)
    royalty_share = ctx.num("royalty_revenue_share_pct", required=False)
    lassonde = ctx.text("lassonde_stage")
    stage, maturity = c.stage, c.maturity

    producing = (revenue is not None and revenue >= cfg["producer_min_revenue_musd"]) or \
        (prod_now is not None and prod_now > 0) or maturity == "production"
    if revenue is not None:
        reasons.append(f"omsättning {revenue:g} MUSD ({'≥' if revenue >= cfg['producer_min_revenue_musd'] else '<'} "
                       f"{cfg['producer_min_revenue_musd']:g})")
    if prod_now is not None:
        reasons.append(f"produktion nu {prod_now:g}")

    if stage == "royalty" or (royalty_share is not None and royalty_share >= cfg["hybrid_royalty_share_pct"]
                              and not producing):
        reasons.append("royalty/stream-modell" + (f" ({royalty_share:g} % av intäkten)" if royalty_share is not None else ""))
        return Classification(dc.ROYALTY, reasons, "high" if stage == "royalty" else "medium", "royalty")
    if royalty_share is not None and royalty_share >= cfg["hybrid_royalty_share_pct"] and producing:
        reasons.append(f"royaltyintäkt {royalty_share:g} % men även egen produktion")
        return Classification(dc.HYBRID, reasons, "medium", "producer")
    if producing:
        reasons.append(f"stage-fält: {stage} · mognad: {ccfg.MATURITY_LABEL.get(maturity, maturity)}")
        return Classification(dc.PRODUCER, reasons, "high" if revenue is not None else "medium", "producer")

    advanced = maturity in _DEV_STAGES or lassonde in _DEV_LASSONDE
    early = maturity == "exploration" or lassonde in _EXP_LASSONDE
    if lassonde:
        reasons.append(f"Lassonde: {dc.LASSONDE_LABEL.get(lassonde, lassonde)}")
    reasons.append(f"mognad: {ccfg.MATURITY_LABEL.get(maturity, maturity)} · stage-fält: {stage}")
    if stage == "producer":
        reasons.append("stage-fältet säger producer men varken omsättning eller produktion är angiven")
        return Classification(dc.UNKNOWN, reasons, "low", "producer")
    if advanced and not (early and lassonde in _EXP_LASSONDE):
        return Classification(dc.DEVELOPER, reasons, "high" if (lassonde or c.has("npv_musd")) else "medium",
                              "developer")
    if early or stage == "explorer":
        return Classification(dc.EXPLORER, reasons, "high" if lassonde else "medium", "explorer")
    if stage == "developer":
        reasons.append("bara stage-fältet talar för developer — ange mognad/Lassonde eller NPV")
        return Classification(dc.DEVELOPER, reasons, "low", "developer")
    reasons.append("ingen omsättning, produktion, mognad eller Lassonde-position känd")
    return Classification(dc.UNKNOWN, reasons, "low", "developer")
