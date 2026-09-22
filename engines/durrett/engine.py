"""
engines/durrett/engine.py — kör hela Durrett-analysen i ordning och
returnerar DurrettAnalysis + EngineResult (engines.contract).

Ordning: klassificering → aktiestruktur (MCap/EV/FD) → resurser →
tio steg → typ-specifik profil → scenarier → red flags → Quality/Risk →
Confidence → thesis. Loggen (SPEC §49) byggs sist. Inget nätverk.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from confidence.data.models import CompanyInput
from engines.contract import EngineResult
from engines.durrett import (balance_sheet, classifier, config as dc, costs, developer, dilution, explorer,
                             financing, growth, jurisdiction, management, momentum, properties, red_flags,
                             scenarios, thesis, upside, valuation)
from engines.durrett import confidence as dconf
from engines.durrett._base import Ctx
from engines.durrett.models import DurrettAnalysis, Score, to_jsonable


def _weighted(key: str, label: str, scores: dict, weights: dict, min_cov: float = 0.5) -> Score:
    s = Score(key, label, None)
    acc, known, total = 0.0, 0.0, 0.0
    for k, w in weights.items():
        if w <= 0:
            continue
        total += w
        sc = scores.get(k)
        s.components[k] = (sc.value if sc else None, w)
        if sc is None or sc.value is None:
            s.unknown.append(f"? {k}: N/A" + (f" ({sc.reason})" if sc and sc.reason else ""))
            continue
        known += w
        acc += sc.value * w
        (s.positive if sc.value >= 50 else s.negative).append(f"{'+' if sc.value >= 50 else '−'} {k} {sc.value:g} × {w}")
    if total <= 0 or known / total < min_cov:
        s.reason = f"för lite känt ({known / total * 100:.0f} % av vikten)" if total else "inga vikter"
    else:
        s.value = round(acc / known, 1)
        if known < total:
            s.unknown.append(f"vikter omfördelade över {known / total * 100:.0f} % känd vikt")
    return s


def risk_level(risk_score: Optional[float], cfg: dict) -> str:
    if risk_score is None:
        return "UNKNOWN"
    return next(lbl for floor, lbl in cfg["risk_levels"] if risk_score >= floor)


def analyze(company: CompanyInput, config: Optional[dict] = None, scenario_overrides: Optional[dict] = None,
            today: Optional[date] = None) -> DurrettAnalysis:
    today = today or date.today()
    ctx = Ctx(company, config)
    cls = classifier.classify(ctx)
    ctype = cls.company_type
    ss = dilution.share_structure(ctx)
    res = properties.resources(ctx)
    pm = properties.valuation_metrics(ctx, ss, res)
    fe = valuation.future_earnings(ctx, ss)
    up = upside.compute(ctx, ss, fe)

    scores = {
        "properties": properties.score(ctx, pm, res),
        "management": management.score(ctx, ss),
        "dilution": dilution.score(ctx, ss),
        "jurisdiction": jurisdiction.score(ctx),
        "growth": growth.score(ctx, res, today),
        "momentum": momentum.score(ctx),
        "costs": costs.score(ctx, ctype, ss),
        "financing": financing.score(ctx, ctype, ss),
        "balance_sheet": balance_sheet.score(ctx, ctype, ss, today),
        "valuation": valuation.score(ctx, ctype, ss, pm, fe),
        "upside": upside.score(ctx, up),
    }
    cases = scenarios.run_all(ctx, ss, fe, scenario_overrides)
    flags = red_flags.detect(ctx, ctype, ss, res, pm, scores, cases, today)
    scores["red_flags"] = red_flags.score(flags)

    weights = dict(ctx.cfg["weights"])
    if ctype == dc.ROYALTY:
        weights["costs"] = 0                                 # royalty: kostnadsstrukturen gäller inte
    if ctx.cfg.get("momentum_in_quality", 0):
        weights["momentum"] = ctx.cfg["momentum_in_quality"]
    quality = _weighted("quality", "Durrett Quality", scores, weights)
    risk = _weighted("risk", "Risk", scores, ctx.cfg["risk_weights"], min_cov=0.4)
    rf = scores["red_flags"].value
    if risk.value is not None and rf is not None:
        risk.value = round(min(risk.value, (risk.value + rf) / 2) if rf < risk.value else risk.value, 1)
        risk.negative.append(f"− red flag-poäng {rf:g} drar ner") if rf < 50 else None
    rob = scenarios.robustness(cases)
    conf, data_conf, _cf = dconf.compute(ctx, list(scores.values()), rob, today)

    dev = developer.checklist(ctx, scores, up, today) if ctype in (dc.DEVELOPER, dc.UNKNOWN) else None
    exp = explorer.profile(ctx, ss, res) if ctype == dc.EXPLORER else None

    a = DurrettAnalysis(
        ticker=company.ticker, name=company.name, commodity=company.commodity, classification=cls,
        properties_score=scores["properties"], management_score=scores["management"],
        dilution_score=scores["dilution"], jurisdiction_score=scores["jurisdiction"],
        growth_score=scores["growth"], momentum_score=scores["momentum"], cost_score=scores["costs"],
        financing_score=scores["financing"], balance_sheet_score=scores["balance_sheet"],
        valuation_score=scores["valuation"], upside_score=scores["upside"], red_flag_score=scores["red_flags"],
        quality_score=quality, risk_score=risk, risk_level=risk_level(risk.value, ctx.cfg),
        confidence_score=conf, data_confidence=data_conf,
        market_cap_musd=ss["mcap_usd"], fd_market_cap_musd=ss["fd_mcap_usd"],
        enterprise_value_musd=ss["ev_usd"], fd_enterprise_value_musd=ss["fd_ev_usd"],
        fair_value_per_share=cases["base"].fair_value_per_share, potential_upside_pct=cases["base"].upside_pct,
        upside_multiple=up["upside_multiple"],
        bear_case=cases["bear"], base_case=cases["base"], bull_case=cases["bull"],
        red_flags=flags, catalysts=thesis.catalysts(company), missing_data=list(dict.fromkeys(ctx.missing)),
        metrics=ctx.metrics, developer_checklist=dev, explorer_profile=exp, generated=today.isoformat(),
    )
    a.thesis = thesis.build(a)
    a.log = _log(a, ctx)
    return a


def _log(a: DurrettAnalysis, ctx: Ctx) -> list:
    def f(v):
        return "N/A" if v is None else f"{v:g}"
    lines = ["DURRETT ENGINE", f"Ticker: {a.ticker}", f"Classification: {a.classification.company_type} "
             f"({a.classification.confidence})"]
    for s in a.scores():
        lines.append(f"{s.label}: {f(s.value)}" + (f" — {s.reason}" if s.value is None and s.reason else ""))
    lines += [f"Quality: {f(a.quality_score.value)}", f"Risk: {f(a.risk_score.value)} ({a.risk_level})",
              f"Upside: {f(a.upside_multiple)}x", f"Confidence: {f(a.confidence_score.value)}% "
              f"(data {f(a.data_confidence)})", f"Red Flags: {len(a.red_flags)}",
              f"Missing: {len(a.missing_data)}"]
    lines += [f"log: {x}" for x in ctx.log]
    return lines


def to_engine_result(a: DurrettAnalysis) -> EngineResult:
    return EngineResult(
        engine="durrett", ticker=a.ticker,
        quality_score=a.quality_score.value, risk_score=a.risk_score.value,
        valuation_score=a.valuation_score.value, upside_score=a.upside_score.value,
        confidence_score=a.confidence_score.value,
        red_flags=[x.as_dict() for x in a.red_flags], catalysts=[c.as_dict() for c in a.catalysts],
        missing_data=list(a.missing_data),
        explanations={s.key: s.as_dict() for s in a.scores() + [a.quality_score, a.risk_score, a.confidence_score]},
        scenarios={k: to_jsonable(c) for k, c in (("bear", a.bear_case), ("base", a.base_case), ("bull", a.bull_case)) if c},
        extras={"classification": to_jsonable(a.classification), "risk_level": a.risk_level,
                "data_confidence": a.data_confidence, "upside_multiple": a.upside_multiple,
                "market_cap_musd": a.market_cap_musd, "fd_market_cap_musd": a.fd_market_cap_musd,
                "enterprise_value_musd": a.enterprise_value_musd, "metrics": a.metrics,
                "developer_checklist": to_jsonable(a.developer_checklist), "explorer_profile": to_jsonable(a.explorer_profile),
                "thesis": a.thesis},
        generated=a.generated,
    )


def run(company: CompanyInput, **kw) -> EngineResult:
    return to_engine_result(analyze(company, **kw))
