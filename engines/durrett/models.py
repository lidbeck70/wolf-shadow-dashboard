"""
engines/durrett/models.py — Durrett-analysens objekt.

Score       en 0–100-poäng med drivrutiner (+/−/?) så explain_score() kan
            svara "Management 84: + CEO byggt 2 gruvor … − optionsprogram".
            value=None betyder N/A med reason — aldrig 0 för okänt.
RedFlag     {flag, severity, reason, source, data, date} (SPEC §20).
Catalyst    {name, type, expected, importance, impact, confidence, source}.
ScenarioCase Bear/Base/Bull med fair value, upside och synliga antaganden.
DurrettAnalysis  allt ovan + klassificering, typ-specifika delar, missing.

Indata är confidence.data.models.CompanyInput (fält = confidence.config.FIELDS),
så ett bolag matas in en gång för båda motorerna. Detta är "MiningCompany"
i masterpromptens §46 — repots befintliga modell återanvänds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Score:
    key: str
    label: str
    value: Optional[float]                  # 0–100, None = N/A
    positive: list = field(default_factory=list)    # "+ CEO byggt 2 gruvor (VERIFIED, …)"
    negative: list = field(default_factory=list)    # "− optionsprogram 8 % av FD"
    unknown: list = field(default_factory=list)     # "? track record ej verifierat"
    components: dict = field(default_factory=dict)  # {namn: (poäng 0–100 | None, vikt)}
    reason: str = ""                        # varför N/A
    missing: list = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.value is not None

    def explain(self) -> str:
        head = f"{self.label}: {self.value:g}" if self.value is not None else f"{self.label}: N/A ({self.reason})"
        lines = [head]
        lines += [f"  {p}" for p in self.positive]
        lines += [f"  {n}" for n in self.negative]
        lines += [f"  {u}" for u in self.unknown]
        lines.append(f"  Positive: {len(self.positive)} · Negative: {len(self.negative)} · Unknown: {len(self.unknown)}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["components"] = {k: {"score": v[0], "weight": v[1]} for k, v in self.components.items()}
        return d


@dataclass
class RedFlag:
    flag: str
    severity: str                           # LOW | MEDIUM | HIGH | CRITICAL
    reason: str
    source: str = ""
    data: str = ""
    date: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Catalyst:
    name: str
    type: str = ""                          # config.CATALYST_TYPES
    expected: str = ""                      # ÅÅÅÅ-MM eller ÅÅÅÅ-Qn
    importance: str = "MEDIUM"              # LOW | MEDIUM | HIGH
    impact: str = ""                        # vad den påverkar (resurs, värdering, risk …)
    confidence: str = "medium"              # high | medium | low
    source: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Assumption:
    name: str
    value: Any
    unit: str = ""
    kind: str = "ASSUMPTION"
    source: str = ""

    def text(self) -> str:
        v = f"{self.value:,.4g}" if isinstance(self.value, (int, float)) and not isinstance(self.value, bool) else str(self.value)
        return f"{self.name}: {v}{(' ' + self.unit) if self.unit else ''} ({self.kind}" + \
               (f", {self.source})" if self.source else ")")


@dataclass
class ScenarioCase:
    key: str                                # bear | base | bull
    label: str
    commodity_price: Optional[float] = None
    production: Optional[float] = None
    aisc: Optional[float] = None
    capex_musd: Optional[float] = None
    multiple: Optional[float] = None
    operating_cf_musd: Optional[float] = None
    future_fcf_musd: Optional[float] = None
    future_ev_musd: Optional[float] = None
    future_mcap_musd: Optional[float] = None
    fair_value_per_share: Optional[float] = None
    upside_pct: Optional[float] = None
    upside_multiple: Optional[float] = None
    assumptions: list = field(default_factory=list)   # [Assumption]
    steps: list = field(default_factory=list)
    reason: str = ""                        # varför N/A

    @property
    def available(self) -> bool:
        return self.upside_pct is not None


@dataclass
class Classification:
    company_type: str
    reasons: list = field(default_factory=list)
    confidence: str = "medium"              # high | medium | low
    profile: str = ""                       # vilken analysprofil som körs


@dataclass
class DurrettAnalysis:
    ticker: str
    name: str
    commodity: str
    classification: Classification
    # tio steg + typ-specifikt
    properties_score: Score
    management_score: Score
    dilution_score: Score
    jurisdiction_score: Score
    growth_score: Score
    momentum_score: Score
    cost_score: Score
    financing_score: Score
    balance_sheet_score: Score
    valuation_score: Score
    upside_score: Score
    red_flag_score: Score
    # sammanvägt
    quality_score: Score
    risk_score: Score
    risk_level: str
    confidence_score: Score                 # Model Confidence (data + robusthet)
    data_confidence: Optional[float]        # ur confidence-motorn
    # värdering
    market_cap_musd: Optional[float] = None
    fd_market_cap_musd: Optional[float] = None
    enterprise_value_musd: Optional[float] = None
    fd_enterprise_value_musd: Optional[float] = None
    fair_value_per_share: Optional[float] = None       # Base
    potential_upside_pct: Optional[float] = None       # Base
    upside_multiple: Optional[float] = None
    bear_case: Optional[ScenarioCase] = None
    base_case: Optional[ScenarioCase] = None
    bull_case: Optional[ScenarioCase] = None
    red_flags: list = field(default_factory=list)
    catalysts: list = field(default_factory=list)
    missing_data: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)        # beräknade nyckeltal {namn: {value, unit, note}}
    developer_checklist: Optional[dict] = None         # {items: [(namn, ok|None, why)], passed, total}
    explorer_profile: Optional[dict] = None            # {play, optionality_score, lassonde, …}
    thesis: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    generated: str = ""

    def scores(self) -> list:
        return [self.properties_score, self.management_score, self.dilution_score, self.jurisdiction_score,
                self.growth_score, self.momentum_score, self.cost_score, self.financing_score,
                self.balance_sheet_score, self.valuation_score, self.upside_score, self.red_flag_score]

    def score(self, key: str) -> Optional[Score]:
        for s in self.scores() + [self.quality_score, self.risk_score, self.confidence_score]:
            if s.key == key:
                return s
        return None

    def explain_score(self, key: str) -> str:
        s = self.score(key)
        return s.explain() if s else f"{key}: okänd poäng"


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "as_dict"):
        return to_jsonable(obj.as_dict())
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
