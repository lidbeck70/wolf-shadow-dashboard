"""
confidence/why_now.py — Why Now 0–100: är timingen rätt just nu? (SPEC)

Sex signaler, alla ur repots egna moduler, alla som DATA in (ingen
nätverkstrafik här — UI:t hämtar och skickar in):
  cykel        blindspot-temats 10-årspercentil → TIDIG/MITTEN/SEN/TOPP   30
  triple       rotationens Triple Signal (hat + fundamenta + katalysator)  25
  ratio        commodity_ratios-status (gummibandet)                       15
  komplex      ember-komplexets utlåtande PÅ/SELEKTIV/AV                   10
  utbud        underskott ur råvaruregistret (samma som Demand & Scarcity) 10
  time-to-money år till första kassaflöde                                  10
Signaler som saknas ger 0 och skalas pro rata; täckningen redovisas och
under 50 % flaggas. Adapters (signals_from_sources) översätter repots
datastrukturer via registrets crosswalk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from confidence import config as cfg
from confidence.commodities import Commodity
from confidence.data.provenance import describe, num
from confidence.scoring._steps import table_lt

_ROT_SUM_MIN, _ROT_SUM_MAX = 3, 15         # rotation.SUM_MIN / SUM_MAX (tre signaler 1–5)
_CYCLE_TIDIG, _CYCLE_SEN, _CYCLE_TOPP = 30.0, 70.0, 90.0   # blindspot.theme_board-trösklar


@dataclass
class Signals:
    """Allt frivilligt; None = DATA_MISSING. Text/”source” per signal för spårbarhet."""
    cycle_label: Optional[str] = None            # TIDIG | MITTEN | SEN | TOPP
    cycle_percentile: Optional[float] = None
    rotation_grade: Optional[dict] = None        # {"hatred","fundamentals","catalyst","case_intact"}
    rotation_month: str = ""
    ratio_status: Optional[str] = None           # RUBBER_BAND_STRETCHED | TENSION_BUILDING | NEUTRAL
    ratio_key: str = ""
    complex_verdict: Optional[str] = None        # PÅ | SELEKTIV | AV
    complex_key: str = ""
    time_to_money_years: Optional[float] = None
    notes: list = field(default_factory=list)


@dataclass
class WhyNow:
    score: float
    band: str
    components: dict
    notes: list
    missing: list
    coverage: float                              # andel av maxpoängen som hade signal
    flags: list = field(default_factory=list)


def cycle_label_from_percentile(pct: Optional[float], ma200w_slope_pct: Optional[float] = None) -> Optional[str]:
    """Samma regel som blindspot.theme_board (TOPP ≥ 90, SEN ≥ 70, TIDIG ≤ 30, annars MITTEN
    om 200v-lutningen är positiv, annars TIDIG)."""
    if pct is None:
        return None
    if pct >= _CYCLE_TOPP:
        return "TOPP"
    if pct >= _CYCLE_SEN:
        return "SEN"
    if pct <= _CYCLE_TIDIG:
        return "TIDIG"
    return "MITTEN" if (ma200w_slope_pct or 0) > 0 else "TIDIG"


def band_for(score: float) -> str:
    return next((lbl for floor, lbl in cfg.WHY_NOW_BANDS if score >= floor), cfg.WHY_NOW_BANDS[-1][1])


def why_now(commodity: Optional[Commodity], signals: Signals) -> WhyNow:
    sub = cfg.WHY_NOW_SUB
    comp: dict = {}
    notes: list = []
    missing: list = []
    covered = 0.0

    # cykel
    if signals.cycle_label in cfg.WHY_NOW_CYCLE:
        comp["Cykelläge"] = cfg.WHY_NOW_CYCLE[signals.cycle_label]
        covered += sub["cycle"]
        pct = f" ({signals.cycle_percentile:.0f}:e percentilen 10 år)" if signals.cycle_percentile is not None else ""
        notes.append(f"Cykelläge {comp['Cykelläge']}/{sub['cycle']} — {signals.cycle_label}{pct}, blindspot-temat")
    else:
        missing.append("cycle_label")
        notes.append(f"Cykelläge 0/{sub['cycle']} — DATA_MISSING (inget blindspot-tema för råvaran)")

    # triple signal
    g = signals.rotation_grade
    if g and all(k in g for k in ("hatred", "fundamentals", "catalyst")):
        s = int(g["hatred"]) + int(g["fundamentals"]) + int(g["catalyst"])
        if g.get("case_intact", True) is False:
            pts = 0.0
            why = "caset markerat som brutet i rotationen"
        else:
            pts = (s - _ROT_SUM_MIN) / (_ROT_SUM_MAX - _ROT_SUM_MIN) * sub["triple_signal"]
            why = f"hat {g['hatred']} + fundamenta {g['fundamentals']} + katalysator {g['catalyst']} = {s}/15"
        comp["Triple Signal"] = round(pts, 2)
        covered += sub["triple_signal"]
        notes.append(f"Triple Signal {pts:g}/{sub['triple_signal']} — {why}"
                     + (f", rotationen {signals.rotation_month}" if signals.rotation_month else ""))
    else:
        missing.append("rotation_grade")
        notes.append(f"Triple Signal 0/{sub['triple_signal']} — DATA_MISSING (råvaran är inte graderad i rotationen)")

    # ratio
    if signals.ratio_status in cfg.WHY_NOW_RATIO:
        comp["Gummiband"] = cfg.WHY_NOW_RATIO[signals.ratio_status]
        covered += sub["ratio"]
        notes.append(f"Gummiband {comp['Gummiband']}/{sub['ratio']} — {signals.ratio_status}"
                     + (f" ({signals.ratio_key})" if signals.ratio_key else ""))
    else:
        missing.append("ratio_status")
        notes.append(f"Gummiband 0/{sub['ratio']} — DATA_MISSING (ingen kvot för exponeringen)")

    # komplex
    if signals.complex_verdict in cfg.WHY_NOW_COMPLEX:
        comp["Komplex"] = cfg.WHY_NOW_COMPLEX[signals.complex_verdict]
        covered += sub["complex"]
        notes.append(f"Komplex {comp['Komplex']}/{sub['complex']} — {signals.complex_verdict}"
                     + (f" ({signals.complex_key})" if signals.complex_key else ""))
    else:
        missing.append("complex_verdict")
        notes.append(f"Komplex 0/{sub['complex']} — DATA_MISSING (ember-komplexet ej beräknat)")

    # utbud
    bal = num(commodity.supply_balance_pct) if commodity else None
    if bal is not None:
        base = table_lt(bal, cfg.SUPPLY_BALANCE_TABLE, cfg.SUPPLY_BALANCE_MAX_POINTS)
        comp["Utbud"] = round(base / cfg.SUPPLY_BALANCE_MAX_POINTS * sub["supply"], 2)
        covered += sub["supply"]
        notes.append(f"Utbud {comp['Utbud']:g}/{sub['supply']} — {describe(commodity.supply_balance_pct, 'balans')}")
    else:
        missing.append("commodity.supply_balance_pct")
        notes.append(f"Utbud 0/{sub['supply']} — DATA_MISSING")

    # time-to-money
    t = signals.time_to_money_years
    if t is not None:
        pts = next((p for lim, p in cfg.WHY_NOW_TTM_STEPS if t <= lim), cfg.WHY_NOW_TTM_BEYOND)
        comp["Time-to-money"] = pts
        covered += sub["time_to_money"]
        notes.append(f"Time-to-money {pts}/{sub['time_to_money']} — {t:g} år")
    else:
        missing.append("time_to_money_years")
        notes.append(f"Time-to-money 0/{sub['time_to_money']} — DATA_MISSING")

    total_max = sum(sub.values())
    raw = sum(comp.values())
    coverage = covered / total_max
    flags: list = []
    if coverage <= 0:
        score = 0.0
        flags.append("Why Now saknar alla signaler — 0")
    elif coverage < cfg.WHY_NOW_MIN_COVERAGE:
        score = raw                                  # för få signaler för att skala upp
        flags.append(f"Why Now bygger på {coverage * 100:.0f} % av signalerna — råpoäng utan uppskalning")
    else:
        score = raw / covered * 100
        if coverage < 1:
            notes.append(f"skalad {covered:g} → {total_max} p: signaler som saknas räknas inte (täckning {coverage * 100:.0f} %)")
            flags.append(f"täckning {coverage * 100:.0f} % — saknade signaler: {', '.join(missing)}")
    for n in signals.notes:
        notes.append(n)
    return WhyNow(round(score, 1), band_for(score), comp, notes, missing, round(coverage, 2), flags)


# ── adapters: repots datastrukturer → Signals (rena, ingen I/O) ─────────────
def signals_from_sources(commodity: Optional[Commodity], rotation_data: Optional[dict] = None,
                         theme_results: Any = None, ratio_results: Optional[dict] = None,
                         complex_results: Optional[dict] = None, exposure_to_ratio: Optional[dict] = None,
                         time_to_money_years: Optional[float] = None) -> Signals:
    """rotation_data = data/rotation.json ({"month", "grades": {key: grade}});
    theme_results = blindspot.theme_board.build_theme_board() (lista med .key/.cykel_label/.percentile_10y);
    ratio_results = alpha_regime.commodity_ratios.fetch_all_ratios() ({key: .status});
    complex_results = ember.regime.compute_all_complex_regimes() ({key: .verdict});
    exposure_to_ratio = alpha_regime.commodity_ratios.EXPOSURE_TO_RATIO (importeras om None)."""
    s = Signals(time_to_money_years=time_to_money_years)
    if commodity is None:
        s.notes.append("okänd råvara — inga signaler kan kopplas")
        return s
    if rotation_data and commodity.rotation_key:
        g = (rotation_data.get("grades") or {}).get(commodity.rotation_key)
        if g:
            s.rotation_grade = dict(g)
            s.rotation_month = str(rotation_data.get("month") or "")
    if theme_results and commodity.theme_key:
        for r in theme_results:
            if _get(r, "key") == commodity.theme_key:
                lbl = _get(r, "cykel_label")
                s.cycle_label = lbl if lbl in cfg.WHY_NOW_CYCLE else None
                s.cycle_percentile = _get(r, "percentile_10y")
                break
    if ratio_results and commodity.ratio_exposure:
        if exposure_to_ratio is None:
            try:
                from alpha_regime.commodity_ratios import EXPOSURE_TO_RATIO as exposure_to_ratio
            except Exception:                                   # yfinance saknas i miljön
                exposure_to_ratio = {}
        for key in exposure_to_ratio.get(commodity.ratio_exposure, []):
            r = ratio_results.get(key)
            if r is not None and _get(r, "status") in cfg.WHY_NOW_RATIO:
                s.ratio_status, s.ratio_key = _get(r, "status"), key
                break
    if complex_results and commodity.ember_complex:
        r = complex_results.get(commodity.ember_complex)
        if r is not None and _get(r, "verdict") in cfg.WHY_NOW_COMPLEX:
            s.complex_verdict, s.complex_key = _get(r, "verdict"), commodity.ember_complex
    return s


def _get(obj: Any, attr: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)
