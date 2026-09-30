"""
asymmetry/quick_leverage.py — Commodity Leverage och break-even skattade ur
bolagets egen historik (Snabbkoll, Asymmetry 1.1 §9–12). Utanför 300.

Metod, synlig för användaren:
  EBITDA_år = a + b × råvarupris_år   (OLS, årssnitt, rapportvalutan)
  FCF_år    = a + b × råvarupris_år
  basnivå   = a + b × dagens pris
  svar      = b × (prob × dagens pris) / basnivå      → poäng 0–10 (config)
  break-even= −a / b  (priset där den skattade nivån blir noll)

FCF väger tyngst (spec §11) när sambandet håller. Ett svagt samband
(R² < LEV_MIN_YEARS/LEV_MIN_R2) poängsätts inte — resultatet styrs då av annat
(volym, kostnader, förvärv) och en siffra vore påhittad. En råvara räknas:
multi-metallbolag får huvudråvaran som proxy tills exponeringsmatrisen finns.
Rena funktioner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from asymmetry import quick_config as qc
from asymmetry.config import ASYMMETRY_CONFIG


@dataclass
class Fit:
    a: float
    b: float
    r2: float
    n: int

    def at(self, price: float) -> float:
        return self.a + self.b * price


@dataclass
class LeverageEstimate:
    commodity: str = ""
    ticker: str = ""
    unit: str = ""
    basis: str = ""                     # "FCF" | "EBITDA" | ""
    score: Optional[int] = None         # 0–10, None när det inte går att skatta
    response_pct: Optional[float] = None
    elasticity: Optional[float] = None
    r2: Optional[float] = None
    n: int = 0
    price_now: Optional[float] = None   # USD (råvarans egen enhet)
    break_even_price: Optional[float] = None
    break_even_margin_pct: Optional[float] = None
    band: str = ""
    break_even_note: str = ""
    downside: dict = field(default_factory=dict)   # {−20: nivå, −30: nivå} i rapportvalutan
    downside_basis: str = ""
    downside_label: str = ""            # STARK | MÅTTLIG | SKÖR
    flag: str = ""
    sensitivity: list = field(default_factory=list)
    fits: dict = field(default_factory=dict)       # {"EBITDA": Fit, "FCF": Fit, "Intäkt": Fit}
    error: Optional[str] = None


def fit(pairs) -> Optional[Fit]:
    """OLS y = a + b·x över [(x, y)]. None vid färre än två punkter eller konstant x."""
    pts = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
    n = len(pts)
    if n < 2:
        return None
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    syy = sum((y - my) ** 2 for _, y in pts)
    b = sxy / sxx
    r2 = (sxy * sxy / (sxx * syy)) if syy > 0 else 0.0
    return Fit(my - b * mx, b, round(r2, 3), n)


def _pairs(series: dict, prices: dict) -> list:
    return [(prices[y], v) for y, v in (series or {}).items() if y in prices and v is not None]


def _usable(f: Optional[Fit], p0: float) -> bool:
    return bool(f and f.n >= qc.LEV_MIN_YEARS and f.b > 0 and f.r2 >= qc.LEV_MIN_R2 and f.at(p0) > 0)


def _score(response_pct: float) -> int:
    for lim, pts in ASYMMETRY_CONFIG["commodity_leverage"]["table"]:
        if response_pct >= lim:
            return pts
    return 0


def fmt_price(v: Optional[float]) -> str:
    """Råvarupris läsbart: decimaler under 100 (6.63 USD/lb), tusental över (4,184 USD/oz)."""
    if v is None:
        return "—"
    return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"


def _band(margin_pct: float) -> str:
    for lim, name in qc.BREAK_EVEN_BANDS:
        if margin_pct >= lim:
            return name
    return qc.BREAK_EVEN_FAILED


def estimate(revenue: dict, ebitda: dict, fcf: dict, prices: dict, p0: Optional[float],
             fx_now: float = 1.0, commodity: str = "", ticker: str = "") -> LeverageEstimate:
    """revenue/ebitda/fcf: {år: belopp} i rapportvalutan. prices: {år: råvarans
    årssnitt} i rapportvalutan, p0 dagens pris i rapportvalutan, fx_now =
    rapportvaluta per USD (för att visa break-even i råvarans egen enhet)."""
    res = LeverageEstimate(commodity=commodity, ticker=ticker, unit=qc.LEV_PRICE_UNITS.get(ticker, ""))
    if not ticker:
        res.error = (f"ingen prisserie för {commodity or 'råvaran'} på Yahoo" if commodity
                     else "råvaran okänd — bolaget har inget tema (lägg det i arket med råvara)")
        return res
    if not p0 or not prices:
        res.error = f"prishistorik för {ticker} saknas"
        return res
    fx = fx_now or 1.0
    res.price_now = round(p0 / fx, 2)
    fits = {k: fit(_pairs(s, prices)) for k, s in (("Intäkt", revenue), ("EBITDA", ebitda), ("FCF", fcf))}
    res.fits = {k: f for k, f in fits.items() if f}
    basis = "FCF" if _usable(fits["FCF"], p0) else "EBITDA" if _usable(fits["EBITDA"], p0) else ""
    if not basis:
        best = max((f for k, f in fits.items() if f and k != "Intäkt"), key=lambda f: f.r2, default=None)
        if best is None or best.n < qc.LEV_MIN_YEARS:
            res.error = f"för få år med både resultat och {commodity}-pris ({best.n if best else 0} < {qc.LEV_MIN_YEARS})"
        elif best.b <= 0:
            res.error = f"resultatet har inte följt {commodity}-priset (negativ lutning) — styrs av annat"
        elif best.r2 < qc.LEV_MIN_R2:
            res.error = (f"för svagt samband med {commodity}-priset (R² {best.r2:.2f} < {qc.LEV_MIN_R2:g}) — "
                         f"volym, kostnader eller förvärv styr mer")
        else:
            res.error = "skattad nivå vid dagens pris ≤ 0 — hävstången kan inte mätas i procent"
        res.sensitivity = _sensitivity(fits, p0, fx)
        return res
    f = fits[basis]
    probe = ASYMMETRY_CONFIG["commodity_leverage"]["probe_pct"] / 100.0
    base = f.at(p0)
    res.basis, res.r2, res.n = basis, f.r2, f.n
    res.response_pct = round(f.b * probe * p0 / base * 100, 1)
    res.elasticity = round(res.response_pct / (probe * 100), 2)
    res.score = _score(res.response_pct)
    be = -f.a / f.b
    if be <= 0:
        # Skattad nivå positiv även vid pris 0 — resten av resultatet kommer från
        # annat än den här råvaran. Ett "100 % UTMÄRKT" vore påhittat.
        res.band = qc.BREAK_EVEN_NOT_MEASURABLE
        res.break_even_note = (f"{basis} ≈ {f.a:,.0f} M även vid {commodity}-pris 0 — resultatet vilar på mer än "
                               f"{commodity} (biprodukter, smältverk, andra metaller)")
    else:
        res.break_even_price = round(be / fx, 2)
        res.break_even_margin_pct = round((p0 - be) / p0 * 100, 1)
        res.band = _band(res.break_even_margin_pct)
    # nedsidan: FCF när det sambandet går att använda, annars basens
    d_fit, d_basis = (fits["FCF"], "FCF") if _usable(fits["FCF"], p0) else (f, basis)
    res.downside_basis = d_basis
    res.downside = {pct: round(d_fit.at(p0 * (1 + pct / 100)), 1) for pct in qc.LEV_DOWNSIDE_PCT}
    lo20, lo30 = res.downside[qc.LEV_DOWNSIDE_PCT[0]], res.downside[qc.LEV_DOWNSIDE_PCT[1]]
    res.downside_label = "STARK" if lo30 > 0 else "MÅTTLIG" if lo20 > 0 else "SKÖR"
    icon = {"STARK": "🟢", "MÅTTLIG": "🟡", "SKÖR": "🔴"}[res.downside_label]
    high = res.score >= qc.LEV_HIGH_SCORE
    res.flag = (f"{icon} {'Hög hävstång' if high else 'Hävstång'} + {res.downside_label.lower()} nedsida")
    res.sensitivity = _sensitivity(fits, p0, fx)
    return res


def _sensitivity(fits: dict, p0: float, fx: float) -> list:
    """Rader för prisstegen (spec §10): pris i råvarans enhet, skattad intäkt,
    EBITDA, FCF och marginaler i rapportvalutan. None där sambandet saknas."""
    rows = []
    for pct in ASYMMETRY_CONFIG["price_steps_pct"]:
        p = p0 * (1 + pct / 100)
        rev = fits["Intäkt"].at(p) if fits.get("Intäkt") else None
        eb = fits["EBITDA"].at(p) if fits.get("EBITDA") else None
        fc = fits["FCF"].at(p) if fits.get("FCF") else None
        rows.append({"pct": pct, "price": round(p / fx, 2),
                     "revenue": None if rev is None else round(rev, 0),
                     "ebitda": None if eb is None else round(eb, 0),
                     "fcf": None if fc is None else round(fc, 0),
                     "ebitda_margin": round(eb / rev * 100, 1) if rev and eb is not None and rev > 0 else None,
                     "fcf_margin": round(fc / rev * 100, 1) if rev and fc is not None and rev > 0 else None})
    return rows


def from_data(d: dict) -> LeverageEstimate:
    """Skattningen ur Snabbkollens datadict (quick_data.fetch)."""
    def _dict(series):
        return {int(y): float(v) for y, v in (series or []) if v is not None}
    cp = d.get("commodity_px") or {}
    return estimate(_dict(d.get("revenue_series")), _dict(d.get("ebitda_series")), _dict(d.get("fcf_series")),
                    cp.get("prices") or {}, cp.get("p0"), cp.get("fx_now") or 1.0,
                    commodity=cp.get("commodity") or "", ticker=cp.get("ticker") or "")
