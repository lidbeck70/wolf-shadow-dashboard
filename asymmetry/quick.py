"""
asymmetry/quick.py — Snabbkollens poäng: Survival, Margin of Safety och
Confidence (auto), fem kort var. Rena funktioner på en platt datadict ur
asymmetry/quick_data.fetch — ingen nätverkstrafik här.

Kort: GRÖN 100 · GUL 50 · RÖD 0 · DATA_GAP räknas inte. Grupp = snittet av
mätta kort. Totalverdikt: alla tre ≥ 65 → GRÖN, någon < 40 → RÖD, annars GUL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Optional

from asymmetry import quick_config as qc


@dataclass
class Pillar:
    key: str
    label: str
    status: str                 # GREEN | AMBER | RED | DATA_GAP
    value: str                  # det faktiska värdet, läsbart
    why: str                    # en rad förklaring med tröskeln

    @property
    def points(self) -> Optional[float]:
        return qc.STATUS_POINTS.get(self.status)


@dataclass
class Group:
    key: str
    label: str
    pillars: list = field(default_factory=list)

    @property
    def measured(self) -> list:
        return [p for p in self.pillars if p.points is not None]

    @property
    def score(self) -> Optional[float]:
        m = self.measured
        if len(m) < qc.MIN_MEASURED:
            return None
        return round(sum(p.points for p in m) / len(m), 1)

    @property
    def coverage(self) -> str:
        return f"{len(self.measured)}/{len(self.pillars)} mätta"


@dataclass
class QuickResult:
    ticker: str
    name: str
    source: str
    groups: list
    verdict: str                # GRÖN | GUL | RÖD | DATA_GAP
    reasons: list

    def group(self, key: str) -> Optional[Group]:
        return next((g for g in self.groups if g.key == key), None)

    @property
    def total(self) -> Optional[float]:
        s = [g.score for g in self.groups if g.score is not None]
        return round(sum(s), 0) if len(s) == len(self.groups) else None


# ── hjälpare ─────────────────────────────────────────────────────────────────
def _n(v) -> Optional[float]:
    try:
        return None if v is None or v != v else float(v)
    except (TypeError, ValueError):
        return None


def _higher(v: float, table: tuple) -> str:
    g, a = table
    return "GREEN" if v >= g else "AMBER" if v >= a else "RED"


def _lower(v: float, table: tuple) -> str:
    g, a = table
    return "GREEN" if v <= g else "AMBER" if v <= a else "RED"


def _gap(key: str, label: str, what: str) -> Pillar:
    return Pillar(key, label, "DATA_GAP", "—", f"DATA_GAP: {what}")


def _vs_median(cur: Optional[float], hist: list) -> tuple:
    """(% mot median, median, antal år) — None när historiken är för kort."""
    vals = [x for x in (_n(v) for v in (hist or [])) if x is not None and 0 < x < 100]
    if cur is None or cur <= 0 or len(vals) < qc.MIN_HISTORY:
        return None, None, len(vals)
    med = median(vals)
    return round((cur / med - 1) * 100, 1), round(med, 1), len(vals)


# ── Survival ────────────────────────────────────────────────────────────────
def survival(d: dict) -> Group:
    g = Group("survival", "Survival")
    fcf, cash = _n(d.get("fcf")), _n(d.get("cash"))
    if fcf is None:
        g.pillars.append(_gap("cashflow", "Kassaflöde", "fritt kassaflöde saknas"))
    elif fcf >= 0:
        g.pillars.append(Pillar("cashflow", "Kassaflöde", "GREEN", f"FCF +{fcf:,.0f} M",
                                "Positivt fritt kassaflöde — ingen runway behövs"))
    elif cash is None:
        g.pillars.append(_gap("cashflow", "Kassaflöde", f"FCF {fcf:,.0f} M men kassan saknas"))
    else:
        years = cash / -fcf if fcf else 99.0
        g.pillars.append(Pillar("cashflow", "Kassaflöde", _higher(years, qc.RUNWAY_YEARS),
                                f"{years:.1f} års runway",
                                f"Kassa {cash:,.0f} M / underskott {-fcf:,.0f} M per år · "
                                f"≥ {qc.RUNWAY_YEARS[0]:g} år grönt, ≥ {qc.RUNWAY_YEARS[1]:g} gult"))
    nd, nd_debt = _n(d.get("nd_ebitda")), _n(d.get("net_debt"))
    if nd_debt is not None and nd_debt <= 0:
        g.pillars.append(Pillar("debt", "Skuld", "GREEN", "Nettokassa", f"Nettoskuld {nd_debt:,.0f} M ≤ 0"))
    elif nd is None:
        g.pillars.append(_gap("debt", "Skuld", "nettoskuld/EBITDA saknas (eller EBITDA ≤ 0)"))
    else:
        g.pillars.append(Pillar("debt", "Skuld", "GREEN" if nd <= 0 else _lower(nd, qc.ND_EBITDA),
                                f"{nd:.1f}× ND/EBITDA",
                                f"≤ {qc.ND_EBITDA[0]:g}× grönt, ≤ {qc.ND_EBITDA[1]:g}× gult, annars rött"))
    cr = _n(d.get("current_ratio"))
    g.pillars.append(_gap("liquidity", "Likviditet", "current ratio saknas") if cr is None else
                     Pillar("liquidity", "Likviditet", _higher(cr, qc.CURRENT_RATIO), f"{cr:.2f} current ratio",
                            f"Omsättningstillgångar / korta skulder · ≥ {qc.CURRENT_RATIO[0]:g} grönt, "
                            f"≥ {qc.CURRENT_RATIO[1]:g} gult"))
    dil = _n(d.get("shares_growth_3y_pct"))
    g.pillars.append(_gap("dilution", "Utspädning", "aktieantal tre år bakåt saknas") if dil is None else
                     Pillar("dilution", "Utspädning", _lower(dil, qc.DILUTION_3Y_PCT), f"{dil:+.0f} % på 3 år",
                            f"Antal aktier · ≤ +{qc.DILUTION_3Y_PCT[0]:g} % grönt, ≤ +{qc.DILUTION_3Y_PCT[1]:g} % gult"))
    er = _n(d.get("equity_ratio_pct"))
    g.pillars.append(_gap("equity", "Soliditet", "soliditet saknas") if er is None else
                     Pillar("equity", "Soliditet", _higher(er, qc.EQUITY_RATIO_PCT), f"{er:.0f} %",
                            f"Eget kapital / tillgångar · ≥ {qc.EQUITY_RATIO_PCT[0]:g} % grönt, "
                            f"≥ {qc.EQUITY_RATIO_PCT[1]:g} % gult"))
    return g


# ── Margin of Safety ────────────────────────────────────────────────────────
def margin_of_safety(d: dict) -> Group:
    g = Group("mos", "Margin of Safety")
    em = _n(d.get("ebitda_margin_pct"))
    if em is None:
        g.pillars.append(_gap("buffer", "Prisbuffert", "EBITDA-marginal saknas"))
    else:
        g.pillars.append(Pillar("buffer", "Prisbuffert", "RED" if em <= 0 else _higher(em, qc.PRICE_BUFFER_PCT),
                                f"{em:.0f} % EBITDA-marginal",
                                f"Priset kan falla ≈ {max(em, 0):.0f} % innan EBITDA blir noll (fasta kostnader) · "
                                f"≥ {qc.PRICE_BUFFER_PCT[0]:g} % grönt, ≥ {qc.PRICE_BUFFER_PCT[1]:g} % gult"))
    pct, med, n = _vs_median(_n(d.get("ev_ebitda")), d.get("ev_ebitda_hist"))
    g.pillars.append(_gap("ev_ebitda", "EV/EBITDA mot historik",
                          f"för kort historik ({n} år) eller negativ EBITDA") if pct is None else
                     Pillar("ev_ebitda", "EV/EBITDA mot historik", _lower(pct, qc.EV_EBITDA_VS_MEDIAN_PCT),
                            f"{d.get('ev_ebitda'):.1f}× ({pct:+.0f} %)",
                            f"Egen median {med:g}× över {n} år · ≥ 25 % under grönt, inom ±25 % gult"))
    pct, med, n = _vs_median(_n(d.get("p_fcf")), d.get("p_fcf_hist"))
    if pct is not None:
        g.pillars.append(Pillar("p_fcf", "P/FCF mot historik", _lower(pct, qc.P_FCF_VS_MEDIAN_PCT),
                                f"{d.get('p_fcf'):.1f}× ({pct:+.0f} %)",
                                f"Egen median {med:g}× över {n} år · ≥ 25 % under grönt, inom ±25 % gult"))
    else:
        fy = _n(d.get("fcf_yield_pct"))
        g.pillars.append(_gap("p_fcf", "FCF-yield", "P/FCF-historik och FCF-yield saknas") if fy is None else
                         Pillar("p_fcf", "FCF-yield", "RED" if fy <= 0 else _higher(fy, qc.FCF_YIELD_PCT),
                                f"{fy:.1f} %", f"Reserv när P/FCF-historik saknas · ≥ {qc.FCF_YIELD_PCT[0]:g} % grönt, "
                                f"≥ {qc.FCF_YIELD_PCT[1]:g} % gult"))
    dd = _n(d.get("from_52w_high_pct"))
    g.pillars.append(_gap("drawdown", "Från 52v-högsta", "kurshistorik saknas") if dd is None else
                     Pillar("drawdown", "Från 52v-högsta", _higher(dd, qc.FROM_52W_HIGH_PCT), f"−{dd:.0f} %",
                            f"≥ {qc.FROM_52W_HIGH_PCT[0]:g} % under toppen grönt, ≥ {qc.FROM_52W_HIGH_PCT[1]:g} % gult"))
    sm = _n(d.get("vs_sma200_pct"))
    g.pillars.append(_gap("sma200", "Mot 200-dagars snitt", "kurshistorik < 200 dagar") if sm is None else
                     Pillar("sma200", "Mot 200-dagars snitt", _lower(sm, qc.VS_SMA200_PCT), f"{sm:+.0f} %",
                            f"Under snittet = hatat = marginal · ≤ {qc.VS_SMA200_PCT[0]:g} % grönt, "
                            f"≤ +{qc.VS_SMA200_PCT[1]:g} % gult"))
    return g


# ── Confidence (auto) ───────────────────────────────────────────────────────
def _stability(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


def confidence(d: dict) -> Group:
    g = Group("confidence", "Confidence (auto)")
    have, total = d.get("coverage", (0, 0))
    cov = have / total * 100 if total else None
    g.pillars.append(_gap("coverage", "Datatäckning", "inga nyckeltal") if cov is None else
                     Pillar("coverage", "Datatäckning", _higher(cov, qc.COVERAGE_PCT), f"{have}/{total} nyckeltal",
                            f"Källa {d.get('source') or '—'} · ≥ {qc.COVERAGE_PCT[0]:g} % grönt, "
                            f"≥ {qc.COVERAGE_PCT[1]:g} % gult"))
    for key, label in (("earnings_stability", "Resultatstabilitet"), ("fcf_stability", "FCF-stabilitet")):
        v = _stability(_n(d.get(key)))
        g.pillars.append(_gap(key, label, f"{label.lower()} saknas (Börsdata)") if v is None else
                         Pillar(key, label, _higher(v, qc.STABILITY), f"{v:.2f}",
                                f"Börsdata 0–1 · ≥ {qc.STABILITY[0]:g} grönt, ≥ {qc.STABILITY[1]:g} gult"))
    fs = _n(d.get("f_score"))
    g.pillars.append(_gap("f_score", "Piotroski F-score", "F-score saknas") if fs is None else
                     Pillar("f_score", "Piotroski F-score", _higher(fs, qc.F_SCORE), f"{fs:.0f}/9",
                            f"≥ {qc.F_SCORE[0]:g} grönt, ≥ {qc.F_SCORE[1]:g} gult"))
    yrs = d.get("report_years")
    gap = _n(d.get("source_gap_pct"))
    if yrs is None and gap is None:
        g.pillars.append(_gap("history", "Historik & källor", "inga rapporter och ingen jämförelse"))
    else:
        st = _higher(float(yrs or 0), qc.REPORT_YEARS) if yrs is not None else "AMBER"
        if gap is not None and abs(gap) > qc.SOURCE_GAP_PCT and st == "GREEN":
            st = "AMBER"
        g.pillars.append(Pillar("history", "Historik & källor", st,
                                (f"{yrs} år" if yrs is not None else "—")
                                + (f" · källor {gap:+.0f} %" if gap is not None else ""),
                                f"Årsrapporter ≥ {qc.REPORT_YEARS[0]} grönt, ≥ {qc.REPORT_YEARS[1]} gult · "
                                f"börsvärde Börsdata mot Yahoo inom ±{qc.SOURCE_GAP_PCT:g} %"))
    return g


# ── Totalt ───────────────────────────────────────────────────────────────────
def score(d: dict) -> QuickResult:
    groups = [survival(d), margin_of_safety(d), confidence(d)]
    scores = {g.label: g.score for g in groups}
    reasons = []
    if any(v is None for v in scores.values()):
        verdict = "DATA_GAP"
        reasons.append("För lite data i " + ", ".join(k for k, v in scores.items() if v is None))
    elif all(v >= qc.VERDICT_GREEN_MIN for v in scores.values()):
        verdict = "GRÖN"
        reasons.append(f"Alla tre ≥ {qc.VERDICT_GREEN_MIN:g}")
    elif any(v < qc.VERDICT_RED_BELOW for v in scores.values()):
        verdict = "RÖD"
        reasons.append("Under " + f"{qc.VERDICT_RED_BELOW:g}: " +
                       ", ".join(f"{k} {v:g}" for k, v in scores.items() if v < qc.VERDICT_RED_BELOW))
    else:
        verdict = "GUL"
        reasons.append("Blandat: " + ", ".join(f"{k} {v:g}" for k, v in scores.items()))
    reds = [f"{p.label} ({p.value})" for g in groups for p in g.pillars if p.status == "RED"]
    if reds:
        reasons.append("Röda kort: " + ", ".join(reds[:4]))
    return QuickResult(str(d.get("ticker") or ""), str(d.get("name") or ""), str(d.get("source") or ""),
                       groups, verdict, reasons)
