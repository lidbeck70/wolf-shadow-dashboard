"""
Royalty-motorn — Royalty-playbookens grindar och en rang byggd av Contrarian Alphas pelare.

Grindarna är playbookens (strategy_rules_masterguide ROYALTY): bruttomarginal > 70 %, EBIT-marginal > 40 %,
nettoskuld/EBITDA < 1,5 — plus FCF > 0 (styrkepelarens egen grind). Rangen 0–100 ordnar bolagen INOM grindarna,
som Deep Contrarian gör, med Royaltys vikter:
  hat 0,35        kurs mot SMA200, nära 52v-lägsta, under eget flerårssnitt, volymtorka, värderingsdepression
                  (contrarian_alpha.hate — samma kod)
  värde 0,30      EV/EBITDA och P/FCF mot bolagets EGEN historik (contrarian_alpha.value) — playbooken:
                  "jämför multipeln mot bolagets egen historik, aldrig mot gruvor"
  kvalitet 0,20   styrkepoängen (contrarian_alpha.strength: FCF, marginal, skuld, Altman)
  katalysator 0,15  vändning och regimfärg (contrarian_alpha.catalyst)
Deep Contrarians vakt "innan marknaden älskar dem igen": kursen högst MAX_ABOVE_SMA200_PCT över SMA200.
Verdikt: BUY CANDIDATE när grindarna och vakten håller och composite >= 65, WATCH >= 50, annars PASS — alltid
med orsaken i klartext. P/NAV och GEO (playbookens köpsignal) finns inte i Börsdata: de ligger kvar i Royalty
C-arket och noteras som manuellt steg. Rena funktioner; datan skickas in.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

WEIGHTS = {"hate": 0.35, "value": 0.30, "quality": 0.20, "catalyst": 0.15}
GATES = {"gross_margin": 70.0, "ebit_margin": 40.0, "nd_ebitda": 1.5}     # playbookens tal
MAX_ABOVE_SMA200_PCT = 10.0                                               # Deep har 5 — royalty rör sig lugnare
BUY_MIN, WATCH_MIN = 65.0, 50.0
BUY, WATCH, PASS = "BUY CANDIDATE", "WATCH", "PASS"
MANUAL_NOTE = "P/NAV-botten och GEO-tillväxt läses i Royalty C-arket (finns inte i Börsdata)"
_KPI_KEYS = ("gross_margin", "ebit_margin", "nd_ebitda", "fcf", "ev_ebitda", "p_fcf", "pb", "ps", "mcap_musd")


def stamp(today=None) -> str:
    """Körningens tidsstämpel: datetime → isoformat, sträng som den är, annars nu (UTC)."""
    if today is None:
        return datetime.now(tz=timezone.utc).isoformat()
    return today.isoformat() if hasattr(today, "isoformat") else str(today)


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def kpis_from_screen(m: Optional[dict]) -> dict:
    """Håvens nyckeltal (screens.json m) i grindarnas enheter: marginaler i %, skuld/EBITDA som kvot."""
    m = m or {}
    return {k: _f(m.get(k)) for k in _KPI_KEYS}


def kpis_from_snapshot(snap: Optional[dict]) -> dict:
    """Börsdatas snapshot (get_fundamentals_snapshot_fast: marginaler som andelar) → grindarnas enheter."""
    s = snap or {}

    def pct(key):
        v = _f(s.get(key))
        return None if v is None else round(v * 100.0, 2)

    mc = _f(s.get("market_cap"))
    return {"gross_margin": pct("gross_margin"), "ebit_margin": pct("operating_margin"),
            "nd_ebitda": _f(s.get("net_debt_ebitda")), "fcf": _f(s.get("fcf_m")), "ev_ebitda": _f(s.get("ev_ebitda")),
            "p_fcf": _f(s.get("p_fcf")), "pb": _f(s.get("pb")), "ps": _f(s.get("ps")),
            "mcap_musd": None if mc is None else mc}


def merge_kpis(primary: dict, fallback: dict) -> dict:
    """Snapshotens värden först, håvens fyller luckorna."""
    out = dict(fallback or {})
    for k, v in (primary or {}).items():
        if v is not None:
            out[k] = v
    return {k: out.get(k) for k in _KPI_KEYS}


def fund_from_kpis(kpi: dict) -> dict:
    """Minsta styrkeunderlag när Börsdatas snapshot saknas (håvens rader): FCF, EV/EBITDA."""
    out = {}
    if kpi.get("fcf") is not None:
        out["fcf"] = float(kpi["fcf"])
    if kpi.get("ev_ebitda") is not None:
        out["ev_ebitda"] = float(kpi["ev_ebitda"])
    return out


def gates(kpi: dict) -> tuple:
    """(håller, [varför inte]) — playbookens tre filter plus FCF > 0; saknat nyckeltal håller inte."""
    fails = []
    gm, em, nd, fcf = (kpi.get(k) for k in ("gross_margin", "ebit_margin", "nd_ebitda", "fcf"))
    if gm is None:
        fails.append("bruttomarginal saknas")
    elif gm <= GATES["gross_margin"]:
        fails.append(f"bruttomarginal {gm:.0f} % (kräver > {GATES['gross_margin']:.0f} %)")
    if em is None:
        fails.append("EBIT-marginal saknas")
    elif em <= GATES["ebit_margin"]:
        fails.append(f"EBIT-marginal {em:.0f} % (kräver > {GATES['ebit_margin']:.0f} %)")
    if nd is None:
        fails.append("skuld/EBITDA saknas")
    elif nd >= GATES["nd_ebitda"]:
        fails.append(f"skuld/EBITDA {nd:.1f} (kräver < {GATES['nd_ebitda']:.1f})")
    if fcf is None:
        fails.append("FCF saknas")
    elif fcf <= 0:
        fails.append("FCF negativt")
    return (not fails), fails


def price_guard(price: dict) -> tuple:
    """(håller, procent över SMA200) — Deep Contrarians vakt: ingen etablerad uppåttrend redan prissatt."""
    close, sma = _f((price or {}).get("close")), _f((price or {}).get("sma200"))
    if not close or not sma:
        return False, None
    above = (close / sma - 1.0) * 100.0
    return above <= MAX_ABOVE_SMA200_PCT, round(above, 1)


def composite(p: dict) -> float:
    return round(sum(WEIGHTS[k] * float(p.get(k) or 0.0) for k in WEIGHTS), 1)


def verdict(ok: bool, comp: float) -> str:
    if not ok:
        return PASS
    return BUY if comp >= BUY_MIN else WATCH if comp >= WATCH_MIN else PASS


def _score(res) -> float:
    try:
        return round(float(res.score), 1)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def evaluate(company: dict, df, kpi: dict, fund: Optional[dict], value_data: Optional[dict]) -> dict:
    """En rad: grindar, vakt, pelare, composite och verdikt med orsak. df = OHLCV (Yahoo), kan vara tom."""
    from contrarian_alpha.catalyst import calculate_catalyst_score, compute_regime_color
    from contrarian_alpha.engine import _compute_price_metrics
    from contrarian_alpha.hate import calculate_hate_score
    from contrarian_alpha.strength import calculate_strength_score
    from contrarian_alpha.value import calculate_value_score

    price = _compute_price_metrics(df) if df is not None and len(df) else {}
    ok_gates, fails = gates(kpi)
    ok_guard, above = price_guard(price)
    flags, notes = [], list(company.get("notes") or [])
    val = None
    if kpi.get("ev_ebitda") is not None and (value_data or {}).get("ev_ebitda_history"):
        val = {"current": kpi["ev_ebitda"], "history": value_data["ev_ebitda_history"], "metric": "ev_ebitda"}
    hate = calculate_hate_score(price, valuation_data=val) if price else None
    value = calculate_value_score(value_data) if value_data else None
    quality = calculate_strength_score(fund) if fund else None
    cat = calculate_catalyst_score(price, df=df) if price else None
    for res in (hate, value, quality):
        flags += [str(x) for x in (getattr(res, "flags", None) or [])]
    if not price:
        flags.append("INGEN_KURS")
        notes.append("ingen kurshistorik — hat och katalysator kan inte räknas")
    pillars = {"hate": _score(hate), "value": _score(value), "quality": _score(quality), "catalyst": _score(cat)}
    comp = composite(pillars)
    ok = ok_gates and ok_guard
    v = verdict(ok, comp)
    why = list(fails)
    if not ok_guard:
        why.append("SMA200 saknas" if above is None else f"{above:+.0f} % mot SMA200 (vakten: högst +{MAX_ABOVE_SMA200_PCT:.0f} %)")
    if ok and v == PASS:
        why.append(f"composite {comp:.0f} under {WATCH_MIN:.0f}")
    low = _f(price.get("low_52w")) if price else None
    close = _f(price.get("close")) if price else None
    try:
        regime = compute_regime_color(df) if df is not None and len(df) else None
    except Exception:
        regime = None
    return {"ticker": company["ticker"], "name": company.get("name") or company["ticker"],
            "kind": company.get("kind"), "theme": company.get("theme"), "source": company.get("source"),
            "ins_id": company.get("ins_id"), "market": company.get("market"), "country": company.get("country"),
            "kpi": {k: kpi.get(k) for k in _KPI_KEYS},
            "gates": {"ok": ok_gates, "fails": fails},
            "guard": {"ok": ok_guard, "above_sma200_pct": above},
            "pillars": pillars, "composite": comp, "verdict": v, "why": why,
            "price": {"close": close, "sma200": _f(price.get("sma200")) if price else None, "low_52w": low,
                      "from_low_52w_pct": (round((close / low - 1.0) * 100.0, 1) if close and low else None)},
            "regime": regime, "flags": flags, "notes": notes + [MANUAL_NOTE]}


def run(companies: list, price_getter: Callable, data_getter: Callable, today=None) -> dict:
    """Hela universumet → bloben. price_getter(ticker) → OHLCV; data_getter(company) → (kpi, fund, value_data).
    Ett bolag som fallerar hamnar i errors — de andra räknas ändå. Raderna sorteras på composite."""
    rows, errors = [], {}
    for c in companies:
        try:
            df = price_getter(c["ticker"])
        except Exception as exc:
            df, errors[c["ticker"]] = None, f"kurs: {exc}"
        try:
            kpi, fund, value_data = data_getter(c)
            rows.append(evaluate(c, df, kpi or {}, fund, value_data))
        except Exception as exc:
            errors[c["ticker"]] = f"{type(exc).__name__}: {exc}"
    rows.sort(key=lambda r: (-r["composite"], r["ticker"]))
    stats = {"n": len(rows), "buy": sum(1 for r in rows if r["verdict"] == BUY),
             "watch": sum(1 for r in rows if r["verdict"] == WATCH), "pass": sum(1 for r in rows if r["verdict"] == PASS)}
    return {"generated": stamp(today), "weights": dict(WEIGHTS), "gates": dict(GATES), "max_above_sma200_pct": MAX_ABOVE_SMA200_PCT,
            "thresholds": {"buy": BUY_MIN, "watch": WATCH_MIN}, "rows": rows, "stats": stats, "errors": errors}
