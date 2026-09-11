#!/usr/bin/env python3
"""
screens_scan.py — håvarna headless.

Guidens fem Börsdata-screeners (reference.SCREENERS) körs idag för hand i
Börsdata. Det här jobbet kör dem schemalagt via KPI-screener-API:t med
EXAKT samma kriterier och sparar träffarna till Gisten (screens.json), så
larmsteget kan säga till när ett nytt bolag kvalar in och arken kan
erbjuda "lägg in i arket". Inga regler ändras — kriterietexten citeras
rad för rad nedan, tolkningarna av "råvarubransch", "~0" och "Metals &
Mining" står vid respektive konstant.

  rule     Överlevarna (Rule): råvarubransch · skuld/EBITDA < 0,5 (olja
           < 1,0) · soliditet > 50 % · EV/EBITDA < 6 · P/B < 1,5 · FCF > 0
  sprott   Optionalitet (Sprott): Kanada/Australien · Metals & Mining ·
           MCap < 200 MUSD · nettokassa · P/B < 1
  durrett  Durrett: Guld/silver · MCap 50–500 MUSD · P/S < 2 ·
           bruttomarginal > 20 % · skuld/EBITDA < 2 · omsättningstillväxt > 0
  tiggre   Tiggre (sweet spot): Kanada/Australien/USA · MCap 50–1 000 MUSD ·
           nettokassa eller byggkredit · omsättning ~0
  royalty  Royalty: Kanada/USA/Australien · bruttomarginal > 70 % ·
           EBIT-marginal > 40 % · skuld/EBITDA < 1,5

Universum: Norden (alltid) + globalt (/instruments/global, kräver Börsdata
Pro+ global). Sprott, Tiggre och Royalty har geografi utanför Norden och
ger därför bara träffar med global licens; utan den får de error satt och
larmbenet fryser sina baslinjer för just dem.

Blob: {"generated", "global_available", "screens": {key: {"label",
"criteria", "rows": [{ticker, name, ins_id, universe, country, branch_id,
currency, mcap_musd, m: {...}, notes: [...]}], "error"}}}

Env: BORSDATA_API_KEY, GITHUB_TOKEN (gist-scope). Flaggor: --dry-run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("screens_scan")

BLOB_NAME = "screens.json"

# ── Tolkningar (dokumenterade, inte dolda) ────────────────────────────────────
# "Råvarubransch" = Börsdatas branscher för olja/gas, kol, uran, gruvor och skog.
RULE_BRANCHES = frozenset({1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 21})
OIL_BRANCHES = frozenset({1, 2, 3, 4, 5})          # "olja < 1,0"
MINING_BRANCHES = frozenset({16, 17, 18})           # "Metals & Mining"
GOLD_SILVER_BRANCHES = frozenset({18})              # "Guld/silver"
REVENUE_ZERO_MAX_MUSD = 5.0                         # "omsättning ~0"

# Grova kurser till USD — trösklarna står i MUSD, börsvärdet kommer i
# kursvalutan. Räcker för intervall som 50–500 MUSD.
FX_TO_USD = {"SEK": 0.095, "NOK": 0.095, "DKK": 0.145, "EUR": 1.08, "USD": 1.0,
             "CAD": 0.73, "AUD": 0.66, "GBP": 1.27, "CHF": 1.13, "PLN": 0.25}

# KPI-id (borsdata_api.KPI)
KPI_ND_EBITDA, KPI_EQUITY_RATIO, KPI_EV_EBITDA, KPI_PB, KPI_FCF = 42, 39, 11, 4, 63
KPI_MCAP, KPI_PS, KPI_GROSS, KPI_EBIT_MARGIN, KPI_NET_DEBT, KPI_REVENUE = 50, 3, 28, 29, 60, 53
_KPIS = {"nd_ebitda": KPI_ND_EBITDA, "equity_ratio": KPI_EQUITY_RATIO,
         "ev_ebitda": KPI_EV_EBITDA, "pb": KPI_PB, "fcf": KPI_FCF,
         "mcap": KPI_MCAP, "ps": KPI_PS, "gross_margin": KPI_GROSS,
         "ebit_margin": KPI_EBIT_MARGIN, "net_debt": KPI_NET_DEBT,
         "revenue": KPI_REVENUE}

_MARKET_SUFFIX = {1: ".ST", 2: ".ST", 3: ".ST", 7: ".ST", 8: ".ST", 9: ".ST",
                  18: ".ST", 19: ".ST", 4: ".OL", 14: ".OL", 5: ".HE", 16: ".HE",
                  6: ".CO", 15: ".CO"}
_INDEX_MARKETS = {7, 8, 13, 19, 28, 31}

CA, AU, US = "canada", "australia", "usa"
_COUNTRY_HINTS = {CA: ("canada", "kanada"), AU: ("australi",),
                  US: ("usa", "united states", "förenta stater", "amerika")}


@dataclass(frozen=True)
class Screen:
    key: str
    label: str
    criteria: str
    countries: tuple           # () = ingen geografi
    check: Callable            # (m, meta) -> (failures, notes)
    sheet: str                 # var träffen hör hemma i panelen


def _f(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _lt(m, k, lim, label):
    v = _f(m.get(k))
    if v is None:
        return f"{label}: saknas"
    return None if v < lim else f"{label}: {v:g} (krav < {lim:g})"


def _gt(m, k, lim, label):
    v = _f(m.get(k))
    if v is None:
        return f"{label}: saknas"
    return None if v > lim else f"{label}: {v:g} (krav > {lim:g})"


# ── Kriterierna, rad för rad ─────────────────────────────────────────────────
def rule_check(m: dict, meta: dict) -> tuple:
    """Överlevarna: råvarubransch · skuld/EBITDA < 0,5 (olja < 1,0) ·
    soliditet > 50 % · EV/EBITDA < 6 · P/B < 1,5 · FCF > 0."""
    fails, notes = [], []
    b = meta.get("branch_id")
    if b not in RULE_BRANCHES:
        fails.append("inte råvarubransch")
    nd_lim = 1.0 if b in OIL_BRANCHES else 0.5
    nd = _f(m.get("nd_ebitda"))
    if nd is None:
        fails.append("skuld/EBITDA: saknas")
    elif nd >= nd_lim:
        fails.append(f"skuld/EBITDA: {nd:g} (krav < {nd_lim:g})")
    for r in (_gt(m, "equity_ratio", 50.0, "soliditet %"),
              _lt(m, "pb", 1.5, "P/B"), _gt(m, "fcf", 0.0, "FCF")):
        if r:
            fails.append(r)
    ev = _f(m.get("ev_ebitda"))
    if ev is None:
        fails.append("EV/EBITDA: saknas")
    elif ev >= 6.0:
        fails.append(f"EV/EBITDA: {ev:g} (krav < 6)")
    elif ev <= 0:
        # Negativ EBITDA ger negativ multipel — det är inte "billigt", det
        # är förlust. Börsdatas egen screener visar dem; här sorteras de bort
        # och sägs varför.
        fails.append(f"EV/EBITDA: {ev:g} (EBITDA ≤ 0)")
    return fails, notes


def sprott_check(m: dict, meta: dict) -> tuple:
    """Optionalitet: Metals & Mining · MCap < 200 MUSD · nettokassa · P/B < 1."""
    fails, notes = [], []
    if meta.get("branch_id") not in MINING_BRANCHES:
        fails.append("inte Metals & Mining")
    mc = _f(m.get("mcap_musd"))
    if mc is None:
        fails.append("MCap: saknas")
    elif mc >= 200.0:
        fails.append(f"MCap: {mc:,.0f} MUSD (krav < 200)")
    nd = _f(m.get("net_debt"))
    if nd is None:
        fails.append("nettoskuld: saknas")
    elif nd >= 0:
        fails.append(f"ingen nettokassa (nettoskuld {nd:,.0f} M)")
    r = _lt(m, "pb", 1.0, "P/B")
    if r:
        fails.append(r)
    return fails, notes


def durrett_check(m: dict, meta: dict) -> tuple:
    """Durrett: Guld/silver · MCap 50–500 MUSD · P/S < 2 · bruttomarginal > 20 %
    · skuld/EBITDA < 2 · omsättningstillväxt > 0."""
    fails, notes = [], []
    if meta.get("branch_id") not in GOLD_SILVER_BRANCHES:
        fails.append("inte guld/silver")
    mc = _f(m.get("mcap_musd"))
    if mc is None:
        fails.append("MCap: saknas")
    elif not (50.0 <= mc <= 500.0):
        fails.append(f"MCap: {mc:,.0f} MUSD (krav 50–500)")
    for r in (_lt(m, "ps", 2.0, "P/S"), _gt(m, "gross_margin", 20.0, "bruttomarginal %"),
              _lt(m, "nd_ebitda", 2.0, "skuld/EBITDA")):
        if r:
            fails.append(r)
    g = m.get("revenue_growth")
    if g is None:
        fails.append("omsättningstillväxt: historik saknas")
    elif g <= 0:
        fails.append(f"omsättningstillväxt: {g * 100:+.0f} % (krav > 0)")
    return fails, notes


def tiggre_check(m: dict, meta: dict) -> tuple:
    """Tiggre: Metals & Mining · MCap 50–1 000 MUSD · nettokassa eller
    byggkredit · omsättning ~0. Skuld som kan vara byggkredit fäller inte —
    den blir en notis, håven säger själv "→ manuell sållning"."""
    fails, notes = [], []
    if meta.get("branch_id") not in MINING_BRANCHES:
        fails.append("inte Metals & Mining")
    mc = _f(m.get("mcap_musd"))
    if mc is None:
        fails.append("MCap: saknas")
    elif not (50.0 <= mc <= 1000.0):
        fails.append(f"MCap: {mc:,.0f} MUSD (krav 50–1 000)")
    nd = _f(m.get("net_debt"))
    if nd is None:
        fails.append("nettoskuld: saknas")
    elif nd > 0:
        notes.append(f"nettoskuld {nd:,.0f} M — är det byggkrediten? Kolla.")
    rev = _f(m.get("revenue_musd"))
    if rev is None:
        notes.append("omsättning saknas — troligen ~0")
    elif rev > REVENUE_ZERO_MAX_MUSD:
        fails.append(f"omsättning {rev:,.0f} MUSD (krav ~0, tolkat som < {REVENUE_ZERO_MAX_MUSD:g})")
    return fails, notes


def royalty_check(m: dict, meta: dict) -> tuple:
    """Royalty: bruttomarginal > 70 % · EBIT-marginal > 40 % · skuld/EBITDA < 1,5."""
    fails = [r for r in (_gt(m, "gross_margin", 70.0, "bruttomarginal %"),
                         _gt(m, "ebit_margin", 40.0, "EBIT-marginal %"),
                         _lt(m, "nd_ebitda", 1.5, "skuld/EBITDA")) if r]
    return fails, []


SCREENS = (
    Screen("rule", "Överlevarna (Rule)",
           "Råvarubransch · skuld/EBITDA < 0,5 (olja < 1,0) · soliditet > 50 % · "
           "EV/EBITDA < 6 · P/B < 1,5 · FCF > 0", (), rule_check, "Rick Rule"),
    Screen("sprott", "Optionalitet (Sprott)",
           "Kanada/Australien · Metals & Mining · MCap < 200 MUSD · nettokassa · P/B < 1",
           (CA, AU), sprott_check, "Poängmodell · Sprott"),
    Screen("durrett", "Durrett",
           "Guld/silver · MCap 50–500 MUSD · P/S < 2 · bruttomarginal > 20 % · "
           "skuld/EBITDA < 2 · omsättningstillväxt > 0", (), durrett_check,
           "Poängmodell · Durrett"),
    Screen("tiggre", "Tiggre (sweet spot)",
           # Metals & Mining står i Tiggre-flikens egen håv-text (tiggre._screener_card)
           "Kanada/Australien/USA · MCap 50–1 000 MUSD · nettokassa eller "
           "byggkredit · omsättning ~0 → manuell FS- och tillståndssållning",
           (CA, AU, US), tiggre_check, "Tiggre"),
    Screen("royalty", "Royalty",
           "Kanada/USA/Australien · bruttomarginal > 70 % · EBIT-marginal > 40 % · "
           "skuld/EBITDA < 1,5", (CA, US, AU), royalty_check, "Royalty C"),
)
SCREEN_BY_KEY = {s.key: s for s in SCREENS}


# ── Metrik-tabell per universum ──────────────────────────────────────────────
def metrics_for(inst: dict, kpi_maps: dict) -> dict:
    """Instrumentets nyckeltal i de enheter kriterierna använder."""
    iid = inst.get("insId")
    ccy = str(inst.get("stockPriceCurrency") or "USD").upper()
    fx = FX_TO_USD.get(ccy, 1.0)
    m = {k: kpi_maps.get(k, {}).get(iid) for k in _KPIS}
    mc = _f(m.get("mcap"))
    m["mcap_musd"] = round(mc * fx, 1) if mc is not None else None
    rev = _f(m.get("revenue"))
    m["revenue_musd"] = round(rev * fx, 1) if rev is not None else None
    m["currency"] = ccy
    return m


def country_ids(countries: list) -> dict:
    """{'canada': {id,...}, 'australia': {...}, 'usa': {...}} ur /countries."""
    out = {k: set() for k in _COUNTRY_HINTS}
    for c in countries or []:
        name = str(c.get("name") or "").lower()
        for key, hints in _COUNTRY_HINTS.items():
            if any(h in name for h in hints):
                out[key].add(c.get("id"))
    return out


def revenue_growth(api, ins_id: int) -> Optional[float]:
    """Senaste årsomsättning mot föregående (KPI 53, year). None utan historik."""
    try:
        rows = api.get_kpi_history(ins_id, KPI_REVENUE, "year", "mean")
    except Exception:
        return None
    vals = sorted(((r.get("y") or 0), _f(r.get("v"))) for r in rows or []
                  if _f(r.get("v")) is not None)
    if len(vals) < 2 or not vals[-2][1]:
        return None
    return vals[-1][1] / vals[-2][1] - 1


def run_universe(api, instruments: list, universe: str, kpi_fetch: Callable,
                 ctry: dict, screens=SCREENS) -> dict:
    """Kör alla håvar på ett universum. Returnerar {key: [rader]}."""
    kpi_maps = {}
    for name, kid in _KPIS.items():
        vals = kpi_fetch(kid) or []
        kpi_maps[name] = {e.get("i"): e.get("n") for e in vals
                          if e.get("i") is not None and e.get("n") is not None}
        log.info("%s KPI %-13s %d värden", universe, name, len(kpi_maps[name]))

    out = {s.key: [] for s in screens}
    for inst in instruments:
        iid = inst.get("insId")
        if iid is None:
            continue
        meta = {"branch_id": inst.get("branchId"), "country_id": inst.get("countryId")}
        m = metrics_for(inst, kpi_maps)
        for s in screens:
            if s.countries and not any(meta["country_id"] in ctry.get(c, set())
                                       for c in s.countries):
                continue
            if s.key == "durrett":
                # dyrt (ett anrop per bolag) — bara när allt annat stämmer
                pre = dict(m, revenue_growth=1.0)
                if durrett_check(pre, meta)[0]:
                    continue
                m = dict(m, revenue_growth=revenue_growth(api, iid))
            fails, notes = s.check(m, meta)
            if fails:
                continue
            ticker = str(inst.get("ticker") or "").strip().upper()
            if universe == "nordic":
                ticker = ticker.replace(" ", "-") + _MARKET_SUFFIX.get(inst.get("marketId"), "")
            out[s.key].append({
                "ticker": ticker, "name": str(inst.get("name") or ""), "ins_id": iid,
                "universe": universe, "country_id": meta["country_id"],
                "branch_id": meta["branch_id"], "currency": m["currency"],
                "mcap_musd": m["mcap_musd"],
                "m": {k: (round(_f(m[k]), 3) if _f(m.get(k)) is not None else None)
                      for k in ("nd_ebitda", "equity_ratio", "ev_ebitda", "pb", "fcf",
                                "ps", "gross_margin", "ebit_margin", "net_debt",
                                "revenue_musd", "revenue_growth")},
                "notes": notes,
            })
    return out


_ROYALTY_NAME = __import__("re").compile(r"royalt|stream|trust", __import__("re").I)


def dedupe_rows(rows: list) -> list:
    """Störst först, och ett bolag en gång: dubbelnoteringar (Cerrado Gold
    på två börser) ger samma namn två gånger — behåll raden med störst
    börsvärde. Skalbolag på 0 MUSD kvalar in på pappret men hamnar sist."""
    rows = sorted(rows, key=lambda r: (-(r.get("mcap_musd") or 0), r.get("ticker") or ""))
    seen, out = set(), []
    for r in rows:
        key = str(r.get("name") or r.get("ticker") or "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def mark_first_seen(result: dict, prev: Optional[dict]) -> dict:
    """Sätt first_seen (datum) per rad så fliken kan visa vad som är NYTT.

    Ett bolag som fanns i förra körningen ärver sitt first_seen därifrån
    (eller förra körningens datum om fältet saknas — så att första körningen
    efter den här ändringen inte märker alla Tiggre-rader som nya). Ett
    bolag som inte fanns förra gången får dagens datum. Larmet listar samma
    tickers, så det man ser i Discord går att hitta i fliken."""
    today = str(result.get("generated") or "")[:10]
    prev_ok = isinstance(prev, dict)
    prev_screens = (prev.get("screens") or {}) if prev_ok else {}
    prev_day = str(prev.get("generated") or today)[:10] if prev_ok else today
    for key, s in (result.get("screens") or {}).items():
        prev_rows = (prev_screens.get(key) or {}).get("rows") or []
        seen = {str(r.get("ticker") or "").upper(): (r.get("first_seen") or prev_day)
                for r in prev_rows if isinstance(r, dict)}
        for r in s.get("rows") or []:
            r["first_seen"] = seen.get(str(r.get("ticker") or "").upper(), today)
        s["new"] = [r["ticker"] for r in s.get("rows") or [] if r["first_seen"] == today]
    return result


def scan(api) -> dict:
    out = {"generated": _now(), "global_available": False,
           "screens": {s.key: {"label": s.label, "criteria": s.criteria,
                               "sheet": s.sheet, "rows": [], "error": None}
                       for s in SCREENS}}
    try:
        ctry = country_ids(api.get_countries())
        nordic = [i for i in api.get_instruments()
                  if i.get("marketId") in _MARKET_SUFFIX
                  and i.get("marketId") not in _INDEX_MARKETS]
        log.info("Norden: %d instrument", len(nordic))
        res_n = run_universe(api, nordic, "nordic",
                             lambda kid: api.get_kpi_screener(kid, "last", "latest"), ctry)
        for k, rows in res_n.items():
            out["screens"][k]["rows"].extend(rows)

        glob = api.get_global_instruments_list()
        if glob:
            out["global_available"] = True
            log.info("Globalt: %d instrument", len(glob))
            res_g = run_universe(api, glob, "global",
                                 lambda kid: api.get_kpi_screener_global(kid, "last", "latest"),
                                 ctry)
            for k, rows in res_g.items():
                out["screens"][k]["rows"].extend(rows)
        else:
            log.warning("Globala instrument saknas — Börsdata Pro+ global krävs för "
                        "Sprott/Tiggre/Royalty (Kanada/Australien/USA).")
            for s in SCREENS:
                if s.countries:
                    out["screens"][s.key]["error"] = (
                        "Kräver Börsdata Pro+ global (Kanada/Australien/USA finns "
                        "inte i det nordiska universumet)")
        for k in out["screens"]:
            out["screens"][k]["rows"] = dedupe_rows(out["screens"][k]["rows"])
        for r in out["screens"]["royalty"]["rows"]:
            if _ROYALTY_NAME.search(r["name"]):
                r["notes"].append("namnet säger royalty/streaming")
    except Exception as e:
        import traceback
        log.error("Håv-skanningen felade:\n%s", traceback.format_exc())
        for k in out["screens"]:
            out["screens"][k]["error"] = str(e)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from borsdata_api import BorsdataAPI
    key = os.environ.get("BORSDATA_API_KEY", "") or os.environ.get("BD_API_KEY", "")
    if not key:
        log.error("BORSDATA_API_KEY saknas.")
        return 1
    result = scan(BorsdataAPI(api_key=key))
    try:
        from gist_storage import load_blob
        prev = load_blob(BLOB_NAME, None)
    except Exception:
        prev = None
    mark_first_seen(result, prev)
    for k, s in result["screens"].items():
        log.info("%-8s %3d träffar, %d nya%s", k, len(s["rows"]), len(s.get("new") or []),
                 f"  (FEL: {s['error']})" if s["error"] else "")
        for r in s["rows"][:12]:
            log.info("   %-14s %-26s %s  %s MUSD  %s", r["ticker"], r["name"][:26],
                     r["universe"], f"{r['mcap_musd']:,.0f}" if r["mcap_musd"] else "–",
                     "; ".join(r["notes"]))
    if args.dry_run:
        log.info("[DRY-RUN] sparar inte till Gisten.")
        return 0
    from gist_storage import save_blob
    if save_blob(BLOB_NAME, result):
        log.info("Sparat till Gisten som %s.", BLOB_NAME)
    else:
        log.error("Kunde inte spara %s till Gisten — kontrollera GITHUB_TOKEN.", BLOB_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
