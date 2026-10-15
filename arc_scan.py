#!/usr/bin/env python3
"""
arc_scan.py — headless körning av Arc-screenerna Wolf och Viking.

Körs av scheduled-scan-workflowen (08/12/18 vardagar) före larmsteget.
Flikarna kör dessa scanningar live när man trycker SCAN — men larm kräver
att någon räknar även när panelen är stängd, med FASTA inställningar:

  Wolf    wolf_shadow_screener.run_screener på de nordiska marknaderna
          (Stockholm/Oslo/Köpenhamn/Helsingfors). Alla rader med
          Total Score >= GOLV sparas — larmribban (default 80) ligger i
          alert_rules/inställningarna, så en ribbändring inte kräver omscan.
  Viking  screener_ovtlyr.run_ovtlyr_screener("Nordic"). Rader med
          Vikings Nine >= 6 eller absolut eligibility sparas.

Resultatet skrivs till Gisten som arc_screeners.json:
  {"wolf": {"generated", "rows": [{"ticker","name","score"}], "error"},
   "viking": {"generated", "rows": [{"ticker","name","v9","eligible",
                                     "signal","composite"}], "error"}}

En screener som felar ger error-fältet satt och rows=[] — larmbenet ser då
ingen läsbar data (alert_scan skickar None) och behåller sin baslinje.

Env: BORSDATA_API_KEY (prisbatch för Wolf; Viking klarar sig på yfinance),
     GITHUB_TOKEN (gist-scope, för save_blob).
Flaggor: --dry-run (räkna, skriv inget till Gisten).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("arc_scan")

BLOB_NAME = "arc_screeners.json"
WOLF_MARKETS = ["stockholm", "oslo", "copenhagen", "helsinki"]
WOLF_REGISTRY_KEY = "registry"   # registrets nordiska huvudlistor som extra marknad i Wolf-screenern (som fliken)
WOLF_EXTRA_MAX = 250             # tak utöver de fyra listorna — Wolf-poängen räknas bolag för bolag
WOLF_SAVE_FLOOR = 50.0    # spara allt fliken kallar MODERATE+ — ribban sätts vid larmet
VIKING_SAVE_FLOOR = 6     # spara V9 >= 6 samt allt absolut-eligible
_BOARD_ORDER = {1: 0, 9: 0, 14: 0, 20: 0, 2: 1, 3: 2}   # Large Cap och huvudbörserna först, sedan Mid, Small


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def registry_wolf(loader=None):
    """{ticker: namn} för Nordens bolag på huvudlistorna ur universum-registret, Large Cap först — None när
    registret saknas i Gisten."""
    from universe import scan as uscan
    members = uscan.members_for("wolf", loader=loader)
    if members is None:
        return None
    members = sorted(members, key=lambda m: (_BOARD_ORDER.get(m.market_id, 1), m.ticker))
    return {m.ticker: (m.name or m.ticker) for m in members}


def registry_viking(loader=None):
    """Nordens alla Börsdata-bolag ur registret som Vikings tickerlista (samma mängd som Vikings egen
    Börsdata-lista, men ur registret) — None när registret saknas."""
    from universe import scan as uscan
    members = uscan.members_for("viking", loader=loader, main_boards=False)
    return None if members is None else tuple(m.ticker for m in members)


def run_wolf(loader=None) -> dict:
    """Wolf 4-lagers regimscore över Norden: de fyra listorna plus registrets huvudlistor. rows sorterade på score."""
    out = {"generated": _now(), "rows": [], "error": None, "universe_source": "listorna"}
    try:
        from wolf_shadow_screener import run_screener, MARKETS, SECTOR_MAP

        markets = list(WOLF_MARKETS)
        MARKETS.pop(WOLF_REGISTRY_KEY, None)
        try:
            reg = registry_wolf(loader)
        except Exception as e:
            log.warning("Wolf: universum-registret gick inte att läsa (%s) — listorna som förut.", e)
            reg = None
        if reg:
            have = {t for m in WOLF_MARKETS for t in MARKETS.get(m, {})}
            extra = {t: n for t, n in reg.items() if t not in have}
            extra = dict(list(extra.items())[:WOLF_EXTRA_MAX])
            MARKETS[WOLF_REGISTRY_KEY] = extra                    # samma grepp som fliken (intl_expanded)
            markets.append(WOLF_REGISTRY_KEY)
            out["universe_source"] = "universum-registret"
            log.info("Wolf: universum-registret ger %d bolag utöver listorna (tak %d).", len(extra), WOLF_EXTRA_MAX)
        else:
            log.info("Wolf: universum-registret saknas — de fyra listorna som förut.")

        pre_fetched = None
        try:
            # Samma prisbatch som fliken använder — Börsdata i klump i stället
            # för yfinance ett-och-ett. Faller tillbaka på yfinance internt.
            from utils.bd_api import BDClient, load_api_key
            key = load_api_key()
            if key:
                bd = BDClient(key)
                stock_tickers = [t for m in markets
                                 for t in MARKETS.get(m, {})]
                fetch = ["SPY"] + list(SECTOR_MAP.values()) + stock_tickers
                pre_fetched = bd.get_price_history_batch(
                    list(dict.fromkeys(fetch)), period="1y")
        except Exception as e:
            log.warning("Wolf: Börsdata-prisbatchen föll — yfinance tar över: %s", e)

        df = run_screener(markets=markets, min_score=WOLF_SAVE_FLOOR,
                          pre_fetched=pre_fetched)
        if df is None or df.empty:
            return out
        for _i, r in df.iterrows():
            score = r.get("Total Score")
            ticker = str(r.get("Ticker", "")).strip()
            if not ticker or score is None:
                continue
            out["rows"].append({"ticker": ticker,
                                "name": str(r.get("Name", "") or ""),
                                "score": round(float(score), 1)})
        out["rows"].sort(key=lambda x: -x["score"])
    except Exception as e:
        import traceback
        log.error("Wolf-scanningen felade:\n%s", traceback.format_exc())
        out["error"] = str(e)
    return out


def run_viking(loader=None) -> dict:
    """Viking (OVTLYR) över Norden — registrets nordiska bolag som tickerlista när registret finns, annars
    Vikings egen Börsdata-lista. V9 parsas ur kolumnen "8/9"-strängen."""
    out = {"generated": _now(), "rows": [], "error": None, "universe_source": "Börsdata"}
    try:
        from screener_ovtlyr import run_ovtlyr_screener

        tickers = None
        try:
            tickers = registry_viking(loader)
        except Exception as e:
            log.warning("Viking: universum-registret gick inte att läsa (%s) — Börsdatas lista som förut.", e)
        if tickers:
            out["universe_source"] = "universum-registret"
            log.info("Viking: universum-registret ger %d nordiska bolag.", len(tickers))
        df = run_ovtlyr_screener(universe="Nordic", ticker_list=tickers)
        if df is None or df.empty:
            return out
        for _i, r in df.iterrows():
            ticker = str(r.get("Ticker", "")).strip()
            if not ticker:
                continue
            raw_v9 = str(r.get("V9", "") or "")
            try:
                v9 = int(raw_v9.split("/")[0])
            except (ValueError, IndexError):
                continue
            eligible = bool(r.get("_eligible", False))
            if v9 < VIKING_SAVE_FLOOR and not eligible:
                continue
            comp = r.get("Composite")
            out["rows"].append({
                "ticker": ticker,
                "name": str(r.get("Name", "") or ""),
                "v9": v9,
                "eligible": eligible,
                "signal": str(r.get("Signal", "") or ""),
                "composite": (round(float(comp), 2)
                              if isinstance(comp, (int, float)) else None),
            })
        out["rows"].sort(key=lambda x: (-x["v9"], -(x["composite"] or 0)))
    except Exception as e:
        import traceback
        log.error("Viking-scanningen felade:\n%s", traceback.format_exc())
        out["error"] = str(e)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wolf = run_wolf()
    log.info("Wolf: %d rader >= %g%s", len(wolf["rows"]), WOLF_SAVE_FLOOR,
             f" (FEL: {wolf['error']})" if wolf["error"] else "")
    viking = run_viking()
    log.info("Viking: %d rader%s", len(viking["rows"]),
             f" (FEL: {viking['error']})" if viking["error"] else "")

    payload = {"wolf": wolf, "viking": viking}
    if args.dry_run:
        log.info("[DRY-RUN] sparar inte till Gisten.")
        return 0

    from gist_storage import save_blob
    if save_blob(BLOB_NAME, payload):
        log.info("Sparat till Gisten som %s.", BLOB_NAME)
    else:
        log.error("Kunde inte spara %s till Gisten — larmbenen ser gammal "
                  "eller ingen data. Kontrollera GITHUB_TOKEN (gist-scope).",
                  BLOB_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
