#!/usr/bin/env python3
"""
arc_scan.py — headless körning av Arc-screenerna Wolf och Viking.

Körs av scheduled-scan-workflowen (08/12/18 vardagar) före larmsteget.
Flikarna kör dessa scanningar live när man trycker SCAN — men larm kräver
att någon räknar även när panelen är stängd, med FASTA inställningar:

  Wolf    wolf_shadow_screener.run_screener på de nordiska marknaderna
          (Stockholm/Oslo/Köpenhamn/Helsingfors) plus registrets nordiska
          huvudlistor. Alla rader med Total Score >= GOLV sparas — larmribban
          (default 80) ligger i alert_rules/inställningarna.
  Viking  screener_ovtlyr.run_ovtlyr_screener("Nordic"). Rader med
          Vikings Nine >= 6 eller absolut eligibility sparas.
  USA     samma två screeners på universum-registrets amerikanska bolag
          (NYSE/Nasdaq) med börsvärde minst USA_MIN_MCAP_MUSD — mid och large
          cap, inte small cap. Sparas som egna nycklar (wolf_usa, viking_usa)
          så larmbenen och Arc-flikarna (Norden) är orörda; Thor läser båda.

Resultatet skrivs till Gisten som arc_screeners.json:
  {"wolf": {"generated", "rows": [{"ticker","name","score"}], "error"},
   "viking": {"generated", "rows": [{"ticker","name","v9","eligible",
                                     "signal","composite"}], "error"},
   "wolf_usa": {...}, "viking_usa": {...}}

En screener som felar ger error-fältet satt och rows=[] — larmbenet ser då
ingen läsbar data (alert_scan skickar None) och behåller sin baslinje.

Env: BORSDATA_API_KEY (prisbatch för Wolf, börsvärden för USA-filtret;
     Viking klarar sig på yfinance), GITHUB_TOKEN (gist-scope, för save_blob).
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
USA_MARKET = "usa"
USA_WOLF_KEY = "registry_usa"    # registrets USA som egen marknad i Wolf-screenern
USA_MIN_MCAP_MUSD = 2000.0       # "utan small cap": mid och large cap (gränsen 2 miljarder USD, Börsdatas KPI 50)


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


def registry_usa(loader=None, api=None):
    """{ticker: namn} för registrets amerikanska bolag på NYSE/Nasdaq med börsvärde >= USA_MIN_MCAP_MUSD, största
    först. Börsvärdet kommer ur Börsdatas globala KPI-screener (KPI 50, MUSD) — utan Börsdata (api None) kan
    small cap inte sållas bort, och då skannas inte USA alls (None). None även när registret saknas."""
    from universe import scan as uscan
    members = uscan.members_for("wolf", loader=loader, markets=(USA_MARKET,))     # huvudlistorna: NYSE, Nasdaq
    if members is None:
        return None
    if api is None:
        log.warning("USA: ingen Börsdata-nyckel — storleksfiltret kan inte räknas, USA hoppas över.")
        return None
    from borsdata_api import KPI
    caps = {}
    for e in api.get_kpi_screener_global(KPI["market_cap"]) or []:
        try:
            if e.get("i") is not None and e.get("n") is not None:
                caps[int(e["i"])] = float(e["n"])
        except (TypeError, ValueError):
            continue
    if not caps:
        log.warning("USA: Börsdatas globala börsvärden saknas (licens?) — USA hoppas över.")
        return None
    keep = [m for m in members if m.ins_id is not None and (caps.get(int(m.ins_id)) or 0.0) >= USA_MIN_MCAP_MUSD]
    keep.sort(key=lambda m: (-caps[int(m.ins_id)], m.ticker))
    log.info("USA: registret %d bolag på NYSE/Nasdaq · %d med börsvärde >= %.0f MUSD.", len(members), len(keep),
             USA_MIN_MCAP_MUSD)
    return {m.ticker: (m.name or m.ticker) for m in keep}


def _wolf_rows(df) -> list:
    """Screenerns DataFrame → [{ticker, name, score}] sorterade på score."""
    rows = []
    for _i, r in df.iterrows():
        score = r.get("Total Score")
        ticker = str(r.get("Ticker", "")).strip()
        if not ticker or score is None:
            continue
        rows.append({"ticker": ticker, "name": str(r.get("Name", "") or ""), "score": round(float(score), 1)})
    rows.sort(key=lambda x: -x["score"])
    return rows


def _viking_rows(df) -> list:
    """Screenerns DataFrame → [{ticker, name, v9, eligible, signal, composite}] för V9 >= golvet eller eligible."""
    rows = []
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
        rows.append({
            "ticker": ticker,
            "name": str(r.get("Name", "") or ""),
            "v9": v9,
            "eligible": eligible,
            "signal": str(r.get("Signal", "") or ""),
            "composite": (round(float(comp), 2) if isinstance(comp, (int, float)) else None),
        })
    rows.sort(key=lambda x: (-x["v9"], -(x["composite"] or 0)))
    return rows


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

        try:
            df = run_screener(markets=markets, min_score=WOLF_SAVE_FLOOR,
                              pre_fetched=pre_fetched)
        finally:
            MARKETS.pop(WOLF_REGISTRY_KEY, None)                 # tillfällig marknad — listorna lämnas som de var
        if df is None or df.empty:
            return out
        out["rows"] = _wolf_rows(df)
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
        out["rows"] = _viking_rows(df)
    except Exception as e:
        import traceback
        log.error("Viking-scanningen felade:\n%s", traceback.format_exc())
        out["error"] = str(e)
    return out


def _usa_out(reg) -> dict:
    out = {"generated": _now(), "rows": [], "error": None, "universe_source": "universum-registret (USA)",
           "min_mcap_musd": USA_MIN_MCAP_MUSD, "universe": len(reg or {})}
    if not reg:
        out["universe_source"] = "hoppades över"
    return out


def run_wolf_usa(loader=None, api=None, reg=None) -> dict:
    """Wolf över registrets USA (mid och large cap). Kurserna kommer ur yfinance i klump (Börsdatas prisbatch
    täcker inte amerikanska aktier). rows sorterade på score. Utan universum (ingen Börsdata) hoppas USA över."""
    if reg is None:
        try:
            reg = registry_usa(loader, api)
        except Exception as e:
            log.warning("Wolf USA: universum-registret gick inte att läsa (%s) — USA hoppas över.", e)
            reg = None
    out = _usa_out(reg)
    if not reg:
        return out
    try:
        from wolf_shadow_screener import run_screener, MARKETS
        MARKETS[USA_WOLF_KEY] = dict(reg)                        # tillfällig marknad — tas bort efteråt
        try:
            log.info("Wolf USA: %d bolag med börsvärde >= %.0f MUSD.", len(reg), USA_MIN_MCAP_MUSD)
            df = run_screener(markets=[USA_WOLF_KEY], min_score=WOLF_SAVE_FLOOR, pre_fetched=None)
        finally:
            MARKETS.pop(USA_WOLF_KEY, None)                      # listorna är universum-registrets källa — rör dem inte
        if df is None or df.empty:
            return out
        out["rows"] = _wolf_rows(df)
    except Exception as e:
        import traceback
        log.error("Wolf USA-scanningen felade:\n%s", traceback.format_exc())
        out["error"] = str(e)
    return out


def run_viking_usa(loader=None, api=None, reg=None) -> dict:
    """Viking (OVTLYR) över registrets USA (mid och large cap) — SPY som marknadsgrind via universe="US"."""
    if reg is None:
        try:
            reg = registry_usa(loader, api)
        except Exception as e:
            log.warning("Viking USA: universum-registret gick inte att läsa (%s) — USA hoppas över.", e)
            reg = None
    out = _usa_out(reg)
    if not reg:
        return out
    try:
        from screener_ovtlyr import run_ovtlyr_screener
        log.info("Viking USA: %d bolag med börsvärde >= %.0f MUSD.", len(reg), USA_MIN_MCAP_MUSD)
        df = run_ovtlyr_screener(universe="US", ticker_list=tuple(reg))
        if df is None or df.empty:
            return out
        out["rows"] = _viking_rows(df)
    except Exception as e:
        import traceback
        log.error("Viking USA-scanningen felade:\n%s", traceback.format_exc())
        out["error"] = str(e)
    return out


def borsdata_for_usa():
    """Börsdata-klienten för USA-filtret (globala börsvärden) — None utan nyckel."""
    try:
        from utils.bd_api import load_api_key
        key = load_api_key()
        if not key:
            return None
        from borsdata_api import BorsdataAPI
        return BorsdataAPI(api_key=key)
    except Exception as e:
        log.warning("USA: Börsdata-klienten gick inte att skapa (%s).", e)
        return None


def scan_all(loader=None, usa_api=None) -> dict:
    """Hela körningen: Norden (som förut) och USA (registret, mid och large cap) — ett universum för USA räknas
    en gång och delas av Wolf och Viking."""
    wolf = run_wolf(loader)
    log.info("Wolf: %d rader >= %g%s", len(wolf["rows"]), WOLF_SAVE_FLOOR,
             f" (FEL: {wolf['error']})" if wolf["error"] else "")
    viking = run_viking(loader)
    log.info("Viking: %d rader%s", len(viking["rows"]),
             f" (FEL: {viking['error']})" if viking["error"] else "")
    try:
        usa = registry_usa(loader, usa_api)
    except Exception as e:
        log.warning("USA: universum-registret gick inte att läsa (%s) — USA hoppas över.", e)
        usa = None
    wolf_usa = run_wolf_usa(loader, usa_api, reg=usa or {})
    log.info("Wolf USA: %d rader >= %g%s", len(wolf_usa["rows"]), WOLF_SAVE_FLOOR,
             f" (FEL: {wolf_usa['error']})" if wolf_usa["error"] else "")
    viking_usa = run_viking_usa(loader, usa_api, reg=usa or {})
    log.info("Viking USA: %d rader%s", len(viking_usa["rows"]),
             f" (FEL: {viking_usa['error']})" if viking_usa["error"] else "")
    return {"wolf": wolf, "viking": viking, "wolf_usa": wolf_usa, "viking_usa": viking_usa}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = scan_all(usa_api=borsdata_for_usa())
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
