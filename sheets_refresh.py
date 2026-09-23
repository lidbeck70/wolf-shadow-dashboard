#!/usr/bin/env python3
"""
sheets_refresh.py — sifferuppdatering och övergångslarm för granskningsarken.

Arken (Insider, Tiggre, Rick Rule, Royalty C, Poängmodellen) håller bara
INMATNINGAR — kurs nu, EV/EBITDA, nettoskuld/EBITDA, börsvärde — och allt
annat räknas vid rendering. De siffrorna åldras mellan besöken, och flera
av arkens egna larmvillkor (Tiggres +100 %, Insiders stopp på −15 % under
klustersnittet, Royaltys "under egen median") kan därför aldrig avfyras
utan att någon skriver in en ny kurs för hand.

Det här jobbet läser arkens datafiler på panel-data-grenen (samma väg som
alert_scan läser inställningarna), hämtar färska tal ur Börsdata och
skriver dem som FÖRSLAG till Gisten (sheets_refresh.json). Arken visar
förslaget bredvid fältet med en "Använd"-knapp — inget skrivs över, och
arkens lås-per-session rörs inte.

Övergångarna räknas med arkens egna funktioner (insider.stop_price /
is_chase, tiggre.free_ride_reached / p_nav, producers.royalty_signal,
lukacs.deleveraging_state, scoring.durrett_buy_ok) — inga regler ändras:

  insider_stop        kurs ≤ klustersnitt × 0,85           (arkets säljregel 2)
  insider_chase       kurs > +30 % mot klustersnittet     (passa-regeln)
  tiggre_free_ride    +100 % mot entry, halva inte såld    (free ride-larmet)
  tiggre_nav_target   P/NAV ≥ 0,8 med färskt börsvärde     ("slutsälj i etapper")
  royalty_signal      signalen byter etikett med färsk EV/EBITDA
  rule_deleveraging   skuld/EBITDA korsar 1,0 (halv position) i Rick Rule
  durrett_buy_rule    MCap/framtida vinst korsar 10× i Durrett

Env: BORSDATA_API_KEY, GITHUB_TOKEN (gist), ALERT_REPO_TOKEN (Contents: Read
på panel-data). Flaggor: --dry-run.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sheets_refresh")

BLOB_NAME = "sheets_refresh.json"
SHEET_FILES = {"insider": "data/insider.json", "tiggre": "data/tiggre.json",
               "producers": "data/producers.json", "scoring": "data/scoring.json",
               "confidence": "data/confidence.json",          # Durrett-/Confidence-arket
               "holdings": "data/holdings.json"}              # registret (positions.py)
_BUCKETS = {"insider": ("signals",), "tiggre": ("candidates", "positions"),
            "producers": ("producers", "royalty"), "scoring": ("sprott", "durrett"),
            "confidence": ("companies",)}
# confidence.json är {"companies": {TICKER: bolag}} — raden får ticker som id.
_SUFFIX_RE = re.compile(r"\.(ST|OL|HE|CO)$", re.I)

FX_TO_USD = {"SEK": 0.095, "NOK": 0.095, "DKK": 0.145, "EUR": 1.08, "USD": 1.0,
             "CAD": 0.73, "AUD": 0.66, "GBP": 1.27, "CHF": 1.13, "PLN": 0.25}


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _f(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def ref_key(sheet: str, row: dict) -> str:
    return f"{sheet}:{row.get('id')}"


def collect_rows(sheets: dict) -> list:
    """Alla arkrader med ticker: [{sheet, bucket, key, ticker, ins_id, row}]."""
    out = []
    for sheet, buckets in _BUCKETS.items():
        data = sheets.get(sheet) or {}
        for b in buckets:
            rows = data.get(b, []) or []
            if sheet == "tiggre" and b == "positions":
                # Tiggre-positionerna bor i registret (Holdings) sedan PR 10;
                # nyckeln förblir tiggre:<id> så arket hittar sina färska tal.
                import positions as _positions
                held = _positions.view_rows_from(sheets.get("holdings"), "Tiggre")
                have = {p.get("id") for p in held}
                rows = held + [p for p in rows if isinstance(p, dict) and p.get("id") not in have]
            if isinstance(rows, dict):                       # confidence: {TICKER: bolag}
                rows = [dict(v, id=k) for k, v in rows.items() if isinstance(v, dict)]
            for row in rows:
                if not isinstance(row, dict) or not row.get("id"):
                    continue
                t = str(row.get("ticker") or "").strip().upper()
                if not t:
                    continue
                out.append({"sheet": sheet, "bucket": b, "key": ref_key(sheet, row),
                            "ticker": t, "ins_id": row.get("ins_id"), "row": row})
    return out


def ticker_forms(ticker: str) -> list:
    """'EKTA-B.ST' → ['EKTA B', 'EKTA-B', 'EKTA B.ST', ...] för Börsdata-uppslag."""
    t = str(ticker or "").strip().upper()
    base = _SUFFIX_RE.sub("", t)
    forms = [base.replace("-", " "), base, base.replace(" ", "-"), t]
    seen, out = set(), []
    for f in forms:
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def resolve(api, ticker: str, ins_id=None) -> Optional[int]:
    if ins_id is not None:
        try:
            return int(ins_id)
        except (TypeError, ValueError):
            pass
    for form in ticker_forms(ticker):
        try:
            iid = api.resolve_instrument_id(form)
        except Exception:
            iid = None
        if iid is not None:
            return int(iid)
    return None


def _last_close(api, ins_id: int):
    try:
        raw = api.get_stockprices(ins_id, max_count=5)
    except Exception:
        return None, None
    bars = sorted((b for b in raw or [] if b.get("c") is not None),
                  key=lambda b: b.get("d") or "")
    if not bars:
        return None, None
    return float(bars[-1]["c"]), str(bars[-1].get("d") or "")[:10]


def _yf_close(ticker: str):
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        if h is None or h.empty:
            return None, None
        return float(h["Close"].iloc[-1]), str(h.index[-1])[:10]
    except Exception:
        return None, None


def refresh(api, sheets: dict) -> dict:
    """Färska tal per arkrad + övergångar. Rena arkfunktioner för händelserna."""
    out = {"generated": _now(), "rows": {}, "events": [], "error": None}
    refs = collect_rows(sheets)
    if not refs:
        return out
    try:
        inst_map = {int(i["insId"]): i for i in api.get_instruments()
                    if i.get("insId") is not None}
        resolved = {}
        for r in refs:
            iid = resolve(api, r["ticker"], r["ins_id"])
            if iid is not None:
                resolved[r["key"]] = iid
        ids = sorted(set(resolved.values()))
        nordic_ids = [i for i in ids if i in inst_map]
        missing_meta = [i for i in ids if i not in inst_map]
        if missing_meta:
            for i in api.get_global_instruments_list():
                if i.get("insId") is not None:
                    inst_map.setdefault(int(i["insId"]), i)
        # KPI-screenern är universum-bunden: globala id:n (Tiggre/Durrett-
        # raderna på TSX/ASX) ger inget ur den nordiska — hämta dem ur den
        # globala.
        snaps = api.get_fundamentals_snapshot_fast(nordic_ids) if nordic_ids else {}
        if missing_meta:
            snaps.update(api.get_fundamentals_snapshot_fast(missing_meta, scope="global") or {})

        price_cache = {}
        for r in refs:
            iid = resolved.get(r["key"])
            s = {"ticker": r["ticker"], "ins_id": iid, "price": None, "asof": None,
                 "currency": None, "source": None, "ev_ebitda": None,
                 "nd_ebitda": None, "mcap_musd": None}
            if iid is not None:
                if iid not in price_cache:
                    price_cache[iid] = _last_close(api, iid)
                s["price"], s["asof"] = price_cache[iid]
                s["source"] = "borsdata"
                meta = inst_map.get(iid) or {}
                ccy = str(meta.get("stockPriceCurrency") or "").upper() or None
                s["currency"] = ccy
                snap = snaps.get(iid) or {}
                s["ev_ebitda"] = _f(snap.get("ev_ebitda"))
                s["nd_ebitda"] = _f(snap.get("net_debt_ebitda"))
                fx = FX_TO_USD.get(ccy or "USD", 1.0)
                mc = _f(snap.get("market_cap"))
                if mc is not None:
                    s["mcap_musd"] = round(mc * fx, 1)
                if r["sheet"] == "confidence":               # Durrett-arket: fler tal ur snapshoten
                    rccy = str(meta.get("reportCurrency") or ccy or "USD").upper()
                    rfx = FX_TO_USD.get(rccy, 1.0)
                    s["fx_to_usd"] = fx if ccy else None
                    s["fx_table"] = "sheets_refresh.FX_TO_USD (fast tabell)"
                    for src_key, out_key, factor in (("ev", "ev_musd", fx), ("net_debt_m", "net_debt_musd", rfx),
                                                     ("revenue_m", "revenue_musd", rfx), ("fcf_m", "fcf_musd", rfx),
                                                     ("ocf_m", "ocf_musd", rfx)):
                        v = _f(snap.get(src_key))
                        s[out_key] = round(v * factor, 1) if v is not None else None
                    for k in ("pe", "ps", "rs_rank", "ebitda_margin"):
                        s[k] = _f(snap.get(k))
            else:
                s["price"], s["asof"] = _yf_close(r["ticker"])
                s["source"] = "yfinance" if s["price"] is not None else None
            out["rows"][r["key"]] = s
        out["events"] = build_events(sheets, out["rows"])
    except Exception as e:
        import traceback
        log.error("Arkuppdateringen felade:\n%s", traceback.format_exc())
        out["error"] = str(e)
    return out


# ── Övergångarna, med arkens egna funktioner ────────────────────────────────
def _ev(kind, sheet, row, title, body) -> dict:
    return {"key": f"{kind}:{ref_key(sheet, row)}", "kind": kind, "sheet": sheet,
            "ticker": str(row.get("ticker") or "").upper(), "title": title, "body": body}


def build_events(sheets: dict, rows: dict) -> list:
    import insider as ins
    import tiggre as tig
    import producers as prod
    import scoring as sc
    import lukacs

    events = []
    for r in collect_rows(sheets):
        s = rows.get(r["key"]) or {}
        row, sheet, bucket = r["row"], r["sheet"], r["bucket"]
        price = _f(s.get("price"))
        t = r["ticker"]

        if sheet == "insider" and price is not None and _f(row.get("cluster_avg")):
            tmp = dict(row, price_now=price)
            stop = ins.stop_price(tmp)
            vs = ins.vs_cluster(tmp)
            if stop is not None and price <= stop:
                events.append(_ev("insider_stop", sheet, row,
                                  f"🛑 Insider: {t} under stoppen",
                                  f"Kurs {price:.2f} ≤ stopp {stop:.2f} (klustersnitt × 0,85). "
                                  f"Arkets säljregel 2 — lägg ordern."))
            elif ins.is_chase(tmp):
                events.append(_ev("insider_chase", sheet, row,
                                  f"⚠️ Insider: {t} {vs:+.0f} % över klustersnittet",
                                  f"Kurs {price:.2f} mot klustersnitt "
                                  f"{_f(row.get('cluster_avg')):.2f} — över +30 % är edgen "
                                  f"förbrukad. Passa."))

        elif sheet == "tiggre" and bucket == "positions":
            entry = _f(row.get("entry"))
            if price is not None and entry and not row.get("half_sold") \
                    and tig.free_ride_reached(entry, price):
                events.append(_ev("tiggre_free_ride", sheet, row,
                                  f"🎯 Tiggre: {t} +100 % — sälj halva",
                                  f"Kurs {price:.2f} mot entry {entry:.2f} "
                                  f"({(price / entry - 1) * 100:+.0f} %). Free ride: sälj "
                                  f"halva, resten åker på husets pengar."))
            mc, nav = _f(s.get("mcap_musd")), _f(row.get("nav"))
            pn = tig.p_nav(mc, nav) if (mc and nav) else None
            if pn is not None and pn >= tig.NAV_TARGET:
                events.append(_ev("tiggre_nav_target", sheet, row,
                                  f"🎯 Tiggre: {t} vid {pn:.2f}× NAV",
                                  f"Börsvärde {mc:,.0f} MUSD mot NAV {nav:,.0f} MUSD — "
                                  f"{tig.NAV_TARGET:g}× nått. Slutsälj i etapper vid "
                                  f"{tig.NAV_EXIT_LO:g}–{tig.NAV_EXIT_HI:g}× NAV eller "
                                  f"produktionsstart."))
            # −40 % från entry = omvärdera från noll (reference.py: Tiggres säljregel).
            if price is not None and entry and tig.drawdown_review(entry, price):
                events.append(_ev("tiggre_drawdown", sheet, row,
                                  f"⚠️ Tiggre: {t} −{tig.REVIEW_DRAWDOWN_PCT:g} % från entry",
                                  f"Kurs {price:.2f} mot entry {entry:.2f} "
                                  f"({(price / entry - 1) * 100:+.0f} %). Omvärdera positionen "
                                  f"från noll: köp mer, behåll eller sälj — inte vänta."))

        elif sheet == "producers" and bucket == "royalty":
            ev = _f(s.get("ev_ebitda"))
            if ev is not None and _f(row.get("ev_median")):
                before = prod.royalty_signal(row).label
                after = prod.royalty_signal(dict(row, ev_now=ev)).label
                if before != after:
                    events.append(_ev("royalty_signal", sheet, row,
                                      f"👑 Royalty: {t} → {after}",
                                      f"Med färsk EV/EBITDA {ev:.1f} (median "
                                      f"{_f(row.get('ev_median')):.1f}) byter signalen "
                                      f"från '{before}' till '{after}'. P/NAV och GEO är "
                                      f"dina siffror — kolla dem."))

        elif sheet == "producers" and bucket == "producers":
            nd = _f(s.get("nd_ebitda"))
            if nd is not None and _f(row.get("nd_ebitda")) is not None:
                before = lukacs.deleveraging_state(row.get("nd_ebitda"))["half_position"]
                after = lukacs.deleveraging_state(nd)["half_position"]
                if before != after:
                    events.append(_ev("rule_deleveraging", sheet, row,
                                      f"🪨 Rick Rule: {t} skuld/EBITDA {nd:.1f}",
                                      (f"Skulden gick över {lukacs.DELEV_ND_MIN:g}× — max halv "
                                       f"position och år till låg skuld under "
                                       f"{lukacs.DELEV_YEARS_MAX:g}." if after else
                                       f"Skulden gick under {lukacs.DELEV_ND_MIN:g}× — "
                                       f"halveringen av positionen gäller inte längre.")))

        elif sheet == "confidence":
            events.extend(_durrett_engine_events(row, s))

        elif sheet == "scoring" and bucket == "durrett":
            mc = _f(s.get("mcap_musd"))
            profit = _f(row.get("profit"))
            if mc is not None and profit and _f(row.get("mcap")):
                before = sc.durrett_buy_ok(sc.mcap_per_earnings(row.get("mcap"), profit))
                ratio = sc.mcap_per_earnings(mc, profit)
                after = sc.durrett_buy_ok(ratio)
                if before != after:
                    events.append(_ev("durrett_buy_rule", sheet, row,
                                      f"🥇 Durrett: {t} {ratio:.1f}× framtida vinst",
                                      (f"Börsvärde {mc:,.0f} MUSD — under 10× framtida vinst, "
                                       f"köpregeln uppfylld." if after else
                                       f"Börsvärde {mc:,.0f} MUSD — över 10× framtida vinst, "
                                       f"köpregeln gäller inte längre.")))
    return events


def _durrett_engine_events(row: dict, s: dict) -> list:
    """Durrett-arket (engines/durrett): köpregeln MCap/framtida vinst korsar 10×
    med färskt börsvärde. Motorn är ren — bolaget byggs ur den sparade raden,
    börsvärdet byts och analysen körs om. Inga regler ändras."""
    mc = _f(s.get("mcap_musd"))
    if mc is None:
        return []
    try:
        from confidence.data.models import CompanyInput
        from confidence.data.provenance import dp
        from engines.durrett.engine import analyze
        from scoring import DURRETT_BUY_MAX
    except Exception:                                   # pragma: no cover
        return []
    try:
        company = CompanyInput.from_dict({k: v for k, v in row.items() if k != "id"})
        if not company.has("market_cap_musd"):
            return []
        before = (analyze(company).metrics.get("mcap_future_earnings") or {}).get("value")
        fresh = CompanyInput.from_dict(company.as_dict())
        fresh.set("market_cap_musd", dp(mc, kind="ACTUAL", source="Börsdata (sifferuppdatering)",
                                        source_type="secondary", pub_date=s.get("asof"), unit="MUSD"))
        after = (analyze(fresh).metrics.get("mcap_future_earnings") or {}).get("value")
    except Exception as exc:                            # motorn får aldrig fälla jobbet
        log.warning("Durrett-händelse för %s hoppades över: %s", row.get("ticker"), exc)
        return []
    if before is None or after is None:
        return []
    ok_before, ok_after = before < DURRETT_BUY_MAX, after < DURRETT_BUY_MAX
    if ok_before == ok_after:
        return []
    t = str(row.get("ticker") or "").upper()
    return [_ev("durrett_engine_buy_rule", "confidence", row,
                f"🐺 Durrett-arket: {t} {after:.1f}× framtida vinst",
                (f"Färskt börsvärde {mc:,.0f} MUSD — under {DURRETT_BUY_MAX:g}× framtida vinst, "
                 f"köpregeln uppfylld (var {before:.1f}×)." if ok_after else
                 f"Färskt börsvärde {mc:,.0f} MUSD — över {DURRETT_BUY_MAX:g}× framtida vinst, "
                 f"köpregeln gäller inte längre (var {before:.1f}×)."))]


def load_sheets() -> dict:
    from alert_scan import _repo_file
    return {k: _repo_file(path, {}) for k, path in SHEET_FILES.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from borsdata_api import BorsdataAPI
    key = os.environ.get("BORSDATA_API_KEY", "") or os.environ.get("BD_API_KEY", "")
    if not key:
        log.error("BORSDATA_API_KEY saknas.")
        return 1
    sheets = load_sheets()
    log.info("Ark: %s", {k: {b: len(v.get(b, []) or []) for b in _BUCKETS[k]}
                         for k, v in sheets.items() if isinstance(v, dict)})
    result = refresh(BorsdataAPI(api_key=key), sheets)
    n_price = sum(1 for s in result["rows"].values() if s.get("price") is not None)
    log.info("Rader: %d, kurs hämtad för %d, händelser: %d%s", len(result["rows"]),
             n_price, len(result["events"]),
             f" (FEL: {result['error']})" if result["error"] else "")
    for e in result["events"]:
        log.info("  %s — %s", e["title"], e["body"][:90])
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
