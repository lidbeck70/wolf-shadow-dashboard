#!/usr/bin/env python3
"""
insider_scan.py — headless Insiderbevakaren.

Körs av scheduled-scan-workflowen (08/12/18 vardagar) före larmsteget, precis
som Deep Contrarian. Arket i GRANSKNING → Insider är helt manuellt; det här
jobbet räknar samma sak ur Börsdatas insynsregister (/holdings/insider) så
att kluster upptäcks och larmas även när panelen är stängd.

INGA REGLER ÄNDRAS. Poäng, status, vs-kluster, stopp och passa-regeln räknas
med insider.py:s egna funktioner (score, status, vs_cluster, stop_price,
is_chase). Det här scriptet fyller bara arkets INMATNINGSFÄLT ur data:

  insiders      distinkta köpare senaste 30 dagarna (riktiga köp: shares > 0,
                inte equityProgram)
  role          högsta roll i klustret ur ownerPosition (VD/CFO · Styrelse · Övrig)
  amount        klustrets totala belopp i tkr, grovt omräknat till SEK
  aterkommande  någon av köparna har köpt på minst två datum senaste 12 mån
  cluster_avg   aktieviktad snittkurs för klustrets köp
  price_now     senaste stängning (Börsdata)
  gate          Ja när alla fem grindkriterier mäts OK, Nej när något mätbart
                faller, "" (= "Kör kvalitetsgrinden!") när något kräver dig —
                F-score ligger inte i licensen, "väg till FCF" och
                "strukturellt fallande" är bedömningar
  trigger       A (stängning > MA20, MA20 planat/vänt) · B (1 mån > 0 och kurs
                över klustersnitt) · Nej
  efter_fall    bara när kursen är ≥ 20 % under 52v-topp OCH F-score känd ≥ 5
                (arkets definition: "i friskt bolag") — annars False + notis
  okar_25       aldrig automatiskt (kräver köparens tidigare innehav) → notis

Resultatet skrivs till Gisten som insider_scan.json:
  {"generated", "universe", "with_buys", "clusters": [{...arkets fält...,
   "score", "status", "vs_cluster", "stop", "chase", "auto": {noter}}],
   "error"}
Bara kluster med poäng ≥ 5 (Bevaka och uppåt) sparas.

Env: BORSDATA_API_KEY, GITHUB_TOKEN (gist-scope). Flaggor: --dry-run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("insider_scan")

BLOB_NAME = "insider_scan.json"

CLUSTER_DAYS = 30          # arkets "Antal insiders 30 dgr"
RECUR_DAYS = 365           # "Återkommande köpare" — samma person, flera köp
ENRICH_MIN_PRELIM = 4      # prelim-poäng (utan efter_fall/okar_25) för pris+fundamenta
SAVE_MIN_SCORE = 5         # spara Bevaka (5–6) och uppåt

# Grindkriterierna (insider.GATE_CRITERIA) i siffror — samma text, samma tal.
MCAP_MIN_MSEK = 300.0
FSCORE_MIN = 5
ND_EBITDA_MAX = 2.0
FALL_PCT = 20.0            # "Efter fall > 20 %"

# Grova kurser till SEK. Arkets trösklar (300 MSEK, tkr) är i kronor;
# för > 300 MSEK och 500/1000 tkr räcker en grov omräkning.
FX_TO_SEK = {"SEK": 1.0, "NOK": 0.95, "DKK": 1.55, "EUR": 11.3,
             "USD": 10.5, "GBP": 13.5}

# Börsdatas marknads-id → yfinance-suffix (samma som ember/universe.py)
_MARKET_SUFFIX = {1: ".ST", 2: ".ST", 3: ".ST", 7: ".ST", 8: ".ST", 9: ".ST",
                  18: ".ST", 19: ".ST", 4: ".OL", 14: ".OL", 5: ".HE", 16: ".HE",
                  6: ".CO", 15: ".CO"}
_INDEX_MARKETS = {7, 8, 13, 19, 28, 31}

ROLE_TOP, ROLE_BOARD, ROLE_OTHER = "VD/CFO", "Styrelse", "Övrig"
_ROLE_RANK = {ROLE_TOP: 2, ROLE_BOARD: 1, ROLE_OTHER: 0}
_TOP_HINTS = ("ceo", "cfo", "verkställande", "finanschef", "chief executive",
              "chief financial", "managing director")
_BOARD_HINTS = ("board", "chairman", "styrelse", "ordförande", "director")


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


def _parse_date(s) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def role_from_position(position) -> str:
    """ownerPosition ('cfo', 'board member', 'external ceo', None) → arkets roll."""
    p = str(position or "").lower()
    if any(h in p for h in _TOP_HINTS):
        return ROLE_TOP
    if any(h in p for h in _BOARD_HINTS):
        return ROLE_BOARD
    return ROLE_OTHER


def yf_ticker(inst: dict) -> str:
    t = str(inst.get("ticker") or "").strip().upper().replace(" ", "-")
    return t + _MARKET_SUFFIX.get(inst.get("marketId"), "")


# ── Klustret ur transaktionerna ──────────────────────────────────────────────
def build_cluster(values: list, today: date, fx: float = 1.0) -> Optional[dict]:
    """Riktiga köp senaste CLUSTER_DAYS → arkets fält. None om inga köp.

    Riktning = tecknet på shares (Börsdatas typkoder är numeriska).
    equityProgram (tilldelning/optioner) räknas inte — arket: "Endast riktiga
    marknadsköp räknas".
    """
    recent, year = [], []
    for t in values or []:
        if not isinstance(t, dict) or t.get("equityProgram"):
            continue
        shares = _f(t.get("shares"))
        if shares is None or shares <= 0:
            continue
        d = _parse_date(t.get("transactionDate") or t.get("verificationDate"))
        if d is None:
            continue
        age = (today - d).days
        if age < 0 or age > RECUR_DAYS:
            continue
        price = _f(t.get("price"))
        amount = _f(t.get("amount"))
        if amount is None and price is not None:
            amount = shares * price
        row = {"date": d.isoformat(), "owner": str(t.get("ownerName") or "?"),
               "position": str(t.get("ownerPosition") or ""),
               "shares": shares, "price": price,
               "amount": abs(amount) if amount is not None else None}
        year.append(row)
        if age <= CLUSTER_DAYS:
            recent.append(row)
    if not recent:
        return None

    owners = {r["owner"] for r in recent}
    amount_local = sum(r["amount"] for r in recent if r["amount"] is not None)
    priced = [r for r in recent if r["price"] is not None]
    tot_sh = sum(r["shares"] for r in priced)
    avg = (sum(r["shares"] * r["price"] for r in priced) / tot_sh) if tot_sh else None
    role = max((role_from_position(r["position"]) for r in recent),
               key=lambda x: _ROLE_RANK[x])

    dates_by_owner = defaultdict(set)
    for r in year:
        dates_by_owner[r["owner"]].add(r["date"])
    recurring = any(len(dates_by_owner[o]) >= 2 for o in owners)

    recent.sort(key=lambda r: r["date"], reverse=True)
    return {
        "insiders": len(owners),
        "role": role,
        "amount_local": round(amount_local / 1000.0, 1),        # t-lokal valuta
        "amount": round(amount_local * fx / 1000.0, 1),          # tkr (SEK)
        "cluster_avg": round(avg, 4) if avg is not None else None,
        "aterkommande": recurring,
        "buys": recent,
        "first_buy": min(r["date"] for r in recent),
        "last_buy": max(r["date"] for r in recent),
    }


# ── Grind, trigger, efter-fall ur data ───────────────────────────────────────
def auto_gate(snap: dict, fx: float = 1.0) -> tuple:
    """(gate, checks). Ja = allt mätbart OK, Nej = något mätbart faller,
    "" = något kräver dig. Ordning och text = insider.GATE_CRITERIA."""
    checks = []

    def add(label, status, detail):
        checks.append({"label": label, "status": status, "detail": detail})

    mcap = _f(snap.get("market_cap"))
    if mcap is None:
        add("Börsvärde > 300 MSEK", "unknown", "börsvärde saknas")
    else:
        m = mcap * fx
        add("Börsvärde > 300 MSEK", "ok" if m > MCAP_MIN_MSEK else "fail",
            f"{m:,.0f} MSEK")

    fs = _f(snap.get("f_score"))
    if fs is None:
        add("F-score ≥ 5", "unknown", "F-score ligger inte i API-licensen — kolla i Börsdata")
    else:
        add("F-score ≥ 5", "ok" if fs >= FSCORE_MIN else "fail", f"F-score {fs:g}")

    nd_m, nde = _f(snap.get("net_debt_m")), _f(snap.get("net_debt_ebitda"))
    if nd_m is not None and nd_m < 0:
        add("Nettoskuld/EBITDA < 2 (eller nettokassa)", "ok", "nettokassa")
    elif nde is not None:
        add("Nettoskuld/EBITDA < 2 (eller nettokassa)",
            "ok" if nde < ND_EBITDA_MAX else "fail", f"{nde:.1f}×")
    else:
        add("Nettoskuld/EBITDA < 2 (eller nettokassa)", "unknown", "nettoskuld saknas")

    fcf = _f(snap.get("fcf_m"))
    if fcf is None:
        add("Positivt FCF eller tydlig väg dit", "unknown", "FCF saknas")
    elif fcf > 0:
        add("Positivt FCF eller tydlig väg dit", "ok", f"FCF {fcf:,.0f} M")
    else:
        add("Positivt FCF eller tydlig väg dit", "unknown",
            f"FCF {fcf:,.0f} M — bedöm vägen dit själv")

    rg = _f(snap.get("revenue_growth"))
    if rg is None:
        add("Ej strukturellt fallande omsättning", "unknown",
            "tillväxt-KPI:n ligger inte i licensen — kolla omsättningstrenden")
    elif rg >= 0:
        add("Ej strukturellt fallande omsättning", "ok", f"omsättning {rg * 100:+.0f} %")
    else:
        add("Ej strukturellt fallande omsättning", "unknown",
            f"omsättning {rg * 100:+.0f} % — strukturellt eller cykliskt? Din bedömning")

    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        return "Nej", checks
    if statuses == {"ok"}:
        return "Ja", checks
    return "", checks


def auto_trigger(closes: list, cluster_avg) -> tuple:
    """(trigger, notis). A = stabilisering, B = bekräftelse, Nej = ingen.
    C (rapport) sätts aldrig automatiskt."""
    if len(closes) < 26:
        return "", "för lite kurshistorik"
    last = closes[-1]
    ma20 = sum(closes[-20:]) / 20.0
    ma20_prev = sum(closes[-25:-5]) / 20.0
    ret_1m = last / closes[-22] - 1 if closes[-22] else 0.0
    if last > ma20 and ma20 >= ma20_prev:
        return "A", (f"stängning {last:.2f} > MA20 {ma20:.2f}, MA20 planat/vänt "
                     f"(1 mån {ret_1m * 100:+.1f} %)")
    if ret_1m > 0 and cluster_avg and last > float(cluster_avg):
        return "B", (f"1 mån {ret_1m * 100:+.1f} %, kurs {last:.2f} över "
                     f"klustersnitt {float(cluster_avg):.2f}")
    return "Nej", (f"stängning {last:.2f} vs MA20 {ma20:.2f}, 1 mån "
                   f"{ret_1m * 100:+.1f} %")


def drawdown_52w(closes: list) -> Optional[float]:
    if not closes:
        return None
    hi = max(closes[-252:])
    return round((hi - closes[-1]) / hi * 100, 1) if hi > 0 else None


def auto_efter_fall(dd, f_score) -> tuple:
    """Arket: 'Kursen har fallit mer än 20 % och F-score ≥ 5'. Båda krävs."""
    if dd is None:
        return False, "kursdata saknas"
    if dd < FALL_PCT:
        return False, f"{dd:.0f} % under 52v-topp (< {FALL_PCT:g} %)"
    if f_score is None:
        return False, (f"{dd:.0f} % under 52v-topp men F-score okänd — kryssa "
                       f"själv om bolaget är friskt (F-score ≥ 5)")
    ok = float(f_score) >= FSCORE_MIN
    return ok, f"{dd:.0f} % under 52v-topp, F-score {float(f_score):g}"


# ── Hela skanningen ──────────────────────────────────────────────────────────
def _closes(api, ins_id: int) -> list:
    try:
        raw = api.get_stockprices(ins_id, max_count=260)
    except Exception as e:
        log.debug("stockprices %s: %s", ins_id, e)
        return []
    bars = sorted((b for b in raw if b.get("c") is not None), key=lambda b: b.get("d") or "")
    return [float(b["c"]) for b in bars]


def scan(api, today: Optional[date] = None) -> dict:
    import insider as ins   # arkets rena funktioner — samma regler

    out = {"generated": _now(), "universe": 0, "with_buys": 0,
           "clusters": [], "error": None}
    today = today or date.today()
    try:
        instruments = api.get_instruments()
        nordic = [i for i in instruments
                  if i.get("marketId") in _MARKET_SUFFIX
                  and i.get("marketId") not in _INDEX_MARKETS]
        by_id = {int(i["insId"]): i for i in nordic if i.get("insId") is not None}
        out["universe"] = len(by_id)
        log.info("Universum: %d nordiska instrument", len(by_id))

        tx = api.get_insider_transactions_batch(list(by_id))
        log.info("Insynsdata för %d instrument", len(tx))

        prelim = []
        for iid, values in tx.items():
            inst = by_id.get(iid)
            if not inst:
                continue
            ccy = str(inst.get("stockPriceCurrency") or "SEK").upper()
            fx = FX_TO_SEK.get(ccy, 1.0)
            cl = build_cluster(values, today, fx)
            if cl is None:
                continue
            out["with_buys"] += 1
            sig = {
                "ticker": yf_ticker(inst), "name": str(inst.get("name") or ""),
                "ins_id": iid, "currency": ccy, "found": today.isoformat(),
                "insiders": cl["insiders"], "role": cl["role"], "amount": cl["amount"],
                "okar_25": False, "efter_fall": False,
                "aterkommande": cl["aterkommande"],
                "cluster_avg": cl["cluster_avg"], "price_now": None,
                "gate": "", "trigger": "", "comment": "",
            }
            pre = ins.score(sig)
            if pre is None or pre < ENRICH_MIN_PRELIM:
                continue
            prelim.append((sig, cl, fx))
        log.info("%d kluster med köp, %d med prelim-poäng >= %d",
                 out["with_buys"], len(prelim), ENRICH_MIN_PRELIM)

        snaps = {}
        if prelim:
            try:
                snaps = api.get_fundamentals_snapshot_fast([s["ins_id"] for s, _, _ in prelim])
            except Exception as e:
                log.warning("fundamenta-batchen föll: %s", e)

        for sig, cl, fx in prelim:
            snap = snaps.get(sig["ins_id"], {}) or {}
            closes = _closes(api, sig["ins_id"])
            gate, checks = auto_gate(snap, fx)
            trig, tnote = auto_trigger(closes, sig["cluster_avg"])
            dd = drawdown_52w(closes)
            ef, efnote = auto_efter_fall(dd, _f(snap.get("f_score")))
            sig.update(price_now=round(closes[-1], 4) if closes else None,
                       gate=gate, trigger=trig, efter_fall=ef)
            sc = ins.score(sig)
            if sc is None or sc < SAVE_MIN_SCORE:
                continue
            vs = ins.vs_cluster(sig)
            stop = ins.stop_price(sig)
            sig.update({
                "score": sc, "status": ins.status(sig),
                "vs_cluster": round(vs, 1) if vs is not None else None,
                "stop": round(stop, 2) if stop is not None else None,
                "chase": ins.is_chase(sig),
                "auto": {
                    "gate_checks": checks, "trigger_note": tnote,
                    "efter_fall_note": efnote,
                    "okar_25_note": "kräver köparens tidigare innehav — kolla insynsfliken",
                    "drawdown_52w": dd, "amount_local": cl["amount_local"],
                    "first_buy": cl["first_buy"], "last_buy": cl["last_buy"],
                    "buys": cl["buys"][:12],
                },
            })
            out["clusters"].append(sig)
        out["clusters"].sort(key=lambda s: (-s["score"], s["ticker"]))
    except Exception as e:
        import traceback
        log.error("Insider-skanningen felade:\n%s", traceback.format_exc())
        out["error"] = str(e)
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
    api = BorsdataAPI(api_key=key)

    result = scan(api)
    log.info("Insider: %d kluster >= %d p av %d med köp%s",
             len(result["clusters"]), SAVE_MIN_SCORE, result["with_buys"],
             f" (FEL: {result['error']})" if result["error"] else "")
    for s in result["clusters"][:15]:
        log.info("  %-12s %-24s %2d p  %-28s %d ins · %s · %s tkr · grind %s · trig %s",
                 s["ticker"], s["name"][:24], s["score"], s["status"][:28],
                 s["insiders"], s["role"], f"{s['amount']:,.0f}",
                 s["gate"] or "—", s["trigger"] or "—")

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
