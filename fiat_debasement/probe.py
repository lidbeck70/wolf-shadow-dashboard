#!/usr/bin/env python3
"""
fiat_debasement/probe.py — datasonden för 🐺 Fiat Debasement (PR 1).

Provar varje kandidatserie för SEK, EUR och USD — penningmängd, KPI,
kärn-KPI, real BNP, statsskuld/BNP och växelkurs — samt guld, silver,
koppar, olja och bitcoin. Skriver för varje serie: status, första och
sista datum, senaste värde, frekvens och om serien verkar nedlagd (sista
datum för gammalt för frekvensen). Kandidaterna står i preferensordning:
primärkällan (centralbank/statistikmyndighet) först, reserver efter.

SCB-tabellernas sökvägar ändras ibland — därför listar sonden även vilka
tabeller som finns under SCB:s mappar för penningmängd, KPI och BNP.

Körs i GitHub Actions (.github/workflows/fiat-probe.yml). Skriver
ingenting, ändrar ingenting, exit 0 alltid. Resultatet hamnar i loggen och
som tabell på körningens sammanfattningssida.

    python -m fiat_debasement.probe
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

import pandas as pd

if __package__ in (None, ""):                                   # python fiat_debasement/probe.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiat_debasement import sources as src  # noqa: E402

# Hur gammalt sista datum får vara innan serien flaggas som nedlagd/eftersläpande (dagar)
STALE_DAYS = {"D": 10, "W": 21, "M": 100, "Q": 220, "A": 640}

M2, CPI, CORE, GDP, DEBT, FX = ("Penningmängd", "KPI", "Kärn-KPI", "Real BNP", "Statsskuld/BNP", "Växelkurs")
GOLD, SILVER, COPPER, OIL, BTC = "Guld", "Silver", "Koppar", "Olja", "Bitcoin"

# SCB-mappar att lista (sökväg, nyckelord i tabellnamnet)
SCB_DISCOVERY = (
    ("FM", ("penningm", "m1", "m2", "m3")),
    ("PR/PR0101", ("kpi", "konsumentprisindex")),
    ("NR/NR0103", ("bnp", "bruttonationalprodukt")),
)


def candidates() -> list:
    """[(begrepp, valuta, hämtfunktion)] i preferensordning per (begrepp, valuta)."""
    f, e, es, y, b = src.fred, src.ecb, src.eurostat, src.yahoo, src.borsdata
    return [
        # ── USD ──
        (M2, "USD", lambda: f("M2SL", unit="mdr USD, säsongsjusterad", currency="USD", label="M2")),
        (M2, "USD", lambda: f("M2NS", unit="mdr USD", currency="USD", label="M2 ej säsongsjusterad")),
        (CPI, "USD", lambda: f("CPIAUCSL", unit="index 1982–84=100", currency="USD", label="CPI-U")),
        (CPI, "USD", lambda: f("CPIAUCNS", unit="index 1982–84=100", currency="USD", label="CPI-U ej s.j.")),
        (CORE, "USD", lambda: f("CPILFESL", unit="index", currency="USD", label="CPI exkl. livsmedel och energi")),
        (GDP, "USD", lambda: f("GDPC1", unit="mdr kedjade 2017 USD", currency="USD", label="Real BNP")),
        (DEBT, "USD", lambda: f("GGGDTAUSA188N", unit="% av BNP", currency="USD",
                                label="Offentlig bruttoskuld (IMF, hela offentliga sektorn)")),
        (DEBT, "USD", lambda: f("GFDEGDQ188S", unit="% av BNP", currency="USD", label="Federal skuld")),
        # ── EUR ──
        (M2, "EUR", lambda: e("BSI", "M.U2.Y.V.M20.X.1.U2.2300.Z01.E", unit="mn EUR", currency="EUR",
                              label="M2 euroområdet, stock")),
        (M2, "EUR", lambda: f("MYAGM2EZM196N", unit="EUR", currency="EUR", label="M2 euroområdet (IMF)")),
        (CPI, "EUR", lambda: e("ICP", "M.U2.N.000000.4.INX", unit="index 2015=100", currency="EUR",
                               label="HICP totalt")),
        (CPI, "EUR", lambda: es("prc_hicp_midx", {"geo": "EA20", "coicop": "CP00", "unit": "I15"},
                                unit="index 2015=100", currency="EUR", label="HICP EA20")),
        (CPI, "EUR", lambda: f("CP0000EZ19M086NEST", unit="index 2015=100", currency="EUR", label="HICP EA19")),
        (CORE, "EUR", lambda: e("ICP", "M.U2.N.XEF000.4.INX", unit="index 2015=100", currency="EUR",
                                label="HICP exkl. energi, livsmedel, alkohol, tobak")),
        (GDP, "EUR", lambda: es("namq_10_gdp", {"geo": "EA20", "unit": "CLV10_MEUR", "s_adj": "SCA",
                                                "na_item": "B1GQ"},
                                unit="mn kedjade 2010 EUR", currency="EUR", label="Real BNP EA20")),
        (GDP, "EUR", lambda: f("CLVMNACSCAB1GQEA19", unit="mn kedjade 2010 EUR", currency="EUR",
                               label="Real BNP EA19")),
        (DEBT, "EUR", lambda: es("gov_10q_ggdebt", {"geo": "EA20", "unit": "PC_GDP", "sector": "S13",
                                                    "na_item": "GD"},
                                 unit="% av BNP", currency="EUR", label="Offentlig bruttoskuld (Maastricht)")),
        (DEBT, "EUR", lambda: f("GGGDTAXMA188N", unit="% av BNP", currency="EUR", label="Offentlig bruttoskuld (IMF)")),
        (FX, "EUR", lambda: e("EXR", "D.USD.EUR.SP00.A", unit="USD per EUR", currency="EUR", label="EUR/USD")),
        (FX, "EUR", lambda: f("DEXUSEU", unit="USD per EUR", currency="EUR", label="EUR/USD")),
        (FX, "EUR", lambda: y("EURUSD=X", unit="USD per EUR", currency="EUR", label="EUR/USD")),
        # ── SEK ──
        (M2, "SEK", lambda: f("MABMM301SEM189S", unit="SEK", currency="SEK", label="M3 Sverige (OECD)")),
        (M2, "SEK", lambda: f("MYAGM2SEM052N", unit="SEK", currency="SEK", label="M2 Sverige (IMF)")),
        (CPI, "SEK", lambda: src.scb_table("PR/PR0101/PR0101A/KPItotM", unit="index 1980=100", currency="SEK",
                                           label="KPI fastställda tal")),
        (CPI, "SEK", lambda: es("prc_hicp_midx", {"geo": "SE", "coicop": "CP00", "unit": "I15"},
                                unit="index 2015=100", currency="SEK", label="HICP Sverige")),
        (CPI, "SEK", lambda: f("CP0000SEM086NEST", unit="index 2015=100", currency="SEK", label="HICP Sverige")),
        (GDP, "SEK", lambda: es("namq_10_gdp", {"geo": "SE", "unit": "CLV10_MNAC", "s_adj": "SCA",
                                                "na_item": "B1GQ"},
                                unit="mn kedjade 2010 SEK", currency="SEK", label="Real BNP Sverige")),
        (GDP, "SEK", lambda: f("CLVMNACSCAB1GQSE", unit="mn kedjade 2010 SEK", currency="SEK",
                               label="Real BNP Sverige")),
        (DEBT, "SEK", lambda: es("gov_10q_ggdebt", {"geo": "SE", "unit": "PC_GDP", "sector": "S13",
                                                    "na_item": "GD"},
                                 unit="% av BNP", currency="SEK", label="Offentlig bruttoskuld (Maastricht)")),
        (DEBT, "SEK", lambda: f("GGGDTASEA188N", unit="% av BNP", currency="SEK", label="Offentlig bruttoskuld (IMF)")),
        (FX, "SEK", lambda: src.riksbank("SEKUSDPMI", unit="SEK per USD", currency="SEK", label="USD/SEK")),
        (FX, "SEK", lambda: src.riksbank("SEKEURPMI", unit="SEK per EUR", currency="SEK", label="EUR/SEK")),
        (FX, "SEK", lambda: e("EXR", "D.SEK.EUR.SP00.A", unit="SEK per EUR", currency="SEK", label="EUR/SEK")),
        (FX, "SEK", lambda: f("DEXSDUS", unit="SEK per USD", currency="SEK", label="USD/SEK")),
        (FX, "SEK", lambda: y("SEK=X", unit="SEK per USD", currency="SEK", label="USD/SEK")),
        # ── Reala tillgångar (USD) ──
        (GOLD, "USD", lambda: b(21031, unit="USD/oz", currency="USD", label="Guld")),
        (GOLD, "USD", lambda: y("GC=F", unit="USD/oz", currency="USD", label="Guld terminer")),
        (GOLD, "USD", lambda: f("GOLDAMGBD228NLBM", unit="USD/oz", currency="USD", label="LBMA AM (troligen nedlagd)")),
        (SILVER, "USD", lambda: b(21032, unit="USD/oz", currency="USD", label="Silver")),
        (SILVER, "USD", lambda: y("SI=F", unit="USD/oz", currency="USD", label="Silver terminer")),
        (COPPER, "USD", lambda: b(21035, unit="USD", currency="USD", label="Koppar")),
        (COPPER, "USD", lambda: y("HG=F", unit="USD/lb", currency="USD", label="Koppar terminer")),
        (COPPER, "USD", lambda: f("PCOPPUSDM", unit="USD/ton", currency="USD", label="Koppar månad (IMF)")),
        (OIL, "USD", lambda: b(21046, unit="USD/fat", currency="USD", label="Brent")),
        (OIL, "USD", lambda: y("BZ=F", unit="USD/fat", currency="USD", label="Brent terminer")),
        (OIL, "USD", lambda: f("POILBREUSDM", unit="USD/fat", currency="USD", label="Brent månad (IMF)")),
        (BTC, "USD", lambda: y("BTC-USD", unit="USD", currency="USD", label="Bitcoin")),
    ]


def stale(sd: src.SeriesData, today: Optional[pd.Timestamp] = None) -> bool:
    """Sista datum äldre än vad frekvensen tillåter (nedlagd eller kraftigt eftersläpande)."""
    if not sd.ok or not sd.frequency:
        return False
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    return (today - sd.values.index[-1]).days > STALE_DAYS.get(sd.frequency, 640)


def row_of(concept: str, currency: str, sd: src.SeriesData, today=None) -> dict:
    status = "OK" if sd.ok else "FEL"
    if sd.ok and stale(sd, today):
        status = "GAMMAL"
    return {"concept": concept, "currency": currency, "source": sd.source, "series": sd.series_id,
            "label": sd.label, "unit": sd.unit, "status": status, "first": sd.first, "last": sd.last,
            "last_value": sd.last_value, "frequency": sd.frequency, "rows": len(sd.values) if sd.ok else 0,
            "error": sd.error, "chosen": (sd.meta or {}).get("chosen")}


def chosen(rows: list) -> dict:
    """(begrepp, valuta) → första raden med status OK (preferensordningen), annars None."""
    out = {}
    for r in rows:
        key = (r["concept"], r["currency"])
        out.setdefault(key, None)
        if out[key] is None and r["status"] == "OK":
            out[key] = r
    return out


def _fmt_val(v) -> str:
    if v is None:
        return "—"
    return f"{v:,.4g}" if abs(v) < 1e5 else f"{v:,.0f}"


def markdown(rows: list, discovery: list) -> str:
    lines = ["## 🐺 Fiat Debasement — datasond", "",
             "| Begrepp | Valuta | Källa | Serie | Status | Från | Till | Senast | Frekv | Rader |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['concept']} | {r['currency']} | {r['source']} | `{r['series']}` | {r['status']} | "
                     f"{r['first'] or '—'} | {r['last'] or '—'} | {_fmt_val(r['last_value'])} | "
                     f"{r['frequency'] or '—'} | {r['rows']} |")
    lines += ["", "### Vald källa per begrepp (första som fungerar)", "",
              "| Begrepp | Valuta | Källa | Serie | Från | Till |", "|---|---|---|---|---|---|"]
    missing = []
    for (concept, cur), r in chosen(rows).items():
        if r is None:
            missing.append(f"{concept} {cur}")
            lines.append(f"| {concept} | {cur} | **DATA UNAVAILABLE** | — | — | — |")
        else:
            lines.append(f"| {concept} | {cur} | {r['source']} | `{r['series']}` | {r['first']} | {r['last']} |")
    if missing:
        lines += ["", "**Saknas:** " + ", ".join(missing)]
    errors = [r for r in rows if r["status"] != "OK" and r["error"]]
    if errors:
        lines += ["", "### Fel", ""] + [f"- {r['source']} `{r['series']}`: {r['error']}" for r in errors]
    if discovery:
        lines += ["", "### SCB-tabeller", ""] + [f"- `{p}` — {t}" for p, t in discovery]
    return "\n".join(lines) + "\n"


def discover_scb(lister: Callable = src.scb_list, max_depth: int = 3) -> list:
    """[(sökväg, tabellnamn)] för SCB-tabeller vars namn matchar nyckelorden."""
    found = []

    def walk(path, words, depth):
        for item_id, kind, text in lister(path):
            sub = f"{path}/{item_id}"
            if kind == "t" and any(w in str(text).lower() for w in words):
                found.append((sub, text))
            elif kind == "l" and depth < max_depth:
                walk(sub, words, depth + 1)

    for root, words in SCB_DISCOVERY:
        walk(root, words, 1)
    return found


def run(cands: Optional[list] = None, lister: Optional[Callable] = None, out=print, today=None) -> tuple:
    rows = []
    for concept, cur, fetch in (cands if cands is not None else candidates()):
        try:
            sd = fetch()
        except Exception as exc:                                    # sonden får aldrig krascha
            sd = src.SeriesData("?", "?", error=f"{type(exc).__name__}: {exc}")
        r = row_of(concept, cur, sd, today)
        rows.append(r)
        out(f"{r['status']:6} {concept:15} {cur:3} {r['source']:10} {r['series'][:48]:48} "
            f"{r['first'] or '—':10} → {r['last'] or '—':10} {r['frequency'] or '-':1} {r['rows']:6} rader"
            + (f"  [{r['error']}]" if r["error"] else "")
            + (f"  val: {r['chosen']}" if r.get("chosen") else ""))
    discovery = discover_scb(lister or src.scb_list)
    for p, t in discovery:
        out(f"SCB-tabell {p} — {t}")
    return rows, discovery


def main() -> int:
    rows, discovery = run()
    md = markdown(rows, discovery)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(md)
        except OSError as exc:
            print(f"Kunde inte skriva sammanfattningen: {exc}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
