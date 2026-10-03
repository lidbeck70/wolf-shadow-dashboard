#!/usr/bin/env python3
"""
fiat_debasement/probe.py — datasonden för 🐺 Fiat Debasement (PR 1).

Provar exakt de serier modulen använder (config.SERIES) för SEK, EUR och
USD — penningmängd, KPI, kärn-KPI, real BNP, statsskuld/BNP och växelkurs —
samt guld, silver, koppar, olja och bitcoin. Skriver för varje serie: status, första och
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

from fiat_debasement import config as cfg  # noqa: E402
from fiat_debasement import data as fd  # noqa: E402
from fiat_debasement import sources as src  # noqa: E402

STALE_DAYS = cfg.STALE_DAYS

# SCB-mappar att lista (sökväg, nyckelord i tabellnamnet)
SCB_DISCOVERY = (
    ("FM", ("penningm", "m1", "m2", "m3")),
    ("PR/PR0101", ("kpi", "konsumentprisindex")),
    ("NR/NR0103", ("bnp", "bruttonationalprodukt")),
)


def candidates() -> list:
    """[(begrepp, valuta, hämtfunktion)] — exakt de källor modulen använder (config.SERIES
    och guld/silver-skarven), i preferensordning per (begrepp, valuta)."""
    out = []
    for (concept, cur), specs in cfg.SERIES.items():
        for spec in specs:
            out.append((cfg.CONCEPT_LABEL[concept], cur, lambda spec=spec: fd.fetch(spec)))
    for name, conf in cfg.ASSET_SPLICE.items():
        for spec in (conf["primary"], conf["backfill"]):
            out.append((cfg.CONCEPT_LABEL[name], "USD", lambda spec=spec: fd.fetch(spec)))
    return out


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
