#!/usr/bin/env python3
"""
berserk/probe.py — datasonden för 🪓 BERSERK (PR 0).

Provar hos Yahoo (samma väg som panelen: market_prices.ohlcv, period "max"):
  1. varje temas drivare (råvaruterminer och reserv-ETF:er), i preferensordning
  2. alla råvaru-ETF:er i universumet
  3. regionernas index (marknadsgrinden) och alla producentbolag per region

Skriver status, första och sista datum, antal rader och senaste kurs, vilken
drivare som väljs per tema (första som fungerar och inte är gammal) och hur
många år historik den har — avgör vilka teman som kan backtestas 2008–2020.

Körs i GitHub Actions (.github/workflows/berserk-probe.yml). Ändrar ingenting,
exit 0 alltid. Resultatet hamnar i loggen och på körningens sammanfattning.

    python -m berserk.probe
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

import pandas as pd

if __package__ in (None, ""):                                   # python berserk/probe.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import themes as th  # noqa: E402
from berserk import universe as uv  # noqa: E402
from fiat_debasement import sources as src  # noqa: E402

STALE_DAYS = 10                    # dagsdata: sista kurs äldre än så = GAMMAL (nedlagd/omdöpt)
OOS_START = "2008-01-01"           # out-of-sample-perioden börjar här


def _fetch(symbol: str, getter: Optional[Callable] = None) -> src.SeriesData:
    try:
        return src.yahoo(symbol, getter=getter)
    except Exception as exc:                                    # sonden får aldrig krascha
        return src.SeriesData("Yahoo", symbol, error=f"{type(exc).__name__}: {exc}")


def row_of(group: str, theme: str, symbol: str, sd: src.SeriesData, today=None) -> dict:
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    status = "OK" if sd.ok else "FEL"
    if sd.ok and (today - sd.values.index[-1]).days > STALE_DAYS:
        status = "GAMMAL"
    years = round((sd.values.index[-1] - sd.values.index[0]).days / 365.25, 1) if sd.ok else 0.0
    return {"group": group, "theme": theme, "label": th.label(theme) if theme else "", "symbol": symbol,
            "status": status, "first": sd.first, "last": sd.last, "years": years,
            "rows": len(sd.values) if sd.ok else 0, "last_value": sd.last_value,
            "oos": bool(sd.ok and sd.first and sd.first <= OOS_START), "error": sd.error}


def candidates() -> list:
    """[(grupp, tema, symbol)] — drivare per tema i preferensordning, sedan ETF:er och producenter."""
    out = []
    for theme in list(th.THEMES) + list(th.BASKETS):
        for s in th.drivers(theme):
            out.append(("Drivare", theme, s))
    for region, sym in uv.REGION_INDEX.items():
        if region != "Norden":                                  # OMXS30 kommer från Börsdata
            out.append(("Index", "", sym))
    for s, theme in uv.ETFS.items():
        out.append(("ETF", theme, s))
    for region, members in uv.REGIONS.items():
        for s, theme in members.items():
            out.append((region, theme, s))
    return out


def chosen_drivers(rows: list) -> dict:
    """tema → första drivaren med status OK (preferensordningen), annars None."""
    out = {theme: None for theme in list(th.THEMES) + list(th.BASKETS)}
    for r in rows:
        if r["group"] == "Drivare" and out.get(r["theme"]) is None and r["status"] == "OK":
            out[r["theme"]] = r
    return out


def run(cands: Optional[list] = None, getter: Optional[Callable] = None, out=print, today=None) -> list:
    rows, cache = [], {}
    for group, theme, symbol in (cands if cands is not None else candidates()):
        if symbol not in cache:
            cache[symbol] = _fetch(symbol, getter)
        r = row_of(group, theme, symbol, cache[symbol], today)
        rows.append(r)
        out(f"{r['status']:6} {group:22} {theme:12} {symbol:12} {r['first'] or '—':10} → {r['last'] or '—':10} "
            f"{r['years']:5.1f} år {r['rows']:6} rader" + (f"  [{r['error']}]" if r["error"] else ""))
    return rows


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:,.4g}" if abs(v) < 1e5 else f"{v:,.0f}")


def markdown(rows: list) -> str:
    lines = ["## 🪓 BERSERK — datasond", "", "### Vald drivare per tema", "",
             "| Komplex | Tema | Drivare | Från | Till | År | Från 2008? |", "|---|---|---|---|---|---|---|"]
    missing = []
    for theme, r in chosen_drivers(rows).items():
        cx = th.COMPLEXES.get(th.complex_of(theme), "")
        if not th.drivers(theme):
            lines.append(f"| {cx} | {th.label(theme)} | ingen prisserie (aktiens egen kurva) | — | — | — | — |")
        elif r is None:
            missing.append(th.label(theme))
            lines.append(f"| {cx} | {th.label(theme)} | **DATA UNAVAILABLE** | — | — | — | — |")
        else:
            lines.append(f"| {cx} | {th.label(theme)} | `{r['symbol']}` | {r['first']} | {r['last']} | {r['years']} | "
                         f"{'ja' if r['oos'] else 'nej'} |")
    if missing:
        lines += ["", "**Drivare saknas:** " + ", ".join(missing)]
    for group in ("Drivare", "Index", "ETF", *uv.REGIONS):
        part = [r for r in rows if r["group"] == group]
        if not part:
            continue
        ok = sum(1 for r in part if r["status"] == "OK")
        lines += ["", f"### {group} — {ok} av {len(part)} OK", "",
                  "| Symbol | Tema | Status | Från | Till | År | Senast |", "|---|---|---|---|---|---|---|"]
        for r in part:
            lines.append(f"| `{r['symbol']}` | {r['label']} | {r['status']} | {r['first'] or '—'} | {r['last'] or '—'} | "
                         f"{r['years']} | {_fmt(r['last_value'])} |")
    bad = [r for r in rows if r["status"] != "OK"]
    if bad:
        lines += ["", "### Saknas eller gammal", ""] + [f"- `{r['symbol']}` ({r['group']}, {r['label']}): "
                                                         f"{r['status']}{' — ' + r['error'] if r['error'] else ''}"
                                                         for r in bad]
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = run()
    md = markdown(rows)
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
