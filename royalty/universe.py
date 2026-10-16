"""
Royalty-universumet: den kurerade listan (config/universes/royalty.csv — royalty-, streaming- och trustbolag i
Yahoo-form med råvarutema i BERSERK-vokabulär) plus Royalty-håvens träffar vars namn säger royalty, streaming
eller trust (screens.json, noten "namnet säger royalty/streaming"). Stora producenter som passerar håvens
marginalfilter hör till Rick Rule och tas inte med. Börsdata-id, marknad och tema fylls ur universum-registret
när de saknas. Rena funktioner — ingen nätverksåtkomst.
"""
from __future__ import annotations

import csv
import os
from typing import Optional

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "universes", "royalty.csv")
SOURCE_LIST, SOURCE_SCREEN = "Royalty-listan", "Royalty-håven"
NAME_NOTE = "namnet säger royalty/streaming"          # screens_scan sätter noten på Royalty-håvens rader
KIND_SCREEN = "håven"
KINDS = ("royalty", "streaming", "trust", KIND_SCREEN)
_COUNTRY = {"canada": "CA", "usa": "US", "australia": "AU", "nordic": "SE"}


def load_csv(path: Optional[str] = None) -> list:
    """Den kurerade listan: {ticker, name, kind, theme, country, notes, source}. Tomma teman blir None."""
    out = []
    with open(path or CSV_PATH, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            t = str(r.get("ticker") or "").strip().upper()
            if not t:
                continue
            out.append({"ticker": t, "name": str(r.get("name") or "").strip() or t,
                        "kind": str(r.get("kind") or "royalty").strip().lower(),
                        "theme": (str(r.get("theme") or "").strip().lower() or None),
                        "country": str(r.get("country") or "").strip().upper() or None,
                        "notes": [str(r.get("notes") or "").strip()] if str(r.get("notes") or "").strip() else [],
                        "source": SOURCE_LIST, "ins_id": None, "market": None, "m": {}})
    return out


def from_screens(blob: Optional[dict]) -> list:
    """Royalty-håvens träffar med noten — bara de vars namn säger royalty/streaming/trust (producenterna är
    Rick Rules). Behåller håvens nyckeltal (m), Börsdata-id och börsvärde."""
    out = []
    rows = (((blob or {}).get("screens") or {}).get("royalty") or {}).get("rows") or []
    for r in rows:
        if not isinstance(r, dict) or NAME_NOTE not in (r.get("notes") or []):
            continue
        t = str(r.get("ticker") or "").strip().upper()
        if not t:
            continue
        out.append({"ticker": t, "name": str(r.get("name") or "").strip() or t, "kind": KIND_SCREEN, "theme": None,
                    "country": _COUNTRY.get(str(r.get("universe") or "").lower()), "notes": [],
                    "source": SOURCE_SCREEN, "ins_id": r.get("ins_id"), "market": None,
                    "m": dict(r.get("m") or {}, mcap_musd=r.get("mcap_musd"))})
    return out


def build(csv_rows: Optional[list] = None, screens: Optional[dict] = None, register=None) -> list:
    """Listan först, håvens träffar fyller på (nyckeltal och Börsdata-id på listans rader, nya bolag efter).
    register = universe.registry.Universe (eller något med .get(ticker)) som fyller ins_id, marknad och tema."""
    rows = [dict(r) for r in (csv_rows if csv_rows is not None else load_csv())]
    by = {r["ticker"]: r for r in rows}
    for s in from_screens(screens):
        r = by.get(s["ticker"])
        if r is None:
            rows.append(s)
            by[s["ticker"]] = s
            continue
        r["m"] = dict(s["m"], **{k: v for k, v in (r.get("m") or {}).items() if v is not None})
        r["ins_id"] = r.get("ins_id") if r.get("ins_id") is not None else s.get("ins_id")
        r["source"] = f"{SOURCE_LIST} + {SOURCE_SCREEN}"
    if register is not None:
        for r in rows:
            try:
                m = register.get(r["ticker"])
            except Exception:
                m = None
            if m is None:
                continue
            if r.get("ins_id") is None:
                r["ins_id"] = getattr(m, "ins_id", None)
            r["market"] = r.get("market") or getattr(m, "market", None)
            r["theme"] = r.get("theme") or (getattr(m, "theme", None) or None)
            if not r.get("name") or r["name"] == r["ticker"]:
                r["name"] = getattr(m, "name", None) or r["name"]
    return rows


def describe(rows: list) -> str:
    """'27 bolag · listan 24 · håven 3 · med Börsdata-id 25'."""
    n_list = sum(1 for r in rows if SOURCE_LIST in str(r.get("source")))
    n_scr = sum(1 for r in rows if r.get("source") == SOURCE_SCREEN)
    n_id = sum(1 for r in rows if r.get("ins_id") is not None)
    return f"{len(rows)} bolag · listan {n_list} · håven {n_scr} · med Börsdata-id {n_id}"
