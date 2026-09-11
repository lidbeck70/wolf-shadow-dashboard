#!/usr/bin/env python3
"""gist_probe.py — diagnostik: vad ligger i Gisten och hur laddar panelen det?

Skriver ut varje fil (storlek, truncated-flagga, content-längd) för både
en oautentiserad läsning (panelens väg i gist_storage.load_blob) och en
autentiserad (GITHUB_TOKEN). Sedan json-parsning av de blobbar arken läser.
Ändrar ingenting."""
import json
import os
import sys

import requests

import gist_storage as g

BLOBS = ("screens.json", "insider_scan.json", "sheets_refresh.json",
         "arc_screeners.json", "alert_state.json")


def _dump(label, r):
    print(f"\n== {label}: HTTP {r.status_code} · ratelimit "
          f"{r.headers.get('x-ratelimit-remaining')}/{r.headers.get('x-ratelimit-limit')}")
    if r.status_code != 200:
        print("  ", r.text[:300])
        return {}
    files = r.json().get("files", {})
    total = 0
    for name, f in sorted(files.items()):
        n = len(f.get("content") or "")
        total += f.get("size") or 0
        print(f"  {name:32} size={f.get('size'):>9} truncated={str(f.get('truncated')):5} "
              f"content_len={n:>9} raw_url={'ja' if f.get('raw_url') else 'nej'}")
    print(f"  totalt {total:,} byte i {len(files)} filer")
    return files


def main() -> int:
    anon = _dump("oautentiserad (panelens väg)", requests.get(g.GIST_API_URL, timeout=20))
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    auth = {}
    if token:
        auth = _dump("autentiserad", requests.get(
            g.GIST_API_URL, headers={"Authorization": f"token {token}"}, timeout=20))

    print("\n== json-parsning (autentiserat svar om det finns, annars oautentiserat)")
    files = auth or anon
    for name in BLOBS:
        f = files.get(name)
        if not f:
            print(f"  {name:22} SAKNAS i Gisten")
            continue
        content = f.get("content") or ""
        if f.get("truncated") and f.get("raw_url"):
            rr = requests.get(f["raw_url"], timeout=30)
            print(f"  {name:22} truncated → raw_url HTTP {rr.status_code}, {len(rr.content):,} byte")
            content = rr.text if rr.status_code == 200 else content
        try:
            d = json.loads(content)
        except Exception as e:
            print(f"  {name:22} JSON-FEL: {e}")
            continue
        if name == "screens.json":
            print(f"  {name:22} generated={d.get('generated')} global={d.get('global_available')} "
                  f"rows={ {k: len(v.get('rows') or []) for k, v in (d.get('screens') or {}).items()} }")
        elif name == "insider_scan.json":
            print(f"  {name:22} generated={d.get('generated')} clusters={len(d.get('clusters') or [])}")
        elif name == "sheets_refresh.json":
            print(f"  {name:22} generated={d.get('generated')} rows={len(d.get('rows') or {})} "
                  f"events={len(d.get('events') or [])}")
        else:
            print(f"  {name:22} nycklar={list(d)[:8]}")

    print("\n== gist_storage.load_blob (exakt panelens kod)")
    for name in BLOBS[:3]:
        b = g.load_blob(name, None)
        print(f"  {name:22} → {type(b).__name__}"
              + (f" med nycklar {list(b)[:6]}" if isinstance(b, dict) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
