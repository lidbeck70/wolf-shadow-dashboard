"""
valhall/turns.py — händelsemotorn (K5): på bekräftad vändning i en ruta — 2 Deep Contrarian + 2 lång + 1 UCITS-ETF.

Analysen 2026-10-24 §6.1 (beslut 9–10). Händelsen är kartans turns-lista (valhall.board.turns): en ruta vars
vändning är BEKRÄFTAD och högst WINDOW_DAYS handelsdagar gammal är i köpfönstret. För varje sådan ruta väljs:

  Deep Contrarian   raderna ur körningen deep_contrarian som hör till rutan (segment via Börsdata-sektorn, komplex
                    via registrets tema), rankade på composite — deep saknar Alpha-verdikt, rankningen är
                    composite + motorns egna grindar (unloved guard, balansräkning). Två bästa, ett aktieslag per bolag.
  Lång              Long Screener STRONG BUY/BUY i rutan (kartans long.segments), annars Quality Contrarian BUY
                    (kartans buys.segments), rankade som i dag. Två bästa.
  ETF               rutans bästa UCITS-exponering (valhall.etf.pick) — en enhet; verifierad linje först.

Utanför fönstret görs inget — det är skillnaden mot dagens tillståndsstyrda köp. Papperet (K7) läser planen som
signaler i kontona komplex_deep / komplex_long / komplex_etf; dubbletter stoppas av kontots karantän.
Riktiga order läggs alltid manuellt.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from valhall import complexes as vcx
from valhall import picks as vp

WINDOW_DAYS = 5          # handelsdagar efter bekräftelsen som rutan ligger i köpfönstret
N_DEEP, N_LONG = 2, 2
BUY_SIGNALS = ("STRONG BUY", "BUY")


def in_window(turn: dict, today=None, window: int = WINDOW_DAYS) -> bool:
    """Bekräftad och högst window handelsdagar sedan bekräftelsen (eller sedan vändningen om datum saknas)."""
    if not turn or not turn.get("confirmed"):
        return False
    since = turn.get("confirmed_since") or turn.get("since")
    if not since:
        return True
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    try:
        return len(pd.bdate_range(pd.Timestamp(since), today)) - 1 <= window
    except Exception:
        return False


def window(turns: list, today=None) -> list:
    """Rutorna i köpfönstret ur board.turns — komplex och teman (råvaror) och Nordens aktiesegment."""
    out = []
    for t in turns or []:
        if not isinstance(t, dict) or not in_window(t, today):
            continue
        if t.get("level") == "segment" and t.get("where") != "Norden":
            continue
        out.append(t)
    return out


def members(turn: dict, universe: dict, complex_map: dict) -> set:
    """Tickers som hör till rutan: komplex via complex_map, tema via komplexets karta + BERSERK-temat, segment via
    universe[t]["segment"]."""
    level, key = turn.get("level"), turn.get("key")
    if level == "komplex":
        return {t for t, c in (complex_map or {}).items() if c == key}
    if level == "tema":
        from berserk import universe as buv
        cx = vcx.complex_of(key)
        own = {t for t, c in (complex_map or {}).items() if c == cx} if cx else set()
        return own | {t for t in buv.by_theme().get(key, []) if buv.region_of(t) == "Norden"}
    return {t for t, u in (universe or {}).items() if (u or {}).get("segment") == key}


def deep_picks(deep_blob: Optional[dict], turn: dict, tickers: set, n: int = N_DEEP) -> list:
    """Två bästa Deep Contrarian-raderna i rutan, på composite (inget verdikt finns i deep-läget)."""
    rows = vp._ca_rows([deep_blob] if deep_blob else [])
    if turn.get("level") == "segment":
        mine = [r for r in rows if r.get("segment") == turn.get("key")]
    else:
        mine = [r for r in rows if r["ticker"] in tickers]
    mine.sort(key=lambda r: (-(r.get("score") or 0), r["ticker"]))
    return [dict(r, source="Deep Contrarian") for r in vp._one_per_company(mine)[:n]]


def long_picks(long_scan: Optional[dict], buys: Optional[dict], turn: dict, tickers: set, n: int = N_LONG) -> list:
    """Två bästa långa: Long Screener STRONG BUY/BUY i rutan, annars Quality Contrarian BUY."""
    segs = (long_scan or {}).get("segments") or {}
    qsegs = (buys or {}).get("segments") or {}
    if turn.get("level") == "segment":
        pool = list((segs.get(turn["key"]) or {}).get("picks") or [])
        quality = [p for p in (qsegs.get(turn["key"]) or {}).get("picks") or [] if p.get("verdict") == "BUY"]
    else:
        pool = [p for e in segs.values() for p in e.get("picks") or [] if p.get("ticker") in tickers]
        quality = [p for e in qsegs.values() for p in e.get("picks") or []
                   if p.get("verdict") == "BUY" and p.get("ticker") in tickers]
    pool.sort(key=lambda p: (p.get("signal") != "STRONG BUY", -(p.get("score_pct") or 0), str(p.get("ticker"))))
    quality.sort(key=lambda p: -(p.get("score") or 0))
    out = vp._one_per_company(pool + quality)[:n]
    return [dict(p, source=p.get("source") or ("Quality Contrarian" if p.get("verdict") else "Long Screener")) for p in out]


def plan(turns: list, universe: dict, complex_map: dict, deep_blob: Optional[dict] = None,
         long_scan: Optional[dict] = None, buys: Optional[dict] = None, etf_pick: Optional[Callable] = None,
         today=None) -> dict:
    """{window: [{...turn, members, deep, long, etf}], generated}. etf_pick(key) → valhall.etf.pick (utbytbar)."""
    out = []
    for t in window(turns, today):
        tickers = members(t, universe, complex_map)
        etf = None
        if etf_pick is not None:
            try:
                etf = etf_pick(t["key"] if t.get("level") != "tema" else (vcx.complex_of(t["key"]) or t["key"]))
            except Exception:
                etf = None
        out.append({**t, "members": len(tickers), "deep": deep_picks(deep_blob, t, tickers),
                    "long": long_picks(long_scan, buys, t, tickers), "etf": etf})
    return {"window": out, "window_days": WINDOW_DAYS,
            "generated": str((pd.Timestamp(today) if today is not None else pd.Timestamp.today()).date())}


def lines(p: Optional[dict]) -> list:
    """Läsbara rader för kartan/loggen: 'Ädelmetaller (komplex, VÄNDER dag 12 ✓): Deep A, B · Lång C, D · ETF IGLN.L'."""
    out = []
    for w in (p or {}).get("window") or []:
        parts = []
        if w.get("deep"):
            parts.append("Deep " + ", ".join(x["ticker"] for x in w["deep"]))
        if w.get("long"):
            parts.append("Lång " + ", ".join(x["ticker"] for x in w["long"]))
        e = w.get("etf")
        if e:
            parts.append(f"ETF {e.get('ticker') or '—'} ({e.get('exposure')}{'' if e.get('verified') else ', att verifiera'})")
        where = {"komplex": "komplex", "tema": "tema"}.get(w.get("level"), w.get("where") or "segment")
        out.append(f"{w.get('label') or w.get('key')} ({where}, {w.get('kind')} dag {w.get('days') or '?'} ✓): "
                   + (" · ".join(parts) if parts else "inga kandidater i rutan i dag"))
    return out
