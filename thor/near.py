"""
thor/near.py — "hur långt från köp": en rad per strategi ur fälten som redan finns i Thors rad och körningens blob.

Inga nya regler och inga nya beräkningar. Texten säger vad strategins egen körning redan räknat — Viking Nines
faktorer och Execution, Wolfs grindar, Momentums rad och setup, EMBER:s tre grindar, Alpha Regimes signaler,
Tiggres grindar, Royaltys composite, Insiders statusflöde — och strategins egen regel för KÖP med dess egna
konstanter. Övervakningen (portfolio/watch_ui) visar den som Avstånd. Rena funktioner.
"""

from __future__ import annotations

import re
from typing import Optional

from thor import candidates as tc


def _raw(data: Optional[dict], key: str, ticker: str, rows_key: str = "rows") -> dict:
    """Körningens egen rad för aktien (data[key][rows_key]) — {} när den saknas."""
    blob = (data or {}).get(key)
    rows = blob.get(rows_key) if isinstance(blob, dict) else None
    if not isinstance(rows, list):
        return {}
    return next((r for r in rows if isinstance(r, dict) and tc.yf(r.get("ticker")) == ticker), {})


def _frac(text) -> Optional[tuple]:
    """'grindar 6/7' → (6, 7); None när texten saknar ett bråk."""
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _left(n: int, what: str) -> str:
    return f"{n} {what} kvar"


# ── Per strategi (bara BEVAKA/HÅLL/TRÄFF — KÖP och SPÄRRAD tas gemensamt) ──
def _viking_nine(c, data) -> str:
    raw = _raw(data, "viking_nine", c.ticker)
    nine = _frac(raw.get("ovtlyr_nine") or c.score_text)
    layers = " · ".join(f"{lbl} {raw[k]}" for k, lbl in (("market", "marknad"), ("sector", "sektor"), ("stock", "aktie"))
                        if raw.get(k))
    if nine and nine[0] >= 9:
        ex = f" (Viking {raw['viking']})" if raw.get("viking") else ""
        return f"9/9 — Execution väntar{ex}" + (f": {c.why}" if c.why else "")
    if nine:
        return f"{nine[0]}/9 — {_left(9 - nine[0], 'faktor(er)')} till 9/9" + (f" ({layers})" if layers else "")
    return c.status


def _wolf(c, data) -> str:
    band = c.status
    gates = re.search(r"grindar (\d+/\d+)", c.score_text or "")
    head = (c.score_text or "").split(" · ")[0]
    return (f"{head} · {band}" + (f" · grindar {gates.group(1)}" if gates else "")
            + " — ingen entry ännu (köpregeln ej uppfylld)")


def _viking(c, data) -> str:
    rule = (c.why or "").split(" — ")[0].replace("Viking-screenern: ", "")     # regeln står redan i radens varför
    return f"{c.status} — {rule}" if rule else c.status


def _momentum(c, data) -> str:
    from valhall.wolf import MOMENTUM_TOP
    m = re.match(r"rad (\d+)", c.status or "")
    rank = int(m.group(1)) if m else None
    if "regimljuset saknas" in (c.why or ""):
        return f"{c.status} — regimljuset saknas, inget köp utan ljus"
    if rank is not None and rank > MOMENTUM_TOP:
        return f"rad {rank} — topp {MOMENTUM_TOP} krävs" + (" (setup A finns)" if "setup A" in c.status else "")
    if rank is not None:
        return f"rad {rank} — setup A saknas"
    return c.status


def _berserk(c, data) -> str:
    return c.why or c.status                         # skannerns eget skäl: 'S2: … väntar på 20-dagarshögsta med volym'


def _ember(c, data) -> str:
    raw = _raw(data, "ember", c.ticker, "near_misses") or _raw(data, "ember", c.ticker, "eligible")
    parts = (("trend_pass", "trend"), ("entry_pass", "entry"), ("notrade_clear", "ingen spärr"))
    if not any(k in raw for k, _l in parts):
        return c.status
    ok = sum(1 for k, _l in parts if raw.get(k))
    marks = " · ".join(f"{lbl} {'✓' if raw.get(k) else '✗'}" for k, lbl in parts)
    return f"{c.status} — {ok}/3 grindar ({marks})"


def _quality(c, data) -> str:
    f = _frac(c.status)
    if f and f[0] < f[1]:
        return f"Alpha Regime {f[0]}/{f[1]} — {_left(f[1] - f[0], 'signal(er)')} till BUY"
    if c.status.startswith(("rank", "listan")):
        return f"{c.status} — inte analyserad av Alpha Regime än"
    return c.status


def _deep(c, data) -> str:
    return f"{c.status} — listan rangordnar bara; inget KÖP-läge i körningen"


def _long(c, data) -> str:
    return " · ".join(x for x in (c.status, c.score_text) if x)


def _insider(c, data) -> str:
    raw = _raw(data, "insider", c.ticker, "clusters")
    full = str(raw.get("status") or c.status)
    return full + (" · passa (jagar kursen)" if raw.get("chase") else "")


def _tiggre(c, data) -> str:
    f = _frac(c.status)
    if f and f[0] < f[1]:
        return f"grindar {f[0]}/{f[1]} — {_left(f[1] - f[0], 'grind(ar)')}"
    return f"{c.status} — {c.why}" if c.why else c.status


def _durrett(c, data) -> str:
    rest = [x for x in (c.score_text or "").split(" · ") if x and x != c.status]
    return " · ".join([c.status] + rest) + " — papperet köper 15-BAGGER eller nordiska"


def _rule(c, data) -> str:
    return f"{c.why}: {c.status}" if c.why and c.why != c.status else c.status


def _royalty(c, data) -> str:
    from royalty.engine import BUY_MIN
    m = re.search(r"composite (\d+)", c.score_text or "")
    if c.status_class == tc.BEVAKA and m:
        return f"composite {m.group(1)} — BUY CANDIDATE från {BUY_MIN:.0f}"
    return c.status


def _hav(c, data) -> str:
    return f"håvträff ({c.status}) — granska i arket; en träff är ingen köpsignal"


NEAR = {"viking_nine": _viking_nine, "wolf": _wolf, "viking": _viking, "momentum": _momentum, "berserk": _berserk,
        "ember": _ember, "quality": _quality, "deep": _deep, "long": _long, "insider": _insider, "tiggre": _tiggre,
        "durrett": _durrett, "rule": _rule, "royalty": _royalty, "hav": _hav}


def near_text(c, data: Optional[dict] = None) -> str:
    """Hur långt raden är från strategins KÖP, i klartext. c = Thors rad (None → tomt), data = Thors data med
    körningarnas blobbar (thor.ui.load). Tål trasiga blobbar: då står strategins eget ord."""
    if c is None:
        return ""
    if c.status_class == tc.KOP:                    # en gammal körning säger inget om i dag (Läge: KÖP · GAMMAL)
        return f"KÖP i en gammal körning — {c.status}" if c.stale else f"köpläge nu — {c.status}"
    if c.status_class == tc.SPARRAD:
        return f"köpsignal men spärrad — {c.status}"
    fn = NEAR.get(c.strategy)
    if fn is None:
        return c.status
    try:
        return fn(c, data) or c.status
    except Exception:
        return c.status
