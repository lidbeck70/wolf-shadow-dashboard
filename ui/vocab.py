"""
ui/vocab.py — ordlistan (designförslaget PR 3, avsnitt 4): ett huvudord per betydelse.

Ordlistan ändrar BARA visningen. Motorerna, arken och datakontrakten behåller sina ord (PASS, READY, GOLDEN TICKET,
CLAUDE.md-statusarna …); panelen visar huvudordet och källans ord som chip bredvid: "KÖP · GOLDEN TICKET".
Ett ord som inte är entydigt (Neutral i Royalty C) mappas inte och visas som det är.

PASS betyder två saker: i en regelkontroll är det OK, som verdikt är det NEJ (avstå) — därför tar main() ett
sammanhang. READY står under BEVAKA: i Viking Nine och Valhalls drivargrind är READY steget före GOLDEN TICKET.
"""
from __future__ import annotations

from typing import Optional

KOP, BEVAKA, HALL, VAKSAM, SALJ, MINSKA, NEJ, SAKNAS = "KÖP", "BEVAKA", "HÅLL", "VAKSAM", "SÄLJ", "MINSKA", "NEJ", "Saknas"
OK, KOLLA = "OK", "KOLLA"
VERDICT, CHECK, HOLDING = "verdikt", "kontroll", "innehav"

WORDS: dict = {
    KOP: ("KÖP", "KÖP-KANDIDAT", "KÖPKANDIDAT", "KÖPLÄGE", "KÖP NU", "KÖP-STÖD", "BUY", "STRONG BUY", "BUY CANDIDATE",
          "GOLDEN TICKET", "KLAR FÖR KÖP", "ENTRY", "AGERA", "GATE OPEN", "OK ATT TA LÅNG POSITION",
          "KÄRNINNEHAV", "ACCUMULATE", "ACCUMULATE 1", "ACCUMULATE 2", "ACCUMULATE 3"),
    BEVAKA: ("BEVAKA", "WATCH", "WAIT", "AVVAKTA", "DEVELOPING", "BEVAKNINGSLISTA", "VÄNTAR PÅ TEKNISK TRIGGER", "READY"),
    HALL: ("HÅLL", "HOLD"),
    SALJ: ("SÄLJ", "EXIT", "SELL", "SÄLJ / TA HEM"),
    MINSKA: ("MINSKA", "REDUCE", "DELVINST", "PARTIAL"),
    NEJ: ("NEJ", "PASSA", "NO TRADE", "INGEN AFFÄR", "SPÄRRAD", "REJECT", "REJECTED", "AVSTÅ", "VILA", "BLOCKERAD"),
    SAKNAS: ("SAKNAS", "DATA UNAVAILABLE", "DATA SAKNAS", "DATA_MISSING", "DATA_GAP", "NO DATA", "OKÄND", "OKÄNT LÄGE"),
}
_INDEX = {w: main for main, ws in WORDS.items() for w in ws}
CHECK_WORDS = {"PASS": OK, "MANUAL": KOLLA, "FAIL": NEJ, "✓": OK, "◌": KOLLA, "✗": NEJ}
UNMAPPED = frozenset({"NEUTRAL"})


def _norm(word) -> str:
    return " ".join(str(word or "").strip().upper().split())


def main(word, context: str = VERDICT) -> Optional[str]:
    """Huvudordet för källans ord, eller None när ordet inte är entydigt eller okänt (visa då källans ord).
    'KÖP — logga i journalen' och 'BUY (4/4)' läses på första ledet."""
    w = _norm(word)
    if not w or w in UNMAPPED:
        return None
    if context == CHECK:
        return CHECK_WORDS.get(w)
    if w == "PASS":
        return NEJ
    if context == HOLDING and w == "WATCH":
        return VAKSAM
    if w in _INDEX:
        return _INDEX[w]
    for sep in (" — ", " - ", " · ", " (", ":"):
        head = w.split(sep, 1)[0].strip()
        if head != w and head in _INDEX:
            return _INDEX[head]
    return None


def pair(word, context: str = VERDICT) -> tuple:
    """(huvudord, källans ord eller '') — källans ord bara när det skiljer sig från huvudordet."""
    src = str(word or "").strip()
    m = main(src, context)
    if m is None:
        return src, ""
    return m, ("" if _norm(src) == _norm(m) else src)
