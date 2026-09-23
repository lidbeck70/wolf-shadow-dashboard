"""
journal_bridge.py — en stängd position i registret blir en journalrad.

positions.close() ropar hit. Raden får Trade Journal-formen (trade_journal.py:
entry/exit, resultat i % och kronor, R ur stoppen, hålltid, säljorsak) och
märks med position_id så samma stängning aldrig skrivs två gånger. Utan
inköps- eller säljkurs finns inget resultat att journalföra — då hoppar vi.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def trade_from_closed(row: dict) -> Optional[dict]:
    """Journalraden ur en closed-rad, eller None när resultatet inte går att räkna."""
    import journal_stats as js
    import positions

    entry = positions._opt(row.get("entry_price"))
    exit_price = positions._opt(row.get("exit_price"))
    if not entry or not exit_price:
        return None
    shares = int(positions._num(row.get("shares")))
    stop = positions._opt(row.get("stop"))
    pnl_pct = (exit_price / entry - 1) * 100
    r = js.r_multiple(entry, stop, exit_price=exit_price) if stop else None
    return {
        "id": str(uuid.uuid4()),
        "position_id": row.get("id"),
        "ticker": str(row.get("ticker") or "").upper(),
        "strategy": positions.TAG_PLAYBOOK.get(row.get("strategy"), "untagged"),
        "entry_date": str(row.get("entry_date") or ""),
        "exit_date": str(row.get("exit_date") or ""),
        "entry_price": round(entry, 2),
        "exit_price": round(exit_price, 2),
        "shares": shares,
        "direction": "long",
        "pnl_pct": round(pnl_pct, 2),
        "pnl_amount": round((exit_price - entry) * shares, 2),
        "r_multiple": round(r, 2) if r is not None else None,
        "stop_loss": round(stop, 2) if stop else None,
        "setup": None,
        "sell_rule": None,
        "holding_days": js.holding_days(row.get("entry_date"), row.get("exit_date")),
        "exit_reason": str(row.get("exit_reason") or "").lower().replace(" ", "_") or "manual",
        "notes": str(row.get("notes") or ""),
        "sector": str(row.get("sector") or ""),
        "source": "holdings",
    }


def record_close(row: dict) -> Optional[dict]:
    """Skriver journalraden för en stängd position. Returnerar raden, eller
    None när den hoppades (inget resultat, eller redan journalförd)."""
    trade = trade_from_closed(row)
    if trade is None:
        return None
    import trade_journal as tj
    trades = list(tj.load_journal() or [])
    if any(t.get("position_id") == trade["position_id"] for t in trades if isinstance(t, dict)):
        return None
    trades.append(trade)
    if not tj.save_journal(trades):
        logger.warning("journalen sparades bara lokalt för %s", trade["ticker"])
    return trade
