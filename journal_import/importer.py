"""
journal_import/importer.py — Convert FIFO-matched closed trades to journal entries
and write them to the existing Trade Journal storage (Gist + local).

Deduplication: trades already imported (identified by _import_hash) are skipped.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import List, Optional, Union

from journal_import.parsers import ParseResult, parse_nordnet, parse_nordea
from journal_import.fifo import ClosedTrade, OpenPosition, run_fifo
from journal_import.ticker_map import resolve_ticker


# ── Import result ─────────────────────────────────────────────────────────────

@dataclass
class ImportReport:
    trades_found: int
    skipped_dupes: int
    skipped_no_ticker: int
    errors: List[str]
    preview: List[dict]           # journal-ready dicts
    open_positions: List[OpenPosition]
    parse_errors: List[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_hash(ct: ClosedTrade) -> str:
    """Stable dedup hash for a closed trade."""
    key = (
        f"{ct.account}|{ct.isin}|{ct.entry_date}|{ct.exit_date}"
        f"|{ct.qty:.6f}|{ct.entry_price:.6f}|{ct.exit_price:.6f}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def _pnl(ct: ClosedTrade) -> tuple[float, float]:
    """
    Returns (pnl_pct, pnl_amount_account_currency).

    pnl_amount is in account currency (SEK for Nordnet, NOK for Nordea).
    Fees are deducted from pnl_amount but not pnl_pct (price-only).
    """
    if ct.entry_price == 0:
        return 0.0, 0.0

    pnl_pct = round((ct.exit_price - ct.entry_price) / ct.entry_price * 100, 2)

    # Convert trade-currency P&L to account currency using fx at exit
    pnl_trade = (ct.exit_price - ct.entry_price) * ct.qty
    pnl_account = pnl_trade * ct.fx_exit - ct.fee_entry - ct.fee_exit
    pnl_amount = round(pnl_account, 2)

    return pnl_pct, pnl_amount


def closed_trade_to_journal(ct: ClosedTrade) -> dict:
    """Convert a ClosedTrade to a journal entry dict compatible with trade_journal.py."""
    pnl_pct, pnl_amount = _pnl(ct)
    ticker = ct.ticker or ct.isin  # fall back to ISIN if unresolved

    return {
        "id":           str(uuid.uuid4()),
        "ticker":       ticker,
        "strategy":     "legacy",
        "entry_date":   ct.entry_date,
        "exit_date":    ct.exit_date,
        "entry_price":  round(ct.entry_price, 4),
        "exit_price":   round(ct.exit_price, 4),
        "shares":       round(ct.qty, 4),
        "direction":    "long",
        "pnl_pct":      pnl_pct,
        "pnl_amount":   pnl_amount,
        "r_multiple":   0.0,
        "exit_reason":  "manual",
        "notes":        f"Imported from {ct.account}",
        "sector":       "",
        "currency":     ct.currency,
        "account":      ct.account,
        "source":       "import",
        "isin":         ct.isin,
        "name":         ct.name,
        "_import_hash": _import_hash(ct),
    }


def _existing_hashes(trades: list) -> set:
    return {t["_import_hash"] for t in trades if "_import_hash" in t}


# ── Main import pipeline ──────────────────────────────────────────────────────

def run_import(
    nordnet_file: Optional[Union[object, str]] = None,
    nordea_file:  Optional[Union[object, str]] = None,
    existing_trades: Optional[list] = None,
) -> ImportReport:
    """
    Parse → FIFO → deduplicate.

    Parameters
    ----------
    nordnet_file     : file path (str), file-like, or None
    nordea_file      : file path (str), file-like, or None
    existing_trades  : current journal list (for dedup); if None, dedup is skipped

    Returns
    -------
    ImportReport — call apply_import(report.preview) to commit
    """
    all_transactions = []
    parse_errors: List[str] = []

    if nordnet_file is not None:
        result = parse_nordnet(nordnet_file)
        all_transactions.extend(result.transactions)
        parse_errors.extend(result.errors)

    if nordea_file is not None:
        result = parse_nordea(nordea_file)
        all_transactions.extend(result.transactions)
        parse_errors.extend(result.errors)

    closed_trades, open_positions, fifo_errors = run_fifo(all_transactions, resolve_ticker)
    errors = parse_errors + fifo_errors

    known_hashes = _existing_hashes(existing_trades or [])

    preview: List[dict] = []
    skipped_dupes = 0
    skipped_no_ticker = 0

    for ct in closed_trades:
        h = _import_hash(ct)
        if h in known_hashes:
            skipped_dupes += 1
            continue
        if ct.ticker is None:
            skipped_no_ticker += 1
            errors.append(
                f"Unresolved ticker: ISIN={ct.isin} name='{ct.name}' — "
                f"add to ISIN_MAP in journal_import/ticker_map.py"
            )
            # Still include in preview so user can see it, but mark it
            entry = closed_trade_to_journal(ct)
            entry["ticker"] = f"[{ct.isin}]"  # visible placeholder
            preview.append(entry)
        else:
            preview.append(closed_trade_to_journal(ct))
            known_hashes.add(h)  # prevent self-dupes when same trade appears in both files

    return ImportReport(
        trades_found=len(closed_trades),
        skipped_dupes=skipped_dupes,
        skipped_no_ticker=skipped_no_ticker,
        errors=errors,
        preview=preview,
        open_positions=open_positions,
        parse_errors=parse_errors,
    )


def apply_import(preview: List[dict]) -> tuple[bool, str]:
    """
    Append preview trades to the live journal (dedup by _import_hash).

    Returns (success, message).
    """
    try:
        from trade_journal import load_journal, save_journal
    except ImportError:
        return False, "Cannot import trade_journal module"

    try:
        trades = load_journal()
        known = _existing_hashes(trades)
        added = 0
        for entry in preview:
            h = entry.get("_import_hash", "")
            if h and h in known:
                continue
            trades.append(entry)
            if h:
                known.add(h)
            added += 1

        ok = save_journal(trades)
        msg = f"Imported {added} trade(s)."
        if not ok:
            msg += " (saved locally — cloud sync failed)"
        return True, msg
    except Exception as exc:
        return False, f"Import failed: {exc}"
