"""
journal_import/fifo.py — FIFO matching of BUY/SELL transactions.

Reconstructs closed trades and open positions per (account, ISIN).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from journal_import.parsers import Transaction


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ClosedTrade:
    account: str
    isin: str
    ticker: Optional[str]
    name: str
    entry_date: str
    exit_date: str
    qty: float
    entry_price: float
    exit_price: float
    currency: str
    fx_entry: float    # trade currency → account currency at entry
    fx_exit: float     # trade currency → account currency at exit
    fee_entry: float   # proportional entry fee in account currency
    fee_exit: float    # proportional exit fee in account currency


@dataclass
class OpenPosition:
    account: str
    isin: str
    ticker: Optional[str]
    name: str
    earliest_entry_date: str
    qty: float
    avg_price: float   # weighted average in trade currency
    currency: str


# ── FIFO engine ───────────────────────────────────────────────────────────────

def run_fifo(
    transactions: List[Transaction],
    ticker_resolver: Callable[[str, str, str], Optional[str]],
) -> Tuple[List[ClosedTrade], List[OpenPosition], List[str]]:
    """
    Match BUY/SELL transactions with FIFO logic.

    Parameters
    ----------
    transactions     : list of Transaction from parsers
    ticker_resolver  : callable(isin, symbol, exchange) → Optional[str]

    Returns
    -------
    closed_trades, open_positions, errors
    """
    closed_trades: List[ClosedTrade] = []
    open_positions: List[OpenPosition] = []
    errors: List[str] = []

    # Sort by account, isin, date so FIFO is chronological
    ordered = sorted(transactions, key=lambda t: (t.account, t.isin, t.date))

    # queue: (account, isin) → deque of [date, qty, price, fx_rate, fee, name, symbol, exchange, currency]
    queue: dict[tuple, deque] = defaultdict(deque)

    for txn in ordered:
        key = (txn.account, txn.isin)
        ticker = ticker_resolver(txn.isin, txn.symbol, txn.exchange)

        if txn.action == "BUY":
            queue[key].append({
                "date":     txn.date,
                "qty":      txn.qty,
                "price":    txn.price,
                "fx":       txn.fx_rate,
                "fee":      txn.fee,
                "ticker":   ticker,
                "name":     txn.name,
                "currency": txn.currency,
                "symbol":   txn.symbol,
                "exchange": txn.exchange,
            })

        elif txn.action == "SELL":
            remaining_sell = txn.qty
            total_sell_qty = txn.qty  # for fee proration

            while remaining_sell > 1e-6 and queue[key]:
                buy = queue[key][0]
                matched = min(remaining_sell, buy["qty"])

                # Proportional fees
                buy_fee_share  = buy["fee"]  * (matched / buy["qty"]) if buy["qty"] > 0 else 0.0
                sell_fee_share = txn.fee * (matched / total_sell_qty) if total_sell_qty > 0 else 0.0

                # Resolve ticker — prefer buy-side if available, else sell-side
                ct_ticker = buy["ticker"] or ticker

                closed_trades.append(ClosedTrade(
                    account=txn.account,
                    isin=txn.isin,
                    ticker=ct_ticker,
                    name=txn.name or buy["name"],
                    entry_date=buy["date"],
                    exit_date=txn.date,
                    qty=matched,
                    entry_price=buy["price"],
                    exit_price=txn.price,
                    currency=txn.currency,
                    fx_entry=buy["fx"],
                    fx_exit=txn.fx_rate,
                    fee_entry=buy_fee_share,
                    fee_exit=sell_fee_share,
                ))

                remaining_sell -= matched
                buy["qty"] -= matched
                buy["fee"] -= buy_fee_share  # reduce remaining fee proportionally

                if buy["qty"] < 1e-6:
                    queue[key].popleft()

            if remaining_sell > 1e-6:
                errors.append(
                    f"SELL without matching BUY: {txn.isin} ({txn.name}) "
                    f"on {txn.date}, {remaining_sell:.4f} shares unmatched — skipped"
                )

    # Remaining queue items → open positions
    for (account, isin), q in queue.items():
        if not q:
            continue
        total_qty   = sum(b["qty"]           for b in q)
        total_cost  = sum(b["qty"] * b["price"] for b in q)
        avg_price   = total_cost / total_qty if total_qty > 0 else 0.0
        first_date  = q[0]["date"]
        first_entry = q[0]
        ticker_hint = first_entry["ticker"]
        name_hint   = first_entry["name"]
        currency    = first_entry["currency"]

        open_positions.append(OpenPosition(
            account=account,
            isin=isin,
            ticker=ticker_hint,
            name=name_hint,
            earliest_entry_date=first_date,
            qty=round(total_qty, 6),
            avg_price=round(avg_price, 4),
            currency=currency,
        ))

    return closed_trades, open_positions, errors
