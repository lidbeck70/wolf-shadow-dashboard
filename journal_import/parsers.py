"""
journal_import/parsers.py — Parse Nordnet ISK and Nordea ASK transaction exports.

Returns ParseResult(transactions, errors) — never raises on bad rows.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import List, Union

import pandas as pd


# ── Shared dataclass ──────────────────────────────────────────────────────────

@dataclass
class Transaction:
    account: str        # "nordnet_isk" | "nordea_ask"
    isin: str
    name: str           # security name (display / fallback)
    symbol: str         # local ticker hint (Nordea: col4, Nordnet: "")
    exchange: str       # exchange name (Nordea: col5, Nordnet: "")
    date: str           # YYYY-MM-DD
    action: str         # "BUY" | "SELL"
    qty: float
    price: float        # in trade currency
    currency: str       # e.g. "SEK", "NOK", "USD"
    fx_rate: float      # trade currency → account currency (1.0 if same)
    fee: float          # in account currency
    gross_amount: float  # abs value in account currency
    raw_hash: str       # hex prefix for dedup


@dataclass
class ParseResult:
    transactions: List[Transaction]
    errors: List[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_decimal(val: object) -> float:
    """Convert Swedish/Norwegian decimal-comma string to float. Returns 0.0 on failure."""
    if val is None:
        return 0.0
    s = str(val).strip().replace('\xa0', '').replace(' ', '').replace(' ', '')
    if s in ('', '-', 'nan', 'None'):
        return 0.0
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        return 0.0


def _make_hash(account: str, isin: str, date: str, qty: float, price: float) -> str:
    key = f"{account}|{isin}|{date}|{qty:.6f}|{price:.6f}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _to_df(file_obj: Union[io.IOBase, str, bytes], encoding: str, sep: str) -> pd.DataFrame:
    """Read CSV from file path, file-like, or bytes into DataFrame."""
    if isinstance(file_obj, str):
        return pd.read_csv(file_obj, encoding=encoding, sep=sep,
                           header=0, dtype=str, on_bad_lines='skip')
    if isinstance(file_obj, bytes):
        text = file_obj.decode(encoding)
        return pd.read_csv(io.StringIO(text), sep=sep, header=0,
                           dtype=str, on_bad_lines='skip')
    # Streamlit UploadedFile or any file-like
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode(encoding)
    return pd.read_csv(io.StringIO(content), sep=sep, header=0,
                       dtype=str, on_bad_lines='skip')


# ── Nordnet ISK parser ────────────────────────────────────────────────────────
# Encoding: UTF-16 LE BOM  Separator: TAB
# BUY = 'KÖPT' (K\xd6PT)   SELL = 'SÅLT' (S\xc5LT)
# Col indices (0-based):
#   [2]  Affärsdag  — trade date YYYY-MM-DD
#   [5]  Transaktionstyp
#   [6]  Värdepapper — security name
#   [7]  ISIN
#   [8]  Antal (qty, decimal comma)
#   [9]  Kurs  (price, decimal comma)
#   [12] Valuta  (account currency of Belopp, always SEK)
#   [13] Belopp  (net amount SEK, decimal comma)
#   [21] Växlingskurs  (FX rate, decimal comma; empty for SEK trades)
#   [26] Courtage  (commission SEK, decimal comma)

_NORDNET_BUY  = 'KÖPT'
_NORDNET_SELL = 'SÅLT'


def parse_nordnet(file_obj: Union[io.IOBase, str, bytes]) -> ParseResult:
    """Parse Nordnet ISK CSV export."""
    transactions: List[Transaction] = []
    errors: List[str] = []

    try:
        df = _to_df(file_obj, encoding='utf-16', sep='\t')
    except Exception as exc:
        return ParseResult([], [f"Nordnet: failed to read file — {exc}"])

    df = df[df.iloc[:, 5].isin([_NORDNET_BUY, _NORDNET_SELL])].copy()

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            vals = row.tolist()

            date_str     = str(vals[2]).strip()
            typ          = str(vals[5]).strip()
            name         = str(vals[6]).strip()
            isin         = str(vals[7]).strip()
            qty          = _parse_decimal(vals[8])
            price        = _parse_decimal(vals[9])
            currency     = str(vals[12]).strip() if len(vals) > 12 else 'SEK'
            gross_amount = abs(_parse_decimal(vals[13])) if len(vals) > 13 else 0.0

            fx_raw  = str(vals[21]).strip() if len(vals) > 21 else ''
            fx_rate = _parse_decimal(fx_raw) if fx_raw not in ('', '0', 'nan') else 1.0

            fee_raw = str(vals[26]).strip() if len(vals) > 26 else ''
            fee     = abs(_parse_decimal(fee_raw)) if fee_raw not in ('', 'nan') else 0.0

            if not isin or isin in ('nan', 'None', ''):
                continue
            if qty == 0:
                continue

            action   = "BUY" if typ == _NORDNET_BUY else "SELL"
            raw_hash = _make_hash("nordnet_isk", isin, date_str, qty, price)

            transactions.append(Transaction(
                account="nordnet_isk",
                isin=isin,
                name=name,
                symbol="",
                exchange="",
                date=date_str,
                action=action,
                qty=qty,
                price=price,
                currency=currency,
                fx_rate=fx_rate,
                fee=fee,
                gross_amount=gross_amount,
                raw_hash=raw_hash,
            ))
        except Exception as exc:
            errors.append(f"Nordnet row {i}: {exc}")

    return ParseResult(transactions, errors)


# ── Nordea ASK parser ─────────────────────────────────────────────────────────
# Encoding: UTF-8 BOM  Separator: semicolon
# BUY = 'Kjøp' (Kj\xf8p)   SELL = 'Salg'   SKIP = 'Innløsning'
# Col indices (0-based):
#   [1]  Kjøp / Salg — type
#   [2]  Navn  — security name
#   [3]  ISIN
#   [4]  Symbol  — local ticker
#   [5]  Børs   — exchange name
#   [6]  Handelsdato  — trade date dd.mm.yyyy
#   [8]  Antall  (qty, decimal comma)
#   [9]  Kurs    (price, decimal comma)
#   [10] Valuta  (trade currency)
#   [11] Valutakurs  (FX rate: trade currency → NOK, decimal comma)
#   [14] Markedsverdi i oppgjørsvaluta  (gross NOK, decimal comma)
#   [16] Kurtasje  (fee NOK, decimal comma)

_NORDEA_BUY  = 'Kjøp'
_NORDEA_SELL = 'Salg'
_NORDEA_SKIP = 'Innløsning'


def parse_nordea(file_obj: Union[io.IOBase, str, bytes]) -> ParseResult:
    """Parse Nordea ASK CSV export."""
    transactions: List[Transaction] = []
    errors: List[str] = []

    try:
        df = _to_df(file_obj, encoding='utf-8-sig', sep=';')
    except Exception as exc:
        return ParseResult([], [f"Nordea: failed to read file — {exc}"])

    df = df[df.iloc[:, 1].isin([_NORDEA_BUY, _NORDEA_SELL])].copy()
    df = df[df.iloc[:, 3].notna() & (df.iloc[:, 3].str.strip() != '')].copy()

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            vals = row.tolist()

            typ      = str(vals[1]).strip()
            name     = str(vals[2]).strip()
            isin     = str(vals[3]).strip()
            symbol   = str(vals[4]).strip() if len(vals) > 4 else ''
            exchange = str(vals[5]).strip() if len(vals) > 5 else ''
            date_raw = str(vals[6]).strip() if len(vals) > 6 else ''
            qty      = _parse_decimal(vals[8]) if len(vals) > 8 else 0.0
            price    = _parse_decimal(vals[9]) if len(vals) > 9 else 0.0
            currency = str(vals[10]).strip() if len(vals) > 10 else 'NOK'

            fx_raw  = str(vals[11]).strip() if len(vals) > 11 else ''
            fx_rate = _parse_decimal(fx_raw) if fx_raw not in ('', 'nan') else 1.0

            gross_raw    = str(vals[14]).strip() if len(vals) > 14 else ''
            gross_amount = abs(_parse_decimal(gross_raw))

            fee_raw = str(vals[16]).strip() if len(vals) > 16 else ''
            fee     = abs(_parse_decimal(fee_raw)) if fee_raw not in ('', 'nan') else 0.0

            # dd.mm.yyyy → YYYY-MM-DD
            if '.' in date_raw:
                d, m, y = date_raw.split('.')
                date_str = f"{y}-{m}-{d}"
            else:
                date_str = date_raw

            if not isin or isin in ('nan', 'None', ''):
                continue
            if qty == 0:
                continue

            action   = "BUY" if typ == _NORDEA_BUY else "SELL"
            raw_hash = _make_hash("nordea_ask", isin, date_str, qty, price)

            transactions.append(Transaction(
                account="nordea_ask",
                isin=isin,
                name=name,
                symbol=symbol,
                exchange=exchange,
                date=date_str,
                action=action,
                qty=qty,
                price=price,
                currency=currency,
                fx_rate=fx_rate,
                fee=fee,
                gross_amount=gross_amount,
                raw_hash=raw_hash,
            ))
        except Exception as exc:
            errors.append(f"Nordea row {i}: {exc}")

    return ParseResult(transactions, errors)
