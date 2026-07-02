#!/usr/bin/env python3
"""
validate_resource_universe.py — CSV integrity + enrichment audit (PR4).

Deterministic, dependency-free validator for
``config/universes/us_ca_resource.csv``. Checks structural integrity (required
columns, valid stages, non-empty primary commodity, duplicate tickers,
yf_ticker format, data_as_of date format) and prints an enrichment-completeness
summary so data gaps are transparent rather than hidden.

Usage:
    python scripts/validate_resource_universe.py [path/to.csv]

Exit code 0 = no hard errors (warnings allowed); 1 = validation errors found.
The heavy lifting lives in :func:`validate_resource_universe` so tests can call
it directly without spawning a subprocess.
"""
from __future__ import annotations

import csv
import datetime
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the loader's schema/vocabulary as the single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contrarian_alpha.universe_static import (  # noqa: E402
    _DEFAULT_CSV, _OPTIONAL_COLUMNS, _REQUIRED_COLUMNS, VALID_STAGES,
)

# Columns whose enrichment completeness we summarize (financials + provenance).
_ENRICHMENT_COLUMNS = (
    "jurisdiction", "project_region", "cash_musd", "debt_musd",
    "quarterly_burn_musd", "shares_out_m", "shares_yoy_growth_pct",
    "data_source", "data_as_of",
)
_YF_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,3})?$")


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    completeness: dict[str, int] = field(default_factory=dict)
    row_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_date(value: str) -> bool:
    try:
        datetime.datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_resource_universe(path: str | Path | None = None) -> ValidationResult:
    """Validate the resource universe CSV and return a ValidationResult."""
    csv_path = Path(path) if path else _DEFAULT_CSV
    res = ValidationResult()

    if not csv_path.exists():
        res.errors.append(f"CSV not found: {csv_path}")
        return res

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    missing_required = _REQUIRED_COLUMNS - set(header)
    if missing_required:
        res.errors.append(f"missing required columns: {sorted(missing_required)}")

    res.row_count = len(rows)
    if not rows:
        res.errors.append("CSV contains no data rows")
        return res

    seen: dict[str, int] = {}
    completeness = {c: 0 for c in _ENRICHMENT_COLUMNS if c in header}

    for i, row in enumerate(rows, start=2):  # row 1 = header
        ticker = (row.get("ticker") or "").strip()
        yf = (row.get("yf_ticker") or "").strip()
        stage = (row.get("stage") or "").strip().lower()
        primary = (row.get("primary_commodity") or "").strip()

        if not ticker:
            res.errors.append(f"row {i}: empty ticker")
            continue

        if ticker in seen:
            res.errors.append(
                f"row {i}: duplicate ticker '{ticker}' (first seen row {seen[ticker]})")
        else:
            seen[ticker] = i

        if stage and stage not in VALID_STAGES:
            res.errors.append(
                f"row {i} ({ticker}): invalid stage '{stage}' "
                f"(allowed: {sorted(VALID_STAGES)})")

        if not primary:
            res.errors.append(f"row {i} ({ticker}): empty primary_commodity")

        if yf and not _YF_TICKER_RE.match(yf):
            res.warnings.append(
                f"row {i} ({ticker}): unusual yf_ticker format '{yf}'")

        as_of = (row.get("data_as_of") or "").strip()
        if as_of and not _parse_date(as_of):
            res.errors.append(
                f"row {i} ({ticker}): data_as_of '{as_of}' not YYYY-MM-DD")

        for col in completeness:
            if (row.get(col) or "").strip():
                completeness[col] += 1

    res.completeness = completeness
    return res


def _format_report(res: ValidationResult) -> str:
    lines = [f"Resource universe validation — {res.row_count} rows"]
    lines.append("")
    if res.errors:
        lines.append(f"ERRORS ({len(res.errors)}):")
        lines += [f"  - {e}" for e in res.errors]
    else:
        lines.append("ERRORS: none")
    lines.append("")
    if res.warnings:
        lines.append(f"WARNINGS ({len(res.warnings)}):")
        lines += [f"  - {w}" for w in res.warnings]
    else:
        lines.append("WARNINGS: none")
    lines.append("")
    lines.append("Enrichment completeness (populated / total):")
    n = res.row_count or 1
    for col in _ENRICHMENT_COLUMNS:
        if col in res.completeness:
            cnt = res.completeness[col]
            lines.append(f"  {col:<24} {cnt:>3}/{res.row_count}  ({cnt * 100 // n}%)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = argv[0] if argv else None
    res = validate_resource_universe(path)
    print(_format_report(res))
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
