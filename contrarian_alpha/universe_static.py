"""
universe_static.py — Static resource-universe loader (PR1 foundation).

Reads a CSV seed list of US/Canada commodity/resource tickers (Rick Rule /
Eric Sprott style screening) and returns lightweight records that
engine._build_universe() can turn into pipeline instruments.

This module is intentionally scoring-agnostic: it only supplies identity and
metadata (stage, commodity, exchange/country, yf_ticker). Stage-aware scoring
is a later PR — see CLAUDE.md / PR2 notes.

CSV schema (config/universes/us_ca_resource.csv):
    ticker,yf_ticker,name,exchange,country,stage,primary_commodity,
    secondary_commodity,notes

Optional enrichment columns (read if present, ignored otherwise — PR3):
    jurisdiction              fine-grained region (e.g. "Quebec", "Nevada")
    cash_musd                 cash on hand, millions USD
    quarterly_burn_musd       cash burn per quarter, millions USD
    debt_musd                 total debt, millions USD
    shares_out_m              shares outstanding, millions
    shares_yoy_growth_pct     YoY share-count growth %
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root = parent of the contrarian_alpha/ package.
_DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "config" / "universes" / "us_ca_resource.csv"
)

# Universe key used by PipelineConfig.universe and the UI preset.
UNIVERSE_KEY = "us_ca_resource"

_REQUIRED_COLUMNS = {"ticker", "yf_ticker", "name", "country", "stage"}

# Conservative stage vocabulary (informational only in PR1).
VALID_STAGES = {"producer", "developer", "explorer", "royalty", "energy", "services"}

# Optional enrichment columns consumed by resource_scoring.py (PR3). Absent =>
# missing-data flags + neutral scores; never fabricated.
_OPTIONAL_COLUMNS = (
    "jurisdiction", "cash_musd", "quarterly_burn_musd", "debt_musd",
    "shares_out_m", "shares_yoy_growth_pct",
)


@dataclass
class ResourceRecord:
    """One row of the static resource universe."""

    ticker: str
    yf_ticker: str
    name: str
    exchange: str = ""
    country: str = ""
    stage: str = ""
    primary_commodity: str = ""
    secondary_commodity: str = ""
    notes: str = ""
    # Optional enrichment (PR3). Absent columns stay as-is (blank/None).
    jurisdiction: str = ""
    cash_musd: str = ""
    quarterly_burn_musd: str = ""
    debt_musd: str = ""
    shares_out_m: str = ""
    shares_yoy_growth_pct: str = ""

    def to_metadata(self) -> dict:
        """Return the metadata dict attached to instrument info for future PRs."""
        return {
            "stage": self.stage,
            "primary_commodity": self.primary_commodity,
            "secondary_commodity": self.secondary_commodity,
            "exchange": self.exchange,
            "country": self.country,
            "notes": self.notes,
            "yf_ticker": self.yf_ticker,
            # Optional enrichment (blank string when the CSV column is absent).
            "jurisdiction": self.jurisdiction,
            "cash_musd": self.cash_musd,
            "quarterly_burn_musd": self.quarterly_burn_musd,
            "debt_musd": self.debt_musd,
            "shares_out_m": self.shares_out_m,
            "shares_yoy_growth_pct": self.shares_yoy_growth_pct,
        }


def _csv_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else _DEFAULT_CSV


def load_resource_universe(path: str | Path | None = None) -> list[ResourceRecord]:
    """
    Load and validate the static resource universe CSV.

    Raises:
        FileNotFoundError — if the CSV is missing (clear message for the UI).
        ValueError        — if required columns are absent or no valid rows.
    """
    csv_path = _csv_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Resource-universum saknas: {csv_path}. "
            f"Skapa filen config/universes/us_ca_resource.csv eller valj ett "
            f"annat universum."
        )

    records: list[ResourceRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = set(reader.fieldnames or [])
        missing = _REQUIRED_COLUMNS - header
        if missing:
            raise ValueError(
                f"Resource-CSV saknar obligatoriska kolumner {sorted(missing)} "
                f"i {csv_path}. Hittade: {sorted(header)}."
            )

        for i, row in enumerate(reader, start=2):  # row 1 = header
            ticker = (row.get("ticker") or "").strip()
            yf_ticker = (row.get("yf_ticker") or "").strip() or ticker
            name = (row.get("name") or "").strip() or ticker
            if not ticker:
                logger.debug("universe_static: skipping empty ticker at row %d", i)
                continue

            stage = (row.get("stage") or "").strip().lower()
            if stage and stage not in VALID_STAGES:
                logger.warning(
                    "universe_static: okand stage '%s' for %s (rad %d) — behalls som-ar",
                    stage, ticker, i,
                )

            opt = {c: (row.get(c) or "").strip() for c in _OPTIONAL_COLUMNS}

            records.append(ResourceRecord(
                ticker=ticker,
                yf_ticker=yf_ticker,
                name=name,
                exchange=(row.get("exchange") or "").strip(),
                country=(row.get("country") or "").strip().upper(),
                stage=stage,
                primary_commodity=(row.get("primary_commodity") or "").strip().lower(),
                secondary_commodity=(row.get("secondary_commodity") or "").strip().lower(),
                notes=(row.get("notes") or "").strip(),
                **opt,
            ))

    if not records:
        raise ValueError(f"Resource-CSV innehaller inga giltiga rader: {csv_path}.")

    logger.info("universe_static: laddade %d resource-tickers fran %s",
                len(records), csv_path.name)
    return records
