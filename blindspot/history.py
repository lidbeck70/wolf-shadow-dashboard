"""
history.py — JSONL append/read for blindspot opportunity score time series.
"""
import json
import os
from datetime import datetime
from typing import List


HISTORY_FILE = os.path.join(os.path.dirname(__file__), "blindspot_history.jsonl")


def append_report(ticker: str, opportunity: float, hat: float, strength: float,
                  catalyst: float, necessity: float, sector: str = "") -> None:
    """Append a single report entry to the JSONL history file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "opportunity": opportunity,
        "hat": hat,
        "strength": strength,
        "catalyst": catalyst,
        "necessity": necessity,
        "sector": sector,
    }
    try:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def read_history(ticker: str = None, limit: int = 500) -> List[dict]:
    """Read history entries, optionally filtered by ticker."""
    entries = []
    if not os.path.exists(HISTORY_FILE):
        return entries
    try:
        with open(HISTORY_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if ticker is None or entry.get("ticker") == ticker:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return entries[-limit:]
