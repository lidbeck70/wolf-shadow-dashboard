"""
Tests for csv_export.py — the offline fallback (migrationsspec §0 och §7).
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv_export as ce

COLS = [("ticker", "Ticker"), ("score", "Poäng"), ("ok", "Grind")]


def test_header_comes_from_the_column_list():
    out = ce.rows_to_csv([], COLS)
    assert out.strip() == "Ticker;Poäng;Grind"


def test_columns_decide_order_and_selection():
    """Storage may grow fields; the export must not shift columns silently."""
    rows = [{"score": 8, "ticker": "ABC", "ok": True, "internt_id": "x"}]
    out = ce.rows_to_csv(rows, COLS)
    assert out.splitlines()[1] == "ABC;8;Ja"


def test_booleans_read_as_swedish_not_python():
    rows = [{"ticker": "A", "score": 1, "ok": False}]
    assert out_line(rows) == "A;1;Nej"


def out_line(rows) -> str:
    return ce.rows_to_csv(rows, COLS).splitlines()[1]


def test_missing_values_become_empty_cells_not_none():
    assert out_line([{"ticker": "A"}]) == "A;;"


def test_semicolons_and_quotes_in_text_survive():
    rows = [{"ticker": "A;B", "score": 'säg "nej"', "ok": True}]
    line = out_line(rows)
    assert '"A;B"' in line
    assert 'säg ""nej""' in line


def test_non_dict_rows_are_skipped_rather_than_crashing():
    out = ce.rows_to_csv([{"ticker": "A", "score": 1, "ok": True}, None, "junk"],
                         COLS)
    assert len(out.strip().splitlines()) == 2


def test_none_input():
    assert ce.rows_to_csv(None, COLS).strip() == "Ticker;Poäng;Grind"


def test_filename_carries_the_date():
    assert ce.filename("insider", date(2026, 8, 15)) == "insider_2026-08-15.csv"
