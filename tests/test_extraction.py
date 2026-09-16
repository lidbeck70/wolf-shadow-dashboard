"""
Copilot-extraktionen ur presentationer: prompten listar arkets fält, svaret
parsas tolerant, förslagen mappas till arkets fältnycklar — och ingenting
av det rör poäng, status eller regler. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ai import extract_prompt as xp
from ai import document as doc


def test_prompt_lists_sheet_fields_and_document():
    p = xp.build_extract_prompt("rule", "BOL.ST", "Boliden", "[Sida 1]\nAISC 1,250 USD/oz")
    assert "Ark: Rick Rule" in p and "unit_cost" in p and "mine_life" in p
    assert '"fields"' in p and "[Sida 1]" in p
    with pytest.raises(ValueError):
        xp.build_extract_prompt("okänt", "X", "", "")
    t = xp.build_extract_prompt("tiggre", "KDK", "Kodiak", "x")
    assert "catalysts" in t and "nav" in t


def test_document_is_clipped():
    big = "a" * (xp.MAX_DOC_CHARS + 500)
    out = xp.clip_document(big)
    assert len(out) < len(big) and "KLIPPT" in out


def test_parse_tolerates_fences_and_prose():
    body = '{"fields": {"unit_cost": {"value": 1250, "unit": "USD/oz", "page": 8, "quote": "AISC of $1,250/oz", "confidence": "high"}}, "notes": []}'
    assert xp.parse_extraction(body)["fields"]["unit_cost"]["value"] == 1250
    assert xp.parse_extraction(f"Här är svaret:\n```json\n{body}\n```\nKlart.")["fields"]
    assert xp.parse_extraction(f"Svar: {body} slut")["fields"]
    with pytest.raises(xp.ExtractionError):
        xp.parse_extraction("Jag hittade inget.")
    with pytest.raises(xp.ExtractionError):
        xp.parse_extraction('{"not": "schema"}')


def test_proposals_keep_sheet_order_and_types():
    parsed = {"fields": {
        "mine_life": {"value": "12 år", "unit": "år", "page": 4, "quote": "12-year LOM",
                      "confidence": "high"},
        "unit_cost": {"value": 1250.5, "unit": "USD/oz", "page": "8", "quote": "AISC $1,250",
                      "confidence": "medium"},
        "rp_ratio": {"value": None},
        "jurisdiction": {"value": "Nevada, USA", "page": 2, "quote": "Nevada"},
        "capital_discipline": {"value": ""},
    }}
    props = xp.proposals("rule", parsed)
    assert [p["key"] for p in props] == ["unit_cost", "mine_life", "jurisdiction"]
    assert props[0]["value"] == 1250.5 and props[0]["page"] == 8 and props[0]["apply"] is True
    assert props[1]["value"] == 12.0                      # "12 år" → tal
    assert props[2]["kind"] == "text" and props[2]["apply"] is False   # läsning, skrivs aldrig


def test_tiggre_bools_and_catalysts():
    parsed = {"fields": {
        "nav": {"value": 650, "unit": "MUSD", "page": 12, "quote": "after-tax NPV5% US$650M",
                "confidence": "high"},
        "fs": {"value": "true", "page": 3, "quote": "DFS completed"},
        "permits": {"value": False, "page": 5, "quote": "permit application pending"},
        "funded": {"value": None}},
        "catalysts": [{"name": "Finansieringsbesked", "date": "2026-Q4", "page": 20},
                      {"name": "utan datum"}, "skräp"]}
    props = {p["key"]: p for p in xp.proposals("tiggre", parsed)}
    assert props["nav"]["value"] == 650.0
    assert props["fs"]["value"] is True and props["permits"]["value"] is False
    assert "funded" not in props
    assert xp.catalysts(parsed) == [{"name": "Finansieringsbesked", "date": "2026-Q4", "page": 20}]


def test_every_sheet_has_specs_and_apply_flags():
    for sheet, specs in xp.FIELDS.items():
        assert specs and all(s.kind in ("number", "bool", "text") for s in specs)
        # textfynd skrivs aldrig in — bedömningen är användarens
        assert all(not s.apply for s in specs if s.kind == "text")


def test_pasted_text_gets_page_markers():
    txt = doc.text_with_pages("x" * 8000, chars_per_page=3500)
    assert doc.page_count(txt) == 3 and txt.startswith("[Sida 1]")
    assert doc.text_with_pages("[Sida 2]\nredan markerad") == "[Sida 2]\nredan markerad"


def test_pdf_to_text_needs_pypdf_or_explains():
    try:
        import pypdf  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError):
            doc.pdf_to_text(b"%PDF-1.4")
