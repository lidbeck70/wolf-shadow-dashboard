"""
Tests for commodity_book.py — Råvarukartboken (Masterguiden Del 5).

The chapters are the depth behind rotation.py's one-liners, so the binding
test is that neither side can grow a row the other does not know about.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commodity_book as book
import rotation as r


# ── Bindningen mot rotationen ────────────────────────────────────────────────
def test_every_chapter_belongs_to_a_real_commodity():
    for c in book.CHAPTERS:
        assert c.key in r.COMMODITY_BY_KEY, f"kartbok-kapitel utan råvara: {c.key}"


def test_every_commodity_has_a_chapter_except_royalty():
    """Royalty is a strategy, not a commodity cycle — Del 4 owns it."""
    missing = [c.key for c in r.COMMODITIES if not book.has_chapter(c.key)]
    assert missing == list(book.NO_CHAPTER) == ["royalty"]
    assert len(book.CHAPTERS) == len(r.COMMODITIES) - 1 == 12


def test_chapter_order_follows_the_rotation_table():
    """So the deep-dive reads in the same order as the grid you grade in."""
    keys = [c.key for c in book.CHAPTERS]
    expected = [c.key for c in r.COMMODITIES if c.key not in book.NO_CHAPTER]
    assert keys == expected


def test_lookup_helpers():
    assert book.chapter("uran").subtitle == "kontraktscykelns metall"
    assert book.chapter("royalty") is None
    assert book.chapter("nonsense") is None
    assert not book.has_chapter("royalty")


# ── Innehållet ───────────────────────────────────────────────────────────────
def test_every_chapter_answers_the_three_core_questions():
    for c in book.CHAPTERS:
        assert len(c.market) > 80, c.key
        assert len(c.play) > 80, c.key
        assert c.timing, c.key
        assert c.subtitle, c.key


def test_every_chapter_names_where_the_number_lives():
    """A timing rule you cannot look up is not a timing rule."""
    for c in book.CHAPTERS:
        assert c.sources, f"{c.key} saknar källa"


def test_the_never_hold_metals_carry_their_warning():
    """The guide singles these out: they are never owned through a top."""
    for key in ("palladium", "litium", "silver"):
        assert book.chapter(key).pitfall, key
    assert "aldrig ägas genom en topp" in book.chapter("palladium").pitfall
    assert "EUFORIN" in book.chapter("litium").pitfall
    assert "mani-topp" in book.chapter("silver").pitfall


def test_gold_is_the_only_chapter_with_a_portfolio_role():
    """Its role — rising in risk aversion — is why it never rotates out."""
    with_role = [c.key for c in book.CHAPTERS if c.role]
    assert with_role == ["guld"]
    assert "riskaversion" in book.chapter("guld").role
    assert r.COMMODITY_BY_KEY["guld"].anchor is True


def test_key_numbers_survive_the_move_from_the_guide():
    assert "$5/lb" in book.chapter("uran").play              # MCap ÷ Mlbs U3O8
    assert "$45 WTI" in book.chapter("olja").play
    assert "$2,5/MMBtu" in book.chapter("gas").market
    assert "20 %" in book.chapter("kol").timing              # FCF-yield
    assert "$50/t" in book.chapter("jarnmalm").play          # C1
    assert "24 månader" in book.chapter("litium").play       # runway


def test_chapters_agree_with_the_rotation_buy_signals():
    """Both sides quote the same trigger price — drift here is a real bug."""
    pairs = [("uran", "$80–90/lb"), ("koppar", "$4,5/lb")]
    for key, number in pairs:
        assert number in r.COMMODITY_BY_KEY[key].buy_signal, key
        ch = book.chapter(key)
        assert number in (ch.timing + ch.play + ch.market), key


# ── Rendering ────────────────────────────────────────────────────────────────
def test_chapter_prose_is_escaped_not_trusted_as_markup():
    """The chapters are prose full of "< 5" / "> 40 %" thresholds.

    A bare "<" followed by a space renders as text in browsers, so today's
    content is safe either way — but the module is data, not markup, and one
    future chapter starting a word right after "<" would silently eat a
    threshold. Escaping makes that impossible.
    """
    html = r.chapter_html(book.chapter("gas"))
    assert "EV/EBITDA &lt; 5" in html
    assert "hedgebok &gt; 40 %" in html


def test_every_chapter_renders_without_losing_its_numbers():
    from html import unescape
    for c in book.CHAPTERS:
        html = r.chapter_html(c)
        for field in (c.market, c.play, c.timing, c.pitfall, c.role):
            if field:
                assert field in unescape(html), c.key
