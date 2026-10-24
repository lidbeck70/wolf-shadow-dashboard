"""Designförslaget PR 3: ordlistan — ett huvudord per betydelse, källans ord som chip."""
from thor import candidates as tc
from thor import cards
from ui import vocab


def test_each_meaning_has_one_main_word():
    for main, words in vocab.WORDS.items():
        for w in words:
            assert vocab.main(w) == main, w


def test_pass_depends_on_context():
    assert vocab.main("PASS", vocab.CHECK) == vocab.OK
    assert vocab.main("MANUAL", vocab.CHECK) == vocab.KOLLA
    assert vocab.main("FAIL", vocab.CHECK) == vocab.NEJ
    assert vocab.main("PASS") == vocab.NEJ


def test_watch_on_a_holding_is_vaksam():
    assert vocab.main("WATCH", vocab.HOLDING) == vocab.VAKSAM
    assert vocab.main("WATCH") == vocab.BEVAKA


def test_neutral_and_unknown_are_shown_as_is():
    assert vocab.main("Neutral") is None and vocab.pair("Neutral") == ("Neutral", "")
    assert vocab.main("INGEN SIGNAL") is None


def test_compound_source_words_read_on_the_first_part():
    assert vocab.main("KÖP — logga i journalen") == vocab.KOP
    assert vocab.main("BUY (4/4)") == vocab.KOP
    assert vocab.main("HOLD · INNEHAV") == vocab.HALL


def test_pair_keeps_source_word_only_when_different():
    assert vocab.pair("GOLDEN TICKET") == ("KÖP", "GOLDEN TICKET")
    assert vocab.pair("KÖP") == ("KÖP", "")
    assert vocab.pair("READY") == ("BEVAKA", "READY")


def _c(status, cls):
    return tc.Candidate("ABC.ST", "ABC", "viking_nine", status, cls)


def test_thor_chip_shows_main_word_and_source():
    html = cards.status_chip(_c("GOLDEN TICKET", tc.KOP))
    assert ">KÖP<" in html and ">GOLDEN TICKET<" in html
    assert cards.status_chip(_c("KÖP", tc.KOP)).count("<span") == 1


def test_blocked_and_hold_classes_use_the_glossary():
    assert cards.main_word(_c("RÖD regim", tc.SPARRAD)) == "NEJ"
    assert cards.main_word(_c("HOLD", tc.HALL)) == "BEVAKA"
    assert cards.main_word(_c("träff", tc.TRAFF)) == "GRANSKA"


def test_verdict_card_shows_main_word_with_source_chip():
    html = cards.verdict_card("NO TRADE", "x")
    assert ">NEJ<" in html and ">NO TRADE<" in html
    assert ">INGEN SIGNAL<" in cards.verdict_card("INGEN SIGNAL", "x")
