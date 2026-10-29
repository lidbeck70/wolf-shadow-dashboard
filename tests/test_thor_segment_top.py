"""
Thor: segmentets största bolag i listan även utan signal ("Största i segmentet", thor/segment_top.py) — temats lista
(uran, palladium), komplexen varvet runt i kartans ordning, sektorlistan ur Universum-registrets blob (5 Norden +
5 USA, 10 på en vald marknad), filtren, bolag med en riktig rad i valet, radslaget utan strategi och korten.
Raderna hamnar aldrig i data["rows"]. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import cards  # noqa: E402
from thor import segment_top as stp  # noqa: E402
from thor import ui as tui  # noqa: E402

DATA = {**fx.BLOBS, "regime": {"regime": "GRÖN"}}


def _rows():
    return tc.build(DATA, today="2026-10-15")


def _top(choice, pick=tui.ALL, filters=(), markets=(), blob=None, rows=None, names=None):
    return stp.top_rows(choice, _rows() if rows is None else rows, pick, list(filters), list(markets), fx.BOARD,
                        blob, names)


def _sector_blob():
    def lst(prefix, n, suffix=""):
        return [{"ticker": f"{prefix}{i}{suffix}", "name": f"{prefix} {i}", "mcap_usd": 1000 * (n - i)} for i in range(1, n + 1)]
    return {"generated": "2026-10-26T05:17:00+00:00", "source": "Börsdata KPI 50/62",
            "segments": {"healthcare": {"norden": lst("NO", 10, ".ST"), "usa": lst("US", 10), "kanada": lst("CA", 3, ".TO")}},
            "themes": {}}


# ── Temana ──────────────────────────────────────────────────────────────────
def test_uranium_shows_the_biggest_without_a_row_and_ccj_keeps_its_real_row():
    rows = _rows()
    before = list(rows)
    extra, note = _top("tema:uran", rows=rows)
    assert [c.ticker for c in extra] == ["BWXT", "PDN.TO", "UROY"]                    # CCJ har EMBER:s KÖP-rad
    assert all(c.strategy == tc.TOP and c.status_class == tc.INGEN and c.theme == "uran" and c.segment == "energy"
               for c in extra)
    assert [c.status for c in extra] == ["STÖRST #2", "STÖRST #3", "STÖRST #4"]       # rangen i temalistan
    assert all(c.entry is None and c.stop is None and c.score is None for c in extra)   # aldrig ett köp
    assert "Har signal ovan: CCJ." in note and "Uran har bara 4 noterade bolag" in note and "Börsdata 2026-10-27" in note
    assert extra[0].why.startswith("Ingen signal i dag · störst börsvärde med positivt kassaflöde")
    assert rows == before and all(c.strategy != tc.TOP for c in rows)                   # data["rows"] rörs inte


def test_palladium_shows_its_four_companies_although_no_row_has_the_theme():
    assert not [c for c in _rows() if c.theme == "palladium"]
    extra, note = _top("tema:palladium")
    assert [c.ticker for c in extra] == ["VALT.L", "SBSW", "UMI.BR", "JMAT.L"]
    assert {c.segment_status for c in extra} == {"RÖD"} and {c.cykel for c in extra} == {"TOPP"}   # kartans läge
    assert "Palladium har bara 4 noterade bolag" in note
    assert stp.title("tema:palladium", tui.ALL) == "STÖRST I PALLADIUM — INGEN SIGNAL I DAG"


def test_complex_is_round_robin_in_map_order_and_deterministic():
    assert stp.ordered_themes({"uran", "olja", "kol"}) == ["olja", "kol", "uran"]
    extra, note = _top("komplex:energi")
    assert [c.ticker for c in extra][:5] == ["XOM", "LNG", "TTE.PA", "GLEN.L", "CVX"]
    assert extra[0].status == "STÖRST #1 i Olja" and extra[3].status == "STÖRST #1 i Kol"
    assert "CCJ" not in [c.ticker for c in extra] and "Har signal ovan: CCJ." in note
    assert [c.ticker for c in _top("komplex:energi")[0]] == [c.ticker for c in extra]
    assert len({c.ticker for c in extra}) == len(extra) <= stp.N                          # TTE.PA står i två teman
    assert stp.title("komplex:adelmetaller", tui.ALL) == "STÖRST I KOMPLEXET ÄDELMETALLER — INGEN SIGNAL I DAG"


def test_blob_themes_are_preferred_with_names_and_market_cap():
    blob = {"generated": "2026-10-26T05:17", "themes": {"uran": [{"ticker": "KAP.L", "name": "Kazatomprom", "mcap_usd": 9000},
                                                                 {"ticker": "CCJ", "name": "Cameco", "mcap_usd": 30000}]}}
    extra, note = _top("tema:uran", blob=blob)
    assert [(c.ticker, c.name, c.status) for c in extra] == [("KAP.L", "Kazatomprom", "STÖRST #1")]
    assert extra[0].score_text == "≈ 9 000 MUSD börsvärde" and "Börsdata 2026-10-26" in extra[0].why
    assert _top("tema:uran", names={"BWXT": "BWX Technologies"})[0][0].name == "BWX Technologies"   # registrets namn


# ── Filtren och valet ───────────────────────────────────────────────────────
def test_buy_new_and_all_segments_hide_the_section():
    for flt in (["bara KÖP"], ["bara nya"], ["bara Norden", "bara KÖP"]):
        assert _top("tema:palladium", filters=flt) == ([], "")
    assert _top(tui.SEGMENT_ALL) == ([], "") and _top(None) == ([], "")


def test_each_pick_dedupes_against_its_own_rows():
    extra, note = _top("tema:uran", pick="BERSERK")                       # CCJ:s rad är EMBER:s, inte BERSERK:s
    assert [c.ticker for c in extra] == ["CCJ", "BWXT", "PDN.TO", "UROY"] and "Har signal ovan" not in note
    assert extra[0].why.startswith("Ingen BERSERK-signal i dag")
    assert stp.title("tema:uran", "BERSERK") == "STÖRST I URAN — INGEN BERSERK-SIGNAL I DAG"
    assert "CCJ" not in [c.ticker for c in _top("tema:uran", pick="EMBER")[0]]
    fam, _n = _top("tema:uran", pick="Råvara (alla)")
    assert "CCJ" not in [c.ticker for c in fam] and fam[0].why.startswith("Ingen Råvara-signal i dag")
    assert "FCX" not in [c.ticker for c in _top("tema:koppar", pick="BERSERK")[0]]       # FCX: BERSERK BEVAKA
    assert "FCX" in [c.ticker for c in _top("tema:koppar", pick="Wolf")[0]]


def test_nordic_only_and_market_pills_apply():
    assert [c.ticker for c in _top("tema:koppar", filters=["bara Norden"])[0]] == ["LUMI.ST"]   # BOL.ST har signal
    assert all(c.nordic for c in _top("komplex:basmetaller", filters=["bara Norden"])[0])
    assert [c.ticker for c in _top("tema:uran", markets=["USA"])[0]] == ["BWXT", "UROY"]
    extra, note = _top("tema:uran", filters=["bara Norden"])
    assert extra == [] and "Inga nordiska bolag bland de största i Uran." in note
    assert [c.ticker for c in _top("tema:palladium", markets=["London", "Europa"])[0]] == ["VALT.L", "UMI.BR", "JMAT.L"]


def test_sector_list_reads_the_blob_five_plus_five_or_ten_on_one_market():
    blob = _sector_blob()
    extra, note = _top("healthcare", blob=blob)
    assert [c.ticker for c in extra] == ["NO1.ST", "NO2.ST", "NO3.ST", "NO4.ST", "NO5.ST", "US1", "US2", "US3", "US4", "US5"]
    assert extra[0].status == "STÖRST #1 i Norden" and extra[5].status == "STÖRST #1 i USA" and extra[0].segment == "healthcare"
    assert extra[0].name == "NO 1" and extra[0].score_text == "≈ 9 000 MUSD börsvärde" and "Börsdata 2026-10-26" in note
    assert [c.ticker for c in _top("healthcare", markets=["USA"], blob=blob)[0]] == [f"US{i}" for i in range(1, 11)]
    assert [c.ticker for c in _top("healthcare", filters=["bara Norden"], blob=blob)[0]] == [f"NO{i}.ST" for i in range(1, 11)]
    three = [c.ticker for c in _top("healthcare", markets=["Norden", "USA", "Kanada"], blob=blob)[0]]
    assert three == ["NO1.ST", "NO2.ST", "NO3.ST", "NO4.ST", "US1", "US2", "US3", "CA1.TO", "CA2.TO", "CA3.TO"]
    extra, note = _top("healthcare", markets=["London"], blob=blob)
    assert extra == [] and "Sektorlistan räknar bara Norden, USA och Kanada." in note
    assert stp.market_split([]) == [("norden", 5), ("usa", 5)] and stp.market_split(["usa"]) == [("usa", 10)]
    assert stp.market_split(["usa", "kanada"], nordic_only=True) == [] and stp.market_split([], True) == [("norden", 10)]


def test_a_company_with_a_row_elsewhere_is_named_not_duplicated():
    blob = _sector_blob()
    blob["segments"]["healthcare"]["usa"][0] = {"ticker": "NEM", "name": "Newmont", "mcap_usd": 60000}
    extra, note = _top("healthcare", blob=blob)
    assert "NEM" not in [c.ticker for c in extra] and "Har en rad i listan men utanför det här urvalet: NEM." in note


def test_missing_sector_list_gives_a_note_not_a_crash():
    extra, note = _top("healthcare")
    assert extra == [] and "Universum-registret (måndagar)" in note and "universe-scan" in note
    assert _top("healthcare", blob={"generated": "x", "segments": "trasig"})[0] == []
    mat, note = _top("materials")                                            # segment med teman: temanas största
    assert mat and all(c.status.startswith("STÖRST #1 i ") for c in mat) and "Sektorlistan saknas än" in note


# ── Radslaget och korten ────────────────────────────────────────────────────
def test_candidate_tolerates_a_key_outside_the_registry():
    c = tc.Candidate("SBSW", "Sibanye", tc.TOP, "STÖRST #2", tc.INGEN)
    assert c.family == "" and c.label == "Största i segmentet" and c.as_dict()["label"] == "Största i segmentet"
    assert tc.Candidate("X", "", "okänd", "", tc.HALL).label == "okänd"
    s = tc.summary([c])
    assert s["n"] == 1 and s["buy"] == 0 and s["by_family"] == {"": 1}
    assert tc.sort_key(c)[0] == tc.CLASS_ORDER[tc.INGEN] == 6
    assert tc.filter_rows([c], only_buy=True) == [] and tc.filter_rows([c], family="råvara") == []
    assert tc.TOP not in tc.reg.STRATEGIES                                    # aldrig i Screening/'Bedöm som'


def test_card_shows_the_label_and_no_signal_never_the_key():
    extra, _n = _top("tema:palladium")
    html = cards.candidate_card(extra[1])
    assert "Största i segmentet" in html and "INGEN SIGNAL" in html and "STÖRST #2" in html and "SBSW" in html
    assert ">storst<" not in html and "storst" not in html
    assert f"border-left:4px solid {cards.GREY}" in html                     # grått kort
    assert cards.main_word(extra[1]) == "INGEN SIGNAL" and cards.family_chip(tc.TOP).count("Största i segmentet") == 1


def test_load_reads_the_segment_blob():
    blob = _sector_blob()
    out = tui.load(lambda name, fb: blob if name == "segment_top.json" else fb)
    assert out["segment_top"] is blob
    assert tui.load(lambda name, fb: fb)["segment_top"] is None
    broken = tui.load(lambda name, fb: (_ for _ in ()).throw(RuntimeError("nere")))
    assert broken["segment_top"] is None
