"""Designförslaget PR 10: två sidor färre — Guld och Sentiment med inre flikar, Marknad märkt som bakgrund."""
import pathlib

from ui import nav

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_two_pages_fewer():
    assert nav.options("odin/Råvaror")[-1] == "🥇 Guld"
    assert "🥇 Guldkvoter" not in nav.leaves() and "Retail Pulse" not in nav.leaves()
    assert nav.options("odin/Makro") == ["🐺 Fiat Debasement", "Sentiment"]


def test_old_links_land_on_the_merged_pages():
    assert nav.resolve("odin/ravaror/guld-silver") == ["odin", "Råvaror", "🥇 Guld"]
    assert nav.resolve("regime/ravaror/guldkvoter") == ["odin", "Råvaror", "🥇 Guld"]
    assert nav.resolve("intelligence/retail-pulse") == ["odin", "Makro", "Sentiment"]
    assert nav.resolve("odin/marknad/retail-pulse") == ["odin", "Makro", "Sentiment"]


def test_only_the_open_inner_tab_is_drawn_and_background_is_marked():
    src = (ROOT / "wolf_panel.py").read_text(encoding="utf-8")
    assert 'on_change="rerun", key=key' in src and 'getattr(tab, "open", None)' in src
    assert '_inner(["🥇🥈 Guld/Silver", "🥇 Guldkvoter"]' in src and '_inner(["Fear & Greed", "Retail"]' in src
    assert "Bakgrund — styr inga beslut" in src


def test_coming_soon_is_gone_from_the_sentiment_page():
    src = (ROOT / "sentiment" / "sentiment_streamlit.py").read_text(encoding="utf-8")
    page = src.split("def render_sentiment_page")[1]
    assert "_render_retail_flow()" not in page and "placeholder" not in page.lower()
