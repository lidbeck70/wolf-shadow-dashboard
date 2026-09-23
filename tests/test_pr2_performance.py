"""
PR 2 — prestanda ur panelgenomgången:
  bara den öppna toppfliken renderas · bannern är liten och kodas en gång ·
  en delad priscache (market_prices) som temabordet, kvotmodulen, Ember och
  Wolf Regime går via · refresh-knappar tömmer bara sin egen cache ·
  Blindspot laddar inte volym per ticker · Contrarian cachar Gist-läsningen.
"""
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel: str) -> str:
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


# ── bara öppen flik ───────────────────────────────────────────────────────────
def test_only_the_open_top_tab_renders():
    src = _src("wolf_panel.py")
    assert 'st.tabs(tab_labels, on_change="rerun", key="main_tabs")' in src
    main = src[src.index("def main():"):]
    tabs = re.findall(r'^    if _is_open\(tabs\["(\w+)"\]\):\n        with tabs\["\1"\]:', main, flags=re.M)
    assert len(tabs) == 10 and len(set(tabs)) == 10, tabs
    assert not re.search(r'^    with tabs\[', main, flags=re.M)          # ingen ovillkorad toppflik kvar
    import wolf_panel
    class _T:
        def __init__(self, o): self.open = o
    assert wolf_panel._is_open(_T(True)) and wolf_panel._is_open(_T(None)) and not wolf_panel._is_open(_T(False))


# ── bannern ───────────────────────────────────────────────────────────────────
def test_banner_is_a_small_jpeg_encoded_once(monkeypatch):
    path = os.path.join(ROOT, "assets", "banner.jpg")
    assert os.path.getsize(path) < 300_000                                 # var 2 018 602 B (PNG)
    with open(path, "rb") as f:
        assert f.read(3) == b"\xff\xd8\xff"                                 # riktig JPEG
    assert not os.path.exists(os.path.join(ROOT, "assets", "banner.png"))   # dubbletten borta
    from ui import theme
    theme._BANNER_B64.clear()
    a = theme._banner_b64()
    monkeypatch.setattr(theme.os.path, "join", lambda *a, **k: "/nope/banner.jpg")
    assert theme._banner_b64() is a and len(a) > 1000                      # cachad — läses inte om
    theme._BANNER_B64.clear()


# ── delad priscache ───────────────────────────────────────────────────────────
def _fake_download(calls):
    def dl(tickers, period="1y", **kw):
        calls.append((tuple(tickers) if isinstance(tickers, list) else tickers, period))
        idx = pd.date_range("2024-01-01", periods=300, freq="B")
        if isinstance(tickers, list) and len(tickers) > 1:
            cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
            df = pd.DataFrame(1.0, index=idx, columns=cols)
            for t in tickers:
                df[(t, "Close")] = range(1, 301)
            return df
        df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": range(1, 301), "Volume": 1000},
                          index=idx)
        return df
    return dl


def test_market_prices_caches_and_shares_downloads(monkeypatch):
    import yfinance as yf
    import market_prices as mp
    calls = []
    monkeypatch.setattr(yf, "download", _fake_download(calls))
    mp.clear()
    a = mp.close("GLD", "10y")
    b = mp.close("GLD", "10y")
    df = mp.ohlcv("GLD", "10y")
    assert len(a) == 300 and a.equals(b) and "Volume" in df and len(calls) == 1   # en nedladdning, tre läsningar
    assert mp.close("GLD", "2y").iloc[-1] == 300 and len(calls) == 2              # annan period = egen post
    d = mp.closes(["GLD", "GDX", "GLD"], "1y")
    assert set(d) == {"GLD", "GDX"} and len(calls) == 3 and calls[-1] == (("GLD", "GDX"), "1y")
    monkeypatch.setattr(yf, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nät")))
    mp.clear()
    assert mp.close("X", "1y").empty and mp.ohlcv("X", "1y").empty and mp.closes(["X"], "1y") == {}
    mp.clear()


def test_modules_download_through_the_shared_cache(monkeypatch):
    import yfinance as yf
    import market_prices as mp
    calls = []
    monkeypatch.setattr(yf, "download", _fake_download(calls))
    mp.clear()
    from blindspot import theme_board as tb
    from alpha_regime import commodity_ratios as cr
    from ember import regime as er
    assert len(tb._download_close("GLD", "10y")) == 300
    assert len(tb._download_volume("GLD", "2y")) == 300 and len(calls) == 1      # volym ur samma ram
    assert len(cr._download_close("GLD", "10y")) == 300 and len(calls) == 1      # delad mellan moduler
    assert len(er._close("GLD", "10y")) == 300 and len(calls) == 1
    assert len(er._close("DX-Y.NYB", "3mo")) == 300 and len(calls) == 2
    for rel, fn in (("blindspot/theme_board.py", "_download_close"), ("blindspot/theme_board.py", "_download_volume"),
                    ("alpha_regime/commodity_ratios.py", "_download_close"), ("ember/regime.py", "_close")):
        src = _src(rel)
        body = src[src.index(f"def {fn}("):]
        body = body[:body.index("\ndef ", 10)]
        assert "yf.download" not in body and "market_prices" in body, (rel, fn)
    assert "import yfinance" not in _src("alpha_regime/commodity_ratios.py")
    src = _src("tabs/regime.py")
    assert 'yf.download(benchmark_ticker, period="3mo"' not in src and "_mp_ohlcv(benchmark_ticker" in src
    mp.clear()


# ── refresh-knappar ───────────────────────────────────────────────────────────
def test_refresh_buttons_clear_only_their_own_caches():
    call = re.compile(r"^\s*st\.cache_data\.clear\(\)", re.M)             # som sats, inte i kommentar
    assert not call.search(_src("ovtlyr/ui/layout.py"))
    assert not call.search(_src("heatmap/heatmap_streamlit.py"))
    assert "_fetch_heatmap_data.clear()" in _src("heatmap/heatmap_streamlit.py")
    assert "_load_ohlcv, _load_sentiment, _load_sector_breadth" in _src("ovtlyr/ui/layout.py")


# ── Blindspot ─────────────────────────────────────────────────────────────────
def test_blindspot_sentiment_uses_batch_volume_not_per_ticker_download(monkeypatch):
    from blindspot import engine as be
    import retail_sentiment.sources.volume as vol
    monkeypatch.setattr(vol, "fetch_volume", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nedladdning per ticker")))
    monkeypatch.setattr("retail_sentiment.sources.reddit.fetch_reddit", lambda: {"data": {}}, raising=False)
    s = be._fetch_sentiment_data("CCJ", {"current_volume": 300.0, "avg_volume_20d": 1000.0})
    assert s["retail_flow_score"] == 10                                     # 0,3 < 0,5
    s = be._fetch_sentiment_data("CCJ", {"current_volume": 700.0, "avg_volume_20d": 1000.0})
    assert s["retail_flow_score"] == 30
    s = be._fetch_sentiment_data("CCJ", {"current_volume": 1200.0, "avg_volume_20d": 1000.0})
    assert s["retail_flow_score"] == 50
    s = be._fetch_sentiment_data("CCJ", None)                               # ingen prisrad → fallback, som sväljs
    assert s["retail_flow_score"] == 50
    assert 'results["price"].get(ticker)' in _src("blindspot/engine.py")


# ── Contrarian Alpha ──────────────────────────────────────────────────────────
def test_contrarian_gist_reads_are_cached_and_status_is_lazy():
    src = _src("contrarian_alpha/ui.py")
    assert "def _saved_results_cached(mode)" in src and src.count("_saved_results_cached(") >= 3
    assert 'st.expander("Cache & lagring", expanded=False, on_change="rerun"' in src and "if _exp.open:" in src
    assert src.count("load_screener_results(") == 2                        # bara inne i den cachade hjälparen


# ── cache-gränser ─────────────────────────────────────────────────────────────
def test_big_frame_caches_have_max_entries():
    for rel in ("cagr/cagr_streamlit.py", "market_cycle/cache.py", "market_prices.py"):
        src = _src(rel)
        assert "max_entries" in src, rel
    assert "@st.cache_data(ttl=3600, show_spinner=False)\n" not in _src("cagr/cagr_streamlit.py")


@pytest.mark.parametrize("rel", ["ui/theme.py", "ui/css.py"])
def test_no_module_reads_the_banner_per_rerun(rel):
    src = _src(rel)
    assert 'open(_banner_path, "rb")' not in src
