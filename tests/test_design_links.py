"""Designförslaget PR 1: varje larm länkar till rätt vy via målnycklarna i ui/nav.TARGETS."""
import ast
import pathlib

import alert_rules as ar
from ui import nav

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_every_target_resolves_to_its_full_path():
    for key, segs in nav.TARGETS.items():
        got = nav.resolve(nav.target_slug(key))
        assert got == list(segs), key


def test_unknown_target_goes_home():
    assert nav.target_slug("finns_inte") == "home"


def test_target_link_with_and_without_base():
    assert nav.target_link("market_risk") == "?p=odin/laget"
    url = nav.target_link("thor", "https://panel.example/", ticker="VOLV-B.ST", strategy="viking_nine")
    assert url == "https://panel.example/?p=thor&t=VOLV-B.ST&s=viking_nine"


def _alert_calls():
    tree = ast.parse((ROOT / "alert_rules.py").read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_alert"]


def test_every_alert_kind_has_a_target():
    calls = _alert_calls()
    assert len(calls) >= 17
    for c in calls:
        assert any(k.arg == "target" for k in c.keywords), f"rad {c.lineno} saknar target"


def test_every_literal_target_exists():
    targets = set()
    for c in _alert_calls():
        for k in c.keywords:
            if k.arg == "target" and isinstance(k.value, ast.Constant):
                targets.add(k.value.value)
    targets |= set(ar.SCREEN_TARGET.values()) | set(ar.SHEET_TARGET.values())
    assert targets <= set(nav.TARGETS), targets - set(nav.TARGETS)


def test_format_without_base_is_unchanged():
    a = ar._alert("x", "Titel", "Kropp", target="thor", ticker="ABC.ST", strategy="wolf")
    assert ar.format_alert(a) == "🐺 Titel\nKropp"
    assert ar.format_alert(a, "") == "🐺 Titel\nKropp"


def test_format_with_base_adds_the_link():
    a = ar._alert("x", "Titel", "Kropp", target="thor", ticker="ABC.ST", strategy="wolf")
    assert ar.format_alert(a, "https://p.example").endswith("\n→ https://p.example/?p=thor&t=ABC.ST&s=wolf")


def test_alert_without_target_has_no_link():
    assert ar.format_alert({"kind": "x", "title": "T", "body": "B"}, "https://p.example") == "🐺 T\nB"


def test_sheet_kinds_map_to_their_sheet():
    assert ar.sheet_target("tiggre_free_ride") == "tiggre"
    assert ar.sheet_target("durrett_engine_buy_rule") == "durrett"
    assert ar.sheet_target("insider_stop") == "insider"
    assert ar.sheet_target("okänd") == ""


def test_sheet_alert_carries_ticker_and_target():
    data = {"events": [{"key": "tiggre_free_ride:x", "kind": "tiggre_free_ride", "ticker": "ABC.ST",
                        "title": "T", "body": "B"}]}
    alerts, _ = ar.sheet_alerts(data, {"events": {}})
    assert alerts[0]["target"] == "tiggre" and alerts[0]["ticker"] == "ABC.ST"


def test_wolf_alert_opens_thor_with_strategy():
    prev = {"qualified": {}}
    alerts, _ = ar.wolf_alerts({"rows": [{"ticker": "abc.st", "name": "A", "score": 100}]}, prev)
    assert alerts[0]["target"] == "thor" and alerts[0]["strategy"] == "wolf" and alerts[0]["ticker"] == "ABC.ST"
    assert "THOR → Wolf" not in alerts[0]["body"]


def test_swing_exit_opens_holdings():
    a = ar._alert("swing_exit", "t", "b", target="holdings", ticker="X")
    assert nav.target_slug(a["target"]) == "portfolio/positioner"


class _St:
    def __init__(self, params):
        self.query_params = params
        self.session_state = {}


def test_thor_link_selects_ticker_and_strategy():
    from thor import ui as tui
    st = _St({"t": "VOLV-B.ST", "s": "viking"})
    tui._apply_links(st)
    assert st.session_state["thor_ticker"] == "VOLV-B.ST"
    assert st.session_state["thor_strategy"] == "Viking" and st.session_state["_thor_strategy_set"]


def test_thor_link_unknown_strategy_only_sets_ticker():
    from thor import ui as tui
    st = _St({"t": "VOLV-B.ST", "s": "nope"})
    tui._apply_links(st)
    assert st.session_state["thor_ticker"] == "VOLV-B.ST" and "thor_strategy" not in st.session_state


def test_thor_link_applies_once():
    from thor import ui as tui
    st = _St({"t": "VOLV-B.ST"})
    tui._apply_links(st)
    st.session_state["thor_ticker"] = ""                     # användaren stängde analysen
    tui._apply_links(st)
    assert st.session_state["thor_ticker"] == ""


def test_workflows_pass_panel_url():
    for f in ("scheduled-scan.yml", "wolf-data.yml", "berserk-scan.yml"):
        assert "PANEL_URL: ${{ secrets.PANEL_URL }}" in (ROOT / ".github" / "workflows" / f).read_text()


def test_kind_target_table_matches_the_alert_calls():
    for c in _alert_calls():
        kind = c.args[0].value if c.args and isinstance(c.args[0], ast.Constant) else None
        target = next((k.value.value for k in c.keywords if k.arg == "target" and isinstance(k.value, ast.Constant)), None)
        if kind and target:
            key = "fiat" if str(kind).startswith("fiat_") else kind
            assert ar.KIND_TARGET[key][1] == target, kind
    for _head, target in ar.KIND_TARGET.values():
        assert target in nav.TARGETS
