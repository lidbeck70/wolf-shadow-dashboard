"""
Larmen till Discord: Deep Contrarian skickade gamla listor (Industri,
Dagligvaror) om och om igen.

  1. Ett enda Discord 429 gjorde att tillståndet aldrig sparades — hela
     skörden skickades igen varje körning. Nu: vänta och försök igen vid 429,
     paus mellan larmen, och tillståndet sparas när något nått fram.
  2. Deep-listan i Gisten var från före råvarugrinden. Nu larmar Deep bara på
     en lista märkt commodity_only=True.
  3. Quality och Deep delade lokal reservfil — nu en per läge.
  4. Schemagrinden hoppade över försenade körningar — nu avgör vilken cron
     som startade jobbet, inte klockslaget.
"""
import io
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import alert_rules as ar  # noqa: E402


# ── 1. Discord 429 och sparlogiken ───────────────────────────────────────────
class _Resp:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http429(retry_after=0.3):
    body = io.BytesIO(json.dumps({"message": "You are being rate limited.", "retry_after": retry_after}).encode())
    return urllib.error.HTTPError("https://discord.com/api/webhooks/x", 429, "Too Many Requests", {}, body)


def test_discord_waits_and_retries_on_429(monkeypatch):
    from alerts.channels import discord
    monkeypatch.setattr(discord, "secret", lambda k: "https://discord.com/api/webhooks/1/abc")
    calls, waits = [], []

    def _open(req, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise _http429(0.3)
        return _Resp()
    monkeypatch.setattr(discord.urllib.request, "urlopen", _open)
    monkeypatch.setattr(discord, "_sleep", waits.append)
    assert discord.send("hej") is True
    assert len(calls) == 3 and waits == [0.3, 0.3]


def test_discord_gives_up_after_max_retries(monkeypatch):
    from alerts.channels import discord
    monkeypatch.setattr(discord, "secret", lambda k: "https://discord.com/api/webhooks/1/abc")
    calls = []

    def _open(req, timeout=None):
        calls.append(1)
        raise _http429(60)                                   # orimligt lång väntan kapas
    waits = []
    monkeypatch.setattr(discord.urllib.request, "urlopen", _open)
    monkeypatch.setattr(discord, "_sleep", waits.append)
    assert discord.send("hej") is False
    assert len(calls) == discord._MAX_RETRIES + 1 and max(waits) == discord._MAX_WAIT_S
    assert "429" in discord.last_error


def test_deliver_paces_counts_and_names_the_missed():
    import alert_scan
    alerts = [{"kind": "k", "title": f"T{i}", "channels": ["discord"]} for i in range(3)]
    results = iter([{"discord": True}, {"discord": False}, {"discord": True}])
    pauses = []
    delivered, failed = alert_scan.deliver(alerts, lambda text, ch, metadata=None: next(results),
                                           lambda a: a["title"], pause=pauses.append)
    assert delivered == 2 and failed == ["T1"]
    assert pauses == [alert_scan.SEND_PAUSE_S] * 2               # ingen paus före det första


def test_state_is_saved_when_something_got_through():
    src = open(os.path.join(ROOT, "alert_scan.py"), encoding="utf-8").read()
    main = src[src.index("def main()"):]
    assert "if alerts and not delivered:" in main                 # sparas inte bara när INGET nådde fram
    assert "delivered_all" not in main


# ── 2. Gamla Deep-listor larmar inte ─────────────────────────────────────────
def _ca(*tickers, commodity_only=True):
    d = {"timestamp": "2026-09-29T10:00", "mode": "deep_contrarian",
         "results": [{"ticker": t, "name": t, "rank": i + 1, "composite_score": 50.0,
                      "necessity_score": 80, "hat_score": 50, "sector": "Råvaror"} for i, t in enumerate(tickers)]}
    if commodity_only is not None:
        d["commodity_only"] = commodity_only
    return d


def test_deep_list_without_the_commodity_gate_is_ignored():
    _a, state = ar.contrarian_alerts(_ca("BOL.ST"), None)
    stale = _ca("BOL.ST", "XANO-B.ST", "SNX.ST", commodity_only=None)     # från före #98
    alerts, state2 = ar.contrarian_alerts(stale, state)
    assert alerts == [] and state2 == state                               # baslinjen fryser
    alerts, _ = ar.contrarian_alerts(_ca("BOL.ST", "LUMI.ST"), state2)
    assert [a["title"] for a in alerts] == ["🎯 Deep Contrarian: LUMI.ST in i listan (#2)"]


def test_saved_payload_carries_mode_and_the_gate(monkeypatch, tmp_path):
    from contrarian_alpha import cache
    from contrarian_alpha.engine import PipelineConfig, PipelineResult
    monkeypatch.setattr(cache, "_LOCAL_FALLBACK", str(tmp_path / ".ca_results.json"))
    monkeypatch.setattr(cache, "_get_github_token", lambda: "")
    for mode, gate in (("deep_contrarian", 12), ("quality", None)):
        res = PipelineResult(results=[], universe_count=100, necessity_passed=0, hate_passed=0, bs_passed=0,
                             composite_ranked=0, run_duration_s=1.0, config=PipelineConfig(mode=mode), eliminated=[], timestamp="t",
                             commodity_passed=gate)
        cache.save_screener_results(res, mode=mode)
    deep = json.load(open(tmp_path / ".ca_results_deep_contrarian.json"))
    qual = json.load(open(tmp_path / ".ca_results_quality.json"))
    assert deep["mode"] == "deep_contrarian" and deep["commodity_only"] is True
    assert qual["mode"] == "quality" and qual["commodity_only"] is False


def test_local_fallback_is_per_mode(monkeypatch, tmp_path):
    """Förut: Deep-läsningen föll tillbaka på den delade filen — med Quality-listan i."""
    from contrarian_alpha import cache
    monkeypatch.setattr(cache, "_LOCAL_FALLBACK", str(tmp_path / ".ca_results.json"))
    monkeypatch.setattr(cache, "_GIST_API_URL", "http://127.0.0.1:9/none")          # Gisten nås inte
    with open(tmp_path / ".ca_results_quality.json", "w") as f:
        json.dump({"timestamp": "t", "mode": "quality", "results": [{"ticker": "XANO-B.ST"}]}, f)
    assert cache.load_screener_results("deep_contrarian").get("results") == []
    assert cache.load_screener_results("quality")["results"][0]["ticker"] == "XANO-B.ST"


# ── 4. Schemagrinden ─────────────────────────────────────────────────────────
def test_schedule_gate_uses_the_triggering_cron_not_the_clock():
    script = open(os.path.join(ROOT, "scripts", "schedule_gate.sh"), encoding="utf-8").read()
    assert "+0200" in script and "+0100" in script and "SCHEDULE" in script
    assert "date +%H" not in script                                       # klockslaget avgör inte
