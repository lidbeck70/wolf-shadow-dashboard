"""
scripts/schedule_gate.sh — säsong + dubblettspärr för de schemalagda jobben.
GitHub försenade/tappade körningarna på jämna klockslag (30/9: 08 aldrig,
12 → 13:26, 18 → 22:21); nu primär + reserv, och reserven hoppar över sig
själv om en riktig körning redan gått. gh ersätts av ett stub-skript.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "schedule_gate.sh")

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="bash")


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(tmp_path, runs=None, event="schedule", schedule="11 6,10,16 * * 1-5", mode="seasonal",
         gh_fails=False):
    """Kör grinden med en låtsad gh (svarar med runs) och returnerar (run, logg)."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    payload = tmp_path / "runs.json"
    payload.write_text(json.dumps({"workflow_runs": runs or []}))
    gh = bindir / "gh"
    if gh_fails:
        gh.write_text("#!/usr/bin/env bash\nexit 1\n")
    else:
        # gh api <url> --jq <filter> → jq på den låtsade svarskroppen
        gh.write_text(f'#!/usr/bin/env bash\nshift 2\n[ "$1" = "--jq" ] && jq -r "$2" "{payload}"\n')
    gh.chmod(0o755)
    out = tmp_path / "out.txt"
    out.write_text("")
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", GITHUB_EVENT_NAME=event, SCHEDULE=schedule,
               GITHUB_REPOSITORY="x/y", GITHUB_RUN_ID="999", GITHUB_OUTPUT=str(out), GH_TOKEN="t")
    args = ["bash", SCRIPT, "scheduled-scan.yml"] + ([mode] if mode else [])
    p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    return out.read_text().strip(), p.stdout.strip()


def _season_cron():
    """Cron för den säsong Stockholm har just nu (testet ska gå året runt)."""
    off = subprocess.run(["bash", "-c", "TZ=Europe/Stockholm date +%z"], capture_output=True, text=True).stdout.strip()
    return ("11 6,10,16 * * 1-5", "11 7,11,17 * * 1-5") if off == "+0200" else ("11 7,11,17 * * 1-5",
                                                                                 "11 6,10,16 * * 1-5")


def _has_jq():
    return subprocess.run(["bash", "-c", "command -v jq"], capture_output=True).returncode == 0


def test_manual_run_always_runs(tmp_path):
    assert _run(tmp_path, event="workflow_dispatch")[0] == "run=true"


def test_other_seasons_cron_is_skipped(tmp_path):
    right, wrong = _season_cron()
    run, log = _run(tmp_path, schedule=wrong)
    assert run == "run=false" and "andra säsongens cron" in log


@pytest.mark.skipif(not _has_jq(), reason="jq saknas")
def test_runs_when_nothing_ran_in_the_window(tmp_path):
    right, _ = _season_cron()
    old = [{"id": 1, "status": "completed", "conclusion": "success", "run_started_at": _iso(400),
            "updated_at": _iso(383)}]
    run, log = _run(tmp_path, runs=old, schedule=right)
    assert run == "run=true" and "kör." in log


@pytest.mark.skipif(not _has_jq(), reason="jq saknas")
def test_backup_skips_when_a_real_run_already_happened(tmp_path):
    right, _ = _season_cron()
    full = [{"id": 1, "status": "completed", "conclusion": "success", "run_started_at": _iso(40),
             "updated_at": _iso(23)}]                                  # 17 min lång
    run, log = _run(tmp_path, runs=full, schedule=right.replace("11 ", "41 ", 1))
    assert run == "run=false" and "reserv" in log
    busy = [{"id": 2, "status": "in_progress", "conclusion": None, "run_started_at": _iso(5), "updated_at": _iso(1)}]
    assert _run(tmp_path, runs=busy, schedule=right)[0] == "run=false"


@pytest.mark.skipif(not _has_jq(), reason="jq saknas")
def test_a_quick_skipped_run_does_not_count(tmp_path):
    right, _ = _season_cron()
    skipped = [{"id": 1, "status": "completed", "conclusion": "success", "run_started_at": _iso(30),
                "updated_at": _iso(29.9)},                              # 6 s = hoppad
               {"id": 999, "status": "in_progress", "conclusion": None, "run_started_at": _iso(0),
                "updated_at": _iso(0)}]                                # körningen själv
    assert _run(tmp_path, runs=skipped, schedule=right)[0] == "run=true"


def test_api_failure_runs_rather_than_drops(tmp_path):
    right, _ = _season_cron()
    run, _ = _run(tmp_path, schedule=right, gh_fails=True)
    assert run == "run=true"


@pytest.mark.skipif(not _has_jq(), reason="jq saknas")
def test_non_seasonal_mode_only_dedupes(tmp_path):
    full = [{"id": 1, "status": "completed", "conclusion": "success", "run_started_at": _iso(30),
             "updated_at": _iso(20)}]
    assert _run(tmp_path, runs=full, schedule="47 4 * * 1-5", mode="")[0] == "run=false"
    assert _run(tmp_path, runs=[], schedule="17 4 * * 1-5", mode="")[0] == "run=true"
