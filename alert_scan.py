#!/usr/bin/env python3
"""
alert_scan.py — schemalagda larm för Swing och Blindspot.

Körs av GitHub Actions efter datajobben (wolf-data 06:00, scheduled-scan
08/12/18 vardagar). Streamlit-appen kör bara när någon tittar på den — det
här skriptet är det som vakar när panelen är stängd.

Flödet:
  1. Läs inställningarna (data/alerts.json på panel-data-grenen; saknas de
     gäller påslaget med Discord).
  2. Läs signalkällorna: wolf_regime/wolf_screener ur Gisten, positionerna
     ur data/swing.json, Blindspots temakarta (räknas här — tio års kurser
     per tema, det är därför jobbet ligger i Actions och inte i en rerun).
  3. Jämför mot förra körningens tillstånd (alert_state.json i Gisten) —
     BARA övergångar larmar. Första körningen lägger baslinjen tyst.
  4. Skicka via alerts.engine (Discord/e-post/webhook) och spara tillståndet.

Tillståndet sparas ENDAST om utskicket inte totalhavererade — annars skulle
ett nätverksfel äta larmet: övergången vore "sedd" men aldrig levererad.

Env:
  GITHUB_TOKEN      gist-scopad PAT (samma som datajobben) — Gist-läs/skriv
  ALERT_REPO_TOKEN  PAT med Contents: Read på repot — för inställningar och
                    positioner på panel-data (gist-tokenen räcker inte om
                    repot är privat)
  ALERT_REPO        ägare/repo (default lidbeck70/wolf-shadow-dashboard)
  ALERT_BRANCH   gren för data/-filerna (default panel-data)
  DISCORD_WEBHOOK_URL, SMTP_*, EMAIL_TO   kanalerna (se alerts/channels/)

Flaggor:
  --dry-run       räkna och skriv ut, skicka inget, spara inget tillstånd
  --no-blindspot  hoppa över temakartan (snabbkörning)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("alert_scan")

STATE_FILE = "alert_state.json"
SETTINGS_PATH = "data/alerts.json"
SWING_PATH = "data/swing.json"


def _repo_file(path: str, default):
    """En data-fil från panel-data-grenen via Contents API.

    404 → default (filen är inte skapad än — inte ett fel). Alla andra fel
    loggas och ger default, så en trasig läsning inte fäller hela skanningen —
    men den syns i loggen i stället för att försvinna.
    """
    import base64
    import requests

    repo = os.environ.get("ALERT_REPO", "lidbeck70/wolf-shadow-dashboard")
    branch = os.environ.get("ALERT_BRANCH", "panel-data")
    # Gist-tokenen (GITHUB_TOKEN) har bara gist-scope och kan inte läsa ett
    # privat repos innehåll — repo-läsningen behöver ALERT_REPO_TOKEN (samma
    # fine-grained PAT som panelens [github]-token, Contents: Read räcker).
    token = (os.environ.get("ALERT_REPO_TOKEN", "").strip()
             or os.environ.get("GITHUB_TOKEN", "").strip())
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        r = requests.get(url, headers=headers, params={"ref": branch},
                         timeout=20)
        if r.status_code == 404:
            log.info("%s finns inte på %s — använder default", path, branch)
            return default
        r.raise_for_status()
        return json.loads(base64.b64decode(r.json()["content"]))
    except Exception as exc:
        log.error("kunde inte läsa %s: %s — använder default", path, exc)
        return default


def _themes(skip: bool) -> list:
    """Temakartan som rena dicts. Tom lista när den hoppas över eller felar —
    blindspot_alerts larmar då ingenting och rör inte sin baslinje."""
    if skip:
        return []
    try:
        from blindspot.theme_board import build_theme_board
        # ThemeResult har label/key — inget name-fält (testet låser detta).
        return [{"name": t.label or t.key, "cykel_label": t.cykel_label,
                 "blindspot_score": t.blindspot_score,
                 "hat_score": t.hat_score}
                for t in build_theme_board()]
    except Exception:
        import traceback
        log.error("temakartan kunde inte byggas:\n%s", traceback.format_exc())
        return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-blindspot", action="store_true")
    args = parser.parse_args()

    import alert_rules
    from alerts.engine import send_alert
    from gist_storage import load_blob, load_wolf_json, save_blob

    settings = _repo_file(SETTINGS_PATH, {})
    regime_data = load_wolf_json("wolf_regime.json") or {}
    screener_data = load_wolf_json("wolf_screener.json") or {}
    swing_data = _repo_file(SWING_PATH, {"positions": [], "market": {}})
    themes = _themes(args.no_blindspot)
    prev_state = load_blob(STATE_FILE, None)

    # En hoppad temakarta får inte radera Blindspot-baslinjen: behåll den
    # gamla, annars larmar nästa fullkörning om övergångar som aldrig skett.
    alerts, new_state = alert_rules.evaluate(
        regime_data, screener_data, swing_data, themes, prev_state, settings)
    if not themes and isinstance(prev_state, dict):
        new_state["blindspot"] = prev_state.get("blindspot",
                                                new_state["blindspot"])

    if prev_state is None:
        log.info("Första körningen — baslinje läggs, inga larm skickas.")
        for name, t in (new_state.get("blindspot", {}).get("themes") or {}).items():
            if t.get("label") == alert_rules.EARLY:
                log.info("OBS: %s ligger REDAN i TIDIG vid baslinjen — larmar "
                         "först vid nästa inträde.", name)

    log.info("%d larm att skicka.", len(alerts))
    delivered_all = True
    for a in alerts:
        text = alert_rules.format_alert(a)
        if args.dry_run:
            log.info("[DRY-RUN] %s -> %s\n%s", a["kind"], a["channels"], text)
            continue
        results = send_alert(text, a["channels"],
                             metadata={"subject": a["title"]})
        log.info("%s -> %s", a["kind"], results)
        if results and not any(results.values()):
            delivered_all = False

    if args.dry_run:
        log.info("[DRY-RUN] tillståndet sparas inte.")
        return 0

    if alerts and not delivered_all:
        # Ingen kanal tog emot något av larmen — spara INTE tillståndet, så
        # övergången larmas om vid nästa körning i stället för att ätas upp.
        log.error("Inget larm nådde någon kanal — tillståndet sparas inte, "
                  "övergångarna prövas igen nästa körning.")
        return 0

    if not save_blob(STATE_FILE, new_state):
        log.error("Tillståndet kunde inte sparas till Gisten — nästa körning "
                  "kan skicka samma larm igen. Kontrollera GITHUB_TOKEN "
                  "(gist-scope).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
