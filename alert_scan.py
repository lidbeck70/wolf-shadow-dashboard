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
     per tema, det är därför jobbet ligger i Actions och inte i en rerun),
     Övervakningens lista (data/watchlist.json) mot Thors rader.
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
  PANEL_DATA_KEY    läser det krypterade registret; tillståndet och larmloggen
                    sparas krypterade i Gisten när den finns (private_data)
  EMBER_GIST_ID     EMBER-resultatens gist (för ember-benet)
  DISCORD_WEBHOOK_URL, SMTP_*, EMAIL_TO   kanalerna (se alerts/channels/)
  PANEL_URL         panelens adress (valfri) — larmen får en länk till rätt vy

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
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("alert_scan")

STATE_FILE = "alert_state.json"
SEND_PAUSE_S = 0.5          # paus mellan larmen (Discord: ~5 per 2 s och webhook)
SETTINGS_PATH = "data/alerts.json"
SWING_PATH = "data/swing.json"
HOLDINGS_PATH = "data/holdings.json"    # registret — positionerna bor här sedan PR 10
WATCH_PATH = "data/watchlist.json"      # 👀 Övervakningen — bevakade par (aktie, strategi)
UNREADABLE_PATHS: set = set()           # filer vars senaste läsning fallerade (nät, fel nyckel) — inte 404


def _repo_file(path: str, default):
    """En data-fil från panel-data-grenen via Contents API.

    404 → default (filen är inte skapad än — inte ett fel). Alla andra fel
    loggas och ger default, så en trasig läsning inte fäller hela skanningen —
    men den syns i loggen i stället för att försvinna, och sökvägen hamnar i
    UNREADABLE_PATHS (ett oläsbart register är inte "inga innehav").
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
            UNREADABLE_PATHS.discard(path)
            return default
        r.raise_for_status()
        import private_data                # privata ark (positionerna) ligger krypterade
        data = private_data.unseal(json.loads(base64.b64decode(r.json()["content"])))
        UNREADABLE_PATHS.discard(path)
        return data
    except Exception as exc:
        log.error("kunde inte läsa %s: %s — använder default", path, exc)
        UNREADABLE_PATHS.add(path)
        return default


def deliver(alerts: list, send, fmt, dry_run: bool = False, pause=None) -> tuple:
    """Skicka larmen ett i taget med paus emellan. (antal levererade, missade titlar).
    Ett larm räknas som levererat när minst en kanal tog emot det."""
    pause = time.sleep if pause is None else pause
    delivered, failed = 0, []
    for i, a in enumerate(alerts):
        text = fmt(a)
        if dry_run:
            log.info("[DRY-RUN] %s -> %s\n%s", a["kind"], a["channels"], text)
            continue
        if i:
            pause(SEND_PAUSE_S)            # Discord-webhooken tål ~5 meddelanden / 2 s
        results = send(text, a["channels"], metadata={"subject": a["title"]})
        log.info("%s -> %s", a["kind"], results)
        if results and not any(results.values()):
            failed.append(a["title"])
        else:
            delivered += 1
    return delivered, failed


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


def _sheet_map(loader):
    """Registrets karta för arkens bara nordiska former ('BOL' → 'BOL.ST') — den Thor bygger raderna med
    (thor.ui.load: ur.load(...).sheet_map()). None när registret inte gick att läsa: då står Rick Rule, Durrett
    och Tiggre still (alert_rules.WATCH_SOURCES) i stället för att jämföra en annan tickerform än Thors."""
    try:
        from universe import registry as ur
        reg_u = ur.load(None, loader)
        return (reg_u.sheet_map() if reg_u is not None else None) or None
    except Exception as exc:
        log.warning("Registret gick inte att läsa (%s).", type(exc).__name__)
        return None


def _watch_rows(watch_data, loader, regime_data, ca_loader=None, ember_loader=None, today=None) -> tuple:
    """(Thors rader, frysta strategier/par, färska listor) för Övervakningen — ur samma blobbar och samma
    tickerkartor som Thor (thor.ui.load/rebuild), men bara när något bevakas. Rader None = körningarna gick inte att
    läsa (eller bygget föll): benet står still, inget larmar. ca_loader/ember_loader = de redan lästa CA- och
    EMBER-listorna, så att de inte hämtas en gång till. today (None = i dag) gäller både radernas GAMMAL och
    listornas egna stämplar (alert_rules.watch_fresh)."""
    if not isinstance(watch_data, dict):
        return None, set(), {}
    if not watch_data.get("items"):
        return [], set(), {}                              # inget bevakas: inget att läsa, tillståndet töms
    try:
        import alert_rules
        from thor import candidates as tc
        from universe import registry as ur
        blobs = tc.load_blobs(loader, ca_loader, ember_loader)
        if blobs.get(tc.UNREADABLE):
            log.warning("Körningarna gick inte att läsa — Övervakningen står stilla.")
            return None, set(), {}
        blobs["regime"] = regime_data
        blobs["sheet_map"] = _sheet_map(loader)
        if blobs["sheet_map"] is None:
            log.warning("Registrets tickerkarta saknas — Rick Rule, Durrett och Tiggre står stilla i Övervakningen.")
        errors: dict = {}
        rows = tc.build(blobs, today=today, errors=errors, sheet_map=blobs["sheet_map"],
                        momentum_map=(ur.load_index(loader) or {}).get("bd_map") or None)
        if errors:
            log.warning("Övervakningen: delar som föll i bygget står stilla: %s", ", ".join(sorted(errors)))
        return rows, alert_rules.watch_frozen(blobs, errors), alert_rules.watch_fresh(blobs, today)
    except Exception:
        import traceback
        log.error("Övervakningens rader kunde inte byggas — benet står stilla:\n%s", traceback.format_exc())
        return None, set(), {}


def _keep_held_baselines(new_state: dict, prev_state, sheets_data) -> None:
    """Ett register som inte gick att läsa (fel PANEL_DATA_KEY, nätfel) är inte "inga innehav": innehavens
    swing-verdikt och Tiggre-positionernas händelser står kvar i baslinjen. Annars larmade varje SÄLJ och varje
    Tiggre-händelse en gång till när registret går att läsa igen (F12). Inget larm skickas ur det gamla läget."""
    def _d(x) -> dict:
        return x if isinstance(x, dict) else {}
    prev = _d(prev_state)
    swing, sheets = new_state.get("swing"), new_state.get("sheets")
    old = _d(prev.get("swing")).get("verdicts")
    if HOLDINGS_PATH in UNREADABLE_PATHS and isinstance(old, dict) and isinstance(swing, dict):
        log.warning("Registret gick inte att läsa — innehavens swing-verdikt står kvar i baslinjen.")
        new_state["swing"] = {**swing, "verdicts": {**old, **_d(swing.get("verdicts"))}}
    old = _d(prev.get("sheets")).get("events")
    if _d(sheets_data).get("holdings_unreadable") and isinstance(old, dict) and isinstance(sheets, dict):
        # Tiggre-händelserna räknas bara på positionerna (sheets_refresh.build_events) — de bär innehaven.
        held = {k: v for k, v in old.items() if str(_d(v).get("kind") or "").startswith("tiggre_")}
        log.warning("Sifferuppdateringen kunde inte läsa registret — Tiggre-positionernas händelser står kvar.")
        new_state["sheets"] = {**sheets, "events": {**held, **_d(sheets.get("events"))}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-blindspot", action="store_true")
    args = parser.parse_args()

    import alert_rules
    from alerts.engine import send_alert
    from gist_storage import load_blob, load_wolf_json, read_blob, save_blob

    # Tillståndet först: en läsning som FALLERAR är inte en första körning. Förut blev ett nätverksfel en ny
    # baslinje — inga larm skickades och nästa körning såg dem som redan sedda (granskning 2, F2).
    state_status, prev_state = read_blob(STATE_FILE)
    if state_status == "unreadable":
        log.error("%s gick inte att läsa (%s) — inga larm skickas och inget sparas; nästa körning försöker igen.",
                  STATE_FILE, prev_state)
        return 1
    if state_status == "missing":
        prev_state = None

    UNREADABLE_PATHS.clear()                       # gäller den här körningens läsningar
    settings = _repo_file(SETTINGS_PATH, {})
    regime_data = load_wolf_json("wolf_regime.json") or {}
    screener_data = load_wolf_json("wolf_screener.json") or {}
    swing_data = _repo_file(SWING_PATH, {"positions": [], "market": {}})
    # Positionerna ur registret (Holdings); en gammal lista i swing.json
    # räknas med tills panelen flyttat den (idempotent på id).
    import positions as _positions
    held = _positions.view_rows_from(_repo_file(HOLDINGS_PATH, {}), "Momentum")
    have = {p.get("id") for p in held}
    swing_data["positions"] = held + [p for p in swing_data.get("positions") or []
                                      if isinstance(p, dict) and p.get("id") not in have]
    themes = _themes(args.no_blindspot)

    # EMBER: gist-läsningen är oautentiserad och billig. timestamp None
    # betyder oläsbar/aldrig sparad — ember_alerts behåller då baslinjen.
    ember_data = None
    try:
        from ember.cache import load_ember_results
        ember_data = load_ember_results()
    except Exception:
        log.warning("EMBER-resultaten kunde inte läsas — benet står stilla.")

    # Wolf/Viking: arc_scan.py:s sparade blob. None → benen behåller sina
    # baslinjer. En screener som felade har error satt — även den ger None,
    # så en kraschad scanning aldrig ser ut som "alla åkte ur listan".
    arc = load_blob("arc_screeners.json", None)
    wolf_data = viking_data = None
    if isinstance(arc, dict):
        w = arc.get("wolf")
        if isinstance(w, dict) and not w.get("error"):
            wolf_data = w
        v = arc.get("viking")
        if isinstance(v, dict) and not v.get("error"):
            viking_data = v

    # Deep Contrarian: scheduled_scan sparar listan till CA-gisten. Tom/
    # oläsbar (ingen timestamp) → benet fryser sin baslinje.
    contrarian_data = None
    try:
        from contrarian_alpha.cache import load_screener_results
        _ca = load_screener_results(mode="deep_contrarian")
        if isinstance(_ca, dict) and _ca.get("timestamp"):
            contrarian_data = _ca
    except Exception:
        log.warning("Deep Contrarian-listan kunde inte läsas — benet står stilla.")

    # Quality-listan med köpsignalerna (scheduled_scan → quality_scan). Saknas
    # signalerna fryser benet sin baslinje.
    quality_data = None
    try:
        from contrarian_alpha.cache import load_screener_results
        _q = load_screener_results(mode="quality")
        if isinstance(_q, dict) and _q.get("timestamp"):
            quality_data = _q
    except Exception:
        log.warning("Quality-listan kunde inte läsas — benet står stilla.")

    # Insiderbevakaren: insider_scan.py:s blob. error → benet fryser.
    insider_data = None
    _ins = load_blob("insider_scan.json", None)
    if isinstance(_ins, dict) and not _ins.get("error"):
        insider_data = _ins

    # Håvarna: screens_scan.py:s blob. Saknas → benet fryser; en håv med
    # error fryser bara sin egen baslinje.
    screens_data = load_blob("screens.json", None)
    if not isinstance(screens_data, dict):
        screens_data = None

    # Arkens övergångar: sheets_refresh.py:s blob. error → benet fryser.
    sheets_data = load_blob("sheets_refresh.json", None)
    if not isinstance(sheets_data, dict):
        sheets_data = None

    # 🌩️ Marknadsrisk: dagens nivå för SPY och OMXS30 (lätt läge). Ingen nivå
    # alls → None, så benet fryser sin baslinje.
    market_risk_data = None
    try:
        import market_risk as _mr
        import market_risk_gate as _mg
        _levels = {m: _mg.current(m) for m in _mr.MARKETS}
        if any(v for v in _levels.values()):
            market_risk_data = _levels
            if not args.dry_run:                        # ODIN → Läget och startsidan läser nivån utan att räkna live
                save_blob("market_risk.json", {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                                               "levels": _levels})
    except Exception:
        log.warning("Marknadsrisken kunde inte räknas — benet står stilla.")

    # 🐺 Fiat Debasement: dagens nyckeltal per valuta och guld/silver-kvoten.
    # Ingenting alls → None, så benet fryser sin baslinje.
    fiat_data = None
    try:
        from fiat_debasement.alerts import collect as _fiat_collect
        fiat_data = _fiat_collect()
    except Exception:
        log.warning("Fiat Debasement kunde inte räknas — benet står stilla.")

    # 👀 Övervakningen: listan ur panel-data (oläsbar → None, benet fryser) mot Thors rader. Bara KÖP och
    # "föll ur listan" larmar (alert_rules.watch_alerts).
    watch_data = _repo_file(WATCH_PATH, None)
    # Brädan läses en gång: händelsemotorn, Thors rader och arken den lästes ur (watch_sheets) ser samma körning.
    board_data = load_blob("valhall_board.json", None)

    def _blob(name, fallback=None):
        return board_data if name == "valhall_board.json" else load_blob(name, fallback)
    watch_rows, frozen_watch, fresh_watch = _watch_rows(
        watch_data, _blob, regime_data,
        ca_loader={"quality": quality_data, "deep_contrarian": contrarian_data}.get,   # None = oläsbar → fryser
        ember_loader=lambda: ember_data)

    # En hoppad temakarta får inte radera Blindspot-baslinjen: behåll den
    # gamla, annars larmar nästa fullkörning om övergångar som aldrig skett.
    alerts, new_state = alert_rules.evaluate(
        regime_data, screener_data, swing_data, themes, prev_state, settings,
        ember_data=ember_data, wolf_data=wolf_data, viking_data=viking_data,
        contrarian_data=contrarian_data, quality_data=quality_data, insider_data=insider_data,
        screens_data=screens_data, sheets_data=sheets_data, market_risk_data=market_risk_data,
        fiat_data=fiat_data, board_data=board_data,
        watch_data=watch_data, watch_rows=watch_rows, frozen_watch=frozen_watch, fresh_watch=fresh_watch,
        sheets_watch=alert_rules.watch_sheets({"board": board_data}))
    if not themes and isinstance(prev_state, dict):
        new_state["blindspot"] = prev_state.get("blindspot",
                                                new_state["blindspot"])
    _keep_held_baselines(new_state, prev_state, sheets_data)

    if prev_state is None:
        log.info("Första körningen — baslinje läggs, inga larm skickas.")
        for name, t in (new_state.get("blindspot", {}).get("themes") or {}).items():
            if t.get("label") == alert_rules.EARLY:
                log.info("OBS: %s ligger REDAN i TIDIG vid baslinjen — larmar "
                         "först vid nästa inträde.", name)

    log.info("%d larm att skicka.", len(alerts))
    # PANEL_URL (hemlighet i GitHub) = panelens adress: varje larm får en länk till rätt vy. Utan den samma text som förut.
    base = os.environ.get("PANEL_URL", "").strip()
    delivered, failed = deliver(alerts, send_alert, lambda a: alert_rules.format_alert(a, base),
                                dry_run=args.dry_run)

    if args.dry_run:
        log.info("[DRY-RUN] tillståndet sparas inte.")
        return 0

    if alerts:
        # Larmloggen i Gisten (ALERTS och startsidan läser den) — även larm som ingen kanal tog emot, märkta.
        from alerts import history
        stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        log_status, old_log = read_blob(history.BLOB)
        if log_status == "unreadable":            # skriv inte över de 50 gamla raderna med bara de nya (F12)
            log.warning("Larmloggen gick inte att läsa (%s) — den uppdateras nästa körning.", old_log)
        else:
            rows = history.append(old_log if log_status == "ok" else None, alerts, stamp,
                                  {a["title"]: a["title"] not in failed for a in alerts})
            if not save_blob(history.BLOB, rows):
                log.warning("Larmloggen kunde inte sparas till Gisten.")

    if alerts and not delivered:
        # Ingen kanal tog emot något av larmen — spara INTE tillståndet, så
        # övergången larmas om vid nästa körning i stället för att ätas upp.
        log.error("Inget larm nådde någon kanal — tillståndet sparas inte, "
                  "övergångarna prövas igen nästa körning.")
        return 0
    if failed:
        # Förut räckte ETT misslyckat larm (Discord 429) för att inget sparades
        # — då skickades hela skörden igen varje körning och rate-limiten slog
        # till ännu hårdare. Nu sparas tillståndet och de missade loggas.
        import private_data                 # Actions-loggen är publik — rubrikerna kan bära innehavens tickers
        log.error("%d av %d larm nådde ingen kanal (sparas ändå): %s",
                  len(failed), len(alerts), private_data.public_log(" | ".join(failed)))

    if not save_blob(STATE_FILE, new_state):
        log.error("Tillståndet kunde inte sparas till Gisten — nästa körning "
                  "kan skicka samma larm igen. Kontrollera GITHUB_TOKEN "
                  "(gist-scope).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
