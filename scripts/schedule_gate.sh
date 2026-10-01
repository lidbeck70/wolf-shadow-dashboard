#!/usr/bin/env bash
# scripts/schedule_gate.sh — avgör om en schemalagd körning ska köras.
#
# GitHub försenar (timmar) eller hoppar helt över schemalagda jobb, värst på
# jämna klockslag. Varje tidpunkt har därför en primär och en reserv-cron
# (30 min senare, båda utanför :00). Grinden:
#   1. Manuell körning (workflow_dispatch) körs alltid.
#   2. "seasonal": sommarcronen (timmarna 6,10,16 UTC) kör bara när Stockholm
#      är UTC+2, vintercronen (7,11,17 UTC) bara vid UTC+1.
#   3. Dubblettspärr: har en riktig körning av samma arbetsflöde redan startat
#      inom WINDOW_MIN minuter (pågår, eller klar och längre än 3 min — inte en
#      hoppad på sekunder) hoppar den här över sig själv.
# Felar API-uppslaget körs jobbet hellre än att tappas — larmen är
# övergångar, så en dubbelkörning skickar inget två gånger.
#
# Användning: schedule_gate.sh <workflow-fil> [seasonal]
# Miljö: GITHUB_EVENT_NAME, SCHEDULE, GITHUB_REPOSITORY, GITHUB_RUN_ID,
#        GITHUB_OUTPUT, GH_TOKEN, valfritt WINDOW_MIN (förval 150).
set -u
WF="${1:?arbetsflödesfil krävs}"
MODE="${2:-}"
SCHED="${SCHEDULE:-}"
WINDOW="${WINDOW_MIN:-150}"

out() { echo "run=$1" >> "$GITHUB_OUTPUT"; echo "$2"; }

if [ "${GITHUB_EVENT_NAME:-}" = "workflow_dispatch" ]; then
  out true "Manuell körning — kör."
  exit 0
fi

if [ "$MODE" = "seasonal" ]; then
  OFF=$(TZ=Europe/Stockholm date +%z)
  case " $SCHED " in
    *" 6,10,16 "*) WANT="+0200" ;;
    *" 7,11,17 "*) WANT="+0100" ;;
    *) WANT="" ;;
  esac
  if [ "$OFF" != "$WANT" ]; then
    out false "Cron '$SCHED', Stockholm $OFF — hoppar (andra säsongens cron)."
    exit 0
  fi
fi

SINCE=$(date -u -d "-${WINDOW} min" +%Y-%m-%dT%H:%M:%SZ)
N=$(gh api "repos/${GITHUB_REPOSITORY}/actions/workflows/${WF}/runs?per_page=20" --jq "
  [.workflow_runs[]
   | select(.id != ${GITHUB_RUN_ID})
   | select(.run_started_at >= \"${SINCE}\")
   | select(.status == \"in_progress\"
            or (.conclusion == \"success\"
                and ((.updated_at | fromdateiso8601) - (.run_started_at | fromdateiso8601)) > 180))
  ] | length" 2>/dev/null) || N=0
case "$N" in ''|*[!0-9]*) N=0 ;; esac

if [ "$N" -gt 0 ]; then
  out false "Cron '$SCHED' — en körning har redan gått de senaste ${WINDOW} min ($N st), hoppar (reserv)."
  exit 0
fi
out true "Cron '$SCHED' — kör."
