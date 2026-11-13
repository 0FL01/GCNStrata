#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
test -x build-906/strata || { echo "Strata binary is missing; see logs/build.log" >&2; exit 1; }
test -f runtime/mtp/.ready || { echo "MTP transfer validation is pending" >&2; exit 1; }
test -f runtime/mtp/dense.txt || { echo "MTP runtime is not ready" >&2; exit 1; }
test -f runtime/shefowl-hybrid/native_experts.txt || { echo "Model pack is not ready" >&2; exit 1; }
if docker container inspect llama-server >/dev/null 2>&1; then
  docker update --restart=no llama-server >/dev/null
  docker stop -t 120 llama-server >/dev/null
fi
docker compose --env-file /home/radneon/llama/.env -f compose.yaml up -d
printf '\nStrata API: 0.0.0.0:8080; startup at system boot enabled via Docker restart policy\nLogs: %s/logs.sh --follow\n' "$PWD"
