#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p logs
if docker container inspect strata-api >/dev/null 2>&1; then
  docker stop -t 120 strata-api >/dev/null
  docker logs --timestamps strata-api > logs/service-last.log 2>&1
fi
# Removing only this compose service also removes its Docker autostart policy.
docker compose --env-file /home/radneon/llama/.env -f compose.yaml down --timeout 120
printf 'Strata stopped; its startup at system boot is disabled\n'
