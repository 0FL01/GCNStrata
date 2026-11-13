#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
case "${1:-}" in
 --follow|-f) exec docker logs --timestamps --tail 200 --follow strata-api ;;
 --build) file=logs/build-retry2.log ;;
 --pack) file=logs/pack.log ;;
 --mtp) file=logs/mtp-prepare.log ;;
 "") file= ;;
 *) echo "Usage: $0 [--follow|--build|--pack|--mtp]" >&2; exit 2 ;;
esac
if [ -z "$file" ] && ! docker container inspect strata-api >/dev/null 2>&1; then file=logs/service-last.log; fi
if [ -n "$file" ]; then
 if command -v less >/dev/null 2>&1 && [ -t 1 ]; then exec less +G "$file"; fi
 exec cat "$file"
fi
if command -v less >/dev/null 2>&1 && [ -t 1 ]; then
 docker logs --timestamps strata-api 2>&1 | less +G
else
 docker logs --timestamps strata-api 2>&1
fi
