#!/usr/bin/env bash
set -euo pipefail
original=/home/radneon/strata/rnd-next-20261017/dflash2
retired=/home/radneon/strata/retired/dflash2-20261020T1013Z
[[ -d "$retired" && ! -e "$original" && ! -L "$original" ]] || { echo 'Restore aborted: retired directory missing or original path occupied.' >&2; exit 1; }
[[ "$(stat -c %d "$retired")" == "$(stat -c %d "$(dirname "$original")")" ]] || { echo 'Restore aborted: different filesystems.' >&2; exit 1; }
mv -T -- "$retired" "$original"
echo 'Restored DFlash2 files only. No process or container restarted.'
