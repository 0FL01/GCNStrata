#!/usr/bin/env bash
# Root runs this on the build host; no API operations and no GPU runtime.
set -euo pipefail
INPUT=$(realpath "${1:?usage: build-host.sh INPUT NEW_WORK}")
WORK=${2:?NEW_WORK required}
IMAGE=sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646
(cd "$INPUT"; sha256sum -c INPUTS.sha256)
[[ "$WORK" == /home/radneon/strata/logs/* ]] || { echo 'work must be a fresh logs directory' >&2; exit 2; }
exec 9>/tmp/strata-direct-ff34-ce.lock
flock -n 9 || { echo 'shared engineering lock busy' >&2; exit 2; }
mkdir "$WORK"
WORK=$(realpath "$WORK")
printf '%s\n' "$IMAGE" > "$WORK/container-image.txt"
cp "$INPUT/INPUTS.sha256" "$INPUT/public-provenance.json" "$WORK/"
set +e
docker run --rm --network none -v "$INPUT:/input:ro" -v "$WORK:/work" -w /work --entrypoint /bin/bash "$IMAGE" /input/build-inner.sh > "$WORK/build.stdout" 2> "$WORK/build.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$WORK/build.exit_code"
exit "$rc"
