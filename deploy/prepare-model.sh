#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
docker run --rm --name strata-pack --user "$(id -u):$(id -g)" --entrypoint python3 -v "$PWD:/src" -v /home/radneon/models-nvme:/models:ro -w /src sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646 tools/iq_pack.py --gguf /models/qwen38-flash-next-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --out /src/runtime/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M --experts-bin
