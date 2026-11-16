#!/bin/sh
# Parent invokes only in its guarded, exclusive idle window after verifying all shard hashes.
set -eu
if [ "${Q4_1_GPU_IDLE_APPROVED:-}" != 1 ]; then
    echo 'Refusing GPU qualification without Q4_1_GPU_IDLE_APPROVED=1' >&2
    exit 2
fi
lab=/home/radneon/strata-rd/q4-1-20261116
weights=/home/radneon/models/bartowski-qwen38-q4_1
image=sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646
shard=/models/Qwen3.8-Flash-Next-Q4_1-00001-of-00003.gguf
run() {
    executable=$1
    shift
    echo "== $executable $* =="
    docker run --rm --label task=q4-1-20261116 --network none --cpus=4 --memory=8g --device=/dev/kfd --device=/dev/dri \
        -v "$lab/candidate-build:/work/build:ro" -v "$lab/fixtures:/work/fixtures:ro" \
        -v "$weights:/models:ro" --entrypoint "/work/build/$executable" "$image" "$@"
}
run iq_parity /work/fixtures
run native_expert_parity --tokens 3 --synthetic q4_1/q4_1
run native_expert_parity --tokens 3 "$shard" 0 1 2 3 20 47
run q5_projection_parity --q4-1 "$shard" blk.0.attn_qkv.weight
run q5_projection_parity --q4-1 "$shard" blk.0.attn_gate.weight
run ple_q5_parity "$shard"
