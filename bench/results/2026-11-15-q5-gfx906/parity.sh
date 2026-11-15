#!/bin/bash
# Run inside the isolated audit container; production must already be stopped.
# Usage: parity.sh baseline|candidate
set -u
variant=$1
case "$variant" in baseline|candidate) ;; *) exit 2;; esac
b=/audit/$variant
out=/audit/artifacts/$variant-parity
mkdir -p "$out"
run() {
    name=$1; shift
    "$@" > "$out/$name.log" 2>&1
    rc=$?
    printf '%s %s\n' "$name" "$rc" | tee -a "$out/status.txt"
}
export HIP_VISIBLE_DEVICES=0 STRATA_EXP_MODE=16
q3=/sata/qwen38-flash-next-ud-q3_k_xl/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
q2=/models/qwen38-flash-next-ud-q2_k_xl/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf
for t in 1 2 3 4 6; do
    run dot-t$t "$b/native_expert_parity" --tokens "$t" --q5_0-dot
    run synthetic-t$t "$b/native_expert_parity" --tokens "$t" --synthetic iq2_s/q5_0 iq3_xxs/q5_0 iq4_xs/q5_0
    run q3-t$t "$b/native_expert_parity" --tokens "$t" "$q3" 0 4 13 16 20 47
    run q2-t$t "$b/native_expert_parity" --tokens "$t" "$q2" 4
done
for mode in 0 2 7 16; do
    export STRATA_EXP_MODE=$mode
    run grouped-mode$mode "$b/native_grouped_parity" --q5_0
done
export STRATA_EXP_MODE=16
for dev in 1 2 3; do
    export HIP_VISIBLE_DEVICES=$dev
    run dot-gpu$dev "$b/native_expert_parity" --tokens 3 --q5_0-dot
done
# Expected baseline arithmetic failures are evidence, not a runner failure.
echo PARITY_RUN_FINISHED
