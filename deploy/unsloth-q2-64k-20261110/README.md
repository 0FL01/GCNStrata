# Prepared benchmark — NOT EXECUTED

controller.py is standalone standard-library Python for amude. Preparation only
created this experiment directory and compiled the script; no engine/GPU workload
or production stop/start was executed.

## Before future execution

Obtain explicit downtime approval AND externally quiesce production ingress.
Status/GPU checks are not an atomic admission barrier against new client requests.
The parent must finish/validate all 3 candidate shards, pack, tokenizer, PLE and
MTP compatibility. Existence checks are not semantic compatibility validation.
Production config, binary, protocol source, container identity, mounts and device
runtime fingerprints must still match setup.json, otherwise execution blocks.

Candidate defaults:
- Native: /sata/qwen38-flash-next-ud-q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf
- Pack: /sata/qwen38-flash-next-ud-q2_0/pack
- PLE: existing shefowl second GGUF shard (Q8_0).

Override candidate paths with --unsloth-native, --unsloth-pack, --unsloth-ple.

Future command, NOT executed during preparation:

    python3 /home/radneon/strata/deploy/unsloth-q2-64k-20261110/controller.py \
      --execute --idle-window-approved --compatibility-approved

Optional --vision-smoke /home/radneon/strata/deploy/unsloth-q2-vision-20261110/request.json
adds one fresh candidate engine before text arms. Parent-prepared JSON has keys:
ids (real image-prompt integer IDs), embeddings (container-visible filename under
/src/deploy/unsloth-q2-vision-20261110/), max_new (normally 64).
The parent prepares/validates CPU encoder output independently; this controller
never runs an encoder. GENI uses identical sampling, --vision is enabled only for
the smoke, and requested nonempty output count, finite positive timings, full
prompt processing, no reuse and no ERR/nonfinite finish are required. Raw streams,
DONE, output IDs and input/embedding hashes are preserved separately.

## Text methodology

- ABBA order: shefowl-1, unsloth-1, unsloth-2, shefowl-2. A fresh isolated engine
  for every case, exactly one 65536-ID prompt and 512 generated outputs per case.
- Same pinned image, qualified binary bind mount, mounts, effective production
  child environment including config overrides, devices, IPC, ulimits, group,
  security option, cwd and other native flags. API credentials are omitted from
  native engines because they need no API authentication. No env file is written.
- Both text arms disable vision, prompt cache/checkpointing and conversation
  parking. EOS sentinel 2147483647; temperature 1, top_p .95, top_k 20, seed 12345.
- Keep expert-cache 9900, prefill 4096, KV int8 resident 32768, layer-split 27,
  max-context 262144, spec 6 and all remaining production flags unchanged except
  candidate native/pack/PLE paths. The production config itself is never edited.
- MTP remains IDENTICAL /src/runtime/mtp: existing fixed Q2_0 experts with Q8_0
  projections. Requested shared-Q8 MTP cannot load directly; this is a limitation,
  NOT a benchmarked shared-Q8 setup.
- Require DONE generated=actual outputs=512, prompt_tokens=prompt_read=65536,
  reused=0, finish=length, valid output IDs and finite positive native timings.
  PP=65536*1000/prompt_ms; TG=512*1000/decode_ms. Native accounting runs the last
  prompt token in the first decode window. Raw DONE is retained.
- Startup/model-load-to-READY and request wall time are reported separately.
  Fresh engines do not mean cold OS page cache. No equal-output or quality claim.
  Optional image smoke is not included in the text averages.

Three stable busy=false/queued=0/in_flight=0 status samples plus GPU busy<=3%
are required before stopping, then a final status/GPU check. Missing GPU counters
fail closed. Each run is bounded (default 5400s including startup; bounded extra
QUIT/cleanup time). A cancel file in this directory stops at the next poll.

SIGINT/SIGTERM and ordinary failures trigger cleanup and finally restart the exact
original production container, wait up to 900s for loaded text+vision readiness,
and verify the production config hash. SIGKILL, host failure or Docker failure
require operator recovery; Python finally cannot handle those. Production is never
removed/recreated. Results go in fresh run-* directories: manifests, stdout/stderr,
DONE/metrics/output IDs, partial results, restoration evidence and per-arm averages.
Secrets and raw production status are never persisted. Fixture provenance, count,
range and hash plus qualified binary/protocol/config hashes are in setup.json.
