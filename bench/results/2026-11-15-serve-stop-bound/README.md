# Native serve stop-bound probe

## Fix and measured boundary

The candidate bounds committed input rows before native length/EOS emission in
the pipeline and non-merged serial serve paths. It ports the meaning of upstream
`243fdb213c270ffd0be76c17c7d1cc3d4a2361dd` and
`13eedf552417a6a791a7d13c6597e24300ee3a09` without a HIP-only guard or a new runtime
switch. Unlike a literal port, it preserves the raw accepted-draft count for
policy/calibration/statistics and uses a separate `kept_rows` for state/replay.
Merged prompt rows keep their full commit. No numerical kernel changed.

On the Q5-fixed baseline `c723df72`, a 58-token synthetic prompt and two sampled
outputs `[24, 11]` left **60** consumed tokens on all four gfx906 stages instead
of **59**. The next continuation reused58/read2 rather than reused59/read1.
The candidate emitted the same IDs and left59; continuation reused59/read1.

The trace proves the affected path, not just the configuration: the second window
had T2, accepted2 input rows, remaining1, speculative B launched and `com=7`.
All three front stages had early full commits. Baseline replayed2 rows; candidate
replayed1 on stages0/1/2, matching the last stage's bounded commit.

Across the sampled budget sweep and greedy control, returned IDs and raw draft
acceptance counters matched baseline. All candidate consumed histories were
exactly `prompt_ids + output_ids[:-1]`. Valid state fingerprints on all four
stages, including GDN/PLE/KV/index state and draft KV `[0,L-1)`, matched a same-
binary reference with `theta=2` (no speculative B). Stale KV beyond L and the
indexer's spare pooled row are excluded. This is short-prefix state evidence,
not model-quality, long-context or performance qualification.

`ple_prev` is checked against the actual consumed tail on **stage0**, the owner
of layer1 in this profile. `Verifier::ple_stage()` (`verify.hpp`) restricts PLE
staging and commit advancement to that owner. `conversation_checkpoint_restore`
reconstructs `ple_prev` on every stage, including non-owners; a cold non-owner
instead retains `-1,-1`. The first EOS comparison exposed precisely that metadata
difference on stages1/2/3 while all payload fingerprints and stage0 history matched.
The checker therefore excludes only those unused non-owner token-history fields,
not PLE payload or the owner's history. This required no runtime change.

**Qualification passed:** `evidence/qualification-summary.json` covers49 native
requests across seven required runs. Both EOS positions, serial decode with the
same startup placement, and uninstrumented release smoke passed. `check.py`
requires exact request sets, direct-live and parked routes, all-stage evidence,
rollback traces and token parity before writing a passing summary. Four extra
startup-serial requests are preserved separately, not used as a matched state
reference. Raw logs remain in the isolated lab.

Recheck the published evidence without GPU access (choose a new output file):

```sh
python3 bench/results/2026-11-15-serve-stop-bound/check.py \
  bench/results/2026-11-15-serve-stop-bound/evidence --output /path/to/new-summary.json
```

## Build, review and limits

- Qualified uninstrumented binary SHA256:
  `2c3da12662846c07fb50bb7aa0f4a3f49fb7cba32917d118c92fc044fb7345bd`.
- `evidence/build-identity.json` records exact source hashes, unchanged baseline
  source verification, pinned image/llama.cpp and production gfx906 flags. Audit
  markers are absent from the release binary. Do not deploy the separately
  identified diagnostic binaries.
- Ported host helper:2736 cases plus zero remaining; actual-source consumer test
  passes, and the c723df72 source compiles but fails its commit/output invariant.
  CMake registration/build/CTest passed. Gfx906 engine build passed; a supplemental
  native-HIP compile passed before hardware scope was clarified. NVIDIA/CUDA and
  SYCL are not qualified or required by this gfx906-only fork.
- Independent runtime review found no blockers. Verification reviews corrected
  missing-route acceptance in the artifact checker and required diagnostic setup;
  PLE ownership and the matched-layout serial reference were independently reviewed.
- No numerical kernels, penalties, DMA/adaptation, model files or clocks changed.
  The existing Q5 fix and memory-lock protections remain. This is not a throughput,
  vision, full256K or broad sampled-quality audit. GPU windows were at most2 rows;
  larger widths are covered by the host policy test only. Python text stops and
  live cancellation timing are not newly qualified by this fix.

`audit.py` uses `serve.server.StrataEngine`, `child_env`, and the existing
`conversation_cache_parity.load_tokenizer`. Each invocation loads **one** native
process. It never contacts the production API. `--help` and dry runs do not import
the runtime or load a model. GPU execution belongs to the approved isolated window.

## Inputs and runs

Use the unchanged container-valid Q3 config:
`/src/configs/unsloth-q3-k-xl-4gpu-f16-iq4-20261114.json`.
Its tokenizer is `/models/qwen38-flash-next-ud-q3_k_xl/pack/tokenizer`.
The runner retains all native arguments/environment from that config; only
explicit `--pipeline-windows` / `--eos-ids` replace the corresponding arguments.
It enables `STRATA_PIPELINE_DEBUG=1` and `STRATA_PIPELINE_LOG=1`.
`--theta 2` writes a private `STRATA_PIPELINE_SWITCH` file to suppress speculative
B; ordinary runs retain the configured threshold. Release runs provide discovery
and control only; all-stage state evidence requires the diagnostic procedure below.

## Required diagnostic builds for state evidence

`instrument.py SOURCE OUTPUT` creates a new audit-only `generate.cpp`; it never
edits its input or overwrites an existing output. Apply it separately to the
`c723df72` baseline source and the fixed candidate source, then compile each in
an isolated copy of the matching gfx906 source/build tree. Keep the release
source and binary untouched. Never deploy either diagnostic binary.

For these runs, use the Docker command below with **both** changes:

- Replace the `/baseline` mount with the diagnostic baseline build directory
  (or select `/candidate/strata` backed by the diagnostic candidate build).
- Add `-e STRATA_STATE_HASH=1`. Without this, even a diagnostic binary does not
  emit `STOP_HISTORY` or per-stage `STOP_STATE`.

Before state comparison, require exactly one `STOP_HISTORY` and `STOP_STATE`
records for stages **0, 1, 2, 3** for each completed request. Require the expected
consumed IDs and logical length, then compare only valid state ranges against a
matched reference. The short-fixture diagnostic excludes the drafter's final
possibly uncomputed cell (`draft_upto=L-1`, capped by its ring size). Confirm
`STOP_BOUND` and, for the rollback case, `STOP_EARLY`/`STOP_UNDO` evidence too.
Missing records are incomplete qualification, never a state pass. The runner's
`--expect pass` alone deliberately does not certify this manual/artifact gate.

Inside a prepared container, baseline discovery is:

```sh
python3 /audit-src/bench/results/2026-11-15-serve-stop-bound/audit.py \
  --config /src/configs/unsloth-q3-k-xl-4gpu-f16-iq4-20261114.json \
  --engine /baseline/strata --output /out/baseline-sweep \
  --scenario sweep --max-new 4 --seed 424242 --expect observe --run
```

The sweep sends sampled limits **1, 2, 3, 4, 5, 7, 8**, with temperature 1,
top_p 0.95, top_k 20. At the selected limit (default 4), it adds direct-live and
park/restore continuations. One greedy request is the control; this is not a
Cartesian sweep. Every sampled/greedy request records its numeric settings.

After finding a real boundary in the trace, run the selected limit in a fresh
invocation with `--scenario case --max-new N --expect fail` against baseline and
the same inputs with `--engine /candidate/strata --expect pass`. Use distinct new
output directories. `--expect fail` requires an actual measured invariant
violation; absent tracing, missed EOS, or a missing parked restore do not satisfy it.
`--expect observe` records exploratory violations without making them an exit
failure; engine exceptions still fail. Passing these checks alone does not prove
the goal's complete GPU-state/boundary coverage.

Selected variants:

- `--sampling greedy`: greedy selected case.
- `--serial-switch`: serial callsite via per-request debug `pw=1`, retaining the
  configured pipeline2 startup allocation and expert-cache capacity. This is the
  `candidate-serial-matched` state-comparison run required by `check.py`.
- `--pipeline-windows 1` (or `--pipeline1`): changes startup allocation too. The
  initial `candidate-serial` run passed IDs/bounds/reuse, but had 23023 expert
  slots versus pipeline2's22698. Its CPU/GPU expert placement therefore differs;
  its state hashes are not a matched numerical reference. Keep it as a separate
  observation, not a passing cross-path state comparison.
- `--theta 2`: no-speculative-B control, using the existing debug switch.
- `--eos-ids ID --scenario case --max-new 8`: replace the native EOS list with an
  observed ID. Confirm no earlier occurrence and the **actual** accepted window
  position from the trace. A token's output index alone is not proof of its window
  position. Use separate invocations for the two actual T2 positions.
- `--route live` / `--route park`: isolate direct A/A+ or parked A/B/A+.
  Default `both` runs A/A+/B/A++: both continuations append **all** returned IDs
  once, with no template regeneration or duplicated pending token.
- `--replay /out/previous/NNN-name.json`: one saved request's exact prompt IDs,
  budget and sampling in a fresh native process. This supplies a cold output
  reference, not an automatic claim of matched FP computation or full-state
  equality. Reapply EOS/pipeline/theta overrides explicitly when appropriate.

All requests save IDs, DONE metadata, RESUME, log offsets and relevant raw log
lines, including optional `STOP_*` and `STATE_HASH`. Existing state lengths, when
present, must equal `len(prompt_ids) + len(returned_ids) - 1`; missing hashes are
not fabricated or treated as native failures. Continuation checks require
`RESUME == reused == len(prompt_ids)-1`, `prompt_read == 1`; park additionally
requires a real restore log. Stage hashes, accepted/kept rows, B launch, early
commit, drain/replay and EOS positions remain trace-review evidence.

Each run creates a new directory and refuses overwrite. Per-request JSON is
written even on a request exception; `engine.log` is retained intact. The manifest
records binary/config/runner hashes, argv, selected tuning settings and options.
Fixtures are synthetic; no API key or production request is used.

## Isolated Docker inputs

On `amude`, prepare existing absolute directories `$AUDIT_ROOT` (whole Strata
source tree containing this runner, `serve/` and `tools/`), `$CANDIDATE_DIR`
(containing `strata`), and `$OUTPUT_PARENT`. Launch only after the parent's GPU
preflight, with production stopped. No API env-file or published port is needed.

```sh
docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --ipc host \
  --ulimit memlock=-1:-1 -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 \
  -v "$AUDIT_ROOT:/audit-src:ro" \
  -v /home/radneon/strata:/src:ro \
  -v /home/radneon/strata/releases/gfx906-q5-fix-c723df72-20261115:/baseline:ro \
  -v "$CANDIDATE_DIR:/candidate:ro" \
  -v /home/radneon/models-nvme:/models:ro \
  -v /home/radneon/models:/sata:ro \
  -v /home/radneon/models-nvme/shared-engram:/home/radneon/models-nvme/shared-engram:ro \
  -v "$OUTPUT_PARENT:/out" \
  sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646 \
  /audit-src/bench/results/2026-11-15-serve-stop-bound/audit.py \
  --config /src/configs/unsloth-q3-k-xl-4gpu-f16-iq4-20261114.json \
  --engine /baseline/strata --output /out/baseline-sweep \
  --scenario sweep --max-new 4 --seed 424242 --expect observe --run
```

Baseline release SHA-256 expected from the runbook:
`fbc52b5228c32709e86b009fe690db5c3edd9f83f0d2433fdc72fffc87f775b2`.
The candidate identity and completed qualification are recorded in `evidence/`.
Remote sources/builds and complete raw artifacts:
`/home/radneon/strata-rd/serve-stop-bound-20261115/`.
