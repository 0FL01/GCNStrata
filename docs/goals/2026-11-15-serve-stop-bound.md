# Goal: bound speculative serve commits to native output

Status: active
Source: user-approved RECON/plan of 2026-11-15; subsequent request to save the plan, implement iteratively, and commit/push to `0FL01/Strata/main`.
Last updated: 2026-11-15

## Objective

Fix native `max_new`/EOS completion in the common CUDA/HIP serve driver: commit only the input rows corresponding to emitted outputs, leaving the final emitted token pending. Qualify the change on the existing four-gfx906 Q3 deployment profile and publish the focused fix and evidence to the fork's `main`.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text. Finish when every required outcome is resolved and affected constraints remain satisfied. Stop substantive work at a proven external blocker, an approved budget boundary, or when no remaining in-scope action has a falsifiable expected result; record the exact evidence and smallest unlock.

## Frozen Contract

### Required Outcomes

- R1: Native length/EOS completion keeps the correct input prefix in pipeline and non-merged serial decode.
  - Source: approved plan, sections "What changes" and phase 2; upstream `243fdb213c270ffd0be76c17c7d1cc3d4a2361dd` and `13eedf552417a6a791a7d13c6597e24300ee3a09`.
  - Acceptance: emitted native IDs are unchanged under matched conditions; committed input rows stop at the output budget/first EOS, last output stays pending; merged prompt rows are not truncated; no zero-row verifier commit.
  - Primary evidence: ported host policy regression and an old-fails/new-passes native boundary reproduction.
  - Status: in_progress
  - Evidence: current `c723df72` commits before emission truncation at `generate.cpp:15008` and `:15329`; upstream-only host test passed 2736 cases in a temporary directory, not yet a test of this fork's candidate.
- R2: Preserve existing verdict consumers and correctly settle all affected GPU stages before reuse.
  - Source: approved plan and its three independent reviews.
  - Acceptance: raw accepted drafts still feed policy/calibration/acceptance diagnostics; bounded input-row count feeds commit/consumed/position/pending/rollback. A launched speculative B is drained and affected front stages restored/replayed; valid state and direct-live/parked continuation agree with a justified prefix reference.
  - Primary evidence: targeted host consumer regression plus trace-backed four-GPU state/continuation qualification, including a sampled request.
  - Status: pending
  - Evidence: planned; existing `STRATA_STATE_HASH` alone covers only the primary stage and is insufficient.
- R3: Publish the verified focused change, tests, provenance and limitations to `0FL01/Strata/main`.
  - Source: user: "коммит пуши делай (в main)"; approved plan phase 3.
  - Acceptance: required builds/checks and independent implementation review are complete (or an explicit user exception is recorded); uninstrumented candidate passes boundary/continuation smoke; remote main SHA matches the published commit, no force push.
  - Primary evidence: build/test/review artifacts, commit and `git ls-remote origin refs/heads/main`.
  - Status: pending
  - Evidence: planning-only commits may precede runtime publication; no candidate is qualified yet.

### Constraints

- Preserve the deployed Q5 correction (`c723df72`), FP16, INT8 KV, MTP, #267 memory locks and `vm.compact_unevictable_allowed=0`. Do not tune clocks or model parameters as part of the fix.
- Production remains Q3_K_XL + IQ4_NL. Deployment is a separate approval, not implied by code publication. Never promote an instrumented audit binary.
- GPU experiments require an explicitly approved downtime window: pause clients, check requests/open API connections, running test jobs, GPU use and KFD holders before lifecycle changes. Before resuming production, confirm test processes have released the GPUs. Idle checks are not an atomic drain.
- Build every affected backend consumer before implementation review, or stop for an explicit exception/toolchain decision. A missing build is not a pass.
- Never expose API secrets, Docker environment contents or private request bodies. Preserve the dirty remote deployment checkout; work in an isolated R&D tree.

### Non-goals

- No upstream merge, model switch, sampler-policy change, DMA/adaptive-cache fix, dense Q4/Q5 arithmetic change or speculative Q6 audit.
- No universal client-visible snapshot claim for Python stop strings, transport cancellation or nonfinite responses; preserve their existing contracts.
- No new permanent runtime switch, dependency, service, model registry or automatic rollback.
- The experimental SYCL driver is a separate source file/project and is not changed or claimed fixed. No broad benchmark or full numerical requalification of unchanged kernels.

## Change Envelope

- Runtime: `include/strata/core/serve_stop_bound.hpp`, the pipeline and non-merged serial callsites in `src/program/generate.cpp`. Keep verifier APIs/kernels unchanged unless a direct induced blocker is proved.
- Tests: upstream `tests/core/serve_stop_bound_test.cpp`, CMake registration, minimal actual-callsite/consumer regression using the existing source-binding convention if needed.
- Evidence: a focused `bench/results/2026-11-15-serve-stop-bound/` report, reproducible runner/fixtures and audit-only instrumentation patch. Diagnostics must not enter the deployed binary.
- Direct consumers: final-stage commit, every front-stage commit/rollback replay, `consumed`, pending token/position, raw-verdict policy/calibration/statistics, live conversation and parked snapshots.

## Approved Plan

### 1. Reproduce on the Q5-fixed baseline

Use `c723df72`, not the older Q5-buggy binary. Freeze prompt IDs and sampling. Capture an accepted two-row window with only one remaining native output, emitted IDs, consumed prefix and the next request's reuse. A baseline run that never crosses the boundary is not a negative control.

### 2. Minimal semantic port and qualification

- Port both upstream changes' meaning without `STRATA_USE_HIP` gating: gfx906 uses CUDA-compatible HIP. Do not cherry-pick unrelated upstream code or the restricted pending-prefix experiment.
- Keep raw accepted **drafts** separate from kept **input rows** (different units). Do not train `policy.observe*()` or rejection statistics on an artificial budget cut. Preserve EOS policy exclusion and merged prompt commits.
- Run the ported host test, a focused verdict-consumer regression, and actual-callsite merged exclusion check. No generic simulator/test framework.
- Build common CUDA and native HIP consumers, including the production gfx906 compatibility configuration. Missing toolchains require an explicit decision before code review/publication.
- On four GPUs after approval, use the unchanged Q3 profile: pipeline2, spec6, mtp-max-t2, no coupled draft, suffix0, PCIe0, adapt0, PIPED_CONCURRENT0. Ordinary windows are at most two rows. PIPE_SHAPE_PREP=1 is configured but its two-stage eligibility does not hold for four stages.
- Short budget sweep 1/2/3/4/5/7/8; EOS at each actual two-row output position; both modified decode paths. No Cartesian product. If a test EOS ID is selected from observed outputs, exclude earlier occurrences and prove the actual stop position and original acceptance from diagnostics.
- Exercise no-B and one actual B-launched case. Prove early full A commit on each reached front stage, B launch, restore, bounded replay and post-drain state. Matching output or a configured pipeline alone is insufficient. Record a real rejected draft as a control where reached.
- Compare exact consumed/pending IDs and valid per-stage GDN/PLE/KV/index state plus applicable draft state. Existing first-stage-only hashes need minimal audit-only per-stage capture. Exclude stale cells; establish the draft KV valid interval instead of blindly using `live.size()`. Exact FP equality requires matched computation; an unjustified reference is a blocker to full-state qualification.
- Continue in the same process from exact returned token IDs, separately proving direct-live and park/restore routes and expected RESUME/REUSED/read counts. Include the actual sampled defaults (temperature1, top_p0.95, top_k20) with recorded seeds, not greedy-only evidence.
- STOP/cancel are narrow nonregression guards, not new semantics. An uninstrumented candidate must pass a boundary/continuation smoke before publication or promotion.

### 3. Review and publish

After the required builds and checks, independently review the actual diff, fix only proven in-scope defects, publish a focused commit with upstream attribution and evidence, push `main`, and verify the remote SHA. Any later approved deployment changes only the selected binary, with a compose backup, and leaves Q3 in place.

## Current Checkpoint

- Closes: R1 and host portion of R2.
- Smallest next action: implement the pure bound and separate raw verdict from committed rows; add the nearest host regressions.
- Expected evidence: policy and actual-consumer tests pass; the old unbounded callsite fails the regression; source diff contains no unrelated change.
- Stop/replan if: tests reveal a direct induced state-consumer defect; do not add adjacent fixes.

## Current State

- Baseline: local and remote fork main `c723df72f54b21d54db073710981163a435d1573`.
- Production: Q3_K_XL + IQ4_NL, healthy, engine SHA256 `fbc52b5228c32709e86b009fe690db5c3edd9f83f0d2433fdc72fffc87f775b2`; no lifecycle changes during RECON.
- Scope evidence: PCIe/adaptation disabled in live argv. No penalties in config, but actual client sampling is unknown (`api_monitor=false`); do not claim penalties universally off.
- Build availability: pinned image `sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646` provides HIP; `nvcc` is absent on local host, remote host and the only available remote image. Native CUDA build remains an external-toolchain gate.
- GPU window: not yet approved. Continue host/source work without stopping production.

## Material Decisions / Checkpoints

- 2026-11-15: two RECON agents and three independent plan reviewers; final plan has no blocking review findings. Incorporated raw-verdict separation, all-stage rollback evidence, valid draft-KV interval, backend scope and operational gates. Plan review is not implementation qualification.
- 2026-11-15: user approved iterative implementation and main publication; saved this contract before runtime edits. Source reference is the supplied audit plus inspected upstream commits; the linked sandbox archive was not available locally.

## Completion

Not complete. Candidate implementation, backend builds, GPU qualification, implementation review and fix publication remain outstanding.
