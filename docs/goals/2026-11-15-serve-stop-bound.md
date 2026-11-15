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
  - Status: verified
  - Evidence: ported host2736+zero and actual-source consumer regression pass (baseline fails); GPU baseline overcommits while candidate passes native budget/EOS, prefix and output checks. Published proof: `bench/results/2026-11-15-serve-stop-bound/evidence/qualification-summary.json`.
- R2: Preserve existing verdict consumers and correctly settle all affected GPU stages before reuse.
  - Source: approved plan and its three independent reviews.
  - Acceptance: raw accepted drafts still feed policy/calibration/acceptance diagnostics; bounded input-row count feeds commit/consumed/position/pending/rollback. A launched speculative B is drained and affected front stages restored/replayed; valid state and direct-live/parked continuation agree with a justified prefix reference.
  - Primary evidence: targeted host consumer regression plus trace-backed four-GPU state/continuation qualification, including a sampled request.
  - Status: verified
  - Evidence: raw verdict consumers preserved; trace proves bounded replay on all three front stages; four-stage valid state and draft KV match no-B reference. Matched-layout serial and both EOS positions pass. Direct-live/parked continuation and uninstrumented smoke pass;49 requests in seven required runs.
- R3: Publish the verified focused change, tests, provenance and limitations to `0FL01/Strata/main`.
  - Source: user: "коммит пуши делай (в main)"; approved plan phase 3.
  - Acceptance: required builds/checks and independent implementation review are complete (or an explicit user exception is recorded); uninstrumented candidate passes boundary/continuation smoke; remote main SHA matches the published commit, no force push.
  - Primary evidence: build/test/review artifacts, commit and `git ls-remote origin refs/heads/main`.
  - Status: in_progress
  - Evidence: gfx906 build and qualification passed, independent implementation/checker reviews have no remaining blockers. Runtime fix and evidence are ready for main publication; binary2c3da126 is uninstrumented and smoke-qualified.
- R4: Deploy the qualified uninstrumented fix and leave production on Q3_K_XL + IQ4_NL.
  - Source: subsequent user approval: "прод отключить, что бы упростить разработку, после тестов и исправлений, развернуть прод с исправлениями версиями бинарей, продолжай".
  - Acceptance: after qualification, manually select the new binary with compose backed up; production is healthy on the same Q3 config, running binary SHA verified, authenticated text/continuation smoke passes.
  - Primary evidence: deployment backup/provenance and `stratactl status` plus authorized smoke results.
  - Status: pending
  - Evidence: downtime and deployment now authorized; preflight required before stopping.

### Constraints

- Preserve the deployed Q5 correction (`c723df72`), FP16, INT8 KV, MTP, #267 memory locks and `vm.compact_unevictable_allowed=0`. Do not tune clocks or model parameters as part of the fix.
- Production remains Q3_K_XL + IQ4_NL. Downtime and eventual deployment were explicitly approved after the plan; never promote an instrumented audit binary.
- GPU experiments require an explicitly approved downtime window: pause clients, check requests/open API connections, running test jobs, GPU use and KFD holders before lifecycle changes. Before resuming production, confirm test processes have released the GPUs. Idle checks are not an atomic drain.
- Build/qualify gfx906 before implementation review. User clarified that this fork targets gfx906 only; native CUDA/NVIDIA is not a gate. Do not download its toolchain. A supplemental native HIP build is not gfx906 GPU qualification.
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
- Build the production gfx906 compatibility configuration. The original broader build gate was superseded by the user's explicit gfx906-only scope; native CUDA and other architectures are not publication requirements.
- On four GPUs after approval, use the unchanged Q3 profile: pipeline2, spec6, mtp-max-t2, no coupled draft, suffix0, PCIe0, adapt0, PIPED_CONCURRENT0. Ordinary windows are at most two rows. PIPE_SHAPE_PREP=1 is configured but its two-stage eligibility does not hold for four stages.
- Short budget sweep 1/2/3/4/5/7/8; EOS at each actual two-row output position; both modified decode paths. No Cartesian product. If a test EOS ID is selected from observed outputs, exclude earlier occurrences and prove the actual stop position and original acceptance from diagnostics.
- Exercise no-B and one actual B-launched case. Prove early full A commit on each reached front stage, B launch, restore, bounded replay and post-drain state. Matching output or a configured pipeline alone is insufficient. Record a real rejected draft as a control where reached.
- Compare exact consumed/pending IDs and valid per-stage GDN/PLE/KV/index state plus applicable draft state. Existing first-stage-only hashes need minimal audit-only per-stage capture. Exclude stale cells; establish the draft KV valid interval instead of blindly using `live.size()`. Exact FP equality requires matched computation; an unjustified reference is a blocker to full-state qualification.
- Continue in the same process from exact returned token IDs, separately proving direct-live and park/restore routes and expected RESUME/REUSED/read counts. Include the actual sampled defaults (temperature1, top_p0.95, top_k20) with recorded seeds, not greedy-only evidence.
- STOP/cancel are narrow nonregression guards, not new semantics. An uninstrumented candidate must pass a boundary/continuation smoke before publication or promotion.

### 3. Review and publish

After the required builds and checks, independently review the actual diff, fix only proven in-scope defects, publish a focused commit with upstream attribution and evidence, push `main`, and verify the remote SHA. Any later approved deployment changes only the selected binary, with a compose backup, and leaves Q3 in place.

## Current Checkpoint

- Closes: R3/R4.
- Smallest next action: publish the qualified fix, then manually select/deploy that exact release binary and verify production.
- Expected evidence: remote main SHA, uninstrumented release hash, healthy Q3 and authenticated text/continuation checks.
- Stop/replan if: source/binary identity differs, GPU holders remain, or deployment verification fails; no automatic rollback/model switch.

## Current State

- Baseline: local and remote fork main `c723df72f54b21d54db073710981163a435d1573`.
- Production: stopped with explicit user approval through `stratactl stop`; preflight found zero inflight, all four GPUs idle and only the production KFD holder. After stop no GPU holders remained. Q3 config unchanged; previous qualified engine SHA256 `fbc52b5228c32709e86b009fe690db5c3edd9f83f0d2433fdc72fffc87f775b2`.
- Scope evidence: PCIe/adaptation disabled in live argv. No penalties in config, but actual client sampling is unknown (`api_monitor=false`); do not claim penalties universally off.
- Build image: `sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646`. gfx906 build and ported host CTest passed; uninstrumented candidate SHA256 `2c3da12662846c07fb50bb7aa0f4a3f49fb7cba32917d118c92fc044fb7345bd`. Supplemental native HIP/gfx1100 build and host CTest also passed, without GPU execution on that backend. No native CUDA build; explicitly outside user scope.
- Lab: `/home/radneon/strata-rd/serve-stop-bound-20261115/`; `src`, `gfx906-build`, `hip-build`, separate `diagnostic-{baseline,candidate}` sources/builds, `artifacts/`. Compose backup: `compose.before.yaml`. All 398 baseline source/include/CMake/cmake files matched `c723df72` before copying the lab seed.
- Host checks: actual-source serial regression passed (raw policy verdict, bounded input prefix, merged bypass); baseline source compiled and failed `committed INPUT rows must equal emitted outputs`. Pure helper passed 2736 cases plus zero remaining. CMake/CTest passed with `STRATA_BUILD_CONVERSATION_TESTS=ON STRATA_NATIVE_EXPERTS=OFF` (header-only check; no dependency download required).
- GPU negative control: `artifacts/baseline-sweep/results.json`, diagnostic baseline SHA256 `393c0e3a6925685d9f5c1ce7ab0a9e5ae0f5f75d41fbe640646dfe6186ca8cf6`. Sampled A-2 emitted `[24,11]`, but all four stages reported L=60 instead of59; continuation reused58/read2 instead of59/read1. Trace proves T2 accepted2, remaining1, B launched, early full commits on stages0/1/2 (`com=7`), then replay keep2. Budget4/8 and greedy-control also overshot.
- Candidate pipeline sweep passed: diagnostic SHA256 `42018a8a25a7bf45cf89d2bcc30bdb2a70fec46938c928587e6121ade89dcbfb`, artifacts `candidate-sweep`. All11 corresponding output-ID sequences and raw accepted-draft counts match baseline; all consumed histories and four-stage lengths are correct. A-2 now L59, replay keep1 on all three front stages, live continuation reused59/read1; park/restore passed.
- State reference `candidate-no-b` passed: same diagnostic binary, theta2, no early speculative commit. All11 requests' valid GDN/PLE/KV/index and conservative draft-KV fingerprints match the normal pipeline on all four stages. Uncut baseline windows A-1/A-3/A-5/A-7 also match candidate valid state.
- Sequential GPU job `1794736607165-73` completed: EOSfirst11, EOSlast220, startup-serial and uninstrumented release smoke passed IDs/bounds/reuse. Aggregate checker correctly stopped on an unmatched serial/pipeline state comparison (different expert placement; see decision below).
- GPU job `1794738527499-74` completed successfully. `candidate-serial-matched` retains pipeline2 startup args/capacities and its valid state matches pipeline. Aggregate checker passed all seven required runs; release smoke also passed. Recheck empty KFD holders before production start. Source runtime/release binary unchanged.

## Material Decisions / Checkpoints

- 2026-11-15: two RECON agents and three independent plan reviewers; final plan has no blocking review findings. Incorporated raw-verdict separation, all-stage rollback evidence, valid draft-KV interval, backend scope and operational gates. Plan review is not implementation qualification.
- 2026-11-15: user approved iterative implementation and main publication; saved this contract before runtime edits. Source reference is the supplied audit plus inspected upstream commits; the linked sandbox archive was not available locally.
- 2026-11-15: published the goal as `fcc2146f`; user subsequently authorized stopping production for development and deploying corrected binaries after tests. Added R4 from that explicit instruction, not from test/review scope expansion.
- 2026-11-15: user explicitly narrowed hardware scope: "наш форк только под gfx906" and requested this rule in `AGENTS.md`. Replaced the inherited all-backend build requirement; no NVIDIA toolchain download or CUDA gate. Native implementation and host checks are complete; GPU evidence, review, fix publication and deployment remain.
- 2026-11-15: EOS-first comparison exposed only unused `ple_prev` metadata on non-owner stages1/2/3 (cold=-1,-1, checkpoint restore reconstructs old token IDs). Verified `Verifier::ple_stage()` and actual split K=11,23,36: only stage0 owns layer1. Checker now requires stage0 history to equal consumed tail and retains all-stage payload comparisons; it excludes non-owner history only. Independent reviewer confirmed the source/actual artifacts and corrected EOS comparison passed. No runtime edit or tolerance relaxation.
- 2026-11-15: startup `--pipeline-windows 1` produced23023 expert slots versus pipeline2's22698; comparing their state hashes is not a matched-math oracle (CPU/GPU expert placement differs). Kept those artifacts as `candidate-serial`, added only a runner option for existing per-request `STRATA_PIPELINE_SWITCH pw=1`, and require identical startup args and expert capacities for `candidate-serial-matched`. Strict payload equality remains; no runtime change or widened tolerance.

## Completion

Not complete: R1/R2 verified; qualified fix publication and production deployment/verification remain outstanding.
