# Exact binary-restoration recovery overlay

Do not rebuild. The diagnostic already compiled successfully as0548008ed5c40a7e69ac65092e0f4767d6da70af1bc6cc72c2d391537c7b0fdd. Original build.json truthfully reports a failed restoration of only the mutable build-906/strata, and remains immutable. Root subsequently restored the original c183 bytes/mtime without changing permissions or ownership.

This package preserves original_package.v2.json byte-for-byte and keeps build_capture.py, capture_arm.py, the capture source/patch, strict200K validator and parser unchanged. The runner adds one explicit recovery path. It does not execute recovery, rebuild, restart the API, change permissions or retry a native request.

## Root command after package verification

python3 run_paired_200k.py --package package.json --execute --binary-sha256 0548008ed5c40a7e69ac65092e0f4767d6da70af1bc6cc72c2d391537c7b0fdd --restore-receipt /home/radneon/strata/logs/perf-campaign-20261009/qsa-200k-capture-build/recovery.json --restore-receipt-sha256 3b8fa0813a5ed5fe603dbe9e34bc1c35646150897a96da5a3122c6210db0a37b

Default invocation without --execute remains offline. INPUTS.sha256 and45 CPU/source tests cover the delivered files. README.v2.md is historical background, not instructions to rerun the build.

## Exact provenance chain

- Original failed build: fe93c7e6588c88069ed318f22cc589f7f0fa44ae42ca6c903b862ef369ea3387
- Original compiled package v2: 0c9fd8de4db8cd9153474c470197af1c1e3250390290239c5697565fbc8e5614
- Candidate: 0548008ed5c40a7e69ac65092e0f4767d6da70af1bc6cc72c2d391537c7b0fdd
- Recovery receipt: 3b8fa0813a5ed5fe603dbe9e34bc1c35646150897a96da5a3122c6210db0a37b
- Raw root restoration evidence: ca909cc67f1f10a47714774e55a3b0735d9cbd7711a3dabcf2c747132241bac1

Only build_artifacts_restored=false and restore_failed_files=['build-906/strata'] may be superseded by verified separate recovery evidence. All original compiler/source/kernel/binary gates remain mandatory. A temporary validation view applies those two logical exceptions without writing or relabeling the original receipt.

Before native execution, recovery_support.py rechecks the original receipt/package, live restored binary bytes/mode/mtime, original backup, all other before-snapshot file hashes, archived baseline, candidate, compiled-source snapshots and exact selector/kernel archive bytes. It binds raw recovery evidence and recorded ownership without requiring ownership changes. Existing idle-state and exact-identity permission preflight then run before the unchanged bounded control/diagnostic pair.

The output summary keeps the original build receipt hash and adds build_recovery containing both original failed status and separate successful live recheck. New runtime package identity is distinct from the v2 package used to compile. PASS_CAPTURE_ONLY, exact counters, native/reference IDs, cleanup and replay contracts are unchanged. Original failed build and all previous packages remain available.

Preparation was controller-only; no preparer MI50 access, build or native/GPU execution. Root already performed the separately recorded one-file restoration.
