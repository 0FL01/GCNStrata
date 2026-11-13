# Primary HC W-negative cache build harness

Preparation only. No MI50 access, HIP compilation, model run, GPU execution or transfer was performed by this worker. Only Python tests and ordinary g++ CPU fixtures were run on the controller.

## Root invocation

Verify SHA256SUMS, independently review this harness, then root may transfer the immutable package to the intended MI50 build host. Run as the existing uid1000 build owner, not OS uid0:

    python3 run_reproduce.py --output /absolute/fresh/output --execute

Without --execute the command only validates. Output must not exist and must be outside the source, package and v4 result trees. No automatic retries. A failed receipt stops admission for analysis.

## Exact build contract

Source is sealed v2 primary-only cache. source-review.json is independently pinned; source-pins.json, patch and public Gemm header are verified against the seal. Both gemm.cu and prefill.cpp are rebuilt because private PrefillImpl gains an atomic flag. Public Gemm layout/header stays unchanged.

Original source/object pins, original compile argv and paths (including HIP CUID-sensitive flags), exact linker argv, pinned image, repaired verifier archive, v4 receipt pins, fresh non-hardlinked minimal closure, read-only originals, CPU/memory/pid/time bounds, no network and no GPU devices are inherited unchanged from c183-hc-cost-build-v1-20261009. Only the two prefill archive members are replaced; every other member and linked input is hash checked.

Complete device equality must preserve all 55 modules, every .hip_fatbin byte, module/kernel metadata, raw wrapper bytes and normalized wrapper relocation binding. TLS gate additionally retains readelf symbols/groups/relocations for both rebuilt objects and symbols for the final linked binary. Both TUs must define and reference the same 8-byte weak/unique COMDAT TLS symbol, and the final symtab must have exactly one definition. Pinned identical header prevents contradictory inline definitions. No relaxation is permitted if any gate fails.

## Receipts and runtime limits

Host: host-receipt.json. Inner: reproduction/receipt.json. Both success statuses are PASS_PRIMARY_CACHE_BUILD_DEVICE_AND_TLS_EQUALITY; passed must be true. Host cleanup must pass, returncode must be zero, inner hash must match. The inner receipt retains observer_sources, observer_objects and observer_binary_sha256 compatibility names. They refer to this cache candidate. New source_review binds the conditional source seal; tls_sharing.passed must be true, two objects and linked_symbol present. reproduction/device-equality.json retains baseline/observer inventories of exactly 55 modules and both equality booleans true. Every evidence file is hash inventoried.

Build success is not model/runtime/performance admission. Only the reviewed generate flow is eligible: one nonconcurrent public run, resident immutable W, no mutating callbacks or reset/delete/reload/rebind races, fixed request context; peers remain baseline. No cross-request cache claims.

Enable cache with exact STRATA_HC_WNEG_PRIMARY=1. Qualification uses STRATA_HC_WNEG_QUALIFY=1. Timing must unset STRATA_HC_WNEG_QUALIFY. Both modes use the SAME binary and original flags; counters remain present in both. Qualification emits at scope end only. No compilation macro, extra compiler option or HIPCUID change is allowed.

## Controller tests

    python3 -m unittest -v test_reproduce test_tls_verify test_tls_cross_tu

Includes isolation, exact commands, source/v4/closure/member rejection, preserved failure receipts, timeout cleanup, real 55-module comparator self/mutation checks, TLS synthetic fail-closed cases, and real g++ cross-TU address sharing/thread-isolation fixture. CPU fixture success does not replace actual ROCm object/link TLS proof.
