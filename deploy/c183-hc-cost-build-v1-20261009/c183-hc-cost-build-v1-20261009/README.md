# HC host-clock cost build delta

Prepared from reviewed c183-counts-observer-build-v1-20261009. This package is not a build result or runtime admission. No MI50 access, HIP compilation, model run, or GPU execution occurred during preparation.

Only cost observer source/header hashes and outer host receipt status differ from the reviewed build path. The exact v4 host/inner receipts, c183 source/object pins, repaired engine verifier, original compile argv and paths, two-object archive replacement, closure copy and link, isolation and full device-byte/metadata comparator remain unchanged. The observer is built into a fresh scratch closure with read-only originals. Full device equality covers 55 modules. No cache patch is included.

Run CPU tests: python3 -m unittest -v test_reproduce

Root alone may authorize run_reproduce.py --output ABSOLUTE_FRESH_DIRECTORY --execute on the intended build host. Default validates offline. The package manifest must verify first. Build success uses PASS_COST_OBSERVER_BUILD_DEVICE_EQUALITY. Independently review the actual host, inner, device gate and binary hashes before runtime admission; no placeholders constitute admission.
