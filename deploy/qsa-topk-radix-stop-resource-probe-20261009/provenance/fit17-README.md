# Root-run public FIT17 component build

This package pins public PR1535 ca3533157d066a13b128f3daa61f56411dcf3155 headers and public parity harness. Candidate selector SHA5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a is unchanged. Public base SHA6432aa2d336496a0ebb3d2d3d14ce177caf67327eecb33bd61801497528d6aad; patchaeba2986cfd4086e79d2ef9532af405886a695d2144017ed46159105a6472b2a.

Extract to a new deploy directory. Root command:
  bash INPUT/build-host.sh INPUT /home/radneon/strata/logs/perf-campaign-20261009/qsa-fit17-public-component-build

It refuses an existing output directory, takes the shared lock, uses a digest-pinned CPU-only ROCm container with no network/GPU mounts, writes only isolated output, and does not operate API or modify an existing source tree. No support archive or private headers are used. Baseline and candidate each link with the pinned public harness plus two separately labelled external harnesses.

External capture/transition harnesses are copied byte-for-byte from previously vetted tests, not public production source. They still require TOPK_TRACE=1 at startup, which is harmless but produces NO selector probe trace in this public source. Any external run wrapper must stop requiring private probe trace lines. FIT17 modes0/1 only are relevant. Do not execute diagnostic mode2 (the unchanged transition harness accepts it syntactically but this production source has no mode2). No runtime commands are included in the build.

Output: public-parity.base/fit17, capture.base/fit17, transition.base/fit17; actual compiler version, preprocessor proof, saved gfx906 images/ISA, resource logs, link maps, hashes and build.json. isa-resources.tar.gz is the compressed export. Review ISA/resources and linkage before any root GPU test. Newly linked public binaries require parity and graph boundary qualification; prior private full-engine model results do not qualify them.
