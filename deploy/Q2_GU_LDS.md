# Q2_0 gate/up LDS experiment

Source: Strata0.1.40 at82f46a8 plus documented gfx906 build fixes. Hardware: two gfx906 GPUs. Model: shefowl Hybrid revision55568c1.

## Implementation

Default STRATA_EXP_MODE=7 remains unchanged. Opt-in mode9 extends native_gu_lds_kernel to Q2_0, with existing vec_dot_q2_0_q8_1, same lane-strided accumulation and warp reduction. Activations staged in LDS, token-inner loop allows weight reuse. Down projection already uses LDS in mode7 and stays identical in mode9. No model/KV precision change. Benchmark selects experts across all512 rather than first256.

## Qualification

90 real-weight component cases: each GPU, layers0/23/47, groups1/8/32, tokens1/2/4/6/8. Final expert outputs bitwise identical mode7/mode9. Timings use upstream interleaved three-round minima; per-phase gate/up timing includes SwiGLU+quantize.

4K candidate smoke passed. A-B-B-A fresh-process model runs:65536input+512output,context204800,INT8KV32768resident,PLERAMlocked,MTP4+suffix3,temperature1/top_p.95/top_k20/seed12345. Same512ids and342/444draftcounts all four. Native timings exclude model load. Model graph path exercised. Full logits were not compared; full occupied200K/other workloads not qualified.

{
  "A_TG": 45.431353845471286,
  "B_TG": 46.223779064781006,
  "TG_gain_pct": 1.7442254131476043,
  "A_PP": 584.5150253789902,
  "B_PP": 584.3873278119004,
  "all512ids_identical": true,
  "cases": 90,
  "median_component_by_tokens": {
    "1": {
      "all": 0.9322548180949026,
      "gu": 0.883089770354906
    },
    "2": {
      "all": 1.1263367526797339,
      "gu": 1.2277587282288986
    },
    "4": {
      "all": 1.3724137931034481,
      "gu": 1.667350973520979
    },
    "6": {
      "all": 1.2594595884367072,
      "gu": 1.4897698209718668
    },
    "8": {
      "all": 1.3768527249597866,
      "gu": 1.6935305321185576
    }
  },
  "production_mode": 7,
  "candidate_mode": 9,
  "promoted_default": false,
  "reason": "T=1 component regression; small single-workload full-model gain",
  "binary_sha256": "b8805f8bdbebd7fb8d8234af8d5212be4731156d09505e59d84a8c58a6513a0a",
  "baseline_sha256": "57d1cb1b7d9a37ebfb5b4363030fb5e88ef83c532173f72cd52769df253ecdb6"
}

## Decision

Candidate retained opt-in; production stays mode7. T1 component regression outweighs claiming universal promotion from a1.74%gain on one workload. API Hybrid200K+CPUFP16vision restored12:26:45UTC. Original binary and source snapshots retained beside this report. Patch:deploy/q2-gu-lds-mode9.patch. Do not push to third-party upstream without authorization.
