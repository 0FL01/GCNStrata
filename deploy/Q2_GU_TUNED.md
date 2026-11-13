# Q2_0 gate/up refinement

## Scope and implementation

Strata 0.1.40 / 82f46a8, two gfx906 GPUs, shefowl Hybrid. No weight or KV precision changes.
- Mode 7: production baseline.
- Mode 9: earlier Q2_0 gate/up LDS/token-inner candidate.
- Modes 10 and 11: 16/32-row blocks with a uniform singleton path bypassing LDS copies and barriers. Rejected for promotion: singleton regressions remained.
- Mode 12: mode-9 layout with exact integer expansion of packed Q2_0 values into signed bytes, replacing byte-permute lookup/interleave. Eight DP4A steps and the float scale/summation order remain unchanged. Down projection is unchanged.

## Evidence

- All 65,536 packed 16-bit codewords matched the original byte-permute semantics (131,072 packed 32-bit outputs).
- Modes 10/11: 288 real-weight cases, bitwise equal.
- Mode 12: 144 real-weight cases, bitwise equal: each GPU, layers 0/23/47, groups 1/8/32, tokens 1 through 8.
- Mode-12 whole-expert median speedup: T1 0.9483x, T2 1.1529x, T4 1.3839x, T8 1.3864x. Gate/up phase includes SwiGLU and quantization.
- Full model: 4K smoke, then A-B-B-A with 65,536 input and 512 output tokens, capacity 204,800. INT8 KV, 32K resident KV, PLE locked in RAM, MTP4/suffix3, fixed seed and sampling, no prompt reuse.
- All 512 output IDs and 342/444 combined draft counters matched across four model runs. This is not a full-logit or broad-quality qualification.

{
  "reference_mode": 7,
  "candidate_mode": 12,
  "A_TG": 45.34063034102885,
  "B_TG": 46.404553446806965,
  "TG_gain_percent": 2.346511501441051,
  "A_PP": 584.2193211534802,
  "B_PP": 584.4389213349278,
  "all512_output_ids_equal": true,
  "mode12_component_cases": 144,
  "mode10_11_component_cases": 288,
  "integer_decode_codewords_exhaustive": 65536,
  "candidate_binary_sha256": "cee2d26fbd0ff259af495932e70ee7a771007c098695520a131227b3217c41ef",
  "production_mode": 7,
  "promoted_default": false,
  "restore": {
    "restored_utc": "2026-10-06T13:14:04.113660+00:00",
    "health": {
      "status": "ok",
      "max_context": 204800,
      "model": "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0",
      "images": true,
      "api_key": true,
      "loaded": true,
      "service": "strata"
    },
    "production_config_unchanged": true,
    "passed": true
  }
}

## Decision

Target of >2% TG reached on this workload (+2.3465%). Keep mode 12 opt-in because singleton microbenchmarks still regress and only one end-to-end workload was measured. Production API restored in mode 7, Hybrid 200K and CPU FP16 vision unchanged. Do not present this as a universal improvement.

Raw data: logs/q2-gu-tuned-20261006. Patch: deploy/q2-gu-tuned.patch. Prior binary and source preserved in the same log folder; original unmodified mode-7 binary remains in logs/q2-gu-lds-20261006/strata-baseline. No third-party push or log transfer performed.
