# Strata0.1.40 deployment and qualification

{
  "source": "82f46a8c8f475f001ad76d92f58f4a4f8ffb0253",
  "version": "0.1.40",
  "PP": 584.3425165888863,
  "TG": 45.52446495416433,
  "PP_change_pct": 0.5397518042252969,
  "TG_change_pct": 2.4967323748299464,
  "all_512_output_ids_identical_across_versions": true,
  "binary_sha256": "57d1cb1b7d9a37ebfb5b4363030fb5e88ef83c532173f72cd52769df253ecdb6",
  "cpu_vision_sha256": "eabb8bfbb49f725cd323905ae7229fefdd796ae4fb64defd3ad5f30c4a2d62cf",
  "server_tests": {
    "run": 452,
    "skipped": 10,
    "failed": 0
  },
  "gpu_parity": "5 selected tests each GPU passed",
  "production_restore": {
    "restored_utc": "2026-10-06T11:20:21.001737+00:00",
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
  },
  "backup": "/home/radneon/strata-backups/20261006-0140-upgrade/strata-0.1.39.tar",
  "local_build_fixes": [
    "fused_gr_read_multi gfx906 return false: caller performs quantization",
    "HIP aliases for stream priority APIs",
    "exclude gfx906 from CUDA-driver VMM path, use existing HIP stubs"
  ],
  "limits": [
    "original user premature-thinking termination not replayed",
    "full occupied200K untested",
    "small throughput difference from two sequential runs, not interleaved A/B"
  ]
}

## Checks
CPU server unit tests, selected GPU parity on each device, CPU FP16 image API red/blue twice, text42, authenticated tool-call/result roundtrip, 4Ksmoke and two64K+512 native runs passed. Model/config unchanged, 204800capacity INT8KV, MTP4+suffix3, PLERAM. ProductionAPI running after benchmarks. Original thinking-only stop remains unproven fixed.

## Backup and rollback
Archive contains source/git/builds/configs/deploy/third_party. Runtime model packs and logs were deliberately left in place, not duplicated. Original build directories are also in backup folder. Existing API credentials remain in original llama/.env and were not copied. To roll back: stop Strata container, restore archive to the original Strata directory, then start.sh; do not replace models or expose credentials.
