CURRENT2026-10-06: production is shefowl Hybrid204800 with CPU FP16 vision; see SHEFOWL_PRODUCTION.md. Prior Coder details below are historical.

# CPU vision deployment —2026-10-06

Current service: strata-api running and healthy, vision enabled through the SAME
OpenAI/Anthropic API on0.0.0.0:8080. Existing API key and API-only/privacy settings
are unchanged. The user's independent inference-engine project remains paused.

## Configuration
- Main native args include --vision (required for GENI image embeddings)
- vision.exe=/src/build-vision-cpu/bin/strata-vision
- vision.mmproj=/src/runtime/vision/mmproj-Qwen3.8-Flash-Next-BF16.gguf
- vision.model=original first coder IQ1_M GGUF split, loaded vocab-only by helper
- vision.gpu=false; threads16; max_tokens1024 per image
- Main text model still runs on both MI50; context131072 and residentKV32768INT8 unchanged
- Main engine has small image-position tables (~1.5MiB/device at this context),
  plus image context/KV usage. CPU encoder weights and working buffers are inRAM.

## Reproducibility
Source6f32ec070f23ced9f50e704d854d775da52591ab; llama dependency
3cf03257f219afbe7334045ff7c6a06ac68c627d. Existing pinned build image is retained.
Run sh deploy/build-cpu-vision.sh to rebuild the helper. Its CMake cache disables
CUDA/HIP/Vulkan/SYCL/Metal/RPC/OpenCL, backend dynamic loading is off; static CPU
backend uses portable AVX2/FMA/F16C. No GPU devices were exposed to its build or
standalone probe containers. ldd shows only libc/libm/libstdc++/libgcc.

mmproj pinned HF revision5348543e0147355ac9cbcb031184a3546350988e:
https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/5348543e0147355ac9cbcb031184a3546350988e/mmproj-Qwen3.8-Flash-Next-BF16.gguf
907,543,008bytes, GGUF magic checked; no full model hashing or conversion.

## Verification
Standalone no-GPU probe encoded a generated1024x1024 red-left/blue-right image
into1024 finite2560-wide embeddings in12.549s; observed VRAM unchanged.
Full authenticated API answered red blue on both requests: first17.159s,
repeat0.198s with image/prompt reuse (not a fresh-image speed claim). Prompt1062,
completion3 tokens. Subsequent text arithmetic returned42. Unauthorized models
still401, UI still404, health images=true. Encoder process showed no GPU
libraries or GPU device descriptors. RSS/peak after test1,180,960KiB (~1.13GiB),
not a bound for every possible image. This is functional smoke, not broad OCR,
vision accuracy, concurrency or occupied128K qualification.

The first API attempt omitted native --vision and rejected GENI; automatic
rollback restored text service. Corrected attempt passed; both receipts retained.
Artifacts: logs/cpu-vision-20261006, deploy/cpu-vision-verification.json.
The enable-cpu-vision.py script is the one-time audited deployment operation,
not a recurring startup command; backups and original failure logs are preserved.
Normal use remains ./start.sh, ./stop.sh, ./logs.sh --follow. Start automatically
spawns the CPU helper through server.py; no separate daemon is needed.
