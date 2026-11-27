<h1 align="center">GCNStrata</h1>

<p align="center"><b>Hardware-specialized LLM inference for AMD gfx906</b><br>
Vega 20 · ROCm / HIP · Multi-GPU · Long-context agents</p>

GCNStrata is an independent, hardware-specialized fork of [Strata](https://github.com/Niko1221/Strata).
Its focus is efficient local inference of **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**
and **[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** on AMD Vega 20 GPUs.

The project develops native wave64 kernels, quantized MoE execution, phase-adaptive multi-GPU scheduling,
speculative decoding, and efficient long-context session handling. The objective is lower end-to-end latency
for coding agents and multi-turn tool workflows, not just higher isolated kernel throughput.

## Hardware scope

- **Target ISA:** AMD gfx906 (Vega 20), including Instinct MI50 / MI60, Radeon VII and Radeon Pro VII.
- **Qualified reference system:** four 16 GiB gfx906 GPUs (64 GiB total VRAM), Intel Xeon E5-2698B v3
  (16 cores / 32 threads), 121 GiB usable system RAM, Linux and community ROCm/HIP 7.14.
- **Multi-GPU:** PCIe peer-to-peer transfers and topology-aware execution. The optimized four-GPU
  Inkling paths require the appropriate peer access and memory budget.
- **Other hardware:** the repository retains inherited Strata backends, but other GCN generations,
  RDNA, NVIDIA, Intel and macOS are outside this fork's qualification scope. Sharing an ISA does
  not by itself qualify a different board, driver, or GPU count.

For RDNA2-specific work, see [RedStrata](https://github.com/0FL01/RedStrata).

## Model backends

### Inkling-2-Mini — `strata-poly`

The current four-GPU production path uses a Q4_0 target, Q8_0 KV cache, a configured 256K context limit
and a native mixed-precision DFlash2 drafter.

- Fully GPU-resident routed experts on a TP2 target pair, with additional co-resident stage weights on the other pair.
- Width-invariant speculative verification, row-shared quantized GEMV and multiplicity-aware expert kernels.
- Native [DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2), fused private vocabulary head and
  zero-lag provisional context updates.
- Phase-adaptive DualTP prefill with overlapped pipeline stages, asynchronous KV deltas and balanced microbatches.
- Four-GPU target vocabulary projection, persistent attention helpers, exact filtered sampling and suffix-aware restore.
- Optional CPU image encoder ([vision build guide](tools/vision/INKLING.md)).

### Qwen3.8-Flash-Next — `strata`

The native Qwen backend targets a much larger hybrid CPU/GPU model. The previously qualified B3 Q4_0
configuration uses four layer-split GPUs, adaptive expert residency in HBM, host-resident experts and
N-gram/PLE data, INT8 KV and MTP speculative decoding.

Its gfx906-specific work includes optimized Gated Residual / hyper-connection operations,
Gated DeltaNet recurrence, Qwen Sparse Attention, grouped MoE MMQ, pipeline scheduling and
adaptive vocabulary head splitting.

Porting the latest Inkling execution improvements to Qwen, and integrating
[Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), are active research
directions. Qwen DFlash2 is not yet a qualified native feature of this fork.

## Measured performance

All numbers below refer to specific measured configurations of the four-GPU reference system,
not guaranteed performance on every gfx906 card or quantization.

### Inkling-2-Mini

**V8-R3, November 27, 2026** — Q4_0 target, mixed-precision DFlash2, Q8_0 KV,
[engine commit `bd13ef75`](https://github.com/0FL01/GCNStrata/commit/bd13ef75708dfa70cf27561d37ce682583db0870).

| Workload | Measured result |
| --- | ---: |
| Cold prefill, 19,127 fresh tokens | **2,113 tok/s** (9.05 s) |
| Long prefill, 49,152 fresh + 16,384 reused | **1,455 tok/s** (33.77 s) |
| Sampled AgentOps decode, 1,553 output tokens across nine turns | **90.1 tok/s** |
| Full nine-turn AgentOps client wall time | **33.30 s** |
| Restore, 23,283 reused + 1 fresh token | **42.3 ms** engine prompt time |
| Filtered decode, 2K prompt, top-k 0 / top-p 0.95 | **46.3 tok/s** (previously 22.7) |
| Filtered decode, 2K prompt, top-k 0 / min-p 0.05 | **133.4 tok/s** (previously 29.4) |

The filtered cases use different sampling distributions; they are not interchangeable measures of model speed.
The V8 qualification matched the checked output SHA fingerprints across eight serving phases and nine
AgentOps turns. Kernel and sampler tests separately check bitwise arithmetic and RNG parity.

[Qualification summary, baselines and limitations](bench/results/2026-11-27-gfx906-inkling2-v8/README.md).

### Qwen3.8-Flash-Next

**Historical B3 Q4_0 baseline, November 22, 2026** —
[engine commit `453e60db`](https://github.com/0FL01/GCNStrata/commit/453e60db),
four gfx906 GPUs, native MTP, INT8 KV, adaptive expert residency.

| Workload | Measured result |
| --- | ---: |
| Cold prefill, ~4K | 729 tok/s |
| Cold prefill, ~16K | 1,270 tok/s |
| Cold prefill, 60–64K | **1,756 tok/s** |
| Append ~4K after 60–64K | 574 tok/s |
| Sampled coding / explanation / review workloads | 67.9–82.1 tok/s |

These Qwen results predate the Inkling V4–V8 research. They are not estimates of the upcoming Qwen port.
[Prefill qualification and raw measurements](bench/results/2026-11-22-gfx906-prefill/after-direct/README.md).

## Phase-adaptive execution

The current Inkling path uses the same physical GPUs differently depending on the inference phase.

```text
Cold / medium prefill
  GPU 0 + GPU 1   Stage A (TP2)
        |        asynchronous activation handoff
  GPU 2 + GPU 3   Stage B (TP2)

Autoregressive / speculative decode
  GPU 0 + GPU 1   canonical target verifier (TP2)
  GPU 2 + GPU 3   DFlash2 + context-dependent attention helpers
  GPU 0..3       sharded target vocabulary projection
```

Stage weights remain resident across phase transitions. The engine preserves canonical target state
while overlapping data movement and useful compute. A scheduler chooses execution geometry for
fresh prompts, incremental appends and decode rather than forcing one layout onto every workload.

Qwen's much larger weight footprint requires a different memory and scheduling strategy: all four
GPUs hold target stages and expert caches, with host RAM supplying non-resident data. Inkling's
co-resident TP2 layout is not simply copied to Qwen.

## Build on Linux

Use a ROCm/HIP toolchain with working gfx906 libraries. The reference build uses the community
[`mixa3607/rocm-gfx906:7.14-complete`](https://hub.docker.com/r/mixa3607/rocm-gfx906) image.
The inherited wave32 HIP backend is not the gfx906 backend.

Inside a suitable build environment with CMake, Ninja, Git, Python and the ROCm Clang toolchain:

```sh
git clone https://github.com/0FL01/GCNStrata.git
cd GCNStrata

cmake -S . -B build-906 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++

cmake --build build-906 --target strata strata-poly
```

CMake fetches the pinned llama.cpp dependency unless `STRATA_GGML_DIR` points to a compatible local
checkout. See [AMD HIP build and kernel notes](docs/AMD_HIP.md) for dependencies, gfx906 compatibility,
optional Qwen HC build flags, and runtime tuning. Container execution requires access to `/dev/kfd`
and `/dev/dri`.

This is the source-build entry point, not a complete model-specific launch preset. Download compatible
model weights separately, prepare the Qwen pack when required, and choose an explicit configuration
for your hardware. Most specialized execution paths are opt-in; compiling the binaries alone does not
reproduce the benchmark settings.

## Serving and AgentOps

The Python server in `serve/` exposes OpenAI- and Anthropic-compatible APIs, including streaming,
reasoning, tool calls and model-dependent image support. Configure the appropriate native executable
(`strata` or `strata-poly`) and model paths before starting it.

Use `http://127.0.0.1:8080/v1` as the OpenAI-compatible base URL for a local server configured on port
8080. Require an API key before exposing it to another machine. See the
[API and server reference](docs/DETAILS.md) and [reverse proxy guide](docs/REVERSE_PROXY.md).
Some inherited setup instructions target the upstream project rather than the qualified gfx906 path.

## Research priorities

- **Qwen3.8-Flash-Next:** native DFlash2 integration, width-invariant GDN/QSA verification, HC fusion,
  residency-aware multi-row MoE and cross-architecture ports from Inkling.
- **Inkling-2-Mini:** long-context attention, adaptive speculation, exact sampling, GPU-resident
  session state and further phase-aware scheduling.
- **Shared infrastructure:** wave64 quantized kernels, PCIe-aware execution, low-overhead state
  transfers, reproducible kernel and end-to-end benchmarks.

Performance changes are evaluated on real hardware with paired controls, output/state correctness
checks and documented negative results. See [AGENTS.md](AGENTS.md) for development and qualification rules.

## Credits and license

GCNStrata builds on [Niko1221/Strata](https://github.com/Niko1221/Strata) and the work of its contributors.
The Inkling backend draws on PolyStrata / [RedStrata](https://github.com/0FL01/RedStrata);
the model and drafter work comes from the respective Qwen, Thinking Machines and Inco AI teams.
The project uses [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) and other dependencies.
Quantized weights and model derivatives are distributed by their respective authors.

The inherited engine is open source under the [MIT License](LICENSE). Individual dependencies
and model weights have their own licenses. Upstream architecture and historical benchmark
documentation remain in the repository for attribution and technical reference.
