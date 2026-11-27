# AGENTS.md

GCNStrata is a hardware-specialized fork of [Strata](https://github.com/Niko1221/Strata) targeting **AMD gfx906 only** (Vega 20: Instinct MI50 / MI60, Radeon VII / Radeon Pro VII). The maintained qualification platform is four 16 GiB gfx906 GPUs on Linux with ROCm/HIP. Other upstream backend sources are retained, but NVIDIA CUDA, other AMD architectures and SYCL are not supported qualification targets.

Focus development and optimization on **both Qwen3.8-Flash-Next and Inkling-2-Mini** on gfx906. Qwen uses the native `strata` engine (`src/program/`); Inkling uses `strata-poly` (`src/poly/`). Use [RedStrata's Inkling architecture reference](https://github.com/0FL01/RedStrata/blob/main/docs/INKLING2.md) as a donor reference, not as a substitute for measurement on gfx906.

For Qwen, compare against the [qualified B3 Q4_0 baseline](bench/results/2026-11-22-gfx906-prefill/after-direct/README.md), not the older Q4_1 deployment. Use `--adapt-every 0` and greedy decoding for token-divergence checks because the adaptive expert cache makes outputs depend on timing. Native DFlash2 for Qwen is a research goal, not an already qualified feature. For Inkling, use the [V8 qualification baseline](bench/results/2026-11-27-gfx906-inkling2-v8/README.md).

## Installing GCNStrata for a user

Use the manual gfx906 source-build route in [README.md](README.md) and [docs/AMD_HIP.md](docs/AMD_HIP.md). Check GPU ISA, ROCm libraries, VRAM, RAM and disk before preparing model weights and an explicit engine/server configuration. Do not assume the inherited one-click installer selects the qualified gfx906 path. Start and verify the server, then connect the user's apps. Never expose the server beyond `127.0.0.1` without `--api-key`. Existing engine names (`strata`, `strata-poly`), environment variables and API contracts remain unchanged.

## Working on the code

- gfx906 experiments use separate Git worktrees; local `main` can lag `origin/main` while holding unfinished work.
  Before resuming, use `git worktree list` and `git status -sb -uall` in the relevant tree. For unrelated work,
  fetch `origin` and create a new worktree from `origin/main`; leave dirty trees and their untracked files in place.
- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- gfx906 builds use `-DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906` (CUDA-source HIP compatibility),
  not the separate wave32 `STRATA_ENABLE_HIP` backend. Multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build and qualify the changed runtime on gfx906 before review. Do not require or download other-backend
  toolchains because a shared file also serves upstream targets; expand hardware scope only on explicit user request.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
