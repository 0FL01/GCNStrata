# AGENTS.md

This fork targets **AMD gfx906 only** (MI50 / Radeon Pro VII). Other upstream backend sources are retained,
but NVIDIA CUDA, other AMD architectures and SYCL are not supported qualification targets. It has a C++/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

Focus development and optimization on **Inkling 2 Mini** inference in `strata-poly` (`src/poly/`) on gfx906.
Use [PolyStrata's Inkling 2 Mini inference graph](https://github.com/0FL01/PolyStrata/blob/main/docs/INKLING2.md)
as the architecture reference. Qwen-specific work is deferred.

For Qwen only: compare four-MI50 prefill against the [B3 Q4_0 baseline](bench/results/2026-11-22-gfx906-prefill/README.md),
not the older Q4_1 deployment. Use `--adapt-every 0` and greedy decoding for token-divergence checks because the
adaptive expert cache makes outputs depend on timing.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

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
