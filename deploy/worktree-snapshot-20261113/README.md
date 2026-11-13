# amude deployment snapshot (2026-11-13)

Preservation commit, not a source change. Nothing here is built by CMake.

- Repo root `compose.yaml`, `configs/`, `deploy/`, `retired/`, `start.sh`,
  `stop.sh`, `logs.sh`, `write-deploy.py`, `deploy-files.json`: the
  deployment root of the production host, at the paths it uses there.
  `compose.yaml` takes the API key from the environment.
- `worktree-on-ab495ec.patch`: uncommitted source changes of the host's
  0.1.40 worktree (17 files). Base commit in `BASE_COMMIT`. Not applicable
  to `main` as is.
- `ab495ec-context-overflow-backport.patch`: the one local commit above
  `perf/gfx906-hybrid-qualified`.
- `untracked/`: source files that were never committed there.
- `iq4nl-iter4/worktree-on-53e6c6a5.patch`: two uncommitted files of the
  IQ4_NL experiment.
- `iq4nl-gfx906-eval-20261111.bundle`: the local-only branch of that name.
- `gpu-duct-fan-control.sh`: the host fan controller, reading four cards.
- `SKIPPED_LARGE_FILES.txt`: files over 512 KiB left on the host (size, path).

Production binary: built from `53e6c6a5`; see
`bench/results/2026-11-13-gfx906-four-gpu/`.
