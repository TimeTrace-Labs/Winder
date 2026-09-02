"""One place the PTB-XL data root's and the shared artifacts directory's defaults live, so they
stop being per-script literals.

Every script that reads raw PTB-XL (`build_manifest.py`, `fit_lead_stats.py`, `pretrain.py`,
`eval_suite.py`, `accept.py`, `detection_battery.py`, `fold10_nominal_eval.py`, `p7_smoke.py`, and
the figure drivers) previously hardcoded its own
`os.path.expanduser("~/anisotropy_scratch/ttl-phase/data/ptbxl")` as its `--data-root` CLI
default -- a path from a different, predecessor project that a fresh clone will not have. All of
them remain overridable on the CLI regardless; this only changes what an UNSET flag resolves to.

`WINDER_DATA_ROOT`, when set, wins outright. Absent that, the old hardcoded path is kept as the
fallback so this machine's existing invocations (and any script or note that still names the old
path) keep working unchanged -- this is a portability improvement, not a behaviour change for
anyone who does nothing differently.

`default_artifacts_dir` is the analogous helper for the shared, top-level `artifacts/` directory
(manifest, theta tokens, lead stats) -- distinct from a single run's OWN `--artifacts-dir` output
directory (e.g. `artifacts/roster/signal_seed0`), which already defaults relative to cwd and needed
no change. `winder.ablations._COMMON_RECIPE_FLAGS` reads the shared directory via this helper so
that `WINDER_ARTIFACTS_DIR`, if set, moves every input path the crowned recipe reads, not just the
per-run output location.
"""

import os

__all__ = ["default_artifacts_dir", "default_data_root"]

_LEGACY_DEFAULT = os.path.expanduser("~/anisotropy_scratch/ttl-phase/data/ptbxl")


def default_data_root() -> str:
    """`$WINDER_DATA_ROOT` if set, else the historical default. Every `--data-root` CLI flag in
    this repo should use this as its `default=`, not re-derive the fallback path itself."""
    return os.environ.get("WINDER_DATA_ROOT", _LEGACY_DEFAULT)


def default_artifacts_dir() -> str:
    """`$WINDER_ARTIFACTS_DIR` if set, else `"artifacts"` -- the same cwd-relative default every
    script's own `--artifacts-dir` flag already uses."""
    return os.environ.get("WINDER_ARTIFACTS_DIR", "artifacts")
