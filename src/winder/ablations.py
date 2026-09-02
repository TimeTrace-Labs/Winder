"""The named-arm registry: one place the crowned recipe's fixed flags live, so a future ablation
is a registry entry, not new plumbing (build plan's "Ablation infrastructure" section).

`ABLATION_ARMS` names the single factor each arm isolates; `resolve_arm` merges that onto the
fixed recipe's own common flags (batch size, steps, folds, checkpoint cadence, operator spectrum,
lead-stats path, encoder/predictor architecture, augmentation stack, device) plus the caller's
`--seed`/`--artifacts-dir`. All six arms (`signal`/`control`/`no_augmentation`/`no_sigreg`/
`shallow_predictor`, each at seeds 0 and 1) have real, complete 30,000-step checkpoints -- the
first two from Phase P8, the mechanism-attribution three from the NeurIPS-workshop-supplement
ablation campaign.

Deliberate deviation from the plan's own pseudocode ordering (`ABLATION_ARMS[name] + the common
flags`, literally arm-flags-then-common): under plain argv concatenation and argparse's own
"last `--flag value` pair wins" behaviour, that order would let the common recipe's OWN default
silently clobber a named arm's override -- e.g. `"shallow_predictor"`'s own
`--predictor-json '{"n_layers":1}'` would be immediately overwritten by the common flags'
`--predictor-json '{"n_layers":4}'` if the common flags were appended AFTER it. `resolve_arm`
therefore merges the common flags first and layers each arm's own flags ON TOP (a dict update,
keyed by flag name, not a list concatenation) -- the only order in which "one factor changed,
everything else held at the fixed recipe" is actually true for every registered arm, including
the ones that override a common flag rather than adding a new one.
"""

from collections.abc import Sequence

from winder.paths import default_artifacts_dir

__all__ = ["ABLATION_ARMS", "parse_flag_pairs", "resolve_arm"]

#: name -> the CLI flags that isolate that arm's single factor, ON TOP of `_common_recipe_flags`
#: (`resolve_arm` below). All six arms (signal/control/no_augmentation/no_sigreg/shallow_predictor,
#: each at seeds 0 and 1) have real, complete 30,000-step checkpoints under `artifacts/roster/`.
ABLATION_ARMS: dict[str, tuple[str, ...]] = {
    "signal": ("--transport-arm", "cyclic", "--lambda-trans", "1.0"),
    "control": ("--transport-arm", "cyclic", "--lambda-trans", "0.0"),
    # Each names the single factor it isolates, holding the crowned recipe
    # (via _common_recipe_flags) fixed otherwise.
    "no_augmentation": ("--transport-arm", "cyclic", "--lambda-trans", "1.0", "--augment", ""),
    "no_sigreg": ("--transport-arm", "cyclic", "--lambda-trans", "1.0", "--lambda-sig", "0.0"),
    "shallow_predictor": (
        "--transport-arm",
        "cyclic",
        "--lambda-trans",
        "1.0",
        "--predictor-json",
        '{"n_layers":1}',
    ),
}


def _common_recipe_flags(*, artifacts_base: str, device: str) -> tuple[str, ...]:
    """The fixed, crowned recipe's own flags (build plan's "Recipe: fixed, not re-derived"
    section), shared by EVERY registered arm -- `--transport-arm`/`--lambda-trans` (the one axis
    `ABLATION_ARMS` varies) and `--seed`/`--artifacts-dir` (per-run identity, `resolve_arm`'s own
    keyword arguments) are deliberately absent here; every other launch-line flag lives in exactly
    this one place, so a new ablation entry cannot accidentally drift from it.

    `artifacts_base` is the SHARED top-level artifacts directory the crowned recipe reads its
    fixed inputs from (`lead_stats_f1to9.json`, `manifest.parquet`, `phase/theta_tokens.npz`) --
    distinct from each run's own `--artifacts-dir` OUTPUT location (`resolve_arm`'s own
    `artifacts_dir` parameter, e.g. `artifacts/roster/signal_seed0`). `device` was previously a
    hardcoded `"cuda"` literal with no way to run this recipe without a GPU present; both are now
    parameters of `resolve_arm`, defaulting to the values this function was hardcoded to before,
    so every existing call site's resolved argv is unchanged unless it opts into something else."""
    return (
        "--batch-size",
        "64",
        "--steps",
        "30000",
        "--device",
        device,
        "--train-folds",
        "1,2,3,4,5,6,7,8,9",
        "--lead-stats-path",
        f"{artifacts_base}/lead_stats_f1to9.json",
        "--manifest-path",
        f"{artifacts_base}/manifest.parquet",
        "--theta-tokens-path",
        f"{artifacts_base}/phase/theta_tokens.npz",
        "--lambda-sig",
        "0.15",
        "--checkpoint-at",
        ",".join(str(2500 * n) for n in range(1, 12)),  # 2500,5000,...,27500
        "--k0",
        "4",
        "--n-j",
        "1,2,3,4,5,6,7,8,9,10",
        "--k-j",
        "24,24,20,16,12,10,8,6,4,2",
        "--encoder-name",
        "conv_trunk",
        "--predictor-json",
        '{"n_layers":4}',
        "--augment",
        "gauss,powerline,wander,ampmod,leaddrop,leadgain",
        "--augment-prob",
        "0.5",
    )


def parse_flag_pairs(flags: Sequence[str]) -> dict[str, str]:
    """`("--a", "1", "--b", "2")` -> `{"--a": "1", "--b": "2"}`: the normal form both `resolve_arm`
    and its own equivalence tests compare against, so two argv lists built in a different order
    (or with a flag repeated, "last wins") are still checked as the SAME resolved configuration
    rather than compared token-by-token. Raises on an odd-length sequence or a value where a
    `--flag` was expected -- a malformed flags tuple in the registry is a bug to catch here, not
    something to silently misparse."""
    if len(flags) % 2 != 0:
        raise ValueError(
            f"flags must be an even-length (--flag, value, ...) sequence, got {flags!r}"
        )
    out: dict[str, str] = {}
    for i in range(0, len(flags), 2):
        flag, value = flags[i], flags[i + 1]
        if not flag.startswith("--"):
            raise ValueError(f"expected a --flag at position {i} of {flags!r}, got {flag!r}")
        out[flag] = value
    return out


def resolve_arm(
    name: str,
    *,
    seed: int,
    artifacts_dir: str,
    artifacts_base: str | None = None,
    device: str = "cuda",
) -> list[str]:
    """`ABLATION_ARMS[name]` layered onto `_common_recipe_flags(...)` (arm-specific flags win on a
    per-flag basis -- module docstring's ordering paragraph), plus `--seed`/`--artifacts-dir`, as
    an argv-style flag list ready for `scripts/pretrain.py`'s own `main(argv)`.

    `artifacts_base` (default: `winder.paths.default_artifacts_dir()`, i.e. `$WINDER_ARTIFACTS_DIR`
    or `"artifacts"`) is where the crowned recipe's FIXED inputs live -- unrelated to
    `artifacts_dir`, this run's own output location. `device` defaults to `"cuda"`, the value
    every existing call site relied on when this was a hardcoded literal; pass `"cpu"` to resolve
    a CPU-runnable argv instead.

    `resolve_arm("signal", seed=0, artifacts_dir=...)` and `resolve_arm("control", seed=0, ...)`
    are, by construction, equivalent (via `parse_flag_pairs`) to the build plan's hardcoded P8
    launch line at `LT=1.0`/`LT=0.0` respectively -- `tests/test_ablations.py` checks this
    directly rather than trusting the two to stay in sync by eye."""
    if name not in ABLATION_ARMS:
        raise KeyError(
            f"unknown ablation arm {name!r} -- registered arms are {sorted(ABLATION_ARMS)}"
        )
    resolved_base = artifacts_base if artifacts_base is not None else default_artifacts_dir()
    merged: dict[str, str] = parse_flag_pairs(
        _common_recipe_flags(artifacts_base=resolved_base, device=device)
    )
    merged.update(parse_flag_pairs(ABLATION_ARMS[name]))

    argv: list[str] = []
    for flag, value in merged.items():
        argv.extend([flag, value])
    argv.extend(["--seed", str(seed), "--artifacts-dir", artifacts_dir])
    return argv
