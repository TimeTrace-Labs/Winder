"""Gate-4 prep: computes and PRINTS a candidate `artifacts/fold10_authorization.json` record --
never writes it. Writing that file is gate 4 itself, a deliberate, human-triggered act (per
`notes/fold10_preregistration.md`); this script only assembles the content a human would review
before doing that, so the record's hashes are computed fresh from real files rather than typed
by hand.

**What this computes, and from where -- every value traceable, nothing invented:**
- `authorized_script`/`authorized_sha256`: the one real, current SHA-256 of `scripts/
  fold10_nominal_eval.py`'s own on-disk bytes.
- `frozen_parameters`: `fold10_nominal_eval.frozen_parameters_dict()` -- the same named constants
  that script's own argparse defaults draw from; this and gate 1's own defaults can never drift
  apart because both read the one source.
- `frozen_inputs`: `fold10_nominal_eval._resolve_frozen_inputs()`, called at the REAL event's own
  full scale (`DEFAULT_ARMS` x `DEFAULT_STEPS`, all 12 (arm, step) checkpoints), using
  `resolve_default_paths`/`resolve_checkpoints` -- the identical resolution logic `main()` itself
  uses, so this printed template describes exactly what a real `--target-fold 10` invocation
  under the same flags would check against.

**What this does NOT compute, and why -- these are the human's own decisions, not derivable from
any file on disk:** `preregistration_doc_commit` (the git commit hash of `notes/
fold10_preregistration.md` AT THE STATE it was signed off on -- printed as a `<FILL IN>`
placeholder, not the current `HEAD`, since sign-off may happen at a later commit), `second_
reviewer_correspondence_commit` (the commit of whichever blind gate-3 review is being certified
as final), `authorized_by` (a name), `authorized_at` (an ISO-8601 timestamp AT actual sign-off
time, not whenever this script happens to run). `acceptance_statement` IS pre-filled -- it is a
fixed ceremony phrase, not a per-signer value, per the pre-registration document's own schema.

Run with no arguments to use every default `scripts/fold10_nominal_eval.py` itself would use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from fold10_nominal_eval import (
    DEFAULT_ARMS,
    DEFAULT_DATA_ROOT,
    DEFAULT_STEPS,
    _resolve_frozen_inputs,
    frozen_parameters_dict,
    resolve_checkpoints,
    resolve_default_paths,
)

AUTHORIZED_SCRIPT = "scripts/fold10_nominal_eval.py"
ACCEPTANCE_STATEMENT = "I accept this number whatever it is."


def build_template(
    *,
    data_root: str,
    artifacts_dir: str,
    roster_dir: str | None,
    lead_stats_path: str | None,
    rpeaks_npz_path: str | None,
    theta_tokens_path: str | None,
    manifest_path: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Assembles the candidate record and returns it alongside the resolved `checkpoints` dict
    (for the caller's own preflight-summary printing). Pure aside from the file reads/hashes it
    performs -- never writes anything, never touches `artifacts/fold10_authorization.json`.
    """
    resolved_paths = resolve_default_paths(
        artifacts_dir,
        roster_dir=roster_dir,
        lead_stats_path=lead_stats_path,
        rpeaks_npz_path=rpeaks_npz_path,
        theta_tokens_path=theta_tokens_path,
        manifest_path=manifest_path,
    )
    checkpoints = resolve_checkpoints(resolved_paths["roster_dir"], DEFAULT_ARMS, DEFAULT_STEPS)
    with open(AUTHORIZED_SCRIPT, "rb") as fh:
        authorized_sha256 = hashlib.sha256(fh.read()).hexdigest()
    frozen_inputs = _resolve_frozen_inputs(
        data_root=data_root,
        roster_dir=resolved_paths["roster_dir"],
        lead_stats_path=resolved_paths["lead_stats_path"],
        rpeaks_npz_path=resolved_paths["rpeaks_npz_path"],
        theta_tokens_path=resolved_paths["theta_tokens_path"],
        manifest_path=resolved_paths["manifest_path"],
        checkpoints=checkpoints,
    )
    template: dict[str, Any] = {
        "authorized_script": AUTHORIZED_SCRIPT,
        "authorized_sha256": authorized_sha256,
        "preregistration_doc_commit": "<FILL IN: git commit hash of notes/"
        "fold10_preregistration.md at the state it is signed off on>",
        "second_reviewer_correspondence_commit": "<FILL IN: commit hash of the blind gate-3 "
        "review being certified as final>",
        "authorized_by": "<FILL IN: name>",
        "authorized_at": "<FILL IN: ISO-8601 timestamp, at actual sign-off time>",
        "acceptance_statement": ACCEPTANCE_STATEMENT,
        "frozen_parameters": frozen_parameters_dict(),
        "frozen_inputs": frozen_inputs,
    }
    return template, checkpoints


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--roster-dir", default=None, help="default <artifacts-dir>/roster")
    ap.add_argument(
        "--lead-stats-path", default=None, help="default <artifacts-dir>/lead_stats_f1to9.json"
    )
    ap.add_argument(
        "--rpeaks-npz-path",
        default=None,
        help="default <artifacts-dir>/reference/phase/rpeaks.npz",
    )
    ap.add_argument(
        "--theta-tokens-path", default=None, help="default <artifacts-dir>/phase/theta_tokens.npz"
    )
    ap.add_argument(
        "--manifest-path", default=None, help="default <artifacts-dir>/manifest.parquet"
    )
    args = ap.parse_args(argv)

    template, checkpoints = build_template(
        data_root=args.data_root,
        artifacts_dir=args.artifacts_dir,
        roster_dir=args.roster_dir,
        lead_stats_path=args.lead_stats_path,
        rpeaks_npz_path=args.rpeaks_npz_path,
        theta_tokens_path=args.theta_tokens_path,
        manifest_path=args.manifest_path,
    )

    print(
        f"[print_fold10_authorization_template] resolved {len(checkpoints)}/"
        f"{len(DEFAULT_ARMS) * len(DEFAULT_STEPS)} real-event (arm, step) checkpoints; "
        f"authorized_sha256 computed from the current on-disk bytes of {AUTHORIZED_SCRIPT!r}.",
        flush=True,
    )
    print(
        "[print_fold10_authorization_template] this is a CANDIDATE record only -- it has NOT "
        "been written to artifacts/fold10_authorization.json. Fill in the four <FILL IN> "
        "fields, confirm gates 1-4 actually ran and passed, then a human writes this file "
        "themselves. This script never does.",
        flush=True,
    )
    print(json.dumps(template, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
