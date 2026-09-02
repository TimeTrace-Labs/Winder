"""Dataset integrity reporting: SHA-256 of PTB-XL's metadata CSVs, current git commit, a config
hash, and a per-split record/patient summary -- so a fitted artifact's provenance can be checked
against a specific download and code state, not trusted by name alone.

Patient-overlap checking is NOT reimplemented here: `winder.data.folds.folds()` already asserts
it unconditionally on every call (see that module's docstring), so `assemble_integrity_report`
gets it for free by calling `folds()` to build the per-split summary.
"""

import hashlib
import os
import subprocess
from typing import Any

import pandas as pd

from winder.data.folds import FoldConfig, folds

__all__ = ["sha256_file", "git_sha", "config_hash", "assemble_integrity_report"]


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo_root: str) -> str | None:
    """Best-effort current commit SHA of the repo rooted at `repo_root`, or `None` if
    unavailable (not a git checkout, git missing, ...) -- provenance should degrade gracefully,
    never crash a run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return None


def config_hash(config_yaml: str) -> str:
    """SHA-256 of a config's YAML text, so an artifact can record exactly which config produced
    it without embedding the whole config verbatim in every downstream file."""
    return hashlib.sha256(config_yaml.encode("utf-8")).hexdigest()


def assemble_integrity_report(
    data_root: str,
    metadata: pd.DataFrame,
    *,
    fold_config: FoldConfig | None = None,
    dataset_version: str = "1.0.3",
    winder_repo_root: str | None = None,
    config_yaml: str | None = None,
) -> dict[str, Any]:
    """Dataset version, per-split record/patient counts (via `folds()`, which also re-asserts
    patient-disjointness), SHA-256 of the metadata CSVs if present, git commit, and a config
    hash if a config's YAML text is supplied."""
    fold_config = fold_config or FoldConfig()
    splits = folds(metadata, fold_config, unseal=False)

    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "n_records_total": int(len(metadata)),
        "splits": {
            name: {
                "n_records": int(len(df)),
                "n_patients": int(df["patient_id"].nunique()),
            }
            for name, df in splits.items()
        },
        "sha256": {},
        "winder_git_sha": git_sha(winder_repo_root) if winder_repo_root else None,
        "config_hash": config_hash(config_yaml) if config_yaml is not None else None,
    }
    for fname in ("ptbxl_database.csv", "scp_statements.csv"):
        path = os.path.join(data_root, fname)
        if os.path.isfile(path):
            report["sha256"][fname] = sha256_file(path)
    return report
