"""`RunProvenance`: a lightweight dataclass capturing what identifies and reproduces one eval run
-- checkpoint identity (dir + SHA-256), the fold protocol, the data artifacts' own hashes, and the
software/hardware environment. New in this port (not extracted from any reference-repo script,
per the design brief).

Replaces `winder.eval.record.EvalRecord`'s exact pydantic schema role with something simpler, for
the promoted-numerics modules in this package (readout/tasks/robustness/gates/comparison) that
have no single fixed artifact shape the way the JEPA MVP's frozen-probe eval does. It does NOT
replace `EvalRecord` itself, which stays the schema for that MVP's own artifact -- `record.py` is
left in place per this project's own P4 commit note (a documented scope-delta, not an oversight);
removing it is a separate release-cleanup decision, not this phase's job.

The outer JSON envelope this module assembles (`assemble_report`) matches the report schema every
other artifact in this project uses: top-level `status`/`milestone_id`/`metrics`/`provenance`/
`decisions`/`questions` keys, so a `RunProvenance`-backed report is directly comparable to one
built around a different inner schema.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import torch

from winder.data.integrity import git_sha, sha256_file

__all__ = ["RunProvenance", "assemble_report"]


@dataclass(frozen=True)
class RunProvenance:
    """Everything needed to identify, and in principle reproduce, one eval run.

    `checkpoint_sha256`/`manifest_sha256`/`lead_stats_sha256` are `None` when the corresponding
    path is not supplied or does not exist on disk (e.g. a synthetic-fixture test run with no
    real manifest) -- provenance degrades gracefully rather than requiring every field to be a
    real file.
    """

    checkpoint_dir: str
    checkpoint_sha256: str | None
    step: int
    train_folds: tuple[int, ...]
    val_fold: int
    test_fold: int
    manifest_sha256: str | None
    lead_stats_sha256: str | None
    git_sha: str | None
    torch_version: str
    device: str
    timestamp: str
    seed: int

    @classmethod
    def collect(
        cls,
        *,
        checkpoint_dir: str,
        step: int,
        train_folds: tuple[int, ...],
        val_fold: int,
        test_fold: int,
        device: str | torch.device,
        seed: int,
        state_path: str | None = None,
        manifest_path: str | None = None,
        lead_stats_path: str | None = None,
        repo_root: str | None = None,
    ) -> RunProvenance:
        """Build a `RunProvenance`, hashing whichever of `state_path`/`manifest_path`/
        `lead_stats_path` are given (each resolves to `None` if its path argument is `None` or
        does not exist on disk, rather than raising)."""

        def _hash_or_none(path: str | None) -> str | None:
            return sha256_file(path) if path and os.path.isfile(path) else None

        return cls(
            checkpoint_dir=checkpoint_dir,
            checkpoint_sha256=_hash_or_none(state_path),
            step=int(step),
            train_folds=tuple(train_folds),
            val_fold=int(val_fold),
            test_fold=int(test_fold),
            manifest_sha256=_hash_or_none(manifest_path),
            lead_stats_sha256=_hash_or_none(lead_stats_path),
            git_sha=git_sha(repo_root or os.getcwd()),
            torch_version=torch.__version__,
            device=str(device),
            timestamp=datetime.now(UTC).isoformat(),
            seed=int(seed),
        )


def assemble_report(
    status: str,
    milestone_id: str,
    metrics: dict[str, Any],
    provenance: RunProvenance,
    decisions: list[str],
    *,
    questions: list[str] | None = None,
) -> dict[str, Any]:
    """The report-schema envelope every artifact in this project conforms to: top-level
    `status`/`milestone_id`/`metrics`/`provenance`/`decisions`/`questions`, with `provenance`
    flattened from the `RunProvenance` dataclass rather than a hand-built dict."""
    return {
        "status": status,
        "milestone_id": milestone_id,
        "metrics": metrics,
        "provenance": asdict(provenance),
        "decisions": list(decisions),
        "questions": list(questions) if questions is not None else [],
    }
