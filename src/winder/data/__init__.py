"""Data layer: WFDB decode, PTB-XL metadata/folds, phase extraction, manifest ledger.

Ported from ttl-phase's disposable research campaign (see tests/fixtures/MANIFEST.json for
the pinned source commit) behind winder's contract conventions. Algorithm bodies (WFDB
decode, R-peak detection, decimation) are near-verbatim, since they're validated only
empirically against real data with no reference library to check against; interfaces and a
named list of correctness/provenance bugs were fixed along the way -- see individual module
docstrings and PR history for specifics.
"""

from winder.data.decimation import decimate_to, out_len
from winder.data.folds import FoldConfig, calibration_subset, folds, train_minus_calibration
from winder.data.integrity import assemble_integrity_report, config_hash, git_sha, sha256_file
from winder.data.manifest import (
    NO_REASON,
    REASON_CODES,
    REASON_DESCRIPTIONS,
    Manifest,
    RecordRow,
    multihot,
)
from winder.data.norm_stats import LeadStats, fit_lead_stats
from winder.data.normalization import (
    NORM_REGISTRY,
    CorpusStatsNormConfig,
    NormConfig,
    PerBeatNormConfig,
    RawNormConfig,
    apply_corpus_stats,
    beat_rms,
    normalize,
    resolve_norm_config,
)
from winder.data.phase import (
    ALL_FLAGS,
    BIN_EXCLUDE,
    DetectorParams,
    PhaseQCConfig,
    PhaseResult,
    bin_phase,
    detect_rpeaks,
    extract_phase,
    jitter_estimate,
    phase_from_rpeaks,
    refine_rpeaks,
)
from winder.data.ptbxl import (
    KEEP_COLS,
    LEAD_ORDER,
    MULTIHOT_COLS,
    SUPERCLASSES,
    UNLABELED,
    WEIGHT_COLS,
    assign_superclass,
    load_metadata,
    load_scp_statements,
    parse_scp_codes,
    read_and_decimate_500hz,
)
from winder.data.wfdb_io import (
    ReadError,
    WfdbHeader,
    read_header,
    read_record,
    write_format16,
)

__all__ = [
    "ReadError",
    "WfdbHeader",
    "read_header",
    "read_record",
    "write_format16",
    "decimate_to",
    "out_len",
    "ALL_FLAGS",
    "BIN_EXCLUDE",
    "DetectorParams",
    "PhaseQCConfig",
    "PhaseResult",
    "bin_phase",
    "detect_rpeaks",
    "extract_phase",
    "jitter_estimate",
    "phase_from_rpeaks",
    "refine_rpeaks",
    "FoldConfig",
    "calibration_subset",
    "folds",
    "train_minus_calibration",
    "SUPERCLASSES",
    "MULTIHOT_COLS",
    "WEIGHT_COLS",
    "UNLABELED",
    "LEAD_ORDER",
    "KEEP_COLS",
    "parse_scp_codes",
    "load_scp_statements",
    "assign_superclass",
    "load_metadata",
    "read_and_decimate_500hz",
    "NO_REASON",
    "REASON_CODES",
    "REASON_DESCRIPTIONS",
    "Manifest",
    "RecordRow",
    "multihot",
    "NORM_REGISTRY",
    "NormConfig",
    "PerBeatNormConfig",
    "RawNormConfig",
    "CorpusStatsNormConfig",
    "beat_rms",
    "normalize",
    "apply_corpus_stats",
    "resolve_norm_config",
    "LeadStats",
    "fit_lead_stats",
    "sha256_file",
    "git_sha",
    "config_hash",
    "assemble_integrity_report",
]
