"""The transport loss: notes/internal/phase_equivariance_notes_v13.pdf Eq. 13,
`L_trans = E_{(t,t')}[1 - cos<R_Delta zhat_t, zhat_t'>]`, over ALL valid within-record token
pairs -- no pair sampler, no RNG. This is a deliberate design choice, not an oversight: a
predecessor prototype's own pair sampler correlated Delta with absolute phase and was 6.3x
non-uniform (see `winder.operators.harmonic`'s module docstring), which is exactly the condition
Eq. 16's non-equivariant floor forbids. All-pairs-within-record has no sampling distribution to
get wrong -- its Delta marginal is whatever the record's own token phases produce, measured (not
assumed) once, corpus-wide, by `scripts/m0_phase_calibration.py` (uniformity ratio 1.076,
Cramer's V 0.0003 against theta_src -- both comfortably inside the pre-registered thresholds).

Averaging convention: per-record mean over that record's own valid pairs, THEN mean over records
-- record-uniform, mirroring `winder.jepa.regularizers.SigReg`'s own per-timestep reduction
convention (that module's docstring) rather than letting a record with more valid tokens
(quadratically more pairs) dominate the batch loss.

**The opt-in radial term** (`radial_weight`, default 0.0 -- `artifacts/campaign_x2x2/
pre_launch_addendum.md`'s pinned convention): `radial_weight * 0.5 * (a/b + b/a - 2)` per valid
pair, `a = ||z_s||`, `b = ||z_t||` UNnormalised, same pair mask and record-uniform mean as the
directional term. `radial_weight = 0` reproduces the directional-only loss bitwise (the radial
branch is structurally skipped, never computed-then-zero-weighted); `radial_weight = 1` realises
the canonical 1:1 geometry, `(1 - cos phi) + 0.5*(a/b + b/a - 2) = ||R_Delta z_s - z_t||^2 /
(2ab)` by the law of cosines -- one half of the geometric-mean-normalised full vector defect, the
same global scale as the directional-only loss, so a caller's lambda_trans dose does not silently
double. The addendum's declared deviation from the radial protocol draft v0.9 applies: ALL valid
within-record pairs, not within-beat pairs only.

**A NaN-theta token is excluded from every quantity this module computes, and from nothing
else.** `winder.transport.dataset.PhaseTaggedDataset` gives an all-NaN theta row to a record
excluded from the phase-QC pool (`scripts/m0_phase_calibration.py`'s own scope, CON-04's
doctrine that phase-clock QC has nothing to do with which records a phase-less JEPA pretrains
on) -- such a record contributes zero pairs here and is otherwise untouched: its `L_pred`/`L_sig`
computation in `winder.jepa.train.train_step` never sees `theta` at all.

**Q = I is exact WLOG for this loss alone, not for the joint objective it sits inside.**
`winder.operators.harmonic.HarmonicTransport`'s own docstring proves the operator side of this
(both `cos<.,.>` and the eps-clamped l2 normaliser below are O(K)-invariant, and `Q^T` is
absorbable into the projector's unconstrained final `nn.Linear`). The bounded exception lives
one layer further downstream, in `winder.jepa.predictor.TransformerPredictor`: its two
`nn.LayerNorm`s are NOT exactly O(K)-equivariant for an arbitrary orthogonal Q -- attention
(query/key/value projections, softmax over dot products), the MLP, the mask token, and the
relative-position bias all conjugate through a change of basis cleanly, but LayerNorm's
mean-subtract step needs `Q @ ones == ones` (a 99.2%-of-the-group subgroup, `O(K-1)` of
`O(K)`, at K=256) and its learned affine `gamma`/`beta` need `Q` to fix them exactly, which only
signed permutations do in general. Since `L_trans` is attached to the PROJECTOR's output `z`
(`winder.jepa.train.train_step`, the same tensor SIGReg already reads), not the predictor's
hidden state, this restriction never actually enters the training objective `Q = I` is
justified against -- it would only matter for a hypothetical variant that also demodulated or
transported the predictor's own output, which this MVP does not do. Logged here as a permanent
caveat because it is a claim about the joint objective, not a claim this loss's own maths
requires anyone to re-derive: it is available as `ln_gamma_cv` in `winder.transport.diagnostics`
(M4), which bounds how much of `O(K)` a trained predictor's own LayerNorm affine would restrict,
were this ever extended to the predictor's output.
"""

from dataclasses import dataclass

import torch

from winder.operators.harmonic import HarmonicTransport

__all__ = ["TransportLossOutput", "transport_loss"]

_EPS = 1e-8


@dataclass(frozen=True)
class TransportLossOutput:
    loss: torch.Tensor  # scalar, attached to the autograd graph
    floor: torch.Tensor  # scalar, Eq. 16's closed form -- diagnostic only, computed under no_grad
    n_valid_pairs: int  # total ordered (s, t) pairs, s != t, across the whole batch
    n_records_with_pairs: int  # records contributing at least one valid pair
    directional_term: torch.Tensor  # detached scalar: the 1 - cos component alone
    radial_term: torch.Tensor  # detached scalar: the 0.5*(a/b + b/a - 2) component alone; NaN
    # when radial_weight == 0 (structurally skipped -- "not applicable", never a silently-wrong
    # 0.0, matching winder.jepa.train.StepMetrics' own NaN-sentinel convention)


def transport_loss(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    *,
    radial_weight: float = 0.0,
    stop_gradient_target: bool = False,
) -> TransportLossOutput:
    """z: (B, L, K) projector output (K == operator.dimension). theta: (B, L), NaN where a
    token's phase is undefined. Returns a record-uniform-averaged loss over every valid,
    non-self (s, t) pair within each record -- see module docstring for why there is no sampler.

    `radial_weight` (default 0.0, every pre-existing caller's implicit value) adds the module
    docstring's opt-in radial term: `loss = directional + radial_weight * radial`, with 0.0
    structurally skipping the radial branch so the directional-only value stays bitwise identical
    to before this parameter existed (eq-28 paired comparability).

    `stop_gradient_target` (default False, every pre-existing caller's implicit value) detaches
    the TARGET branch of every ordered pair -- token t's normalised vector in the directional
    term, and its norm in the radial term -- so gradient flows only through the transported
    source. Pair enumeration is symmetric (every ordered (s, t) with s != t), so each token still
    receives gradient as the SOURCE of its reverse pairs: the symmetrised stop-gradient
    convention (BYOL/SimSiam-style), here WITHOUT any EMA or predictor asymmetry. Forward values
    are bitwise unchanged -- the flag reshapes only the backward pass, which is what makes a
    flag-on arm a clean paired contrast against its flag-off sibling (campaign_x2x2 X4).
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, L, K), got shape {tuple(z.shape)}")
    b, n_tok, k = z.shape
    if theta.shape != (b, n_tok):
        raise ValueError(
            f"theta shape {tuple(theta.shape)} must equal z's leading dims {(b, n_tok)}"
        )
    if k != operator.dimension:
        raise ValueError(f"z's last dim {k} != operator.dimension {operator.dimension}")

    # A floor, not a forced downcast -- mirrors winder.jepa.regularizers.SigReg's own dtype
    # floor: promote anything below fp32 (e.g. bf16 from an autocast context) up to fp32, but
    # leave fp32/fp64 input as-is so an fp64 caller (torch.autograd.gradcheck) is not silently
    # downcast.
    compute_dtype = z.dtype if z.dtype in (torch.float32, torch.float64) else torch.float32
    z = z.to(compute_dtype)
    theta = theta.to(compute_dtype)

    valid = torch.isfinite(theta)  # (B, L)
    eye = torch.eye(n_tok, dtype=torch.bool, device=z.device)
    # pair_valid[b, s, t] = token s and token t of record b both have a defined phase, s != t.
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye.unsqueeze(0)

    # Sanitise BOTH theta and z at invalid positions before anything derived from them is
    # computed -- not only theta. A caller's z is never expected to be non-finite at an
    # excluded token (the encoder always produces a real number; only theta is NaN there), but
    # "never read" must hold even if it were: without this, `torch.where(pair_valid, ...)`
    # below correctly zeroes the FORWARD value at an invalid pair, but the BACKWARD pass would
    # still multiply an incoming zero gradient by that branch's own (NaN) local Jacobian entry
    # -- 0 * NaN = NaN in IEEE arithmetic -- silently poisoning z.grad at the excluded token's
    # position despite the loss VALUE being provably unaffected. Filling z here, before zhat is
    # computed, keeps NaN out of the graph entirely rather than trying to mask it out later.
    z_filled = torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))
    token_norm = z_filled.norm(dim=-1, keepdim=True)  # (B, L, 1); exactly 0 at invalid tokens
    zhat = z_filled / (token_norm + _EPS)  # Eq. 10's clamped form

    theta_filled = torch.where(valid, theta, torch.zeros_like(theta))
    # delta[b, s, t] = theta[b, t] - theta[b, s]: transport FROM source token s TO target token t.
    delta = theta_filled.unsqueeze(1) - theta_filled.unsqueeze(2)

    # stop_gradient_target: the target branch reads a detached view; source branch unchanged.
    zhat_tgt = zhat.detach() if stop_gradient_target else zhat
    src = zhat.unsqueeze(2).expand(b, n_tok, n_tok, k)
    tgt = zhat_tgt.unsqueeze(1).expand(b, n_tok, n_tok, k)
    transported = operator.transport(src, delta)  # (B, L, L, K)
    pair_defect = 1.0 - (transported * tgt).sum(dim=-1)  # (B, L, L), Eq. 13's per-pair term

    k0 = operator.k0
    src_k0 = zhat[..., :k0].unsqueeze(2).expand(b, n_tok, n_tok, k0)
    tgt_k0 = zhat[..., :k0].unsqueeze(1).expand(b, n_tok, n_tok, k0)
    with torch.no_grad():
        floor_defect = 1.0 - (src_k0 * tgt_k0).sum(dim=-1)  # (B, L, L), Eq. 16's closed form

    directional = _record_uniform_mean(pair_defect, pair_valid)
    with torch.no_grad():
        floor = _record_uniform_mean(floor_defect, pair_valid)

    if radial_weight != 0.0:
        # a = ||z_s||, b = ||z_t|| on the UNnormalised tokens. An invalid token's z_filled norm
        # is exactly 0 (z_filled above), so an unguarded ratio would put inf into the forward
        # graph and 0 * inf = NaN into the backward one -- the same IEEE bug class the z_filled
        # comment above documents. REPLACE those norms with 1.0 before any division (the same
        # fill-don't-mask pattern): torch.where routes the backward to the constant branch at
        # invalid positions, so an excluded token's z.grad stays exactly zero, not merely finite.
        squeezed_norm = token_norm.squeeze(-1)  # (B, L)
        norm_safe = torch.where(valid, squeezed_norm, torch.ones_like(squeezed_norm))
        norm_safe_tgt = norm_safe.detach() if stop_gradient_target else norm_safe
        norm_s = norm_safe.unsqueeze(2)  # a: (B, L, 1), source token s, broadcast over targets
        norm_t = norm_safe_tgt.unsqueeze(1)  # b: (B, 1, L), target token t, broadcast over sources
        # 0.5 * (a/b + b/a - 2) per pair, eps-clamped like zhat above. At radial_weight = 1 the
        # per-pair total is (1 - cos phi) + 0.5*(a/b + b/a - 2) = ||R_Delta z_s - z_t||^2 / (2ab)
        # -- the pinned canonical 1:1 geometry (module docstring's radial paragraph).
        radial_defect = 0.5 * (
            norm_s / (norm_t + _EPS) + norm_t / (norm_s + _EPS) - 2.0
        )  # (B, L, L)
        radial = _record_uniform_mean(radial_defect, pair_valid)
        loss = directional + radial_weight * radial
        radial_term = radial.detach()
    else:
        # Structural skip (winder.jepa.train's own doctrine): radial_weight == 0.0 runs the exact
        # pre-radial op sequence -- `loss` IS the directional tensor, bitwise -- and reports the
        # radial component as the NaN "not applicable" sentinel (TransportLossOutput's own field
        # comment), never a silently-wrong 0.0.
        loss = directional
        radial_term = directional.detach().new_full((), float("nan"))

    n_valid_pairs = int(pair_valid.sum())
    n_records_with_pairs = int((pair_valid.sum(dim=(1, 2)) > 0).sum())
    return TransportLossOutput(
        loss=loss,
        floor=floor,
        n_valid_pairs=n_valid_pairs,
        n_records_with_pairs=n_records_with_pairs,
        directional_term=directional.detach(),
        radial_term=radial_term,
    )


def _record_uniform_mean(pair_values: torch.Tensor, pair_valid: torch.Tensor) -> torch.Tensor:
    """(B, L, L) pair-wise values -> a single scalar: mean over each record's own valid pairs,
    then mean over records that have at least one -- record-uniform, not pair-uniform (module
    docstring). A record with zero valid pairs contributes exactly nothing (M3-A5's zero-pair
    safety): it is excluded from the final mean by boolean indexing, which also excludes it from
    the backward pass rather than merely zeroing its forward value."""
    per_record_sum = (pair_values * pair_valid).sum(dim=(1, 2))
    per_record_count = pair_valid.sum(dim=(1, 2))
    has_pairs = per_record_count > 0
    if not bool(has_pairs.any()):
        return pair_values.new_zeros(())
    per_record_mean = per_record_sum[has_pairs] / per_record_count[has_pairs]
    return per_record_mean.mean()
