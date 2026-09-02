"""The JEPA training step and the loop that drives it.

Implements a single causal forward pass per step: one shared `Encoder` call on the unmasked
waveform, one shared `ProjectionHead` call on its output -- not the dual-forward-pass a
bidirectional-infilling design would need. This is possible *because* the encoder is causal
(`winder.jepa.encoder`'s module docstring): token `j`'s value already depends on no sample after
its own timestamp, so there is nothing a second, raw-waveform-masked encoder pass could remove
that the encoder didn't already exclude. A context cutoff is instead enforced at the predictor:
`model.mask_sampler` returns a `CausalMaskPlan` (context prefix and the single target token
immediately after the sampled cutoff `c`, architecture-primer.html §5-6); every position at or after
the cutoff -- not only the target token -- is replaced by the predictor's own `mask_token`
(`~plan.context`), so no context-branch computation ever reads the target token's true projected
value. The masked MSE loss then scores only the target token (`plan.target`) against the same,
un-detached projected tokens.

SIGReg runs on every projected token, no floor exclusion (architecture-primer.html §5-6): the CM-07
floor existed because the (now-retired) overlapping `ResidualCnnEncoder` gave early tokens a
receptive field that ran off the record start, systematically different second moments from a real,
fully- supported token's. The non-overlapping `PatchEncoder` has no such run-in -- token 0 is
exactly as well-supported as any other. This is scoped to the encoder this MVP actually trains going
forward; a caller who instead builds a NEW `residual_cnn` run would need to reintroduce a
floor-exclusion of their own, since none of the six primitives here can name one generically for
an encoder they don't yet know the geometry of. Existing `residual_cnn` checkpoints stay loadable
and evaluable under their own geometry regardless (`winder.jepa.checkpoint`'s own docstring).

SIGReg is called on `(T, B, K)` -- transposed from the natural `(B, T, K)` token layout, never
flattened -- so its own per-timestep reduction (`N` = batch size, averaged over `T`; see
`winder.jepa.regularizers`'s module docstring) is what actually runs, not a `B*T`-pooled
statistic (architecture-primer.html §7's "the reduction is being corrected").

`L = L_pred + lambda_sig * L_sig` (design spec Sec 12) -- a straight sum with one multiplier on
the regularizer term, **not** a convex combination. An earlier design draft used
`(1 - lambda) * pred + lambda * sig`; that is not what this MVP implements.

Both forms are legitimate within this exact research lineage, not a choice between "correct" and
"deviant": LeJEPA's own paper (arXiv:2511.08544, Eq. LeJEPA) uses the convex-combination form, but
LeWorldModel (arXiv:2603.19312, Algorithm 1: `return pred_loss + lambd * sigreg_loss`) -- from an
overlapping author set, published later -- uses this exact straight sum. This MVP's choice matches
the latter, not an unexamined departure from the former.

Transport (`winder.transport.loss.transport_loss`, `L = lambda_pred*L_pred + lambda_sig*L_sig +
lambda_trans*L_trans`): attached to the SAME `z` SIGReg already reads (the projector's output,
before the predictor), never the predictor's hidden state -- see `winder.transport.loss`'s module
docstring for why. `lambda_trans == 0.0` (the default, and every pre-existing caller's implicit
value) STRUCTURALLY skips this block: `theta`/`operator` are not read even if passed, no
`+ 0.0 * L_trans` term is ever added to `total_loss`, and `StepMetrics`' four new `trans_*`/
`closure_residual` fields are set to `float("nan")`, matching `grad_norm`'s own "filled in
elsewhere, or not applicable" convention. This is deliberately NOT the same as computing
`L_trans` and multiplying by a zero weight: `0.0 * NaN` is `NaN`, so a real skip -- not merely a
zero coefficient -- is what keeps every pre-existing (transport-unaware) call to this function
bitwise identical to before this feature existed (`tests/test_jepa_train.py`'s own regression
check).

`lambda_pred` (default `1.0`, matching this function's own pre-existing implicit weight on
`pred_loss`) is the predictor's structural-skip twin, motivated by theory_closeout_v1.html §8
(eq-28): for phase-equivariant transport, the learned predictor's optimum is known in closed form
(`z_hat_t = R_Delta @ z_hat_s`), so the transport defect above IS the prediction loss with a
zero-parameter operator standing in for the predictor -- eq-28 promotes `L_trans` to the sole
prediction loss and sets `lambda_pred=0`. `lambda_pred == 0.0` STRUCTURALLY skips the predictor
forward pass, `pred_loss`, and the persistence baseline (never computed, not computed-then-
discarded): `StepMetrics.pred_loss`/`persistence_loss` are set to `float("nan")`, the SAME
convention as the four `trans_*` fields above. Mask sampling (`model.mask_sampler`) and SIGReg
stay unconditional either way -- both are decoupled from the predictor already (see
`train_step`'s own inline comment at the mask-plan draw). The predictor module itself is still
constructed and still receives its dummy-forward validation in `assemble_jepa`; only its use
inside `train_step` is gated. `AdamW` skips any parameter with a `None` gradient, so the predictor
does not need its own exclusion from an optimizer's param groups merely because `lambda_pred=0`
leaves its parameters outside `total_loss`'s autograd graph for that step.

`StepMetrics.persistence_loss` (architecture-primer.html §0) is the "nothing changes" baseline --
each record's own context-cutoff latent, repeated and scored against the same `plan.target` mask
`pred_loss` uses -- logged every step from the first run rather than reconstructed later from a
saved checkpoint. Never enters `total_loss`; a pure diagnostic, computed under `torch.no_grad()`.
Under `PatchEncoder` this is exactly `z_{t+1} = z_t` on the previous patch embedding
(architecture-primer.html §0's own framing), the same formula regardless of encoder.

`sigreg_frame` (default `"raw"`, this function's own pre-existing implicit frame) selects WHICH
tensor SIGReg reads, motivated by theory_closeout_v1.html §8.2/§8.3 (eq-28)'s closure term
`L = L_trans + lambda*L_SIG({u_t})`, `u_t = B_{-theta_t} z_t`: raw-frame SIGReg sees only the
phase-pooled shadow of the latent's covariance and is provably blind to anisotropy that co-rotates
with phase (every token arrives pre-rotated by its own theta, so the population statistic
confounds covariance with the phase trajectory itself); canonical-frame SIGReg first demodulates
every token to its own record's phase-zero frame, so it sees the full law. `sigreg_frame ==
"canonical"` demodulates `z` via `operator.transport(z, -theta_filled)` BEFORE the `(T, B, K)`
transpose -- the exact `-theta` convention `winder.transport.procrustes.
demodulated_within_record_pairs` already establishes ("every token demodulated to its OWN
record's phase-zero frame"), reused here rather than re-derived. `sigreg_frame == "raw"` (the
default) leaves `z_for_sigreg` bit-for-bit as it was before this feature existed -- a structural
addition, not a modification of the default path, matching `lambda_trans`/`lambda_pred`'s own
"every pre-existing caller is unaffected" discipline. `sigreg_frame == "canonical"` requires both
`theta` and `operator`, independent of `lambda_trans`'s value: canonical framing is a property of
what SIGReg reads, not of whether the transport term itself is in `total_loss`.

NaN-theta tokens (~10% of tokens, pre-first-R-peak slices per M0's own measurement) are filled to
`theta=0` before demodulation -- `R_0 = I`, so a filled token passes through canonical framing
UN-rotated rather than being excluded. This is deliberate "declared, structured dilution"
(theory_closeout_v1.html §8), not the masked-exclusion discipline `winder.transport.loss` uses:
that loss discards invalid pairs entirely (a well-defined operation on PAIRS), whereas SIGReg's
per-timestep statistic has no analogous per-token exclusion that preserves `N` across every
`T`-slice -- dropping filled tokens would leave a different, ragged `N` per slice, which is the
ragged-`N` exact variant theory_closeout_v1.html defers, not this MVP's scope.
`StepMetrics.theta_valid_frac` reports the batch's own fraction of finite-theta tokens (NaN when
`sigreg_frame == "raw"`, the same "not applicable" convention as the four `trans_*` fields) so a
caller can log it once per run as a sanity check, not as a per-step signal.

`sigreg_frame == "record_canonical"` (campaign_x2x2's X6) applies SIGReg to the per-record
demodulated TEMPLATE rather than to tokens: `u_t = B_{-theta_t} z_t` exactly as above, then
`ubar_r = mean over record r's VALID tokens of u_t` -- one `(B, K)` sample stack handed to
`model.regularizer` as its own documented `T = 1` case (`winder.jepa.regularizers`' module
docstring), so `N` = the number of contributing records (64 at the campaign's batch size), never
`B * n_tokens` pooled and never "N = batch size, averaged over T".

Why a THIRD frame instead of a re-dosed second one: the token-level canonical closure was
falsified twice (transport gain_fraction -0.17..-0.38, i.e. below the no-mechanism floor of -0.10,
with effective rank collapsing to 2.9-11.2). What the escape route IS was itself corrected by the
X-panel read (`artifacts/campaign_x2x2/pre_launch_addendum.md`, "Cell 3"): NOT healthy
within-record dispersion of full-mass tokens, but NORM COLLAPSE WITH RESIDUAL DIRECTION SPREAD --
the canonical arms EMPTIED their templates (X1c at 0.19*sqrt(K) mass, near-isotropic directions
only once rescaled), which is why pooled rank fell instead of holding. The old
"dispersion escape" phrasing is superseded and must not be cited.

The route-blocking property survives that correction, and reads more directly under it: a record
mean penalises BOTH failure modes from the same unit-variance target. Emptying a template
violates it from BELOW (mass loss), and incoherent within-record energy shrinks `ubar_r` toward
zero at fixed token energy -- measured on synthetic latents, injecting within-record incoherence
into diverse templates leaves the token-level statistic flat (1.06 -> 1.07) while driving the
record-level statistic 1.07 -> 11.7. Within-record COHERENCE -- the thing `L_trans` exists to
build -- is untouched by construction. It also points the regulariser at the exact feature the
`z/demodulated` readout consumes (`winder.eval.pooling.demodulated_pool`'s Proposition-4.2
estimator).

One measured indifference of this frame, stated because it is the residual escape route: the
constraint is on the TEMPLATE's distribution, not on the dispersion RATIO, so dividing by
`sqrt(1 + s**2/T)` instead of `sqrt(1 + s**2)` while dispersing at strength `s` leaves the record
statistic at its floor (1.01 -> 1.08 for s = 0 -> 16 at N=64, K=256) with the coherent share of
template energy down to 0.33 and token norms inflated 9x. The token-level frames pinned token
scale directly (that same sweep drives the token statistic 1.05 -> 73.5); this frame does not, so
once it REPLACES them the only remaining anchors against global latent inflation are `L_pred`
(quadratic in latent scale) and weight decay. `scripts/s2_pretrain_jepa.py`'s end-of-run
`spectrum_report` (`mean_norm`, `effective_rank` on pooled tokens) is the monitor for it;
`scripts/scratch_x6_escape_probe.py` is the sweep itself.

NaN-theta handling here is a DELIBERATE DIVERGENCE from the token-level canonical path above: an
invalid token is EXCLUDED from its record's mean, numerator AND denominator -- not filled to
theta=0 and passed through un-rotated. The "declared, structured dilution" argument that licenses
filling at token level does not survive the reduction: a filled token would contribute an
arbitrary phase-zero-frame vector straight INTO the template, corrupting (not diluting) a sample
of the very quantity the statistic constrains. Nor does the ragged-`N` objection apply, because
the record-level statistic has exactly one `N` -- the record count -- and dropping a record from
it is the same well-defined exclusion `winder.transport.loss` already performs on invalid pairs.
This is `winder.eval.pooling`'s demodulated-pooling convention exactly, which is also what keeps
the trained statistic and the probe's own feature the same object.

A record with ZERO valid tokens is excluded from the statistic entirely --
`record_canonical_templates` returns an all-NaN row plus a `False` in its `has_valid` mask, and
`train_step` boolean-indexes it away before the regularizer sees it -- mirroring
`winder.transport.loss._record_uniform_mean`'s zero-pair safety, and specifically NOT emitting a
zero vector, which would enter the statistic as a false "collapsed template" sample.
`StepMetrics.sigreg_n_records` reports how many records survived, so the statistic's own
data-dependent `N` (and hence its floor, which is an `N`-dependent quantity) is visible per step.

The one degenerate case is a batch where NO record has a single valid theta: the record-level
statistic is undefined there, so `sigreg_loss` is exactly zero -- the empty selection's own
`.sum()`, still attached to the autograd graph, so backward is a well-defined no-op rather than a
NaN -- and `gen_sigreg` is NOT advanced on that step, because how much RNG a call consumes is the
regularizer's own property (`NoRegularizer` consumes none), not something `train_step` may
hardcode. Reachable only if every record in the batch was excluded from the phase-QC pool
(`winder.transport.dataset.PhaseTaggedDataset`'s all-NaN row); `sigreg_n_records == 0.0` marks
such a step in the history file.

`lambda_sig_record` (default `0.0`, campaign_x2x2's X7) ADDS the record-level statistic ALONGSIDE
the token-level one instead of replacing it:

    L = lambda_pred*L_pred + lambda_sig*S({z_t}) + lambda_sig_record*S({ubar_r})
        + lambda_trans*L_trans

-- the token term in whatever `sigreg_frame` names (default, and X7's intended use: `"raw"`, the
working incumbent W3's own frame), the record term always on the demodulated templates
`record_canonical_templates` builds. Why an ADDITION and not the `record_canonical` FRAME above:
X6, which substituted one for the other, collapsed at all three doses -- transport gain +0.0000
with `trans_floor == trans_loss` in every arm (the latents fell into the non-rotating `k0`
invariant block, where the objective is trivially satisfiable) and `pred_loss` ~1e-4 at the two low
doses. A record-level statistic constrains only the ~63 per-record MEANS, so it is structurally
BLIND to within-record degeneracy: if every token of a record is identical, `ubar_r` is just that
vector and the template distribution can still look isotropic. The token-level term had been doing
anti-collapse work across 125 timesteps x 64 records that no record-level statistic can do. So the
two terms have a division of labour, not a rivalry -- the record term supplies template isotropy
(and blocks the norm-collapse route the token-level CANONICAL frame was falsified on), the token
term supplies anti-collapse.

`lambda_sig_record == 0.0` (every pre-existing caller's implicit value) is a STRUCTURAL SKIP in the
exact sense `lambda_trans`/`lambda_pred`/`transport_radial_weight` already are: the record branch is
not computed, no `+ 0.0 * S_rec` term is added (`0.0 * NaN` is `NaN` -- the whole reason a real
branch, not a zero coefficient, is what keeps a NaN-theta batch clean), the second regularizer call
never happens, and `StepMetrics.sigreg_record_loss` carries the NaN "not applicable" sentinel. It
requires `theta` and `operator` when nonzero, the same rule the demodulating frames enforce and for
the same reason (the templates are demodulated), independent of `lambda_trans`.

RNG streams: the record term's `model.regularizer` call draws its own `randn(K, n_directions)`, so
it gets its OWN named stream -- `gen_sigreg_record`, `"sigreg_record"` by `fit`'s own default
construction -- never `gen_sigreg`. A second draw on the shared stream would advance it and
desynchronise every subsequent token-level draw in the run, destroying draw-for-draw comparability
with the paired arms (W3, the X ladder) this arm is read against; `tests/test_jepa_train.py` pins
that invariant. `train_step` REFUSES to invent the generator itself when the term is active (it has
no seed to build one from, and a per-step fresh generator would freeze the direction draw across the
whole run, silently deleting the per-call resampling LeJEPA Sec 4.3 makes load-bearing). Because
the stream is consumed on every active step, it is training-run state a CKPT-01 resume MUST restore
alongside `"mask"`/`"sigreg"` -- `scripts/s2_pretrain_jepa.py` saves it in the bundle's
`generator_states`, and a resume that replayed it from seed would diverge from an uninterrupted run
(`tests/test_jepa_train.py`'s own CKPT-04 sibling test measures that divergence rather than
asserting the requirement in prose).

`sigreg_frame == "record_canonical"` TOGETHER with `lambda_sig_record != 0.0` raises: that is the
same statistic twice over two independent direction draws, i.e. a silently double-dosed record term
-- the one combination of the two mechanisms that means nothing. X7 is `"raw"` + a record dose; X6
is the frame alone.

`StepMetrics.sigreg_record_loss` reports the UNWEIGHTED record statistic every step (NaN when the
term is off), next to the token-level `sigreg_loss`, because the ladder is adjudicated on both
numbers -- and on neither alone: the record statistic's gradient was measured 16x WEAKER than the
token-level one at init but 2.5-4x STRONGER at trained checkpoints, so loss values by themselves
misstate the dose. `sigreg_n_records` (above) is filled in by EITHER the record frame or this term
-- both read the same `record_canonical_templates` helper, so it is the same quantity either way.

`sigreg_frame` is typed `str`, not `typing.Literal["raw", "canonical"]` -- a deliberate deviation
from a `Literal` closed-set type, matching `winder.config.ArmConfig.operator_name`'s own
convention (also `str`, also dict/branch-validated at runtime, never `Literal`), because
`TrainConfig` is a dataclass `winder.jepa.checkpoint.resolved_config_yaml` feeds to
`OmegaConf.structured(...)` for CKPT-02's config.yaml -- OmegaConf 2.3.1 cannot serialize a
`typing.Literal`-annotated dataclass field at all (raises `ValidationError: Unexpected type
annotation`, confirmed empirically: every real-data-mode `tests/test_s2_pretrain_smoke.py` test
failed immediately with a `Literal` annotation here). Both `TrainConfig.sigreg_frame` and
`train_step`'s own `sigreg_frame` parameter are `str`; the closed set is enforced by `train_step`'s
own `raise ValueError` below, not by the type checker.

`TrainConfig.augment` (default `""`, the V5 arm of Amendment 14's eighth addendum) names a
comma-separated subset of `AUGMENT_VOCABULARY` -- a closed vocabulary of THETA-SAFE waveform
augmentations (`augment_waveform` below), applied by `fit` to each raw waveform batch BEFORE
`train_step` sees it, training-time only by construction (nothing outside `fit`'s loop ever calls
it, so eval/checkpointing paths are structurally untouched). Every entry is pointwise in time or
lead space -- additive noise/tones/drift, multiplicative gain, lead zeroing -- so the R-peak
timing grid, and with it every theta label, stays valid; time-reversal/warping/cropping are
deliberately absent from the vocabulary because each would silently invalidate theta. Theta
tensors are never touched: `augment_waveform` does not even receive them. `""` (every
pre-existing caller's implicit value) is a STRUCTURAL SKIP in the exact `lambda_trans`/
`lambda_sig_record` sense -- `augment_waveform` is not called, and the `"augment"` RNG stream
below is never drawn from, so every pre-existing arm is bitwise unaffected
(`tests/test_s2_augment_flags.py` pins that against pre-change golden hashes).
`TrainConfig.augment_prob` (default 0.5) is the per-record, per-augmentation application
probability -- each listed augmentation gates independently per record.

RNG: the augmentation draws come from their OWN named stream -- `gen_augment`, `"augment"` by
`fit`'s own default construction -- never `gen_mask`/`gen_sigreg`/`gen_sigreg_record`, for the
same desynchronisation reason `"sigreg_record"` got its own stream. The per-step draw COUNT is
fixed given the batch size and the configured list (gates and parameters are drawn even for
records the gate turns off) so paired arms sharing an augment list stay draw-for-draw aligned.
Because the stream is consumed on every step when active, it is training-run state a CKPT-01
resume must restore alongside `"mask"`/`"sigreg"`/`"sigreg_record"` --
`scripts/s2_pretrain_jepa.py` saves it in the bundle's `generator_states`.

Determinism: explicit `torch.Generator` per named stream (`winder.determinism.generator`), never
`torch.manual_seed`. Four independent streams per step -- `"mask"`, `"sigreg"`, (only when
`lambda_sig_record != 0.0` draws from it) `"sigreg_record"`, and (only when `augment != ""` draws
from it) `"augment"` -- so a change to one source of randomness cannot silently shift the others'
draws.
"""

import itertools
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from winder.determinism import generator
from winder.jepa.model import JepaModel
from winder.operators.harmonic import HarmonicTransport
from winder.transport.loss import transport_loss

__all__ = [
    "AUGMENT_VOCABULARY",
    "TrainConfig",
    "StepMetrics",
    "lr_schedule",
    "parse_augment_spec",
    "augment_waveform",
    "record_canonical_templates",
    "train_step",
    "fit",
]


@dataclass
class TrainConfig:
    n_steps: int = 100
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-4
    warmup_steps: int = 5
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    lambda_sig: float = 0.1
    lambda_pred: float = 1.0
    lambda_trans: float = 0.0
    # Weight on the RECORD-level SIGReg term, added alongside the token-level one (campaign_x2x2's
    # X7 two-term repair) -- 0.0, the default, structurally skips it: see the module docstring's
    # own lambda_sig_record paragraphs.
    lambda_sig_record: float = 0.0
    # radial_weight for winder.transport.loss.transport_loss (campaign_x2x2 pre-launch addendum's
    # pinned convention) -- 0.0, the default, keeps the transport loss bitwise identical to the
    # directional-only formula every eq-28 arm trained with.
    transport_radial_weight: float = 0.0
    # stop_gradient_target for winder.transport.loss.transport_loss (campaign_x2x2 X4): detach
    # the target branch of every transport pair. Backward-only -- forward values bitwise equal.
    transport_stop_gradient: bool = False
    # "raw" | "canonical" | "record_canonical" -- see the module docstring's own paragraphs
    sigreg_frame: str = "raw"
    # Comma-separated subset of AUGMENT_VOCABULARY (V5's theta-safe augmentation stack) -- "",
    # the default, structurally skips augment_waveform and leaves the "augment" RNG stream
    # untouched: see the module docstring's own augment paragraphs. `str` (not a list) for the
    # same OmegaConf-serialization reason sigreg_frame is `str`, parsed by parse_augment_spec.
    augment: str = ""
    # Per-record, per-augmentation application probability (each listed augmentation gates
    # independently per record) -- read only when `augment != ""`.
    augment_prob: float = 0.5
    seed_pretrain: int = 0
    log_every: int = 1


@dataclass
class StepMetrics:
    step: int
    lr: float
    pred_loss: float
    persistence_loss: float
    sigreg_loss: float
    total_loss: float
    n_context: int
    n_target: int
    cutoff_mean: float
    grad_norm: float
    trans_loss: float
    trans_floor: float
    trans_gain: float
    trans_directional: float  # the 1 - cos component of trans_loss alone
    trans_radial: float  # the radial component alone; NaN unless transport_radial_weight != 0 --
    # the campaign_x2x2 baby-run abort criterion reads the trans_radial:trans_directional ratio
    closure_residual: float
    theta_valid_frac: float
    # Number of records contributing to the record-level statistic -- filled in by EITHER
    # sigreg_frame == "record_canonical" (X6) or lambda_sig_record != 0.0 (X7), which read the same
    # record_canonical_templates helper; NaN -- "not applicable" -- when neither is active, the same
    # sentinel convention as the four trans_* fields above. Data-dependent by construction: a
    # record with zero valid-theta tokens is excluded, so N < batch size is expected, and N is
    # what sets the statistic's own floor (winder.jepa.regularizers' module docstring). Defaulted
    # to that sentinel so a caller building a StepMetrics by hand (e.g. a test fixture) does not
    # have to name a field that only one frame ever fills in.
    sigreg_n_records: float = float("nan")
    # The UNWEIGHTED record-level statistic itself (X7's lambda_sig_record term), reported next to
    # the token-level sigreg_loss so a dose ladder can be read on both numbers -- NaN, the same
    # sentinel, when lambda_sig_record == 0.0. Exactly 0.0 (not NaN) on a batch where no record has
    # a single valid theta: that is "computed, and the statistic is undefined so the term is a
    # no-op", which is a different fact from "not computed".
    sigreg_record_loss: float = float("nan")


def lr_schedule(step: int, cfg: TrainConfig) -> float:
    """Explicit linear warmup then cosine decay to `cfg.min_lr`, as a pure function of `step` --
    not `torch.optim.lr_scheduler`, so the learning rate at any step is reproducible and loggable
    without stepping through a stateful schedule object."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    total_decay_steps = max(cfg.n_steps - cfg.warmup_steps, 1)
    progress = min((step - cfg.warmup_steps) / total_decay_steps, 1.0)
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


# --------------------------------------------------------------------------------------------
# V5: theta-safe pretraining augmentations (Amendment 14, eighth addendum; Reverso's measured
# augmentation lever, ER-JEPA corroborating). Applied by fit() to raw waveform batches only --
# never to theta, never outside the training loop.
# --------------------------------------------------------------------------------------------

#: The closed augmentation vocabulary, in canonical application order. Every entry is pointwise
#: in time or lead space, so the R-peak timing grid (and every theta label) stays valid --
#: time-reversal/warp/crop are deliberately absent (module docstring's augment paragraphs).
#: Additive artifacts (gauss/powerline/wander) are injected before the multiplicative ones
#: (ampmod/leadgain) and before lead zeroing (leaddrop) -- physically, gain and electrode
#: failure act at the electrode/amplifier on signal plus interference.
AUGMENT_VOCABULARY: tuple[str, ...] = (
    "gauss",
    "powerline",
    "wander",
    "ampmod",
    "leaddrop",
    "leadgain",
)

#: The "powerline" tone's frequency in Hz. Mains interference is 50 Hz, but 50 Hz IS the Nyquist
#: frequency at this pipeline's fs = 100 Hz: sin(2*pi*50*(n/100) + phi) = sin(phi) * (-1)^n, an
#: alternating-sign CONSTANT whose amplitude depends entirely on the drawn phase (exactly zero at
#: phi = 0) -- a degenerate, phase-gauged artifact, not a tone. 25 Hz (4 samples/cycle, well
#: inside the band) stands in as the documented "powerline-like" narrowband interference instead.
_POWERLINE_HZ: float = 25.0


def parse_augment_spec(spec: str) -> tuple[str, ...]:
    """`"wander,gauss"` -> `("gauss", "wander")`: split a comma-separated `TrainConfig.augment`
    value, validate every token against `AUGMENT_VOCABULARY` (closed set -- `ValueError` naming
    the offending token otherwise), dedupe, and return in CANONICAL vocabulary order regardless
    of input order, so the same set always applies identically. `""` -> `()`, the structural
    skip."""
    if not spec:
        return ()
    tokens = [tok.strip() for tok in spec.split(",")]
    unknown = [tok for tok in tokens if tok not in AUGMENT_VOCABULARY]
    if unknown:
        raise ValueError(
            f"unknown augmentation(s) {unknown!r} in {spec!r} -- the closed vocabulary is "
            f"{list(AUGMENT_VOCABULARY)} (module docstring's augment paragraphs)"
        )
    return tuple(name for name in AUGMENT_VOCABULARY if name in tokens)


def augment_waveform(
    waveform: torch.Tensor,
    augmentations: tuple[str, ...],
    *,
    prob: float,
    generator: torch.Generator,
    fs: float = 100.0,
) -> torch.Tensor:
    """Apply the listed theta-safe augmentations to a `(B, n_leads, n_samples)` waveform batch,
    each gated independently per record at probability `prob`, drawing ONLY from `generator`
    (the `"augment"` stream). Returns a new tensor; the input is never mutated. All amplitudes
    scale with each record's own CLEAN-input RMS (one scalar per record, over leads and samples),
    so the dose is invariant to per-record signal scale. Per step, the draw COUNT is a fixed
    function of `(B, augmentations)` -- gates and parameters are drawn even for records the gate
    turns off -- keeping paired arms draw-for-draw aligned (module docstring).

    The vocabulary (canonical order; `parse_augment_spec` emits it):
      gauss     additive white noise, sigma = 0.05 x record RMS
      powerline additive `_POWERLINE_HZ` tone (25 Hz Nyquist-safe stand-in for 50 Hz mains --
                see `_POWERLINE_HZ`), amplitude U[0.05, 0.2] x RMS, phase U[0, 2*pi), shared
                across leads
      wander    additive baseline drift: 1-3 sinusoids, f ~ U[0.05, 0.4] Hz, amplitude
                U[0.05, 0.2] x RMS each, shared across leads
      ampmod    multiplicative slow gain 1 + a*sin(2*pi*f*t + phi), a ~ U[0.02, 0.1],
                f ~ U[0.05, 0.3] Hz, record-global
      leaddrop  zero out k ~ U{1, 2, 3} randomly chosen leads for the whole record
      leadgain  per-lead constant gain g_l ~ U[0.7, 1.3] (electrode-contact variation --
                per-lead and time-constant, vs ampmod's record-global time-varying gain)

    Usage:
        augmented = augment_waveform(w, parse_augment_spec("gauss,wander"), prob=0.5,
                                     generator=gen_augment)
    """
    if waveform.ndim != 3:
        raise ValueError(f"waveform must be (B, n_leads, n_samples), got {tuple(waveform.shape)}")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    b, n_leads, n_samples = waveform.shape
    device, dtype = waveform.device, waveform.dtype
    two_pi = 2.0 * math.pi
    # One scalar per record, from the CLEAN input (not sequentially re-measured), so every
    # additive dose refers to the same physical scale however many augmentations stack.
    rms = waveform.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # (B, 1, 1)
    t = torch.arange(n_samples, dtype=dtype, device=device) / fs  # (n_samples,)

    out = waveform
    for name in augmentations:
        # Every draw happens on the generator's own (CPU) device, then moves to the waveform's --
        # the mask sampler's own convention -- so the draw sequence is device-independent.
        gate = (torch.rand(b, generator=generator) < prob).view(b, 1, 1).to(device)
        if name == "gauss":
            noise = torch.randn(b, n_leads, n_samples, generator=generator)
            out = torch.where(gate, out + 0.05 * rms * noise.to(device=device, dtype=dtype), out)
        elif name == "powerline":
            u = torch.rand(b, 2, generator=generator).to(device=device, dtype=dtype)
            amp = (0.05 + 0.15 * u[:, 0]).view(b, 1, 1) * rms
            phase = (two_pi * u[:, 1]).view(b, 1, 1)
            tone = amp * torch.sin(two_pi * _POWERLINE_HZ * t + phase)  # (B, 1, n_samples)
            out = torch.where(gate, out + tone, out)
        elif name == "wander":
            n_sins = torch.randint(1, 4, (b,), generator=generator)  # U{1, 2, 3}
            u = torch.rand(b, 3, 3, generator=generator).to(device=device, dtype=dtype)
            freq = 0.05 + 0.35 * u[:, :, 0]  # (B, 3) in [0.05, 0.4] Hz
            amp = 0.05 + 0.15 * u[:, :, 1]  # (B, 3) in [0.05, 0.2] x RMS
            phase = two_pi * u[:, :, 2]  # (B, 3)
            # A fixed 3 components are always drawn (fixed draw count); the per-record n_sins
            # merely masks the unused ones out of the sum.
            active = (torch.arange(3).unsqueeze(0) < n_sins.unsqueeze(1)).to(device, dtype)
            comps = torch.sin(two_pi * freq.unsqueeze(-1) * t + phase.unsqueeze(-1))  # (B, 3, S)
            drift = ((active * amp).unsqueeze(-1) * comps).sum(dim=1, keepdim=True) * rms
            out = torch.where(gate, out + drift, out)
        elif name == "ampmod":
            u = torch.rand(b, 3, generator=generator).to(device=device, dtype=dtype)
            a = (0.02 + 0.08 * u[:, 0]).view(b, 1, 1)
            f = (0.05 + 0.25 * u[:, 1]).view(b, 1, 1)
            phase = (two_pi * u[:, 2]).view(b, 1, 1)
            gain = 1.0 + a * torch.sin(two_pi * f * t + phase)  # (B, 1, n_samples)
            out = torch.where(gate, out * gain, out)
        elif name == "leaddrop":
            k = torch.randint(1, 4, (b,), generator=generator)  # U{1, 2, 3} leads to drop
            scores = torch.rand(b, n_leads, generator=generator)
            # rank[r, l] = position of lead l in record r's random ordering; the k lowest-ranked
            # leads are dropped -- a uniform draw of a k-subset without replacement.
            rank = scores.argsort(dim=1).argsort(dim=1)
            drop = (rank < k.unsqueeze(1)).unsqueeze(-1).to(device)  # (B, n_leads, 1)
            out = torch.where(gate & drop, torch.zeros_like(out), out)
        elif name == "leadgain":
            g = 0.7 + 0.6 * torch.rand(b, n_leads, generator=generator)
            out = torch.where(gate, out * g.unsqueeze(-1).to(device=device, dtype=dtype), out)
        else:
            raise ValueError(
                f"unknown augmentation {name!r} -- the closed vocabulary is "
                f"{list(AUGMENT_VOCABULARY)}; parse_augment_spec validates CLI input"
            )
    return out


def record_canonical_templates(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(B, n_tokens, K)`, `(B, n_tokens)` -> `(templates (B, K), has_valid (B,) bool)`: each
    record's own demodulated TEMPLATE, `ubar_r = mean over r's VALID tokens of B_{-theta_t} z_t`
    -- the `sigreg_frame == "record_canonical"` statistic's input (module docstring), and the same
    quantity `winder.eval.pooling.demodulated_pool` (Proposition 4.2) hands the `z/demodulated`
    probe.

    A record with zero valid tokens gets an all-NaN row and `has_valid[r] == False`; callers must
    boolean-index with `has_valid` before using the stack (`templates[has_valid]`), exactly as
    `winder.eval.pooling`'s own all-NaN rows must be dropped before a probe is fit. A zero vector
    is deliberately NOT emitted there -- it would read as a genuine "collapsed template" sample.

    Public (not a `_`-private helper) because the mechanism tests exercise it on synthetic latents
    a real encoder cannot be steered to produce, and the campaign's scale-measurement scripts need
    the identical estimator the training step uses, not a re-derivation of it.

    Arithmetic duplicated from `winder.eval.pooling._valid_masked_mean` rather than imported, for
    the same reason the persistence baseline below is inlined: `train.py` is the core training
    path and must not depend on `winder.eval`.

    Usage:
        templates, has_valid = record_canonical_templates(z, theta, operator)
        stat = regularizer(templates[has_valid], generator=gen_sigreg)  # (N_valid, K), T=1 case
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, n_tokens, K), got shape {tuple(z.shape)}")
    if theta.shape != z.shape[:2]:
        raise ValueError(
            f"theta shape {tuple(theta.shape)} must equal z's leading dims {tuple(z.shape[:2])}"
        )
    if z.shape[-1] != operator.dimension:
        raise ValueError(f"z's last dim {z.shape[-1]} != operator.dimension {operator.dimension}")

    valid = torch.isfinite(theta)  # (B, n_tokens)
    theta_filled = torch.where(valid, theta, torch.zeros_like(theta))
    # Fill z at invalid tokens BEFORE the rotation, not merely mask the result afterwards --
    # winder.transport.loss's own `z_filled` doctrine: `torch.where` on the output would zero the
    # forward value correctly but still multiply an incoming zero gradient by that branch's local
    # Jacobian (0 * NaN = NaN) if z were ever non-finite at an excluded token. Filling keeps NaN
    # out of the graph instead of trying to mask it out later.
    z_filled = torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))
    u = operator.transport(z_filled, -theta_filled)  # (B, n_tokens, K), the -theta convention
    # Redundant given the fill above (an excluded token's rotated value is already exactly zero),
    # kept because it is the obviously-correct statement of "excluded from the numerator".
    masked = torch.where(valid.unsqueeze(-1), u, torch.zeros_like(u))
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)  # excluded from the DENOMINATOR too
    pooled = masked.sum(dim=1) / counts  # (B, K)
    has_valid = valid.any(dim=1)  # (B,)
    templates = torch.where(has_valid.unsqueeze(-1), pooled, torch.full_like(pooled, float("nan")))
    return templates, has_valid


def train_step(
    model: JepaModel,
    waveform: torch.Tensor,
    *,
    lambda_sig: float,
    gen_mask: torch.Generator,
    gen_sigreg: torch.Generator,
    gen_sigreg_record: torch.Generator | None = None,
    theta: torch.Tensor | None = None,
    operator: HarmonicTransport | None = None,
    lambda_trans: float = 0.0,
    lambda_pred: float = 1.0,
    lambda_sig_record: float = 0.0,
    transport_radial_weight: float = 0.0,
    transport_stop_gradient: bool = False,
    sigreg_frame: str = "raw",
) -> tuple[torch.Tensor, StepMetrics]:
    """One single-causal-pass step. Returns `(total_loss, metrics)` with `total_loss` still
    attached to the autograd graph -- the caller (`fit`) owns `zero_grad`/`backward`/clip/`step`,
    so this function stays a pure forward+loss computation, testable without an optimizer.

    `theta`/`operator` are read when `lambda_trans != 0.0` OR `sigreg_frame` is either
    demodulating frame (`"canonical"`/`"record_canonical"`) -- see module docstring's "structural
    skip" and `sigreg_frame` paragraphs. `theta` is `(B, n_tokens)`, NaN where a token's phase is
    undefined (`winder.transport.dataset.PhaseTaggedDataset`'s own convention).

    `lambda_pred == 0.0` structurally skips the predictor forward pass, `pred_loss`, and the
    persistence baseline -- module docstring's eq-28 paragraph. Mask sampling and SIGReg are
    unaffected: both are already decoupled from the predictor (see the mask-plan draw below).

    `transport_radial_weight` (default 0.0) is handed straight to `transport_loss`'s own
    `radial_weight` (campaign_x2x2's pinned radial-term convention) -- at 0.0 the transport term
    stays bitwise identical to the directional-only formula, and `StepMetrics.trans_radial` is
    the NaN "not applicable" sentinel; nonzero, the separated components land in
    `StepMetrics.trans_directional`/`trans_radial` so a caller can log their ratio (the
    pre-registered baby-run abort criterion).

    `sigreg_frame == "canonical"` demodulates `z` to each token's own phase-zero frame BEFORE
    SIGReg reads it -- module docstring's `sigreg_frame` paragraph. `sigreg_frame ==
    "record_canonical"` (X6) goes one reduction further and hands SIGReg the per-record
    demodulated TEMPLATE stack `(N_valid, K)`, with invalid tokens excluded from each record's
    mean and token-less records dropped -- module docstring's `record_canonical` paragraphs. Both
    require `theta` and `operator` independent of `lambda_trans`.

    `lambda_sig_record != 0.0` (X7) instead ADDS that record-level statistic as a SECOND term
    alongside whichever token-level frame is configured -- module docstring's own
    `lambda_sig_record` paragraphs. It requires `theta`, `operator`, and its own
    `gen_sigreg_record` generator stream (never `gen_sigreg`: the second direction draw must not
    advance the token-level stream), and refuses to combine with
    `sigreg_frame == "record_canonical"`, which would dose the same statistic twice. `0.0`, the
    default, structurally skips the whole branch."""
    if waveform.ndim != 3:
        raise ValueError(f"waveform must be (B, n_leads, n_samples), got {tuple(waveform.shape)}")
    b = waveform.shape[0]
    n_tokens = int(model.config.n_tokens)

    plan = model.mask_sampler(b, n_tokens, generator=gen_mask)
    z = model.projector.forward(model.encoder.forward(waveform))  # (B, n_tokens, K)

    theta_valid_frac_val = float("nan")
    sigreg_n_records_val = float("nan")
    if sigreg_frame == "canonical":
        if theta is None or operator is None:
            raise ValueError(
                "sigreg_frame == 'canonical' requires both theta and operator to be provided "
                f"(got theta={theta!r}, operator={operator!r})"
            )
        # u_t = B_{-theta_t} z_t (theory_closeout_v1.html §8.2, eq-28), the same -theta
        # convention as winder.transport.procrustes.demodulated_within_record_pairs: every token
        # demodulated to its OWN record's phase-zero frame. NaN-theta tokens (M0's ~10%
        # pre-first-R-peak dilution) are filled to theta=0 -- R_0 = I, so a filled token passes
        # through UN-rotated rather than being excluded (module docstring's "declared, structured
        # dilution" paragraph: SIGReg's per-timestep statistic has no ragged-N exclusion, unlike
        # winder.transport.loss's pairwise masking -- out of this MVP's scope).
        theta_valid = torch.isfinite(theta)
        theta_filled = torch.where(theta_valid, theta, torch.zeros_like(theta))
        u = operator.transport(z, -theta_filled)  # (B, n_tokens, K), demodulated per token
        z_for_sigreg = u.transpose(0, 1)  # (T, B, K)
        theta_valid_frac_val = float(theta_valid.float().mean())
    elif sigreg_frame == "record_canonical":
        if theta is None or operator is None:
            raise ValueError(
                "sigreg_frame == 'record_canonical' requires both theta and operator to be "
                f"provided (got theta={theta!r}, operator={operator!r})"
            )
        # One demodulated TEMPLATE per record (module docstring's record_canonical paragraphs):
        # invalid tokens excluded from numerator AND denominator -- the declared divergence from
        # the token-level canonical path's fill-to-zero dilution -- and records with no valid
        # token dropped from the statistic rather than entered as zero vectors.
        templates, has_valid = record_canonical_templates(z, theta, operator)
        z_for_sigreg = templates[has_valid]  # (N_valid, K): the regularizer's own T = 1 case
        theta_valid_frac_val = float(torch.isfinite(theta).float().mean())
        sigreg_n_records_val = float(int(has_valid.sum()))
    elif sigreg_frame == "raw":
        z_for_sigreg = z.transpose(0, 1)  # (T, B, K): per-timestep reduction, N = batch size
    else:
        raise ValueError(
            f"sigreg_frame must be 'raw', 'canonical', or 'record_canonical', got {sigreg_frame!r}"
        )

    if z_for_sigreg.shape[0] == 0:
        # Zero-record safety, reachable ONLY in the record_canonical frame (a token-level frame's
        # leading dim is n_tokens >= 1, so this branch is inert there and both token-level frames
        # stay bitwise as they were): the empty selection's own sum is exactly 0.0 and still
        # attached to the autograd graph, so backward is a well-defined no-op instead of the NaN
        # an empty-N statistic would produce. gen_sigreg is deliberately not advanced -- module
        # docstring's degenerate-batch paragraph.
        sigreg_loss = z_for_sigreg.sum()
    else:
        sigreg_loss = model.regularizer(z_for_sigreg, generator=gen_sigreg)

    total_loss = lambda_sig * sigreg_loss

    sigreg_record_loss_val = float("nan")
    if lambda_sig_record != 0.0:
        # X7's SECOND term (module docstring's lambda_sig_record paragraphs), added to whatever the
        # token-level frame above computed rather than replacing it.
        if theta is None or operator is None:
            raise ValueError(
                "lambda_sig_record != 0.0 requires both theta and operator to be provided (the "
                "record-level statistic reads demodulated templates) "
                f"-- got theta={theta!r}, operator={operator!r}"
            )
        if gen_sigreg_record is None:
            raise ValueError(
                "lambda_sig_record != 0.0 requires its own gen_sigreg_record generator stream: the "
                "record term's regularizer call draws directions, and drawing them from gen_sigreg "
                "would advance the token-level stream and desynchronise every paired arm. Pass "
                "winder.determinism.generator(seed, 'sigreg_record') (fit() does this by default)."
            )
        if sigreg_frame == "record_canonical":
            raise ValueError(
                "sigreg_frame == 'record_canonical' with lambda_sig_record != 0.0 would apply the "
                "SAME record-level statistic twice, over two independent direction draws -- a "
                "silently double-dosed record term. X7 is a token-level frame ('raw') PLUS "
                f"lambda_sig_record={lambda_sig_record}; X6 is the frame alone, at "
                "lambda_sig_record=0.0."
            )
        # Built from the RAW z in every frame -- record_canonical_templates demodulates internally,
        # so handing it the canonical frame's already-demodulated u would rotate twice.
        record_templates, record_has_valid = record_canonical_templates(z, theta, operator)
        # The same "declared, structured dilution" sanity number the demodulating frames report --
        # meaningful now that theta is actually read, and identical to whatever the canonical frame
        # above may already have written (one expression, one quantity).
        theta_valid_frac_val = float(torch.isfinite(theta).float().mean())
        record_input = record_templates[record_has_valid]  # (N_valid, K): the T = 1 case
        if record_input.shape[0] == 0:
            # Same zero-record safety as the record_canonical frame above: the empty selection's own
            # sum is exactly 0.0 and still attached to the graph, so backward is a well-defined
            # no-op instead of the NaN an empty-N statistic would produce, and gen_sigreg_record is
            # deliberately not advanced (how much RNG a call consumes is the regularizer's own
            # property, not something train_step may hardcode).
            sigreg_record_loss = record_input.sum()
        else:
            sigreg_record_loss = model.regularizer(record_input, generator=gen_sigreg_record)
        total_loss = total_loss + lambda_sig_record * sigreg_record_loss
        sigreg_record_loss_val = float(sigreg_record_loss.detach())
        sigreg_n_records_val = float(int(record_has_valid.sum()))

    pred_loss_val = float("nan")
    persistence_loss_val = float("nan")
    if lambda_pred != 0.0:
        predictor_mask = ~plan.context  # every position >= cutoff, not only the target
        z_hat = model.predictor.forward(z, predictor_mask)
        pred_loss = model.prediction_loss(z_hat, z, plan.target)

        # Persistence baseline (architecture-primer.html §0): z_{t+1} = z_t, repeating each
        # record's own context-cutoff latent at every position and scoring it with the SAME
        # masked loss and target mask as pred_loss -- directly comparable to it, "logged from the
        # first run rather than reconstructed later." No gradient: this never contributes to
        # total_loss, so tracking it would only cost memory. Same formula as
        # winder.eval.forecast.persistence_predict, inlined rather than imported -- train.py is
        # the core training path and must not depend on winder.eval (an evaluation-side module),
        # and that harness is itself scheduled to retire (architecture-primer.html §5-6) once the
        # encoder swap lands.
        with torch.no_grad():
            idx = torch.arange(b, device=z.device)
            persistence_pred = z[idx, plan.cutoff.to(z.device), :].unsqueeze(1).expand_as(z)
            persistence_loss = model.prediction_loss(persistence_pred, z, plan.target)

        total_loss = lambda_pred * pred_loss + total_loss
        pred_loss_val = float(pred_loss.detach())
        persistence_loss_val = float(persistence_loss)

    trans_loss_val = float("nan")
    trans_floor_val = float("nan")
    trans_gain_val = float("nan")
    trans_directional_val = float("nan")
    trans_radial_val = float("nan")
    closure_residual_val = float("nan")
    if lambda_trans != 0.0:
        if theta is None or operator is None:
            raise ValueError(
                "lambda_trans != 0.0 requires both theta and operator to be provided "
                f"(got theta={theta!r}, operator={operator!r})"
            )
        trans_out = transport_loss(
            z,
            theta,
            operator,
            radial_weight=transport_radial_weight,
            stop_gradient_target=transport_stop_gradient,
        )
        total_loss = total_loss + lambda_trans * trans_out.loss
        trans_loss_val = float(trans_out.loss.detach())
        trans_floor_val = float(trans_out.floor.detach())
        trans_gain_val = trans_floor_val - trans_loss_val
        trans_directional_val = float(trans_out.directional_term)
        # NaN at transport_radial_weight == 0.0 (transport_loss's own structural-skip sentinel).
        trans_radial_val = float(trans_out.radial_term)
        closure_residual_val = float(operator.closure_residual().detach())

    metrics = StepMetrics(
        step=-1,  # filled in by fit()
        lr=float("nan"),  # filled in by fit()
        pred_loss=pred_loss_val,
        persistence_loss=persistence_loss_val,
        sigreg_loss=float(sigreg_loss.detach()),
        total_loss=float(total_loss.detach()),
        n_context=int(plan.context.sum()),
        n_target=int(plan.target.sum()),
        cutoff_mean=float(plan.cutoff.float().mean()),
        grad_norm=float("nan"),  # filled in by fit()
        trans_loss=trans_loss_val,
        trans_floor=trans_floor_val,
        trans_gain=trans_gain_val,
        trans_directional=trans_directional_val,
        trans_radial=trans_radial_val,
        closure_residual=closure_residual_val,
        theta_valid_frac=theta_valid_frac_val,
        sigreg_n_records=sigreg_n_records_val,
        sigreg_record_loss=sigreg_record_loss_val,
    )
    return total_loss, metrics


def fit(
    model: JepaModel,
    batches: Iterable[torch.Tensor],
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    *,
    on_step: Callable[[StepMetrics], None] | None = None,
    start_step: int = 0,
    gen_mask: torch.Generator | None = None,
    gen_sigreg: torch.Generator | None = None,
    gen_sigreg_record: torch.Generator | None = None,
    gen_augment: torch.Generator | None = None,
    theta_batches: Iterable[torch.Tensor] | None = None,
    operator: HarmonicTransport | None = None,
) -> list[StepMetrics]:
    """Drives `train_step` over `batches` (each a `(B, n_leads, n_samples)` waveform tensor),
    applying the explicit warmup+cosine schedule, gradient clipping, and one optimizer step per
    batch. `"mask"` and `"sigreg"` generator streams advance across steps -- resampling every
    call is load-bearing for both (see their own docstrings), so they are not reset per step.

    By default (`gen_mask`/`gen_sigreg` omitted) both streams are freshly seeded from
    `cfg.seed_pretrain`, exactly as before this function accepted these two arguments -- every
    pre-existing caller that never resumes is unaffected. A caller resuming from a CKPT-01
    checkpoint (`winder.jepa.checkpoint`) instead builds its own `torch.Generator`, restores its
    saved state via `set_state`, and passes it in here so training continues those streams
    mid-sequence rather than replaying them from seed.

    `cfg.n_steps` is the schedule's total length (`lr_schedule` reads it for the cosine decay's
    denominator), NOT "how many steps this call runs" -- `start_step` (default 0, the position
    this call's first yielded batch is treated as) lets a resumed call continue the SAME schedule
    from where a prior, now-checkpointed call left off. How many steps this call actually runs is
    controlled by `batches`' own length: `zip(..., strict=False)` stops as soon as either
    `range(start_step, cfg.n_steps)` or `batches` is exhausted, whichever comes first -- a
    resumed call typically hands `fit` a `batches` iterator with exactly
    `cfg.n_steps - start_step` items, not a truncated `cfg`.

    `theta_batches`/`operator` are optional and additive: omitted (the pre-existing default),
    every step calls `train_step` exactly as it did before either existed, with `cfg.lambda_trans`
    at its own default 0.0 -- structurally skipping the transport path (module docstring). When
    provided, `theta_batches` must yield one `(B, n_tokens)` theta tensor per `batches` item, in
    lockstep -- the same `zip(..., strict=False)` early-stop contract as `batches` vs. `range(...)`
    above applies here too, so a `theta_batches` shorter than `batches` silently truncates the run
    rather than raising, matching that existing contract's own shape.

    `operator`'s gradient is clipped SEPARATELY from `model.parameters()` (its own
    `clip_grad_norm_` call, same `cfg.grad_clip_norm` bound) because it is deliberately not a
    submodule of `model` (`winder.jepa.model`'s own docstring: the operator/JEPA decoupling is
    preserved by construction) -- a single joint clip would silently let a large encoder gradient
    rescale the operator's own effective step, which is not the intended trade-off (see
    `winder.operators.free.FreeOperatorConfig`'s docstring on the SEPARATE optimizer param group
    a caller building `optimizer` should give `operator.parameters()`, at `weight_decay=0.0`).

    `cfg.sigreg_frame` is read every step and passed straight through to `train_step` -- module
    docstring's `sigreg_frame` paragraph. At its own default `"raw"`, every pre-existing caller is
    bitwise unaffected. `"canonical"` and `"record_canonical"` both need `theta_batches`/`operator`
    supplied here for the same reason `lambda_trans != 0.0` does (`train_step`'s own guard raises
    otherwise), independent of `cfg.lambda_trans`'s value -- a caller may run a demodulated-frame
    SIGReg with the transport term itself weighted to 0.0, though this campaign never exercises
    that combination.

    `cfg.lambda_sig_record` is likewise read every step and passed straight through (X7's record
    term -- module docstring). At its own 0.0 default the branch is structurally skipped and every
    pre-existing caller is bitwise unaffected, including the `gen_sigreg_record` stream built below,
    which is then never drawn from. Nonzero, it needs `theta_batches`/`operator` here for the same
    reason the demodulating frames do. A resuming caller must restore `gen_sigreg_record` from its
    checkpoint exactly as it restores `gen_mask`/`gen_sigreg`: the record term consumes that stream
    on every active step, so replaying it from seed would repeat direction draws an uninterrupted
    run had already spent.

    `cfg.augment` (V5 -- module docstring's augment paragraphs) is parsed ONCE, before the loop;
    when non-empty, every waveform batch passes through `augment_waveform` (drawing only from
    `gen_augment`, `"augment"` by the default construction below) before `train_step` sees it --
    theta batches are never touched. At its own `""` default the call site is structurally
    skipped and `gen_augment`'s state is never advanced, so every pre-existing caller is bitwise
    unaffected. A resuming caller must restore `gen_augment` from its checkpoint for the same
    reason as `gen_sigreg_record` above: when active, the stream is consumed on every step.
    """
    gen_mask = gen_mask if gen_mask is not None else generator(cfg.seed_pretrain, "mask")
    gen_sigreg = gen_sigreg if gen_sigreg is not None else generator(cfg.seed_pretrain, "sigreg")
    gen_sigreg_record = (
        gen_sigreg_record
        if gen_sigreg_record is not None
        else generator(cfg.seed_pretrain, "sigreg_record")
    )
    gen_augment = (
        gen_augment if gen_augment is not None else generator(cfg.seed_pretrain, "augment")
    )
    augmentations = parse_augment_spec(cfg.augment)
    theta_iter: Iterable[torch.Tensor | None] = (
        theta_batches if theta_batches is not None else itertools.repeat(None)
    )

    history: list[StepMetrics] = []
    for step, waveform, theta in zip(
        range(start_step, cfg.n_steps), batches, theta_iter, strict=False
    ):
        lr = lr_schedule(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if augmentations:
            # V5's theta-safe stack, waveform only -- theta passes through below untouched
            # (module docstring's augment paragraphs; structural skip at cfg.augment == "").
            waveform = augment_waveform(
                waveform, augmentations, prob=cfg.augment_prob, generator=gen_augment
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics = train_step(
            model,
            waveform,
            lambda_sig=cfg.lambda_sig,
            gen_mask=gen_mask,
            gen_sigreg=gen_sigreg,
            gen_sigreg_record=gen_sigreg_record,
            theta=theta,
            operator=operator,
            lambda_trans=cfg.lambda_trans,
            lambda_pred=cfg.lambda_pred,
            lambda_sig_record=cfg.lambda_sig_record,
            transport_radial_weight=cfg.transport_radial_weight,
            transport_stop_gradient=cfg.transport_stop_gradient,
            sigreg_frame=cfg.sigreg_frame,
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        if operator is not None:
            operator_params = list(operator.parameters())
            if operator_params:  # the cyclic arm has none (omega is a frozen buffer, not a
                # Parameter) -- clip_grad_norm_ on an empty list is a harmless no-op but warns
                torch.nn.utils.clip_grad_norm_(operator_params, cfg.grad_clip_norm)
        optimizer.step()

        metrics.step = step
        metrics.lr = lr
        metrics.grad_norm = float(grad_norm)
        history.append(metrics)
        if on_step is not None and step % cfg.log_every == 0:
            on_step(metrics)
    return history
