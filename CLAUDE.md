# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDM-UniPS is a **CVPR 2023 Highlight** paper implementation for **Universal Photometric Stereo** — recovering surface normal maps from multiple images captured under arbitrary, spatially-varying lighting with a fixed camera. The upstream repository is inference-only; this fork adds a training pipeline (`sdm_unips/train.py`, `modules/builder/trainer.py`, `modules/io/dataloader/`, `modules/loss/`) with train/val/test all drawn from the same synthetic mixed pool.

**Scope:** this fork targets **surface-normal prediction only**. The upstream BRDF heads (baseColor / roughness / metallic from Appendix C of the paper) and novel-view relighting (`relighting.py`, `modules/utils/render.py`) have been removed.

The current branch (`architecture/training-pipeline`) targets **Model A** (the baseline SDM-UniPS architecture, no modifications), trained on a mix of `hdlong-complexv1` and `PolarPS` for the author's thesis: *"Optimizing Universal Photometric Stereo via Wavelet-Energy Saliency and Latent Carrier Tokens."* Models B and C (WTConv, WESS, latent-carrier deformable attention) will live on dedicated branches. A side experiment continues Model A on **MerlMix** (see **Fine-tuning from a pretrained model**) to test whether more data improves DiLiGenT MAE.

## Running Inference

**Normal map recovery:**
```bash
python sdm_unips/main.py \
  --session_name SESSION_NAME \
  --test_dir YOUR_DATA_PATH \
  --checkpoint CHECKPOINT_PATH
```

**Key inference flags:**
- `--scalable`: Enable memory-efficient mode for high-resolution images (512×512 patch decomposition; trades some accuracy for lower VRAM)
- `--max_image_res`: Max input resolution (default 4096)
- `--max_image_num`: Max images to load per object (default 10)
- `--canonical_resolution`: Internal encoding resolution (default 256)
- `--pixel_samples`: Pixels sampled per batch during inference (default 10000)

The checkpoint directory must contain `<checkpoint>/normal/*.pytmodel`.

## Running Training

**Model A (thesis recipe — hdlong-complexv1 + PolarPS mix, held-out val + test splits):**
```bash
python sdm_unips/train.py \
  --session_name modelA_full \
  --hdlong_dir /path/to/hdlong-complexv1 \
  --polarps_dir /path/to/PolarPS \
  --max_scenes 8000 --val_fraction 0.1 --test_fraction 0.1 \
  --train_resolution 512 --canonical_resolution 256 \
  --batch_size 8 --pixel_samples 2048 \
  --epochs 60 --lr 1e-4 --weight_decay 0.05 \
  --lr_schedule wsd --decay_fraction 0.2 \
  --warmup_epochs 5 --min_lr_ratio 0.01 \
  --grad_clip 1.0 --amp_dtype bf16
```
(`--lr_schedule` defaults to `cosine`; pass `wsd` explicitly — see **LR
schedule** below for why.)

**Smoke test on Kaggle (10 epochs, optionally shrink dataset):**
```bash
python sdm_unips/train.py \
  --session_name modelA_smoke --smoke_test --smoke_epochs 10 \
  --hdlong_dir /kaggle/input/hdlong-complexv1 \
  --polarps_dir /kaggle/input/polarps \
  --max_scenes 8000 --amp_dtype bf16
```

**Model selection & evaluation protocol:** the mixed pool is split at the
**scene level** into train / val / test (`--val_fraction` and
`--test_fraction`, both default 0.1), split proportionally within each source
(hdlong and PolarPS split separately, so the mix ratio is preserved in all
three splits) and deterministic given `--seed`. A source present in the pool
must supply at least one scene per split it is asked to fill (3 normally, 2
with `--test_fraction 0`) or training aborts, as does a requested split that
comes out empty. `best.pt` is selected by **held-out validation loss** at the
end of every `--val_every_epochs` epochs. The **held-out test split** is
evaluated **exactly once, after the final epoch** via the same `val_step` path,
on the weights loaded from `best.pt` (not the last-epoch weights, so late
overfitting or early stopping never biases the headline number), and it never
influences checkpoint selection.

**`--test_fraction 0` switches the test split off**, which is what lets a side
experiment skip it without needing a mode of its own: validation still selects
`best.pt` while everything else trains, and the verdict comes from an external
benchmark (`sdm_unips/eval_diligent.py`, run against `<session>/checkpoints`
after the run exports them). **`--val_fraction` has no such switch and must be
> 0** — validation is what chooses which epoch to keep, so a run without it
produces no deliverable.

Test and validation report the **same quantities aggregated the same way** —
both sweep `train.py:run_eval_pass`, which takes a scene-count-weighted mean
over batches (so a short final batch is not over-weighted, and the metric does
not depend on `--batch_size`) and drops any non-finite value rather than
letting one `±inf` batch poison the average and silently freeze `best.pt`
selection. The two numbers are therefore directly comparable; they differ only
in which scenes they cover and in the number of trials per scene.

**A/B fairness contract (Models A vs B/C).** The thesis varies the *pixel
sampler*, so the evaluation is built to be invariant to it. Two guarantees,
each in one place:

1. **Fixed evaluation pixels.** `Trainer._eval_sample_ids` draws the val/test
   pixel set **outside the network** — uniformly over the mask, seeded per
   *item* (`eval_seed + item_index`, CPU generator), so the draw for a given
   scene is independent of `--batch_size`, of the worker count, and of how much
   RNG training consumed first. `Net.forward` accepts these as `sample_ids` and
   then never calls its own sampler. `Trainer.begin_eval_pass()` resets the
   item counter at the head of each sweep. Without this, a saliency sampler
   would change *which pixels the metric is computed on*, and a "better" val
   curve could be nothing but an easier pixel draw.
2. **Fixed evaluation renders.** Both held-out splits are wrapped by
   `MixedEvalDataset`, whose per-index seeding pins each scene's camera, K
   lights and Dirichlet mix. This matters because `MixedTrainDataset` passes
   `rng=None` to the scene loaders, which falls back to the global
   `np.random`: `augment=False` disables only the horizontal flip, so val
   scenes would otherwise be **re-rendered every epoch**. Val uses
   `n_trials=1` (one fixed render per scene, identical every epoch and every
   run); test uses `--test_trials` (each trial a *different* but fixed draw,
   averaged for variance reduction — reproducible across runs).

Net effect: two runs that differ only in the model are evaluated on byte-identical
data. `Net.sample_train_pixels` is the single override point for a new sampler
(training only); nothing else needs to change to keep the comparison fair.

**LR schedule — why `wsd` and not `cosine`.** A cosine schedule's LR at step
`s` is a function of `s / total_steps`, so it must know the endpoint in
advance, and a cycle length that does not match the actual run costs final
loss (Chinchilla, Hoffmann et al. 2022, App. B). That makes "train a while,
measure throughput, then decide the length" *lossy*: resuming with a different
`--epochs` rebuilds the lambda from the new value while the scheduler's step
count comes from the old run, so every step already taken is retroactively
re-priced (at step 46,359 of the thesis config, shortening 40 → 16 epochs
moves that step's LR by ×0.73) and the resulting curve has a kink nobody can
reproduce without replaying the exact kill point.

`--lr_schedule wsd` (warmup → **stable** → decay) removes the coupling: the LR
does not depend on the endpoint until the decay begins, so the run is
genuinely extendable and the decision of when to stop can be made *from the
val curve* rather than committed before the first step.

- `--decay_fraction` (0.2) is the **length** of the decay ramp, as a fraction
  of the planned run. It is not a stopping rule.
- `--decay_from_step` is where the ramp **begins**; 0 derives it so the ramp
  lands exactly at the end of `--epochs`. Left alone, the run is therefore
  fully hands-off and finishes annealed exactly like a cosine run — watching
  the curve is *optional*, not required.
- `--decay_now` (resume only) begins the ramp at the step being resumed from
  and truncates `--epochs` so the run ends when the ramp does (rounded up to
  an epoch boundary, since the loop is epoch-driven). This is the one-flag
  form of "val has plateaued, anneal and finish", and it is **continuous in
  LR** — the stable phase sits at exactly `1.0 × --lr`, which is where the
  cosine ramp starts, so nothing jumps and no step already taken changes.

- `--auto_decay` (**off by default**, wsd only) fires the same branch
  **in process**, with no cancel-and-relaunch, once held-out val loss stops
  improving. It is the automated form of watching the curve by hand, and
  Model A was trained without it — leaving it off keeps B/C on a byte-identical
  code path.

`--decay_fraction` and `--decay_from_step` are in `SCHEDULE_SENSITIVE_ARGS`,
not `SPLIT_CRITICAL_ARGS`: changing them at resume is the intended use, so
they warn rather than abort.

**The auto-decay rule.** A check is *flat* when the relative improvement over
the **running best**, `(best − cur) / |best|`, is below `--auto_decay_rel`
(0.01); the ramp branches after `--auto_decay_patience` (2) consecutive flat
checks, but never before `--auto_decay_min_epoch` (8) completed epochs — so
`[VAL EPOCH 7]` is the earliest check that can trigger. The floor absorbs an
early noise-flat pair; the two-check requirement absorbs a single bad render.

The comparison is against the running best and **not** against the previous
check, because a sawtooth defeats the latter: a curve alternating 0.0300 /
0.0285 shows a *+5% improvement* on every other check, resetting the counter
forever while the model plateaus. Against the best, both halves of the
oscillation read as flat and the rule fires. Model A's observed history
(0.1072 → 0.0236, then 0.0255 at epoch 8) does **not** fire under this rule —
one flat check, not two — which matches the decision that was taken manually.

Firing truncates `--epochs` so the run ends with the ramp. Note that the epoch
loop's `range()` is fixed at entry, so the truncation is enforced by a separate
break; if the ramp needs more epochs than the launch allowed, a warning is
printed and `--resume auto` finishes it (the branch is in the checkpoint — see
below).

**Branched ramps are persisted.** A ramp moved by `--decay_now` or
`--auto_decay` exists only in the scheduler closure —
`LambdaLR.state_dict()` stores `last_epoch`/`_last_lr` and `None` for a lambda
that is a plain function — so it cannot be rebuilt from `args`, and
`--decay_from_step` alone is not enough (`decay_len` derives from `--epochs`,
which the branch truncates). Checkpoints therefore carry
`sched_state = {decay_start, decay_end, branched}`, and a resume restores those
endpoints **authoritatively**, ignoring whatever `--epochs` and
`--decay_fraction` the relaunch passes. Otherwise a mid-ramp resume would put
the LR back to the full `--lr` and `--decay_now` would re-branch at the new
step, so a machine that crashes during the decay could never finish it.
Re-passing `--decay_now` against a branched checkpoint is an announced no-op.

A **derived** ramp (`--decay_from_step 0`, never branched) is deliberately
*not* persisted: it is a pure function of the args, rebuilds identically, and
pinning it would destroy the extendability WSD exists for.

Note that WSD is a more aggressive schedule in aggregate — it holds the peak
LR for ~80% of the run where cosine averages about half of it. Watch
`grad_norm` and `avg_grad_skipped` over the few hundred steps after warmup
ends.

**A/B use.** Whatever decay point Model A ends up with, pin it for B and C
with an explicit `--decay_from_step` + the same `--epochs`. Choosing it from
A's val curve is legitimate — it is one shared hyperparameter chosen once, not
a per-model tune — but all three variants must run the identical schedule for
the comparison to mean anything. `--auto_decay` is therefore a *discovery*
tool, not a per-run setting: let it fire once, read `decay_start` out of the
`{"kind": "decay_trigger", ...}` record, then pin every variant with an
explicit `--decay_from_step` and matching `--epochs`. Leaving it enabled on
B and C would let each variant pick its own schedule and confound the result.

**Early stopping (safety cutoff):** `--patience` (default 10, in units of
validation checks; 0 disables) stops training once val loss has not improved
by more than `--min_delta` (default 0.0) for that many consecutive checks. The
LR schedule still spans the full `--epochs`; patience only trims the
unproductive tail, and `best.pt` already holds the best-val weights, so
stopping early never costs the deliverable.

**Non-finite gradients (skip + abort).** A non-finite `grad_norm` **skips the
optimizer step** rather than writing NaN into the weights — the same policy
`GradScaler` applies for fp16, extended to bf16/none where nothing else
provides it. Clipping cannot substitute: `clip_grad_norm_` scales by
`total_norm`, and `NaN/NaN` is `NaN`. The scheduler and `global_step` still
advance on a skipped step, so the LR schedule stays aligned with
`--epochs`; gradients are cleared by the next accumulation cycle's
`zero_grad`.

Because a model that never steps never learns — and its loss curve is
indistinguishable from one that has plateaued — **two independent guards**
abort the run with `NonFiniteGradientAbort`. Neither subsumes the other:

- `--max_consecutive_skips` (default 20, 0 disables) — an unbroken run of
  skips. Trips within seconds of a hard breakage. The default is generous
  because fp16 legitimately skips a handful of steps at startup while
  `GradScaler` calibrates its scale down from 65536.
- `--max_skip_rate` (default 0.3, 0 disables) over `--skip_rate_window`
  (default 200 optimizer steps) — the fraction of *recent* steps skipped,
  compared strictly (`>`), so exactly 30% does not fire. This catches what a
  consecutive counter is blind to: an intermittent NaN that halves the
  effective run without ever skipping twice in a row. The window is rolling,
  not cumulative, so a bad patch that genuinely recovers ages out, and a
  mid-run degradation is caught rather than diluted by earlier healthy steps.
  It stays **disarmed until the window is full**, which doubles as its grace
  period — otherwise one skip in the first two steps would read as 50%.

On a hard breakage the consecutive guard fires first by construction (20 steps
vs a 200-step window). The abort is written to `train.jsonl` as
`{"kind": "abort", ...}` before it propagates. Weights are never corrupted by a
skipped step, so the last checkpoint is always sound.

Every skip also prints a `[grad-skip]` line carrying the consecutive count and
the rolling rate, and each epoch summary carries **`avg_grad_skipped`** (that
epoch's skip rate, 0.0 = healthy, 1.0 = nothing learned) plus cumulative
`skipped_steps`. Check `avg_grad_skipped` before trusting a loss curve: a run
sitting just under `--max_skip_rate` never aborts but is still training at a
fraction of its nominal rate, with the LR schedule advancing regardless.

**Diagnosing where a NaN came from.** `Trainer._report_bad_grads` fires once,
on the first non-finite gradient, *before* the optimizer step. Backward runs
loss → input, so a NaN contaminates every parameter **upstream** of its origin
and leaves everything downstream clean; `named_parameters()` is registration
(forward) order, so the contaminated region **ends** at the origin. The report
prints that end, the first clean gradients past it, and a per-submodule
`bad/total` breakdown. `--detect_anomaly` additionally installs per-parameter
backward hooks that name the first non-finite gradient in true backward order
(no ordering assumption), alongside the existing forward hooks and
`torch.autograd.detect_anomaly`. Note that a **post-hoc weight** scan cannot
localize anything: one optimizer step after the origin, every parameter is
NaN.

**Reproducibility:** an *uninterrupted* run is reproducible for a fixed config
— all RNGs are
seeded from `--seed` (python/numpy/torch/cuda), cuDNN is pinned to
deterministic kernels, and the DataLoaders seed their workers' numpy RNG
(`worker_init_fn`) with a seeded shuffle `generator`.

A **resumed** run is not bit-identical to an uninterrupted one, by design
decision (verified 2026-08-07). `--resume` restores the RNG streams and the
shuffle `generator`, so the model-side draws continue exactly, but
`persistent_workers=True` seeds each worker's numpy RNG **once**, at the first
epoch, from a `base_seed` drawn off the shuffle generator; those RNGs then
evolve inside the worker processes across epochs. A resumed run spawns fresh
workers mid-schedule and draws a new `base_seed`, so its shuffle permutation
and per-scene random renders (Dirichlet mix, camera, K-image draw) diverge from
what the uninterrupted run would have seen — same distribution, different
sample. `persistent_workers=False` would close the gap (every epoch would then
consume the generator identically) at the cost of respawning workers each
epoch; deliberately not done, because it changes no reported number: val and
test go through `MixedEvalDataset` + `Trainer._eval_sample_ids`, both seeded
per *item* and independent of all of this, so the A/B contract and the headline
test figure are untouched.

The test evaluation is
strongly reproducible: `MixedEvalDataset` seeds each scene's random
camera/lights from its flat index (independent of worker count), and the
final eval re-seeds torch so the model's pixel sampling is fixed too. To cut
the variance of the random K-image draw, each test scene is rendered
`--test_trials` times (default 3), each trial a different fixed K-image draw,
and their MAE is averaged with a scene-count-weighted (unbiased) mean.

**Malformed scenes (validated before the split).** Discovery admits a
directory on a single marker file (`normal.exr` / `light_means.config`), which
is far less than the loaders need — a directory carrying only the marker enters
the pool, is assigned to a split, and raises the first time the sampler reaches
it, potentially days in. `modules/io/dataloader/scene_check.py` therefore
validates the whole pool at startup, **before the `--max_scenes` cap and before
the split**, so both operate on a fully readable pool and the val/test
fractions and the hdlong:PolarPS ratio stay exact.

A scene is valid iff it can supply one sample of K (`--k_per_scene`) images:

- **PolarPS** — `normal.exr`, plus ≥K `light-*` dirs holding an `S0.exr` under
  `sorted(subdirs)[0]`. That first material-mix directory specifically, because
  it is the only one `PolarPSLoader.load` ever reads.
- **hdlong** — `light_means.config`, plus ≥1 `cam_*` dir with
  `binary_mask.exr`, `local_normal.exr` and `min(#point, #dir, #env) ≥ K`.

Checks are stat-only (no EXR decode — decoding one image per scene across ~18k
scenes would take hours) and run on a thread pool, since each lookup on a
network mount is a round trip. `ThreadPoolExecutor.map` preserves input order,
which the split depends on.

**The validator and the loaders share their definition of "usable"**, via
`usable_cam_dirs` / `usable_light_dirs` in `scene_check.py`. This is not
tidiness: the rule is "≥1 usable camera", so a scene with one good camera out
of three passes — and would still fail two reads in three if `HdlongLoader`
kept drawing from all `cam_*`. The loaders draw only from what those helpers
return, so "passed validation" and "every read succeeds" cannot drift apart.

**Corrupt file contents** cannot be caught by a stat check, so
`MixedTrainDataset._get_with_fallback` also steps forward to the next scene on
any read failure, logging once per scene per worker. The substitution is a pure
function of the scene list and the failing index — no RNG, no model state — so
an A/B pair still sees identical data. The failing scene is **not** removed
from the pool: the split is a permutation over scene *positions*, so dropping
one mid-run would reshuffle everything. More than 8 consecutive failures raises
instead (that is an unmounted dataset, not a few bad files). Per-worker counts
are not aggregated into the epoch summary — workers are separate processes and
that would require widening the batch tuple — so grep `[SceneSkip]` in the log.

**Pool identity across runs.** The split is a permutation over pool *positions*,
so a pool that gains or loses even one scene relands every surviving scene:
empirically, removing 1 scene of 20 moved 4 of the other 19, and ~75% of a
freshly-permuted test split turns out to be old-train scenes. Three mechanisms
keep this from silently corrupting an A/B comparison:

- `<session>/logs/scene_manifest.json` — the validated pool, plus every
  rejected scene with its reason (the record of what was excluded).
- `args.scene_pool_fingerprint` (sha1 of the ordered valid list) in
  `config.json` and in every checkpoint, and **split-critical** on `--resume`:
  a mismatch aborts, exactly as a changed `--seed` does.
- `--scene_manifest PATH` — reuse a pinned pool. Point Models B and C at Model
  A's manifest so all three provably train on the same split regardless of
  filesystem drift; missing scenes in a reused manifest are a hard error, never
  a silent drop. A **resumed run reuses its own session's manifest
  automatically**, so a repaired scene between launches cannot change the split
  underneath it (and startup skips the scan).

Both the manifest and the fingerprint describe scenes **relative to their
dataset root** (`scene_check.relative_to_roots`, posix separators), never by
absolute path. Absolute paths would make both machine-specific: training Model
B on a second box that mounts the datasets elsewhere would produce a different
fingerprint for the *same* pool — the resume/A-B guard flagging a difference
that does not exist — and Model A's manifest could not be reused there at all.
With relative paths a fresh scan at a different mount point yields the
identical fingerprint (verified), and `read_manifest` rebases a manifest onto
the current roots, so differing roots are a rebase rather than an error. A
uniform root change also leaves discovery order untouched (all paths share the
prefix), so the split itself is unchanged — which is what makes the fingerprint
match meaningful rather than coincidental.

**Several roots per source.** `--hdlong_dir` and `--polarps_dir` are
`nargs='+'`: each takes one or more roots, swept in the order given and pooled
into a **single source**. MerlMix has hdlong's exact directory layout and
marker file (`light_means.config`), so it is added as an extra `--hdlong_dir`
root rather than as a new dataset kind — no new loader, no new `kind`. The
consequence to keep in mind is that `_proportional_cap` and the per-source
split key on *kind*, so `--max_scenes` balances hdlong-total against PolarPS,
not hdlong-complexv1 against MerlMix.

Scene paths are de-duplicated across roots (a root nested inside another would
otherwise land two copies of a scene in different splits), and each root's
discovered and post-validation counts are printed separately — a combined total
would let a healthy root hide a mistyped or half-mounted one.

Each root gets its **own manifest key**: `hdlong_dir`, `hdlong_dir#1`,
`hdlong_dir#2`, … Without that, `hdlong/SCENE_0001` and `merlmix/SCENE_0001`
would collapse to the same `(kind, root_key, relpath)` entry — a fingerprint
collision, and a `read_manifest` that rebases both onto whichever root came
first, i.e. training on one dataset twice and never touching the other. The
fingerprint folds in only the key's `#n` **suffix**, so the *first* root of each
flag hashes exactly as it did when the flag was single-valued: a single-root
pool keeps its old fingerprint, and Model A's checkpoints (where the
fingerprint is split-critical) still resume and its manifest is still reusable.
`scene_check.build_roots` owns the keying; `sdm_unips/tests/test_scene_check_roots.py`
pins both the collision case and the single-root invariant (stdlib-only, so it
runs without a training environment).

Root **order** is part of the pool identity, since the split is a permutation
over pool *positions*. Passing the same roots in a different order is a
different split; the fingerprint records which order was used, and `--resume`
aborts on a change.

**Key training flags** (defaults shown are thesis values):
- `--hdlong_dir`, `--polarps_dir`: roots for the two mixed sources (either or
  both). Each takes **one or more** roots (`nargs='+'`) — see **Several roots
  per source** below
- `--train_dir`: auto-detect root (scenes classified into hdlong/polarps by on-disk markers)
- `--max_scenes`: cap on the **combined** scene pool before the split (thesis uses 8000)
- `--k_per_scene`: 10 — images drawn per scene, and the minimum a scene must
  supply to enter the pool. Split-critical: raising it shrinks the pool
- `--scene_manifest`: reuse a pinned scene list instead of rescanning
- `--strict_scenes`: abort if any scene fails validation (default: skip it)
- `--max_bad_scene_frac`: 0.05 — abort if more than this fraction fails
  validation (0 disables); catches a half-mounted dataset or a wrong `--k_per_scene`
- `--val_fraction`: held-out fraction for validation (default 0.1; **0 = no
  val split**, which also means no `best.pt`)
- `--test_fraction`: held-out fraction for the final test report (default 0.1;
  **0 = no test split**, for a run judged by an external benchmark)
- `--test_trials`: deterministic K-image draws per test scene, averaged (default 3; keep small)
- `--val_every_epochs`: epochs between validation passes / best.pt checks (default 1)
- `--patience`: early-stopping patience in validation checks (default 10; 0 disables)
- `--min_delta`: min val-loss decrease counted as improvement (default 0.0)
- `--batch_size`: 8 (**micro**-batch) — `--pixel_samples`: m=2048
- `--accum_steps`: 1 — micro-batches per optimizer step; effective batch is
  `--batch_size × --accum_steps`. Use it to keep the recipe's effective batch
  of 8 on a GPU too small to hold it in one pass (`--batch_size 2
  --accum_steps 4`). `global_step`, the LR schedule, `--log_every` and
  `--ckpt_every` all count **optimizer** steps, so the schedule is unchanged.
- `--num_workers`: 4 — `--prefetch_factor`: 2 (batches prefetched per worker;
  in-flight host memory ≈ `num_workers × prefetch_factor × batch_size × 36 MB`
  at K=10/512², passed through `/dev/shm`)
- `--max_vram_gib`: 0 (no cap) — hard per-process VRAM ceiling; over-large
  batches raise OOM instead of consuming the whole card. Worth setting when
  the GPU also drives a display.
- `--train_resolution`: 512 — `--canonical_resolution`: 256
- K is `--k_per_scene` (default 10), enforced in `HdlongLoader` / `PolarPSLoader` and as the pool's validity threshold.
- `--lr`: 1e-4 — `--weight_decay`: 0.05
- `--lr_schedule`: default `cosine`; `wsd` is recommended and `step` exists —
  see **LR schedule** below
- `--decay_fraction`: 0.2 — `--decay_from_step`: 0 — `--decay_now` (wsd only)
- `--auto_decay` (off) — `--auto_decay_rel`: 0.01 — `--auto_decay_patience`: 2
  — `--auto_decay_min_epoch`: 8. In-process decay trigger; see **LR schedule**
- `--warmup_epochs`: 5.0 (linear)
- `--amp_dtype`: `bf16` (recommended on H100), `fp16` (legacy), or `none`
- `--grad_clip`: 1.0
- `--max_consecutive_skips`: 20 — `--max_skip_rate`: 0.3 over
  `--skip_rate_window`: 200 (either 0 disables that guard) — see below
- `--keep_last`: 3 (epoch checkpoints to retain; `best.pt` is kept separately)
  — `--ckpt_every`: 1000 optimizer steps, `--keep_last_steps`: 2 (mid-epoch
  crash insurance; see **Checkpoints** below)
- `--resume`: continue an interrupted run (`auto`, a checkpoint dir, or a
  file) — see **Resuming** below. `--allow_config_change` downgrades the
  split-config guard to a warning
- `--pretrained`: start a **new** run from an existing checkpoint's weights —
  see **Fine-tuning** below
- `--smoke_test` + `--smoke_epochs`: short dry-run for Kaggle

The final test evaluation runs automatically after the last epoch on the
held-out `--test_fraction` split, averaged over `--test_trials` deterministic
trials — unless `--test_fraction 0`, in which case the run ends after the
export and the verdict comes from an external benchmark.

Checkpoints are written to `<session>/checkpoints/`. The most recent `--keep_last` `epoch_*.pt` files plus `best.pt` are retained; older epoch checkpoints are pruned (`final.pt` is never auto-pruned), **ordered by parsed epoch number** — string-sorting the filenames puts `epoch_10.pt` before `epoch_7.pt`, which would delete the newest checkpoint the moment it was written.

**One inference file, not three.** `builder.load_models` needs a bare `state_dict` in a directory holding exactly one `*.pytmodel`; a `.pt` training checkpoint will not do, since its top-level keys are `model`/`optimizer`/… rather than parameter names. So `*.pytmodel` is the **interchange format**, not a legacy one — but only one copy of it is needed. `Trainer.export_weights` serializes the model **in memory** straight to `<session>/checkpoints/normal/normal.pytmodel`, called at each `best.pt` update and once after the final epoch. Consequences:

- The filename is fixed and written atomically, so "exactly one `*.pytmodel` in that directory" holds by construction — no stale sibling to clean up, and no second copy that could disagree with the first.
- A run killed mid-schedule leaves weights that are **immediately usable** (`--checkpoint <session>/checkpoints`), instead of files needing to be moved by hand.
- Per-save weight writes are gone. Previously every epoch checkpoint *and* every `--ckpt_every` step checkpoint rewrote a full `normal.pytmodel` that nothing read until the run ended.

**Mid-epoch checkpoints.** `--ckpt_every` (default 1000 optimizer steps, 0
disables) writes `step_*.pt`, because resume granularity is one epoch and an
epoch on the full pool is hours long — without it a crash discards everything
since the last epoch boundary. They are rotated to `--keep_last_steps`
(default 2) **immediately after each write**, not at the epoch boundary: an
epoch is thousands of steps, so waiting would let a whole epoch's worth of
full checkpoints (model + both AdamW moments each) accumulate first.

`find_latest_checkpoint` picks the most recently *written* of {newest
`epoch_*.pt`, newest `step_*.pt`}, so `--resume auto` actually uses that
insurance. A `step_*.pt` is never a worse choice than the epoch checkpoint
preceding it — both carry the same `loop_state` epoch (a step checkpoint
stores `epoch - 1`, so its epoch replays from the top either way) but its
model and optimizer are further along. Selection is by mtime, not by parsed
number, because the two series are not numerically comparable and mtime
correctly prefers a fresh epoch checkpoint over a `step_*.pt` left by an
earlier resume generation.

Every save is atomic (`torch.save` to `*.tmp`, then `os.replace`): the file most likely to be half-written when a machine dies is the newest one, which is exactly the one `--resume` picks.

The epoch checkpoint is written **after** that epoch's validation, not before. It carries `best_val_loss` and the early-stopping counter, both of which the validation check decides; saving first would persist the pre-check values, so a resume from that file would let a worse epoch overwrite `best.pt` and would silently reset patience. The cost is that a crash *during* validation loses that epoch.

## Resuming an interrupted run

A run spans days, so a crash near the end must not cost the run:

```bash
python sdm_unips/train.py ... --resume auto     # same command that launched it
```

`--resume` accepts `auto` (this session's `checkpoints/`), any checkpoint
directory, or a specific file. A directory resolves to the most recently
written of {highest-numbered `epoch_*.pt`, highest-numbered `step_*.pt`}, else
`final.pt`; **never `best.pt`**, which is a *selection*
artifact and normally lags the training frontier. `auto` on a session with no
checkpoints starts a fresh run, so the same command works for launch and
relaunch; any other unresolvable path is a hard error (a typo must not silently
train from scratch for two days).

Two modes, chosen by inspecting the file and **printed at startup**:

- **full resume** — the checkpoint has an `optimizer` entry. Model, optimizer
  moments, LR-schedule position, GradScaler scale, non-finite-skip counters,
  every RNG stream (python / numpy / torch / cuda / the DataLoader shuffle
  generator), and the loop's epoch, `best_val_loss`, patience counter and
  `elapsed_sec` are all restored; training continues at the next epoch.
- **warm start** — the file holds weights only (a bare `*.pytmodel`
  `state_dict`), so nothing else *can* be restored. This is the `--pretrained`
  path; pointing `--resume` at such a file falls through to it with a printed
  warning.

Weights alone are worse than useless for *continuing* a run: a fresh AdamW has
no moment estimates and a fresh `LambdaLR` restarts at step 0, so the first
resumed step lands at the warmup LR with no gradient history. That is why the
two are separate flags.

**Config guard.** The checkpoint stores its own `args`. Resuming with a
different `--seed`, `--val_fraction`, `--test_fraction`, `--max_scenes` or
dataset root **aborts**: those redefine the deterministic scene-level split, so
the resumed run would train on a different partition and could pull held-out
test scenes into training. `--allow_config_change` downgrades it to a warning.
Differences in schedule-shaping args (`--epochs`, `--lr`, `--batch_size`, …)
only warn — the LR lambda is rebuilt from the new values while the step count
comes from the old run, so the LR curve bends at the resume point. That is
intentional when extending `--epochs`, and must be visible otherwise.

**Granularity is one epoch.** An interrupted epoch is replayed from its start;
mid-epoch resume would require fast-forwarding the sampler to a specific
micro-batch. A `step_*.pt` checkpoint (`--ckpt_every`) is resumable but
restarts its epoch from the top, so `global_step` overshoots `total_steps` by
the partial epoch — harmless, since cosine clamps its progress to 1.0 and the
loop is epoch-driven.

**Effect on the A/B graphs.** `epoch_summary`, `val_summary` and
`test_summary` are written once per *completed* epoch, so they are gap-free and
free of duplicates, and `elapsed_sec` continues across a restart (`t0` is
rewound by the stored elapsed time — it measures compute time and excludes
downtime). The convergence-vs-time curves therefore need no special handling.
The one artifact is in **per-step** records: the logs are append-only, so the
interrupted epoch's steps remain, followed by the replayed epoch, and raw
`elapsed_sec` rewinds at that seam. Every record carries `resume_count`
(0 for the original run) and a `{"kind": "resume", ...}` marker is written at
the seam, so the stale tail is a one-line filter. Each resumed run also writes
`config_resume_<n>.json` rather than overwriting the original `config.json`.

## Fine-tuning from a pretrained model

`--pretrained PATH` starts a **new** run from an existing checkpoint's weights:
fresh optimizer, fresh LR schedule, epoch 0, `best_val_loss` reset. `--resume`
means "continue *this* run"; `--pretrained` means "begin a new one from that
model". The MerlMix experiment is exactly this — continue Model A (selected on
the hdlong+PolarPS val split) on MerlMix alone and see whether DiLiGenT MAE
improves:

```bash
python sdm_unips/train.py --session_name modelA_merlmix \
  --hdlong_dir /path/to/MerlMix \
  --pretrained modelA_full/checkpoints/best.pt --resume auto \
  --val_fraction 0.1 --test_fraction 0 \
  --lr 2e-5 --warmup_epochs 0.5 --lr_schedule wsd --epochs 15 \
  --amp_dtype bf16
```

Then evaluate the exported weights externally:

```bash
python sdm_unips/eval_diligent.py --diligent_dir DATA \
  --checkpoint modelA_merlmix/checkpoints
```

**How `--pretrained` and `--resume` compose.** The two answer different
questions — `--pretrained` says *where a new run begins*, `--resume` says
*whether this is a new run at all* — so they are applied in a fixed order at
startup:

1. **Resolve both paths**, before the (slow) scene scan, so a typo dies in a
   second. `--resume auto` on a session with no checkpoints resolves to
   **None** (not an error: that is what makes one command work for launch and
   relaunch). `--pretrained` resolves a directory to `best.pt` and rejects
   `auto`.
2. **Build the Trainer** — random init, fresh AdamW, `LambdaLR` at step 0.
3. **`load_pretrained`**, if `--pretrained` was given. Model weights only.
   Records `args.pretrained_weights_sha1`.
4. **`resume_from`**, if `--resume` resolved to anything. A full checkpoint
   overwrites the model from step 3 and additionally restores optimizer
   moments, scheduler position, GradScaler scale, skip counters and every RNG
   stream, then returns the loop state.
5. **Loop state applied** only when step 4 returned one: `start_epoch`,
   `best_val_loss`, patience counter, `elapsed_sec`, `resume_count`, the
   shuffle generator, the auto-decay counters, the branched decay ramp, and the
   `check_resume_compat` split guard. Otherwise the run starts at epoch 0 with
   `best_val_loss = inf`.

| `--pretrained` | `--resume` resolves to | weights come from | starts at | optimizer + schedule |
|---|---|---|---|---|
| — | nothing | random init | epoch 0 | fresh |
| `X` | nothing | `X` | epoch 0 | fresh |
| — | `epoch_N.pt` | `epoch_N.pt` | epoch N+1 | restored |
| `X` | `epoch_N.pt` | `epoch_N.pt` (`X` discarded) | epoch N+1 | restored |

So `--pretrained X --resume auto` is one relaunchable command. Launch 1: the
checkpoint dir is empty, `--resume` resolves to nothing, and the run warm-starts
from `X`. It crashes at epoch 8. Launch 2, *same command*: `--resume auto` finds
`epoch_7.pt`, which wins outright — its weights are strictly downstream of `X`'s
— and training continues at epoch 8 with the moment estimates and LR position
intact.

Points that are load-bearing:

- **Why the order is pretrained-then-resume, and not a branch.** Whenever
  `--resume` finds a real training checkpoint it must win, because those
  weights already contain everything `--pretrained` would have supplied. Doing
  it by unconditional overwrite rather than by an `if/else` means there is no
  state in which the run can end up with neither. The cost is that a relaunch
  reads and hashes the pretrained file and then immediately discards it (a few
  seconds); that is deliberate, because it keeps `pretrained` and
  `pretrained_weights_sha1` in the argument snapshot of *every* checkpoint the resumed
  run writes, instead of the provenance dropping off at the first crash.
- **Nothing about `--pretrained` needs persisting**, unlike the WSD decay ramp.
  A branched ramp lives only in a lambda closure and must be carried in
  `sched_state` or it cannot be rebuilt; a warm start, by contrast, is already
  baked into the weights the checkpoint stores. `pretrained` is therefore in
  neither `SPLIT_CRITICAL_ARGS` nor `SCHEDULE_SENSITIVE_ARGS` — re-passing it
  on a relaunch is silent, which is correct, since it changes nothing.
- **`--resume` pointing at a bare `*.pytmodel`** finds no optimizer state, so
  it falls through to `load_pretrained` and returns None: a warm start at epoch
  0, announced. If `--pretrained` was also given, the `--resume` file wins (it
  is loaded second). Ambiguous and rare — name one or the other.
- **A directory resolves to `best.pt`** — the *selected* model, which is what
  a fine-tune should start from. This is the exact opposite of `--resume`'s
  directory rule, which never picks `best.pt` because that file lags the
  training frontier and resuming from it would discard epochs. Order is
  `best.pt` → `final.pt` → newest `epoch_*.pt` → an exported
  `normal/normal.pytmodel`.
- **Loading nothing is an error.** Tensors are matched by name *and shape*;
  the loaded / missing / unexpected / shape-mismatched counts are printed, and
  a match of zero raises rather than training from random init. (A full
  checkpoint's top-level keys are `model`/`optimizer`/…, not parameter names,
  so handing one to `load_state_dict(strict=False)` matches nothing and trains
  from scratch in silence — the guard exists for exactly that.) Shape filtering
  is deliberate: it lets a later architecture variant inherit the layers it
  shares with Model A instead of `load_state_dict` refusing the whole file.
- **Provenance is over the weights, not the file.** `best.pt` is overwritten as
  a run improves, so a path alone does not identify a model;
  `args.pretrained_weights_sha1` pins it, and goes into `config.json` and every
  checkpoint. `Trainer.weights_sha1` digests the `state_dict`'s sorted keys,
  shapes, dtypes and raw bytes — deliberately not the file, because the same
  weights arrive in two containers: `best.pt` wraps them alongside both AdamW
  moment buffers, the RNG streams, `sched_state` and the args snapshot, while
  `normal/normal.pytmodel` is the bare `state_dict`. A file digest would give
  those different values for identical weights and would fold in optimizer
  state the fine-tune discards. So `--pretrained <run>/checkpoints/best.pt` and
  `--pretrained <run>/checkpoints/normal/normal.pytmodel` are provably the same
  starting point, and record the same digest.
- **Use a lower `--lr`.** The recipe's peak 1e-4 with 5 warmup epochs would
  largely undo the pretrained weights before they learn anything from the new
  data.
- The new run rescans the scene pool. The session-manifest reuse in
  `_validate_pool` fires only for a genuine `--resume`, never for
  `--pretrained`, which is usually pointed at a different dataset entirely.
- The config guard does **not** compare against the pretrained checkpoint's
  args. A different dataset and split is the whole point of a fine-tune;
  `check_resume_compat` runs on full resumes only.

After the final epoch, `export_for_inference` republishes `<session>/checkpoints/normal/normal.pytmodel` from the model in memory — `best.pt` when a best was recorded, else the final-epoch weights — so the exported file and the reported test number cannot describe different models (see **One inference file, not three** above). Inference runs directly against the training output:

```bash
python sdm_unips/main.py --session_name SESSION --test_dir DATA \
    --checkpoint <session>/checkpoints
```

Logs land in `<session>/logs/`:
- `train.jsonl` / `train.log`  — per-step loss, MAE, LR, grad norm, step seconds; per-epoch and per-validation summaries

**ETA (always on).** `total_steps` alone says nothing about wall clock, which
is how a 40-epoch run on the full 51.5k-scene pool turned out to be ~10 days
without anyone noticing until it was already running. Two records now convert
steps into time from **measured** throughput:

- `{"kind": "eta", ...}` — printed once, ~100 optimizer steps in. Early enough
  to kill a run whose length is unacceptable before it has cost anything, late
  enough that the worker-pool warmup has washed out. Carries `sec_per_step`,
  the implied hours/epoch, `eta_sec` and `eta_finish_unix`.
- `sec_per_epoch_cycle`, `eta_sec`, `eta_finish_unix` on every epoch summary,
  measured over **full epoch cycles** (training *plus* the validation and
  checkpointing between them) rather than `epoch_sec`, which excludes both and
  therefore under-counts a run's real length when summed.

Both measure from a baseline taken at the top of the epoch loop, not from
`t0`: `t0` is rewound across a resume and on a fresh run still carries the
scene-check and dataset construction, neither of which recurs per step, so
folding it in would inflate every estimate.

`elapsed_sec` (wall-clock since training began) is carried on every step
record, epoch summary, validation summary and the final test summary. It is
the x-axis for a convergence-vs-time comparison between model variants, which
`epoch_sec` cannot supply on its own: `epoch_sec` is measured before the
checkpoint save and the validation pass, so its cumulative sum under-counts
real elapsed time. Deliberately *not* accumulated into the per-epoch averages
(a mean of a monotonically increasing quantity is meaningless).

For an A/B convergence comparison, note that `avg_loss` / `avg_mae_deg` on the
epoch summary are computed on each model's **own** sampled pixels and are
therefore *not* comparable across variants — use them as per-run diagnostics
only. `val_*` and `test_*` are the cross-variant-valid metrics (see the A/B
fairness contract above). Run comparison runs with `--patience 0` so both
curves span the full `--epochs`.
- `eval.jsonl` / `eval.log`    — the single final held-out test summary (mean loss + MAE)
- `config.json`                — frozen CLI arguments for the run

**Resource profiling (`--log_memory`, off by default).** Adds
`mem_peak_gib` / `mem_reserved_peak_gib` (CUDA high-water marks, reset each
epoch), `host_rss_gib` (self + DataLoader workers, an upper bound since
copy-on-write pages are counted per process), and `data_wait_sec` (seconds the
step spent blocked on the loader) to every `train.jsonl` record, plus
`epoch_*` peaks, `data_wait_sec_total` / `step_sec_total`, `data_wait_share`
and `unaccounted_sec` on each epoch summary. It is opt-in
because these fields roughly double the width of every record; enable it only
when sizing `--batch_size` / `--num_workers`, then turn it off.

Reading it: `mem_peak_gib` vs the card's capacity is the headroom for a
larger batch (`mem_reserved_peak_gib − mem_peak_gib` is allocator
fragmentation). `data_wait_share ≈ 0` means the loader keeps up and more
workers buy nothing; a large share means input-bound, so raise
`--num_workers` / `--prefetch_factor` or move the data off a slow disk.
`step_sec` alone cannot distinguish the two. `--log_memory` also inserts a
`torch.cuda.synchronize()` before reading `step_sec` (and only then — a
per-step sync costs throughput), without which async kernel launches make
`step_sec` time launches rather than compute and leak real GPU time into the
next step's `data_wait_sec`, leaving both non-comparable across runs. Ignore
epoch 0's
`data_wait_share`: its first `data_wait_sec` absorbs worker-pool startup and
the initial fill, which `persistent_workers=True` pays only once.

`data_wait_share`'s denominator is the **accounted** time
(`data_wait_sec_total + step_sec_total`), not `epoch_sec`. The difference is
reported as `unaccounted_sec` — per-step JSONL writes, the `/proc` reads for
`host_rss_gib`, the epoch checkpoint, validation, and the trailing
`flush_accum`, none of which either timer covers. Check it before trusting a
tuning decision: a small residual means the accounting closes; a large one
means profiling overhead (or validation) dominates the epoch, so re-measure
with `--log_memory` off once the sizing is settled.

`_seed_worker` calls `cv2.setNumThreads(0)`: OpenCV otherwise spawns one
thread per core inside *each* worker, so `W` workers oversubscribe the CPU by
a factor of `W` and raising `--num_workers` can reduce throughput. Any
worker-count benchmark taken without this measures contention, not the loader.
`train.py` also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(via `setdefault`, so an explicit env var still wins) to reduce the
fragmentation-driven OOMs that occur while `nvidia-smi` still shows free VRAM.

## Data Format

**Inference layout:**
```
YOUR_DATA_PATH/
└── OBJECT_NAME.data/
    ├── L*.png          # input images (8-bit or 16-bit)
    ├── mask.png        # optional object mask
    └── Normal_gt.png   # optional ground-truth for MAE evaluation
```

**Inference output layout** (written to `SESSION_NAME/results/OBJECT_NAME.data/`):
```
normal.png
error.png      # only if Normal_gt.png provided
```

**Training layout (hdlong-complexv1 / MerlMix + PolarPS):**
```
HDLONG_ROOT/            # one or more; MerlMix uses this exact layout
└── SCENE_NAME/
    ├── light_means.config        # required marker; lists point/dir/env means
    ├── cam_0000N/
    │   ├── binary_mask.exr
    │   ├── local_normal.exr      # GT, RGB-encoded as (n+1)/2 in [0,1]
    │   ├── point_light_000NN.exr # 20 per cam, divided by point_mean
    │   ├── dir_light_000NN.exr   # 20 per cam, divided by dir_mean
    │   └── env_light_000NN.exr   # 10 per cam, divided by env_mean
    └── ...

POLARPS_ROOT/
└── SCENE_NAME/
    ├── normal.exr                # required marker; mask derived from magnitude
    └── <material-mix-subdir>/
        ├── light-01/S0.exr
        ├── light-02/S0.exr
        └── ... (32 lights total per scene)
```
The dataset class auto-detects scene type via these markers (`light_means.config` ⇒ hdlong, `normal.exr` ⇒ PolarPS). **MerlMix carries the hdlong layout and marker**, so it is passed as an additional `--hdlong_dir` root and needs no loader, `kind`, or preprocessing of its own. It synthesizes each training render via a Dirichlet (α, β, γ) mix of one randomly chosen point/dir/env light triple (hdlong) or by drawing one of 32 `S0.exr` images (PolarPS). hdlong is upsampled from 256×256 to `--train_resolution`.

**Per-image normalization:** each observation is divided by a single scalar — the **max, over its foreground pixels, of that pixel's mean across the three colour channels**. This is deliberately the same statistic `realdata.py` uses at inference (`temp = mean over channels; mx = max over pixels; I /= mx`), so observations land in ~[0, 1] in both paths. That matters in two places the architecture is built around: `Net.forward` concatenates the 0/1 mask as a 4th input channel (a differently-scaled RGB drowns it out), and `Net._decode_pixels` concatenates raw observations with the 256-d GLC features before an `ln=True` attention block (a large-magnitude observation channel dominates the LayerNorm statistics and flattens the GLC signal).

PolarPS caches the scalar in an `image_scales.config` sidecar per scene — its observations are fixed `S0.exr` files, so the value depends only on file + mask and is computed once, at the on-disk resolution so it survives a change of `--train_resolution`. **hdlong does not cache**: its observation is a random Dirichlet mix drawn afresh each epoch, and unlike the mean, the max is not linear (`max(Σ wᵢxᵢ) ≠ Σ wᵢ max(xᵢ)` — empirically ~12% off), so there is no per-component quantity to recombine; the scale is measured directly on each composite pre-resize. Read-only dataset mounts (e.g. Kaggle inputs) simply skip the sidecar write and keep an in-process cache.

> Migration: **delete any `image_means.config` files still present in the dataset trees.** They hold means from the pre-2026-08-05 normalization and are dead weight; the current sidecar is `image_scales.config`.

## Architecture

The system has three logical stages.

### 1. Encoding — `modules/model/model.py` → `ScaleInvariantSpatialLightImageEncoder`
Each input image is encoded at a fixed 256×256 canonical resolution regardless of original resolution. A **ConvNeXt-T** backbone (`modules/model/convnext.py`, depths `[3,3,9,3]`, dims `[96,192,384,768]`) extracts 4-scale features fused by **UPerHead** (`modules/model/uper.py`). Light-axis transformer blocks are stacked at counts `[0,1,2,4]` per scale. The output is a *Global Light-aware Context* (GLC) feature map at ¼ resolution. For high-res inputs (or `--scalable` inference), images are decomposed into G×G sub-tensors via `modules/model/decompose_tensors.py`, processed independently, then recomposed; a downsized-image feature map is added back to promote inter-tile interaction.

### 2. Aggregation — `modules/model/model.py` → `GLC_Aggregation`
A `CommunicationBlock` transformer (`modules/model/transformer.py`) performs **cross-image attention**: at each sampled pixel, K image features attend to each other along the light axis. PMA collapses K → 1 (paper's pixel-sampling Transformer). No positional embeddings on samples (paper Sec. 3).

### 3. Regression — `modules/model/model.py` → `Regressor`
At the full decoder resolution (up to 4096×4096 in inference, 512 by default in training), per-pixel prediction. A spatial-axis communication transformer refines features across pixels in a sample set, then a `384→192→3` MLP predicts the unit normal.

### Inference orchestration — `modules/builder/builder.py`
`Builder` loads the checkpoint, wraps `Net`, and runs the pixel-sampling loop in `--pixel_samples` chunks across all valid pixels. In scalable mode it handles patch decomposition at the encoder level and Gaussian feature smoothing (`modules/utils/gauss_filter.py`) at patch boundaries.

### Training orchestration — `modules/builder/trainer.py`
`Trainer` builds `Net`, `AdamW`, the LR scheduler (step decay or cosine, both with epoch-based warmup), and `GradScaler` (AMP only for fp16). `Net.forward(..., training=True)` samples exactly `pixel_samples` valid-mask pixels per batch element with gradients on (no `.detach()`), and returns flat per-pixel predictions plus the sampled flat indices. The loss (`modules/loss/losses.py:normal_loss`) gathers GT at those indices and computes masked MSE on normals (Sec. 4). Each save also writes `normal.pytmodel` so checkpoints are drop-in for inference.

`train_step` runs one **micro**-batch: it zeroes gradients only at the start of an accumulation cycle, divides the loss by `--accum_steps` before `backward()` (so the accumulated gradient is the *mean* over the effective batch), and optimizes on every `accum_steps`-th call, flagging that in `log['stepped']`. Clipping therefore sees the complete effective-batch gradient, which is what `--grad_clip 1.0` is calibrated against. `Trainer.flush_accum()` applies a short trailing cycle at each epoch boundary so the last micro-batches are not discarded by the next `zero_grad`; `steps_per_epoch` is `ceil(len(train_loader) / accum_steps)` to match.

### Data loading
- Inference: `modules/io/dataloader/realdata.py` — auto bounding-box cropping, square aspect ratio, mean-luminance normalization, optional masking, GT MAE (`modules/utils/compute_mae.py`).
- Training (thesis mix): `modules/io/dataloader/hdlong.py` (per-camera Dirichlet light mixing for hdlong-complexv1), `modules/io/dataloader/polarps.py` (random K of 32 S0 lights), unified by `modules/io/dataloader/mixed.py:MixedTrainDataset`. `modules/io/dataloader/scene_check.py` validates the pool before the split, owns the shared "usable unit" helpers the loaders draw from, and owns multi-root keying (`as_roots` / `build_roots`). Every hdlong-structured dataset — hdlong-complexv1 and MerlMix alike — goes through `hdlong.py`; MerlMix is an extra `--hdlong_dir` root, not a new module. The per-image normalization scalar and its PolarPS-only sidecar cache live in `modules/io/dataloader/scale_cache.py`. `mixed.py:build_mixed_split` performs the deterministic, per-source-proportional scene-level train/val/test split and returns all three disjoint datasets in one pass (same seed ⇒ identical, disjoint splits) and `train.py` wraps the two held-out splits for evaluation. K is `--k_per_scene` (default 10) per scene. (`modules/io/dataio.py` is upstream's *inference* loader, untouched — it has no train/val/test accessors.)
- Held-out val and test: both wrapped by `mixed.py:MixedEvalDataset` (length `n_scenes × n_trials`, per-index-seeded so renders are worker-count-independent and reproducible). Val uses `n_trials=1` and runs every `--val_every_epochs`; test uses `--test_trials` and runs **once after the final epoch** via `train.py:run_test_eval`. Both sweep `train.py:run_eval_pass` → `Trainer.val_step` (scene-count-weighted mean, non-finite dropped). `--test_fraction 0` builds no test loader (`--val_fraction` must be > 0, so the val loader always exists); `sdm_unips/eval_diligent.py` is the external benchmark used when the test split is off.

## Environment

- Python 3.11, PyTorch 2.0, CUDA 11.8
- Dependencies pinned in `requirements.txt` (`torch`, `numpy`, `opencv-python`, `einops`).
- Tested on NVIDIA RTX A6000 (48 GB VRAM); CPU fallback supported for inference. Training expects a CUDA GPU.
- Platforms: Ubuntu 20.04.5 (WSL2) and Windows 11
