# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDM-UniPS is a **CVPR 2023 Highlight** paper implementation for **Universal Photometric Stereo** — recovering surface normal maps from multiple images captured under arbitrary, spatially-varying lighting with a fixed camera. The upstream repository is inference-only; this fork adds a training pipeline (`sdm_unips/train.py`, `modules/builder/trainer.py`, `modules/io/dataloader/`, `modules/loss/`) with train/val/test all drawn from the same synthetic mixed pool.

**Scope:** this fork targets **surface-normal prediction only**.

The current branch (`modelB-wtconv`) targets **Model B1** (WTConv). **Model A** (the baseline SDM-UniPS architecture, no modifications) is trained on a mix of `hdlong-complexv1` and `PolarPS` for the author's thesis: *"Optimizing Universal Photometric Stereo via Wavelet-Energy Saliency and Latent Carrier Tokens."* The branch `modelA-vflip-mean-max-scale` adds more versatile augmentations, the model is also to be trained on the `MerlMix` dataset. Models B and C (WTConv, WESS, latent-carrier deformable attention) live on dedicated branches — see **Model variants** below.

## Model variants

Three modifications sit on top of Model A. **Each is a branch, not a flag.**
The variant is chosen by checking out its branch; `train.py` carries training
hyperparameters only, and the architecture is not one of them. There is
deliberately no runtime switch that can select the baseline from a variant
branch — a flag-gated architecture puts the identity of the trained model in
the launch command rather than in the code, which is where a thesis comparison
cannot afford it.

| Variant | Branch | What changes |
|---|---|---|
| **A** — baseline | `modelA-vflip-mean-max-scale` | — |
| **B1** — WTConv | `modelB-wtconv` | `convnext.Block.dwconv`: 7×7 depthwise conv → wavelet convolution |
| **B2** — WESS | `modelC-wess` | `Net.sample_train_pixels`: uniform draw → wavelet sub-band energy over the mask **interior**; see **B2 — WESS** below |
| **C** — carrier tokens + deformable attention | (planned) | `Regressor`'s pixel-sampling transformer |

**What this costs, and why it is the right trade.** A checkpoint is only
meaningful under the branch that produced it. `main.py` and `eval_diligent.py`
on `modelB-wtconv` will not load Model A's `normal.pytmodel` — `builder.
load_models`' partial-match guard raises, loudly, rather than running a
half-initialised network. To evaluate A, check out A's branch. Likewise
`--resume` against a foreign checkpoint dies in `resume_from`'s
`load_state_dict`, which is strict: a name-and-shape guard, strictly stronger
than the args-comparison guard a `--dw` flag would have needed.

The one cross-variant path that still works is `--pretrained`, which matches by
name *and* shape under `strict=False`: B1 warm-started from Model A inherits
everything except the 12 depthwise kernels (A's 7×7 `dwconv.weight` has no
counterpart in a WTConv block, whose base conv is 5×5).

### B1 — WTConv (`modelB-wtconv`)

`modules/model/wtconv.py`. Replaces the ConvNeXt block's 7×7 depthwise
convolution with a cascaded Haar DWT: each level halves resolution and splits
into {LL, LH, HL, HH}, a small depthwise kernel (5×5) is applied to all four
independently, and the levels are folded back with the inverse transform and
added to a full-resolution base convolution. The point is that a large spatial
kernel low-passes exactly the intensity gradients photometric stereo reads
normals from, whereas the sub-band form never averages low and high frequencies
together.

**The diff is two files.** `wtconv.py` (new) and `convnext.py` (one attribute).
`model.py`, `builder.py` and `train.py` are byte-identical to Model A's branch;
`trainer.py` differs only in a log line naming the backbone. That is not
tidiness — it is the drop-in claim, made checkable by `git diff`. `WTConv2d`
maps `[N,C,H,W] → [N,C,H,W]`, identically to the `nn.Conv2d` it replaces, so
*nothing* downstream can observe it: the UPerHead fusion, the light-axis
attention, the 128×128 GLC merge and the 2048-pixel decode are unchanged by
construction. `sdm_unips/tests/test_wtconv_shapes.py` pins it.

Three things are load-bearing:

- **The Haar filters are buffers, not `nn.Parameter(requires_grad=False)`**
  (which is what the reference implementation uses). `Net.with_grad()` →
  `model_utils.mode_change` does a blanket
  `for param in net.parameters(): param.requires_grad = True`, so a
  non-trainable Parameter would be silently promoted and swept into the AdamW
  parameter list — the fixed orthonormal basis the whole method rests on would
  drift during training, with no error and no log line.
- **`wt_levels` is per stage.** Both encoder paths feed the backbone a
  `--canonical_resolution` square (`x_resized` is interpolated to it,
  `x_grid`'s tiles are exactly it), so stage resolutions are a fixed
  64/32/16/8 at R=256 and the budget can be settled at construction.
  `convnext.derive_wt_levels` picks the largest depth that keeps every
  sub-band ≥ 4×4 — `(3, 3, 2, 1)` at 256, which is `ConvNeXt.__init__`'s
  default. A flat `3` would give stage 3 a **1×1** sub-band, where a 5×5
  depthwise kernel is all padding around one real tap. Training at a different
  canonical resolution means editing that default, which is correct: it is an
  architecture decision and belongs in the architecture file.
- **The cascade recurses on the raw LL**, not the convolved one, so every level
  decomposes the input band-limited rather than the accumulated output of the
  levels above it. The *filtered* LL is what the synthesis pass consumes. Swap
  the two and the pyramid becomes a serial chain of convolutions, losing the
  independent-frequency-processing property the method exists for.

**Cost.** The backbone runs **5N times per scene** (N resized + 4N tiles), so a
per-block slowdown is multiplied by 50 at K=10 before `--batch_size` is applied.
Parameters rise only modestly (depthwise is small next to the 1×1s) but DWT/IWT
are unfused `conv2d`/`conv_transpose2d` at 3 levels × 4 stages × 12 blocks. Read
the `{"kind": "eta", ...}` record ~100 optimizer steps into the first launch and
kill the run there if the projected length is unacceptable.

**Hook for B2.** `WTConv2d.tap_subbands` (off by default, so B1 holds no extra
tensors) makes `WTConv2d.subbands` hold the level-0 detail bands
`[N, C, 3, H/2, W/2]`, **detached** — the saliency draw is an index selection
and carries no gradient, and keeping the live tensor would pin the encoder
graph. Note that the encoder calls the backbone twice and the *tile* path
(`x_grid`, 4N) is last, so that is what the attribute holds after a forward;
assert on `subbands.shape[0]` rather than relying on call order.

**Shape probe:** `python sdm_unips/tests/test_wtconv_shapes.py` (needs torch, so
it runs on the training box). Checks Haar orthonormality and perfect
reconstruction, sub-band channel ordering, shape preservation at all four stage
geometries, the odd-size padding path, the buffer-vs-parameter guarantee, the
backbone's four stage outputs against Model A's shapes, and the derived level
budget. Also prints the depthwise parameter count against Model A's.

### B2 — WESS (`modelC-wess`)

`modules/model/wess.py` plus a `Net.sample_train_pixels` body. The training
pixel draw stops being uniform over the mask and instead follows the level-0
Haar sub-band energy of the *first* WTConv block (stage 0, block 0), reduced
over the K images by the mean of the top 2, so each optimizer step spends its
2048-pixel budget on creases and shadow boundaries rather than on flat regions
that contribute almost no gradient.

**The tap is a forward pre-hook, not a model edit.** It sits on
`stages[0][0].dwconv.wavelet_convs[0]`, whose input is exactly the raw
`bands.reshape(n, 4C, h, w)` — the pure Haar coefficients, before any learned
parameter. `WTConv2d.tap_subbands` cannot supply that: it stores the *filtered*
bands (scaled by a trainable gain that drifts during training) and is
last-writer-wins across all 12 blocks, so after a forward it holds stage 3's
4×4 bands. The hook adds no parameters and no buffers, so B1's `best.pt` loads
here under the exact-match guard and `--pretrained` warm-starts cleanly.

The energy is captured on the **tile** path (`x_grid`, K·N maps), never the
resized one: `x_resized` is a bilinear downsample, i.e. a low-pass filter that
destroys precisely the frequencies being measured. The four 32² tile maps merge
through `merge_tensor_spatial` into a 64² full-frame map. Nominal cell size is
8×8 image pixels; **true localization is ~16 px**, because the four tiles' cell
(i,j) all summarize the same 16×16 region at four different phases. WESS
concentrates the budget on crease *neighbourhoods*, not crease pixels.

**The silhouette is excluded from the reweighting — this is the load-bearing
decision, and it came out of Phase 0.** `Net.forward` feeds the backbone
`cat([I * M, M])`, so the object boundary is a hard step in all four input
channels and its level-0 detail bands are enormous *by construction*. Left
alone the sampler put ~40% of the draw on that ring (τ=1) against 7–9% for a
uniform draw, and — worse — the ring inflated `e.std()` enough to squash every
interior z-score, so the interior softmax came out nearly flat. Sampling the
silhouette harder needs no wavelets and is not the thesis claim. So the
distribution is:

```
p(rim pixel)      = 1 / n_valid                        # Model A, untouched
p(interior pixel) = (n_int / n_valid) * [ lam / n_int + (1 - lam) * softmax_int ]
```

The two blocks sum to `n_rim/n_valid + n_int/n_valid = 1` exactly, so nothing is
renormalized. The rim keeps Model A's rate, which means **any A/B difference is
attributable to interior reallocation alone** — a win cannot be explained away
as "B2 just trained harder on the boundary, where GT normals sit at grazing
angles and the mask edge is antialiased". `--wess_erode_cells` (2, in 64²
energy-grid cells ≈ 16 px at R=512) sets the excluded band; the erosion runs on
the energy grid because that is the resolution at which the ring is defined.

Order of operations inside the interior is fixed and was settled before
implementation: **upsample → gather at the mask → standardize → softmax → mix
with uniform**. Softmax before the mask would spend the budget on background
(`exp(0) = 1`); softmax before standardizing would make `--wess_tau` mean
something different at every epoch, since `E` is an L2 norm over 96 channels of
a *learned* feature map whose scale drifts as the encoder trains. Standardizing
puts τ in units of σ.

**`--wess_lam` is a mixture weight, not a second draw.** The proposal split the
budget into 512 uniform + 1536 saliency-drawn, which needs bookkeeping to keep
the two disjoint. The mixture has the same expected count per pixel, takes one
`multinomial`, and converts the support argument from asymptotic into a hard
guarantee: every masked pixel keeps `p ≥ lam/n_valid` however peaked the
softmax gets. The sampler changes the *rate* at which a pixel is supervised,
never whether it can be — which is what makes training under `E_P` while
reporting `E_uniform` defensible. **`--wess_lam 1` (or a large `--wess_tau`)
recovers Model A's uniform draw exactly**, on the rim and in the interior
alike; that is a free correctness check, and worth running for a few hundred
steps before the real launch.

**The A/B fairness contract is untouched.** The tap is armed only when
`training=True and sample_ids is None`. Evaluation always supplies `sample_ids`
(`Trainer._eval_sample_ids`), so the hook never fires on a val/test forward and
those numbers stay byte-identical to Model A's. WESS lives entirely inside the
training draw.

**Instrumentation (Phase 1).** `sample_train_pixels` accumulates per-scene
diagnostics that `_forward_losses` folds into the ordinary per-step log, so
they land in `train.jsonl` and are averaged into each epoch summary by the same
code that averages `loss` — no new plumbing:

- `wess_ess` — effective sample size `1/Σp²` as a fraction of the mask. **The
  one to watch on launch.** Below ~0.1 the draw has collapsed onto a handful of
  pixels and the per-step gradient is high-variance; near 1.0 the sampler is
  inert. Phase 0 measured 0.21 at τ=1 *with* the rim included, and excluding the
  rim removes what was inflating σ, so the shipped sampler will read lower at
  the same τ. If the first hundred steps show `wess_ess < 0.1`, raise
  `--wess_tau`.
- `wess_tilt_int` — energy tilt achieved *within the interior*, the only region
  WESS may reweight. The honest headline number.
- `wess_tilt` — the same over the whole mask, kept for comparability with the
  Phase-0 records; diluted by the rim samples, which are uniform by construction.
- `wess_interior_frac` — share of the draw landing inside the eroded interior.
  Should sit near the uniform value (~0.90); a drift away from it means the
  erosion is not doing what it claims.
- `wess_top10_frac` — share of the draw in the mask's top energy decile
  (uniform reads 0.10).

**Phase 0 (premise validation), n = 431, the full val split.** `sdm_unips/wess_probe.py`
loads B1's `best.pt`, taps the sub-bands and asks whether `E` points at geometry
before any GPU time is spent training on it. It answered yes: ρ(E, GT curvature)
median **+0.46**, positive in 99% of scenes and above +0.2 in 90%, and
consistent across both sources (hdlong +0.44, PolarPS +0.47). B1's MAE on
WESS-drawn pixels is **+0.79°** above its MAE on uniform-drawn pixels at τ=1
(higher in 80% of scenes), i.e. the sampler does find pixels the model currently
gets wrong — and across scenes ρ(that gain, interior curvature tilt) = **+0.39**,
the strongest coupling in the probe, so the gain tracks the mechanism rather
than the silhouette.

Two limitations to report rather than let a reviewer find:

- **Albedo contamination is real but not dominant.** The partial correlation
  ρ(E, |∇I| | curvature) is +0.29 overall (+0.22 hdlong, +0.33 PolarPS). The
  panels for the worst offenders are reassuring — on `soap-002@fabric_159` and
  `blob01-uv@metalplate_4` the GT normal is smooth, `|∇I|` lights up the painted
  ornament, and `E` visibly does *not* follow it — so much of that partial
  correlation is the silhouette, where E and `|∇I|` both spike and where
  `np.gradient`-based curvature is itself an artifact. Still, E is a feature-map
  energy, not a shading-gradient estimator, and nothing forces it to ignore paint.
- **The gain is largest where B1 is already good.** ρ(MAE gain, uniform MAE) is
  −0.13 to −0.22. WESS targets hard *pixels within* a scene; it does not target
  hard scenes.

**Probe:** `python -u sdm_unips/wess_probe.py --checkpoint <B1>/checkpoints/best.pt
<same --hdlong_dir/--polarps_dir/--scene_manifest as the run> --num_scenes 0
--tau 1.0 2.0 --band raw --no_figures --out_dir <dir>`. Roughly 3 s/scene plus
~1–2 min of startup; drop `--no_figures` for per-scene panels. Note `python -u`:
the probe's `print`s are block-buffered into a redirected file otherwise, and a
redirected run looks hung for minutes.

### Comparing A against B1

`eval_diligent.py` is the verdict, and it is comparable across the two branches
provided the runs share a schedule and a split:

- Pin the pool: `--scene_manifest <modelA-session>/logs/scene_manifest.json`
  and the same `--seed`.
- Pin the schedule: same `--epochs`, an explicit `--decay_from_step` (A's), no
  `--auto_decay`, and `--patience 0` so both curves span the full run.
- DiLiGenT decodes **every** valid pixel (`builder.run` chunks through all of
  them), so the pixel-sampling question the A/B fairness contract exists for
  does not arise there at all; and at DiLiGenT resolution P ≤ 4, so the GLC
  Gaussian fires for neither. Do check that A's number was produced with the
  **current** code — the P > 4 gate changed what inference does at P ≤ 4.

One irreducible caveat: B1 has more parameters, so initialisation consumes the
torch RNG stream differently and the two runs see different *training* pixel
draws from step 0. Same distribution, different sample — noise, not bias, but
it means a small MAE gap on one seed each is not a result.
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
after the run exports them). **`--val_fraction` has no such switch and must be > 0** — validation is what chooses which epoch to keep, so a run without it
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
   `np.random`: `augment=False` disables only the augmentations (the two flips
   and the random normalization scale), so val scenes would otherwise be
   **re-rendered every epoch**. Val uses
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
  (There are deliberately **no architecture flags**: the variant is the branch.
  See **Model variants** above.)
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
- `--hf_backup`: mirror the resume-critical files to a HuggingFace repo at
  each epoch boundary (**off by default**) — `--hf_repo`
  (`culacgiontan0312/UniPS`), `--hf_repo_type` (`dataset`),
  `--hf_backup_every` (1). See **Off-box backup** below
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

## Off-box backup (`--hf_backup`)

`--resume` protects against a *crash*; it does nothing about the machine
itself disappearing. On rented GPU time that is the likelier loss, and the
whole session directory goes with it.

Almost everything on the box is regenerable for free — the datasets come back
from HuggingFace, the code from git, `scene_manifest.json` from a rescan (the
manifest and its fingerprint are relative to the dataset roots, so a fresh scan
of the same data reproduces the same pool and the same split), and
`normal.pytmodel` is re-exported from `best.pt`. **The trained weights are the
only thing that cannot be reconstructed at any price**, so they are the only
thing that has to leave the box.

`--hf_backup` uploads to `sdm-ckpt/<model-name>/` in `--hf_repo`, where
`<model-name>` is the last path component of `--session_name` (a session at
`$HOME/runs/modelA` backs up to `sdm-ckpt/modelA/`):

- `latest.pt` — the epoch checkpoint just written. A **full** checkpoint, so
  `--resume <path>/latest.pt` on a fresh box continues the run with optimizer
  moments, scheduler position and RNG streams intact rather than warm-starting.
- `best.pt` — the deliverable. Not needed to resume, but if the run never
  improves again after a crash there would otherwise be no selected model.
- `train.jsonl`, `eval.jsonl`, `config*.json` — the curves and the exact args.

Deliberately **not** uploaded: `step_*.pt` (resume granularity is one epoch, so
a mid-epoch snapshot buys nothing once `latest.pt` exists), `normal.pytmodel`
(re-exportable from `best.pt`), `scene_manifest.json` (regenerable, and large).

**Where the call sits is load-bearing.** The upload runs in the epoch loop
*after* `trainer.save`, the `best.pt` copy and `prune_checkpoints` — so every
file it reads is complete (`_atomic_save` has already renamed its `*.tmp` into
place) and none is about to be rotated away. Two further forced uploads run at
the end: one after `export_for_inference`, and one after the test eval so
`eval.jsonl` is included (everything unchanged since the previous upload is
skipped by a size/mtime check).

**It can never kill a run.** Every failure path prints `[HF-BACKUP] …` and
returns; `huggingface_hub` is imported *inside* the call, so a box without it
trains normally. `preflight()` checks credentials and repo access once at
startup rather than hours in at the first backup point — and warns rather than
aborting. With `--hf_backup` absent the object is inert.

**`--hf_backup_every` is a storage control, not a safety dial.** HuggingFace
keeps LFS history even though the remote filenames are overwritten, so every
upload adds a revision that counts against the repo (HF asks that dataset repos
stay under ~300 GB). At roughly 1.5–2 GB per backup, `--hf_backup_every 5` over
a 200-epoch run costs ~70 GB of history and risks losing at most 5 epochs.
Collapse the accumulated history with
`HfApi().super_squash_history(repo_id=..., repo_type='dataset')` when it grows;
that is irreversible and keeps only the current tree.

**Restoring on a fresh box**: download `latest.pt` and `best.pt`, put `best.pt`
in `<session>/checkpoints/` **first** (otherwise a later, worse epoch
overwrites the selected model), then relaunch with
`--resume <path>/latest.pt`.

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

**Augmentation (training only).** Two label-preserving symmetries and one
intensity transform, all gated on the loaders' `augment` flag, which
`build_mixed_split` sets False for both held-out splits:

- **Horizontal and vertical flips**, drawn *independently* (so all four
  orientations occur). Reflecting the image plane about an axis negates the
  normal's in-plane component along that axis and leaves z alone, so the flip
  is exact, not approximate: hflip reverses the width axis and negates
  `N[..., 0]`, vflip reverses the height axis and negates `N[..., 1]`. The
  reversed views are materialized with `ascontiguousarray` *before* the sign
  flip, since the negation is an in-place write. Note the datasets render with
  lights over the upper hemisphere, so the vertical flip is what supplies
  below-lit examples — legitimate here because UniPS assumes unknown,
  arbitrary lighting, and real captures are lit from anywhere.
- **A random normalization scale**, described next.

**Per-image normalization.** Each observation is divided by a single scalar,
drawn per image from **U[mean, max]** of its foreground intensity while
training and pinned to the **max** everywhere else, where both statistics are
taken over the object's foreground pixels of that pixel's mean across the three
colour channels. This follows the paper (Sec. 3.1: *"each image is normalized
by a random value between its maximum and mean"*), whose own inference path
uses the max end: `realdata.py` does `temp = mean over channels; mx = max over
pixels; I /= mx`. So the eval/inference condition lands observations in ~[0, 1],
and the training draw *widens* the range around it (a smaller divisor pushes
values above 1) so the network cannot assume its input is exactly
max-normalized.

The eval end of that range matters in two places the architecture is built
around: `Net.forward` concatenates the 0/1 mask as a 4th input channel (a
differently-scaled RGB drowns it out), and `Net._decode_pixels` concatenates
raw observations with the 256-d GLC features before an `ln=True` attention
block (a large-magnitude observation channel dominates the LayerNorm statistics
and flattens the GLC signal). Note the asymmetry this creates and live with it
deliberately: because the range is `[mean, max]` rather than centred on the
max, the inference condition sits at the *edge* of the training distribution,
hit exactly when `t = 1`. That is what the paper specifies, and matching it is
the point.

Held-out val and test never draw: `augment=False` pins them to the max, which
is both the inference condition and a *fixed* one — a random normalizer at eval
would add variance to the very number that selects `best.pt`, for no
information. It also means the eval renders are byte-identical to those of runs
made before this augmentation existed (`augment=False` short-circuits every new
RNG draw), so val/test numbers stay comparable across the change.

PolarPS caches **both statistics** in an `image_scale_stats.config` sidecar per
scene (one `<key> <mean> <max>` line per image) — its observations are fixed
`S0.exr` files, so the pair depends only on file + mask and is computed once, at
the on-disk resolution so it survives a change of `--train_resolution`. Only the
statistics are cached, never the sampled scale, which is redrawn every epoch.
**hdlong does not cache**: its observation is a random Dirichlet mix drawn
afresh each epoch, and neither statistic can be recombined from per-component
ones — the max is not linear (`max(Σ wᵢxᵢ) ≠ Σ wᵢ max(xᵢ)` — empirically ~12%
off), and the mean is linear only in the mixing weights, which are themselves
redrawn — so both are measured directly on each composite pre-resize.
Read-only dataset mounts (e.g. Kaggle inputs) simply skip the sidecar write and
keep an in-process cache.

> Migration: **delete any `image_means.config` and `image_scales.config` files
> still present in the dataset trees.** The first holds means from the
> pre-2026-08-05 normalization, the second max-only scalars from before the
> mean/max pair; nothing reads either. The filename changed each time precisely
> so an older sidecar is ignored rather than misread — `read_scale_stats`
> requires three tokens per line, so a two-token `image_scales.config` line
> cannot be mistaken for a mean.

## Architecture

The system has three logical stages.

### 1. Encoding — `modules/model/model.py` → `ScaleInvariantSpatialLightImageEncoder`
Each input image is encoded at a fixed 256×256 canonical resolution regardless of original resolution. A **ConvNeXt-T** backbone (`modules/model/convnext.py`, depths `[3,3,9,3]`, dims `[96,192,384,768]`) extracts 4-scale features fused by **UPerHead** (`modules/model/uper.py`). On the `modelB-wtconv` branch each block's depthwise convolution is `modules/model/wtconv.py:WTConv2d` instead of the 7×7 conv; it is shape-preserving, so everything in this section and below is unchanged. Light-axis transformer blocks are stacked at counts `[0,1,2,4]` per scale. The output is a *Global Light-aware Context* (GLC) feature map at ¼ resolution. For high-res inputs (or `--scalable` inference), images are decomposed into G×G sub-tensors via `modules/model/decompose_tensors.py`, processed independently, then recomposed; a downsized-image feature map is added back to promote inter-tile interaction.

### 2. Aggregation — `modules/model/model.py` → `GLC_Aggregation`
A `CommunicationBlock` transformer (`modules/model/transformer.py`) performs **cross-image attention**: at each sampled pixel, K image features attend to each other along the light axis. PMA collapses K → 1 (paper's pixel-sampling Transformer). No positional embeddings on samples (paper Sec. 3).

### 3. Regression — `modules/model/model.py` → `Regressor`
At the full decoder resolution (up to 4096×4096 in inference, 512 by default in training), per-pixel prediction. A spatial-axis communication transformer refines features across pixels in a sample set, then a `384→192→3` MLP predicts the unit normal.

### Inference orchestration — `modules/builder/builder.py`
`Builder` loads the checkpoint, wraps `Net`, and runs the pixel-sampling loop in `--pixel_samples` chunks across all valid pixels. In scalable mode it handles patch decomposition at the encoder level and Gaussian feature smoothing (`modules/utils/gauss_filter.py`) at patch boundaries.

**GLC smoothing is gated on P > 4, as the paper specifies — a deliberate
deviation from upstream's released code.** The paper's scale-invariant encoder
says *"Optionally, when P is larger than 4, we apply depth-wise Gaussian
filtering ... to the feature maps to further enhance the interaction"*, where
`P = decoder_resolution / canonical_resolution` is the mosaic factor.
Upstream applied it **unconditionally** (`self.glc_smoothing = True`, no P
test), so at the training configuration (512/256 ⇒ P=2) a 21×21 σ=1 depthwise
blur landed on the GLC that every sampled pixel is `grid_sample`d from — σ=1 at
quarter resolution is ≈4 px at 512 — to suppress sub-tensor block artifacts
that barely exist at 4 sub-tensors. It cost detail and bought nothing, which is
squarely in the path of what Models B and C are meant to improve.

The **kernel size stays at upstream's `10*P+1`**, not the paper's `P−1`. With σ
hard-coded to 1, 99.95% of the kernel's mass sits inside the central 7×7 at any
declared size (a 21×21 and a 7×7 differ by <1e-4 per tap), so the two formulas
are numerically equivalent and differ only in compute; keeping upstream's value
leaves the P > 4 path byte-identical to the released model. (`P−1` is also
*even* for odd P, and `padding = kernel_size//2` with an even kernel shifts the
output size by one.)

Blast radius: for any inference at **P ≤ 4** — decoder resolution ≤ 1024, which
includes DiLiGenT via `eval_diligent.py` and every `--scalable` patch
(`patch_size = 512`) — the filter no longer fires. Numbers produced for
**existing** checkpoints before this change are therefore not comparable to
numbers produced after it. Re-run the full ablation rather than mixing the two.

### Training orchestration — `modules/builder/trainer.py`
`Trainer` builds `Net`, `AdamW`, the LR scheduler (step decay or cosine, both with epoch-based warmup), and `GradScaler` (AMP only for fp16). `Net.forward(..., training=True)` samples exactly `pixel_samples` valid-mask pixels per batch element with gradients on (no `.detach()`), and returns flat per-pixel predictions plus the sampled flat indices. The loss (`modules/loss/losses.py:normal_loss`) gathers GT at those indices and computes masked MSE on normals (Sec. 4). Each save also writes `normal.pytmodel` so checkpoints are drop-in for inference.

`train_step` runs one **micro**-batch: it zeroes gradients only at the start of an accumulation cycle, divides the loss by `--accum_steps` before `backward()` (so the accumulated gradient is the *mean* over the effective batch), and optimizes on every `accum_steps`-th call, flagging that in `log['stepped']`. Clipping therefore sees the complete effective-batch gradient, which is what `--grad_clip 1.0` is calibrated against. `Trainer.flush_accum()` applies a short trailing cycle at each epoch boundary so the last micro-batches are not discarded by the next `zero_grad`; `steps_per_epoch` is `ceil(len(train_loader) / accum_steps)` to match.

### Data loading
- Inference: `modules/io/dataloader/realdata.py` — auto bounding-box cropping, square aspect ratio, mean-luminance normalization, optional masking, GT MAE (`modules/utils/compute_mae.py`).
- Training (thesis mix): `modules/io/dataloader/hdlong.py` (per-camera Dirichlet light mixing for hdlong-complexv1), `modules/io/dataloader/polarps.py` (random K of 32 S0 lights), unified by `modules/io/dataloader/mixed.py:MixedTrainDataset`. `modules/io/dataloader/scene_check.py` validates the pool before the split, owns the shared "usable unit" helpers the loaders draw from, and owns multi-root keying (`as_roots` / `build_roots`). Every hdlong-structured dataset — hdlong-complexv1 and MerlMix alike — goes through `hdlong.py`; MerlMix is an extra `--hdlong_dir` root, not a new module. The per-image normalization statistics, the U[mean, max] draw taken from them, and the PolarPS-only sidecar cache all live in `modules/io/dataloader/scale_cache.py`. `mixed.py:build_mixed_split` performs the deterministic, per-source-proportional scene-level train/val/test split and returns all three disjoint datasets in one pass (same seed ⇒ identical, disjoint splits) and `train.py` wraps the two held-out splits for evaluation. K is `--k_per_scene` (default 10) per scene. (`modules/io/dataio.py` is upstream's *inference* loader, untouched — it has no train/val/test accessors.)
- Held-out val and test: both wrapped by `mixed.py:MixedEvalDataset` (length `n_scenes × n_trials`, per-index-seeded so renders are worker-count-independent and reproducible). Val uses `n_trials=1` and runs every `--val_every_epochs`; test uses `--test_trials` and runs **once after the final epoch** via `train.py:run_test_eval`. Both sweep `train.py:run_eval_pass` → `Trainer.val_step` (scene-count-weighted mean, non-finite dropped). `--test_fraction 0` builds no test loader (`--val_fraction` must be > 0, so the val loader always exists); `sdm_unips/eval_diligent.py` is the external benchmark used when the test split is off.

## Environment

- Python 3.11, PyTorch 2.0, CUDA 11.8
- Dependencies pinned in `requirements.txt` (`torch`, `numpy`, `opencv-python`, `einops`).
- Tested on NVIDIA RTX A6000 (48 GB VRAM); CPU fallback supported for inference. Training expects a CUDA GPU.
- Platforms: Ubuntu 20.04.5 (WSL2) and Windows 11
