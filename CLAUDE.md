# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDM-UniPS is a **CVPR 2023 Highlight** paper implementation for **Universal Photometric Stereo** — recovering surface normal maps from multiple images captured under arbitrary, spatially-varying lighting with a fixed camera. The upstream repository is inference-only; this fork adds a training pipeline (`sdm_unips/train.py`, `modules/builder/trainer.py`, `modules/io/dataloader/`, `modules/loss/`) with train/val/test all drawn from the same synthetic mixed pool.

**Scope:** this fork targets **surface-normal prediction only**. The upstream BRDF heads (baseColor / roughness / metallic from Appendix C of the paper) and novel-view relighting (`relighting.py`, `modules/utils/render.py`) have been removed.

The current branch (`architecture/training-pipeline`) targets **Model A** (the baseline SDM-UniPS architecture, no modifications), trained on a mix of `hdlong-complexv1` and `PolarPS` for the author's thesis: *"Optimizing Universal Photometric Stereo via Wavelet-Energy Saliency and Latent Carrier Tokens."* Models B and C (WTConv, WESS, latent-carrier deformable attention) will live on dedicated branches once Model A's smoke test is green.

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
  --lr_schedule cosine --warmup_epochs 5 --min_lr_ratio 0.01 \
  --grad_clip 1.0 --amp_dtype bf16
```

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
must supply at least 3 scenes (one per split) or training aborts with an
informative error (and training also aborts if the val or test split comes
out empty). `best.pt` is selected by **held-out validation loss** at the end
of every `--val_every_epochs` epochs. The **held-out test split** is evaluated
**exactly once, after the final epoch** via the same `val_step` path, on the
weights loaded from `best.pt` (not the last-epoch weights, so late overfitting
or early stopping never biases the headline number), and it never influences
checkpoint selection. There is no external benchmark; DiLiGenT is no longer
used.

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

**Early stopping (safety cutoff):** `--patience` (default 10, in units of
validation checks; 0 disables) stops training once val loss has not improved
by more than `--min_delta` (default 0.0) for that many consecutive checks. The
cosine LR schedule still spans the full `--epochs`; patience only trims the
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

**Reproducibility:** a run is reproducible for a fixed config — all RNGs are
seeded from `--seed` (python/numpy/torch/cuda), cuDNN is pinned to
deterministic kernels, and the DataLoaders seed their workers' numpy RNG
(`worker_init_fn`) with a seeded shuffle `generator`. The test evaluation is
strongly reproducible: `MixedEvalDataset` seeds each scene's random
camera/lights from its flat index (independent of worker count), and the
final eval re-seeds torch so the model's pixel sampling is fixed too. To cut
the variance of the random K-image draw, each test scene is rendered
`--test_trials` times (default 3), each trial a different fixed K-image draw,
and their MAE is averaged with a scene-count-weighted (unbiased) mean.

**Key training flags** (defaults shown are thesis values):
- `--hdlong_dir`, `--polarps_dir`: roots for the two mixed sources (either or both)
- `--train_dir`: auto-detect root (scenes classified into hdlong/polarps by on-disk markers)
- `--max_scenes`: cap on the **combined** scene pool before the split (thesis uses 8000)
- `--val_fraction`: held-out fraction for validation (default 0.1)
- `--test_fraction`: held-out fraction for the final test report (default 0.1)
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
- K is fixed at 10 per scene inside `HdlongLoader` / `PolarPSLoader`.
- `--lr`: 1e-4 — `--weight_decay`: 0.05
- `--lr_schedule`: `cosine` (1e-4 → 1e-4·`min_lr_ratio` over the full run) or `step`
- `--warmup_epochs`: 5.0 (linear)
- `--amp_dtype`: `bf16` (recommended on H100), `fp16` (legacy), or `none`
- `--grad_clip`: 1.0
- `--max_consecutive_skips`: 20 — `--max_skip_rate`: 0.3 over
  `--skip_rate_window`: 200 (either 0 disables that guard) — see below
- `--keep_last`: 3 (epoch checkpoints to retain; `best.pt` is kept separately)
- `--resume`: warm-start from a `*.pytmodel` or `*.pt` checkpoint dir
- `--smoke_test` + `--smoke_epochs`: short dry-run for Kaggle

The final test evaluation runs automatically after the last epoch on the
held-out `--test_fraction` split, averaged over `--test_trials` deterministic
trials.

Checkpoints are written to `<session>/checkpoints/`. Each save also drops a bare-`state_dict` `normal.pytmodel` copy (overwritten every save, so it tracks the *latest* weights); each `best.pt` update additionally copies it to `best_normal.pytmodel`. The most recent `--keep_last` `epoch_*.pt` files plus `best.pt` are retained; older epoch checkpoints are pruned (`final.pt` and `step_*.pt` are never auto-pruned).

After the final epoch, `train.py:export_for_inference` publishes the weights the test number was reported on to `<session>/checkpoints/normal/normal.pytmodel` — `best_normal.pytmodel` when a best was recorded, else `normal.pytmodel` (the final-epoch weights). Inference then runs directly against the training output:

```bash
python sdm_unips/main.py --session_name SESSION --test_dir DATA \
    --checkpoint <session>/checkpoints
```

The dedicated `normal/` subdirectory is required because `builder.load_models` globs `*.pytmodel` and `"".join`s the matches — the directory it points at must hold exactly one file, which `<session>/checkpoints/` itself does not.

Logs land in `<session>/logs/`:
- `train.jsonl` / `train.log`  — per-step loss, MAE, LR, grad norm, step seconds; per-epoch and per-validation summaries
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

**Training layout (hdlong-complexv1 + PolarPS):**
```
HDLONG_ROOT/
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
The dataset class auto-detects scene type via these markers (`light_means.config` ⇒ hdlong, `normal.exr` ⇒ PolarPS). It synthesizes each training render via a Dirichlet (α, β, γ) mix of one randomly chosen point/dir/env light triple (hdlong) or by drawing one of 32 `S0.exr` images (PolarPS). hdlong is upsampled from 256×256 to `--train_resolution`.

**Per-image normalization:** each observation is divided by a single scalar — the **max, over its foreground pixels, of that pixel's mean across the three colour channels**. This is deliberately the same statistic `realdata.py` uses at inference (`temp = mean over channels; mx = max over pixels; I /= mx`), so observations land in ~[0, 1] in both paths. That matters in two places the architecture is built around: `Net.forward` concatenates the 0/1 mask as a 4th input channel (a differently-scaled RGB drowns it out), and `Net._decode_pixels` concatenates raw observations with the 256-d GLC features before an `ln=True` attention block (a large-magnitude observation channel dominates the LayerNorm statistics and flattens the GLC signal).

PolarPS caches the scalar in an `image_scales.config` sidecar per scene — its observations are fixed `S0.exr` files, so the value depends only on file + mask and is computed once, at the on-disk resolution so it survives a change of `--train_resolution`. **hdlong does not cache**: its observation is a random Dirichlet mix drawn afresh each epoch, and unlike the mean, the max is not linear (`max(Σ wᵢxᵢ) ≠ Σ wᵢ max(xᵢ)` — empirically ~12% off), so there is no per-component quantity to recombine; the scale is measured directly on each composite pre-resize. Read-only dataset mounts (e.g. Kaggle inputs) simply skip the sidecar write and keep an in-process cache.

> Was mean-normalization before 2026-08-05. The sidecar was renamed from `image_means.config` so stale caches can't be reused as if they held maxima — but **delete any `image_means.config` files already written into the dataset trees**, they are now dead weight.

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
- Training (thesis mix): `modules/io/dataloader/hdlong.py` (per-camera Dirichlet light mixing for hdlong-complexv1), `modules/io/dataloader/polarps.py` (random K of 32 S0 lights), unified by `modules/io/dataloader/mixed.py:MixedTrainDataset`. The per-image normalization scalar and its PolarPS-only sidecar cache live in `modules/io/dataloader/scale_cache.py`. `mixed.py:build_mixed_split` performs the deterministic, per-source-proportional scene-level train/val/test split and returns all three disjoint datasets in one pass (same seed ⇒ identical, disjoint splits) and `train.py` wraps the two held-out splits for evaluation. K is fixed at 10 per scene. (`modules/io/dataio.py` is upstream's *inference* loader, untouched — it has no train/val/test accessors.)
- Held-out val and test: both wrapped by `mixed.py:MixedEvalDataset` (length `n_scenes × n_trials`, per-index-seeded so renders are worker-count-independent and reproducible). Val uses `n_trials=1` and runs every `--val_every_epochs`; test uses `--test_trials` and runs **once after the final epoch** via `train.py:run_test_eval`. Both sweep `train.py:run_eval_pass` → `Trainer.val_step` (scene-count-weighted mean, non-finite dropped). No external benchmark is involved.

## Environment

- Python 3.11, PyTorch 2.0, CUDA 11.8
- Dependencies pinned in `requirements.txt` (`torch`, `numpy`, `opencv-python`, `einops`).
- Tested on NVIDIA RTX A6000 (48 GB VRAM); CPU fallback supported for inference. Training expects a CUDA GPU.
- Platforms: Ubuntu 20.04.5 (WSL2) and Windows 11
