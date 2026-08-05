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

Test and validation report the **same quantities** (identical `val_step`
forward + loss path) but **aggregate them differently**, so treat them as
close-but-not-identical estimators rather than exactly comparable numbers:
`run_test_eval` takes a scene-count-weighted mean over batches and drops any
non-finite value, while the per-epoch val loop takes a plain per-batch
`np.nanmean` (short final batch over-weighted; NaN dropped but `±inf`
propagates). The two coincide exactly only when the split size is a multiple
of `--batch_size`. Neither gap affects `best.pt` selection — the val loop's
bias is a fixed reweighting of a fixed batch partition, hence consistent
across epochs.

**Early stopping (safety cutoff):** `--patience` (default 10, in units of
validation checks; 0 disables) stops training once val loss has not improved
by more than `--min_delta` (default 0.0) for that many consecutive checks. The
cosine LR schedule still spans the full `--epochs`; patience only trims the
unproductive tail, and `best.pt` already holds the best-val weights, so
stopping early never costs the deliverable.

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
- `--batch_size`: 8 — `--pixel_samples`: m=2048
- `--train_resolution`: 512 — `--canonical_resolution`: 256
- K is fixed at 10 per scene inside `HdlongLoader` / `PolarPSLoader`.
- `--lr`: 1e-4 — `--weight_decay`: 0.05
- `--lr_schedule`: `cosine` (1e-4 → 1e-4·`min_lr_ratio` over the full run) or `step`
- `--warmup_epochs`: 5.0 (linear)
- `--amp_dtype`: `bf16` (recommended on H100), `fp16` (legacy), or `none`
- `--grad_clip`: 1.0
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

**Per-image normalization:** each observation is divided by a single scalar — the mean intensity over its foreground pixels across all three colour channels. That scalar depends only on the source file + mask, so it is computed once and cached in an `image_means.config` sidecar next to the images (per camera for hdlong, per scene for PolarPS); the same value is reused every epoch and for both train and val/test. hdlong's composite scalar is the Dirichlet-weighted sum of its three cached component means (mean is linear over a shared mask). Read-only dataset mounts (e.g. Kaggle inputs) simply skip the sidecar write and keep an in-process cache.

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

### Data loading
- Inference: `modules/io/dataloader/realdata.py` — auto bounding-box cropping, square aspect ratio, mean-luminance normalization, optional masking, GT MAE (`modules/utils/compute_mae.py`).
- Training (thesis mix): `modules/io/dataloader/hdlong.py` (per-camera Dirichlet light mixing for hdlong-complexv1), `modules/io/dataloader/polarps.py` (random K of 32 S0 lights), unified by `modules/io/dataloader/mixed.py:MixedTrainDataset`. The per-image mean cache lives in `modules/io/dataloader/mean_cache.py`. `mixed.py:build_mixed_split` performs the deterministic, per-source-proportional scene-level train/val/test split and returns all three disjoint datasets in one pass (same seed ⇒ identical, disjoint splits); `dataio.py:build_train_dataset` / `build_val_dataset` / `build_test_dataset` are thin single-split accessors. K is fixed at 10 per scene.
- Held-out test: the test third of the same mixed split, wrapped by `mixed.py:MixedEvalDataset` (length `n_scenes × --test_trials`, per-index-seeded so it is worker-count-independent and reproducible) and evaluated **once after the final epoch** via `train.py:run_test_eval` (reuses `Trainer.val_step`; scene-count-weighted mean over trials). No external benchmark is involved.

## Environment

- Python 3.11, PyTorch 2.0, CUDA 11.8
- Dependencies pinned in `requirements.txt` (`torch`, `numpy`, `opencv-python`, `einops`).
- Tested on NVIDIA RTX A6000 (48 GB VRAM); CPU fallback supported for inference. Training expects a CUDA GPU.
- Platforms: Ubuntu 20.04.5 (WSL2) and Windows 11
