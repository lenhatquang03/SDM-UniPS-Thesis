# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDM-UniPS is a **CVPR 2023 Highlight** paper implementation for **Universal Photometric Stereo** — recovering surface normal maps from multiple images captured under arbitrary, spatially-varying lighting with a fixed camera. The upstream repository is inference-only; this fork adds a training pipeline (`sdm_unips/train.py`, `modules/builder/trainer.py`, `modules/io/dataloader/`, `modules/loss/`) plus a DiLiGenT K-sweep evaluation hook.

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

**Model A (thesis recipe — hdlong-complexv1 + PolarPS mix, DiLiGenT eval):**
```bash
python sdm_unips/train.py \
  --session_name modelA_full \
  --hdlong_dir /path/to/hdlong-complexv1 \
  --polarps_dir /path/to/PolarPS \
  --eval_dir   /path/to/DiLiGenT/pmsData \
  --max_scenes 8000 --dataset_backend mixed \
  --train_resolution 512 --canonical_resolution 256 \
  --batch_size 8 --pixel_samples 2048 \
  --min_image_num 3 --max_image_num 6 \
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
  --eval_dir   /kaggle/input/diligent/pmsData \
  --max_scenes 8000 --amp_dtype bf16
```

**Legacy PS-Mix path** (single-scene synthetic loader): use `--dataset_backend synthetic --train_dir /path/to/PS-Mix`. Step decay ×0.8 every 10 epochs is available via `--lr_schedule step`.

**Key training flags** (defaults shown are thesis values):
- `--dataset_backend`: `mixed` (hdlong + PolarPS) or `synthetic` (legacy PS-Mix)
- `--hdlong_dir`, `--polarps_dir`: roots for the two mixed-backend sources (either or both)
- `--max_scenes`: cap on the training set size (thesis uses 8000)
- `--batch_size`: 8 — `--pixel_samples`: m=2048
- `--train_resolution`: 512 — `--canonical_resolution`: 256
- `--min_image_num` / `--max_image_num`: 3 / 6 (K is sampled uniformly per batch)
- `--lr`: 1e-4 — `--weight_decay`: 0.05
- `--lr_schedule`: `cosine` (1e-4 → 1e-4·`min_lr_ratio` over the full run) or `step`
- `--warmup_epochs`: 5.0 (linear)
- `--amp_dtype`: `bf16` (recommended on H100), `fp16` (legacy), or `none`
- `--grad_clip`: 1.0
- `--keep_last`: 3 (epoch checkpoints to retain; `best.pt` is kept separately)
- `--resume`: warm-start from a `*.pytmodel` or `*.pt` checkpoint dir
- `--smoke_test` + `--smoke_epochs`: short dry-run for Kaggle

**Evaluation flags (DiLiGenT K-sweep, end of every epoch when `--eval_dir` is set):**
- `--eval_dir`: DiLiGenT `pmsData` root (10 `*PNG` scene directories)
- `--eval_K_list`: comma-separated K values (default `2,4,8,16,32,64,96`)
- `--eval_trials`: random subsets per (scene, K) (default 10)
- `--eval_side`: center-crop side (default 512; DiLiGenT is 612×512)
- `--eval_best_K`: the K used to decide the best-by-MAE checkpoint (default 16)

Checkpoints are written to `<session>/checkpoints/`. Each save also drops a `normal.pytmodel` copy that the inference `Builder` can load directly. The most recent `--keep_last` `epoch_*.pt` files plus `best.pt` are retained; older epoch checkpoints are pruned.

Logs land in `<session>/logs/`:
- `train.jsonl` / `train.log`  — per-step loss, MAE, LR, grad norm, step seconds
- `eval.jsonl` / `eval.log`    — per-scene, per-K, per-trial DiLiGenT MAE plus per-K summaries
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

**Training layout (legacy `synthetic` backend, one directory per scene, suffix `--train_ext`, default `.data`):**
```
TRAIN_DATA_PATH/
└── SCENE_NAME.data/
    ├── L_*.png          # multi-light renders (16-bit recommended; PS-Mix is 512×512)
    ├── normal.png       # required: GT surface normal, RGB-encoded as (n+1)/2
    └── mask.png         # optional foreground mask (derived from normal magnitude if absent)
```

**Training layout (thesis `mixed` backend):**
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

**Evaluation layout (`DiLiGenT/pmsData/`):**
```
DILIGENT_ROOT/
└── <obj>PNG/
    ├── filenames.txt         # 96 lines, one image filename each
    ├── light_intensities.txt # 96 RGB triples, one per line
    ├── light_directions.txt  # 96 unit-vector triples (unused — SDM-UniPS is uncalibrated)
    ├── mask.png
    ├── Normal_gt.png         # uint8, encoded as (n+1)/2
    └── 001.png .. 096.png    # uint16 16-bit observations (612×512)
```
The eval loader center-crops to 512×512 and divides each image by its per-channel light intensity before per-image max-luminance normalization.

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
- Training (legacy PS-Mix): `modules/io/dataloader/synthetic.py` (wrapped by `modules/io/dataio.py:TrainDataio`) — square mask-bbox crop with jitter, resize to `--train_resolution`, random horizontal flip (with normal x-component sign flip), random K image subset, paper-spec per-image normalization.
- Training (thesis mix): `modules/io/dataloader/hdlong.py` (per-camera Dirichlet light mixing for hdlong-complexv1), `modules/io/dataloader/polarps.py` (random K of 32 S0 lights), unified by `modules/io/dataloader/mixed.py:MixedTrainDataset`. Selected via `--dataset_backend mixed` and built by `modules/io/dataio.py:build_train_dataset`.
- DiLiGenT eval: `modules/io/dataloader/diligent.py:DiligentLoader` + `modules/io/dataio.py:DiligentEvalDataset` — 1 scene preload, then per-call random K subsets for the K-sweep MAE evaluation invoked at the end of every epoch from `train.py:run_diligent_eval`.

## Environment

- Python 3.11, PyTorch 2.0, CUDA 11.8
- Dependencies pinned in `requirements.txt` (`torch`, `numpy`, `opencv-python`, `einops`).
- Tested on NVIDIA RTX A6000 (48 GB VRAM); CPU fallback supported for inference. Training expects a CUDA GPU.
- Platforms: Ubuntu 20.04.5 (WSL2) and Windows 11
