"""
Training entry point for SDM-UniPS (Model A baseline).

Quickstart for the thesis recipe (hdlong-complexv1 + PolarPS):

    python sdm_unips/train.py \
        --session_name modelA_smoke \
        --hdlong_dir /kaggle/input/hdlong-complexv1 \
        --polarps_dir /kaggle/input/polarps \
        --max_scenes 8000 --val_fraction 0.1 --test_fraction 0.1 \
        --train_resolution 512 --canonical_resolution 256 \
        --batch_size 8 --pixel_samples 2048 \
        --lr 1e-4 --weight_decay 0.05 --lr_schedule cosine \
        --warmup_epochs 5 --min_lr_ratio 0.01 --grad_clip 1.0 \
        --amp_dtype bf16 --smoke_test

`--smoke_test` clamps the run to `--smoke_epochs` (default 10) for a quick
Kaggle dry-run before committing to the full 60-epoch schedule.

Train, validation, and test are all disjoint scene-level slices of the same
mixed (hdlong + PolarPS) pool. Checkpoint selection uses the held-out
validation split; the held-out test split is evaluated exactly once, after
the final epoch, so it never influences model selection.

Logging
-------
Per-step JSONL lines are appended to `<session>/logs/train.jsonl` and a
human-readable mirror to `<session>/logs/train.log`. The single final
held-out test evaluation lands in `<session>/logs/eval.jsonl`.

`--log_memory` (off by default) adds resource fields to those records —
`mem_peak_gib`, `mem_reserved_peak_gib`, `host_rss_gib`, `data_wait_sec`, and
a per-epoch `data_wait_share`. They exist to size `--batch_size` and
`--num_workers` from measurement rather than guesswork: peak memory says how
much VRAM headroom is left, and `data_wait_share` says whether the GPU is
starved by the loader (raise workers) or already saturated (do not).

Fitting a large effective batch on a small GPU
----------------------------------------------
`--batch_size` is the micro-batch; `--accum_steps` micro-batches make one
optimizer step. `--batch_size 2 --accum_steps 4` trains the same effective
batch of 8 as `--batch_size 8`, at roughly a quarter of the activation
memory. `--max_vram_gib` caps this process's VRAM so an over-large batch
fails fast instead of exhausting a card that may also be driving a display.
"""

from __future__ import print_function, division

import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.builder.trainer import Trainer, prune_checkpoints
from modules.io.dataloader.mixed import build_mixed_split, MixedEvalDataset


def _seed_worker(worker_id):
    """Seed a DataLoader worker's RNGs and stop OpenCV from oversubscribing.

    Seeding: PyTorch gives each worker a unique base seed derived from the main
    generator and uses it to seed `torch` and python `random`, but it does
    NOT seed numpy — the classic gotcha that makes np.random-based loaders
    non-reproducible (and duplicated across workers). Mirroring
    `torch.initial_seed()` into numpy/random fixes both.

    Threading: `cv2` defaults to one thread per core for `imread`/`resize`, so
    W workers each spawn ~ncore threads and the loader thrashes the CPU
    (W x ncore threads over ncore cores). One thread per worker is the right
    policy when parallelism already comes from the worker pool — without this,
    raising `--num_workers` can make throughput *worse*, and any
    num_workers benchmark is measuring contention rather than the loader.
    """
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Resource instrumentation (opt-in via --log_memory)
# ---------------------------------------------------------------------------
def _gpu_mem_stats(device) -> dict:
    """Peak/current CUDA memory in GiB. Counter reads only — no device sync.

    `alloc` is what tensors hold right now; `peak` is the high-water mark since
    the last `reset_peak_memory_stats` (this script resets per epoch), and is
    the number to compare against the VRAM budget when sizing `--batch_size`.
    `reserved_peak` is what the caching allocator took from the driver — the
    gap between it and `peak` is fragmentation.
    """
    if device.type != 'cuda':
        return {}
    gib = 1024 ** 3
    return {
        'mem_alloc_gib': torch.cuda.memory_allocated() / gib,
        'mem_peak_gib': torch.cuda.max_memory_allocated() / gib,
        'mem_reserved_peak_gib': torch.cuda.max_memory_reserved() / gib,
    }


def _host_rss_gib() -> float:
    """Resident host memory of this process plus its DataLoader workers (GiB).

    Sums `VmRSS` over self + child PIDs found via `/proc/self/task/*/children`.
    Shared copy-on-write pages are counted once per process, so this is an
    upper bound on true footprint — useful as a ceiling check against
    available RAM and `/dev/shm`, not as an exact figure. Returns 0.0 where
    /proc is unavailable (non-Linux).
    """
    def rss_kib(pid) -> int:
        try:
            with open(f'/proc/{pid}/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
        return 0

    total_kib = rss_kib('self')
    try:
        for task in os.listdir('/proc/self/task'):
            with open(f'/proc/self/task/{task}/children') as f:
                for child in f.read().split():
                    total_kib += rss_kib(child)
    except OSError:
        pass
    return total_kib / (1024 ** 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()

    # I/O ----------------------------------------------------------------
    p.add_argument('--session_name', default='train_run')
    p.add_argument('--hdlong_dir', default=None,
                   help='Root of hdlong-complexv1 (scenes have light_means.config)')
    p.add_argument('--polarps_dir', default=None,
                   help='Root of PolarPS (scenes have normal.exr)')
    p.add_argument('--train_dir', default=None,
                   help='Auto-detect root: scenes are classified into hdlong/polarps '
                        'by their on-disk markers.')
    p.add_argument('--val_fraction', type=float, default=0.1,
                   help='Held-out fraction of the mixed pool for validation '
                        '(scene-level split, split proportionally per source).')
    p.add_argument('--test_fraction', type=float, default=0.1,
                   help='Held-out fraction of the mixed pool for the final '
                        'test report (scene-level, proportional per source).')
    p.add_argument('--test_trials', type=int, default=3,
                   help='Number of deterministic trials (independent K-image '
                        'draws) per test scene; their MAE is averaged for an '
                        'unbiased, low-variance test number (keep small, e.g. 3).')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Defaults to <session_name>/checkpoints')
    p.add_argument('--log_dir', default=None,
                   help='Defaults to <session_name>/logs')
    p.add_argument('--resume', default=None)
    p.add_argument('--log_every', type=int, default=10,
                   help='Optimizer steps between log records')
    p.add_argument('--log_memory', action='store_true',
                   help='Add resource-usage fields to every log record: peak '
                        'CUDA memory, allocator reservation, host RSS, and the '
                        'seconds each step spent WAITING on the DataLoader. '
                        'Off by default to keep train.jsonl lean; turn it on '
                        'when tuning --batch_size / --num_workers.')
    p.add_argument('--ckpt_every', type=int, default=0,
                   help='Step interval for extra checkpoints. 0 disables.')
    p.add_argument('--val_every_epochs', type=int, default=1)
    p.add_argument('--patience', type=int, default=10,
                   help='Early-stopping patience, counted in validation checks: '
                        'stop once val loss has not improved for this many '
                        'consecutive checks. 0 disables (train the full '
                        '--epochs). The cosine LR schedule still spans --epochs; '
                        'this is only a safety cutoff, and best.pt keeps the '
                        'best-val weights regardless.')
    p.add_argument('--min_delta', type=float, default=0.0,
                   help='Minimum decrease in val loss to count as an improvement '
                        'for early stopping (default 0.0 = any improvement).')
    p.add_argument('--keep_last', type=int, default=3,
                   help='Number of recent epoch checkpoints to keep on disk')

    # Network ------------------------------------------------------------
    p.add_argument('--canonical_resolution', type=int, default=256)
    p.add_argument('--pixel_samples', type=int, default=2048)

    # Data ---------------------------------------------------------------
    # K is fixed at 10 per scene inside HdlongLoader / PolarPSLoader.
    p.add_argument('--train_resolution', type=int, default=512)
    p.add_argument('--max_scenes', type=int, default=8000,
                   help='Cap the training set size. 0 = no cap.')

    # Optimization (thesis recipe defaults) ------------------------------
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=8,
                   help='MICRO-batch size (scenes per forward pass). The '
                        'effective batch is --batch_size x --accum_steps.')
    p.add_argument('--accum_steps', type=int, default=1,
                   help='Gradient accumulation: number of micro-batches per '
                        'optimizer step. Use it to keep the recipe effective '
                        'batch of 8 on a GPU that cannot hold it in one pass '
                        '(e.g. --batch_size 2 --accum_steps 4). The LR '
                        'schedule counts optimizer steps, so it is unchanged.')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--prefetch_factor', type=int, default=2,
                   help='Batches each worker prefetches. In-flight host memory '
                        'is roughly num_workers x prefetch_factor x batch_size '
                        'x 36 MB (K=10 at 512x512), and it passes through '
                        '/dev/shm — raise only if data-wait is nonzero AND '
                        'shm has room. Ignored when --num_workers 0.')
    p.add_argument('--max_vram_gib', type=float, default=0.0,
                   help='Hard cap on CUDA memory for this process (0 = no '
                        'cap). Allocations beyond it raise OOM instead of '
                        'consuming the whole card — worth setting on a GPU '
                        'that also drives a display, where an uncapped run can '
                        'take down the desktop session.')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=float, default=5.0)
    p.add_argument('--lr_schedule', default='cosine', choices=['cosine', 'step'])
    p.add_argument('--min_lr_ratio', type=float, default=0.01)
    p.add_argument('--lr_decay_every', type=int, default=10)
    p.add_argument('--lr_decay_gamma', type=float, default=0.8)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--amp', action='store_true',
                   help='Legacy alias: equivalent to --amp_dtype fp16')
    p.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp16', 'none'])
    p.add_argument('--detect_anomaly', action='store_true',
                   help='Wrap each train step in torch.autograd.detect_anomaly so '
                        'a NaN/Inf in forward or backward raises at the exact op '
                        '(with the forward stack trace). Slow; use for debugging.')
    p.add_argument('--seed', type=int, default=42)

    # Smoke test ---------------------------------------------------------
    p.add_argument('--smoke_test', action='store_true',
                   help='Run for --smoke_epochs to sanity-check the pipeline.')
    p.add_argument('--smoke_epochs', type=int, default=10)
    p.add_argument('--smoke_max_scenes', type=int, default=0,
                   help='If >0 and --smoke_test, override --max_scenes to this value.')
    return p


# ---------------------------------------------------------------------------
# Collation + helpers
# ---------------------------------------------------------------------------
def _collate(batch):
    """Stack a list of 4-tuples (I, N, M, n_imgs) produced by training datasets."""
    I = torch.from_numpy(np.stack([b[0] for b in batch], 0))
    N = torch.from_numpy(np.stack([b[1] for b in batch], 0))
    M = torch.from_numpy(np.stack([b[2] for b in batch], 0))
    n_imgs = torch.tensor([int(b[3]) for b in batch], dtype=torch.long)
    return I, N, M, n_imgs


def _to_scalar(v) -> float:
    """"Convert a tensor to a float"""
    if torch.is_tensor(v):
        return float(v.detach().cpu().item())
    return float(v)


def _fmt_value(v: float) -> str:
    """
    If v >= 0.001, use the 4-decimal format
    else, use scientific notation (e.g, 5.00e-0.5)
    """
    a = abs(v)
    if a == 0:
        return f'{v:.4f}'
    if a >= 1e-3:
        return f'{v:.4f}'
    return f'{v:.2e}'


def _format_log(log: dict) -> str:
    """
    Return a presentable format for the log records
    e.g,  {"loss": 0.0412, "mae_deg": 11.8734, "lr": 4.1e-05, "grad_norm": 2.3145,
   "step_sec": 0.9123, "epoch": 3, "it": 40, "global_step": 1240}
    """
    return ' | '.join(f'{k}={_fmt_value(_to_scalar(v))}'
                     for k, v in log.items()
                     if isinstance(v, (int, float)) or torch.is_tensor(v))


class JSONLLogger:
    """Append-only JSONL + plain-text mirror logger."""

    def __init__(self, log_dir, filename_stem):
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, f'{filename_stem}.jsonl')
        self.text_path = os.path.join(log_dir, f'{filename_stem}.log')

    def write(self, record, text=None):
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        if text is not None:
            with open(self.text_path, 'a') as f:
                f.write(text + '\n')


# ---------------------------------------------------------------------------
# Held-out test evaluation
# ---------------------------------------------------------------------------
def run_test_eval(
        trainer: Trainer, test_loader: torch.utils.data.DataLoader, 
        logger: JSONLLogger, global_step: int, n_trials: int
) -> dict:
    """Evaluate the held-out mixed test split and return its summary.

    Reuses the trainer's validation path (`val_step`), so the per-batch loss
    and MAE are the same quantities the validation loop reports. The loader is
    `MixedEvalDataset`, whose length is `n_scenes * n_trials`; a single sweep
    therefore averages every scene over `n_trials` deterministic trials.

    Note the *aggregation* differs from the validation loop, so the two
    headline numbers are close but not identical estimators:

    - here, batches are weighted by their scene count, making the mean exactly
      the per-(scene, trial) average (unbiased) even when the last batch is
      short; the val loop takes a plain per-batch mean, which over-weights a
      short final batch. The two coincide only when the split size is an exact
      multiple of `--batch_size`.
    - here, any non-finite value is dropped (`np.isfinite`); the val loop uses
      `np.nanmean`, which drops NaN but propagates +/-inf.

    Both gaps are small (sub-percent for realistic split sizes) and neither
    affects `best.pt` selection, since the val loop's bias is a fixed
    reweighting of a fixed batch partition and so is consistent across epochs.
    """
    totals = {}  # key -> [weighted_sum, weight]
    for test_batch in test_loader:
        batch_size = int(test_batch[0].shape[0])
        for k, v in trainer.val_step(test_batch).items():
            # MAE and MSE from a batch
            val = _to_scalar(v)
            if not np.isfinite(val):
                continue  # skip a non-finite batch instead of poisoning the mean
            acc = totals.setdefault(k, [0.0, 0.0])
            acc[0] += val * batch_size
            acc[1] += batch_size
    if not totals:
        return {}
    summary = {f'test_{k}': (s / w if w > 0 else float('nan'))
               for k, (s, w) in totals.items()}
    summary['kind'] = 'test_summary'
    summary['global_step'] = global_step
    summary['n_trials'] = int(n_trials)
    logger.write(summary, text='[TEST] ' + _format_log(summary))
    print('[TEST SUMMARY] ' + _format_log(summary))
    return summary


# ---------------------------------------------------------------------------
# Inference export
# ---------------------------------------------------------------------------
def export_for_inference(ckpt_dir: str) -> str | None:
    """Publish the reported weights to `<ckpt_dir>/normal/normal.pytmodel`.

    `builder.load_models` globs `*.pytmodel` under `<--checkpoint>/normal` and
    `"".join`s the matches, so that directory must contain exactly one file —
    hence the dedicated subdirectory rather than pointing inference at
    `ckpt_dir`, which holds both `normal.pytmodel` and `best_normal.pytmodel`.

    Source preference mirrors the checkpoint the final test number is reported
    on: `best_normal.pytmodel` (written alongside every `best.pt` update, so
    its presence means a best was recorded) when it exists, otherwise
    `normal.pytmodel`, which `Trainer.save(tag='final')` last overwrote with
    the final-epoch weights.

    Returns the exported path, or None if neither source exists.
    """
    best_src = os.path.join(ckpt_dir, 'best_normal.pytmodel')
    final_src = os.path.join(ckpt_dir, 'normal.pytmodel')
    if os.path.isfile(best_src):
        src, provenance = best_src, 'best_normal.pytmodel (val-selected)'
    elif os.path.isfile(final_src):
        src, provenance = final_src, 'normal.pytmodel (final-epoch fallback)'
    else:
        print(f'[EXPORT] WARNING: no *.pytmodel found in {ckpt_dir}; '
              f'skipping the inference export.')
        return None

    out_dir = os.path.join(ckpt_dir, 'normal')
    os.makedirs(out_dir, exist_ok=True)
    # Keep exactly one .pytmodel here (see docstring): drop anything a previous
    # export left behind, e.g. a best export superseded by a fallback one.
    for stale in glob.glob(os.path.join(out_dir, '*.pytmodel')):
        os.remove(stale)
    dst = os.path.join(out_dir, 'normal.pytmodel')
    shutil.copyfile(src, dst)
    print(f'[EXPORT] Inference weights = {provenance} → {dst}\n'
          f'[EXPORT] Run inference with --checkpoint {ckpt_dir}')
    return dst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()

    # Read by the caching allocator on its first allocation (which has not
    # happened yet), so setting it here still takes effect. Expandable segments
    # let the allocator grow a block instead of stranding memory in the wrong
    # size class — the usual cause of an OOM while `nvidia-smi` still shows
    # free VRAM. `setdefault` keeps an explicit env override authoritative.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # Same rationale as in `_seed_worker`, for the --num_workers 0 path where
    # loading runs in this process.
    cv2.setNumThreads(0)

    # Smoke-test configurations
    if args.smoke_test:
        args.epochs = min(args.epochs, args.smoke_epochs)
        if args.smoke_max_scenes > 0:
            args.max_scenes = args.smoke_max_scenes

    # Reproducibility: seed every RNG the pipeline touches and pin cuDNN to deterministic kernels. 
    # Combined with the DataLoader `generator` +`_seed_worker` below (numpy in workers)
    # And the deterministic MixedEvalDataset, a run is reproducible for a fixed config.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create dedicated directories for logs and checkpoints
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_dir = args.log_dir or os.path.join(args.session_name, 'logs')
    ckpt_dir = args.checkpoint_dir or os.path.join(args.session_name, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'[TRAIN] Device = {device}  Session = {args.session_name}')
    print(f'[TRAIN] Dataset = mixed (hdlong + PolarPS) | Automatic Mixed Precision (AMP) = {args.amp_dtype}  '
          f'Epochs = {args.epochs} | Batch size = {args.batch_size} '
          f'x {max(1, args.accum_steps)} accum '
          f'(effective {args.batch_size * max(1, args.accum_steps)})  K=10')
    print(f'[TRAIN] DataLoader workers = {args.num_workers} | '
          f'prefetch_factor = {args.prefetch_factor}')
    print(f'[TRAIN] Log directory = {log_dir} | Checkpoint directory = {ckpt_dir}')
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        total_gib = gpu_props.total_memory / 2**30
        print(f'[TRAIN] GPU = {torch.cuda.get_device_name(0)} | '
              f'Memory = {total_gib:.1f} GiB | '
              f'Capability = {gpu_props.major}.{gpu_props.minor}')
        if args.max_vram_gib > 0:
            # set_per_process_memory_fraction takes a fraction of TOTAL VRAM,
            # counting only this process's allocations — memory already held by
            # other processes (a desktop compositor, another job) is not
            # deducted, so leave headroom when choosing the cap.
            fraction = min(max(args.max_vram_gib / total_gib, 0.0), 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            print(f'[TRAIN] VRAM cap = {args.max_vram_gib:.1f} GiB '
                  f'({fraction:.1%} of the card); allocations past it raise OOM.')
    if args.log_memory:
        print('[TRAIN] Resource logging ENABLED (--log_memory): peak CUDA '
              'memory, host RSS, and DataLoader wait time per record.')

    # Seeded generator makes the train shuffle order reproducible; combined
    # with _seed_worker (numpy in workers) the whole loader stream is fixed.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)

    # Deterministic scene-level splits of the mixed PolarPS/hdlong-complexv1 pool 
    # produces train (augmented), held-out val (drives best.pt), and held-out test (final report).
    train_set, val_set, test_set = build_mixed_split(args)

    def _worker_kwargs(n_workers):
        """DataLoader worker options. `prefetch_factor` is only a legal
        argument when workers exist (torch raises otherwise)."""
        if n_workers <= 0:
            return {'num_workers': 0}
        return {'num_workers': n_workers,
                'prefetch_factor': max(1, args.prefetch_factor)}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'),
        drop_last=True, collate_fn=_collate,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_seed_worker, generator=loader_gen,
        **_worker_kwargs(args.num_workers),
    )

    # Validation set drives best.pt and test set is the final report, so neither may be empty.
    if len(val_set) == 0:
        raise RuntimeError(
            'Validation split is empty. --val_fraction '
            f'({args.val_fraction}) rounded every present source to 0 val '
            'scenes. Increase --val_fraction or provide more scenes per source .'
        )
    if len(test_set) == 0:
        raise RuntimeError(
            'Test split is empty. --test_fraction '
            f'({args.test_fraction}) rounded every present source to 0 test '
            'scenes. Increase --test_fraction or provide more scenes per source.'
        )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=(device.type == 'cuda'),
        drop_last=False, collate_fn=_collate,
        worker_init_fn=_seed_worker, generator=loader_gen,
        **_worker_kwargs(max(1, args.num_workers // 2)),
    )

    # Test eval: a deterministic multi-trial wrapper over the test scenes (unbiased).
    # Its per-item RNG is seeded from the index, so results are worker-count independent.
    test_eval_set = MixedEvalDataset(args, test_set.scenes,
                                     n_trials=args.test_trials, seed=args.seed)
    test_loader = torch.utils.data.DataLoader(
        test_eval_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=(device.type == 'cuda'),
        drop_last=False, collate_fn=_collate,
        worker_init_fn=_seed_worker, generator=loader_gen,
        **_worker_kwargs(max(1, args.num_workers // 2)),
    )

    # The scheduler and every *_every counter run on OPTIMIZER steps, so with
    # accumulation they must be derived from the micro-batch count, not from
    # len(train_loader). ceil, because `flush_accum` applies the short trailing
    # cycle at each epoch boundary rather than dropping it.
    accum_steps = max(1, args.accum_steps)
    micro_per_epoch = max(1, len(train_loader))
    steps_per_epoch = max(1, math.ceil(micro_per_epoch / accum_steps))
    total_steps = steps_per_epoch * args.epochs
    print(f'[TRAIN] scenes: train = {len(train_set):,} | val = {len(val_set):,} | '
          f'test = {len(test_set):,} (x{args.test_trials} trials)')
    print(f'[TRAIN] micro-batches/epoch = {micro_per_epoch:,} | '
          f'optimizer steps/epoch = {steps_per_epoch:,} | '
          f'total steps = {total_steps:,} | '
          f'effective batch = {args.batch_size * accum_steps}')

    trainer = Trainer(args, device, total_steps=total_steps,
                       steps_per_epoch=steps_per_epoch)

    train_logger = JSONLLogger(log_dir, 'train')
    eval_logger = JSONLLogger(log_dir, 'eval')

    # Store the run arguments
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump({k: v for k, v in sorted(vars(args).items())
                   if isinstance(v, (str, int, float, bool, type(None)))},
                  f, indent=2)

    best_val_loss = float('inf')
    best_path = os.path.join(ckpt_dir, 'best.pt')
    epochs_no_improve = 0   # consecutive validation checks without improvement
    stopped_early = False
    # Run-wide high-water marks, carried across the per-epoch peak resets.
    run_peak = {'mem_peak_gib': 0.0, 'mem_reserved_peak_gib': 0.0,
                'host_rss_gib': 0.0}
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        epoch_running = {}
        last_ckpt_step = -1
        # Peak memory is a high-water mark: reset it per epoch so each summary
        # reports THIS epoch's peak rather than the run's.
        if args.log_memory and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        # Wall clock at the end of the previous iteration; the gap to the top of
        # the next one is time spent blocked on the loader (see data_wait_sec).
        iter_end = time.time()
        for it, batch in enumerate(train_loader):
            data_wait = time.time() - iter_end
            step_t0 = time.time()
            log = trainer.train_step(batch)  # MSE, MAE (+ lr, grad_norm on steps)
            step_dt = time.time() - step_t0
            stepped = bool(log.pop('stepped', True))

            scalar_log = {k: _to_scalar(v) for k, v in log.items()}
            scalar_log['step_sec'] = step_dt
            scalar_log['epoch'] = epoch
            scalar_log['it'] = it
            scalar_log['global_step'] = trainer.global_step
            if args.log_memory:
                # data_wait_sec ~ 0 => the loader keeps up and more workers buy
                # nothing; a large fraction of step_sec => input-bound, so raise
                # --num_workers / --prefetch_factor or move the data off a slow
                # disk. step_sec alone cannot tell these apart.
                scalar_log['data_wait_sec'] = data_wait
                scalar_log['host_rss_gib'] = _host_rss_gib()
                scalar_log.update(_gpu_mem_stats(device))

            for k, v in scalar_log.items():
                if isinstance(v, float):
                    # MSE, MAE, lr, grad_norm (+ resource fields if enabled)
                    epoch_running.setdefault(k, []).append(v)

            # Log on optimizer-step boundaries only: mid-cycle micro-batches
            # share a global_step, so an unguarded modulo would emit a burst of
            # near-duplicate records for every logged step.
            if (stepped and trainer.global_step % args.log_every == 0) or it == 0:
                elapsed = time.time() - t0
                head = (f'[STEP {trainer.global_step}/{total_steps}] '
                        f'epoch={epoch} | it={it} | elapsed={elapsed:.1f}s')
                text = head + ' | ' + _format_log(scalar_log)
                train_logger.write(scalar_log, text=text)
                print(text)

            # `stepped` guards against re-saving the same step for every
            # micro-batch of an accumulation cycle.
            if (args.ckpt_every > 0 and stepped
                    and trainer.global_step % args.ckpt_every == 0
                    and trainer.global_step != last_ckpt_step):
                last_ckpt_step = trainer.global_step
                path = trainer.save(ckpt_dir, tag=f'step_{trainer.global_step}')
                print(f'[CHECKPOINT] Step checkpoint saved {path}')

            iter_end = time.time()

        # Apply a short trailing accumulation cycle so the epoch's last
        # micro-batches are not discarded by the next cycle's zero_grad.
        if trainer.flush_accum() is not None:
            print(f'[TRAIN] epoch {epoch}: flushed a partial accumulation cycle '
                  f'({micro_per_epoch % accum_steps} micro-batches).')

        epoch_summary = {
            'kind': 'epoch_summary',
            'epoch': epoch,
            'global_step': trainer.global_step,
            'epoch_sec': time.time() - epoch_t0,
        }
        for k, vals in epoch_running.items():
            if vals:
                # Average MAE, MSE, lr, and grad_norm over all training batches
                epoch_summary[f'avg_{k}'] = float(np.mean(vals))
        if args.log_memory:
            # Peaks, not averages: the peak is what has to fit in VRAM, and the
            # averaged `avg_mem_peak_gib` above understates it badly.
            epoch_mem = _gpu_mem_stats(device)
            epoch_mem['host_rss_gib'] = _host_rss_gib()
            for k, v in epoch_mem.items():
                epoch_summary[f'epoch_{k}'] = v
                if k in run_peak:
                    run_peak[k] = max(run_peak[k], v)
            waits = epoch_running.get('data_wait_sec', [])
            steps = epoch_running.get('step_sec', [])
            if waits and steps:
                total_wait, total_step = float(np.sum(waits)), float(np.sum(steps))
                # The headline tuning number: share of wall time the GPU spent
                # idle waiting for input.
                epoch_summary['data_wait_share'] = (
                    total_wait / max(total_wait + total_step, 1e-9))
        train_logger.write(
            epoch_summary,
            text=f'[EPOCH {epoch} DONE] ' + _format_log(epoch_summary),
        )

        # Save end-of-epoch checkpoint.
        path = trainer.save(ckpt_dir, tag=f'epoch_{epoch}')
        print(f'[CHECKPOINT] Epoch checkpoint saved: {path}')

        # Held-out validation drives best.pt selection (lower loss = better).
        if val_loader is not None and (epoch + 1) % args.val_every_epochs == 0:
            v_logs = []
            for vb in val_loader:
                # MSEs and MAEs of all validation batches 
                v_logs.append({k: _to_scalar(v)
                                for k, v in trainer.val_step(vb).items()})
            if v_logs:
                # nanmean so a single non-finite val batch (already flagged by
                # the trainer's nan-debug guard) doesn't poison the whole average
                # and silently block best.pt selection.
                avg = {f'val_{k}': float(np.nanmean([d[k] for d in v_logs]))
                       for k in v_logs[0]}
                avg['kind'] = 'val_summary'
                avg['epoch'] = epoch
                avg['global_step'] = trainer.global_step
                train_logger.write(
                    avg,
                    text=f'[VAL EPOCH {epoch}] ' + _format_log(avg),
                )
                avg_val_loss = avg.get('val_loss', float('inf'))
                if avg_val_loss < best_val_loss - args.min_delta:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    shutil.copyfile(path, best_path)
                    legacy = os.path.join(ckpt_dir, 'normal.pytmodel')
                    if os.path.isfile(legacy):
                        shutil.copyfile(
                            legacy,
                            os.path.join(ckpt_dir, 'best_normal.pytmodel'),
                        )
                    print(f'[BEST] Epoch {epoch} | '
                          f'val_loss={avg_val_loss:.4f} → {best_path}')
                else:
                    epochs_no_improve += 1
                    if args.patience > 0:
                        print(f'[EARLY STOPPING]: No val improvement '
                              f'({epochs_no_improve}/{args.patience}) | '
                              f'best_val_loss={best_val_loss:.4f}')

        prune_checkpoints(ckpt_dir, keep_last=args.keep_last, protect=('best',))

        # Early stopping (option 1): a safety cutoff that leaves the cosine
        # schedule spanning --epochs but bails once val loss plateaus.
        if args.patience > 0 and epochs_no_improve >= args.patience:
            stopped_early = True
            msg = (f'[EARLY STOPPING] Stopping at epoch {epoch}: val loss did not '
                   f'improve by > {args.min_delta} for {args.patience} '
                   f'consecutive validation check(s) | '
                   f'best_val_loss={best_val_loss:.4f}')
            print(msg)
            train_logger.write(
                {'kind': 'early_stop', 'epoch': epoch,
                 'global_step': trainer.global_step,
                 'best_val_loss': best_val_loss,
                 'epochs_no_improve': epochs_no_improve},
                text=msg,
            )
            break

    final = trainer.save(ckpt_dir, tag='final')
    reason = 'early-stopped' if stopped_early else 'completed all epochs'
    print(f'[TRAIN] DONE ({reason}) | Final checkpoint: {final} | '
          f'best_val_loss={best_val_loss:.4f}')
    if args.log_memory and device.type == 'cuda':
        # Max over the per-epoch peaks (peak stats are reset each epoch).
        # Compare against the card's capacity to see how much room is left for
        # a larger --batch_size.
        print(f'[MEM] run peak allocated = {run_peak["mem_peak_gib"]:.2f} GiB | '
              f'peak reserved = {run_peak["mem_reserved_peak_gib"]:.2f} GiB | '
              f'peak host RSS (self+workers) = {run_peak["host_rss_gib"]:.2f} GiB')

    # Report the final test number on the SELECTED model (best.pt, chosen by
    # val loss). Fall back to the in-memory weights only if best.pt is somehow absent (e.g. val never yielded a
    # finite loss so no best was ever written).
    if os.path.isfile(best_path):
        print(f'[TRAIN] loading best.pt (val_loss={best_val_loss:.4f}) '
              f'for the final held-out test evaluation')
        trainer.load(best_path)
    else:
        print('[TRAIN] WARNING: best.pt not found; running the final test '
              'evaluation on the last-epoch weights instead.')

    # Publish the same weights the test number is reported on as a drop-in
    # inference checkpoint. `trainer.load` above only updates the in-memory
    # model — `normal.pytmodel` on disk still holds the final-epoch weights —
    # so the export reads from `best_normal.pytmodel` when a best exists.
    export_for_inference(ckpt_dir)

    # The held-out mixed test split is the final benchmark.
    # Re-seed here so the eval's pixel sampling (torch RNG inside the model) is
    # reproducible independent of how many RNG draws training consumed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    summary = run_test_eval(trainer, test_loader, eval_logger,
                            global_step=trainer.global_step,
                            n_trials=args.test_trials)
    if summary:
        print(f'[TRAIN] Final held-out test (avg of {args.test_trials} trials) | '
              f'mae={summary.get("test_mae_deg", float("nan")):.4f} | '
              f'loss={summary.get("test_loss", float("nan")):.4f}')


if __name__ == '__main__':
    main()
