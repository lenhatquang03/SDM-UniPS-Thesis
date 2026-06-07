"""
Training entry point for SDM-UniPS (Model A baseline).

Quickstart for the thesis recipe (hdlong-complexv1 + PolarPS):

    python sdm_unips/train.py \
        --session_name modelA_smoke \
        --hdlong_dir /kaggle/input/hdlong-complexv1 \
        --polarps_dir /kaggle/input/polarps \
        --eval_dir /kaggle/input/diligent/pmsData \
        --max_scenes 8000 --val_fraction 0.1 \
        --train_resolution 512 --canonical_resolution 256 \
        --batch_size 8 --pixel_samples 2048 \
        --lr 1e-4 --weight_decay 0.05 --lr_schedule cosine \
        --warmup_epochs 5 --min_lr_ratio 0.01 --grad_clip 1.0 \
        --amp_dtype bf16 --smoke_test

`--smoke_test` clamps the run to `--smoke_epochs` (default 10) for a quick
Kaggle dry-run before committing to the full 60-epoch schedule.

Checkpoint selection uses a held-out scene-level validation split of the
mixed (hdlong + PolarPS) pool. DiLiGenT is the held-out *test* benchmark and
is evaluated exactly once, after the final epoch, to avoid leaking the test
set into model selection.

Logging
-------
Per-step JSONL lines are appended to `<session>/logs/train.jsonl` and a
human-readable mirror to `<session>/logs/train.log`. The single final
DiLiGenT evaluation lands in `<session>/logs/eval.jsonl`.
"""

from __future__ import print_function, division

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.builder.trainer import Trainer, prune_checkpoints
from modules.io.dataio import (
    build_train_dataset,
    build_val_dataset,
    DiligentEvalDataset,
)


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
    p.add_argument('--mask_margin', type=int, default=8)
    p.add_argument('--checkpoint_dir', default=None,
                   help='Defaults to <session_name>/checkpoints')
    p.add_argument('--log_dir', default=None,
                   help='Defaults to <session_name>/logs')
    p.add_argument('--resume', default=None)
    p.add_argument('--log_every', type=int, default=10,
                   help='Steps between log records')
    p.add_argument('--ckpt_every', type=int, default=0,
                   help='Step interval for extra checkpoints. 0 disables.')
    p.add_argument('--val_every_epochs', type=int, default=1)
    p.add_argument('--keep_last', type=int, default=3,
                   help='Number of recent epoch checkpoints to keep on disk')

    # Eval (DiLiGenT) — run once after the final epoch -------------------
    p.add_argument('--eval_dir', default=None,
                   help='DiLiGenT pmsData root (10 *PNG scene dirs). '
                        'Evaluated once, after training, as the held-out test set.')
    p.add_argument('--eval_K_list', default='2,4,8,16,32,64,96')
    p.add_argument('--eval_trials', type=int, default=10)
    p.add_argument('--eval_side', type=int, default=512)
    p.add_argument('--eval_best_K', type=int, default=16,
                   help='K whose mean MAE is reported as the headline final test number')

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
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
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


def _to_scalar(v):
    if torch.is_tensor(v):
        return float(v.detach().cpu().item())
    return float(v)


def _fmt_value(v):
    a = abs(v)
    if a == 0:
        return f'{v:.4f}'
    if a >= 1e-3:
        return f'{v:.4f}'
    return f'{v:.2e}'


def _format_log(log):
    return '  '.join(f'{k}={_fmt_value(_to_scalar(v))}'
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
# DiLiGenT evaluation hook
# ---------------------------------------------------------------------------
@torch.no_grad()
def _eval_one_pack(trainer, I_np, N_np, M_np, K_used, device):
    """Run inference on one (scene, K) pack and return masked-mean MAE in deg."""
    I = torch.from_numpy(I_np).unsqueeze(0).to(device).float()       # (1, 3, h, w, K)
    M = torch.from_numpy(M_np).unsqueeze(0).to(device).float()       # (1, 1, h, w)
    N = torch.from_numpy(N_np).unsqueeze(0).to(device).float()       # (1, 3, h, w)
    n_imgs = torch.tensor([[int(K_used)]], dtype=torch.long, device=device)
    H = I.shape[2]
    dec_res = torch.full((1, 1), H, dtype=torch.long, device=device)
    can_res = torch.full((1, 1), trainer.args.canonical_resolution,
                         dtype=torch.long, device=device)

    trainer.net.eval()
    with trainer._autocast():
        nout = trainer.net(
            I, M, n_imgs,
            decoder_resolution=dec_res,
            canonical_resolution=can_res,
            training=False,
        )
    nout = nout.float()
    nrm = torch.linalg.norm(nout, dim=1, keepdim=True).clamp_min(1e-8)
    nout = nout / nrm
    mask_bool = (M[:, 0] > 0.5)
    dot = (nout * N).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    ang = torch.acos(dot) * (180.0 / 3.141592653589793)
    err = ang[mask_bool]
    if err.numel() == 0:
        return float('nan')
    return float(err.mean().detach().cpu().item())


def run_diligent_eval(trainer, eval_set, device, logger, epoch, global_step,
                       best_K=16):
    """Run the K-sweep and return mean MAE at `best_K`."""
    per_scene_K_trials = {}
    obj_names = {}
    for i in range(len(eval_set)):
        meta = eval_set.get_meta(i)
        I, N, M, K_used, K, obj_idx, trial = eval_set[i]
        mae = _eval_one_pack(trainer, I, N, M, int(K_used), device)
        obj_names[obj_idx] = meta['objname']
        per_scene_K_trials.setdefault((obj_idx, K), []).append(mae)
        logger.write(
            {
                'epoch': epoch,
                'global_step': global_step,
                'obj_idx': obj_idx,
                'objname': meta['objname'],
                'K': int(K),
                'trial': trial,
                'mae_deg': mae,
            },
            text=(f'[eval] epoch={epoch} step={global_step} '
                  f'obj={meta["objname"]} K={K} trial={trial} mae={mae:.4f}'),
        )

    K_list = sorted({K for (_, K) in per_scene_K_trials})
    per_K_mean = {}
    per_obj_mean = {}
    for K in K_list:
        per_obj = {}
        for obj_idx in sorted(obj_names):
            trials = per_scene_K_trials.get((obj_idx, K), [])
            if trials:
                per_obj[obj_names[obj_idx]] = float(np.nanmean(trials))
        per_obj_mean[K] = per_obj
        per_K_mean[K] = (float(np.nanmean(list(per_obj.values())))
                         if per_obj else float('nan'))

    summary = {
        'kind': 'eval_summary',
        'epoch': epoch,
        'global_step': global_step,
        'per_K_mean_mae_deg': per_K_mean,
        'per_K_per_obj_mae_deg': per_obj_mean,
    }
    summary_text = ('[eval-summary] epoch={}  '.format(epoch)
                    + '  '.join(f'K={K}:{per_K_mean[K]:.4f}' for K in K_list))
    logger.write(summary, text=summary_text)
    print(summary_text)
    return per_K_mean.get(best_K, float('nan'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()

    # Smoke-test configurations
    if args.smoke_test:
        args.epochs = min(args.epochs, args.smoke_epochs)
        if args.smoke_max_scenes > 0:
            args.max_scenes = args.smoke_max_scenes

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create dedicated directories for logs and checkpoints
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_dir = args.log_dir or os.path.join(args.session_name, 'logs')
    ckpt_dir = args.checkpoint_dir or os.path.join(args.session_name, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'[Train] Device = {device}  Session = {args.session_name}')
    print(f'[Train] Dataset = mixed (hdlong + PolarPS)  Automatic Mixed Precision (AMP) = {args.amp_dtype}  '
          f'Epochs = {args.epochs}  Batch size = {args.batch_size}  K=10')
    print(f'[Train] Log directory = {log_dir}  Checkpoint directory = {ckpt_dir}')
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f'[Train] GPU = {torch.cuda.get_device_name(0)}  '
              f'Memory = {gpu_props.total_memory / 2**30:.1f} GiB  '
              f'Capability = {gpu_props.major}.{gpu_props.minor}')

    # Train Dataloader
    train_set = build_train_dataset(args, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        drop_last=True, collate_fn=_collate,
        persistent_workers=(args.num_workers > 0),
    )

    # Held-out scene-level split drives checkpoint selection (best.pt).
    val_set = build_val_dataset(args)
    val_loader = None
    if len(val_set) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=(device.type == 'cuda'),
            drop_last=False, collate_fn=_collate,
        )
    else:
        print('[Train] Validatoin split is empty (--val_fraction too small); '
              'best.pt will not be updated.')

    eval_set = None
    if args.eval_dir:
        K_list = tuple(int(k) for k in args.eval_K_list.split(',') if k)
        eval_set = DiligentEvalDataset(
            args.eval_dir, K_list=K_list, trials_per_K=args.eval_trials,
            side=args.eval_side, seed=args.seed,
        )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    print(f'[Train] steps_per_epoch = {steps_per_epoch:,}  total_steps = {total_steps:,}')

    trainer = Trainer(args, device, total_steps=total_steps,
                       steps_per_epoch=steps_per_epoch)

    train_logger = JSONLLogger(log_dir, 'train')
    eval_logger = JSONLLogger(log_dir, 'eval')

    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump({k: v for k, v in sorted(vars(args).items())
                   if isinstance(v, (str, int, float, bool, type(None)))},
                  f, indent=2)

    best_val_loss = float('inf')
    best_path = os.path.join(ckpt_dir, 'best.pt')
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        epoch_running = {}
        for it, batch in enumerate(train_loader):
            step_t0 = time.time()
            log = trainer.train_step(batch)
            step_dt = time.time() - step_t0

            scalar_log = {k: _to_scalar(v) for k, v in log.items()}
            scalar_log['step_sec'] = step_dt
            scalar_log['epoch'] = epoch
            scalar_log['it'] = it
            scalar_log['global_step'] = trainer.global_step

            for k, v in scalar_log.items():
                if isinstance(v, float):
                    epoch_running.setdefault(k, []).append(v)

            if trainer.global_step % args.log_every == 0 or it == 0:
                elapsed = time.time() - t0
                head = (f'[step {trainer.global_step}/{total_steps}] '
                        f'epoch={epoch} it={it} elapsed={elapsed:.1f}s')
                text = head + '  ' + _format_log(scalar_log)
                train_logger.write(scalar_log, text=text)
                print(text)

            if args.ckpt_every > 0 and trainer.global_step % args.ckpt_every == 0:
                path = trainer.save(ckpt_dir, tag=f'step_{trainer.global_step}')
                print(f'[ckpt] step checkpoint saved {path}')

        epoch_summary = {
            'kind': 'epoch_summary',
            'epoch': epoch,
            'global_step': trainer.global_step,
            'epoch_sec': time.time() - epoch_t0,
        }
        for k, vals in epoch_running.items():
            if vals:
                epoch_summary[f'avg_{k}'] = float(np.mean(vals))
        train_logger.write(
            epoch_summary,
            text=f'[epoch {epoch} done] ' + _format_log(epoch_summary),
        )

        # Save end-of-epoch checkpoint.
        path = trainer.save(ckpt_dir, tag=f'epoch_{epoch}')
        print(f'[ckpt] epoch checkpoint saved {path}')

        # Held-out validation drives best.pt selection (lower loss = better).
        if val_loader is not None and (epoch + 1) % args.val_every_epochs == 0:
            v_logs = []
            for vb in val_loader:
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
                    text=f'[val epoch {epoch}] ' + _format_log(avg),
                )
                avg_val_loss = avg.get('val_loss', float('inf'))
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    shutil.copyfile(path, best_path)
                    legacy = os.path.join(ckpt_dir, 'normal.pytmodel')
                    if os.path.isfile(legacy):
                        shutil.copyfile(
                            legacy,
                            os.path.join(ckpt_dir, 'best_normal.pytmodel'),
                        )
                    print(f'[best] epoch {epoch}  '
                          f'val_loss={avg_val_loss:.4f}  → {best_path}')

        prune_checkpoints(ckpt_dir, keep_last=args.keep_last, protect=('best',))

    final = trainer.save(ckpt_dir, tag='final')
    print(f'[Train] done. final checkpoint: {final}  '
          f'best_val_loss={best_val_loss:.4f}')

    # DiLiGenT is the held-out test benchmark: evaluate exactly once, after
    # training, so the test set never influences checkpoint selection.
    if eval_set is not None:
        final_mae = run_diligent_eval(
            trainer, eval_set, device, eval_logger,
            epoch=args.epochs - 1, global_step=trainer.global_step,
            best_K=args.eval_best_K,
        )
        print(f'[Train] final DiLiGenT mae@K{args.eval_best_K}={final_mae:.4f}')


if __name__ == '__main__':
    main()
