"""
Training-side counterpart of `builder.py`.

`Trainer` builds a `Net`, optimizer, scheduler, AMP autocast context, and
exposes `train_step` / `val_step` / `save` / `load`. The training script
drives the loop, logs metrics, and runs evaluation. The Trainer owns the
model state and the optimization step.

Mixed-precision policy
----------------------
- bf16  : `torch.amp.autocast(dtype=torch.bfloat16)` with NO GradScaler
          (bf16 has fp32-equivalent exponent range; loss scaling is
          unneeded and `GradScaler.unscale_` is unsupported).
- fp16  : legacy `torch.cuda.amp.autocast(dtype=torch.float16)` with a
          `GradScaler` (matches the upstream paper code).
- none  : no autocast; model runs in the optimizer's default dtype.
"""

import glob
import math
import os
from contextlib import nullcontext

import torch

from modules.model import model
from modules.model.model_utils import loadmodel, mode_change, get_n_params
from modules.loss import losses


def _step_decay_with_warmup(optimizer, warmup_steps, steps_per_epoch,
                             decay_every_epochs=10, gamma=0.8):
    """SDM-UniPS paper schedule: linear warmup then x`gamma` every N epochs."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        epoch = step // max(1, steps_per_epoch)
        return gamma ** (epoch // decay_every_epochs)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """Thesis schedule: linear warmup -> cosine annealing to `min_lr_ratio`."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, args, device, total_steps, steps_per_epoch):
        self.args = args
        self.device = device
        self.target = 'normal'

        self.net = model.Net(args.pixel_samples, device).to(device)
        self.net.with_grad()
        print(f"[Trainer] Target = Normal  Params = {get_n_params(self.net):,}")

        if getattr(args, 'resume', None):
            ckpt_paths = (glob.glob(os.path.join(args.resume, '*.pytmodel'))
                          + glob.glob(os.path.join(args.resume, '*.pt')))
            if ckpt_paths:
                print(f'[Trainer] Resuming weights from {ckpt_paths[0]}')
                self.net = loadmodel(self.net, ckpt_paths[0], strict=False)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad],
            lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
        )
        warmup_steps = int(args.warmup_epochs * steps_per_epoch)
        if args.lr_schedule == 'cosine':
            self.scheduler = _cosine_with_warmup(
                self.optimizer, warmup_steps, total_steps,
                min_lr_ratio=float(getattr(args, 'min_lr_ratio', 0.01)),
            )
        else:
            self.scheduler = _step_decay_with_warmup(
                self.optimizer, warmup_steps, steps_per_epoch,
                decay_every_epochs=args.lr_decay_every, gamma=args.lr_decay_gamma,
            )

        self.amp_dtype = getattr(args, 'amp_dtype', 'none')
        if self.amp_dtype not in ('bf16', 'fp16', 'none'):
            raise ValueError(f"--amp_dtype must be bf16|fp16|none (got {self.amp_dtype})")
        # Legacy --amp flag == fp16 autocast.
        if getattr(args, 'amp', False) and self.amp_dtype == 'none':
            self.amp_dtype = 'fp16'
        self.amp_enabled = (self.amp_dtype != 'none' and device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.amp_enabled and self.amp_dtype == 'fp16'),
        )
        print(f'[Trainer] AMP = {self.amp_dtype}  Enabled = {self.amp_enabled}  '
              f'schedule = {args.lr_schedule}  warmup_steps = {warmup_steps}  '
              f'total_steps = {total_steps}')

        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.global_step = 0
        self._nan_reported = False
        self.detect_anomaly = bool(getattr(args, 'detect_anomaly', False))
        if self.detect_anomaly:
            print('[Trainer] torch.autograd anomaly detection ENABLED '
                  '(slow; raises at the first NaN/Inf op).')
            self._install_forward_nan_hooks()

    def _install_forward_nan_hooks(self):
        """Print the FIRST leaf module whose forward output goes non-finite.

        Anomaly detection only flags backward ops, so a pure forward explosion
        surfaces only at the loss. These post-forward hooks fire in execution
        order; the first non-finite output localizes the culprit. If that
        module's inputs were already non-finite, the blow-up is in a functional
        op (einsum/exp/grid_sample/interpolate) just upstream of it.
        """
        self._fwd_nan_found = False

        def make_hook(name):
            def hook(module, inp, out):
                if self._fwd_nan_found:
                    return
                outs = out if isinstance(out, (tuple, list)) else (out,)
                bad = any(torch.is_tensor(t) and t.numel() > 0
                          and not torch.isfinite(t).all() for t in outs)
                if not bad:
                    return
                ins = [t for t in inp if torch.is_tensor(t) and t.numel() > 0]
                in_finite = all(torch.isfinite(t).all() for t in ins)
                self._fwd_nan_found = True
                print(f'[fwd-nan] FIRST non-finite forward output: '
                      f'{name} ({module.__class__.__name__})  '
                      f'inputs_finite={in_finite}')
                if in_finite:
                    print('[fwd-nan] => this module is the culprit.')
                else:
                    print('[fwd-nan] => inputs already non-finite; culprit is a '
                          'functional op (einsum/exp/grid_sample/interpolate) '
                          'just upstream of this module.')
            return hook

        n = 0
        for name, module in self.net.named_modules():
            if not list(module.children()):  # leaf modules only
                module.register_forward_hook(make_hook(name))
                n += 1
        print(f'[Trainer] forward NaN hooks installed on {n} leaf modules.')

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        dtype = torch.bfloat16 if self.amp_dtype == 'bf16' else torch.float16
        return torch.amp.autocast(device_type='cuda', dtype=dtype)

    def _move_batch(self, batch):
        I, N, M, n_imgs = batch
        I = I.to(self.device, non_blocking=True).float()
        N = N.to(self.device, non_blocking=True).float()
        M = M.to(self.device, non_blocking=True).float()
        n_imgs = n_imgs.to(self.device, non_blocking=True).long().reshape(-1, 1)
        return I, N, M, n_imgs

    def _forward_losses(self, batch):
        I, N, M, n_imgs = self._move_batch(batch)
        H = I.shape[2]
        dec_res = torch.full((I.shape[0], 1), H, dtype=torch.long, device=self.device)
        can_res = torch.full((I.shape[0], 1), self.args.canonical_resolution,
                             dtype=torch.long, device=self.device)

        pred_n, sample_idx, _ = self.net(
            I, M, n_imgs, decoder_resolution=dec_res,
            canonical_resolution=can_res, training=True,
        )

        loss = losses.normal_loss(pred_n, N, M, sample_idx)
        with torch.no_grad():
            mae = losses.angular_error_deg(pred_n, N, M, sample_idx)
        return loss, {'loss': loss.detach(), 'mae_deg': mae.detach()}

    def _report_nan(self, batch):
        """One-time diagnostic: localize a non-finite loss to inputs vs forward."""
        # Only report the FIRST NaN/Inf occurrence, to avoid spamming.
        if self._nan_reported:
            return
        self._nan_reported = True
        
        I, N, M, n_imgs = self._move_batch(batch)
        def stat(name, t):
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            print(f'[nan-debug] {name}: shape={tuple(t.shape)} '
                  f'nan={n_nan} inf={n_inf} '
                  f'min={t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else float("nan"):.4g} '
                  f'max={t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else float("nan"):.4g}')
        print('[nan-debug] non-finite loss detected; inspecting this batch:')
        stat('I (images)', I)
        stat('N (gt normal)', N)
        stat('M (mask)', M)
        any_input_bad = any(not torch.isfinite(t).all() for t in (I, N, M))
        # Scan parameters: NaN weights mean a PREVIOUS backward/step corrupted
        # the model (e.g. an exploding/NaN gradient), so the current forward NaNs
        # even with clean inputs. Clean weights + NaN loss => pure forward issue.
        bad_params = [name for name, p in self.net.named_parameters()
                      if not torch.isfinite(p).all()]
        if bad_params:
            print(f'[nan-debug] {len(bad_params)} parameter tensors are NaN/Inf, '
                  f'e.g. {bad_params[:3]}')
        if any_input_bad:
            print('[nan-debug] => NaN/Inf is in the INPUT data (loader bug), '
                  'not the forward pass.')
        elif bad_params:
            print('[nan-debug] => model WEIGHTS are corrupted: a prior backward '
                  'produced a NaN/Inf gradient (grad_clip cannot fix a NaN). '
                  'Look at the loss/normalization gradient, not the forward.')
        else:
            print('[nan-debug] => inputs and weights are finite; NaN arises in '
                  'this forward pass itself (model numerics).')

    def train_step(self, batch):
        self.net.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Anomaly detection must wrap BOTH forward and backward: it records the
        # forward op stacks so a backward NaN/Inf raises pointing at the exact
        # originating op.
        anomaly_ctx = (torch.autograd.detect_anomaly()
                       if self.detect_anomaly else nullcontext())
        with anomaly_ctx:
            with self._autocast():
                loss, log = self._forward_losses(batch)

            if not torch.isfinite(loss):
                self._report_nan(batch)

            clip_val = self.args.grad_clip if self.args.grad_clip > 0 else float('inf')
            if self.amp_dtype == 'fp16':
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_val)
                self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1
        log['lr'] = self.optimizer.param_groups[0]['lr']
        log['grad_norm'] = grad_norm.detach() if torch.is_tensor(grad_norm) else float(grad_norm)
        return log

    @torch.no_grad()
    def val_step(self, batch):
        self.net.eval()
        with self._autocast():
            loss, log = self._forward_losses(batch)
        if not torch.isfinite(loss):
            self._report_nan(batch)
        return log

    def save(self, ckpt_dir, tag):
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'{tag}.pt')
        torch.save({
            'global_step': self.global_step,
            'target': self.target,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
        }, path)
        # Drop-in copy for the inference Builder.
        legacy = os.path.join(ckpt_dir, f'{self.target}.pytmodel')
        torch.save(self.net.state_dict(), legacy)
        return path

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        if ckpt.get('scaler') is not None and self.scaler is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
        self.global_step = ckpt.get('global_step', 0)


def prune_checkpoints(ckpt_dir, keep_last=3, protect=()):
    """Delete `epoch_*.pt` checkpoints except the most recent `keep_last` and
    anything in `protect` (e.g. {'best'}). The legacy `<target>.pytmodel` and
    explicit step checkpoints are never auto-deleted.
    """
    paths = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))
    if len(paths) <= keep_last:
        return
    keep = set(paths[-keep_last:])
    for name in protect:
        keep.add(os.path.join(ckpt_dir, f'{name}.pt'))
    for p in paths:
        if p not in keep and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass
