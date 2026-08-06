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

Gradient accumulation
---------------------
`--accum_steps > 1` splits one *effective* batch across that many forward /
backward passes, so a GPU that cannot hold `--batch_size 8` can still train
the thesis recipe at `--batch_size 2 --accum_steps 4`. Each micro-batch loss
is divided by `accum_steps` before `.backward()`, so the accumulated gradient
is the mean over the effective batch — numerically equivalent to a single
large batch (up to per-sample-count rounding when micro-batches differ in
size, which `drop_last=True` prevents on the train loader).

`global_step`, the scheduler, and `--log_every` / `--ckpt_every` all count
*optimizer* steps, not micro-batches, so the LR schedule is identical to an
un-accumulated run at the same effective batch size.
"""

import glob
import math
import os
from collections import deque
from contextlib import nullcontext

import torch

from modules.model import model
from modules.model.model_utils import loadmodel, mode_change, get_n_params
from modules.loss import losses


class NonFiniteGradientAbort(RuntimeError):
    """Too many optimizer steps were skipped for non-finite gradients.

    Skipping a bad step protects the weights, but a model that never steps
    never learns, and across a multi-day run that failure is *silent*: the loss
    curve of a model receiving no updates looks like a model that has plateaued.
    This aborts instead, so the run dies loudly and early rather than burning
    days producing nothing.

    Two independent guards raise it, because neither subsumes the other:

    * `--max_consecutive_skips` — an unbroken run of skips. Fast to trip, so it
      catches a hard breakage within seconds.
    * `--max_skip_rate` over `--skip_rate_window` — the fraction of recent steps
      skipped. Catches the case the consecutive counter is blind to: an
      intermittent failure (say every other batch) that halves the effective
      training run without ever producing two skips in a row.
    """


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
        self.global_step = 0          # counts OPTIMIZER steps, not micro-batches
        self.accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
        self._micro_in_cycle = 0      # micro-batches accumulated since last step
        if self.accum_steps > 1:
            print(f'[Trainer] Gradient accumulation = {self.accum_steps} x '
                  f'batch_size {args.batch_size} => effective batch '
                  f'{self.accum_steps * args.batch_size}')
        self._nan_reported = False
        self._grad_nan_reported = False
        # Non-finite-gradient skip policy (see `_note_skip` / `_check_skip_rate`).
        self.max_consecutive_skips = int(getattr(args, 'max_consecutive_skips', 20))
        self.max_skip_rate = float(getattr(args, 'max_skip_rate', 0.3))
        self.skip_rate_window = max(1, int(getattr(args, 'skip_rate_window', 200)))
        self.skipped_steps = 0        # cumulative over the run
        self._consecutive_skips = 0
        # Rolling 0/1 record of the last `skip_rate_window` OPTIMIZER steps. The
        # rate guard stays disarmed until this is full, which doubles as the
        # grace period: without it a single skip among the first two steps would
        # read as a 50% rate and abort a healthy run instantly.
        self._skip_window = deque(maxlen=self.skip_rate_window)
        guards = []
        if self.max_consecutive_skips > 0:
            guards.append(f'{self.max_consecutive_skips} consecutive')
        if self.max_skip_rate > 0:
            guards.append(f'>{self.max_skip_rate:.0%} of the last '
                          f'{self.skip_rate_window} steps')
        print('[Trainer] Non-finite gradients skip the optimizer step; abort on '
              + (' or '.join(guards) if guards else 'NOTHING (both guards disabled)'))
        self.detect_anomaly = bool(getattr(args, 'detect_anomaly', False))
        if self.detect_anomaly:
            print('[Trainer] torch.autograd anomaly detection ENABLED '
                  '(slow; raises at the first NaN/Inf op).')
            self._install_forward_nan_hooks()
            self._install_grad_nan_hooks()

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

    def _install_grad_nan_hooks(self):
        """Name the FIRST parameter whose gradient goes non-finite, in true
        BACKWARD execution order.

        `_report_bad_grads` below infers the origin from registration order,
        which is only a proxy for the backward topology (the net is a DAG —
        encoder, aggregation and regressor with skip paths — not a chain).
        These per-parameter hooks fire exactly when each gradient is produced,
        so the first one to see a non-finite value *is* the origin, with no
        ordering assumption at all. One `isfinite` per parameter per step is far
        too expensive to leave on, hence --detect_anomaly only.
        """
        self._grad_hook_nan_found = False

        def make_hook(name):
            def hook(grad):
                if self._grad_hook_nan_found or torch.isfinite(grad).all():
                    return grad
                self._grad_hook_nan_found = True
                print(f'[grad-nan] FIRST non-finite gradient in BACKWARD order: '
                      f'{name}  ({int(torch.isnan(grad).sum())} NaN, '
                      f'{int(torch.isinf(grad).sum())} Inf of {grad.numel()})')
                print('[grad-nan] => created by the backward of an op between '
                      'this parameter and the loss; every parameter reported '
                      'after this one is downstream contamination.')
                return grad
            return hook

        n = 0
        for name, p in self.net.named_parameters():
            if p.requires_grad:
                p.register_hook(make_hook(name))
                n += 1
        print(f'[Trainer] gradient NaN hooks installed on {n} parameters.')

    def _report_bad_grads(self, grad_norm):
        """Localize a non-finite GRADIENT, called *before* the optimizer step.

        Backward runs loss -> input, so a NaN born in some module's backward
        contaminates every parameter UPSTREAM of it (earlier in the forward) and
        leaves everything downstream — already differentiated — clean.
        `named_parameters()` yields registration order, which follows the
        forward, so the contaminated region ENDS at the origin. The old
        `bad_params[:3]` printed the other end of that region, which is why it
        always named the ConvNeXt stem regardless of where the NaN came from.
        """
        if self._grad_nan_reported:
            return
        self._grad_nan_reported = True
        self._nan_reported = True   # suppress the redundant post-hoc weight scan

        print(f'[grad-nan] non-finite gradient before optimizer step '
              f'{self.global_step + 1} (total_norm={float(grad_norm):.4g}); '
              'reported BEFORE the step, so the weights are still clean.')

        stats = []   # (name, n_bad, numel, norm of the finite part)
        for name, p in self.net.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            finite = torch.isfinite(g)
            gn = float(g[finite].norm()) if bool(finite.any()) else float('nan')
            stats.append((name, int((~finite).sum()), g.numel(), gn))

        bad = [s for s in stats if s[1] > 0]
        if not bad:
            # clip_grad_norm_ sums squares across all tensors; that sum can
            # overflow to +Inf while every individual tensor is finite.
            print('[grad-nan] every individual grad tensor is finite — the '
                  'TOTAL norm overflowed while accumulating them. Largest '
                  'per-tensor grad norms:')
            for name, _, _, gn in sorted(stats, key=lambda s: -s[3])[:5]:
                print(f'[grad-nan]   {name}: |g|={gn:.4g}')
            return

        print(f'[grad-nan] {len(bad)}/{len(stats)} grad tensors non-finite.')
        print('[grad-nan] nearest the LOSS (origin end, most informative):')
        for name, n_bad, numel, gn in reversed(bad[-3:]):
            print(f'[grad-nan]   {name}: {n_bad}/{numel} bad, '
                  f'finite-part |g|={gn:.4g}')

        names = [s[0] for s in stats]
        clean_after = [s for s in stats[names.index(bad[-1][0]) + 1:] if s[1] == 0]
        if clean_after:
            print('[grad-nan] first CLEAN grads past that point (differentiated '
                  'earlier in backward, so the NaN is born between these and '
                  'the block above):')
            for name, _, _, gn in clean_after[:3]:
                print(f'[grad-nan]   {name}: |g|={gn:.4g}')
        else:
            print('[grad-nan] nothing clean past that point: the NaN is born at '
                  'or after the LAST parameter of the network — look at the '
                  'loss and the unit-normalization in Net._decode_pixels.')

        # A total wipeout still says something if it is broken down by stage.
        by_mod = {}
        for name, n_bad, _, _ in stats:
            top = '.'.join(name.split('.')[:2])
            b, t = by_mod.get(top, (0, 0))
            by_mod[top] = (b + (1 if n_bad else 0), t + 1)
        print('[grad-nan] per-submodule bad/total: '
              + '  '.join(f'{k}={b}/{t}' for k, (b, t) in by_mod.items()))
        if not self.detect_anomaly:
            print('[grad-nan] re-run with --detect_anomaly for the exact '
                  'backward-order origin and the raising op.')

    def _note_skip(self, grad_norm):
        """Record a skipped optimizer step; abort if they run consecutively.

        One line per skip, deliberately: a run that skips 40% of its steps
        without ever hitting the consecutive threshold is still broken, and the
        only way to see that in a log read days later is for every skip to leave
        a mark. `avg_grad_skipped` on each epoch summary is the same signal
        aggregated (0.0 = healthy, 1.0 = nothing learned this epoch).
        """
        self.skipped_steps += 1
        self._consecutive_skips += 1
        # Rolling rate so far — reported even while the window is still filling,
        # so a bad start is visible before the guard is armed to act on it.
        seen = len(self._skip_window)
        rate = (sum(self._skip_window) / seen) if seen else 0.0
        print(f'[grad-skip] step {self.global_step + 1}: non-finite gradient '
              f'(total_norm={float(grad_norm):.4g}) — optimizer step SKIPPED  '
              f'[consecutive={self._consecutive_skips}'
              f'/{self.max_consecutive_skips or "off"}  '
              f'rate={rate:.1%} of last {seen}/{self.skip_rate_window}  '
              f'total={self.skipped_steps}]')
        if (self.max_consecutive_skips > 0
                and self._consecutive_skips >= self.max_consecutive_skips):
            raise NonFiniteGradientAbort(
                f'{self._consecutive_skips} consecutive optimizer steps skipped '
                f'for non-finite gradients (limit --max_consecutive_skips='
                f'{self.max_consecutive_skips}); the model has received no '
                f'update in that span, so training is aborting rather than '
                f'silently learning nothing. See the [grad-nan] report above '
                f'for the origin, and re-run with --detect_anomaly to get the '
                f'exact op. Weights were never corrupted (the bad steps were '
                f'skipped), so the last checkpoint is sound.')

    def _check_skip_rate(self):
        """Abort on a sustained skip *rate*, which the consecutive counter misses.

        An intermittent failure — every other batch, say — never produces two
        skips in a row, so `--max_consecutive_skips` never fires while half the
        run silently evaporates. This measures a rolling window instead, and
        stays disarmed until that window is full so the noisy opening steps
        (and fp16's scale calibration) cannot trip it.
        """
        if self.max_skip_rate <= 0 or len(self._skip_window) < self.skip_rate_window:
            return
        rate = sum(self._skip_window) / self.skip_rate_window
        if rate <= self.max_skip_rate:
            return
        raise NonFiniteGradientAbort(
            f'{rate:.1%} of the last {self.skip_rate_window} optimizer steps '
            f'were skipped for non-finite gradients, over the limit '
            f'--max_skip_rate={self.max_skip_rate:.1%}; the run is only '
            f'training at {1.0 - rate:.0%} of its nominal rate and the LR '
            f'schedule has advanced regardless, so it is aborting rather than '
            f'silently under-training. This guard is deliberately separate from '
            f'--max_consecutive_skips: an intermittent NaN never trips a '
            f'consecutive counter. See the [grad-nan] report for the origin. '
            f'Weights were never corrupted, so the last checkpoint is sound.')

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
        all_params = [name for name, _ in self.net.named_parameters()]
        bad_params = [name for name, p in self.net.named_parameters()
                      if not torch.isfinite(p).all()]
        if bad_params:
            # Print the end of the contaminated region NEAREST THE LOSS, not
            # `bad_params[:3]`: that is registration order, so the first entries
            # are always the ConvNeXt stem no matter where the NaN came from.
            # Weight corruption still can't localize as well as `_report_bad_grads`,
            # which fires one step earlier on the gradients themselves.
            print(f'[nan-debug] {len(bad_params)}/{len(all_params)} parameter '
                  f'tensors are NaN/Inf; nearest the loss: {bad_params[-3:][::-1]}')
            if len(bad_params) == len(all_params):
                print('[nan-debug] EVERY parameter is corrupted — this is at '
                      'least one optimizer step downstream of the origin, so '
                      'these names carry no information about where it started. '
                      'The `[grad-nan]` report from the failing step does.')
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

    def _apply_optimizer_step(self, log):
        """Unscale (fp16), clip, step, advance the schedule, close the cycle.

        Called once per `accum_steps` micro-batches — and by `flush_accum` for
        a short final cycle — so gradient clipping sees the *complete*
        effective-batch gradient, which is what `--grad_clip 1.0` is calibrated
        against.
        """
        clip_val = self.args.grad_clip if self.args.grad_clip > 0 else float('inf')
        if self.amp_dtype == 'fp16':
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_val)
            # Post-unscale_, so this is the true gradient norm — the same test
            # GradScaler applies internally to decide whether to skip. A handful
            # of consecutive skips is NORMAL here at startup while the scale
            # calibrates down from 65536, which is why the abort threshold is
            # generous. No detailed [grad-nan] report: under loss scaling an
            # overflow is a scale problem, not a numerics bug.
            skipped = not bool(torch.isfinite(grad_norm))
            if skipped:
                self._note_skip(grad_norm)
            self.scaler.step(self.optimizer)   # itself a no-op when non-finite
            self.scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_val)
            # Check BEFORE stepping. `optimizer.step()` with a NaN gradient
            # writes NaN into that parameter, the next forward spreads it
            # through every activation, and one step later all 373 tensors are
            # non-finite — which is why a post-hoc weight scan could only ever
            # report "everything is broken" and name whichever parameter was
            # registered first. Clipping cannot rescue it either: clip_grad_norm_
            # scales by total_norm, and NaN/NaN is NaN. The isfinite() costs no
            # extra sync — train.py already calls .item() on grad_norm to log it.
            skipped = not bool(torch.isfinite(grad_norm))
            if skipped:
                self._report_bad_grads(grad_norm)   # full detail, first time only
                self._note_skip(grad_norm)          # may raise the abort
            else:
                self.optimizer.step()

        if not skipped:
            self._consecutive_skips = 0
        # Record every optimizer step, skipped or not — the window's denominator
        # is steps *attempted*, so the rate is meaningful. `_note_skip` above may
        # already have aborted on the consecutive guard, which is the more urgent
        # signal and trips far sooner.
        self._skip_window.append(1.0 if skipped else 0.0)
        self._check_skip_rate()
        # The scheduler advances on skipped steps too. It is parameterized by
        # `total_steps`, so freezing it would stretch the cosine tail past the
        # end of the run and desynchronize the LR from --epochs; this is also
        # the standard GradScaler + LRScheduler idiom. The skipped step simply
        # contributes no update. Gradients are not zeroed here — `train_step`
        # does that at the head of the next accumulation cycle, so the
        # non-finite values cannot survive into it.
        self.scheduler.step()
        self.global_step += 1
        self._micro_in_cycle = 0
        log['stepped'] = True
        # 0.0/1.0 per optimizer step, so `avg_grad_skipped` on the epoch summary
        # reads directly as this epoch's skip rate.
        log['grad_skipped'] = float(skipped)
        log['lr'] = self.optimizer.param_groups[0]['lr']
        log['grad_norm'] = (grad_norm.detach() if torch.is_tensor(grad_norm)
                            else float(grad_norm))
        return log

    def train_step(self, batch):
        """Run one MICRO-batch. Only every `accum_steps`-th call optimizes.

        The returned log always carries `loss` / `mae_deg` for this
        micro-batch; `lr` / `grad_norm` appear only on calls that actually
        stepped, flagged by `log['stepped']`.
        """
        self.net.train()
        # Zero only at the start of an accumulation cycle — mid-cycle the
        # gradients from previous micro-batches must survive.
        if self._micro_in_cycle == 0:
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

            # Divide so the accumulated gradient is the MEAN over the effective
            # batch rather than its sum; `log['loss']` stays the unscaled
            # per-micro-batch loss so logged values are accum-independent.
            scaled_loss = loss / self.accum_steps
            if self.amp_dtype == 'fp16':
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        self._micro_in_cycle += 1
        log['stepped'] = False
        if self._micro_in_cycle >= self.accum_steps:
            self._apply_optimizer_step(log)
        return log

    def flush_accum(self):
        """Apply a partial accumulation cycle left over at the end of an epoch.

        Without this, the trailing `len(loader) % accum_steps` micro-batches
        would have their gradients silently discarded by the next cycle's
        `zero_grad`. Their contribution is under-weighted (divided by
        `accum_steps` rather than the true short-cycle count), which is the
        standard trade-off and affects at most `accum_steps - 1` micro-batches
        per epoch.

        Returns the step's log, or None if the epoch ended on a cycle boundary.
        """
        if self._micro_in_cycle == 0:
            return None
        return self._apply_optimizer_step({})

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
