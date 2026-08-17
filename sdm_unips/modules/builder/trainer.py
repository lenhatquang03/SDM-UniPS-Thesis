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

Resuming an interrupted run
---------------------------
A thesis run spans days, so a crash at epoch 50 must not cost 50 epochs.
`save` therefore writes the *entire* run state — weights, optimizer moments,
scheduler position, GradScaler scale, skip counters, every RNG stream, and the
training loop's own bookkeeping (epoch, best val loss, patience counter,
elapsed seconds) — and `resume_from` restores all of it.

Restoring only the weights, which is what this file used to do, is worse than
useless for a mid-run restart: a fresh AdamW has no moment estimates and a
fresh LambdaLR restarts at step 0, so the first resumed step lands at the
warmup LR with no gradient history and undoes much of what the run had
learned.

Two distinct modes, chosen by inspecting the file:

* **full resume** — the checkpoint carries an `optimizer` entry, so it came
  from `Trainer.save`. Everything is restored and training continues at the
  next epoch.
* **warm start** — a bare `state_dict` (`*.pytmodel`, e.g. the upstream
  released weights). Weights only, `strict=False`, fresh optimizer, epoch 0.

Which one happened is printed loudly, because silently warm-starting a run
that was meant to resume looks like normal training until the loss curve is
read days later.
"""

import glob
import math
import os
import random
import re
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch

from modules.model import model
from modules.model.model_utils import mode_change, get_n_params
from modules.loss import losses


# Bumped when the checkpoint payload changes shape. `resume_from` accepts
# older versions and fills in what it can, so a run started before a format
# change can still be resumed.
CHECKPOINT_FORMAT_VERSION = 3


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


def _wsd_with_warmup(optimizer, warmup_steps, bounds, min_lr_ratio=0.01):
    """Warmup-Stable-Decay: linear warmup -> CONSTANT lr -> decay to
    `min_lr_ratio`.

    Unlike cosine, the LR at step `s` does not depend on the total length of
    the run until the decay begins, so the run is genuinely extendable: a
    resume may lengthen or shorten `--epochs` without retroactively changing
    the LR of any step already taken. That is the whole reason this schedule
    exists here -- see the note in CLAUDE.md on why a cosine cycle whose
    length does not match the actual run underperforms.

    `bounds` is a MUTABLE dict {'start', 'end'} rather than two captured ints,
    so `Trainer.set_decay_from` can branch the decay at an arbitrary step on a
    resume without rebuilding the scheduler (rebuilding would lose the restored
    `last_epoch`). The closure reads it on every call.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        start, end = bounds['start'], bounds['end']
        if step < start:
            return 1.0                      # stable phase: LR does not move
        if step >= end:
            return min_lr_ratio
        progress = (step - start) / max(1, end - start)
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

        # NOTE: resuming is NOT done here. It has to happen after the optimizer
        # and scheduler exist, or their state cannot be restored — which is
        # exactly the bug this used to have. `train.py` calls `resume_from`
        # once construction is complete.

        self.optimizer = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad],
            lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
        )
        warmup_steps = int(args.warmup_epochs * steps_per_epoch)
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = float(getattr(args, 'min_lr_ratio', 0.01))
        # Populated only by the wsd branch; None means "this schedule has no
        # branchable decay", which is what `set_decay_from` refuses to touch.
        self._decay_bounds = None
        self.decay_len = 0
        # True once `set_decay_from` has moved the ramp off the point the args
        # derive. Only a BRANCHED ramp has to be persisted: a derived one is a
        # pure function of the args and rebuilds identically, and restoring it
        # would break the extendability that is the whole point of wsd.
        self._decay_branched = False
        if args.lr_schedule == 'cosine':
            self.scheduler = _cosine_with_warmup(
                self.optimizer, warmup_steps, total_steps,
                min_lr_ratio=self.min_lr_ratio,
            )
        elif args.lr_schedule == 'wsd':
            # Length of the decay ramp, as a fraction of the planned run. The
            # START is either explicit (--decay_from_step) or derived so the
            # ramp lands exactly at the end of --epochs.
            self.decay_len = max(1, int(round(
                float(getattr(args, 'decay_fraction', 0.2)) * total_steps)))
            explicit = int(getattr(args, 'decay_from_step', 0) or 0)
            start = explicit if explicit > 0 else max(
                warmup_steps, total_steps - self.decay_len)
            self._decay_bounds = {'start': start, 'end': start + self.decay_len}
            self.scheduler = _wsd_with_warmup(
                self.optimizer, warmup_steps, self._decay_bounds,
                min_lr_ratio=self.min_lr_ratio,
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
        if self._decay_bounds is not None:
            self._print_decay_plan(total_steps, steps_per_epoch)

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
        # Evaluation pixel sampling: seeded per item from this base, counter
        # reset by `begin_eval_pass()`. Offset from --seed so it cannot collide
        # with the dataset's own per-scene render seeds.
        self.eval_seed = int(getattr(args, 'seed', 42)) + 10_007
        self._eval_item = 0
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

    # -- WSD decay branch ---------------------------------------------------

    def _print_decay_plan(self, total_steps, steps_per_epoch):
        b = self._decay_bounds
        spe = max(1, steps_per_epoch)
        print(f'[Trainer] WSD: warmup 0->{self.warmup_steps} '
              f'(epoch {self.warmup_steps / spe:.2f}) | constant lr to step '
              f'{b["start"]} (epoch {b["start"] / spe:.2f}) | decay '
              f'{self.decay_len} steps to step {b["end"]} '
              f'(epoch {b["end"] / spe:.2f}), ending at '
              f'{self.min_lr_ratio:g} x lr')
        if b['end'] > total_steps:
            # Not fatal, but the run would stop mid-ramp and never reach
            # min_lr, which is exactly the annealed endpoint the schedule
            # exists to produce.
            print(f'[Trainer] WARNING: the decay ends at step {b["end"]} but '
                  f'the run is only {total_steps} steps. The LR will be cut '
                  f'off at {self.lr_at(total_steps):.3e} instead of reaching '
                  f'{self.min_lr_ratio * self.args.lr:.3e}. Raise --epochs to '
                  f'>= {math.ceil(b["end"] / spe)} or lower --decay_fraction.')
        elif b['end'] < total_steps:
            print(f'[Trainer] NOTE: {total_steps - b["end"]} steps '
                  f'({(total_steps - b["end"]) / spe:.2f} epochs) run at the '
                  f'floor lr after the decay completes.')

    def lr_at(self, step):
        """The LR this schedule would produce at `step` (for reporting)."""
        lam = self.scheduler.lr_lambdas[0]
        return float(self.args.lr) * float(lam(step))

    def set_decay_from(self, step):
        """Branch the WSD decay so it begins at `step` and runs `decay_len`.

        Called on a resume with --decay_now. This is the operation cosine
        cannot express: it changes only the FUTURE of the schedule, leaving
        every step already taken at the LR it was actually taken at, so the
        resumed run is a legitimate continuation rather than a re-shaping.

        Returns the new (start, end). The LR is re-applied to the optimizer
        immediately, so the first resumed step does not run one step behind at
        the restored constant-phase LR.
        """
        if self._decay_bounds is None:
            raise ValueError('--decay_now requires --lr_schedule wsd '
                             f'(this run uses {self.args.lr_schedule}).')
        start = max(int(step), self.warmup_steps)
        self._decay_bounds['start'] = start
        self._decay_bounds['end'] = start + self.decay_len
        # Mark it so `save` persists the resolved endpoints. Without this the
        # branch lives only in the closure's `bounds` dict and dies with the
        # process: LambdaLR.state_dict() carries `last_epoch`/`_last_lr` and
        # stores None for a lambda that is a plain function, so a resume would
        # rebuild `bounds` from args that never recorded the branch and the LR
        # would jump back to the constant phase.
        self._decay_branched = True
        lam = self.scheduler.lr_lambdas[0]
        scale = float(lam(self.scheduler.last_epoch))
        for group, base in zip(self.optimizer.param_groups,
                               self.scheduler.base_lrs):
            group['lr'] = base * scale
        self.scheduler._last_lr = [g['lr'] for g in self.optimizer.param_groups]
        return start, self._decay_bounds['end']

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

    def begin_eval_pass(self):
        """Reset the deterministic evaluation pixel sampler.

        Call once before each validation / test sweep. Pixel sets are seeded
        from a per-item counter that restarts here, so every sweep scores the
        same pixels as every other sweep -- across epochs, across runs, and
        across model variants.
        """
        self._eval_item = 0

    @torch.no_grad()
    def _eval_sample_ids(self, M, decoder_resolution):
        """Pick evaluation pixels OUTSIDE the model, uniformly over the mask.

        This is the other half of the A/B fairness contract (see
        `Net.sample_train_pixels`). Evaluation must never route through the
        model's own sampler: if it did, swapping in a saliency sampler would
        change *which pixels the metric is computed on*, and the val curves of
        two variants would no longer be measuring the same thing.

        Determinism is per *item*, not per batch: the seed is `eval_seed +
        item_index`, and items are visited in dataset order (the eval loaders
        use `shuffle=False`), so the draw for a given scene is independent of
        `--batch_size`, of the worker count, and of how much RNG the training
        loop consumed beforehand. A CPU generator is used so the draw is also
        independent of the device.
        """
        M_dec = torch.nn.functional.interpolate(
            M, size=(decoder_resolution, decoder_resolution), mode='nearest')
        B = M_dec.shape[0]
        HW = decoder_resolution * decoder_resolution
        m = int(self.args.pixel_samples)
        out = torch.zeros(B, m, dtype=torch.long, device=M_dec.device)
        gen = torch.Generator()   # CPU: same stream regardless of device
        for b in range(B):
            gen.manual_seed((self.eval_seed + self._eval_item) % (2 ** 31 - 1))
            self._eval_item += 1
            m_ = M_dec[b, :, :, :].reshape(-1, HW).permute(1, 0)
            valid_ids = torch.nonzero(m_ > 0, as_tuple=False)[:, 0]
            n_valid = int(valid_ids.numel())
            if n_valid == 0:
                continue          # leave zeros; loss masking discards them
            if n_valid >= m:
                sel = torch.randperm(n_valid, generator=gen)[:m]
            else:
                sel = torch.randint(0, n_valid, (m,), generator=gen)
            out[b] = valid_ids[sel.to(valid_ids.device)]
        return out

    def _forward_losses(self, batch, deterministic_eval=False):
        I, N, M, n_imgs = self._move_batch(batch)
        H = I.shape[2]
        dec_res = torch.full((I.shape[0], 1), H, dtype=torch.long, device=self.device)
        can_res = torch.full((I.shape[0], 1), self.args.canonical_resolution,
                             dtype=torch.long, device=self.device)

        # Training lets the model choose (that policy is what the thesis
        # varies); evaluation hands the model a fixed, model-independent set.
        sample_ids = self._eval_sample_ids(M, H) if deterministic_eval else None

        pred_n, sample_idx, _ = self.net(
            I, M, n_imgs, decoder_resolution=dec_res,
            canonical_resolution=can_res, training=True, sample_ids=sample_ids,
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
        """One validation / test batch, scored on model-independent pixels.

        `begin_eval_pass()` must have been called at the head of the sweep.
        """
        self.net.eval()
        with self._autocast():
            loss, log = self._forward_losses(batch, deterministic_eval=True)
        if not torch.isfinite(loss):
            self._report_nan(batch)
        return log

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    @staticmethod
    def _atomic_save(payload, path):
        """`torch.save` to a temp file, then rename over the target.

        A rename within one filesystem is atomic, so an interrupted save can
        never leave a truncated `.pt` behind. That matters more here than
        usual: the file most likely to be half-written when a machine dies is
        the newest one, which is precisely the one `--resume` will pick.
        """
        tmp = path + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _capture_rng(self):
        """Snapshot every RNG stream the training loop draws from.

        Without this, a resumed run re-derives its data order and pixel
        samples from wherever the RNGs happen to sit, so an interrupted run is
        no longer the same experiment as an uninterrupted one. Kept on CPU:
        `torch.load(map_location=cuda)` would otherwise hand `set_rng_state` a
        CUDA tensor and raise.
        """
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        }

    def restore_rng(self, rng):
        """Reinstate the RNG streams captured by `_capture_rng`.

        Call immediately before the training loop, once the datasets and
        loaders have been constructed — building them consumes numpy draws,
        which would otherwise overwrite what was just restored.
        """
        if not rng:
            return
        try:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'].cpu())
            cuda_states = rng.get('cuda')
            if cuda_states and torch.cuda.is_available():
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
                else:
                    # Resumed on a host with a different GPU count: seed what
                    # exists rather than raising. The run stays valid, it is
                    # just no longer bit-identical to the original.
                    print(f'[Trainer] RNG: checkpoint holds {len(cuda_states)} '
                          f'CUDA state(s) but this host has '
                          f'{torch.cuda.device_count()}; restoring device 0 only.')
                    torch.cuda.set_rng_state(cuda_states[0].cpu(), 0)
            print('[Trainer] RNG streams restored (python/numpy/torch/cuda).')
        except (KeyError, TypeError, RuntimeError) as exc:
            print(f'[Trainer] WARNING: could not restore RNG state ({exc}). '
                  f'Training continues, but the resumed data order will differ '
                  f'from an uninterrupted run.')

    def _sched_state(self):
        """The RESOLVED decay endpoints, when the ramp has been branched.

        `args` cannot express this on its own: `--decay_from_step` would pin
        `start`, but `decay_len` is derived from `--epochs`, which `--decay_now`
        truncates — so a faithful relaunch off the resumed config would rebuild
        a shorter ramp. Storing both endpoints as integers makes the restore
        independent of `--epochs` and `--decay_fraction` entirely.

        None for a derived (unbranched) ramp, which is reproducible from args
        and must stay extendable.
        """
        if self._decay_bounds is None or not self._decay_branched:
            return None
        return {'decay_start': int(self._decay_bounds['start']),
                'decay_end': int(self._decay_bounds['end']),
                'branched': True}

    def restore_decay_bounds(self, start, end):
        """Re-apply persisted decay endpoints and the LR they imply.

        Mirrors the tail of `set_decay_from`: the optimizer's LR is refreshed
        immediately so the first resumed step does not run one step behind at
        the pre-restore value.
        """
        if self._decay_bounds is None:
            return None
        start, end = int(start), int(end)
        self._decay_bounds['start'] = start
        self._decay_bounds['end'] = end
        self.decay_len = max(1, end - start)
        self._decay_branched = True
        lam = self.scheduler.lr_lambdas[0]
        scale = float(lam(self.scheduler.last_epoch))
        for group, base in zip(self.optimizer.param_groups,
                               self.scheduler.base_lrs):
            group['lr'] = base * scale
        self.scheduler._last_lr = [g['lr'] for g in self.optimizer.param_groups]
        return start, end

    def _args_snapshot(self):
        """JSON-ish copy of the run's arguments, for the resume compat check."""
        return {k: v for k, v in vars(self.args).items()
                if isinstance(v, (str, int, float, bool, type(None)))}

    def save(self, ckpt_dir, tag, state=None):
        """Write a complete, resumable checkpoint.

        `state` carries the *training loop's* bookkeeping (epoch,
        best_val_loss, patience counter, elapsed seconds, DataLoader shuffle
        generator). The Trainer does not own those, but they have to travel
        with the optimizer state: without them a resumed run restarts the
        schedule at epoch 0, forgets its best val loss, and overwrites
        `best.pt` with a worse model.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'{tag}.pt')
        payload = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'target': self.target,
            'global_step': self.global_step,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
            'trainer_state': {
                'global_step': self.global_step,
                'skipped_steps': self.skipped_steps,
                'consecutive_skips': self._consecutive_skips,
                'skip_window': list(self._skip_window),
                'micro_in_cycle': self._micro_in_cycle,
            },
            'rng': self._capture_rng(),
            'sched_state': self._sched_state(),
            'args': self._args_snapshot(),
            'loop_state': dict(state) if state else None,
        }
        self._atomic_save(payload, path)
        # Drop-in copy for the inference Builder.
        legacy = os.path.join(ckpt_dir, f'{self.target}.pytmodel')
        self._atomic_save(self.net.state_dict(), legacy)
        return path

    def resume_from(self, path, weights_only=False):
        """Restore a run from `path`. Returns the loop state, or None.

        A returned dict means a **full resume** happened and the caller must
        continue from `state['epoch'] + 1`. `None` means only weights were
        loaded (**warm start**) and the caller starts from scratch at epoch 0.

        `weights_only=True` forces the warm-start reading of a full
        checkpoint — for deliberately beginning a *new* run from a previous
        run's weights, where inheriting stale optimizer moments and an
        already-finished LR schedule would be wrong.
        """
        # weights_only=False is explicit: newer torch defaults it to True and
        # would reject the RNG/args payload, which is not a plain tensor dict.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        is_full = (isinstance(ckpt, dict) and 'model' in ckpt
                   and 'optimizer' in ckpt)
        if not is_full or weights_only:
            sd = ckpt['model'] if (isinstance(ckpt, dict) and 'model' in ckpt) else ckpt
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            why = ('forced by --resume_weights_only' if (is_full and weights_only)
                   else 'file holds weights only, no optimizer state')
            print(f'[Trainer] WARM START from {path} ({why}).')
            print(f'[Trainer]   weights loaded | missing={len(missing)} '
                  f'unexpected={len(unexpected)} | optimizer, LR schedule and '
                  f'epoch counter all start FRESH at 0.')
            if len(missing) > 0:
                # strict=False is what makes a warm start possible at all, but
                # it is also what silently loaded *nothing* when this code was
                # handed a full training checkpoint. Report what actually landed.
                print(f'[Trainer]   WARNING: {len(missing)} parameter tensors '
                      f'were absent from the file and keep their random init, '
                      f'e.g. {missing[:3]}')
            return None

        self.net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        if ckpt.get('scaler') is not None and self.scaler is not None:
            self.scaler.load_state_dict(ckpt['scaler'])

        ts = ckpt.get('trainer_state') or {}
        self.global_step = int(ts.get('global_step', ckpt.get('global_step', 0)))
        self.skipped_steps = int(ts.get('skipped_steps', 0))
        self._consecutive_skips = int(ts.get('consecutive_skips', 0))
        self._skip_window = deque(ts.get('skip_window', []),
                                  maxlen=self.skip_rate_window)
        # NOT restored from the checkpoint: a resume always begins at an epoch
        # boundary with no accumulated gradients in flight, so the next
        # micro-batch must open a fresh cycle (and `zero_grad`). Carrying a
        # stale count over would make the first cycle short and let it step on
        # a partial effective batch.
        self._micro_in_cycle = 0

        loop = dict(ckpt.get('loop_state') or {})
        if 'epoch' not in loop:
            # Pre-v2 checkpoint, or a step_* one saved mid-epoch. global_step
            # counts optimizer steps, so integer division recovers how many
            # epochs completed.
            done = self.global_step // max(1, self.steps_per_epoch)
            loop['epoch'] = done - 1        # `epoch` means "last COMPLETED epoch"
            print(f'[Trainer] checkpoint carries no loop state; inferring '
                  f'{done} completed epoch(s) from global_step='
                  f'{self.global_step}. best_val_loss and the early-stopping '
                  f'counter restart from scratch, so best.pt can be overwritten '
                  f'by a worse epoch.')
        loop['rng'] = ckpt.get('rng')
        # A BRANCHED decay ramp is authoritative over whatever the current
        # args would derive: --epochs and --decay_fraction on the relaunch are
        # irrelevant once the endpoints are known.
        sched = ckpt.get('sched_state') or {}
        loop['sched_restored'] = False
        if sched.get('branched') and self._decay_bounds is not None:
            s, e = self.restore_decay_bounds(sched['decay_start'],
                                             sched['decay_end'])
            loop['sched_restored'] = True
            loop['decay_start'] = s
            loop['decay_end'] = e
            print(f'[Trainer]   decay ramp restored from the checkpoint: '
                  f'steps {s} -> {e} (lr {self.lr_at(s):.3e} -> '
                  f'{self.lr_at(e):.3e}); --epochs / --decay_fraction on this '
                  f'launch do not reshape it.')

        loop['args'] = ckpt.get('args') or {}
        loop['path'] = path
        loop['format_version'] = int(ckpt.get('format_version', 1))

        print(f'[Trainer] FULL RESUME from {path}')
        print(f'[Trainer]   model + optimizer + scheduler + scaler restored | '
              f'global_step={self.global_step} | '
              f'lr={self.optimizer.param_groups[0]["lr"]:.3e} | '
              f'skipped_steps={self.skipped_steps}')
        return loop

    def load(self, path):
        """Load weights + optimizer/scheduler for the final test evaluation.

        Distinct from `resume_from`: this is the end-of-run `best.pt` load,
        which only needs the model in the right state and deliberately does
        NOT touch the loop's epoch / best-val bookkeeping.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        if ckpt.get('scaler') is not None and self.scaler is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
        self.global_step = ckpt.get('global_step', 0)


def _epoch_index(path):
    """Epoch number encoded in an `epoch_<n>.pt` filename, or -1."""
    m = re.search(r'epoch_(\d+)\.pt$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _step_index(path):
    """Optimizer step encoded in a `step_<n>.pt` filename, or -1."""
    m = re.search(r'step_(\d+)\.pt$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_latest_checkpoint(ckpt_dir):
    """Newest resumable checkpoint in `ckpt_dir`, or None.

    The most recently WRITTEN of {newest `epoch_*.pt`, newest `step_*.pt`},
    then `final.pt`. `best.pt` is deliberately never chosen: it is a
    *selection* artifact and normally lags the training frontier, so resuming
    from it would silently replay epochs.

    `step_*.pt` is included because with `--ckpt_every` on by default it is the
    newest state on disk after a mid-epoch crash, and ignoring it would throw
    away every step since the last epoch boundary — hours, on the full pool —
    which is the entire reason those checkpoints are written. It is never a
    worse choice than the epoch checkpoint it follows: both carry the same
    `loop_state` epoch (a step checkpoint stores `epoch - 1`, so its epoch is
    replayed from the top either way), but the step file's model and optimizer
    are further along.

    Chosen by mtime rather than by parsed number because the two series are not
    comparable numerically, and mtime correctly prefers a fresh epoch
    checkpoint over a `step_*.pt` left behind by an earlier resume generation.
    """
    candidates = []
    epochs = glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt'))
    if epochs:
        candidates.append(max(epochs, key=_epoch_index))
    steps = glob.glob(os.path.join(ckpt_dir, 'step_*.pt'))
    if steps:
        candidates.append(max(steps, key=_step_index))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    final = os.path.join(ckpt_dir, 'final.pt')
    return final if os.path.isfile(final) else None


def prune_checkpoints(ckpt_dir, keep_last=3, protect=(), keep_last_steps=None):
    """Rotate `epoch_*.pt` (and, when `keep_last_steps` is set, `step_*.pt`).

    Keeps the most recent `keep_last` epoch checkpoints and anything in
    `protect` (e.g. {'best'}). `final.pt` and the legacy `<target>.pytmodel`
    are never touched.

    `step_*.pt` is rotated separately and only when `keep_last_steps` is not
    None: it is mid-epoch crash insurance, so a handful is enough, and with
    `--ckpt_every` on by default an unrotated series would fill the disk (one
    full checkpoint carries the model plus both AdamW moments). Passing None
    preserves the historical never-delete behaviour for callers that want it.

    Ordering is by the parsed epoch/step NUMBER, not by filename. Sorting the
    paths as strings put `epoch_10.pt` before `epoch_7.pt`
    (['epoch_10.pt', 'epoch_11.pt', 'epoch_7.pt', 'epoch_8.pt', 'epoch_9.pt']),
    so from epoch 10 onward the newest checkpoint was deleted the moment it was
    written and the disk froze at epochs 7/8/9 — which would have made
    `--resume` replay the same epoch forever.
    """
    protected = {os.path.join(ckpt_dir, f'{name}.pt') for name in protect}

    def rotate(pattern, key, keep_n):
        paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)), key=key)
        if len(paths) <= keep_n:
            return
        keep = set(paths[len(paths) - keep_n:]) | protected
        for p in paths:
            if p not in keep and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    rotate('epoch_*.pt', _epoch_index, keep_last)
    if keep_last_steps is not None:
        rotate('step_*.pt', _step_index, max(0, int(keep_last_steps)))
