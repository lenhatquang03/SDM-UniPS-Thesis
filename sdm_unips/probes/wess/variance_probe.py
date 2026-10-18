"""WESS variance probe: does the importance-weighted WESS draw give a less
noisy gradient than Model A's uniform draw, at fixed weights?

Pre-registered (context_handoff.md, approved 2026-10-18 with two amendments:
dropout off via eval(), and the interpretation grid row for "Run 1 worse
than B1"):

  Claim     At fixed theta, the WESS-IW minibatch gradient has lower variance
            than the uniform minibatch gradient. Both are unbiased for the same
            mean (the uniform loss over the mask), so variance is the only thing
            the sampler can change.
  Metric    Per scene, V = (1/R) * sum_r ||g_r - g_bar||^2 over R repeated pixel
            draws, g_r = gradient of the step loss w.r.t. every trainable
            parameter for draw r (same render, same theta). Ratio
            V_arm / V_uniform per scene; summarised as the geometric mean over
            scenes, with the SE of the mean log-ratio.
  Arms      uniform  Model A's draw: 2048 pixels uniformly without replacement
            iw       WESS tau=1 lam=0.25 S0 energy, with replacement, loss x 1/(n p)
            shuffled same as iw with the 64x64 energy map's cells permuted
                     (still unbiased, no geometric content)
  Null      ratio ~ 1; shuffled >= 1.
  Decision  SUPPORTED iff GM(iw) < 1 AND GM(shuffled) >= 1 AND
            mean_s[log r_iw - log r_shuffled] < -2 SE (paired over scenes).
            Otherwise the thesis cannot call WESS a variance-reduction sampler.
  Reading   ratio(iw) >= 1          -> energy is not a gradient-size proxy; the
                                       weights added noise, which explains
                                       B2-IW < B2 tau=1.
            ratio(iw) < 1, shuf >= 1 -> WESS does lower gradient noise yet still
                                       loses; the cost is elsewhere (clustering),
                                       which is Model C's question.

Integrity checks printed per checkpoint:
  * unbiasedness: D = ||g_bar_arm - g_bar_uniform||^2 / ((V_arm + V_uniform)/R).
    Under equal means E[D] = 1; a median far above ~2 means an arm is biased.
  * importance-weight mean per draw ~ 1, max <= 1/lam.
  * eval() is on (dropout off), so the draw is the only source of randomness.

Checkpoints (one run each):
  late   B1 best.pt                      /workspace/runs/modelB1_wtconv/checkpoints/best.pt
  early  Run 1 (B2-IW) epoch 19, from HF /workspace/runs/modelB2_iw/early_ckpt/epoch_19/sdm-ckpt/modelB2_iw/latest.pt

Run on the training box only (full forward + backward at 512, K=10):
  python -u sdm_unips/probes/wess/variance_probe.py --checkpoint <ckpt> \
      --out_dir <dir> [--num_scenes 20 --draws 16]
  python sdm_unips/probes/wess/variance_probe.py --summarize <dir>/variance_probe.jsonl
Dataset flags default to the probe machine's (PROBE_DATA_DEFAULTS), exactly as
wess_site_probe.py; pass --hdlong_dir/--polarps_dir/--scene_manifest if the
roots differ.
"""

from __future__ import print_function, division

import os

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import json
import math
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..'))

ARMS = ('uniform', 'iw', 'shuffled')


# ---------------------------------------------------------------------------
# Summary (stdlib only, so it also runs off the box)
# ---------------------------------------------------------------------------
def summarize(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if r.get('kind') == 'scene']
    if not recs:
        raise SystemExit(f'no scene records in {path}')
    li = [math.log(r['V']['iw'] / r['V']['uniform']) for r in recs]
    ls = [math.log(r['V']['shuffled'] / r['V']['uniform']) for r in recs]
    d = [a - b for a, b in zip(li, ls)]
    n = len(recs)

    def mse(x):
        return st.mean(x), (st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float('nan'))

    mi, si = mse(li)
    ms, ss = mse(ls)
    md, sd = mse(d)
    gm_i, gm_s = math.exp(mi), math.exp(ms)
    print(f'checkpoint: {recs[0]["checkpoint"]}')
    print(f'{n} scenes, R = {recs[0]["draws"]} draws per arm\n')
    print(f'  V_iw / V_uniform        GM {gm_i:.3f}   (mean log {mi:+.3f} +- {si:.3f})')
    print(f'  V_shuffled / V_uniform  GM {gm_s:.3f}   (mean log {ms:+.3f} +- {ss:.3f})')
    print(f'  log r_iw - log r_shuf   {md:+.3f} +- {sd:.3f}  (paired over scenes)\n')
    Di = st.median(r['D']['iw'] for r in recs)
    Ds = st.median(r['D']['shuffled'] for r in recs)
    wmean = st.mean(r['w_mean']['iw'] for r in recs)
    wmax = max(r['w_max']['iw'] for r in recs)
    print('INTEGRITY')
    print(f'  unbiasedness D (median, expect ~1): iw {Di:.2f}   shuffled {Ds:.2f}'
          f'   {"OK" if max(Di, Ds) < 2.5 else "CHECK: an arm looks biased"}')
    print(f'  iw weights: mean over scenes {wmean:.3f} (expect ~1), max {wmax:.3f} (expect <= 4)\n')
    ok = gm_i < 1 and gm_s >= 1 and md < -2 * sd
    print('DECISION (pre-registered):')
    print(f'  GM(iw) < 1: {gm_i < 1}   GM(shuffled) >= 1: {gm_s >= 1}   '
          f'paired gap < -2 SE: {md < -2 * sd}')
    if ok:
        print('  -> SUPPORTED: WESS-IW lowers gradient variance; the energy map carries '
              'gradient-size information (shuffled does not).')
    elif gm_i >= 1:
        print('  -> NOT SUPPORTED: WESS-IW does not lower gradient variance. The energy map is '
              'not a gradient-size proxy; the weights add noise (consistent with B2-IW < B2 tau=1).')
    else:
        print('  -> NOT SUPPORTED: the reduction is not distinguishable from the shuffled control.')


# ---------------------------------------------------------------------------
# Probe (needs torch + the training box)
# ---------------------------------------------------------------------------
class Welford:
    """Streaming mean vector + summed squared deviation (float64, on device)."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, 0.0

    def add(self, g):
        self.n += 1
        if self.mean is None:
            self.mean = g.clone()
            return
        delta = g - self.mean
        self.mean += delta / self.n
        self.m2 += float((delta * (g - self.mean)).sum())

    def var_trace(self):
        """(1/R) * sum_r ||g_r - g_bar||^2, the population form used in the spec."""
        return self.m2 / self.n


def run(args):
    import numpy as np
    import torch

    from modules.loss import losses
    from modules.model import model as model_mod
    from modules.model import wess
    from modules.io.dataloader.mixed import build_mixed_split, MixedEvalDataset
    import tap_sites
    from wess_probe import load_weights, shuffle_map
    from wess_site_probe import prepare

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'variance_probe.jsonl')
    device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = model_mod.Net(int(args.pixel_samples), device).to(device)
    load_weights(net, args.checkpoint, device)
    net.with_grad()      # every parameter requires grad
    net.eval()           # amendment (a): dropout off, so the draw is the only noise
    params = [p for p in net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    amp = torch.bfloat16 if args.amp_dtype == 'bf16' else (
        torch.float16 if args.amp_dtype == 'fp16' else None)

    _, val_set, _ = build_mixed_split(args)
    eval_set = MixedEvalDataset(args, val_set.scenes, n_trials=1, seed=args.seed + 5_000_011,
                                subset_name='VarianceProbe/val', noun='probe', announce=True)
    n_scenes = min(args.num_scenes, len(eval_set))
    m = int(args.pixel_samples)
    print(f'[var-probe] {n_scenes} scenes x {args.draws} draws x {len(ARMS)} arms | '
          f'{n_params:,} params | tau {args.tau} lam {args.lam} | amp {args.amp_dtype}', flush=True)

    def energy_map(sc):
        """S0 energy, identical to the shipped sampler's (tap_sites.site_map('S0'))."""
        ids0 = sc['valid_ids'][:m]
        with tap_sites.LevelTaps(net, [(0, 0, 0)]) as taps, torch.no_grad():
            net(sc['I'], sc['M'], sc['n_imgs'].to(device), decoder_resolution=sc['dec'],
                canonical_resolution=sc['can'], training=True, sample_ids=ids0[None])
            learned = tap_sites.learned_level_maps(taps, sc['n'], sc['mosaic'])
        return tap_sites.site_map('S0', learned, {}, sc['M'][0, 0], args.top_k)

    def draw(arm, sc, E):
        if arm == 'uniform':
            perm = torch.randperm(sc['valid_ids'].numel(), device=sc['valid_ids'].device)
            return sc['valid_ids'][perm[:m]], None
        ids, _, w = wess.wess_sample_train(E, sc['M'][0, 0], sc['valid_ids'], sc['H'], sc['H'], m,
                                           tau=args.tau, lam=args.lam,
                                           erode_cells=args.erode_cells, return_weights=True)
        return ids, w

    def grad_vector(sc, ids, w):
        net.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=amp, enabled=amp is not None):
            pred, idx, _ = net(sc['I'], sc['M'], sc['n_imgs'].to(device),
                               decoder_resolution=sc['dec'], canonical_resolution=sc['can'],
                               training=True, sample_ids=ids[None])
        loss = losses.normal_loss(pred.float(), sc['N_dev'], sc['M'], idx,
                                  weights=None if w is None else w[None])
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(grads, params)]).double()

    t0 = time.time()
    with open(out_path, 'a') as fh:
        for i in range(n_scenes):
            sc = prepare(eval_set, i, device, args.canonical_resolution)
            if sc['valid_ids'].numel() < m:
                print(f'[var-probe] scene {i}: {sc["valid_ids"].numel()} valid pixels < {m}, skipped')
                continue
            E = energy_map(sc)
            g_sh = torch.Generator()
            g_sh.manual_seed(tap_sites.shuffle_seed(args.seed, i, 'S0'))
            E_arm = {'uniform': None, 'iw': E, 'shuffled': shuffle_map(E, g_sh)}

            acc = {a: Welford() for a in ARMS}
            w_mean = {a: [] for a in ARMS}
            w_max = {a: 0.0 for a in ARMS}
            ts = time.time()
            for ai, arm in enumerate(ARMS):
                for r in range(args.draws):
                    torch.manual_seed((args.seed * 1_000_003 + i * 9_973 + ai * 101 + r) % (2 ** 31 - 1))
                    ids, w = draw(arm, sc, E_arm[arm])
                    if w is not None:
                        w_mean[arm].append(float(w.mean()))
                        w_max[arm] = max(w_max[arm], float(w.max()))
                    acc[arm].add(grad_vector(sc, ids, w))

            V = {a: acc[a].var_trace() for a in ARMS}
            R = args.draws
            D = {a: float(((acc[a].mean - acc['uniform'].mean) ** 2).sum()) /
                 ((V[a] + V['uniform']) / R) for a in ('iw', 'shuffled')}
            kind, scene_dir = eval_set.scenes[i % eval_set.n_scenes]
            rec = {'kind': 'scene', 'index': i, 'source': kind,
                   'scene': os.path.basename(scene_dir.rstrip('/')),
                   'checkpoint': args.checkpoint, 'draws': R, 'tau': args.tau, 'lam': args.lam,
                   'n_valid': int(sc['valid_ids'].numel()), 'V': V, 'D': D,
                   'grad_mean_norm': {a: float(acc[a].mean.norm()) for a in ARMS},
                   'w_mean': {a: (st.mean(w_mean[a]) if w_mean[a] else 1.0) for a in ARMS},
                   'w_max': {a: (w_max[a] if w_mean[a] else 1.0) for a in ARMS},
                   'sec': round(time.time() - ts, 1)}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            for a in ARMS:
                acc[a].mean = None
            torch.cuda.empty_cache()
            print(f'[var-probe {i + 1}/{n_scenes}] {rec["scene"][:30]:30s} '
                  f'iw/u {V["iw"] / V["uniform"]:.3f}  shuf/u {V["shuffled"] / V["uniform"]:.3f}  '
                  f'D iw {D["iw"]:.2f} shuf {D["shuffled"]:.2f}  ({rec["sec"]}s, '
                  f'total {time.time() - t0:.0f}s)', flush=True)

    print(f'\n[var-probe] records: {out_path}')
    summarize(out_path)


def build_parser():
    import train as train_mod
    from wess_probe import PROBE_DATA_DEFAULTS
    from modules.model import wess
    p = train_mod.build_argparser()
    p.set_defaults(**PROBE_DATA_DEFAULTS)
    g = p.add_argument_group('WESS variance probe')
    g.add_argument('--checkpoint', required=True,
                   help='A .pt training checkpoint or bare .pytmodel built on this branch (WTConv).')
    g.add_argument('--out_dir', required=True)
    g.add_argument('--num_scenes', type=int, default=20)
    g.add_argument('--draws', type=int, default=16, help='R: pixel draws per arm per scene.')
    g.add_argument('--tau', type=float, default=1.0)
    g.add_argument('--lam', type=float, default=wess.DEFAULT_LAM)
    g.add_argument('--erode_cells', type=int, default=wess.DEFAULT_ERODE_CELLS)
    g.add_argument('--top_k', type=int, default=2)
    return p


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--summarize':
        summarize(sys.argv[2])
    else:
        run(build_parser().parse_args())
