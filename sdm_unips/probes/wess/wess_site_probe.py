"""WESS tap-site and sampler-setting screen.

Runs any subset of the sites in `tap_sites.SITES` through the shipped sampler
on held-out scenes, loading Model B1's weights, and records per scene what
`wess_site_summary.py` turns into the screen table and the tap-site gate.

Two modes, mutually exclusive:

* `--target_ess E` -- each site runs at the temperature whose mean ESS is E
  (sites compared at equal concentration; the tap-site screen).
* `--tau T [T ...]` -- fixed temperatures (the lambda / tau rows, and the
  regression against wess_tau_grid_v2).

Every configuration also gets a NULL arm unless `--no_null`: the same site's
map with its cells shuffled, run at the temperature that matches the real
arm's mean ESS. `wess_probe.py --shuffle_energy` matched the null by tau
instead, and its shuffled draw came out more concentrated than the real one
(ESS 0.162 vs 0.388 at tau=1), over-stating the clustering offset. Here the
two arms share scenes, renders and uniform draw, so the net MAE gain is a
paired, per-scene difference.

Two passes over the scenes:

1. render, one forward with every tap attached, the uniform-draw MAE, each
   site's map (and shuffled map), its correlations with GT curvature and with
   the image gradient, and its mean ESS over TAU_GRID;
2. choose the temperatures, then render again, draw at those temperatures,
   and measure tilts and (unless `--no_mae`) the MAE on the drawn pixels.

Maps are cached on the CPU between the passes (~1.3 MB per scene for all nine
sites with their null maps, ~0.6 GB over the 431-scene val split), so both
passes use identical maps.

Outputs in --out_dir: `site_probe.jsonl` (one record per scene),
`tau_search.json` (ESS curves and the chosen temperatures), `run_args.json`.

Dataset flags are train.py's own, so `--config sdm_unips/configs/modelB2_wess.yaml`
supplies the roots, manifest, seed and split of the B2 runs.
"""

from __future__ import print_function, division

import os

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import json
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# sdm_unips/, two levels up -- for `modules` and `train`.
sys.path.append(os.path.join(HERE, '..', '..'))

from modules.io.dataloader.mixed import build_mixed_split, MixedEvalDataset
from modules.loss import losses
from modules.model import model as model_mod
from modules.model import wess

import tap_sites
import train as train_mod
from wess_probe import (curvature_map, observation_gradient, spearman,
                        partial_spearman, shuffle_map, load_weights,
                        resolve_checkpoint)


def build_parser():
    p = train_mod.build_argparser()
    g = p.add_argument_group('WESS tap-site screen')
    g.add_argument('--checkpoint', required=True,
                   help='Model B1 weights (best.pt, final.pt, a .pytmodel, or a '
                        'checkpoints/ directory). This branch builds WTConv blocks.')
    g.add_argument('--out_dir', required=True,
                   help='Where site_probe.jsonl, tau_search.json and run_args.json go.')
    g.add_argument('--num_scenes', type=int, default=0,
                   help='Held-out scenes to probe (0 = the whole split).')
    g.add_argument('--split', default='val', choices=['val', 'test'])
    g.add_argument('--sites', nargs='+', default=['S0'], choices=list(tap_sites.SITES),
                   help='Tap sites to run (see tap_sites.py).')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tau', type=float, nargs='+',
                      help='Fixed temperature(s); one configuration per site per value.')
    mode.add_argument('--target_ess', type=float,
                      help='Run each site at the temperature whose mean ESS is this '
                           '(0.388 = S0 at tau=1, lam=0.25).')
    g.add_argument('--lam', type=float, default=wess.DEFAULT_LAM,
                   help='Uniform mixture weight inside the interior (shipped: 0.25).')
    g.add_argument('--erode_cells', type=int, default=wess.DEFAULT_ERODE_CELLS,
                   help='Silhouette band in S0 grid cells, shared by every site.')
    g.add_argument('--top_k', type=int, default=2,
                   help='Reduction over the K images: mean of the top-k per cell.')
    g.add_argument('--erode', type=int, default=9,
                   help='Mask erosion (px) for the correlation statistics and the '
                        'curvature tilts; the same definition as wess_probe.py.')
    g.add_argument('--no_null', action='store_true',
                   help='Skip the shuffled-map arm.')
    g.add_argument('--no_mae', action='store_true',
                   help='Skip decoding the drawn pixels (tilts and ESS only; much faster).')
    return p


class CurvRef:
    """GT curvature of one scene, and the tilt formulas of wess_probe.scene_statistics."""

    def __init__(self, N_np, M_np, erode):
        self.mask = (M_np[0] > 0).astype(np.uint8)
        self.curv = curvature_map(N_np) * self.mask
        interior = (cv2.erode(self.mask, np.ones((erode, erode), np.uint8), iterations=1)
                    if erode > 0 else self.mask)
        self.sel = interior > 0
        flat, in_mask = self.curv.reshape(-1), self.mask.reshape(-1) > 0
        self.flat_curv = flat
        self.flat_interior = self.sel.reshape(-1)
        self.mean_curv = float(flat[in_mask].mean()) if in_mask.any() else 0.0
        self.top_decile = float(np.percentile(flat[in_mask], 90)) if in_mask.any() else 0.0
        inner = flat[self.flat_interior]
        self.mean_curv_in = float(inner.mean()) if inner.size else 0.0

    def tilts(self, ids):
        ids = np.asarray(ids, np.int64)
        v, inside = self.flat_curv[ids], self.flat_interior[ids]
        t_in = (float(v[inside].mean() / (self.mean_curv_in + 1e-12))
                if inside.any() and self.mean_curv_in > 0 else float('nan'))
        return {'curv_tilt': float(v.mean() / (self.mean_curv + 1e-12)),
                'curv_top10_frac': float((v >= self.top_decile).mean()),
                'curv_tilt_interior': t_in,
                'interior_frac': float(inside.mean())}


def site_correlations(E, H, ref, imgrad, r_gc):
    """rho(E, curvature) and rho(E, |grad I| | curvature) over the eroded interior."""
    if ref.sel.sum() < 32:
        return {'rho_E_curvature': float('nan'), 'rho_E_imgrad_given_curvature': float('nan')}
    E_up = F.interpolate(E[None, None].float().cpu(), size=(H, H), mode='bilinear',
                         align_corners=False).reshape(H, H).numpy()
    e = E_up[ref.sel]
    r_ec = spearman(e, ref.curv[ref.sel])
    r_eg = spearman(e, imgrad[ref.sel])
    return {'rho_E_curvature': r_ec,
            'rho_E_imgrad_given_curvature': partial_spearman(r_eg, r_ec, r_gc)}


def prepare(eval_set, i, device, canonical):
    I_np, N_np, M_np, n_i = eval_set[i]
    I = torch.from_numpy(I_np)[None].to(device)
    M = torch.from_numpy(M_np)[None].to(device)
    H = I.shape[2]
    m_ = M[0].reshape(-1, H * H).permute(1, 0)
    return dict(I_np=I_np, N_np=N_np, M_np=M_np, n=int(n_i), I=I, M=M, H=H,
                N_dev=torch.from_numpy(N_np)[None].to(device),
                n_imgs=torch.tensor([int(n_i)], dtype=torch.long),
                valid_ids=torch.nonzero(m_ > 0, as_tuple=False)[:, 0],
                dec=torch.full((1, 1), H, dtype=torch.long, device=device),
                can=torch.full((1, 1), int(canonical), dtype=torch.long, device=device),
                mosaic=H // int(canonical))


def decode_mae(net, sc, ids, device):
    with torch.no_grad():
        pred, idx, _ = net(sc['I'], sc['M'], sc['n_imgs'].to(device),
                           decoder_resolution=sc['dec'], canonical_resolution=sc['can'],
                           training=True, sample_ids=ids[None])
    return float(losses.angular_error_deg(pred, sc['N_dev'], sc['M'], idx))


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.log_dir is None:
        args.log_dir = args.out_dir
    with open(os.path.join(args.out_dir, 'run_args.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = model_mod.Net(int(args.pixel_samples), device).to(device)
    load_weights(net, resolve_checkpoint(args.checkpoint), device)
    net.eval()
    net.no_grad()

    train_set, val_set, test_set = build_mixed_split(args)
    del train_set
    chosen = val_set if args.split == 'val' else test_set
    if len(chosen.scenes) == 0:
        raise RuntimeError(f'the {args.split} split is empty')
    seed = args.seed + 5_000_011 if args.split == 'val' else args.seed
    eval_set = MixedEvalDataset(args, chosen.scenes, n_trials=1, seed=seed,
                                subset_name=f'WESSSiteProbe/{args.split}',
                                noun='probe', announce=True)
    n_scenes = len(eval_set) if args.num_scenes <= 0 else min(args.num_scenes, len(eval_set))

    arms = ['real'] if args.no_null else ['real', 'null']
    if args.tau:
        cfgs = [dict(name=f'{s}|tau={t:g}', site=s, mode='tau', tau_fixed=float(t))
                for s in args.sites for t in args.tau]
    else:
        cfgs = [dict(name=f'{s}|ess={args.target_ess:g}', site=s, mode='ess', tau_fixed=None)
                for s in args.sites]
    keys = tap_sites.learned_taps_needed(args.sites)
    raw_max = tap_sites.raw_max_level(args.sites)
    n_samp = int(args.pixel_samples)
    print(f'[site-probe] {n_scenes} scene(s) | sites {args.sites} | lam {args.lam} | '
          f'{"tau " + str(args.tau) if args.tau else "target ESS " + str(args.target_ess)} | '
          f'arms {arms} | mae {"off" if args.no_mae else "on"}', flush=True)

    # ---- pass 1: maps, correlations, ESS curves --------------------------
    curves = {(s, a): [] for s in args.sites for a in arms}
    fixed = {(s, t): [] for s in args.sites for t in (args.tau or [])}
    cache = []
    t0 = time.time()
    for i in range(n_scenes):
        sc = prepare(eval_set, i, device, args.canonical_resolution)
        if sc['valid_ids'].numel() < 32:
            cache.append(None)
            print(f'[pass 1 {i + 1}/{n_scenes}] skipped: {sc["valid_ids"].numel()} valid pixels', flush=True)
            continue
        gen = torch.Generator()
        gen.manual_seed((args.seed + i) % (2 ** 31 - 1))
        ids_u = wess.uniform_sample(sc['valid_ids'], n_samp, gen)
        with tap_sites.LevelTaps(net, keys) as taps, torch.no_grad():
            pred_u, idx_u, _ = net(sc['I'], sc['M'], sc['n_imgs'].to(device),
                                   decoder_resolution=sc['dec'], canonical_resolution=sc['can'],
                                   training=True, sample_ids=ids_u[None])
            learned = tap_sites.learned_level_maps(taps, sc['n'], sc['mosaic'])
        mae_u = float(losses.angular_error_deg(pred_u, sc['N_dev'], sc['M'], idx_u))
        with torch.no_grad():
            raw = tap_sites.raw_level_maps(sc['I'], sc['M'], sc['n'], raw_max) if raw_max >= 0 else {}

        mask_hw = sc['M'][0, 0]
        interior_flat = tap_sites.shared_interior(mask_hw, args.erode_cells).reshape(-1)
        ref = CurvRef(sc['N_np'], sc['M_np'], args.erode)
        imgrad = observation_gradient(sc['I_np'], sc['n']) * ref.mask
        r_gc = spearman(imgrad[ref.sel], ref.curv[ref.sel]) if ref.sel.sum() >= 32 else float('nan')

        entry = {'mae_uniform_deg': mae_u, 'maps': {}, 'rho': {}}
        for s in args.sites:
            E = tap_sites.site_map(s, learned, raw, mask_hw, args.top_k)
            arm_maps = {'real': E}
            if 'null' in arms:
                g_sh = torch.Generator()
                g_sh.manual_seed(tap_sites.shuffle_seed(args.seed, i, s))
                arm_maps['null'] = shuffle_map(E, g_sh)
            entry['maps'][s] = {a: m.float().cpu() for a, m in arm_maps.items()}
            entry['rho'][s] = {a: site_correlations(m, sc['H'], ref, imgrad, r_gc)
                               for a, m in arm_maps.items()}
            for a, m in arm_maps.items():
                curves[(s, a)].append([tap_sites.ess_fraction(wess.wess_train_probabilities(
                    m, sc['valid_ids'], interior_flat, sc['H'], sc['H'], tau=float(t), lam=args.lam)[0])
                    for t in tap_sites.TAU_GRID])
            for t in (args.tau or []):
                fixed[(s, t)].append(tap_sites.ess_fraction(wess.wess_train_probabilities(
                    E, sc['valid_ids'], interior_flat, sc['H'], sc['H'], tau=float(t), lam=args.lam)[0]))
        del learned, raw
        cache.append(entry)
        print(f'[pass 1 {i + 1}/{n_scenes}] mae_uniform {mae_u:.2f} deg | '
              f'{time.time() - t0:.0f}s', flush=True)

    if not any(c is not None for c in cache):
        raise RuntimeError('no scene had enough valid pixels')

    # ---- temperatures ------------------------------------------------------
    mean_curve = {k: np.mean(np.asarray(v), axis=0) for k, v in curves.items() if v}
    for c in cfgs:
        s = c['site']
        if c['mode'] == 'tau':
            c['tau_real'], c['real_ok'] = c['tau_fixed'], True
            c['target_ess'] = float(np.mean(fixed[(s, c['tau_fixed'])]))
        else:
            c['target_ess'] = float(args.target_ess)
            c['tau_real'], c['real_ok'] = tap_sites.tau_for_ess(mean_curve[(s, 'real')], c['target_ess'])
        if 'null' in arms:
            c['tau_null'], c['null_ok'] = tap_sites.tau_for_ess(mean_curve[(s, 'null')], c['target_ess'])
        else:
            c['tau_null'], c['null_ok'] = None, None
    search = {'tau_grid': tap_sites.TAU_GRID.tolist(), 'lam': args.lam,
              'mean_ess': {f'{s}|{a}': mean_curve[(s, a)].tolist() for (s, a) in mean_curve},
              'configs': {c['name']: {k: c[k] for k in ('site', 'mode', 'target_ess', 'tau_real',
                                                        'real_ok', 'tau_null', 'null_ok')}
                          for c in cfgs}}
    with open(os.path.join(args.out_dir, 'tau_search.json'), 'w') as fh:
        json.dump(search, fh, indent=2)
    print('\n[site-probe] operating points (ESS matched per arm):')
    for c in cfgs:
        flag = '' if c['real_ok'] and c['null_ok'] in (True, None) else '   ** TARGET OUTSIDE ESS RANGE **'
        null = f'{c["tau_null"]:.3f}' if c['tau_null'] is not None else '-'
        print(f'  {c["name"]:18s} target ESS {c["target_ess"]:.3f} | tau real {c["tau_real"]:.3f} '
              f'null {null}{flag}', flush=True)

    # ---- pass 2: draws, tilts, MAE -----------------------------------------
    out_jsonl = os.path.join(args.out_dir, 'site_probe.jsonl')
    t1 = time.time()
    with open(out_jsonl, 'w') as fh:
        for i in range(n_scenes):
            entry = cache[i]
            if entry is None:
                continue
            sc = prepare(eval_set, i, device, args.canonical_resolution)
            interior_flat = tap_sites.shared_interior(sc['M'][0, 0], args.erode_cells).reshape(-1)
            ref = CurvRef(sc['N_np'], sc['M_np'], args.erode)
            kind, scene_dir = eval_set.scenes[i % eval_set.n_scenes]
            rec = {'index': i, 'kind': kind, 'scene': os.path.basename(scene_dir.rstrip('/')),
                   'n_valid': int(sc['valid_ids'].numel()), 'lam': args.lam,
                   'erode_cells': args.erode_cells, 'top_k': args.top_k,
                   'mae_uniform_deg': entry['mae_uniform_deg'], 'configs': {}}
            for c in cfgs:
                s = c['site']
                out = {'site': s, 'mode': c['mode'], 'target_ess': c['target_ess'],
                       'tau_real': c['tau_real'], 'tau_null': c['tau_null'],
                       'grid': list(entry['maps'][s]['real'].shape),
                       **entry['rho'][s]['real']}
                if 'null' in arms:
                    out['null_rho_E_curvature'] = entry['rho'][s]['null']['rho_E_curvature']
                for ai, a in enumerate(arms):
                    tau = c['tau_real'] if a == 'real' else c['tau_null']
                    E = entry['maps'][s][a].to(device)
                    p, e, is_int = wess.wess_train_probabilities(
                        E, sc['valid_ids'], interior_flat, sc['H'], sc['H'], tau=tau, lam=args.lam)
                    g = torch.Generator(device=device)
                    g.manual_seed(tap_sites.draw_seed(args.seed, i, c['name'], ai))
                    sel = torch.multinomial(p, n_samp, replacement=p.numel() < n_samp, generator=g)
                    ids = sc['valid_ids'][sel]
                    st = wess.train_draw_stats(e, p, sel, is_int)
                    st.update(ref.tilts(ids.cpu().numpy()))
                    st['tau'] = tau
                    if not args.no_mae:
                        st['mae_deg'] = decode_mae(net, sc, ids, device)
                        st['mae_delta_vs_uniform_deg'] = st['mae_deg'] - entry['mae_uniform_deg']
                    out[a] = st
                rec['configs'][c['name']] = out
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            cache[i] = None
            print(f'[pass 2 {i + 1}/{n_scenes}] {rec["scene"][:32]:32s} | '
                  f'{time.time() - t1:.0f}s', flush=True)

    print(f'\n[site-probe] records: {out_jsonl}')
    print(f'[site-probe] summarise: python {os.path.join(HERE, "wess_site_summary.py")} --run {out_jsonl}')


if __name__ == '__main__':
    main()
