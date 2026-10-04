"""DiLiGenT MAE per GT-curvature decile: does biased WESS win where it aims?

Pre-registered (2026-10-04, before any number was seen):

  Claim     B2 tau=1 (unweighted WESS) predicts normals more accurately than B1
            on the most curved pixels of each DiLiGenT object.
  Metric    dTop = MAE_top(B2 tau=1) - MAE_top(B1): MAE over each object's
            top curvature decile, K=16, mean over 10 objects x 10 trials,
            paired by trial (same seeds as eval_diligent.py).
  Null      dTop_ctrl = MAE_top(B2 lam=1) - MAE_top(B1): two trainings of the
            same model, i.e. the noise reference for this bucket.
  Decision  dTop < 0 AND |dTop| > |dTop_ctrl| AND |dTop| > 2 * SE(dTop)
              -> focus claim supported: final run 3 (C + WESS) uses the
                 UNWEIGHTED loss whatever run 1 shows, and curvature-decile MAE
                 becomes a secondary metric in the thesis.
            otherwise -> the running plan is unchanged.
  The other nine deciles are reported, never used to decide.

Curvature is kappa = |grad N_gt| summed over x, y, z (the same definition as
wess_probe.grad_magnitude), on the GT mask eroded 9x9 so that the
silhouette, where np.gradient produces fake curvature, is excluded. Deciles
are per object, so each object contributes its own 10% most curved pixels.

Integrity gate: the whole-mask MAE of every (object, K, trial) must equal the
value in that run's diligent_eval.jsonl, because the seeds are identical. A
mismatch means a different model or code path, and the run must not be used.

Run (once per model, on the training box, branch modelC-wess):
    python -u sdm_unips/probes/curvature_buckets.py run \
        --diligent_dir <pmsData> --checkpoint <run>/checkpoints \
        --out <run>/diligent_eval/curvature_buckets.json
Compare:
    python sdm_unips/probes/curvature_buckets.py compare \
        --b1 <B1 json> --b2 <B2 tau=1 json> --ctrl <B2 lam=1 json>
"""

import argparse
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..'))

K = 16
N_DECILES = 10
ERODE = 9


def grad_magnitude(field):
    """Same as wess_probe.grad_magnitude: |grad| summed over channels."""
    import numpy as np
    total = np.zeros(field.shape[:2], np.float64)
    for c in range(field.shape[2]):
        gy, gx = np.gradient(field[:, :, c].astype(np.float64))
        total += gy * gy + gx * gx
    return np.sqrt(total)


def run(args):
    import cv2
    import numpy as np
    import torch
    from types import SimpleNamespace
    import eval_diligent as ed
    from modules.builder import builder as builder_mod
    from modules.io import dataio

    out_dir = os.path.abspath(os.path.dirname(args.out))
    work = os.path.join(out_dir, '_curv_work')
    os.makedirs(work, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    staged = ed.stage_scenes(ed.discover_scenes(args.diligent_dir), os.path.join(work, 'staging'))
    weights_root, info = ed.stage_best_weights(args.checkpoint, os.path.join(work, '_weights'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ns = SimpleNamespace(session_name=os.path.join(work, '_scratch'), checkpoint=weights_root,
                         max_image_res=4096, max_image_num=K, test_ext='.data',
                         test_dir=os.path.join(work, 'staging'), test_prefix='L*', mask_margin=8,
                         canonical_resolution=256, pixel_samples=args.pixel_samples, scalable=False)
    net = builder_mod.builder(ns, device)
    data = dataio.dataio('Test', ns)

    records = []
    for trial in range(args.trials):
        for objkey, objdir, _ in staged:
            seed = ed.seed_everything(args.seed, objkey, K, trial)
            data.objlist = [objdir]
            data.data.numberOfImages = K
            rec = net.run(testdata=data, max_image_resolution=4096, canonical_resolution=256)[0]
            n_gt, err = rec['normal_gt'], rec['error_map']
            mask = (np.abs(1 - np.linalg.norm(n_gt, axis=2)) < 0.5).astype(np.uint8)
            inner = cv2.erode(mask, np.ones((ERODE, ERODE), np.uint8)) > 0
            kappa = grad_magnitude(n_gt)[inner]
            e = err[inner]
            edges = np.quantile(kappa, np.linspace(0, 1, N_DECILES + 1))
            bucket = np.clip(np.searchsorted(edges, kappa, side='right') - 1, 0, N_DECILES - 1)
            dec = [float(e[bucket == d].mean()) if (bucket == d).any() else float('nan')
                   for d in range(N_DECILES)]
            records.append({'objname': objkey, 'K': K, 'trial': trial, 'seed': seed,
                            'mae_deg': rec['mae'], 'decile_mae': dec,
                            'kappa_edges': [float(x) for x in edges]})
            print(f'[curv] {objkey:<8} trial {trial}  MAE {rec["mae"]:.3f}  '
                  f'flat {dec[0]:.2f}  top {dec[-1]:.2f}', flush=True)

    with open(args.out, 'w') as fh:
        json.dump({'checkpoint': args.checkpoint, 'weights': info, 'K': K,
                   'trials': args.trials, 'seed': args.seed, 'records': records}, fh)
    print('wrote', args.out)

    # Integrity gate against the existing DiLiGenT run of the same model.
    ref_path = os.path.join(out_dir, 'diligent_eval.jsonl')
    if os.path.isfile(ref_path):
        ref = {}
        for line in open(ref_path):
            r = json.loads(line)
            if r['K'] == K:
                ref[(r['objname'], r['trial'])] = r['mae_deg']
        diffs = [abs(r['mae_deg'] - ref[(r['objname'], r['trial'])]) for r in records
                 if (r['objname'], r['trial']) in ref]
        print(f'[gate] {len(diffs)} matched trials, max |dMAE| vs diligent_eval.jsonl = '
              f'{max(diffs) if diffs else float("nan"):.2e}  '
              f'{"PASS" if diffs and max(diffs) < 1e-6 else "FAIL -- do not use"}')


def per_trial(path):
    d = json.load(open(path))
    by = {}
    for r in d['records']:
        by.setdefault(r['trial'], []).append(r['decile_mae'])
    # mean over objects, per trial and decile
    return {t: [st.mean(v[i] for v in rows) for i in range(N_DECILES)] for t, rows in by.items()}


def paired(a, b, i):
    diffs = [a[t][i] - b[t][i] for t in sorted(a) if t in b]
    return st.mean(diffs), st.stdev(diffs) / math.sqrt(len(diffs))


def compare(args):
    b1, b2, ctrl = per_trial(args.b1), per_trial(args.b2), per_trial(args.ctrl)
    print(f'{"decile":>6} {"kappa":>9} {"B1":>7} {"B2 t=1":>7} {"ctrl":>7}   '
          f'{"B2-B1":>14} {"ctrl-B1":>14}')
    for i in range(N_DECILES):
        m = lambda x: st.mean(x[t][i] for t in x)
        d, se = paired(b2, b1, i)
        dc, sec = paired(ctrl, b1, i)
        lab = 'flat' if i == 0 else ('TOP' if i == N_DECILES - 1 else '')
        print(f'{i:>6} {lab:>9} {m(b1):7.2f} {m(b2):7.2f} {m(ctrl):7.2f}   '
              f'{d:+7.3f}+-{se:.3f} {dc:+7.3f}+-{sec:.3f}')
    d, se = paired(b2, b1, N_DECILES - 1)
    dc, _ = paired(ctrl, b1, N_DECILES - 1)
    ok = d < 0 and abs(d) > abs(dc) and abs(d) > 2 * se
    print(f'\nDECISION (pre-registered): dTop = {d:+.3f} +- {se:.3f}, |null| = {abs(dc):.3f}')
    print('  -> focus claim SUPPORTED: run 3 uses the unweighted loss; add decile MAE '
          'as a secondary metric.' if ok else
          '  -> not supported: running plan unchanged.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--diligent_dir', required=True)
    r.add_argument('--checkpoint', required=True, help='<run>/checkpoints (holds best.pt)')
    r.add_argument('--out', required=True)
    r.add_argument('--trials', type=int, default=10)
    r.add_argument('--seed', type=int, default=2024)
    r.add_argument('--pixel_samples', type=int, default=2048)
    c = sub.add_parser('compare')
    c.add_argument('--b1', required=True)
    c.add_argument('--b2', required=True)
    c.add_argument('--ctrl', required=True)
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else compare(a)
