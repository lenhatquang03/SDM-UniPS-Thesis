"""
DiLiGenT benchmark evaluation for SDM-UniPS, mimicking Table 2 of the paper.

Standalone: it needs nothing but an exported checkpoint directory, so it can be
run at any time after training, as many times as you like. It drives the exact
upstream inference path (`dataio` -> `realdata` -> `builder`) that produced the
published numbers, rather than a bespoke loader, so our figures are directly
comparable to the paper's row.

Protocol notes (deliberate choices, each with a reason):

* **No light-intensity division.** DiLiGenT ships `light_intensities.txt`, but
  `realdata.py` never reads it and SDM-UniPS is an *uncalibrated* method. Using
  it would inject calibration the paper's protocol does not use and make the
  MAE incomparable to Table 2.
* **MAE against the normal-derived mask** (`|1 - ||N|||< 0.5`) at the original
  612x512 resolution -- upstream's own choice in `builder.run`, and what the
  published numbers were computed with. Note this is *not* `mask.png`, so these
  figures are not directly comparable to papers that use the shipped mask.
* **Two RNG streams are seeded per trial, not one.** `realdata.load` draws the
  K-image subset with `np.random.permutation`, and `Net.forward` shuffles the
  valid pixels before chunking them into `--pixel_samples` groups
  (`model.py:344`). The decoder runs a spatial-axis transformer *across the
  pixels within a chunk*, so the grouping perturbs the output -- this is the
  source of upstream's "results change with every prediction" warning. Seeding
  numpy alone would leave the run irreproducible.

Scenes are staged as a tree of symlinks rather than read in place. This costs
nothing, never writes to the dataset, and buys two things: DiLiGenT ships the
ground truth as `Normal_gt.png` for most objects but `normal_gt.png` for at
least one (`pot2PNG`), and `realdata.py:69` is case-sensitive -- an unresolved
mismatch silently yields a zero GT and the object is dropped from the average
with no error. Staging also renames `001.png` -> `L001.png` so the upstream
`--test_prefix L*` / `--test_ext .data` defaults apply unchanged.
"""

from __future__ import print_function, division

import os

# Must precede the first `import cv2` for EXR support (as hdlong.py/polarps.py
# do); nothing on the inference path sets it.
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import argparse
import glob
import hashlib
import json
import random
import re
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.builder import builder as builder_mod
from modules.io import dataio


# ---------------------------------------------------------------------------
# Paper reference (Table 2, "Ours" rows). Column order is the paper's.
# ---------------------------------------------------------------------------
OBJECT_ORDER = ['ball', 'bear', 'buddha', 'cat', 'cow',
                'goblet', 'harvest', 'pot1', 'pot2', 'reading']
DISPLAY_NAME = {'ball': 'Ball', 'bear': 'Bear', 'buddha': 'Buddha',
                'cat': 'Cat', 'cow': 'Cow', 'goblet': 'Goblet',
                'harvest': 'Harvest', 'pot1': 'Pot1', 'pot2': 'Pot2',
                'reading': 'Reading'}

PAPER_TABLE2 = {
    96: [1.5, 3.6, 7.5, 5.4, 4.5, 8.5, 10.2, 4.7, 4.1, 8.2],
    64: [1.5, 3.6, 7.6, 5.5, 4.6, 8.6, 10.2, 4.7, 4.1, 8.3],
    32: [1.5, 3.6, 7.7, 5.5, 4.7, 8.6, 10.4, 4.8, 4.2, 8.4],
    16: [1.5, 3.8, 7.7, 6.0, 4.8, 8.5, 10.8, 4.9, 4.4, 8.7],
    8:  [1.6, 4.0, 8.2, 6.3, 5.2, 8.4, 11.5, 5.2, 4.8, 9.4],
    4:  [1.7, 4.1, 10.0, 8.6, 6.3, 9.0, 14.1, 6.1, 5.9, 11.4],
    2:  [1.9, 6.8, 14.4, 13.6, 8.3, 12.8, 21.2, 9.0, 9.2, 16.9],
}


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------
def _resolve_ci(scene_dir, wanted):
    """Return the path of `wanted` inside `scene_dir`, ignoring filename case."""
    target = wanted.lower()
    for fn in os.listdir(scene_dir):
        if fn.lower() == target:
            return os.path.join(scene_dir, fn)
    return None


def _link(src, dst):
    # Absolute target: a relative one resolves against the *link's* directory,
    # not the cwd, so `stage/ballPNG.data/L001.png -> pmsData/ballPNG/001.png`
    # would dangle and every image would silently vanish from the glob.
    src = os.path.abspath(src)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def discover_scenes(root):
    """Find DiLiGenT scene directories under `root` (typically `.../pmsData`)."""
    scenes = sorted(glob.glob(os.path.join(root, '*PNG')))
    if not scenes:
        scenes = sorted(d for d in glob.glob(os.path.join(root, '*'))
                        if os.path.isdir(d)
                        and os.path.isfile(os.path.join(d, 'filenames.txt')))
    if not scenes:
        raise RuntimeError(f'No DiLiGenT scenes found under {root}')
    return scenes


def stage_scenes(scenes, stage_dir):
    """Build `<stage_dir>/<obj>.data/` symlink trees in the upstream layout.

    Returns [(objkey, staged_dir, n_images)], sorted by objkey. `objkey` is the
    scene directory name lowercased with a trailing 'png' stripped, so
    `ballPNG` -> `ball`, which is what `OBJECT_ORDER` keys on.
    """
    os.makedirs(stage_dir, exist_ok=True)
    staged = []
    for scene in scenes:
        name = os.path.basename(scene.rstrip('/'))
        key = re.sub(r'png$', '', name.lower())
        dst = os.path.join(stage_dir, f'{name}.data')
        os.makedirs(dst, exist_ok=True)

        imgs = sorted(fn for fn in os.listdir(scene)
                      if re.fullmatch(r'\d+\.png', fn, flags=re.IGNORECASE))
        if not imgs:
            raise RuntimeError(f'{scene}: no NNN.png observation images found')
        for fn in imgs:
            _link(os.path.join(scene, fn), os.path.join(dst, f'L{fn}'))

        gt = _resolve_ci(scene, 'Normal_gt.png')
        if gt is None:
            raise RuntimeError(f'{scene}: no Normal_gt.png (any case) -- '
                               f'MAE cannot be computed')
        _link(gt, os.path.join(dst, 'Normal_gt.png'))

        mask = _resolve_ci(scene, 'mask.png')
        if mask is not None:
            _link(mask, os.path.join(dst, 'mask.png'))

        staged.append((key, dst, len(imgs)))

    staged.sort(key=lambda t: (OBJECT_ORDER.index(t[0])
                               if t[0] in OBJECT_ORDER else 99, t[0]))
    return staged


# ---------------------------------------------------------------------------
# EXR output
# ---------------------------------------------------------------------------
_EXR_F32 = ([int(cv2.IMWRITE_EXR_TYPE), int(cv2.IMWRITE_EXR_TYPE_FLOAT)]
            if hasattr(cv2, 'IMWRITE_EXR_TYPE_FLOAT') else [])


def write_exr(path, arr):
    """Write float32 EXR. 3-channel input is RGB and gets flipped for cv2's BGR."""
    arr = np.ascontiguousarray(np.float32(arr))
    if arr.ndim == 3:
        arr = arr[:, :, ::-1]
    if not cv2.imwrite(path, arr, _EXR_F32):
        raise IOError(f'Failed to write {path} -- is OpenEXR support enabled '
                      f'in this OpenCV build?')


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(base_seed, objkey, K, trial):
    """Deterministic per-(object, K, trial) seed, stable across runs and orders."""
    digest = hashlib.sha1(f'{base_seed}|{objkey}|{K}|{trial}'.encode()).hexdigest()
    seed = int(digest[:8], 16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _cell(v, width):
    return ('-' if v is None else f'{v:.2f}').rjust(width)


def format_table(per_obj_K, objkeys, K_list, show_paper):
    names = [DISPLAY_NAME.get(k, k) for k in objkeys]
    widths = [max(len(n), 5) for n in names]
    lab_w = max(14, max(len(f'Ours (K={K})') for K in K_list) + 2)

    def row(label, values, ave):
        cells = ' | '.join(_cell(v, w) for v, w in zip(values, widths))
        return f'| {label:<{lab_w}} | {cells} | {_cell(ave, 5)} |'

    lines = [
        f'| {"":<{lab_w}} | '
        + ' | '.join(n.rjust(w) for n, w in zip(names, widths)) + ' | Ave.  |',
        f'|{"-" * (lab_w + 2)}|'
        + '|'.join('-' * (w + 2) for w in widths) + '|-------|',
    ]
    for K in K_list:
        vals = [per_obj_K[K].get(k) for k in objkeys]
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        lines.append(row(f'Ours (K={K})', vals,
                         float(np.mean(finite)) if finite else None))
        if show_paper and K in PAPER_TABLE2:
            ref = PAPER_TABLE2[K]
            ref_vals = [ref[OBJECT_ORDER.index(k)] if k in OBJECT_ORDER else None
                        for k in objkeys]
            fin = [v for v in ref_vals if v is not None]
            lines.append(row(f'  paper (K={K})', ref_vals,
                             float(np.mean(fin)) if fin else None))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description='DiLiGenT evaluation for SDM-UniPS (paper Table 2 protocol).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--diligent_dir', required=True,
                   help="DiLiGenT 'pmsData' root holding the *PNG scene dirs")
    p.add_argument('--checkpoint', required=True,
                   help="Checkpoint dir containing normal/*.pytmodel "
                        "(i.e. <session>/checkpoints)")
    p.add_argument('--out_dir', default='diligent_eval',
                   help='Where the report, JSONL log, EXRs and staging tree go')

    p.add_argument('--K_list', default='10',
                   help='Comma-separated image counts to evaluate')
    p.add_argument('--trials', type=int, default=10,
                   help='Random K-image subsets per (object, K), averaged '
                        '(the paper uses 10)')
    p.add_argument('--objects', default='',
                   help='Comma-separated object keys to restrict to '
                        '(e.g. ball,bear); empty means all')
    p.add_argument('--seed', type=int, default=2024,
                   help='Base seed; per-trial seeds are derived from it')

    p.add_argument('--visualize', action='store_true',
                   help='Write normal/GT/error EXRs for VERIV')
    p.add_argument('--vis_k_list', default='',
                   help='K values to visualize; empty means all of --K_list')
    p.add_argument('--vis_trial', type=int, default=0,
                   help='Which trial index the visualized EXRs come from')

    # Inference knobs -- defaults mirror main.py; canonical_resolution must
    # match training.
    p.add_argument('--canonical_resolution', type=int, default=256)
    p.add_argument('--pixel_samples', type=int, default=10000)
    p.add_argument('--max_image_res', type=int, default=4096)
    p.add_argument('--mask_margin', type=int, default=8)
    p.add_argument('--scalable', action='store_true')
    return p


def main():
    args = build_argparser().parse_args()

    K_list = sorted({int(k) for k in args.K_list.split(',') if k.strip()})
    if not K_list:
        raise SystemExit('--K_list is empty')
    vis_K = (sorted({int(k) for k in args.vis_k_list.split(',') if k.strip()})
             if args.vis_k_list.strip() else list(K_list))
    unknown = set(vis_K) - set(K_list)
    if unknown:
        raise SystemExit(f'--vis_k_list values not in --K_list: {sorted(unknown)}')

    out_dir = os.path.abspath(args.out_dir)
    stage_dir = os.path.join(out_dir, 'staging')
    vis_root = os.path.join(out_dir, 'visualize')
    scratch = os.path.join(out_dir, '_scratch')   # absorbs builder's per-run PNGs
    os.makedirs(out_dir, exist_ok=True)

    # Deterministic kernels, matching the training pipeline's policy.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    scenes = discover_scenes(args.diligent_dir)
    staged = stage_scenes(scenes, stage_dir)
    if args.objects.strip():
        available = [k for k, _, _ in staged]
        keep = {o.strip().lower() for o in args.objects.split(',') if o.strip()}
        staged = [s for s in staged if s[0] in keep]
        if not staged:
            raise SystemExit(f'--objects matched nothing; available: {available}')
    objkeys = [k for k, _, _ in staged]
    n_avail = min(n for _, _, n in staged)
    print(f'[DiLiGenT] {len(staged)} scenes staged at {stage_dir}')
    print(f'[DiLiGenT] objects: {", ".join(objkeys)}')
    print(f'[DiLiGenT] K={K_list}  trials={args.trials}  '
          f'(min images available per scene: {n_avail})')
    for K in K_list:
        if K > n_avail:
            print(f'[DiLiGenT] WARNING: K={K} exceeds the {n_avail} images some '
                  f'scene provides; it will be clipped to that scene\'s count.')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ns = SimpleNamespace(
        session_name=scratch,
        checkpoint=args.checkpoint,
        max_image_res=args.max_image_res,
        max_image_num=K_list[0],
        test_ext='.data',
        test_dir=stage_dir,
        test_prefix='L*',
        mask_margin=args.mask_margin,
        canonical_resolution=args.canonical_resolution,
        pixel_samples=args.pixel_samples,
        scalable=args.scalable,
    )

    net = builder_mod.builder(ns, device)
    test_data = dataio.dataio('Test', ns)

    # Append-only, so repeated runs accumulate; `run_id` makes them separable.
    jsonl_path = os.path.join(out_dir, 'diligent_eval.jsonl')
    jsonl = open(jsonl_path, 'a')
    run_id = time.strftime('%Y%m%dT%H%M%S')

    per_obj_K = {K: {} for K in K_list}
    trials_K = {K: {k: [] for k in objkeys} for K in K_list}
    gt_written = set()
    total = len(K_list) * args.trials * len(staged)
    done = 0
    t0 = time.time()

    for K in K_list:
        for trial in range(args.trials):
            for objkey, objdir, _ in staged:
                seed = seed_everything(args.seed, objkey, K, trial)
                test_data.objlist = [objdir]
                test_data.data.numberOfImages = K

                records = net.run(testdata=test_data,
                                  max_image_resolution=args.max_image_res,
                                  canonical_resolution=args.canonical_resolution)
                rec = records[0]
                mae = rec['mae']
                if mae is None or not np.isfinite(mae):
                    print(f'[DiLiGenT] WARNING: {objkey} K={K} trial={trial} '
                          f'produced no finite MAE; dropped from the average.')
                else:
                    trials_K[K][objkey].append(float(mae))

                done += 1
                jsonl.write(json.dumps({
                    'run_id': run_id, 'objname': objkey, 'K': K, 'trial': trial,
                    'seed': seed, 'mae_deg': mae,
                    'elapsed_sec': round(time.time() - t0, 3),
                }) + '\n')
                jsonl.flush()
                rate = (time.time() - t0) / done
                print(f'[{done}/{total}] {objkey:<8} K={K:<3} trial={trial:<2} '
                      f'MAE={mae if mae is None else round(mae, 3)}  '
                      f'(eta {(total - done) * rate / 60:.1f} min)')

                if args.visualize and K in vis_K and trial == args.vis_trial:
                    obj_dir = os.path.join(vis_root, objkey)
                    kt_dir = os.path.join(obj_dir, f'K{K:03d}_t{trial:02d}')
                    os.makedirs(kt_dir, exist_ok=True)
                    write_exr(os.path.join(kt_dir, 'normal.exr'), rec['normal'])
                    if rec['error_map'] is not None:
                        write_exr(os.path.join(kt_dir, 'error.exr'),
                                  rec['error_map'])
                    # GT depends on neither K nor trial -- write it once.
                    if objkey not in gt_written and rec['normal_gt'] is not None:
                        write_exr(os.path.join(obj_dir, 'normal_gt.exr'),
                                  rec['normal_gt'])
                        gt_written.add(objkey)

    jsonl.close()

    for K in K_list:
        for objkey in objkeys:
            vals = trials_K[K][objkey]
            per_obj_K[K][objkey] = float(np.mean(vals)) if vals else None

    table = format_table(per_obj_K, objkeys, K_list, show_paper=True)
    header = (f'# DiLiGenT evaluation (MAE, degrees)\n\n'
              f'- checkpoint: `{args.checkpoint}`\n'
              f'- trials per (object, K): {args.trials}   base seed: {args.seed}\n'
              f'- canonical_resolution: {args.canonical_resolution}   '
              f'pixel_samples: {args.pixel_samples}\n'
              f'- MAE uses the normal-derived mask at native resolution '
              f'(upstream `builder.run`), no light-intensity division.\n\n')
    report = header + table + '\n'
    report_path = os.path.join(out_dir, 'diligent_table.md')
    with open(report_path, 'w') as f:
        f.write(report)

    print('\n' + table)
    print(f'\n[DiLiGenT] report  -> {report_path}')
    print(f'[DiLiGenT] per-trial -> {jsonl_path}')
    if args.visualize:
        print(f'[DiLiGenT] EXRs    -> {vis_root}')
    print(f'[DiLiGenT] total time: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
