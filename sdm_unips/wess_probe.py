"""Phase 0 of Model B2 (WESS): does the sub-band energy point at geometry?

WESS reweights the training gradient toward pixels with high wavelet sub-band
energy, on the premise that those are the creases and shadow boundaries where
a surface-normal network actually has headroom. The premise has one obvious way
to fail, and it fails silently: photometric stereo reads normals from *shading*
gradients, but the sub-band energy of a learned feature map will also fire on
*albedo* edges. A painted stripe on a flat surface carries enormous
high-frequency energy and zero geometric content. If `E` mostly tracks texture,
WESS spends its whole budget on pixels that are already easy and the
formulation has to change -- energy computed on cross-image *consistency*
rather than per-image intensity.

That question costs an afternoon to answer and a multi-day training run to
discover the hard way, so it is answered first. This probe trains nothing and
writes nothing into the model: it loads Model B1's checkpoint, runs the encoder
over held-out scenes, and reports, per scene:

* **Rank correlations** of `E` against a geometric reference (the gradient
  magnitude of the ground-truth normal field, i.e. surface curvature) and
  against a photometric one (the gradient magnitude of the K-mean observation,
  which mixes shading and albedo). Plus the **partial** correlation of `E` with
  the image gradient *holding curvature fixed* -- the albedo-sensitivity number.
  All computed on the mask **eroded** by `--erode` pixels: the object silhouette
  is simultaneously the strongest edge in the frame and a place where both
  gradients are meaningless, and leaving it in would manufacture a positive
  correlation out of nothing.

* **`curv_tilt`** -- mean GT curvature over the WESS-drawn pixels divided by
  mean curvature over the mask. This is the headline: it says, in one number,
  whether the sampler aims at geometry. A uniform draw reads 1.0.

* The three sampler diagnostics (`wess_tilt`, `wess_ess`, `wess_top10_frac`)
  that will be logged per step during the B2 run, so `--tau` can be chosen here
  rather than guessed there.

* **MAE of the B1 checkpoint on WESS-drawn pixels against uniform-drawn ones.**
  If the high-energy pixels are not measurably harder for the current model,
  there is no headroom for a saliency sampler to exploit and that is worth
  knowing before spending the GPU time.

* **raw vs filtered bands** (`--band both`): the rank correlation between the
  energy of the pure Haar coefficients and of the same bands after the level-0
  depthwise conv and its trainable 0.1-initialised gain. If they rank pixels
  the same, the choice does not matter; if they diverge, `raw` is preferred
  because it cannot drift as training proceeds.

Scope: the mixed training pool (hdlong + PolarPS) at `--train_resolution`,
which is where WESS actually operates and the only place the tile decomposition
has the geometry the saliency map assumes. DiLiGenT is deliberately not probed
here -- `realdata.py` resizes each object to an arbitrary square, which the
encoder's `divide_tensor_spatial` cannot always decompose, and the Phase-0
question is about the *training* distribution.

Nothing in the repository is modified by running this. The dataset flags are
`train.py`'s own, so pointing `--scene_manifest` and `--seed` at the training
run's values probes provably the same held-out scenes the run validated on.

Usage:

    python sdm_unips/wess_probe.py \
      --checkpoint ~/runs/modelB1_wtconv/checkpoints/best.pt \
      --hdlong_dir /path/to/hdlong-complexv1 \
      --polarps_dir /path/to/PolarPS \
      --scene_manifest ~/runs/modelB1_wtconv/scene_manifest.json \
      --max_scenes 8000 --seed 42 \
      --num_scenes 10 --tau 0.5 1.0 2.0 --band both \
      --out_dir ~/result_overview/wess_phase0
"""

from __future__ import print_function, division

import os

# Must precede the first `import cv2`, as the dataset loaders do -- the scene
# loaders read EXR and OpenCV's EXR codec is off unless this is set.
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import contextlib
import glob
import json
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.io.dataloader.mixed import build_mixed_split, MixedEvalDataset
from modules.loss import losses
from modules.model import model as model_mod
from modules.model import wess

import train as train_mod


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def resolve_checkpoint(spec):
    """A `.pt` / `.pytmodel` file, or a directory holding one."""
    if os.path.isfile(spec):
        return spec
    if os.path.isdir(spec):
        for cand in ('best.pt', 'final.pt'):
            p = os.path.join(spec, cand)
            if os.path.isfile(p):
                return p
        exported = sorted(glob.glob(os.path.join(spec, 'normal', '*.pytmodel')))
        if len(exported) == 1:
            return exported[0]
        epochs = sorted(glob.glob(os.path.join(spec, 'epoch_*.pt')))
        if epochs:
            return epochs[-1]
    raise FileNotFoundError(
        f'--checkpoint {spec!r}: expected a .pt/.pytmodel file, or a directory '
        f'holding best.pt / final.pt / normal/*.pytmodel / epoch_*.pt.')


def load_weights(net, path, device):
    """Load a bare `state_dict` or a training checkpoint into an unwrapped `Net`.

    Mirrors `builder.load_models`' guards, which exist because both silent
    failure modes here produce a *randomly initialised* network that still runs
    and reports ~76 deg MAE: a `module.` prefix mismatch (the trainer saves from
    a bare `Net`, upstream's weights came from a DataParallel one), and a
    partial match. Anything short of a full match raises.
    """
    params = torch.load(path, map_location=device)
    if isinstance(params, dict) and 'model' in params and 'optimizer' in params:
        params = params['model']
    if any(k.startswith('module.') for k in params):
        params = {k[len('module.'):]: v for k, v in params.items()}

    missing, unexpected = net.load_state_dict(params, strict=False)
    n_total = len(net.state_dict())
    if missing or unexpected:
        raise RuntimeError(
            f'{path}: partial load -- {n_total - len(missing)}/{n_total} '
            f'parameters matched, {len(missing)} missing, {len(unexpected)} '
            f'unexpected. Refusing to probe a half-initialised network. A '
            f'checkpoint only loads under the branch that produced it: this '
            f'branch builds WTConv blocks, so Model A weights will not fit. '
            f'Missing e.g. {missing[:3]}; unexpected e.g. {unexpected[:3]}.')
    print(f'[probe] loaded {n_total}/{n_total} parameters from {path}')
    return net


# ---------------------------------------------------------------------------
# Reference fields and statistics
# ---------------------------------------------------------------------------
def grad_magnitude(field):
    """Gradient magnitude of an (H, W) or (H, W, C) float field, summed over C."""
    if field.ndim == 2:
        field = field[:, :, None]
    total = np.zeros(field.shape[:2], np.float64)
    for c in range(field.shape[2]):
        gy, gx = np.gradient(field[:, :, c].astype(np.float64))
        total += gy * gy + gx * gx
    return np.sqrt(total)


def curvature_map(normal_chw):
    """Surface-curvature proxy: |grad N| of the GT unit-normal field.

    Model-independent by construction, which is what makes it a fair reference
    for every variant -- and the same bucketing Phase 2 will use to report
    per-curvature-decile DiLiGenT MAE for A, B1 and B2.
    """
    return grad_magnitude(np.transpose(normal_chw, (1, 2, 0)))


def observation_gradient(I_chwk, n_imgs):
    """Photometric reference: |grad| of the mean-over-K observation intensity.

    Mixes shading with albedo, which is exactly the point -- correlating `E`
    against this, with curvature partialled out, is the albedo-sensitivity test.
    """
    obs = I_chwk[..., :n_imgs].mean(axis=(0, 3))
    return grad_magnitude(obs)


def rankdata(x):
    """Ordinal ranks (ties broken by order). Avoids a scipy dependency."""
    order = np.argsort(x, kind='stable')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return float('nan')
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float('nan')


def partial_spearman(r_eg, r_ec, r_gc):
    """Rank correlation of E with G, holding C fixed, from the 3x3 rank matrix."""
    denom = np.sqrt(max(1 - r_ec ** 2, 0.0) * max(1 - r_gc ** 2, 0.0))
    if denom <= 1e-9:
        return float('nan')
    return float((r_eg - r_ec * r_gc) / denom)


def shuffle_map(E, generator):
    """Falsification control: permute the energy map's cells.

    A positive result is only evidence if the *same pipeline* reads null on
    input that cannot carry the signal. Permuting `E` destroys its spatial
    correspondence with the surface while preserving its value multiset
    **exactly**, so the softmax is as peaked as before and the draw is as
    concentrated as before -- what changes is only *where* the concentrated
    mass sits.

    Under the shuffle, `rho(E, curvature)`, `rho(E, |grad I| | curvature)`,
    both curvature tilts and the MAE gain must collapse to their nulls (0, 0,
    1.0, 1.0, 0 deg), and the rim share of the draw must fall back to the
    uniform draw's ~0.085.

    RESULT (2026-09-07, n=431, B1 best.pt). Every geometric null passes, and
    tightly: rho(E, curvature) +0.43 -> -0.008, rho partial +0.29 -> -0.001,
    curv_tilt 1.0019, curv_tilt_interior 1.0007, top-decile 0.1004, interior
    share within 0.0011 of the uniform arm. The MAE gain does NOT fully
    collapse: +0.162 +- 0.037 deg survives at tau=1 (4.4 SE from zero), +0.059
    +- 0.025 at tau=2. It is not leaked signal -- rho(real gain, shuffled gain)
    = -0.047 -- but an offset that tracks peakedness (ESS 0.100 -> +0.162 deg;
    ESS 0.614 -> +0.059 deg), most likely because `Regressor`'s spatial-axis
    transformer attends across pixels within the sample set, so a clustered
    draw degrades the decoder's context wherever the clusters land. Subtract it
    before quoting the gain.

    `wess_ess` is **not** a null target here, and it will not stay put --
    measured, not assumed. `wess_probabilities` upsamples the 64x64 map to the
    decoder grid *before* standardizing, and bilinear interpolation of a
    spatially decorrelated field averages unrelated neighbours, shrinking
    `std(e)`; standardizing by the smaller sigma then amplifies whatever
    outliers survive, so the softmax comes out MORE peaked, not less. Measured
    on the real split: ESS 0.231 -> 0.100 at tau=1, 0.715 -> 0.614 at tau=2.
    (An earlier version of this docstring predicted 0.107 -> 0.008 from a
    stdlib simulation of the chain -- right direction, wrong by ~12x, because
    the simulation's sigma-shrinkage is far stronger on a synthetic field than
    on a real energy map. Corrected against the run.) That is an artefact of
    permuting a field that is normally smooth; read ESS from the real run only.

    INTEGRITY GATE for any re-run: `uniform_sample` is called before this
    function consumes the generator, so the control's uniform arm is
    bit-identical to the real run's, scene for scene. Verify that first (it
    held exactly on all 431); if it does not, the two runs are not the same
    checkpoint or scene list and nothing downstream is comparable.

    The permutation runs on the 64x64 map rather than the 512x512 upsample so
    the bilinear interpolation that follows still produces a field of the same
    resolution as the real one. Shuffling after the upsample would hand the
    control a white-noise field the real map never is -- an easier null to pass
    than the honest one.

    Per-scene nulls are noisy, because the draw concentrates on wherever the
    permutation happened to put the high cells. Measured on the run, the tilt
    null has a per-scene sd of 0.085 at tau=1 (0.054 at tau=2) and the MAE
    delta a sd of 0.77 deg -- so a ten-scene control resolves nothing, and the
    MAE arm is the slowest to converge. Run over the whole split
    (`--num_scenes 0`), where the standard error is 0.004 on the tilts and
    0.037 deg on the MAE delta. (An earlier simulation put the tilt sd at
    ~0.29; the real field is far better behaved.)
    """
    flat = E.reshape(-1)
    perm = torch.randperm(flat.numel(), generator=generator).to(flat.device)
    return flat[perm].reshape(E.shape)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _to_u8(x, mask=None, pct=99.0):
    """Percentile-normalised uint8, so one specular outlier cannot flatten a map."""
    x = np.asarray(x, np.float64)
    sel = x[mask > 0] if mask is not None and mask.sum() > 0 else x.reshape(-1)
    lo = float(sel.min()) if sel.size else 0.0
    hi = float(np.percentile(sel, pct)) if sel.size else 1.0
    if hi <= lo:
        hi = lo + 1e-6
    return np.uint8(np.clip((x - lo) / (hi - lo), 0, 1) * 255)


def _heat(x, mask=None):
    img = cv2.applyColorMap(_to_u8(x, mask), cv2.COLORMAP_INFERNO)
    if mask is not None:
        img = img * (mask > 0)[:, :, None].astype(np.uint8)
    return img


def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _overlay(base_bgr, ids, H, W, colour):
    """Scatter a draw over a base image, one 2x2 dot per sampled pixel."""
    img = base_bgr.copy()
    ys, xs = np.divmod(np.asarray(ids, np.int64), W)
    for dy in (0, 1):
        for dx in (0, 1):
            yy = np.clip(ys + dy, 0, H - 1)
            xx = np.clip(xs + dx, 0, W - 1)
            img[yy, xx] = colour
    return img


def write_panel(path, tiles, ncol=4):
    """Grid of equally sized labelled tiles, padded with black to a full row."""
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    pad = np.zeros((h, w, 3), np.uint8)
    rows = []
    for i in range(0, len(tiles), ncol):
        row = tiles[i:i + ncol]
        row = row + [pad] * (ncol - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(path, np.vstack(rows))


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
def build_parser():
    p = train_mod.build_argparser()
    g = p.add_argument_group('WESS Phase-0 probe')
    g.add_argument('--checkpoint', required=True,
                   help="Model B1 weights: a .pt / .pytmodel file, or a "
                        "checkpoints/ directory. Must come from THIS branch's "
                        "architecture (WTConv blocks); Model A weights do not fit.")
    g.add_argument('--out_dir', default='wess_phase0',
                   help='Where the panels and wess_probe.jsonl are written.')
    g.add_argument('--num_scenes', type=int, default=10,
                   help='Held-out scenes to probe (0 = all).')
    g.add_argument('--split', default='val', choices=['val', 'test'],
                   help='Which held-out split to draw the scenes from.')
    g.add_argument('--tau', type=float, nargs='+', default=[0.5, 1.0, 2.0],
                   help='Softmax temperatures, in units of sigma (the energy is '
                        'standardized over the mask first, which is what makes '
                        'the unit stable across scenes and across training).')
    g.add_argument('--lam', type=float, default=wess.DEFAULT_LAM,
                   help='Uniform mixing weight. 0.25 reproduces the proposal\'s '
                        '512-of-2048 base set; 1.0 recovers Model A exactly.')
    g.add_argument('--band', default='both', choices=['raw', 'filtered', 'both'],
                   help='Raw Haar coefficients, the level-0 filtered bands, or '
                        'both (reports how differently they rank pixels).')
    g.add_argument('--stage', type=int, default=wess.DEFAULT_STAGE,
                   help='Backbone stage to tap (0 = 64x64 features, 8x8-pixel '
                        'saliency cells; 3 = 8x8 features, 64x64-pixel cells).')
    g.add_argument('--block', type=int, default=wess.DEFAULT_BLOCK,
                   help='Block within the stage. 0 is closest to the stem.')
    g.add_argument('--top_k', type=int, default=2,
                   help='Reduction over the K images: mean of the top-k per '
                        'pixel. 1 = max, 2 = the agreed default.')
    g.add_argument('--erode', type=int, default=9,
                   help='Mask erosion (px) for the correlation statistics. The '
                        'silhouette is a spurious edge in both references.')
    g.add_argument('--shuffle_energy', action='store_true',
                   help='FALSIFICATION CONTROL. Randomly permute the energy '
                        'map before it is used, so it keeps its value '
                        'histogram but loses all alignment with the surface. '
                        'Every geometric metric must collapse to its null; '
                        'wess_ess must not move. Records are tagged '
                        '"shuffled": true so a control run cannot be mistaken '
                        'for a real one.')
    g.add_argument('--no_figures', action='store_true',
                   help='Statistics only; skip the PNG panels.')
    return p


def probe_scene(net, batch, args, device, gen):
    """Run one scene through the encoder and score every configured sampler."""
    I, N, M, n_imgs = batch
    I, M = I.to(device), M.to(device)
    N_dev = N.to(device)
    H = I.shape[2]
    B = I.shape[0]
    assert B == 1, 'the probe runs one scene at a time'
    n = int(n_imgs[0])
    mosaic_scale = H // int(args.canonical_resolution)

    dec_res = torch.full((B, 1), H, dtype=torch.long, device=device)
    can_res = torch.full((B, 1), int(args.canonical_resolution),
                         dtype=torch.long, device=device)

    m_ = M[0].reshape(-1, H * H).permute(1, 0)
    valid_ids = torch.nonzero(m_ > 0, as_tuple=False)[:, 0]

    bands = ['raw', 'filtered'] if args.band == 'both' else [args.band]
    taps = {b: wess.SubbandTap(net, args.stage, args.block, band=b) for b in bands}

    # Pass 1: the uniform baseline. Also what populates the taps -- WESS reads
    # the energy of the very forward it is about to sample from, so no second
    # encoder pass is needed at training time either.
    ids_uniform = wess.uniform_sample(valid_ids, int(args.pixel_samples), gen)
    with contextlib.ExitStack() as stack:
        for t in taps.values():
            stack.enter_context(t)
        with torch.no_grad():
            pred_u, idx_u, _ = net(I, M, n_imgs.to(device),
                                   decoder_resolution=dec_res,
                                   canonical_resolution=can_res,
                                   training=True,
                                   sample_ids=ids_uniform[None])
        maps = {b: wess.saliency_maps(t, n_imgs, mosaic_scale, top_k=args.top_k)[0]
                for b, t in taps.items()}

    mae_uniform = float(losses.angular_error_deg(pred_u, N_dev, M, idx_u))

    primary = maps['raw' if 'raw' in maps else args.band]
    if args.shuffle_energy:
        # Consumes `gen`, so a control run's WESS draws sit at a different RNG
        # position than a real run's. That is intended and harmless: the
        # control is compared against its nulls, never sample-for-sample
        # against the real run. With the flag off nothing here executes and
        # the stream is untouched.
        primary = shuffle_map(primary, gen)
    draws = []
    for tau in args.tau:
        ids, stats = wess.wess_sample(primary, valid_ids, H, H,
                                      int(args.pixel_samples),
                                      tau=tau, lam=args.lam, generator=gen)
        with torch.no_grad():
            pred_w, idx_w, _ = net(I, M, n_imgs.to(device),
                                   decoder_resolution=dec_res,
                                   canonical_resolution=can_res,
                                   training=True, sample_ids=ids[None])
        stats['tau'] = float(tau)
        stats['mae_deg'] = float(losses.angular_error_deg(pred_w, N_dev, M, idx_w))
        draws.append((ids, stats))

    return {
        'H': H, 'n_imgs': n, 'mosaic_scale': mosaic_scale,
        'valid_ids': valid_ids, 'maps': maps, 'primary': primary,
        'ids_uniform': ids_uniform, 'mae_uniform': mae_uniform,
        'draws': draws,
    }


def scene_statistics(res, I_np, N_np, M_np, erode):
    """Correlations and tilts against the model-independent reference fields."""
    H = res['H']
    mask = (M_np[0] > 0).astype(np.uint8)
    curv = curvature_map(N_np) * mask
    imgrad = observation_gradient(I_np, res['n_imgs']) * mask

    E_up = F.interpolate(
        res['primary'][None, None].float().cpu(), size=(H, H),
        mode='bilinear', align_corners=False).reshape(H, H).numpy()

    # Correlations run on the INTERIOR only. The silhouette is the strongest
    # edge in every one of these three fields at once, so including it would
    # manufacture agreement between them that says nothing about creases.
    if erode > 0:
        k = np.ones((erode, erode), np.uint8)
        interior = cv2.erode(mask, k, iterations=1)
    else:
        interior = mask
    sel = interior > 0

    stats = {'n_valid': int(mask.sum()), 'n_interior': int(sel.sum())}
    if sel.sum() >= 32:
        e, c, g = E_up[sel], curv[sel], imgrad[sel]
        r_ec = spearman(e, c)
        r_eg = spearman(e, g)
        r_gc = spearman(g, c)
        stats.update({
            'rho_E_curvature': r_ec,
            'rho_E_imgrad': r_eg,
            'rho_imgrad_curvature': r_gc,
            'rho_E_imgrad_given_curvature': partial_spearman(r_eg, r_ec, r_gc),
        })

    # Tilts are measured on the FULL mask, because that is what the sampler
    # actually draws from.
    flat_curv = curv.reshape(-1)
    mean_curv = float(flat_curv[mask.reshape(-1) > 0].mean()) if mask.sum() else 0.0
    top_decile = (float(np.percentile(flat_curv[mask.reshape(-1) > 0], 90))
                  if mask.sum() else 0.0)

    flat_interior = sel.reshape(-1)
    interior_curv = flat_curv[flat_interior]
    mean_curv_in = float(interior_curv.mean()) if interior_curv.size else 0.0

    def tilt(ids):
        ids = np.asarray(ids, np.int64)
        v = flat_curv[ids]
        inside = flat_interior[ids]
        # The interior tilt is the honest one: it excludes the silhouette,
        # which is both the strongest edge in the frame and a place where
        # np.gradient of the normal field is a boundary artefact. A sampler
        # that merely traces the outline scores 1.0 there and high on the
        # full-mask number.
        t_in = (float(v[inside].mean() / (mean_curv_in + 1e-12))
                if inside.any() and mean_curv_in > 0 else float('nan'))
        return (float(v.mean() / (mean_curv + 1e-12)),
                float((v >= top_decile).mean()), t_in,
                float(inside.mean()))

    t_u, d_u, ti_u, f_u = tilt(res['ids_uniform'].cpu().numpy())
    stats['curv_tilt_uniform'] = t_u
    stats['curv_top10_uniform'] = d_u
    stats['curv_tilt_interior_uniform'] = ti_u
    stats['interior_frac_uniform'] = f_u
    stats['mae_uniform_deg'] = res['mae_uniform']

    per_tau = []
    for ids, s in res['draws']:
        t_w, d_w, ti_w, f_w = tilt(ids.cpu().numpy())
        s = dict(s)
        s['curv_tilt'] = t_w
        s['curv_top10_frac'] = d_w
        s['curv_tilt_interior'] = ti_w
        s['interior_frac'] = f_w
        s['mae_delta_vs_uniform_deg'] = s['mae_deg'] - res['mae_uniform']
        per_tau.append(s)
    stats['per_tau'] = per_tau

    if len(res['maps']) == 2:
        # On the object only. Both maps are near-zero over the background,
        # which is most of the frame, so a whole-map correlation would read
        # ~1.0 however differently they rank the pixels that matter.
        hs = res['maps']['raw'].shape[-1]
        m_small = cv2.resize(mask, (hs, hs), interpolation=cv2.INTER_NEAREST) > 0
        a = res['maps']['raw'].float().cpu().numpy()[m_small]
        b = res['maps']['filtered'].float().cpu().numpy()[m_small]
        stats['rho_raw_vs_filtered'] = spearman(a, b)

    return stats, curv, imgrad, E_up, mask


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # `build_mixed_split` writes a scene manifest under <log_dir> when
    # --scene_manifest is not given; keep it beside the probe output instead of
    # dropping a `train_run/` tree into the cwd.
    if args.log_dir is None:
        args.log_dir = args.out_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net = model_mod.Net(int(args.pixel_samples), device).to(device)
    load_weights(net, resolve_checkpoint(args.checkpoint), device)
    net.eval()
    net.no_grad()

    train_set, val_set, test_set = build_mixed_split(args)
    del train_set
    chosen = val_set if args.split == 'val' else test_set
    if len(chosen.scenes) == 0:
        raise RuntimeError(f'the {args.split} split is empty')
    # Same seed convention train.py uses, so these are the very renders the
    # training run validated on.
    seed = args.seed + 5_000_011 if args.split == 'val' else args.seed
    eval_set = MixedEvalDataset(args, chosen.scenes, n_trials=1, seed=seed,
                                subset_name=f'WESSProbe/{args.split}',
                                noun='probe', announce=True)

    n_scenes = len(eval_set) if args.num_scenes <= 0 else min(args.num_scenes,
                                                              len(eval_set))
    print(f'[probe] {n_scenes} scene(s) | tap = stage {args.stage} block '
          f'{args.block} | bands = {args.band} | top_k = {args.top_k} | '
          f'tau = {args.tau} | lam = {args.lam}')
    if args.shuffle_energy:
        print('[probe] *** SHUFFLED-ENERGY FALSIFICATION CONTROL *** these are '
              'NULL measurements, not results')

    out_jsonl = os.path.join(args.out_dir, 'wess_probe.jsonl')
    records = []
    t0 = time.time()
    with open(out_jsonl, 'w') as fh:
        for i in range(n_scenes):
            I_np, N_np, M_np, n_i = eval_set[i]
            batch = (torch.from_numpy(I_np)[None], torch.from_numpy(N_np)[None],
                     torch.from_numpy(M_np)[None],
                     torch.tensor([int(n_i)], dtype=torch.long))

            gen = torch.Generator()
            gen.manual_seed((args.seed + i) % (2 ** 31 - 1))
            res = probe_scene(net, batch, args, device, gen)
            stats, curv, imgrad, E_up, mask = scene_statistics(
                res, I_np, N_np, M_np, args.erode)

            kind, scene_dir = eval_set.scenes[i % eval_set.n_scenes]
            rec = {'index': i, 'kind': kind,
                   'scene': os.path.basename(scene_dir.rstrip('/')),
                   'shuffled': bool(args.shuffle_energy), **stats}
            records.append(rec)
            fh.write(json.dumps(rec) + '\n')
            fh.flush()

            best = rec['per_tau'][0]
            print(f"[{i + 1}/{n_scenes}] {rec['scene'][:28]:28s} "
                  f"rho(E,curv)={rec.get('rho_E_curvature', float('nan')):+.3f} "
                  f"rho(E,img|curv)={rec.get('rho_E_imgrad_given_curvature', float('nan')):+.3f} "
                  f"curv_tilt={best['curv_tilt']:.2f} "
                  f"(uniform {rec['curv_tilt_uniform']:.2f}) "
                  f"mae {rec['mae_uniform_deg']:.2f} -> {best['mae_deg']:.2f} deg")

            if not args.no_figures:
                H = res['H']
                obs = I_np[..., :res['n_imgs']].mean(axis=(0, 3))
                obs_bgr = cv2.cvtColor(_to_u8(obs, mask), cv2.COLOR_GRAY2BGR)
                nrm = np.uint8(np.clip(
                    0.5 * (1 + np.transpose(N_np, (1, 2, 0))[:, :, ::-1]), 0, 1)
                    * 255) * mask[:, :, None]
                tiles = [
                    _label(obs_bgr, 'mean observation'),
                    _label(nrm, 'GT normal'),
                    _label(_heat(curv, mask), 'GT curvature |grad N|'),
                    _label(_heat(imgrad, mask), '|grad I| (shading + albedo)'),
                    _label(_heat(E_up, mask), f'E (stage {args.stage} blk '
                                              f'{args.block}, top{args.top_k})'),
                    _label(_overlay(obs_bgr, res['ids_uniform'].cpu().numpy(),
                                    H, H, (80, 200, 80)), 'uniform draw'),
                ]
                for (ids, _s), tau in zip(res['draws'], args.tau):
                    tiles.append(_label(
                        _overlay(obs_bgr, ids.cpu().numpy(), H, H, (60, 60, 240)),
                        f'WESS draw  tau={tau:g}  lam={args.lam:g}'))
                write_panel(os.path.join(
                    args.out_dir, f'panel_{i:02d}_{rec["scene"][:32]}.png'), tiles)

    # ---- aggregate ------------------------------------------------------
    def agg(key, src=None):
        vals = [(r if src is None else src(r)).get(key) for r in records]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float('nan')

    print(f'\n[probe] {len(records)} scenes in {time.time() - t0:.0f}s\n')
    if args.shuffle_energy:
        print('  *** SHUFFLED-ENERGY CONTROL. Nulls measured 2026-09-07 at '
              'n=431 (tau=1): rho ~ -0.008, curv_tilt 1.002, interior 1.001,\n'
              '      top-decile 0.100, interior share within 0.001 of the '
              'uniform row. MAE delta does NOT reach zero -- it reads +0.162\n'
              '      +/-0.037 deg, a peakedness offset, not leaked signal '
              '(rho with the real gain = -0.047). ESS is NOT a null here and\n'
              '      falls 0.231 -> 0.100. Compare against these, and check the '
              'uniform arm matches the real run exactly. See shuffle_map(). ***\n')
    print('  Does E point at geometry?  (interior only, silhouette eroded)')
    print(f"    rho(E, curvature)                 {agg('rho_E_curvature'):+.3f}"
          '   <- want clearly positive')
    print(f"    rho(E, image gradient)            {agg('rho_E_imgrad'):+.3f}")
    print(f"    rho(E, image grad | curvature)    "
          f"{agg('rho_E_imgrad_given_curvature'):+.3f}"
          '   <- albedo sensitivity; want small')
    if any('rho_raw_vs_filtered' in r for r in records):
        print(f"    rho(raw bands, filtered bands)    "
              f"{agg('rho_raw_vs_filtered'):+.3f}")
    print('\n  Where does the budget land?  (full mask)')
    print(f"    uniform draw   curv_tilt {agg('curv_tilt_uniform'):.3f}"
          f"  interior {agg('curv_tilt_interior_uniform'):.3f}"
          f"  top-decile {agg('curv_top10_uniform'):.3f}"
          '   (1.000 / 0.100 by construction)')
    for j, tau in enumerate(args.tau):
        pick = lambda r, j=j: r['per_tau'][j]
        print(f"    tau={tau:<5g}        curv_tilt {agg('curv_tilt', pick):.3f}"
              f"  interior {agg('curv_tilt_interior', pick):.3f}"
              f"  top-decile {agg('curv_top10_frac', pick):.3f}"
              f"  E-tilt {agg('wess_tilt', pick):.2f}"
              f"  ESS {agg('wess_ess', pick):.3f}"
              f"  rim {1 - agg('interior_frac', pick):.2f}")
    print('\n  Is there headroom?  B1 MAE on the drawn pixels')
    print(f"    uniform                           {agg('mae_uniform_deg'):.3f} deg")
    for j, tau in enumerate(args.tau):
        pick = lambda r, j=j: r['per_tau'][j]
        print(f"    tau={tau:<5g}                        "
              f"{agg('mae_deg', pick):.3f} deg"
              f"  ({agg('mae_delta_vs_uniform_deg', pick):+.3f})")
    print(f'\n[probe] per-scene records: {out_jsonl}')
    if not args.no_figures:
        print(f'[probe] panels: {args.out_dir}/panel_*.png')


if __name__ == '__main__':
    main()
