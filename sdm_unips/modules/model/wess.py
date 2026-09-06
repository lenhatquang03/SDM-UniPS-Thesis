"""Model C (WESS) -- Wavelet-Energy Saliency Sampling.

The training pixel sampler under study. `Net.sample_train_pixels` draws its
2048 pixels uniformly over the mask; WESS draws them from a distribution built
from the sub-band energy of the WTConv backbone's *first* block, so the
gradient of each optimizer step concentrates on creases, shadow boundaries and
other high-frequency structure rather than on the flat regions that dominate a
uniform draw and contribute almost no gradient.

**Nothing here is wired into `Net`.** This module is imported by
`wess_probe.py` (Phase 0) and, later, by `Net.sample_train_pixels`. Keeping it
standalone means the Phase-0 measurements are taken on exactly the code that
will ship, while `model.py`, `convnext.py` and `wtconv.py` stay byte-identical
to Model B1's -- so the eventual B1 -> B2 diff is still readable as a `git diff`.

Design decisions, each settled before implementation:

* **Stage 0, block 0, level 0.** Stage resolutions at R=256 are 64/32/16/8, so
  level-0 detail bands are 32/16/8/4 per tile. Only stage 0 has usable
  localization: its 32x32 bands merge to a 64x64 full-frame map whose cells are
  8x8 image pixels, against stage 3's 64x64-pixel cells, which would select
  whole object quadrants and degenerate the distribution to uniform. Block 0
  because its input is closest to the raw stem output -- most photometric,
  least semantic, and photometric stereo reads normals from intensity
  gradients. Level 0 because it is the finest band; folding levels 1-2 in is a
  legitimate multi-scale variant, and a separate experiment.

* **The tile path, not the resized path.** `ScaleInvariantSpatialLightImageEncoder`
  runs the backbone twice: on `x_resized` (N images, `F.interpolate`d down to
  the canonical resolution) and on `x_grid` (4N interleaved stride-2 tiles).
  `x_resized` is a bilinear *downsample*, i.e. a low-pass filter -- it destroys
  precisely the high frequencies this method measures. The tiles subsample
  instead, so the energy survives (aliased in phase, present in magnitude), and
  `merge_tensor_spatial` re-interleaves the four 32x32 maps into a coherent
  64x64.

  Honest caveat, worth stating in the thesis rather than leaving to a reviewer:
  the four tiles' cell (i,j) all summarize the *same* 16x16 image region at
  four different phases, and the merge assigns them to four adjacent 8x8
  quadrants. Nominal cell size is 8x8; true localization is ~16 px. WESS
  concentrates the budget on crease *neighbourhoods*, not on crease pixels.
  (The GLC is 128x128 and `_decode_pixels` interpolates it bilinearly, so a
  pixel drawn a few px off a crease still pulls a feature vector substantially
  shaped by that crease.)

* **mean-of-top-2 over the K images.** A crease is only visible under *some*
  lighting; under a light that leaves it flatly shaded it produces no gradient
  at all. A mean over K would divide a strong response present in 1 of 10
  images by 10 -- the wrong question. `max` asks "is this location informative
  under at least one light", which is the criterion photometric stereo actually
  operates on, but it is also maximally sensitive to a single blown-out
  specular highlight. mean-of-top-2 keeps the semantics and survives one
  outlier image.

* **Flat draw, not hierarchical.** The map is 64x64; `ids` index the 512x512
  decoder grid. A hierarchical draw (pick a cell proportional to its energy,
  then a pixel inside it) gives every cell the same mass regardless of how much
  of it the mask covers -- so a silhouette cell holding 5 masked pixels of 64
  receives a full cell's budget and duplicates those 5. Silhouette cells are
  high-energy by construction (the object outline is the strongest edge in the
  frame) and the silhouette is the least interesting structure to supervise.
  The flat draw's coverage weighting is the correct behaviour, not a defect,
  and it composes with the existing `sample_train_pixels(valid_ids, n_sample)`
  signature: `valid_ids` *is* the mask, so masking is a gather.

* **Mask before normalizing.** `exp(0) = 1`, so a softmax over the full frame
  hands background pixels -- zero energy -- a share of the budget proportional
  to their count. With a 40% object mask that is ~60% of the draw thrown away.

* **Standardize before the softmax.** `E` is an L2 norm over 96 channels of a
  *learned* feature map, so its scale drifts as the encoder trains; `tau = 0.1`
  at epoch 1 and at epoch 80 would be different samplers. Standardizing over
  the masked set puts `tau` in units of sigma, stable across scenes and across
  training.

* **A uniform floor, `lam`, rather than a separate base set.** The proposal
  splits the budget into M_base = 512 uniform + M_adapt = 1536 saliency-drawn,
  which needs bookkeeping to keep the two draws disjoint. The mixture
  `P = lam * Uniform + (1 - lam) * P_wess` with `lam = 512 / 2048 = 0.25` has
  the same expected count per pixel, takes one `multinomial`, and turns the
  support argument into a hard guarantee: every masked pixel keeps
  `p >= lam / n_valid` no matter how peaked the softmax gets. That is what
  makes training under E_P while reporting E_uniform defensible -- the sampler
  changes the *rate* at which a pixel is supervised, never whether it can be.
  `lam = 1` (or `tau -> inf`) recovers Model A's uniform sampler exactly, which
  is a free correctness check.

* **The draw runs on a CPU generator**, mirroring `Trainer._eval_sample_ids`:
  it makes the sample a function of the seed alone, independent of the device
  and of how much CUDA RNG the rest of the step consumed.

Everything is `detach()`ed. The draw is an index selection and carries no
gradient; keeping the live tensor would pin the whole encoder graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decompose_tensors import merge_tensor_spatial


# The agreed tap site. Not exposed as a training flag -- like `wt_levels`, it is
# an architecture decision, and the stage sweep is a separate experiment.
DEFAULT_STAGE = 0
DEFAULT_BLOCK = 0

# Defaults for the draw. `lam` reproduces the proposal's 512/2048 base set.
DEFAULT_TAU = 1.0
DEFAULT_LAM = 0.25


def unwrap(net):
    """The bare `Net`, whether or not it is DataParallel-wrapped."""
    return net.module if isinstance(net, nn.DataParallel) else net


def wtconv_block(net, stage=DEFAULT_STAGE, block=DEFAULT_BLOCK):
    """The `WTConv2d` of one ConvNeXt block of the image encoder's backbone.

    `ImageFeatureExtractor.backbone` is an `nn.Sequential` holding the single
    `ConvNeXt`, hence the `[0]`.
    """
    convnext = unwrap(net).image_encoder.backbone.backbone[0]
    return convnext.stages[stage][block].dwconv


class SubbandTap:
    """Capture one WTConv block's level-0 sub-bands, without touching the model.

    `WTConv2d.tap_subbands` exists for this, but it is last-writer-wins across
    all 12 blocks (so after a forward it holds stage 3's 4x4 bands, which are
    useless here) and it only ever exposes the *filtered* bands. Hooks give both
    variants and a single, explicit tap site, and leave `wtconv.py` untouched:

    * `band='raw'`   -- a forward **pre**-hook on `wavelet_convs[0]`, whose
      input is exactly the raw DWT output `bands.reshape(n, 4C, h, w)`. This is
      the `X^LH / X^HL / X^HH` of the proposal: pure Haar coefficients, no
      learned parameters involved.
    * `band='filtered'` -- a forward hook on `wavelet_scale[0]`, i.e. the bands
      after the level-0 depthwise 5x5 and the per-channel gain. That gain is
      initialised at 0.1 and is trainable, so this variant is scaled by
      something that drifts during training. Phase 0 measures whether the two
      rank pixels differently; `raw` is the default because it is the quantity
      the proposal defines and the one that cannot drift.

    Both hooks fire on **each** of the encoder's two backbone calls. `last`
    therefore holds the tile path (`x_grid`, 4N), which is the one WESS wants,
    because that call is second -- but `saliency_maps` asserts on the leading
    dimension rather than relying on the ordering.
    """

    def __init__(self, net, stage=DEFAULT_STAGE, block=DEFAULT_BLOCK, band='raw'):
        if band not in ('raw', 'filtered'):
            raise ValueError(f"band must be 'raw' or 'filtered', got {band!r}")
        self.dwconv = wtconv_block(net, stage, block)
        if self.dwconv.wt_levels < 1:
            raise RuntimeError('the tapped WTConv2d has no wavelet levels')
        self.band = band
        self.channels = self.dwconv.channels
        self.last = None
        self._handle = None

    def _capture(self, t):
        self.last = t.detach()

    def __enter__(self):
        if self.band == 'raw':
            def pre_hook(_module, inputs):
                self._capture(inputs[0])
            self._handle = self.dwconv.wavelet_convs[0].register_forward_pre_hook(pre_hook)
        else:
            def hook(_module, _inputs, output):
                self._capture(output)
            self._handle = self.dwconv.wavelet_scale[0].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def subband_energy(bands, channels):
    """[N, 4C, h, w] -> [N, h, w]: `sqrt(sum_c (LH^2 + HL^2 + HH^2))`, Eq. 1.4.

    The channel layout is `4*c + k` (filter `k` of input channel `c`), which is
    what `haar_filters`' `repeat(C, 1, 1, 1)` produces and what `WTConv2d`
    reshapes back to; band 0 is LL and is dropped.
    """
    n, c4, h, w = bands.shape
    if c4 != 4 * channels:
        raise RuntimeError(f'expected {4 * channels} band channels, got {c4}')
    detail = bands.reshape(n, channels, 4, h, w)[:, :, 1:4, :, :].float()
    return torch.sqrt((detail * detail).sum(dim=(1, 2)) + 1e-12)


def merge_tile_energy(energy, n_images, mosaic_scale):
    """[K*N, h, w] tile energies -> [N, h*ms, w*ms] full-frame maps.

    The encoder builds `x_grid` as `divide_tensor_spatial(...).permute(1,0,...)`,
    so the leading axis runs `k*N + n`; `merge_tensor_spatial` inverts exactly
    that with `nn.Fold`, re-interleaving the four stride-2 phases.
    """
    K = mosaic_scale * mosaic_scale
    if energy.shape[0] != K * n_images:
        raise RuntimeError(
            f'sub-band tap holds {energy.shape[0]} maps, expected K*N = '
            f'{K}*{n_images} = {K * n_images}. The tap fires on BOTH backbone '
            f'calls; this looks like the resized path (N maps), which is a '
            f'low-passed downsample and must not feed the saliency map.')
    h, w = energy.shape[-2:]
    tiles = energy.reshape(K, n_images, 1, h, w)
    return merge_tensor_spatial(tiles, method='tile_stride')[:, 0]


def reduce_over_lights(energy, top_k=2):
    """[K_imgs, H, W] -> [H, W] by the mean of the `top_k` largest per pixel."""
    k = max(1, min(int(top_k), energy.shape[0]))
    return energy.topk(k, dim=0).values.mean(dim=0)


def saliency_maps(tap, nImgArray, mosaic_scale, top_k=2):
    """Per-batch-element saliency maps from a populated `SubbandTap`.

    Returns a list of B tensors, each [h*ms, w*ms] (64x64 at R=512/canonical=256).
    """
    if tap.last is None:
        raise RuntimeError('the sub-band tap is empty -- run a forward inside '
                           'the `with SubbandTap(...)` block')
    n_imgs = [int(n) for n in nImgArray]
    total = sum(n_imgs)
    energy = subband_energy(tap.last, tap.channels)          # [K*N, h, w]
    merged = merge_tile_energy(energy, total, mosaic_scale)  # [N, H', W']
    out, p = [], 0
    for n in n_imgs:
        out.append(reduce_over_lights(merged[p:p + n], top_k=top_k))
        p += n
    return out


def wess_probabilities(E, valid_ids, H, W, tau=DEFAULT_TAU, lam=DEFAULT_LAM):
    """The sampling distribution over `valid_ids`, plus the raw energy there.

    Order of operations is load-bearing and was agreed before implementation:
    upsample -> gather at the mask -> standardize over the masked set ->
    softmax -> mix with uniform. Doing the softmax before the mask would spend
    the budget on background; doing it before standardizing would make `tau`
    mean something different at every epoch.

    Returns `(p, e)` -- `p` sums to 1 over `valid_ids`, `e` is the unstandardized
    energy at those pixels (the reference for the tilt diagnostic).
    """
    E_up = F.interpolate(E[None, None].float(), size=(H, W),
                         mode='bilinear', align_corners=False).reshape(-1)
    e = E_up[valid_ids]
    n_valid = e.numel()
    if n_valid == 0:
        return e.new_zeros(0), e

    z = (e - e.mean()) / (e.std() + 1e-6)
    logits = z / max(float(tau), 1e-6)
    p_wess = torch.softmax(logits - logits.max(), dim=0)

    lam = float(min(max(lam, 0.0), 1.0))
    p = lam / n_valid + (1.0 - lam) * p_wess
    return p / p.sum(), e


def _draw(p, n_sample, generator):
    """`n_sample` indices into `p`, without replacement where the mask allows.

    Drawn on CPU so the sample depends on the seed alone -- not on the device,
    and not on how much CUDA RNG the rest of the step consumed. That mirrors
    `Trainer._eval_sample_ids`, which the A/B contract already relies on.
    """
    p_cpu = p.detach().float().cpu()
    replace = p_cpu.numel() < n_sample
    return torch.multinomial(p_cpu, n_sample, replacement=replace,
                             generator=generator)


def wess_sample(E, valid_ids, H, W, n_sample,
                tau=DEFAULT_TAU, lam=DEFAULT_LAM, generator=None):
    """Draw `n_sample` flat decoder-grid indices. Returns `(ids, stats)`.

    `stats` is the per-step instrumentation agreed for `train.jsonl`:

    * `wess_tilt` -- mean energy of the drawn set over mean energy of the mask.
      1.0 means the sampler is inert; this is the one number that says whether
      `tau` is set sanely.
    * `wess_ess` -- effective sample size `1 / sum(p^2)` as a fraction of the
      mask. Catches both failure modes at once: collapse (all mass on a handful
      of pixels) and degeneracy (indistinguishable from uniform, which reads
      1.0).
    * `wess_top10_frac` -- share of the draw landing in the mask's top energy
      decile. Uniform reads 0.10.
    """
    if valid_ids.numel() == 0:
        # No valid pixels: a placeholder, exactly as `sample_train_pixels` does.
        # Loss masking discards these.
        zeros = torch.zeros(n_sample, dtype=torch.long, device=valid_ids.device)
        return zeros, {'wess_tilt': float('nan'), 'wess_ess': float('nan'),
                       'wess_top10_frac': float('nan'), 'n_valid': 0}

    p, e = wess_probabilities(E, valid_ids, H, W, tau=tau, lam=lam)
    sel = _draw(p, n_sample, generator).to(valid_ids.device)
    return valid_ids[sel], draw_stats(e, p, sel)


def uniform_sample(valid_ids, n_sample, generator=None):
    """Model A's sampler, seeded. The paired baseline for every WESS diagnostic.

    Identical in distribution to `Net.sample_train_pixels`; it takes an explicit
    generator so a probe can draw the uniform and the WESS sets from the same
    seed and compare them on one scene.
    """
    n_valid = int(valid_ids.numel())
    if n_valid == 0:
        return torch.zeros(n_sample, dtype=torch.long, device=valid_ids.device)
    if n_valid >= n_sample:
        sel = torch.randperm(n_valid, generator=generator)[:n_sample]
    else:
        sel = torch.randint(0, n_valid, (n_sample,), generator=generator)
    return valid_ids[sel.to(valid_ids.device)]


def draw_stats(e, p, sel):
    """The three sampler diagnostics, for a draw `sel` into the masked set."""
    e = e.detach().float()
    n_valid = e.numel()
    mean_all = e.mean()
    thresh = torch.quantile(e, 0.9) if n_valid > 1 else e.max()
    sel = sel.to(e.device)
    return {
        'wess_tilt': float(e[sel].mean() / (mean_all + 1e-12)),
        'wess_ess': float(1.0 / (p.detach().float().pow(2).sum() + 1e-12) / n_valid),
        'wess_top10_frac': float((e[sel] >= thresh).float().mean()),
        'n_valid': int(n_valid),
    }


# ---------------------------------------------------------------------------
# Training sampler (Model B2)
#
# Phase 0 (n=431, the full val split, both sources) settled the shape of this.
# Two findings drove it:
#
# 1. `E` at stage 0 block 0 is dominated by the OCCLUDING CONTOUR. The panels
#    show a bright ring one to two cells wide and a near-flat interior. That is
#    not a discovery about geometry -- `Net.forward` feeds the backbone
#    `cat([I * M, M])`, so the mask itself is a hard step at the boundary in
#    all four input channels, and the level-0 Haar detail bands of a step edge
#    are enormous by construction. Left alone the sampler spends ~40% of the
#    budget (tau=1) tracing a discontinuity that the mask channel put there,
#    against 7-9% for a uniform draw. "Sample the silhouette more" needs no
#    wavelets and is not the thesis claim.
#
# 2. The interior signal is real but was being crushed by that ring. Over the
#    eroded interior the curvature tilt is 1.20 at tau=1, and across scenes
#    rho(MAE gain, interior curvature tilt) = +0.39 -- the strongest coupling in
#    the probe. The reason it is only 1.20 is the standardization: `e.std()`
#    over the whole mask is inflated by the rim, so every interior z-score is
#    squashed toward 0 and the softmax over the interior comes out nearly flat.
#
# So the rim is excluded from E's statistics and from the softmax, but NOT from
# the draw -- it keeps exactly its uniform share. WESS reallocates only the
# interior's own share, among interior pixels:
#
#     p(rim pixel)      = 1 / n_valid                        (Model A, unchanged)
#     p(interior pixel) = (n_int / n_valid) * [ lam / n_int
#                                             + (1 - lam) * softmax_int ]
#
# The two blocks sum to n_rim/n_valid + n_int/n_valid = 1 exactly, so no
# renormalization is needed and none is done. Properties this buys:
#
# * `lam = 1` or `tau -> inf` still recovers Model A's uniform draw exactly, on
#   the rim and in the interior alike -- the free correctness check survives.
# * The hard per-pixel floor survives and is unchanged: an interior pixel keeps
#   p >= (n_int/n_valid) * lam/n_int = lam/n_valid, a rim pixel keeps 1/n_valid.
#   Every masked pixel is still reachable, which is what makes training under
#   E_P while reporting E_uniform defensible.
# * Any A/B difference is attributable to interior reallocation alone. The
#   silhouette, where GT normals sit at grazing angles and the mask edge is
#   antialiased, is sampled identically to Model A -- so a win cannot be
#   explained away as "B2 just trained harder on the boundary".
#
# The erosion runs on the 64x64 energy grid rather than at 512x512 because that
# is the resolution at which the ring is actually defined, and because one cell
# there is ~16 px of true localization anyway (see the module docstring).
# ---------------------------------------------------------------------------

# Ring width to exclude, in energy-grid cells. 2 cells ~ 16 px nominal / ~24 px
# effective at R=512. Phase 0's `--erode 9` (~1 cell) still left tau=1 with 40%
# of the draw outside the interior, so the ring is wider than one cell.
DEFAULT_ERODE_CELLS = 2

# Below this many interior PIXELS the object is a thin sliver and eroding it
# leaves nothing to reallocate; fall back to the uniform draw for that scene
# rather than to a softmax whose mean and sigma come from a handful of pixels.
MIN_INTERIOR_PIXELS = 256


def interior_mask(mask_hw, grid_hw, erode_cells=DEFAULT_ERODE_CELLS):
    """[H, W] 0/1 mask -> [H, W] 0/1 interior mask, eroded on the energy grid.

    Area-pools the decoder-resolution mask down to the energy grid, keeps only
    cells that are FULLY inside the object (`>= 1 - 1e-6`, so a cell straddling
    the boundary is already dropped before any erosion), erodes by
    `erode_cells` with a max-pool on the complement, then upsamples back with
    nearest so the result is exactly a union of energy cells.
    """
    m = mask_hw[None, None].float()
    cells = F.adaptive_avg_pool2d(m, grid_hw)
    full = (cells >= 1.0 - 1e-6).float()
    k = 2 * int(erode_cells) + 1
    if k > 1:
        # Erosion = -dilation(-x): a cell survives only if every cell within
        # `erode_cells` of it is also full.
        full = 1.0 - F.max_pool2d(1.0 - full, kernel_size=k, stride=1, padding=k // 2)
    up = F.interpolate(full, size=mask_hw.shape[-2:], mode='nearest')
    return (up[0, 0] * mask_hw).clamp_(0.0, 1.0)


def wess_train_probabilities(E, valid_ids, interior_flat, H, W,
                             tau=DEFAULT_TAU, lam=DEFAULT_LAM):
    """The rim-preserving sampling distribution over `valid_ids`.

    `interior_flat` is the flattened [H*W] interior mask from `interior_mask`.
    Returns `(p, e, is_int)`: `p` sums to 1 over `valid_ids`, `e` is the raw
    energy there, `is_int` is the boolean interior selector into that same set.
    """
    E_up = F.interpolate(E[None, None].float(), size=(H, W),
                         mode='bilinear', align_corners=False).reshape(-1)
    e = E_up[valid_ids]
    n_valid = e.numel()
    is_int = interior_flat.reshape(-1)[valid_ids] > 0
    n_int = int(is_int.sum())

    lam = float(min(max(lam, 0.0), 1.0))
    # Degenerate cases fall back to Model A rather than to a distribution whose
    # statistics are estimated from a handful of pixels.
    if n_int < MIN_INTERIOR_PIXELS or lam >= 1.0:
        return e.new_full((n_valid,), 1.0 / n_valid), e, is_int

    e_int = e[is_int]
    z = (e_int - e_int.mean()) / (e_int.std() + 1e-6)
    logits = z / max(float(tau), 1e-6)
    p_int = torch.softmax(logits - logits.max(), dim=0)

    share = n_int / n_valid
    p = e.new_full((n_valid,), 1.0 / n_valid)          # rim keeps Model A's rate
    p[is_int] = share * (lam / n_int + (1.0 - lam) * p_int)
    return p, e, is_int


def wess_sample_train(E, mask_hw, valid_ids, H, W, n_sample,
                      tau=DEFAULT_TAU, lam=DEFAULT_LAM,
                      erode_cells=DEFAULT_ERODE_CELLS, generator=None):
    """Draw `n_sample` flat decoder-grid indices for one training scene.

    Unlike `wess_sample` (the Phase-0 probe path) the draw runs on the tensors'
    own device using the ambient RNG, exactly as Model A's `randperm` does. A
    CPU generator would be more reproducible in isolation but would make B2's
    draw depend on a different RNG stream than A's, and would force a device
    sync per batch element per step. `--resume` restores the CUDA and CPU
    streams alike, so this is resumable on the same terms Model A is.

    Returns `(ids, stats)`.
    """
    if valid_ids.numel() == 0:
        zeros = torch.zeros(n_sample, dtype=torch.long, device=valid_ids.device)
        return zeros, {'wess_tilt': float('nan'), 'wess_tilt_int': float('nan'),
                       'wess_ess': float('nan'), 'wess_top10_frac': float('nan'),
                       'wess_interior_frac': float('nan'), 'n_valid': 0}

    interior = interior_mask(mask_hw, E.shape[-2:], erode_cells)
    p, e, is_int = wess_train_probabilities(E, valid_ids, interior, H, W,
                                            tau=tau, lam=lam)
    replace = p.numel() < n_sample
    sel = torch.multinomial(p, n_sample, replacement=replace, generator=generator)
    return valid_ids[sel], train_draw_stats(e, p, sel, is_int)


def train_draw_stats(e, p, sel, is_int):
    """Per-step sampler instrumentation for `train.jsonl` (Phase 1).

    `wess_tilt_int` is the honest one: the energy tilt achieved *within the
    interior*, which is the only region WESS is allowed to reweight. `wess_tilt`
    over the whole mask stays for comparability with the Phase-0 numbers, but it
    is diluted by the rim samples, which are uniform by construction.
    `wess_interior_frac` should sit near the uniform value -- a drift away from
    it means the erosion is not doing what it claims.
    """
    e = e.detach().float()
    n_valid = e.numel()
    sel = sel.to(e.device)
    sel_int = is_int[sel]
    e_sel = e[sel]
    thresh = torch.quantile(e, 0.9) if n_valid > 1 else e.max()

    tilt_int = float('nan')
    if bool(sel_int.any()) and bool(is_int.any()):
        tilt_int = float(e_sel[sel_int].mean() / (e[is_int].mean() + 1e-12))
    return {
        'wess_tilt': float(e_sel.mean() / (e.mean() + 1e-12)),
        'wess_tilt_int': tilt_int,
        'wess_ess': float(1.0 / (p.detach().float().pow(2).sum() + 1e-12) / n_valid),
        'wess_top10_frac': float((e_sel >= thresh).float().mean()),
        'wess_interior_frac': float(sel_int.float().mean()),
        'n_valid': int(n_valid),
    }
