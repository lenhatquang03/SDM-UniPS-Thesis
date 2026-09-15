"""Energy-map tap sites for the WESS site screen. Probe-only: nothing in
training imports this file.

A site turns one scene into one [g, g] saliency map E. Every site's E then goes
through the SHIPPED sampler unchanged -- `wess.wess_train_probabilities`, with
one interior mask shared by all sites -- so sites differ only in where E comes
from, never in how it becomes a draw.

    ID  site                            map at R=512
    S0  stage 0, block 0, level 0       64x64    the shipped B2 map
    S1  raw input, level 0              256x256  finest localization
    S2  raw input, level 1              128x128
    S3  raw input, level 2              64x64    same grid as S0: learned stem vs raw
    S4  raw input, levels 0+1+2         256x256  multi-scale
    S5  stage 0, block 0, level 1       32x32
    S6  stage 0, block 0, level 2       16x16
    S7  stage 0, block 0, levels 0+1+2  64x64    multi-scale on the learned stem
    S8  stage 0, block 2, level 0       64x64    one "more learned depth" point

Learned sites read the RAW Haar bands entering `wavelet_convs[level]` of the
chosen WTConv2d. The cascade recurses on the raw LL, so that input is exactly
DWT^level of the block's input, before any WTConv parameter touches it. They
are taken on the tile path and merged exactly as `wess.saliency_maps` does, so
S0 reproduces the shipped map bit for bit (`test_tap_sites.py` checks it).

Decisions fixed before the screen ran:

1. One silhouette band for every site. The interior is `wess.interior_mask` on
   S0's grid (H // 8) with `--erode_cells`, whatever the site's own grid, so
   every site excludes exactly the ring the shipped sampler excludes.
2. Raw sites use the RGB of I*M only. The mask channel's only edge is the rim,
   which is excluded anyway.
3. Raw sites divide each image's map by its own masked mean before the top-k
   reduction over images. Training divides each image by a random U[mean, max]
   scalar, which would otherwise decide which images win the top-k. Learned
   sites are not rescaled: S0 must stay identical to the shipped map.
4. Level roll-up: E = sqrt(sum_l up(E_l^2) / 4^(l - l0)) on the finest level
   l0's grid, nearest-neighbour upsampled -- each level's detail energy spread
   evenly over the area its coefficients cover (Parseval), so a coarse level is
   not over-counted for having fewer, larger cells.
"""
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from modules.model import wess
from modules.model.wtconv import haar_filters


SITES = {
    'S0': dict(kind='learned', stage=0, block=0, levels=(0,), desc='stage 0 block 0 L0 (shipped)'),
    'S1': dict(kind='raw', levels=(0,), desc='raw input L0'),
    'S2': dict(kind='raw', levels=(1,), desc='raw input L1'),
    'S3': dict(kind='raw', levels=(2,), desc='raw input L2'),
    'S4': dict(kind='raw', levels=(0, 1, 2), desc='raw input L0+L1+L2'),
    'S5': dict(kind='learned', stage=0, block=0, levels=(1,), desc='stage 0 block 0 L1'),
    'S6': dict(kind='learned', stage=0, block=0, levels=(2,), desc='stage 0 block 0 L2'),
    'S7': dict(kind='learned', stage=0, block=0, levels=(0, 1, 2), desc='stage 0 block 0 L0+L1+L2'),
    'S8': dict(kind='learned', stage=0, block=2, levels=(0,), desc='stage 0 block 2 L0'),
}
SITE_ORDER = list(SITES)

# Temperatures searched for the matched-ESS operating point, log-spaced.
TAU_GRID = np.geomspace(0.1, 20.0, 33)


# ---------------------------------------------------------------------------
# Learned sites
# ---------------------------------------------------------------------------
def learned_taps_needed(site_ids):
    """Sorted (stage, block, level) keys the chosen learned sites read."""
    return sorted({(SITES[s]['stage'], SITES[s]['block'], lv)
                   for s in site_ids if SITES[s]['kind'] == 'learned'
                   for lv in SITES[s]['levels']})


class LevelTaps:
    """Forward pre-hooks capturing the raw bands entering `wavelet_convs[level]`.

    One hook per (stage, block, level). Each fires on both backbone calls; the
    tile call (`x_grid`) is second, so it is what remains after a forward, and
    `wess.merge_tile_energy` asserts the leading dimension rather than trusting
    that order. Captures are detached and nothing is returned from a hook, so
    the forward itself is untouched.
    """

    def __init__(self, net, keys):
        self.bands, self.channels, self._mods, self._handles = {}, {}, [], []
        for key in keys:
            stage, block, level = key
            dwconv = wess.wtconv_block(net, stage, block)
            if level >= dwconv.wt_levels:
                raise ValueError(f'stage {stage} block {block} has {dwconv.wt_levels} '
                                 f'wavelet level(s); level {level} does not exist')
            self.channels[key] = dwconv.channels
            self._mods.append((key, dwconv.wavelet_convs[level]))

    def __enter__(self):
        for key, mod in self._mods:
            def hook(_module, inputs, key=key):
                self.bands[key] = inputs[0].detach()
            self._handles.append(mod.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def learned_level_maps(taps, n_images, mosaic_scale):
    """{(stage, block, level): [N, g, g]} full-frame energy per image."""
    out = {}
    for key, bands in taps.bands.items():
        energy = wess.subband_energy(bands, taps.channels[key])          # [T*N, h, w]
        out[key] = wess.merge_tile_energy(energy, n_images, mosaic_scale)
    return out


# ---------------------------------------------------------------------------
# Raw-input sites
# ---------------------------------------------------------------------------
def haar_dwt(x):
    """[N, C, H, W] -> [N, C, 4, H/2, W/2]; the same operation as WTConv2d._dwt."""
    n, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError(f'haar_dwt needs even sizes, got {h}x{w}')
    filt = haar_filters(c, dtype=x.dtype).to(x.device)
    return F.conv2d(x, filt, stride=2, groups=c).reshape(n, c, 4, h // 2, w // 2)


def raw_max_level(site_ids):
    levels = [lv for s in site_ids if SITES[s]['kind'] == 'raw' for lv in SITES[s]['levels']]
    return max(levels) if levels else -1


def raw_level_maps(I, M, n_images, max_level):
    """{level: [N, H/2^(l+1), W/2^(l+1)]} detail energy of the RGB of I*M.

    `I` is the batch tensor [1, 3, H, W, Nmax] and `M` [1, 1, H, W], as fed to
    `Net`. The cascade recurses on the raw LL, like WTConv2d.
    """
    x = I[0, :, :, :, :n_images].permute(3, 0, 1, 2).float() * M[0].float()
    out, ll = {}, x
    for level in range(max_level + 1):
        bands = haar_dwt(ll)
        n, c, _, h, w = bands.shape
        out[level] = wess.subband_energy(bands.reshape(n, 4 * c, h, w), c)
        ll = bands[:, :, 0]
    return out


# ---------------------------------------------------------------------------
# Site map
# ---------------------------------------------------------------------------
def roll_up(level_maps, levels):
    """Decision 4. A single level is returned unchanged."""
    if len(levels) == 1:
        return level_maps[levels[0]]
    l0 = min(levels)
    grid = level_maps[l0].shape[-2:]
    acc = torch.zeros_like(level_maps[l0])
    for level in levels:
        e2 = level_maps[level] ** 2
        if level != l0:
            e2 = F.interpolate(e2[:, None], size=grid, mode='nearest')[:, 0] / (4 ** (level - l0))
        acc = acc + e2
    return torch.sqrt(acc)


def per_image_mean_normalize(E_imgs, mask_hw):
    """Decision 3: [N, g, g] divided by each image's mean over masked cells."""
    cells = F.adaptive_avg_pool2d(mask_hw[None, None].float(), E_imgs.shape[-2:])[0, 0] >= 0.5
    if not bool(cells.any()):
        return E_imgs
    mean = E_imgs[:, cells].mean(dim=1).clamp_min(1e-12)
    return E_imgs / mean[:, None, None]


def site_map(site_id, learned_maps, raw_maps, mask_hw, top_k=2):
    """One scene's [g, g] saliency map for `site_id`."""
    s = SITES[site_id]
    if s['kind'] == 'learned':
        per_level = {lv: learned_maps[(s['stage'], s['block'], lv)] for lv in s['levels']}
    else:
        per_level = {lv: raw_maps[lv] for lv in s['levels']}
    E_imgs = roll_up(per_level, s['levels'])
    if s['kind'] == 'raw':
        E_imgs = per_image_mean_normalize(E_imgs, mask_hw)
    return wess.reduce_over_lights(E_imgs, top_k=top_k)


def shared_interior(mask_hw, erode_cells):
    """Decision 1: the shipped interior, on S0's grid, for every site."""
    H, W = mask_hw.shape[-2:]
    return wess.interior_mask(mask_hw, (H // 8, W // 8), erode_cells)


# ---------------------------------------------------------------------------
# Operating point and seeds
# ---------------------------------------------------------------------------
def ess_fraction(p):
    """1 / sum(p^2) / n -- the same float32 formula as `wess.train_draw_stats`."""
    return float(1.0 / (p.detach().float().pow(2).sum() + 1e-12) / p.numel())


def tau_for_ess(mean_ess_curve, target):
    """Temperature whose mean ESS (over scenes) hits `target`, interpolated in
    log tau on TAU_GRID. Returns (tau, ok); ok is False when the target lies
    outside the curve's range, in which case the nearest grid end is returned.
    """
    y = np.maximum.accumulate(np.asarray(mean_ess_curve, dtype=float))
    log_tau = np.log(TAU_GRID)
    if target <= y[0]:
        return float(TAU_GRID[0]), False
    if target >= y[-1]:
        return float(TAU_GRID[-1]), False
    xs, first = np.unique(y, return_index=True)
    return float(np.exp(np.interp(target, xs, log_tau[first]))), True


def shuffle_seed(seed, scene_index, site_id):
    """Per (scene, site): the same permutation whatever other sites run."""
    return (seed * 7_919 + scene_index * 104_729 + SITE_ORDER.index(site_id) * 31 + 1) % (2 ** 31 - 1)


def draw_seed(seed, scene_index, config_name, arm_index):
    """Per (scene, configuration, arm), stable across runs and --sites order."""
    return (seed * 1_000_003 + scene_index * 9_973
            + zlib.crc32(config_name.encode()) + arm_index) % (2 ** 31 - 1)
