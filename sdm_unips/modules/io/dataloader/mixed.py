"""
Mixed training-dataset wrapper for hdlong-complexv1 + PolarPS.

The Dataset discovers scene directories under one or more roots, infers
which loader each scene belongs to (hdlong has `light_means.config`,
PolarPS has `normal.exr`), and dispatches to the appropriate per-scene
loader. `__getitem__` returns the 4-tuple `(I, N, M, n_imgs)` consumed by
`train._collate`.

`build_mixed_split` performs a deterministic, scene-level train/val split
of the discovered pool so checkpoint selection uses a held-out subset of
the synthetic data instead of leaking the DiLiGenT test set.

Required attrs on `args`:
    - hdlong_dir (optional)
    - polarps_dir (optional)
    - train_dir (optional fallback: auto-detect both kinds under one root)
    - train_resolution, mask_margin, session_name, seed
    - max_scenes (optional cap; default unlimited)
    - val_fraction (held-out fraction for the val split; default 0.1)

K is fixed at 10 per scene inside both `HdlongLoader` and `PolarPSLoader`.
"""

import os

import numpy as np
import torch.utils.data as data

from typing import Callable

from .hdlong import HdlongLoader
from .polarps import PolarPSLoader

K_PER_SCENE = 10


def _is_hdlong_scene(path: str) -> bool:
    return os.path.isfile(os.path.join(path, 'light_means.config'))


def _is_polarps_scene(path: str) -> bool:
    return os.path.isfile(os.path.join(path, 'normal.exr'))


def _find_scenes(root: str, is_scene: Callable[[str], bool]) -> list[str]:
    """Find every directory at or under `root` that *directly* holds a scene
    marker, regardless of nesting depth.

    Scenes live at different depths across sources: hdlong scenes are direct
    children of `hdlong-complexv1/`, while PolarPS scenes sit one level deeper
    under a per-object group dir (`PolarPS/<group>/<scene>/normal.exr`). A
    marker-based walk handles both. We prune below a recognized scene so we
    never descend into the heavy `cam_*/` (hdlong) or `light-NN/` (PolarPS)
    leaf trees, and we sort for a deterministic, seed-reproducible order.
    """
    if not root or not os.path.isdir(root):
        return []
    found = []
    # os.walk(root) returns the name of the directory, a list of sub-directories, and a list of files in that directory
    # It decides the next directory to traverse by looking at the previously found list of sub-directories.
    for dirpath, subdirs, _ in os.walk(root):
        if is_scene(dirpath):
            found.append(dirpath)
            subdirs[:] = []  # scene found: don't recurse into its leaves
    return sorted(found)


def _discover(roots: list[str], kinds: list[str]) -> list[tuple[str, str]]:
    """Return list of (kind, scene_dir) for matching roots."""
    predicate = {'hdlong': _is_hdlong_scene, 'polarps': _is_polarps_scene}
    out = []
    for kind, root in zip(kinds, roots):
        for d in _find_scenes(root, predicate[kind]):
            out.append((kind, d))
    return out


def _auto_discover(root: str) -> list[tuple[str, str]]:
    out = [('hdlong', d) for d in _find_scenes(root, _is_hdlong_scene)]
    out += [('polarps', d) for d in _find_scenes(root, _is_polarps_scene)]
    return out


def _proportional_cap(
        hd: list[tuple[str, str]], pp: list[tuple[str, str]], 
        max_scenes: int, seed: int
    ):
    """Sub-sample (hd, pp) down to `max_scenes` while preserving the full-pool
    hdlong:polarps ratio.

    A single uniform draw over the concatenated pool (the previous approach)
    could wipe out the minority source entirely when the cap is small —
    PolarPS is ~1% of the combined pool, so `permutation(len)[:max_scenes]`
    can easily return zero PolarPS scenes. Instead we allocate the budget
    between the two sources in proportion to their full-pool sizes, then draw
    within each source independently. Every source that is present keeps at
    least one scene, so the mix never silently collapses to a single source.

    Both draws use the same seed for a deterministic, reproducible sub-pool.
    """
    total = len(hd) + len(pp)
    if total <= max_scenes:
        return hd, pp

    # n_hd / max_scenes ~ len(hd) / total
    n_hd = int(round(max_scenes * len(hd) / total)) if hd else 0
    # Clamp n_hd between 1 and len(hd). If n_hd < 1, n_hd=1. If n_hd > len(hd), n_hd=len(hd). Else n_hd
    n_hd = min(max(n_hd, 1), len(hd)) if hd else n_hd
    # Clamp n_pp beteween 1 and len(pp)
    n_pp = min(max(max_scenes - n_hd, 1), len(pp)) if pp else 0
    # If n_hd + n_pp < max_scenes still, take more hdlong-complexv1 scenes.
    n_hd = min(max(max_scenes - n_pp, 1 if hd else 0), len(hd))

    rng = np.random.RandomState(seed)
    hd_sampled = [hd[i] for i in sorted(rng.permutation(len(hd))[:n_hd].tolist())]
    pp_sampled = [pp[i] for i in sorted(rng.permutation(len(pp))[:n_pp].tolist())]
    return hd_sampled, pp_sampled


def _discover_scenes(args):
    """Discover and (optionally) cap the combined hdlong + PolarPS scene pool.

    The cap (`--max_scenes`) is applied here, before any train/val split, so
    the split halves are carved out of the capped pool. The cap is allocated
    per source in proportion to the full-pool ratio (see `_proportional_cap`)
    rather than as a uniform draw, so the minority source cannot be sampled
    out of existence.
    """
    hd_dir = getattr(args, 'hdlong_dir', None)
    pp_dir = getattr(args, 'polarps_dir', None)
    hd = _discover([hd_dir], ['hdlong'])
    pp = _discover([pp_dir], ['polarps'])

    # Fail loud on a configured-but-empty root.
    if hd_dir and os.path.isdir(hd_dir) and not hd:
        print(f"[MixedTrainDataset] WARNING: --hdlong_dir='{hd_dir}' exists but "
              f"yielded 0 scenes (no 'light_means.config' marker found at any "
              f"depth). Did you extract hdlong_config.zip into the same tree?")
    if pp_dir and os.path.isdir(pp_dir) and not pp:
        print(f"[MixedTrainDataset] WARNING: --polarps_dir='{pp_dir}' exists but "
              f"yielded 0 scenes (no 'normal.exr' marker found at any depth). "
              f"Check that the flag points at the PolarPS root.")

    # Execute only if hd = pp = [] with non-NaN train_dir
    if not (hd or pp) and getattr(args, 'train_dir', None):
        auto = _auto_discover(args.train_dir)
        hd = [s for s in auto if s[0] == 'hdlong']
        pp = [s for s in auto if s[0] == 'polarps']

    if len(hd) + len(pp) == 0:
        raise RuntimeError(
            'MixedTrainDataset found no scenes. Pass --hdlong_dir and/or '
            '--polarps_dir, or place scenes under --train_dir.'
        )

    max_scenes = getattr(args, 'max_scenes', None)
    if max_scenes is not None and max_scenes > 0:
        hd, pp = _proportional_cap(hd, pp, max_scenes, getattr(args, 'seed', 42))
    return hd + pp


def build_mixed_split(args, augment=True):
    """Deterministic scene-level train/val split of the mixed pool.

    The pool is split proportionally *within* each source (hdlong split
    separately from PolarPS) so the mix ratio is preserved in both halves.
    The split is fully determined by `--seed`, so calling this twice with the
    same args yields identical, disjoint train/val scene lists.

    Returns (train_dataset, val_dataset); the val set never augments.
    """
    # Validate the cheap argument before the (expensive) filesystem walk, so a
    # bad flag dies immediately instead of after discovering ~1200 scenes.
    val_fraction = float(getattr(args, 'val_fraction', 0.1))
    seed = int(getattr(args, 'seed', 42))
    if not 0 < val_fraction < 1:
        raise ValueError(
            f"--val_fraction must be in the open interval (0, 1); got "
            f"{val_fraction}. Use e.g. 0.1 for a 10% held-out split."
        )

    scenes = _discover_scenes(args)
    hd = [s for s in scenes if s[0] == 'hdlong']
    pp = [s for s in scenes if s[0] == 'polarps']

    def _split(
        items: list[tuple[str, str]], rng: np.random.RandomState, 
        name: str, flag: str
    ):
        n = len(items)
        if n == 0:
            return [], []
        # A source present in the pool must land in *both* halves. That needs
        # at least one scene per side, so a single-scene source is unsplittable
        # and n_val is clamped into [1, n-1] (>=1 val, >=1 train) otherwise.
        if n == 1:
            raise RuntimeError(
                f"The {name} pool has only 1 scene, which cannot be placed in "
                f"both the train and val splits. Provide at least 2 {name} "
                f"scenes, or drop {flag} to train without {name}."
            )
        perm = rng.permutation(n)
        n_val = int(round(n * val_fraction))
        n_val = min(max(n_val, 1), n - 1)
        val_pos = set(perm[:n_val].tolist())
        train = [items[i] for i in range(n) if i not in val_pos]
        val = [items[i] for i in range(n) if i in val_pos]
        return train, val

    hd_train, hd_val = _split(hd, np.random.RandomState(seed + 1),
                              'hdlong', '--hdlong_dir')
    pp_train, pp_val = _split(pp, np.random.RandomState(seed + 2),
                              'polarps', '--polarps_dir')

    # Ensure the ratio between hdlong-complexv1 and PolarPS scenes are the same
    train_ds = MixedTrainDataset(args, augment=augment,
                                 scenes=hd_train + pp_train,
                                 subset_name='MixedTrain', noun='train')
    val_ds = MixedTrainDataset(args, augment=False,
                               scenes=hd_val + pp_val,
                               subset_name='MixedVal', noun='val')
    return train_ds, val_ds

# Inhertis from Pytorch's base Dataset class
# Must override the "magic" methods: __len__ and __getitem__
# Allows us to plug directly into Pytorch's DataLoader for automatic batching, shuffling, and parallel data loading.
class MixedTrainDataset(data.Dataset):
    """Mixed hdlong-complexv1 + PolarPS training dataset.

    Either discovers scenes from `args` (when `scenes is None`) or wraps a
    pre-split scene list handed in by `build_mixed_split`. `__getitem__`
    returns the 4-tuple expected by `train._collate`: (I, N, M, n_imgs).
    """

    def __init__(self, args, augment=True, 
                 scenes: list[tuple[str, str]]|None=None,
                 subset_name='MixedTrain', noun='train'):
        self.args = args
        self.augment = augment
        self.train_resolution = int(args.train_resolution)
        self.mask_margin = getattr(args, 'mask_margin', 8)
        self.outdir = args.session_name
        # Default to max_scenes scenes with respected hdlong-polarps ratio
        self.scenes = scenes if scenes is not None else _discover_scenes(args)
        n_hd = sum(1 for k, _ in self.scenes if k == 'hdlong')
        n_pp = sum(1 for k, _ in self.scenes if k == 'polarps')
        print(f'[{subset_name}] {len(self.scenes):,} {noun} scenes  '
              f'({n_hd:,} hdlong + {n_pp:,} polarps)')

        self._hdlong = HdlongLoader(
            self.train_resolution, outdir=self.outdir,
            mask_margin=self.mask_margin,
        )
        self._polarps = PolarPSLoader(
            self.train_resolution, outdir=self.outdir,
            mask_margin=self.mask_margin,
        )

    def __len__(self):
        return len(self.scenes)

    def _load(self, kind, scene_dir):
        loader = self._hdlong if kind == 'hdlong' else self._polarps
        loader.load(scene_dir, augment=self.augment)
        return loader

    def __getitem__(self, idx):
        kind, scene_dir = self.scenes[idx]
        loader = self._load(kind, scene_dir)
        h, w = loader.h, loader.w
        n = loader.numberOfImages

        # I: (H, W, 3, K) -> pad to (3, h, w, K_PER_SCENE). Padding slots
        # are zero so any scene that fell short of 10 on disk still stacks.
        I = np.zeros((h, w, 3, K_PER_SCENE), np.float32)
        I[..., :n] = loader.I
        I = I.transpose(2, 0, 1, 3)

        N = loader.N.transpose(2, 0, 1).astype(np.float32)         # (3, H, W)
        M = loader.mask.transpose(2, 0, 1).astype(np.float32)      # (1, H, W)
        return I, N, M, np.int64(n)
