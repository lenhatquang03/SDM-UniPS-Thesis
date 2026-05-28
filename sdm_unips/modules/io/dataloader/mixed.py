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

import glob
import os

import numpy as np
import torch.utils.data as data

from .hdlong import HdlongLoader
from .polarps import PolarPSLoader

K_PER_SCENE = 10


def _list_dir(root):
    if not root or not os.path.isdir(root):
        return []
    return sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))


def _is_hdlong_scene(path):
    return os.path.isfile(os.path.join(path, 'light_means.config'))


def _is_polarps_scene(path):
    return os.path.isfile(os.path.join(path, 'normal.exr'))


def _discover(roots, kinds):
    """Return list of (kind, scene_dir) for matching roots."""
    out = []
    for kind, root in zip(kinds, roots):
        for d in _list_dir(root):
            if kind == 'hdlong' and _is_hdlong_scene(d):
                out.append(('hdlong', d))
            elif kind == 'polarps' and _is_polarps_scene(d):
                out.append(('polarps', d))
    return out


def _auto_discover(root):
    out = []
    for d in _list_dir(root):
        if _is_hdlong_scene(d):
            out.append(('hdlong', d))
        elif _is_polarps_scene(d):
            out.append(('polarps', d))
    return out


def _discover_scenes(args):
    """Discover and (optionally) cap the combined hdlong + PolarPS scene pool.

    The cap (`--max_scenes`) is applied to the combined pool here, before any
    train/val split, so the split halves are carved out of the capped pool.
    """
    scenes = []
    scenes += _discover([getattr(args, 'hdlong_dir', None)], ['hdlong'])
    scenes += _discover([getattr(args, 'polarps_dir', None)], ['polarps'])
    if not scenes and getattr(args, 'train_dir', None):
        scenes = _auto_discover(args.train_dir)

    if len(scenes) == 0:
        raise RuntimeError(
            'MixedTrainDataset found no scenes. Pass --hdlong_dir and/or '
            '--polarps_dir, or place scenes under --train_dir.'
        )

    max_scenes = getattr(args, 'max_scenes', None)
    if max_scenes is not None and max_scenes > 0 and len(scenes) > max_scenes:
        rng = np.random.RandomState(getattr(args, 'seed', 42))
        idx = rng.permutation(len(scenes))[:max_scenes]
        scenes = [scenes[i] for i in idx]
    return scenes


def build_mixed_split(args, augment=True):
    """Deterministic scene-level train/val split of the mixed pool.

    The pool is split proportionally *within* each source (hdlong split
    separately from PolarPS) so the mix ratio is preserved in both halves.
    The split is fully determined by `--seed`, so calling this twice with the
    same args yields identical, disjoint train/val scene lists.

    Returns (train_dataset, val_dataset); the val set never augments.
    """
    scenes = _discover_scenes(args)
    val_fraction = float(getattr(args, 'val_fraction', 0.1))
    seed = int(getattr(args, 'seed', 42))

    hd = [s for s in scenes if s[0] == 'hdlong']
    pp = [s for s in scenes if s[0] == 'polarps']

    def _split(items, rng):
        if not items:
            return [], []
        perm = rng.permutation(len(items))
        n_val = int(round(len(items) * val_fraction))
        n_val = min(max(n_val, 0), len(items))
        val_pos = set(perm[:n_val].tolist())
        train = [items[i] for i in range(len(items)) if i not in val_pos]
        val = [items[i] for i in range(len(items)) if i in val_pos]
        return train, val

    hd_train, hd_val = _split(hd, np.random.RandomState(seed + 1))
    pp_train, pp_val = _split(pp, np.random.RandomState(seed + 2))

    # Ensure the ratio between hdlong-complexv1 and PolarPS scenes are the same
    train_ds = MixedTrainDataset(args, augment=augment,
                                 scenes=hd_train + pp_train,
                                 subset_name='MixedTrain', noun='train')
    val_ds = MixedTrainDataset(args, augment=False,
                               scenes=hd_val + pp_val,
                               subset_name='MixedVal', noun='val')
    return train_ds, val_ds


class MixedTrainDataset(data.Dataset):
    """Mixed hdlong-complexv1 + PolarPS training dataset.

    Either discovers scenes from `args` (when `scenes is None`) or wraps a
    pre-split scene list handed in by `build_mixed_split`. `__getitem__`
    returns the 4-tuple expected by `train._collate`: (I, N, M, n_imgs).
    """

    def __init__(self, args, augment=True, scenes=None,
                 subset_name='MixedTrain', noun='train'):
        self.args = args
        self.augment = augment
        self.train_resolution = int(args.train_resolution)
        self.mask_margin = getattr(args, 'mask_margin', 8)
        self.outdir = args.session_name

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
