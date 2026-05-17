"""
Mixed training-dataset wrapper for hdlong-complexv1 + PolarPS.

The Dataset discovers scene directories under one or more roots, infers
which loader each scene belongs to (hdlong has `light_means.config`,
PolarPS has `normal.exr`), and dispatches to the appropriate per-scene
loader. The output is the same 8-tuple shape as `TrainDataio` so it can
be consumed by `train.py` without any other changes.

Required attrs on `args`:
    - hdlong_dir (optional)
    - polarps_dir (optional)
    - train_dir (optional fallback: auto-detect both kinds under one root)
    - max_image_num, min_image_num, train_resolution, mask_margin
    - session_name, seed
    - max_scenes (optional cap; default unlimited)
"""

import glob
import os

import numpy as np
import torch.utils.data as data

from .hdlong import HdlongLoader
from .polarps import PolarPSLoader


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


class MixedTrainDataset(data.Dataset):
    """Mixed hdlong-complexv1 + PolarPS training dataset.

    __getitem__ returns the 4-tuple expected by `train._collate`:
        (I, N, M, n_imgs)
    """

    def __init__(self, args, augment=True):
        self.args = args
        self.augment = augment
        self.max_image_num = args.max_image_num
        self.min_image_num = getattr(args, 'min_image_num', None)
        self.train_resolution = int(args.train_resolution)
        self.mask_margin = getattr(args, 'mask_margin', 8)
        self.outdir = args.session_name

        # Discover scenes from the explicit per-backend roots, falling back to
        # auto-detection under --train_dir.
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

        # Optional reduced subset (thesis ablation uses 8000 scenes).
        rng = np.random.RandomState(getattr(args, 'seed', 42))
        max_scenes = getattr(args, 'max_scenes', None)
        if max_scenes is not None and max_scenes > 0 and len(scenes) > max_scenes:
            idx = rng.permutation(len(scenes))[:max_scenes]
            scenes = [scenes[i] for i in idx]

        self.scenes = scenes
        n_hd = sum(1 for k, _ in self.scenes if k == 'hdlong')
        n_pp = sum(1 for k, _ in self.scenes if k == 'polarps')
        print(f'[MixedTrain] {len(self.scenes)} scenes  (hdlong={n_hd}  polarps={n_pp})')

        self._hdlong = HdlongLoader(
            self.max_image_num, self.train_resolution,
            outdir=self.outdir, mask_margin=self.mask_margin,
        )
        self._polarps = PolarPSLoader(
            self.max_image_num, self.train_resolution,
            outdir=self.outdir, mask_margin=self.mask_margin,
        )

    def __len__(self):
        return len(self.scenes)

    def _load(self, kind, scene_dir):
        loader = self._hdlong if kind == 'hdlong' else self._polarps
        loader.load(scene_dir, augment=self.augment,
                    min_image_num=self.min_image_num)
        return loader

    def __getitem__(self, idx):
        kind, scene_dir = self.scenes[idx]
        loader = self._load(kind, scene_dir)
        h, w = loader.h, loader.w
        n = loader.numberOfImages

        # I: (h, w, 3, K) -> pad -> (3, h, w, Nmax)
        I = np.zeros((h, w, 3, self.max_image_num), np.float32)
        I[..., :n] = loader.I
        I = I.transpose(2, 0, 1, 3)

        N = loader.N.transpose(2, 0, 1).astype(np.float32)         # (3, h, w)
        M = loader.mask.transpose(2, 0, 1).astype(np.float32)      # (1, h, w)
        return I, N, M, np.int64(n)
