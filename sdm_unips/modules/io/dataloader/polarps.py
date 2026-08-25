"""
PolarPS dataset loader (per the project's `dataset/framework.ipynb`).

Layout assumption (one scene per directory under the dataset root):

    SCENE/
        normal.exr
        <material-mix sub-dir>/
            light-01/S0.exr
            light-02/S0.exr
            ...
            light-32/S0.exr

The mask is derived from `normal.exr`: encoded as (n+1)/2, so background
pixels (rgb=0.5) decode to the zero vector and have magnitude 0. We
threshold magnitude > 0.5 to isolate foreground.

For one __getitem__ call we draw a fixed K=10 light directions out of
the 32, mask each image, resize to --train_resolution if needed,
optionally flip horizontally and/or vertically (training only), and divide
each image by a per-image scalar drawn from its (cached) foreground
U[mean, max] intensity when training (paper Sec. 3.1) and pinned to the max
otherwise, matching the inference-time normalization in `realdata.py`.
"""

import os
import re

import cv2
import numpy as np

from .scale_cache import (masked_scale_stats, read_scale_stats,
                          resolve_scales, write_scale_stats)
from .scene_check import usable_light_dirs

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


def _read_exr(path):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise IOError(f'Could not read EXR: {path}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    # Rendered HDR EXRs can carry Inf/NaN (clipped specular highlights, failed
    # samples). Left in, a single Inf pixel poisons the per-image max-luminance
    # normalization (Inf/Inf -> NaN) and produces NaN losses. Drop them to 0.
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)


class PolarPSLoader:
    """Per-scene loader for PolarPS.

    Public attributes:
        I (h, w, 3, K), N (h, w, 3), mask (h, w, 1), roi, objname,
        data_workspace, h, w, numberOfImages.
    """

    def __init__(self, train_resolution=512, outdir='.', k=10):
        self.train_resolution = int(train_resolution)
        self.outdir = outdir
        self.k = max(1, int(k))
        # Per-scene {rel_path: (foreground_mean, foreground_max)} caches,
        # memoized so the sidecar is read from disk at most once per scene per
        # worker. Only the statistics are cached; the scale drawn from them is
        # redrawn every epoch.
        self._stats_by_dir = {}

    def _get_stats(self, cache_dir: str) -> dict[str, tuple[float, float]]:
        stats = self._stats_by_dir.get(cache_dir)
        if stats is None:
            stats = read_scale_stats(cache_dir)
            self._stats_by_dir[cache_dir] = stats
        return stats

    @staticmethod
    def _is_same_resolution(img: np.ndarray, expected: tuple[int, int]=(512, 512)):
        if len(img.shape) < 2: raise ValueError("Input array must have > 1 dimension!")

        if (img.shape[0] == expected[0]) and (img.shape[1] == expected[1]):
            return True
        return False

    def _sample_K(self, total_available):
        # K renders per scene (capped by what's actually on disk; startup
        # validation guarantees >= K, so the cap is defensive only).
        return min(self.k, total_available)

    def load(self, scene_dir, augment=True, rng=None):
        rng = rng if rng is not None else np.random
        # Resolve data paths
        self.objname = re.split(r'\\|/', scene_dir)[-1]
        self.data_workspace = f'{self.outdir}/results/{self.objname}'

        nml_path = os.path.join(scene_dir, 'normal.exr')
        if not os.path.isfile(nml_path):
            raise RuntimeError(f'Missing normal.exr in {scene_dir}')
        # Extract binary mask from surface normal vectors
        n_raw = _read_exr(nml_path)
        n_vec = 2.0 * n_raw - 1.0
        mag = np.linalg.norm(n_vec, axis=2)
        mask = (mag >= 0.5).astype(np.float32)
        # Normalization
        N = (n_vec * mask[..., None]) / (mag[..., None] + 1e-12)

        # Randomly sample K images/scene, drawing ONLY from light dirs that
        # actually hold an S0.exr. `usable_light_dirs` is the same helper the
        # startup validator uses, so "this scene passed validation" and "every
        # draw here succeeds" cannot drift apart. Without the filter, a scene
        # with 31 of 32 lights intact would pass validation and still fail
        # roughly one read in 32.
        img_subdir, light_dirs = usable_light_dirs(scene_dir)
        if img_subdir is None:
            raise RuntimeError(f'No image sub-directory in {scene_dir}')
        if not light_dirs:
            raise RuntimeError(f'No light-*/S0.exr under {img_subdir}')

        K = self._sample_K(len(light_dirs))
        chosen = rng.permutation(len(light_dirs))[:K]

        out_h = out_w = self.train_resolution

        # Upsample surface normals and masks (if needed) + normalize surface normals
        mask_r = mask
        if not self._is_same_resolution(mask):
            mask_r = (cv2.resize(mask_r, (out_h, out_w), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.float32)

        N_r = N
        if not self._is_same_resolution(N_r):
            N_r = cv2.resize(N_r, (out_h, out_w), interpolation=cv2.INTER_LINEAR)
            N_r = (N_r * mask_r[..., None]) / (np.linalg.norm(N_r, axis=2, keepdims=True) + 1e-12)

        composed = np.zeros((K, out_h, out_w, 3), np.float32)
        stats = self._get_stats(scene_dir)
        new_entries = {}
        # Per-image foreground intensity statistics. The scale actually divided
        # through is drawn from them below, once the flips are settled.
        mn_stat = np.zeros(K, np.float32)
        mx_stat = np.ones(K, np.float32)
        for i, k in enumerate(chosen):
            img_path = os.path.join(light_dirs[k], 'S0.exr')
            img = _read_exr(img_path)
            # Apply mask
            img *= mask[..., None]
            # Per-image foreground (mean, max), computed once per file then
            # cached. Taken at the ON-DISK resolution (before any resize below)
            # so the cached values stay valid across --train_resolution
            # settings.
            rel = os.path.relpath(img_path, scene_dir)
            if rel in stats:
                mn_stat[i], mx_stat[i] = stats[rel]
            else:
                mn_stat[i], mx_stat[i] = masked_scale_stats(img, mask)
                stats[rel] = new_entries[rel] = (mn_stat[i], mx_stat[i])
            # Upsample to the desired shape (512, 512) if needed
            if not self._is_same_resolution(img):
                img = cv2.resize(img, (out_h, out_w), interpolation=cv2.INTER_CUBIC)
                img *= mask_r[..., None]
            img = np.maximum(img, 0.0)
            composed[i] = img # (K, H, W, 3)
        if new_entries:
            write_scale_stats(scene_dir, stats)


        # Flips are label-preserving symmetries here: reflecting the image
        # plane about an axis negates the normal's in-plane component along
        # that axis and leaves z untouched.
        do_flip_h = bool(augment and rng.rand() < 0.5)
        do_flip_v = bool(augment and rng.rand() < 0.5)
        if do_flip_h:
            composed = composed[:, :, ::-1, :]
            mask_r = mask_r[:, ::-1]
            N_r = N_r[:, ::-1, :]
        if do_flip_v:
            composed = composed[:, ::-1, :, :]
            mask_r = mask_r[::-1, :]
            N_r = N_r[::-1, :, :]
        if do_flip_h or do_flip_v:
            composed = np.ascontiguousarray(composed)
            mask_r = np.ascontiguousarray(mask_r)
            N_r = np.ascontiguousarray(N_r)
            if do_flip_h:
                N_r[..., 0] = -N_r[..., 0]
            if do_flip_v:
                N_r[..., 1] = -N_r[..., 1]

        # Per-image normalization: divide each image by a scalar drawn from
        # U[mean, max] of its foreground intensity while training (paper
        # Sec. 3.1) and equal to the max otherwise.
        scale = resolve_scales(mn_stat, mx_stat, augment, rng)
        I_flat = composed.reshape(K, -1, 3) # (K, H * W, 3)
        I_flat = I_flat / (scale.reshape(-1, 1, 1) + 1e-6)
        composed = I_flat.reshape(K, out_h, out_w, 3)

        I = np.transpose(composed, (1, 2, 3, 0)) # (H, W, 3, K)

        self.h = out_h
        self.w = out_w
        self.I = I.astype(np.float32)
        self.N = N_r.astype(np.float32)
        self.mask = mask_r.reshape(out_h, out_w, 1).astype(np.float32)
        self.numberOfImages = K
        self.roi = np.array([out_h, out_w, 0, out_h, 0, out_w])
