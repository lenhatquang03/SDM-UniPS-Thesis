"""
DiLiGenT evaluation dataset.

Layout assumption (one scene per `*PNG` directory under
`<root>/pmsData/`):

    <objPNG>/
        filenames.txt           # one image filename per line, 96 lines
        light_intensities.txt   # one '<r> <g> <b>' triple per line, 96 lines
        light_directions.txt    # not used by Model A (SDM-UniPS is uncalibrated)
        mask.png
        Normal_gt.png           # uint8, encoded as (n+1)/2
        001.png .. 096.png      # uint16 16-bit observations

Each image is divided by its per-channel light intensity (the standard
DiLiGenT pre-processing). Everything is center-cropped to a square of
side `side` (default 512, since DiLiGenT renders are 612x512).

Use `DiligentLoader.load(scene_dir)` once per scene, then call
`sample(K, rng)` repeatedly to draw K-image subsets for the K-sweep
evaluation described in the thesis spec.
"""

import os
import re

import cv2
import numpy as np


def _read_image(path):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise IOError(f'Could not read image: {path}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint8:
        scale = 255.0
    elif img.dtype == np.uint16:
        scale = 65535.0
    else:
        scale = 1.0
    return img.astype(np.float32) / scale


def _center_crop_square(img, side):
    h, w = img.shape[:2]
    r0 = (h - side) // 2
    c0 = (w - side) // 2
    if img.ndim == 2:
        return img[r0:r0 + side, c0:c0 + side]
    return img[r0:r0 + side, c0:c0 + side, :]


class DiligentLoader:
    """Per-scene DiLiGenT loader.

    Public attributes after `load()`:
        objname, h, w, total_images, N (h, w, 3), mask (h, w),
        fnames, intens (N, 3).
    """

    def __init__(self, side=512):
        self.side = int(side)

    def load(self, scene_dir):
        self.objname = re.split(r'\\|/', scene_dir)[-1]

        with open(os.path.join(scene_dir, 'filenames.txt'), 'r') as f:
            fnames = [l.strip() for l in f if l.strip()]

        intens = np.loadtxt(os.path.join(scene_dir, 'light_intensities.txt'),
                            dtype=np.float32)
        if intens.ndim == 1:
            intens = intens.reshape(-1, 3)

        mask = _read_image(os.path.join(scene_dir, 'mask.png'))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0.5).astype(np.float32)

        N = _read_image(os.path.join(scene_dir, 'Normal_gt.png'))
        N = 2.0 * N - 1.0
        nrm = np.linalg.norm(N, axis=2, keepdims=True)
        N = N / (nrm + 1e-12)
        N = N * mask[..., None]

        side = self.side
        self.mask = _center_crop_square(mask, side).astype(np.float32)
        self.N = _center_crop_square(N, side).astype(np.float32)
        self.h = self.w = side
        self.fnames = [os.path.join(scene_dir, fn) for fn in fnames]
        self.intens = intens
        self.total_images = len(self.fnames)

    def sample(self, K, rng=None):
        """Return (I, N, mask, K_used) for one random subset of K images.

        I is layout (h, w, 3, K) so it matches the train pipeline.
        """
        rng = rng if rng is not None else np.random
        K = min(int(K), self.total_images)
        idx = rng.permutation(self.total_images)[:K]

        side = self.side
        I = np.zeros((side, side, 3, K), np.float32)
        for j, k in enumerate(idx):
            img = _read_image(self.fnames[k])
            img = _center_crop_square(img, side)
            img = img / (self.intens[k].reshape(1, 1, 3) + 1e-6)
            img = img * self.mask[..., None]
            I[:, :, :, j] = img.astype(np.float32)

        # Per-image normalization (deterministic at eval: divide by max luminance).
        flat = I.transpose(3, 0, 1, 2).reshape(K, -1, 3)
        m_flat = self.mask.reshape(-1)
        if m_flat.sum() > 0:
            lum = np.mean(flat[:, m_flat == 1, :], axis=2)
        else:
            lum = np.mean(flat, axis=2)
        mx = np.max(lum, axis=1)
        flat = flat / (mx.reshape(-1, 1, 1) + 1e-6)
        I = flat.reshape(K, side, side, 3).transpose(1, 2, 3, 0)
        return I.astype(np.float32), self.N, self.mask[..., None].astype(np.float32), K
