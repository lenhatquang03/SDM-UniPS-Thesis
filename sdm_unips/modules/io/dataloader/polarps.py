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

For one __getitem__ call we sample K (default 3..6) light directions
out of the 32, mask each image, resize to --train_resolution if needed,
and apply the paper-spec per-image (mean..max) luminance normalization.
"""

import glob
import os
import re

import cv2
import numpy as np

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


def _read_exr(path):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise IOError(f'Could not read EXR: {path}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


class PolarPSLoader:
    """Per-scene loader for PolarPS.

    Public attributes mirror `synthetic.SyntheticDataset`:
        I (h, w, 3, K), N (h, w, 3), mask (h, w, 1), roi, objname,
        data_workspace, h, w, numberOfImages.
    """

    def __init__(self, max_image_num, train_resolution=512, outdir='.', mask_margin=8):
        self.max_image_num = max_image_num
        self.train_resolution = int(train_resolution)
        self.outdir = outdir
        self.mask_margin = mask_margin

    def _sample_K(self, total_available, min_image_num, rng):
        n_max = min(total_available, self.max_image_num)
        n_min = min_image_num if min_image_num is not None else max(2, n_max // 2)
        n_min = min(n_min, n_max)
        return int(rng.randint(n_min, n_max + 1))

    def load(self, scene_dir, augment=True, min_image_num=None, rng=None):
        rng = rng if rng is not None else np.random
        self.objname = re.split(r'\\|/', scene_dir)[-1]
        self.data_workspace = f'{self.outdir}/results/{self.objname}'

        nml_path = os.path.join(scene_dir, 'normal.exr')
        if not os.path.isfile(nml_path):
            raise RuntimeError(f'Missing normal.exr in {scene_dir}')
        n_raw = _read_exr(nml_path)
        n_vec = 2.0 * n_raw - 1.0
        mag = np.linalg.norm(n_vec, axis=2)
        mask = (mag > 0.5).astype(np.float32)
        N = n_vec / (mag[..., None] + 1e-12)
        N = N * mask[..., None]

        subdirs = sorted(d for d in glob.glob(os.path.join(scene_dir, '*'))
                         if os.path.isdir(d))
        if not subdirs:
            raise RuntimeError(f'No image sub-directory in {scene_dir}')
        img_subdir = subdirs[0]
        light_dirs = sorted(d for d in glob.glob(os.path.join(img_subdir, 'light-*'))
                            if os.path.isdir(d))
        if not light_dirs:
            raise RuntimeError(f'No light-* dirs under {img_subdir}')

        K = self._sample_K(len(light_dirs), min_image_num, rng)
        chosen = rng.permutation(len(light_dirs))[:K]

        out_h = out_w = self.train_resolution
        composed = np.zeros((K, out_h, out_w, 3), np.float32)
        mask_b = mask[..., None]
        for i, k in enumerate(chosen):
            img = _read_exr(os.path.join(light_dirs[k], 'S0.exr'))
            img = np.maximum(img, 0.0) * mask_b
            if img.shape[0] != out_h or img.shape[1] != out_w:
                img = cv2.resize(img, (out_h, out_w), interpolation=cv2.INTER_CUBIC)
            composed[i] = img

        mask_r = (cv2.resize(mask, (out_h, out_w), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.float32)
        N_r = cv2.resize(N, (out_h, out_w), interpolation=cv2.INTER_CUBIC)
        N_r = N_r / (np.linalg.norm(N_r, axis=2, keepdims=True) + 1e-12)
        N_r = N_r * mask_r[..., None]

        do_flip = bool(augment and rng.rand() < 0.5)
        if do_flip:
            composed = composed[:, :, ::-1, :].copy()
            mask_r = mask_r[:, ::-1].copy()
            N_r = N_r[:, ::-1, :].copy()
            N_r[:, :, 0] = -N_r[:, :, 0]

        m_flat = mask_r.reshape(-1)
        I_flat = composed.reshape(K, -1, 3)
        if m_flat.sum() > 0:
            valid = I_flat[:, m_flat == 1, :]
        else:
            valid = I_flat
        lum = np.mean(valid, axis=2)
        mx = np.max(lum, axis=1)
        mn = np.mean(lum, axis=1)
        if augment:
            t = rng.rand(K).astype(np.float32)
            scale = (1.0 - t) * mn + t * mx
        else:
            scale = mx
        I_flat = I_flat / (scale.reshape(-1, 1, 1) + 1e-6)
        composed = I_flat.reshape(K, out_h, out_w, 3)

        I = np.transpose(composed, (1, 2, 3, 0))

        self.h = out_h
        self.w = out_w
        self.I = I.astype(np.float32)
        self.N = N_r.astype(np.float32)
        self.mask = mask_r.reshape(out_h, out_w, 1).astype(np.float32)
        self.numberOfImages = K
        self.roi = np.array([out_h, out_w, 0, out_h, 0, out_w])
