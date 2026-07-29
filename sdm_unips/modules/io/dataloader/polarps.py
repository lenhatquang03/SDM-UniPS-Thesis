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
the 32, mask each image, resize to --train_resolution if needed, and
apply the paper-spec per-image (mean..max) luminance normalization.
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

    def __init__(self, train_resolution=512, outdir='.', mask_margin=8):
        self.train_resolution = int(train_resolution)
        self.outdir = outdir
        self.mask_margin = mask_margin

    def _is_same_resolution(img: np.ndarray, expected: tuple[int, int]=(512, 512)):
        if len(img.shape) < 2: raise ValueError("Input array must have > 1 dimension!")

        if (img.shape[0] == expected[0]) and (img.shape[1] == expected[1]):
            return True
        return False

    def _sample_K(self, total_available):
        # Fixed K=10 renders per scene (capped by what's actually on disk).
        return min(10, total_available)

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
        mask = (mag > 0.5).astype(np.float32)
        # Normalization
        N = (n_vec * mask[..., None]) / (mag[..., None] + 1e-12)

        # Randomply sample 10 images/scene
        # There should only be 1 sub-directory
        subdirs = sorted(d for d in glob.glob(os.path.join(scene_dir, '*'))
                         if os.path.isdir(d))
        if not subdirs:
            raise RuntimeError(f'No image sub-directory in {scene_dir}')
        img_subdir = subdirs[0]
        light_dirs = sorted(d for d in glob.glob(os.path.join(img_subdir, 'light-*'))
                            if os.path.isdir(d))
        if not light_dirs:
            raise RuntimeError(f'No light-* dirs under {img_subdir}')

        K = self._sample_K(len(light_dirs))
        chosen = rng.permutation(len(light_dirs))[:K]

        out_h = out_w = self.train_resolution
        composed = np.zeros((K, out_h, out_w, 3), np.float32)
        for i, k in enumerate(chosen):
            img = _read_exr(os.path.join(light_dirs[k], 'S0.exr'))
            # Apply mask
            img = np.maximum(img, 0.0) * mask[..., None]
            # Upsample to the desired shape (512, 512) if needed
            if not self._is_same_resolution(img):
                img = cv2.resize(img, (out_h, out_w), interpolation=cv2.INTER_CUBIC)
            composed[i] = img # (K, H, W, 3)

        # Upsample surface normals and masks (if needed) + normalize surface normals
        mask_r = mask
        if not self._is_same_resolution(mask_r):
            mask_r = (cv2.resize(mask_r, (out_h, out_w), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.float32)

        N_r = N
        if not self._is_same_resolution(N_r):
            N_r = cv2.resize(N_r, (out_h, out_w), interpolation=cv2.INTER_LINEAR)
            N_r = (N_r * mask_r[..., None]) / (np.linalg.norm(N_r, axis=2, keepdims=True) + 1e-12)

        do_flip = bool(augment and rng.rand() < 0.5)
        if do_flip:
            composed = composed[:, :, ::-1, :].copy()
            mask_r = mask_r[:, ::-1].copy()
            N_r = N_r[:, ::-1, :].copy()
            N_r[:, :, 0] = -N_r[:, :, 0]

        # Per-image normalization
        m_flat = mask_r.reshape(-1) 
        I_flat = composed.reshape(K, -1, 3) # (K, H * W, 3)
        if m_flat.sum() > 0:
            valid = I_flat[:, m_flat == 1, :] # (K, V, 3); V = # foreground pixels
        else:
            valid = I_flat
        lum = np.mean(valid, axis=2) # (K, V)
        mx = np.percentile(lum, 99.5, axis=1).astype(np.float32) # (K,)
        # Max-scaled for both train and val sets.
        scale = mx
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
