"""
Synthetic training dataset loader for SDM-UniPS (normals only).

Expected layout per object directory (suffix configurable, default `.data`):

    OBJECT.data/
        L_*.png         # rendered images under varied lighting
        normal.png      # GT normal map (RGB-encoded normals in [-1, 1] via (n+1)/2)
        mask.png        # optional, foreground mask
"""

import glob
import os
import re
import numpy as np
import cv2


def _read_image(path):
    if path is None or not os.path.isfile(path):
        return None
    img = cv2.imread(path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint8:
        scale = 255.0
    elif img.dtype == np.uint16:
        scale = 65535.0
    else:
        scale = 1.0
    return np.float32(img) / scale


def _decode_normal(rgb01):
    n = 2.0 * rgb01 - 1.0
    return n / (np.sqrt(np.sum(n * n, axis=2, keepdims=True)) + 1e-12)


class SyntheticDataset:
    """In-memory loader for one synthetic training sample.

    Mirrors `realdata.dataloader`'s public surface (fields `I`, `N`, `mask`,
    `roi`, `objname`, `data_workspace`, `h`, `w`, `numberOfImages`).
    """

    def __init__(self, max_image_num, train_resolution=256, outdir='.', mask_margin=8):
        self.max_image_num = max_image_num
        self.train_resolution = train_resolution
        self.outdir = outdir
        self.mask_margin = mask_margin

    def _sample_n_images(self, total_available, min_image_num, rng):
        n_max = min(total_available, self.max_image_num)
        n_min = min_image_num if min_image_num is not None else max(2, n_max // 2)
        n_min = min(n_min, n_max)
        return int(rng.randint(n_min, n_max + 1))

    def _square_bbox(self, mask, rng, augment):
        h0, w0 = mask.shape
        ys, xs = np.nonzero(mask)
        if len(ys) > 0:
            r0, r1 = int(ys.min()), int(ys.max())
            c0, c1 = int(xs.min()), int(xs.max())
        else:
            r0, r1, c0, c1 = 0, h0 - 1, 0, w0 - 1
        side = max(r1 - r0, c1 - c0) + 2 * self.mask_margin
        side = max(side, self.train_resolution)
        cy = (r0 + r1) // 2
        cx = (c0 + c1) // 2
        if augment:
            jit = max(1, side // 16)
            cy += int(rng.randint(-jit, jit + 1))
            cx += int(rng.randint(-jit, jit + 1))
        rs = max(0, cy - side // 2)
        re_ = min(h0, rs + side)
        rs = max(0, re_ - side)
        cs = max(0, cx - side // 2)
        ce_ = min(w0, cs + side)
        cs = max(0, ce_ - side)
        return rs, re_, cs, ce_

    def load(self, objdir, prefix='L*', max_image_resolution=None,
             min_image_num=None, augment=True, rng=None):
        rng = rng if rng is not None else np.random
        self.objname = re.split(r'\\|/', objdir)[-1]
        self.data_workspace = f'{self.outdir}/results/{self.objname}'

        directlist = sorted(p for p in glob.glob(f'{objdir}/{prefix}') if os.path.isfile(p))
        if len(directlist) == 0:
            raise RuntimeError(f'No input images found in {objdir} with prefix {prefix}')

        # GT normal: required.
        nml_path = None
        for cand in ('normal.png', 'Normal_gt.png', 'normal.tif'):
            if os.path.isfile(os.path.join(objdir, cand)):
                nml_path = os.path.join(objdir, cand)
                break
        if nml_path is None:
            raise RuntimeError(f'No GT normal in {objdir} (normal.png / Normal_gt.png / normal.tif)')

        nml_rgb = _read_image(nml_path)
        N = _decode_normal(nml_rgb)
        h0, w0 = N.shape[:2]

        mask_path = os.path.join(objdir, 'mask.png')
        if os.path.isfile(mask_path):
            mask = _read_image(mask_path)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0.5).astype(np.float32)
        else:
            # Derive a mask from the GT normal: encoded background pixels (rgb=0.5)
            # decode to n=0, so the residual magnitude departs from 1.
            n_raw = 2.0 * nml_rgb - 1.0
            mag = np.sqrt(np.sum(n_raw * n_raw, axis=2))
            mask = (np.abs(1.0 - mag) < 0.5).astype(np.float32)

        # Crop to a square around the mask.
        rs, re_, cs, ce_ = self._square_bbox(mask, rng, augment)
        N = N[rs:re_, cs:ce_]
        mask = mask[rs:re_, cs:ce_]

        # Resize to the training resolution.
        h = w = int(self.train_resolution)
        N = cv2.resize(N, (h, w), interpolation=cv2.INTER_CUBIC)
        N = N / (np.linalg.norm(N, axis=2, keepdims=True) + 1e-12)
        mask = (cv2.resize(mask, (h, w), interpolation=cv2.INTER_CUBIC) > 0.5).astype(np.float32)

        # One flip flag, applied uniformly to the normal map + input images.
        do_flip = bool(augment and rng.rand() < 0.5)
        if do_flip:
            N = N[:, ::-1, :].copy()
            N[:, :, 0] = -N[:, :, 0]
            mask = mask[:, ::-1].copy()

        # Load and crop the chosen input images.
        n_imgs = self._sample_n_images(len(directlist), min_image_num, rng)
        chosen = rng.permutation(len(directlist))[:n_imgs]
        I = np.zeros((n_imgs, h, w, 3), np.float32)
        for i, k in enumerate(chosen):
            img = _read_image(directlist[k])
            img = img[rs:re_, cs:ce_]
            img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
            if do_flip:
                img = img[:, ::-1, :].copy()
            I[i] = img

        # Per-image normalization: paper Sec. 3.1 says each image is normalized
        # by a random value between its max and its mean (over masked pixels).
        I_flat = I.reshape(n_imgs, -1, 3)
        m_flat = mask.reshape(-1)
        valid = I_flat[:, m_flat == 1, :] if m_flat.sum() > 0 else I_flat
        per_pixel_lum = np.mean(valid, axis=2)                 # [N, num_valid]
        mx = np.max(per_pixel_lum, axis=1)                     # [N]
        mn = np.mean(per_pixel_lum, axis=1)                    # [N]
        if augment:
            t = rng.rand(n_imgs).astype(np.float32)
            scale = (1.0 - t) * mn + t * mx
        else:
            scale = mx
        I = I_flat / (scale.reshape(-1, 1, 1) + 1e-6)
        I = I.reshape(n_imgs, h, w, 3)

        # Final tensor layout (h, w, 3, N) — same as realdata.
        I = np.transpose(I, (1, 2, 3, 0))

        self.h = h
        self.w = w
        self.I = I
        self.N = N.astype(np.float32)
        self.mask = mask.reshape(h, w, 1).astype(np.float32)
        self.roi = np.array([h, w, 0, h, 0, w])
        self.numberOfImages = n_imgs
