"""
hdlong-complexv1 dataset loader (per the project's `dataset/framework.ipynb`).

Layout assumption (one scene per directory under the dataset root):

    SCENE/
        cams.config, dir_lights.config, env_lights.config, point_lights.config,
        light_means.config, object.config
        cam_0000N/
            binary_mask.exr
            depth_map.exr
            local_normal.exr          # used as GT (object-space normal)
            global_normal.exr
            point_light_000NN.exr     # 20 per camera
            dir_light_000NN.exr       # 20 per camera
            env_light_000NN.exr       # 10 per camera

For one __getitem__ call we:
    1. Pick one camera at random.
    2. Build a fixed K=10 composite renders. Each render is a Dirichlet
       (alpha,beta,gamma) mix of one randomly chosen point, directional,
       and environment light image, each pre-normalized by its per-scene
       `light_means.config` value and masked.
    3. Upsample 256x256 -> --train_resolution (default 512).
    4. Optionally flip horizontally and/or vertically (training only).
    5. Divide each composite by a per-image scalar drawn from U[mean, max] of
       its foreground intensity when training (paper Sec. 3.1) and pinned to
       the max otherwise, matching the inference-time normalization in
       `realdata.py`.

"""

import glob
import os
import re

import cv2
import numpy as np

from .scale_cache import masked_scale_stats, resolve_scales
from .scene_check import usable_cam_dirs

# Required for OpenCV EXR support.
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


def _read_exr(path: str):
    # cv2.IMREAD_ANYCOLOR: Read the image using its own color format without forcing it to cv2.IMREAD_GRAYSCALE for example
    # cv2.IMREAD_ANYDEPTH: Preserve the image's bit depth (float32 for EXR)
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) # np.ndarray (256. 256. 3)
    if img is None:
        raise IOError(f'Could not read EXR: {path}')
    # By default, OpenCV stores color images in BGR format. We convert it back to RGB.
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    # Rendered HDR EXRs can carry Inf/NaN. Drop them to 0.
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)


def _read_light_means(path):
    means = {}
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                try:
                    means[tokens[0]] = float(tokens[1])
                except ValueError:
                    continue
    return means


class HdlongLoader:
    """Per-scene loader for hdlong-complexv1.

    After `load()` the loader exposes:
        I: (H, W, 3, K), N: (H, W, 3), Mask: (h, w, 1), Region of Interest (roi),
        objname, data_workspace, h, w, numberOfImages

    Region Of Interest is [canvas_h, canvas_w, row_min, row_max, col_min, col_max]
        - (canvas_h, canvas_w): The full frame the prediction eventually lives in
        - (row_min, row_max, col_min, col_max: The bounding box fed to the network
    """

    def __init__(self, train_resolution=512, outdir='.', k=10):
        self.train_resolution = int(train_resolution)
        self.outdir = outdir
        self.k = max(1, int(k))

    def _sample_K(self, total_available):
        # K rendered images per scene (capped by what's on disk; startup
        # validation guarantees >= K, so the cap is defensive only).
        return min(self.k, total_available)

    def load(self, scene_dir, augment=True, rng=None):
        rng = rng if rng is not None else np.random

        # Resolve data paths
        self.objname = re.split(r'\\|/', scene_dir)[-1]
        self.data_workspace = f'{self.outdir}/results/{self.objname}'

        # For each scene, randomly choose one camera -- but only among cameras
        # that are actually complete (mask + normal + >= K each of
        # point/dir/env). `usable_cam_dirs` is the same helper the startup
        # validator uses, so "this scene passed validation" and "every draw
        # here succeeds" cannot drift apart. Drawing from ALL cam_* dirs would
        # make a scene with 1 good camera out of 3 fail two reads in three,
        # despite validating cleanly.
        cam_dirs = usable_cam_dirs(scene_dir, self.k)
        if not cam_dirs:
            raise RuntimeError(
                f'No usable cam_* subdirectory in {scene_dir} '
                f'(need binary_mask.exr + local_normal.exr and >= {self.k} '
                f'each of point/dir/env images)')
        cam_dir = cam_dirs[rng.randint(0, len(cam_dirs))]

        # Extract different lighting condition means
        cfg_path = os.path.join(scene_dir, 'light_means.config')
        if not os.path.isfile(cfg_path):
            raise RuntimeError(f'Missing light_means.config in {scene_dir}')
        
        means = _read_light_means(cfg_path)
        point_mean = means.get('point_mean', 1.0)
        dir_mean = means.get('dir_mean', 1.0)
        env_mean = means.get('env_mean', 1.0)

        # Process anti-aliased mask
        mask_soft = _read_exr(os.path.join(cam_dir, 'binary_mask.exr')) # (256, 256)
        # If the mask has 3 channels, take the first channel only
        if mask_soft.ndim == 3:
            mask_soft = mask_soft[..., 0]
        mask = (mask_soft >= 0.5).astype(np.float32)
        if mask.sum() == 0:
            raise ValueError(f'{cam_dir}: empty mask — per-image scale is undefined.')

        # Normalize surface normal vectors
        N_raw = _read_exr(os.path.join(cam_dir, 'local_normal.exr'))
        N = 2.0 * N_raw - 1.0
        N = N * mask[..., None]

        # Generate K rendered images/scene
        point_paths = sorted(glob.glob(os.path.join(cam_dir, 'point_light_*.exr')))
        dir_paths = sorted(glob.glob(os.path.join(cam_dir, 'dir_light_*.exr')))
        env_paths = sorted(glob.glob(os.path.join(cam_dir, 'env_light_*.exr')))
        if not (point_paths and dir_paths and env_paths):
            raise RuntimeError(f'Missing point/dir/env light images in {cam_dir}')

        K = self._sample_K(min(len(point_paths), len(dir_paths), len(env_paths)))

        out_h = out_w = self.train_resolution
        composed = np.zeros((K, out_h, out_w, 3), np.float32)

        # For correct broadcasting
        # Upsample the anti-aliased mask using LINEAR INTERPOLATION
        mask_soft_r = cv2.resize(mask_soft, (out_h, out_w), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mask_r = (mask_soft_r >= 0.5).astype(np.float32)

        # Per-composite foreground intensity statistics. The scale actually
        # divided through is drawn from them below, once the flip is settled.
        mn_stat = np.zeros(K, np.float32)
        mx_stat = np.ones(K, np.float32)

        for k in range(K):
            # Randomly sample
            pi = rng.randint(0, len(point_paths))
            di = rng.randint(0, len(dir_paths))
            ei = rng.randint(0, len(env_paths))
            p_img = _read_exr(point_paths[pi]) / (point_mean + 1e-8)
            d_img = _read_exr(dir_paths[di]) / (dir_mean + 1e-8)
            e_img = _read_exr(env_paths[ei]) / (env_mean + 1e-8)
            # Apply mask: (256, 256, 3) * (256, 256, 1)
            p_img *= mask[..., None]
            d_img *= mask[..., None]
            e_img *= mask[..., None]
            # Rendered image = a convex combination of three component images
            w = rng.dirichlet(np.ones(3))
            img = (w[0] * p_img + w[1] * d_img + w[2] * e_img).astype(np.float32)
            # Foreground (mean, max) of THIS composite.
            # Taken pre-resize (256x256, the native render size) 
            # so the values do not depend on--train_resolution or on cubic overshoot.
            mn_stat[k], mx_stat[k] = masked_scale_stats(img, mask)
            # hdlong-complexv1's images have spatial size (256, 256), need upsampling
            if img.shape[0] != out_h or img.shape[1] != out_w:
                img = cv2.resize(img, (out_h, out_w), interpolation=cv2.INTER_CUBIC)
                # Apply the mask once again
                img *= mask_r[..., None]
            img = np.maximum(img, 0.0)
            composed[k] = img # composed: (K, H, W, 3)

        # Upsample + Normalize surface normals
        N_r = cv2.resize(N, (out_h, out_w), interpolation=cv2.INTER_LINEAR)
        N_r = N_r * mask_r[..., None]
        N_r = N_r / (np.linalg.norm(N_r, axis=2, keepdims=True) + 1e-12)

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

        # Per-image normalization: divide each composite by a scalar drawn
        # from U[mean, max] of its foreground intensity while training (paper
        # Sec. 3.1) and equal to the max otherwise
        scale = resolve_scales(mn_stat, mx_stat, augment, rng) # (K,)
        I_flat = composed.reshape(K, -1, 3) # (K, H * W, 3)
        I_flat = I_flat / (scale.reshape(-1, 1, 1) + 1e-6)
        composed = I_flat.reshape(K, out_h, out_w, 3)

        I = np.transpose(composed, (1, 2, 3, 0))  # (H, W, 3, K)

        self.h = out_h
        self.w = out_w
        self.I = I.astype(np.float32)
        self.N = N_r.astype(np.float32)
        self.mask = mask_r.reshape(out_h, out_w, 1).astype(np.float32)
        self.numberOfImages = K
        self.roi = np.array([out_h, out_w, 0, out_h, 0, out_w])
