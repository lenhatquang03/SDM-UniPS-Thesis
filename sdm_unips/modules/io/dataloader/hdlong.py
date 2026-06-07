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
    4. Apply the paper-spec per-image (mean..max) luminance normalization.
"""

import glob
import os
import re

import cv2
import numpy as np

# Required for OpenCV EXR support.
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
    """

    def __init__(self, train_resolution=512, outdir='.', mask_margin=8):
        self.train_resolution = int(train_resolution)
        self.outdir = outdir
        self.mask_margin = mask_margin

    def _sample_K(self, total_available):
        # Fixed 10 rendered images per scene (capped by what's actually on disk).
        return min(10, total_available)

    def load(self, scene_dir, augment=True, rng=None):
        rng = rng if rng is not None else np.random

        # Resolve data paths
        self.objname = re.split(r'\\|/', scene_dir)[-1]
        self.data_workspace = f'{self.outdir}/results/{self.objname}'

        # For each scene, randomly choose one camera
        cam_dirs = sorted(d for d in glob.glob(os.path.join(scene_dir, 'cam_*'))
                          if os.path.isdir(d))
        if not cam_dirs:
            raise RuntimeError(f'No cam_* subdirectories in {scene_dir}')
        cam_dir = cam_dirs[rng.randint(0, len(cam_dirs))]

        cfg_path = os.path.join(scene_dir, 'light_means.config')
        if not os.path.isfile(cfg_path):
            raise RuntimeError(f'Missing light_means.config in {scene_dir}')
        
        # Different lighting condition means
        means = _read_light_means(cfg_path)
        point_mean = means.get('point_mean', 1.0)
        dir_mean = means.get('dir_mean', 1.0)
        env_mean = means.get('env_mean', 1.0)

        # Process binary mask
        mask = _read_exr(os.path.join(cam_dir, 'binary_mask.exr'))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask >= 0.5).astype(np.float32)

        # Normalize surface normal vectors
        N_raw = _read_exr(os.path.join(cam_dir, 'local_normal.exr'))
        N = 2.0 * N_raw - 1.0
        N = N / (np.linalg.norm(N, axis=2, keepdims=True) + 1e-12)
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
        mask_b = mask[..., None]

        for k in range(K):
            # Randomly sample 
            pi = rng.randint(0, len(point_paths))
            di = rng.randint(0, len(dir_paths))
            ei = rng.randint(0, len(env_paths))
            p_img = _read_exr(point_paths[pi]) / (point_mean + 1e-8)
            d_img = _read_exr(dir_paths[di]) / (dir_mean + 1e-8)
            e_img = _read_exr(env_paths[ei]) / (env_mean + 1e-8)
            # Apply mask
            p_img *= mask_b
            d_img *= mask_b
            e_img *= mask_b
            # Rendered image = a convex combination of three component images
            w = rng.dirichlet(np.ones(3))
            img = (w[0] * p_img + w[1] * d_img + w[2] * e_img).astype(np.float32)
            img = np.maximum(img, 0.0)
            # hdlong-complexv1's images have spatial size (256, 256), need upsampling
            if img.shape[0] != out_h or img.shape[1] != out_w:
                img = cv2.resize(img, (out_h, out_w), interpolation=cv2.INTER_CUBIC)
            composed[k] = img # composed: (K, H, W, 3)

        # Upsample surface normals and masks + Normalize surface normals
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

        # Paper Sec. 3.1: Per-image normalization (mean..max scale).
        m_flat = mask_r.reshape(-1)
        I_flat = composed.reshape(K, -1, 3) # (K, H * W, 3)
        if m_flat.sum() > 0:
            valid = I_flat[:, m_flat == 1, :] # (K, V, 3); V = # foregound pixels
        else:
            valid = I_flat
        lum = np.mean(valid, axis=2) # (K, V)
        mx = np.max(lum, axis=1) # (K,)
        mn = np.mean(lum, axis=1) # (K,)
        if augment:
            t = rng.rand(K).astype(np.float32)
            scale = (1.0 - t) * mn + t * mx
        else:
            scale = mx
        # Normalize image
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
