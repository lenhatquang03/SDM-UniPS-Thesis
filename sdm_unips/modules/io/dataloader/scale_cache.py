"""Persistent per-image normalization-scale cache for the training loaders.

Per-image normalization divides each observation by a single scalar: the
**maximum, over the object's foreground pixels, of that pixel's mean across
the three colour channels** (an (H, W, 3) image -> one scalar). This mirrors
the inference path exactly -- see `realdata.py`::

    temp = np.mean(I[:, mask.flatten()==1, :], axis=2)   # per-pixel channel mean
    mx   = np.max(temp, axis=1)                          # per-image max
    temp = mx
    I   /= (temp.reshape(-1,1,1) + 1.0e-6)

so a checkpoint trained here meets the same input distribution at inference
time, and the observations stay in ~[0, 1] -- commensurate with both the mask
channel that `Net.forward` concatenates onto them and the O(1) GLC features
that `Net._decode_pixels` concatenates them with before an `ln=True`
attention block.

Caching. The scalar depends only on the image file and its mask, so where the
observation *is* a fixed file on disk (PolarPS `S0.exr`) it is computed once
and persisted beside the images in a human-readable ``image_scales.config``
(one ``<key> <value>`` line per image, mirroring the existing
``light_means.config`` convention) and reused every epoch.

hdlong deliberately does NOT use this cache: its observation is a random
Dirichlet mix of three component renders, and unlike the mean, the max is
**not** linear -- ``max(w0*p + w1*d + w2*e) != w0*max(p) + w1*max(d) +
w2*max(e)`` -- so there is no per-component quantity to cache. hdlong computes
the scale directly on each composite instead (a masked max over 256x256, which
is negligible next to the three EXR reads that produced it).

Datasets mounted read-only (e.g. Kaggle ``/kaggle/input/...``) cannot receive
the sidecar; writes there fail silently and the caller's in-memory cache still
spares recomputation within the run.

NOTE: the sidecar filename changed from ``image_means.config`` when this
switched from mean- to max-normalization, precisely so any sidecar written by
the older code is ignored rather than silently reused as if it held maxima.
"""

import os

import numpy as np

CACHE_FILENAME = 'image_scales.config'


def read_scales(cache_dir: str) -> dict[str, float]:
    """Load the ``{key: scale}`` sidecar for `cache_dir` (empty if absent)."""
    path = os.path.join(cache_dir, CACHE_FILENAME)
    scales = {}
    if os.path.isfile(path):
        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    try:
                        scales[tokens[0]] = float(tokens[1])
                    except ValueError:
                        continue
    return scales


def write_scales(cache_dir: str, scales: dict[str, float]):
    """Persist `scales` to `cache_dir` atomically; a no-op on read-only mounts.

    Content is deterministic (image + mask fully determine each scalar), so
    concurrent DataLoader workers racing to write produce identical files and
    the atomic replace makes last-writer-wins harmless.
    """
    path = os.path.join(cache_dir, CACHE_FILENAME)
    tmp = path + f'.tmp{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            for key in sorted(scales):
                f.write(f'{key} {scales[key]:.8g}\n')
        os.replace(tmp, path)
    except OSError:
        # Read-only dataset mount: the in-memory cache still applies.
        try:
            os.remove(tmp)
        except OSError:
            pass


def masked_scale(img: np.ndarray, mask: np.ndarray) -> float:
    """Max over foreground pixels of the per-pixel channel mean -> scalar.

    Falls back to the whole-image statistic when the mask is empty so a
    degenerate frame never yields NaN, and clamps at 0 so an all-negative HDR
    frame cannot return a negative scale (which would flip the sign of every
    observation when divided through).
    """
    m = mask.reshape(-1) >= 0.5                    # (H * W,)
    flat = img.reshape(-1, img.shape[-1])          # (H * W, 3)
    per_pixel = flat[m] if m.sum() > 0 else flat   # (n_fg, 3)
    return float(max(per_pixel.mean(axis=-1).max(), 0.0))
