"""Persistent per-image mean cache for the training loaders.

Per-image normalization divides each observation by a single scalar: the
mean intensity over the object's foreground pixels, averaged across all
three colour channels (an (H, W, 3) image -> one scalar). That scalar
depends only on the image file and its mask, so it is stable across the
whole training run. We therefore compute it once and persist it beside the
images in a human-readable ``image_means.config`` (one ``<key> <value>``
line per image, mirroring the existing ``light_means.config`` convention)
so it can be reused every epoch and inspected easily.

Datasets mounted read-only (e.g. Kaggle ``/kaggle/input/...``) cannot
receive the sidecar; writes there fail silently and the caller's in-memory
cache still spares recomputation within the run.
"""

import os

import numpy as np

CACHE_FILENAME = 'image_means.config'


def read_means(cache_dir: str) -> dict[str, float]:
    """Load the ``{key: mean}`` sidecar for `cache_dir` (empty if absent)."""
    path = os.path.join(cache_dir, CACHE_FILENAME)
    means = {}
    if os.path.isfile(path):
        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    try:
                        means[tokens[0]] = float(tokens[1])
                    except ValueError:
                        continue
    return means


def write_means(cache_dir: str, means: dict[str, float]):
    """Persist `means` to `cache_dir` atomically; a no-op on read-only mounts.

    Content is deterministic (image + mask fully determine each scalar), so
    concurrent DataLoader workers racing to write produce identical files and
    the atomic replace makes last-writer-wins harmless.
    """
    path = os.path.join(cache_dir, CACHE_FILENAME)
    tmp = path + f'.tmp{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            for key in sorted(means):
                f.write(f'{key} {means[key]:.8g}\n')
        os.replace(tmp, path)
    except OSError:
        # Read-only dataset mount: the in-memory cache still applies.
        try:
            os.remove(tmp)
        except OSError:
            pass


def masked_mean(img: np.ndarray, mask: np.ndarray) -> float:
    """Mean over foreground pixels across all colour channels -> scalar.

    Falls back to the whole-image mean when the mask is empty so a degenerate
    frame never yields NaN.
    """
    m = mask.reshape(-1) >= 0.5 # (H * W)
    flat = img.reshape(-1, img.shape[-1]) # (H * W, 3)
    if m.sum() > 0:
        return float(flat[m].mean())
    return float(flat.mean())
