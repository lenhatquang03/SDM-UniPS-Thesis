"""Per-image normalization statistics (and their PolarPS sidecar cache).

Per-image normalization divides each observation by a single scalar. Following
SDM-UniPS Sec. 3.1 -- *"each image is normalized by a random value between its
maximum and mean"* -- that scalar is drawn per image from ``U[mean, max]``
while **training**, and pinned to the **max** everywhere else. Both statistics
are taken over the object's foreground pixels, of that pixel's mean across the
three colour channels (an (H, W, 3) image -> one scalar).

The eval/inference end of that range mirrors `realdata.py` exactly::

    temp = np.mean(I[:, mask.flatten()==1, :], axis=2)   # per-pixel channel mean
    mx   = np.max(temp, axis=1)                          # per-image max
    temp = mx
    I   /= (temp.reshape(-1,1,1) + 1.0e-6)

so a checkpoint trained here meets the same input distribution at inference
time, and the observations stay in ~[0, 1] -- commensurate with both the mask
channel that `Net.forward` concatenates onto them and the O(1) GLC features
that `Net._decode_pixels` concatenates them with before an `ln=True`
attention block. Dividing by a scalar *below* the max is what widens the
training distribution: it pushes observations above 1, so the network cannot
assume its input is exactly max-normalized. Held-out val and test therefore
run at the max (`augment=False`), which is both the inference condition and a
fixed one -- a random normalizer at eval would add variance to the number that
selects `best.pt` for no information.

Caching. Both statistics depend only on the image file and its mask, so where
the observation *is* a fixed file on disk (PolarPS `S0.exr`) they are computed
once and persisted beside the images in a human-readable
``image_scale_stats.config`` (one ``<key> <mean> <max>`` line per image,
mirroring the existing ``light_means.config`` convention) and reused every
epoch. Only the *statistics* are cached, never the sampled scale: that is
redrawn every epoch, which is the whole point of the augmentation.

hdlong deliberately does NOT use this cache: its observation is a random
Dirichlet mix of three component renders, and unlike the mean, the max is
**not** linear -- ``max(w0*p + w1*d + w2*e) != w0*max(p) + w1*max(d) +
w2*max(e)`` -- so there is no per-component quantity to cache. hdlong computes
both statistics directly on each composite instead (one masked pass over
256x256, negligible next to the three EXR reads that produced it).

Datasets mounted read-only (e.g. Kaggle ``/kaggle/input/...``) cannot receive
the sidecar; writes there fail silently and the caller's in-memory cache still
spares recomputation within the run.

NOTE: the sidecar filename has changed twice, each time so that a file written
by older code is ignored rather than silently reused as something it is not:
``image_means.config`` (mean-normalization) -> ``image_scales.config``
(max-normalization, one value per line) -> ``image_scale_stats.config`` (two
values per line). Delete any of the older files still lying in the dataset
trees; nothing reads them.
"""

import os

import numpy as np

CACHE_FILENAME = 'image_scale_stats.config'


def read_scale_stats(cache_dir: str) -> dict[str, tuple[float, float]]:
    """Load the ``{key: (mean, max)}`` sidecar for `cache_dir` (empty if absent)."""
    path = os.path.join(cache_dir, CACHE_FILENAME)
    stats = {}
    if os.path.isfile(path):
        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                # Three tokens exactly: a two-token line is an `image_scales`
                # sidecar from the max-only era, which must NOT be read as if
                # its single value were a mean.
                if len(tokens) >= 3:
                    try:
                        stats[tokens[0]] = (float(tokens[1]), float(tokens[2]))
                    except ValueError:
                        continue
    return stats


def write_scale_stats(cache_dir: str, stats: dict[str, tuple[float, float]]):
    """Persist `stats` to `cache_dir` atomically; a no-op on read-only mounts.

    Content is deterministic (image + mask fully determine both scalars), so
    concurrent DataLoader workers racing to write produce identical files and
    the atomic replace makes last-writer-wins harmless.
    """
    path = os.path.join(cache_dir, CACHE_FILENAME)
    tmp = path + f'.tmp{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            for key in sorted(stats):
                mn, mx = stats[key]
                f.write(f'{key} {mn:.8g} {mx:.8g}\n')
        os.replace(tmp, path)
    except OSError:
        # Read-only dataset mount: the in-memory cache still applies.
        try:
            os.remove(tmp)
        except OSError:
            pass


def masked_scale_stats(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Foreground (mean, max) of the per-pixel channel mean -> two scalars.

    Falls back to the whole-image statistic when the mask is empty so a
    degenerate frame never yields NaN, and clamps at 0 so an all-negative HDR
    frame cannot return a negative scale (which would flip the sign of every
    observation when divided through). The mean is additionally clamped to the
    max so `resolve_scales` always interpolates over a non-inverted interval.
    """
    m = mask.reshape(-1) >= 0.5                    # (H * W,)
    flat = img.reshape(-1, img.shape[-1])          # (H * W, 3)
    per_pixel = flat[m] if m.sum() > 0 else flat   # (n_fg, 3)
    lum = per_pixel.mean(axis=-1)                  # (n_fg,)
    mx = float(max(lum.max(), 0.0))
    mn = float(min(max(lum.mean(), 0.0), mx))
    return mn, mx


def resolve_scales(means: np.ndarray, maxes: np.ndarray, augment: bool, rng) -> np.ndarray:
    """Per-image normalization scalars: ``U[mean, max]`` if `augment`, else max.

    Paper Sec. 3.1. The draw is per *image* (not per scene), and comes from
    `rng` rather than the global `np.random` so a caller that pins its RNG --
    `MixedEvalDataset` -- gets a reproducible result. Eval never augments, so
    in practice this returns the max there whatever the RNG is doing.
    """
    mx = np.asarray(maxes, np.float32) # (K,)
    if not augment:
        return mx
    mn = np.clip(np.asarray(means, np.float32), 0.0, mx) # (K,)
    t = np.asarray(rng.rand(len(mx)), np.float32) # (K,)
    return ((1.0 - t) * mn + t * mx).astype(np.float32) # (K, )
