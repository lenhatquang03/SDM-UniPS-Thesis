"""Structural validation of scene directories, run BEFORE the train/val/test split.

Discovery in `mixed.py` admits a directory on a single marker file
(`normal.exr` for PolarPS, `light_means.config` for hdlong). That is far less
than the loaders actually need, so a directory carrying the marker and nothing
else enters the pool, gets assigned to a split, and raises the first time the
sampler reaches it — potentially days into a run.

This module answers, cheaply, "will the loader be able to read this scene?".

Design rule
-----------
The validator and the loader must agree by CONSTRUCTION, not by two parallel
lists of conditions that drift apart. So the "which pieces of this scene are
usable" helpers live here and are called by BOTH:

    usable_cam_dirs()    -> hdlong  : complete cam_* directories
    usable_light_dirs()  -> PolarPS : light-* directories that hold an S0.exr

A scene is valid iff those helpers return enough units for one training sample;
the loaders then draw only from the units the helpers returned. Without the
second half, "at least one usable camera" guarantees nothing — a scene with one
good camera out of three would pass validation and still fail two reads in
three.

Cost
----
Stat-only: directory listings and `isfile`, never an image decode. Decoding
even one EXR per scene across ~18k scenes would take hours. The trade-off is
that a *corrupt* file cannot be detected here — only a missing one — which is
why `mixed.py` also carries a runtime fallback.

The walk is I/O-latency bound rather than CPU bound (each lookup on a network
mount is a round trip), so `filter_valid_scenes` runs it on a thread pool.
`ThreadPoolExecutor.map` preserves input order, so the surviving list is
deterministic regardless of scheduling — which the split depends on.

Paths are stored and hashed RELATIVE to their dataset root
------------------------------------------------------------
The manifest and the fingerprint both describe scenes as
`(kind, root_key, relative/path)` rather than by absolute path, and relative
paths are normalized to forward slashes. Absolute paths would make both
machine-specific: the same dataset mounted at a different point on a second
GPU box would produce a different fingerprint and an unusable manifest, so the
resume guard would flag a difference that isn't one, and Models B and C could
not reuse Model A's pool when trained elsewhere.

Relative paths make the fingerprint answer the question actually being asked —
"is this the same *set of scenes*?" — independently of where the dataset is
mounted, and let `read_manifest` rebase a manifest onto the current roots.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

# v2: scene entries are (kind, root_key, relpath) instead of (kind, abspath).
MANIFEST_VERSION = 2
DEFAULT_MANIFEST_NAME = 'scene_manifest.json'

# Root arguments a scene can live under. `train_dir` is last so that a more
# specific root wins when it is nested inside it (see `relative_to_roots`).
ROOT_KEYS = ('hdlong_dir', 'polarps_dir', 'train_dir')


def _norm(path):
    return os.path.normpath(os.path.abspath(path)) if path else None


def relative_to_roots(path, roots):
    """`(root_key, 'relative/path')` for an absolute scene path.

    The LONGEST matching root wins, because `--train_dir` may be a parent of
    `--hdlong_dir` / `--polarps_dir` and the more specific root is the one that
    will still resolve if the tree is reorganized.

    A scene under no configured root falls back to `('', abspath)`. That keeps
    it usable, at the cost of being machine-specific — which is honest, since
    nothing else can be said about where it would live elsewhere.
    """
    ap = _norm(path)
    best = None
    for key in ROOT_KEYS:
        root = _norm(roots.get(key))
        if not root:
            continue
        if ap == root or ap.startswith(root + os.sep):
            if best is None or len(root) > len(best[1]):
                best = (key, root)
    if best is None:
        return '', ap
    rel = os.path.relpath(ap, best[1])
    return best[0], rel.replace(os.sep, '/')   # posix form: OS-independent hash


def resolve_from_roots(root_key, rel, roots):
    """Inverse of `relative_to_roots`, against the CURRENT roots."""
    if not root_key:
        return rel                      # stored absolute; nothing to rebase
    base = roots.get(root_key)
    if not base:
        raise RuntimeError(
            f'manifest references scenes under --{root_key.replace("_dir", "")}'
            f'_dir, but that flag was not given for this run.')
    return os.path.normpath(os.path.join(_norm(base), rel.replace('/', os.sep)))


# ---------------------------------------------------------------------------
# "Usable unit" helpers — shared by the validator and the loaders
# ---------------------------------------------------------------------------
def _listdir(path):
    """Sorted directory entries, or None if the directory cannot be read.

    None (unreadable) is kept distinct from [] (readable but empty) so the
    caller can report a transient mount failure differently from a genuinely
    malformed scene.
    """
    try:
        return sorted(os.listdir(path))
    except OSError:
        return None


def usable_cam_dirs(scene_dir, k):
    """hdlong: sorted `cam_*` directories complete enough for one sample.

    Complete = has `binary_mask.exr` and `local_normal.exr`, and at least `k`
    each of point / dir / env light images (the loader draws `k` composites,
    each mixing one image of each type).
    """
    names = _listdir(scene_dir)
    if names is None:
        return []
    out = []
    for name in names:
        if not name.startswith('cam_'):
            continue
        cam = os.path.join(scene_dir, name)
        entries = _listdir(cam)
        if entries is None:
            continue
        entry_set = set(entries)
        if not {'binary_mask.exr', 'local_normal.exr'} <= entry_set:
            continue
        n_point = n_dir = n_env = 0
        for e in entries:
            if not e.endswith('.exr'):
                continue
            if e.startswith('point_light_'):
                n_point += 1
            elif e.startswith('dir_light_'):
                n_dir += 1
            elif e.startswith('env_light_'):
                n_env += 1
        if min(n_point, n_dir, n_env) >= k:
            out.append(cam)
    return out


def usable_light_dirs(scene_dir, k=None):
    """PolarPS: `(img_subdir, [light dirs that hold an S0.exr])`, sorted.

    The image sub-directory is `sorted(subdirs)[0]` — the FIRST material-mix
    directory — because that is the only one `PolarPSLoader.load` ever reads.
    Checking "any sub-directory" here would pass a scene whose first directory
    is the empty one and let the loader fail on it.

    `k` short-circuits the scan once `k` usable light dirs are found, which is
    what keeps validation cheap on a scene with 32 lights. Pass None (the
    loaders do) to enumerate all of them.
    """
    names = _listdir(scene_dir)
    if names is None:
        return None, []
    subdirs = [os.path.join(scene_dir, n) for n in names
               if os.path.isdir(os.path.join(scene_dir, n))]
    if not subdirs:
        return None, []
    img_subdir = subdirs[0]          # `names` is sorted, so this is sorted()[0]
    entries = _listdir(img_subdir)
    if entries is None:
        return img_subdir, []
    out = []
    for name in entries:
        if not name.startswith('light-'):
            continue
        d = os.path.join(img_subdir, name)
        if os.path.isfile(os.path.join(d, 'S0.exr')):
            out.append(d)
            if k is not None and len(out) >= k:
                break
    return img_subdir, out


# ---------------------------------------------------------------------------
# Per-scene validation
# ---------------------------------------------------------------------------
def validate_polarps(scene_dir, k):
    """Reason the scene is unusable, or None if it is fine."""
    if not os.path.isfile(os.path.join(scene_dir, 'normal.exr')):
        return 'missing normal.exr'
    img_subdir, lights = usable_light_dirs(scene_dir, k=k)
    if img_subdir is None:
        return 'no material-mix sub-directory'
    if len(lights) < k:
        return f'only {len(lights)} light-*/S0.exr under {os.path.basename(img_subdir)} (need {k})'
    return None


def validate_hdlong(scene_dir, k):
    """Reason the scene is unusable, or None if it is fine."""
    if not os.path.isfile(os.path.join(scene_dir, 'light_means.config')):
        return 'missing light_means.config'
    if not usable_cam_dirs(scene_dir, k):
        return (f'no cam_* directory with binary_mask.exr + local_normal.exr '
                f'and >= {k} each of point/dir/env images')
    return None


def validate_scene(kind, scene_dir, k):
    if kind == 'hdlong':
        return validate_hdlong(scene_dir, k)
    return validate_polarps(scene_dir, k)


# ---------------------------------------------------------------------------
# Pool-level filtering
# ---------------------------------------------------------------------------
def filter_valid_scenes(scenes, k, max_workers=16, progress_every=2000):
    """Split `[(kind, dir), ...]` into (valid, invalid) preserving input order.

    `invalid` entries are `(kind, dir, reason)`. Order is preserved (via
    `ThreadPoolExecutor.map`) because the train/val/test permutation is taken
    over scene *positions*, so a scheduling-dependent order would make the
    split non-reproducible.
    """
    if not scenes:
        return [], []
    t0 = time.time()
    valid, invalid = [], []
    n_total = len(scenes)
    workers = max(1, min(int(max_workers), 32))
    print(f'[SceneCheck] validating {n_total:,} scenes (K={k}, '
          f'{workers} threads, stat-only)...')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reasons = pool.map(lambda s: validate_scene(s[0], s[1], k), scenes)
        for i, ((kind, path), reason) in enumerate(zip(scenes, reasons), 1):
            if reason is None:
                valid.append((kind, path))
            else:
                invalid.append((kind, path, reason))
            if progress_every and i % progress_every == 0:
                print(f'[SceneCheck]   {i:,}/{n_total:,} checked, '
                      f'{len(invalid):,} rejected ({time.time() - t0:.0f}s)')
    print(f'[SceneCheck] done in {time.time() - t0:.1f}s: {len(valid):,} valid, '
          f'{len(invalid):,} rejected.')
    return valid, invalid


def scene_fingerprint(scenes, roots):
    """Short, stable digest of an ordered scene list.

    Stored in the checkpoint and in config.json. It is what lets a resume prove
    it is training on the same pool (and therefore the same split) as the run
    it continues, and lets two model variants be shown to have shared a pool by
    diffing one line.

    Hashed over root-RELATIVE paths, so the same dataset mounted at a different
    point on another machine yields the same fingerprint. Hashing absolute
    paths would make every cross-machine comparison look like a pool change.
    """
    h = hashlib.sha1()
    for kind, path in scenes:
        root_key, rel = relative_to_roots(path, roots)
        h.update(kind.encode())
        h.update(b'\0')
        h.update(rel.encode())
        h.update(b'\0')
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(path, valid, invalid, k, roots):
    """Persist the validated pool. Best-effort: a read-only log dir is not fatal.

    Scenes are written as `[kind, root_key, relative/path]` so the file can be
    reused on a machine that mounts the dataset elsewhere. `roots` is recorded
    too, but only as provenance — `read_manifest` rebases onto the roots of the
    run that reads it.
    """
    payload = {
        'version': MANIFEST_VERSION,
        'k_per_scene': int(k),
        'roots': roots,                      # provenance only; not a constraint
        'fingerprint': scene_fingerprint(valid, roots),
        'n_valid': len(valid),
        'n_invalid': len(invalid),
        'scenes': [[kind, *relative_to_roots(p, roots)] for kind, p in valid],
        'invalid': [[kind, *relative_to_roots(p, roots), why]
                    for kind, p, why in invalid],
    }
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
        print(f'[SceneCheck] manifest -> {path}')
    except OSError as exc:
        print(f'[SceneCheck] WARNING: could not write manifest ({exc}). '
              f'Reruns will rescan; pass --scene_manifest to pin the pool.')
    return payload


def read_manifest(path, k, roots, max_workers=16):
    """Load a manifest, rebase it onto this run's roots, and verify it.

    Reusing a manifest is what guarantees two model variants train on the same
    pool even if the filesystem drifts between their launches. That guarantee
    is only worth anything if a stale or foreign manifest is rejected rather
    than silently used — so K mismatches and missing scenes raise.

    Different dataset ROOTS are not a mismatch: scenes are stored relative to
    their root, so the same dataset at a different mount point rebases cleanly.
    That is what makes the file portable to a second training machine.
    """
    with open(path) as f:
        man = json.load(f)

    if int(man.get('version', 0)) != MANIFEST_VERSION:
        raise RuntimeError(
            f'{path}: manifest version {man.get("version")} != '
            f'{MANIFEST_VERSION}. Delete it and let this run regenerate it '
            f'(v1 stored absolute paths and is not portable).')
    if int(man.get('k_per_scene', -1)) != int(k):
        raise RuntimeError(
            f'{path}: manifest was built with --k_per_scene '
            f'{man.get("k_per_scene")}, this run uses {k}. K changes which '
            f'scenes qualify, so it changes the pool and the split. Use the '
            f'original K, or regenerate the manifest and start a fresh run.')

    entries = man.get('scenes', [])
    if not entries:
        raise RuntimeError(f'{path}: manifest lists no scenes.')
    scenes = [(kind, resolve_from_roots(root_key, rel, roots))
              for kind, root_key, rel in entries]

    man_roots = man.get('roots') or {}
    rebased = {kk: (man_roots.get(kk), vv) for kk, vv in roots.items()
               if vv and man_roots.get(kk) and man_roots.get(kk) != vv}
    if rebased:
        detail = '; '.join(f'{kk}: {a!r} -> {b!r}' for kk, (a, b) in rebased.items())
        print(f'[SceneCheck] manifest was written against different roots '
              f'({detail}); rebasing onto this run\'s roots. The pool is the '
              f'same set of scenes, so the split and the fingerprint are '
              f'unchanged.')

    # A manifest is a promise that these exact scenes exist. Verify rather than
    # discovering a missing mount one epoch in. One stat per scene, threaded.
    workers = max(1, min(int(max_workers), 32))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        present = list(pool.map(lambda s: os.path.isdir(s[1]), scenes))
    missing = [s for s, ok in zip(scenes, present) if not ok]
    if missing:
        raise RuntimeError(
            f'{path}: {len(missing)} of {len(scenes)} manifest scenes are '
            f'missing on disk after rebasing onto this run\'s roots, e.g. '
            f'{[m[1] for m in missing[:3]]}. Either the roots point somewhere '
            f'else than the manifest describes, or the dataset is incomplete. '
            f'Dropping them would change the split, so this is a hard error — '
            f'fix the roots/mount, or start a fresh run without '
            f'--scene_manifest.')

    print(f'[SceneCheck] reusing manifest {path}: {len(scenes):,} scenes, '
          f'fingerprint={man.get("fingerprint")} (scan skipped)')
    return scenes, man
