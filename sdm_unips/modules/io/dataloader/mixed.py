"""
Mixed training-dataset wrapper for hdlong-complexv1 + PolarPS.

The Dataset discovers scene directories under one or more roots, infers
which loader each scene belongs to (hdlong has `light_means.config`,
PolarPS has `normal.exr`), and dispatches to the appropriate per-scene
loader. `__getitem__` returns the 4-tuple `(I, N, M, n_imgs)` consumed by
`train._collate`.

`build_mixed_split` performs a deterministic, scene-level train/val/test
split of the discovered pool. Both the held-out val (checkpoint selection)
and the held-out test (final reporting) come from the same synthetic mix,
so no external benchmark is needed and no split leaks into another.

Malformed scenes
----------------
Discovery admits a directory on a single marker file, which is far less than
the loaders need, so the pool is validated by `scene_check` BEFORE the
`--max_scenes` cap and before the split. Order matters: filtering after the
split would leave the val/test fractions and the hdlong:PolarPS ratio
approximate, while filtering first keeps both exact.

The consequence to keep in mind is that the split is a permutation over scene
*positions*, so changing the pool at all changes where every surviving scene
lands. The pool fingerprint is therefore recorded on `args` and guarded on
`--resume`.

Required attrs on `args`:
    - hdlong_dir (optional)
    - polarps_dir (optional)
    - train_dir (optional fallback: auto-detect both kinds under one root)
    - train_resolution, session_name, seed
    - max_scenes (optional cap; default unlimited)
    - val_fraction (held-out fraction for the val split; default 0.1)
    - test_fraction (held-out fraction for the test split; default 0.1)
    - k_per_scene (images drawn per scene; default 10)
    - scene_manifest / strict_scenes / max_bad_scene_frac (optional)

K (`--k_per_scene`) is the number of images drawn per scene, and doubles as
the minimum a scene must be able to supply to enter the pool.
"""

import os

import numpy as np
import torch.utils.data as data

from typing import Callable

from .hdlong import HdlongLoader
from .polarps import PolarPSLoader
from .scene_check import (
    DEFAULT_MANIFEST_NAME, filter_valid_scenes, read_manifest,
    scene_fingerprint, write_manifest,
)

DEFAULT_K_PER_SCENE = 10

# Consecutive scenes tried before giving up when one raises at read time.
# Small on purpose: a handful of corrupt files should be stepped over, but a
# systemic failure (dataset unmounted mid-run) must surface as an error rather
# than spin through the entire split looking for something readable.
MAX_SCENE_SUBSTITUTIONS = 8


def _is_hdlong_scene(path: str) -> bool:
    return os.path.isfile(os.path.join(path, 'light_means.config'))


def _is_polarps_scene(path: str) -> bool:
    return os.path.isfile(os.path.join(path, 'normal.exr'))


def _find_scenes(root: str, is_scene: Callable[[str], bool]) -> list[str]:
    """Find every directory at or under `root` that *directly* holds a scene
    marker, regardless of nesting depth.

    Scenes live at different depths across sources: hdlong scenes are direct
    children of `hdlong-complexv1/`, while PolarPS scenes sit one level deeper
    under a per-object group dir (`PolarPS/<group>/<scene>/normal.exr`). A
    marker-based walk handles both. We prune below a recognized scene so we
    never descend into the heavy `cam_*/` (hdlong) or `light-NN/` (PolarPS)
    leaf trees, and we sort for a deterministic, seed-reproducible order.
    """
    if not root or not os.path.isdir(root):
        return []
    found = []
    # os.walk(root) returns the name of the directory, a list of sub-directories, and a list of files in that directory
    # It decides the next directory to traverse by looking at the previously found list of sub-directories.
    for dirpath, subdirs, _ in os.walk(root):
        if is_scene(dirpath):
            found.append(dirpath)
            subdirs[:] = []  # scene found: don't recurse into its leaves
    return sorted(found)


def _discover(roots: list[str], kinds: list[str]) -> list[tuple[str, str]]:
    """Return list of (kind, scene_dir) for matching roots."""
    predicate = {'hdlong': _is_hdlong_scene, 'polarps': _is_polarps_scene}
    out = []
    for kind, root in zip(kinds, roots):
        for d in _find_scenes(root, predicate[kind]):
            out.append((kind, d))
    return out


def _auto_discover(root: str) -> list[tuple[str, str]]:
    out = [('hdlong', d) for d in _find_scenes(root, _is_hdlong_scene)]
    out += [('polarps', d) for d in _find_scenes(root, _is_polarps_scene)]
    return out


def _proportional_cap(
        hd: list[tuple[str, str]], pp: list[tuple[str, str]], 
        max_scenes: int, seed: int
    ):
    """Sub-sample (hd, pp) down to `max_scenes` while preserving the full-pool
    hdlong:polarps ratio.

    A single uniform draw over the concatenated pool (the previous approach)
    could wipe out the minority source entirely when the cap is small — with
    the full thesis pool (1,198 hdlong vs 17,047 PolarPS) hdlong is only ~6.6%
    of the scenes, so `permutation(len)[:max_scenes]` can easily return very
    few hdlong scenes, or none at a small cap. Instead we allocate the budget
    between the two sources in proportion to their full-pool sizes, then draw
    within each source independently. Every source that is present keeps at
    least one scene, so the mix never silently collapses to a single source.

    Both draws use the same seed for a deterministic, reproducible sub-pool.
    """
    total = len(hd) + len(pp)
    if total <= max_scenes:
        return hd, pp

    # n_hd / max_scenes ~ len(hd) / total
    n_hd = int(round(max_scenes * len(hd) / total)) if hd else 0
    # Clamp n_hd between 1 and len(hd). If n_hd < 1, n_hd=1. If n_hd > len(hd), n_hd=len(hd). Else n_hd
    n_hd = min(max(n_hd, 1), len(hd)) if hd else n_hd
    # Clamp n_pp beteween 1 and len(pp)
    n_pp = min(max(max_scenes - n_hd, 1), len(pp)) if pp else 0
    # If n_hd + n_pp < max_scenes still, take more hdlong-complexv1 scenes.
    n_hd = min(max(max_scenes - n_pp, 1 if hd else 0), len(hd))

    rng = np.random.RandomState(seed)
    hd_sampled = [hd[i] for i in sorted(rng.permutation(len(hd))[:n_hd].tolist())]
    pp_sampled = [pp[i] for i in sorted(rng.permutation(len(pp))[:n_pp].tolist())]
    return hd_sampled, pp_sampled


def _k_per_scene(args) -> int:
    return max(1, int(getattr(args, 'k_per_scene', DEFAULT_K_PER_SCENE)))


def _validate_pool(args, scenes):
    """Reduce the discovered pool to scenes the loaders can actually read.

    Runs before the `--max_scenes` cap and before the split, so both operate on
    a pool that is entirely readable and the fractions/ratio stay exact.

    Three ways to obtain the valid list, in priority order:

    1. `--scene_manifest PATH` — reuse a pinned list. This is how Models B and
       C are made to train on provably the same pool as Model A, immune to
       filesystem drift between their launches.
    2. This session's own manifest, when resuming. A resumed run MUST reproduce
       its original pool, and rescanning could differ (a flaky mount, a
       repaired scene); reusing is both safer and faster.
    3. A fresh scan, whose result is written to the session manifest.

    Sets `args.scene_pool_fingerprint` so it flows into `config.json` and the
    checkpoint's argument snapshot, where `--resume`'s config guard compares it.
    """
    k = _k_per_scene(args)
    roots = {
        'hdlong_dir': getattr(args, 'hdlong_dir', None),
        'polarps_dir': getattr(args, 'polarps_dir', None),
        'train_dir': getattr(args, 'train_dir', None),
    }
    log_dir = (getattr(args, 'log_dir', None)
               or os.path.join(getattr(args, 'session_name', '.'), 'logs'))
    session_manifest = os.path.join(log_dir, DEFAULT_MANIFEST_NAME)
    workers = max(1, int(getattr(args, 'num_workers', 4)) * 4)

    explicit = getattr(args, 'scene_manifest', None)
    reuse = None
    if explicit:
        reuse = explicit
    elif getattr(args, 'resume', None) and os.path.isfile(session_manifest):
        reuse = session_manifest
        print('[SceneCheck] resuming: reusing this session\'s manifest so the '
              'pool (and therefore the split) is identical to the original run.')

    if reuse:
        valid, _man = read_manifest(reuse, k, roots, max_workers=workers)
        args.scene_pool_fingerprint = scene_fingerprint(valid, roots)
        return valid

    valid, invalid = filter_valid_scenes(scenes, k, max_workers=workers)

    if invalid:
        by_kind = {}
        for kind, _p, why in invalid:
            by_kind[kind] = by_kind.get(kind, 0) + 1
        print('[SceneCheck] rejected per source: '
              + ', '.join(f'{kk}={vv:,}' for kk, vv in sorted(by_kind.items())))
        for kind, path, why in invalid[:10]:
            print(f'[SceneCheck]   SKIP {kind}: {path}  ({why})')
        if len(invalid) > 10:
            print(f'[SceneCheck]   ... and {len(invalid) - 10:,} more; the full '
                  f'list with reasons is in the manifest.')

    write_manifest(session_manifest, valid, invalid, k, roots)

    # Guardrails. A few malformed scenes are a data-quality fact; a large
    # fraction is a broken mount, and training on whatever survived would look
    # exactly like a normal (but much worse) run.
    if invalid and getattr(args, 'strict_scenes', False):
        raise RuntimeError(
            f'--strict_scenes: {len(invalid):,} of {len(scenes):,} scenes '
            f'failed validation (see {session_manifest}). Repair them, or drop '
            f'--strict_scenes to train on the {len(valid):,} valid scenes.')
    max_bad = float(getattr(args, 'max_bad_scene_frac', 0.05))
    bad_frac = len(invalid) / max(1, len(scenes))
    if max_bad > 0 and bad_frac > max_bad:
        raise RuntimeError(
            f'{bad_frac:.1%} of the discovered scenes ({len(invalid):,} of '
            f'{len(scenes):,}) failed validation, over --max_bad_scene_frac='
            f'{max_bad:.1%}. That is usually a half-mounted dataset or a wrong '
            f'--k_per_scene ({k}), not data quality — training on the remainder '
            f'would silently use a fraction of the data. See '
            f'{session_manifest} for the reasons; raise the threshold to '
            f'proceed anyway.')
    if not valid:
        raise RuntimeError(
            f'Every one of the {len(scenes):,} discovered scenes failed '
            f'validation. See {session_manifest} for per-scene reasons.')

    args.scene_pool_fingerprint = scene_fingerprint(valid, roots)
    return valid


def _discover_scenes(args):
    """Discover, validate, and (optionally) cap the hdlong + PolarPS pool.

    Order is discover -> validate -> cap, so the cap draws only from readable
    scenes and still reaches its target count. The cap (`--max_scenes`) is
    applied before any train/val split, so the split is carved out of the
    capped pool, and it is allocated per source in proportion to the full-pool
    ratio (see `_proportional_cap`) rather than as a uniform draw, so the
    minority source cannot be sampled out of existence.
    """
    hd_dir = getattr(args, 'hdlong_dir', None)
    pp_dir = getattr(args, 'polarps_dir', None)
    hd = _discover([hd_dir], ['hdlong'])
    pp = _discover([pp_dir], ['polarps'])

    # Fail loud on a configured-but-empty root.
    if hd_dir and os.path.isdir(hd_dir) and not hd:
        print(f"[MixedTrainDataset] WARNING: --hdlong_dir='{hd_dir}' exists but "
              f"yielded 0 scenes (no 'light_means.config' marker found at any "
              f"depth). Did you extract hdlong_config.zip into the same tree?")
    if pp_dir and os.path.isdir(pp_dir) and not pp:
        print(f"[MixedTrainDataset] WARNING: --polarps_dir='{pp_dir}' exists but "
              f"yielded 0 scenes (no 'normal.exr' marker found at any depth). "
              f"Check that the flag points at the PolarPS root.")

    # Execute only if hd = pp = [] with non-NaN train_dir
    if not (hd or pp) and getattr(args, 'train_dir', None):
        auto = _auto_discover(args.train_dir)
        hd = [s for s in auto if s[0] == 'hdlong']
        pp = [s for s in auto if s[0] == 'polarps']

    if len(hd) + len(pp) == 0:
        raise RuntimeError(
            'MixedTrainDataset found no scenes. Pass --hdlong_dir and/or '
            '--polarps_dir, or place scenes under --train_dir.'
        )

    # Validate BEFORE the cap so the cap draws only from readable scenes.
    valid = _validate_pool(args, hd + pp)
    hd = [s for s in valid if s[0] == 'hdlong']
    pp = [s for s in valid if s[0] == 'polarps']
    print(f'[SceneCheck] pool = {len(hd):,} hdlong + {len(pp):,} polarps '
          f'(fingerprint {getattr(args, "scene_pool_fingerprint", "?")})')

    max_scenes = getattr(args, 'max_scenes', None)
    if max_scenes is not None and max_scenes > 0:
        hd, pp = _proportional_cap(hd, pp, max_scenes, getattr(args, 'seed', 42))
    return hd + pp

def build_mixed_split(args, augment=True):
    """Deterministic scene-level train/val/test split of the mixed pool.

    The pool is split proportionally *within* each source (hdlong split
    separately from PolarPS) so the hdlong:PolarPS mix ratio is preserved in
    all three splits. The split is fully determined by `--seed`, so calling
    this repeatedly with the same args yields identical, disjoint
    train/val/test scene lists.

    Returns (train_dataset, val_dataset, test_dataset); val and test never
    augment.
    """
    # Validate the cheap arguments before the (expensive) filesystem walk, so a
    # bad flag dies immediately instead of after discovering ~1200 scenes.
    val_fraction = float(getattr(args, 'val_fraction', 0.1))
    test_fraction = float(getattr(args, 'test_fraction', 0.1))
    seed = int(getattr(args, 'seed', 42))
    for flag, frac in (('--val_fraction', val_fraction),
                       ('--test_fraction', test_fraction)):
        if not 0 < frac < 1:
            raise ValueError(
                f"{flag} must be in the open interval (0, 1); got {frac}. "
                f"Use e.g. 0.1 for a 10% held-out split."
            )
    if val_fraction + test_fraction >= 1:
        raise ValueError(
            f"--val_fraction ({val_fraction}) + --test_fraction "
            f"({test_fraction}) must be < 1 to leave scenes for training."
        )

    scenes = _discover_scenes(args)
    hd = [s for s in scenes if s[0] == 'hdlong']
    pp = [s for s in scenes if s[0] == 'polarps']

    def _split(
        items: list[tuple[str, str]], rng: np.random.RandomState,
        name: str, flag: str
    ):
        n = len(items)
        if n == 0:
            return [], [], []
        # A source present in the pool must land in *all three* splits, which
        # needs at least one scene each. Fewer than 3 scenes is unsplittable.
        if n < 3:
            raise RuntimeError(
                f"The {name} pool has only {n} VALID scene(s), which cannot "
                f"fill the train, val, and test splits (one scene each "
                f"minimum). Note this count is post-validation: check the "
                f"[SceneCheck] lines above and the manifest, since scenes that "
                f"cannot supply K images were removed. Provide at least 3 "
                f"usable {name} scenes, or drop {flag} to train without {name}."
            )
        perm = rng.permutation(n)
        # Clamp so every split keeps >=1 scene: val in [1, n-2], then test in
        # [1, n-1-n_val], leaving n_train = n - n_val - n_test >= 1.
        n_val = min(max(int(round(n * val_fraction)), 1), n - 2)
        n_test = min(max(int(round(n * test_fraction)), 1), n - 1 - n_val)
        val_pos = set(perm[:n_val].tolist())
        test_pos = set(perm[n_val:n_val + n_test].tolist())
        train = [items[i] for i in range(n)
                 if i not in val_pos and i not in test_pos]
        val = [items[i] for i in range(n) if i in val_pos]
        test = [items[i] for i in range(n) if i in test_pos]
        return train, val, test

    # args.seed is used directly for the split, allowing CONSISTENCY ACROSS RUNS, 
    # given identical scene pool
    hd_train, hd_val, hd_test = _split(hd, np.random.RandomState(seed + 1),
                                       'hdlong', '--hdlong_dir')
    pp_train, pp_val, pp_test = _split(pp, np.random.RandomState(seed + 2),
                                       'polarps', '--polarps_dir')

    # The hdlong:PolarPS ratio is preserved across all three splits.
    train_ds = MixedTrainDataset(args, augment=augment,
                                 scenes=hd_train + pp_train,
                                 subset_name='MixedTrain', noun='train')

    # val_ds and test_ds scenes will be wrapped around MixedEvalDataset.
    val_ds = MixedTrainDataset(args, augment=False,
                               scenes=hd_val + pp_val,
                               subset_name='MixedVal', noun='val')
    test_ds = MixedTrainDataset(args, augment=False,
                                scenes=hd_test + pp_test,
                                subset_name='MixedTest', noun='test',
                                announce=False)
    return train_ds, val_ds, test_ds

# Inhertis from Pytorch's base Dataset class
# Must override the "magic" methods: __len__ and __getitem__
# Allows us to plug directly into Pytorch's DataLoader for automatic batching, shuffling, and parallel data loading.
class MixedTrainDataset(data.Dataset):
    """Mixed hdlong-complexv1 + PolarPS training dataset.

    Either discovers scenes from `args` (when `scenes is None`) or wraps a
    pre-split scene list handed in by `build_mixed_split`. `__getitem__`
    returns the 4-tuple expected by `train._collate`: (I, N, M, n_imgs).
    """

    def __init__(self, args, augment: bool=True,
                 scenes: list[tuple[str, str]]|None=None,
                 subset_name='MixedTrain', noun='train', announce: bool=True):
        self.args = args
        self.augment = augment
        self.train_resolution = int(args.train_resolution)
        self.outdir = args.session_name
        # Default to max_scenes scenes with respected hdlong-polarps ratio
        self.scenes = scenes if scenes is not None else _discover_scenes(args)
        if announce:
            n_hd = sum(1 for k, _ in self.scenes if k == 'hdlong')
            n_pp = sum(1 for k, _ in self.scenes if k == 'polarps')
            print(f'[{subset_name}] {len(self.scenes):,} {noun} scenes  '
                  f'({n_hd:,} hdlong + {n_pp:,} polarps)')

        self.k = _k_per_scene(args)
        self._hdlong = HdlongLoader(
            self.train_resolution, outdir=self.outdir, k=self.k,
        )
        self._polarps = PolarPSLoader(
            self.train_resolution, outdir=self.outdir, k=self.k,
        )
        # Scenes already reported as unreadable, so a scene that fails on every
        # epoch produces one line per worker rather than one per access.
        self._reported_bad = set()

    def __len__(self):
        return len(self.scenes)

    def _load(self, kind, scene_dir):
        loader = self._hdlong if kind == 'hdlong' else self._polarps
        loader.load(scene_dir, augment=self.augment)
        return loader

    def _pack(self, loader):
        """Loader state -> the (I, N, M, n_imgs) 4-tuple `train._collate` wants."""
        h, w = loader.h, loader.w
        n = loader.numberOfImages

        # I: (H, W, 3, n) -> pad to (3, H, W, K). Validation guarantees n == K,
        # so the padding is now defensive only; it costs nothing and keeps the
        # tuple shape fixed if a loader ever returns short.
        I = np.zeros((h, w, 3, self.k), np.float32)
        I[..., :n] = loader.I
        I = I.transpose(2, 0, 1, 3)

        N = loader.N.transpose(2, 0, 1).astype(np.float32)         # (3, H, W)
        M = loader.mask.transpose(2, 0, 1).astype(np.float32)      # (1, H, W)
        return I, N, M, np.int64(n)

    def _note_bad(self, scene_dir, exc):
        if scene_dir in self._reported_bad:
            return
        self._reported_bad.add(scene_dir)
        print(f'[SceneSkip] {scene_dir}: {type(exc).__name__}: {exc} '
              f'-- substituting the next scene. (Structural checks run at '
              f'startup, so this is a corrupt/unreadable FILE, not a missing '
              f'one.)', flush=True)

    def _get_with_fallback(self, idx, load_fn):
        """Return the sample at `idx`, stepping forward on a read failure.

        Startup validation catches missing files but cannot catch a corrupt or
        truncated one without decoding every image in the dataset. Left
        unhandled, a single bad file kills a multi-day run, so a failing scene
        is replaced by the next one in the list.

        The substitution is a pure function of the scene list and the failing
        index -- no RNG, no model state -- so two runs that differ only in the
        model see identical data and the A/B contract holds. The bad scene is
        NOT removed from the pool: the split is a permutation over scene
        positions, so dropping one mid-run would reshuffle everything.
        """
        n = len(self.scenes)
        last = None
        for attempt in range(min(MAX_SCENE_SUBSTITUTIONS, n) + 1):
            j = (idx + attempt) % n
            try:
                return load_fn(j)
            except Exception as exc:      # noqa: BLE001 - any read failure
                self._note_bad(self.scenes[j][1], exc)
                last = exc
        raise RuntimeError(
            f'{MAX_SCENE_SUBSTITUTIONS + 1} consecutive scenes failed to load '
            f'starting at index {idx}. That is a systemic failure (dataset '
            f'unmounted, permissions, disk) rather than a few bad files, so it '
            f'is raised instead of skipped. Last error: {last!r}') from last

    # To be called by CPU worker processes defined in torch.utils.data.DataLoader
    def __getitem__(self, idx):
        def load(j):
            kind, scene_dir = self.scenes[j]
            return self._pack(self._load(kind, scene_dir))
        return self._get_with_fallback(idx, load)


class MixedEvalDataset(MixedTrainDataset):
    """Deterministic, multi-trial wrapper over a fixed scene list, used for BOTH
    held-out evaluations.

    `MixedTrainDataset` passes `rng=None` to the scene loaders, which makes them
    fall back to the global `np.random`: `augment=False` only disables the
    horizontal flip, so the camera, the K lights and the Dirichlet mix are still
    re-drawn on every access. For validation that means a *different render of
    every scene each epoch*, and the val curve then mixes model progress with
    render noise. This wrapper pins the draw instead:

    - validation (`n_trials=1`): one fixed render per scene, identical every
      epoch, every run and every model variant.
    - test (`n_trials=3`): each trial is a different but fixed draw, so
      averaging them cuts the variance of the K-image draw without bias, and
      the three draws reproduce across runs.

    Length = len(scenes) * n_trials. For a flat index i, the trial = i // n_scenes 
    and the scene = i % n_scenes; that scene's random camera
    and lights are drawn from an RNG seeded PURELY by (seed, trial, scene).
    Two consequences:

    - REPRODUCIBLE: The draw is a pure function of the index, so the result
      is identical run-to-run and independent of the DataLoader worker count
      (no reliance on per-worker global RNG state).
    - UNBIASED, LOW VARIANCE: Each trial uses a different but fixed seed
      offset, so it draws a different K-image subset of the scene; averaging the
      `n_trials` renders reduces the variance of that random draw without
      biasing the estimate.

    Returns the same `(I, N, M, n_imgs)` 4-tuple as `MixedTrainDataset`, so the
    trainer's `val_step` path consumes it unchanged.
    """

    def __init__(self, args, scenes, n_trials=3, seed=42,
                 subset_name='MixedTestEval', noun='test-eval',
                 announce: bool=True):
        super().__init__(args, augment=False, scenes=scenes,
                         subset_name=subset_name, noun=noun, announce=announce)
        self.n_trials = max(1, int(n_trials))
        self.n_scenes = len(self.scenes)
        self.eval_seed = int(seed)

    def __len__(self):
        return self.n_scenes * self.n_trials

    def __getitem__(self, idx):
        trial_idx, scene_idx = divmod(idx, self.n_scenes)

        def load(j):
            kind, scene_dir = self.scenes[j]
            loader = self._hdlong if kind == 'hdlong' else self._polarps
            # Deterministic per-(trial, scene) RNG -> worker-count-independent
            # and reproducible; the trial offset makes each trial a distinct
            # fixed draw. The seed follows the SUBSTITUTED scene index, so a
            # fallback stays deterministic too.
            seed = (self.eval_seed + 1_000_003 * trial_idx + j) % (2 ** 32)
            loader.load(scene_dir, augment=False, rng=np.random.RandomState(seed))
            return self._pack(loader)

        # Substitute within the scene axis, keeping the trial fixed. A replaced
        # eval scene is scored twice, which is deterministic and identical for
        # every model variant -- unlike letting the sweep die at epoch 1.
        return self._get_with_fallback(scene_idx, load)
