"""Root-relative manifest/fingerprint behaviour, especially with SEVERAL roots.

`scene_check` is deliberately stdlib-only (no numpy, no torch), so this file
runs anywhere — including a laptop with no training environment installed. It
is loaded straight from its path rather than imported as
`modules.io.dataloader.scene_check`, because that package's `__init__` chain
pulls in cv2/numpy/torch.

What is being protected:

* Two roots holding identically-named scenes must NOT collapse to the same
  manifest entry. They would silently share a fingerprint, and `read_manifest`
  would rebase both onto whichever root came first — i.e. train on one dataset
  twice and never touch the other.
* A SINGLE-root pool must hash exactly as it did before multi-root support
  existed, or Model A's fingerprint changes, its checkpoints stop resuming
  (the fingerprint is split-critical) and its manifest stops being reusable
  by Models B and C.
* A manifest must stay portable across mount points, and must refuse to
  silently drop scenes it cannot find.

Run: python sdm_unips/tests/test_scene_check_roots.py
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, '..', 'modules', 'io', 'dataloader', 'scene_check.py')
_spec = importlib.util.spec_from_file_location('scene_check', _SRC)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


class Args:
    """Stand-in for the argparse namespace `build_roots` reads."""

    def __init__(self, hdlong_dir=None, polarps_dir=None, train_dir=None):
        self.hdlong_dir = hdlong_dir
        self.polarps_dir = polarps_dir
        self.train_dir = train_dir


def make_hdlong_scene(root, name, k=2, cams=1):
    """A structurally valid hdlong scene: marker + a complete cam_* dir."""
    scene = os.path.join(root, name)
    os.makedirs(scene, exist_ok=True)
    with open(os.path.join(scene, 'light_means.config'), 'w') as f:
        f.write('point_mean 1.0\ndir_mean 1.0\nenv_mean 1.0\n')
    for c in range(cams):
        cam = os.path.join(scene, f'cam_{c:05d}')
        os.makedirs(cam, exist_ok=True)
        for fname in ('binary_mask.exr', 'local_normal.exr'):
            open(os.path.join(cam, fname), 'w').close()
        for i in range(k):
            for prefix in ('point_light_', 'dir_light_', 'env_light_'):
                open(os.path.join(cam, f'{prefix}{i:05d}.exr'), 'w').close()
    return scene


# ---------------------------------------------------------------------------
def test_as_roots_normalizes_and_dedupes():
    assert sc.as_roots(None) == []
    assert sc.as_roots('/a') == ['/a']            # pre-nargs checkpoints
    assert sc.as_roots(['/a', '/b']) == ['/a', '/b']
    # Same root twice would enter every scene into the pool twice.
    assert sc.as_roots(['/a', '/a']) == ['/a']
    assert sc.as_roots(['/a', '/a/']) == ['/a']   # normalized before comparing


def test_first_root_keeps_the_bare_key():
    """Single-root runs must be byte-identical to the pre-multi-root code."""
    roots = sc.build_roots(Args(hdlong_dir=['/data/hd'], polarps_dir='/data/pp'))
    assert roots == {'hdlong_dir': '/data/hd', 'polarps_dir': '/data/pp'}

    roots = sc.build_roots(Args(hdlong_dir=['/data/hd', '/data/merlmix']))
    assert roots == {'hdlong_dir': '/data/hd', 'hdlong_dir#1': '/data/merlmix'}


def test_single_root_fingerprint_is_unchanged():
    """The hash of a one-root pool must not move.

    It is computed over `(kind, relpath)` pairs, and the first root still keys
    as `hdlong_dir`, so this is the invariant that keeps Model A's checkpoints
    resumable across this change.
    """
    roots = sc.build_roots(Args(hdlong_dir=['/data/hd']))
    scenes = [('hdlong', '/data/hd/SCENE_0001'), ('hdlong', '/data/hd/SCENE_0002')]
    # Reference value computed the old way: one root, key irrelevant to the
    # hash, paths relative to it.
    import hashlib
    h = hashlib.sha1()
    for kind, rel in (('hdlong', 'SCENE_0001'), ('hdlong', 'SCENE_0002')):
        h.update(kind.encode()); h.update(b'\0')
        h.update(rel.encode()); h.update(b'\0')
    assert sc.scene_fingerprint(scenes, roots) == h.hexdigest()[:16]


def test_same_scene_name_under_two_roots_stays_distinct():
    """The collision this whole keying scheme exists to prevent."""
    roots = sc.build_roots(Args(hdlong_dir=['/data/hd', '/data/merlmix']))
    a = sc.relative_to_roots('/data/hd/SCENE_0001', roots)
    b = sc.relative_to_roots('/data/merlmix/SCENE_0001', roots)
    assert a == ('hdlong_dir', 'SCENE_0001')
    assert b == ('hdlong_dir#1', 'SCENE_0001')
    assert a != b
    # ...and they resolve back to different places.
    assert sc.resolve_from_roots(*a, roots) != sc.resolve_from_roots(*b, roots)
    # A pool of the two must not hash like a pool of one scene twice.
    both = [('hdlong', '/data/hd/SCENE_0001'),
            ('hdlong', '/data/merlmix/SCENE_0001')]
    twice = [('hdlong', '/data/hd/SCENE_0001'),
             ('hdlong', '/data/hd/SCENE_0001')]
    assert sc.scene_fingerprint(both, roots) != sc.scene_fingerprint(twice, roots)


def test_root_order_changes_the_pool_identity():
    """Swapping the roots is a different pool, and the fingerprint says so."""
    fwd = sc.build_roots(Args(hdlong_dir=['/data/hd', '/data/merlmix']))
    rev = sc.build_roots(Args(hdlong_dir=['/data/merlmix', '/data/hd']))
    scenes = [('hdlong', '/data/hd/S1'), ('hdlong', '/data/merlmix/S1')]
    assert sc.scene_fingerprint(scenes, fwd) != sc.scene_fingerprint(scenes, rev)


def test_longest_root_wins_when_nested():
    """--train_dir may be a parent; the specific root must still claim a scene."""
    roots = sc.build_roots(Args(hdlong_dir=['/data/all/hd'], train_dir='/data/all'))
    assert sc.relative_to_roots('/data/all/hd/S1', roots) == ('hdlong_dir', 'S1')


def test_scene_under_no_root_falls_back_to_absolute():
    roots = sc.build_roots(Args(hdlong_dir=['/data/hd']))
    key, rel = sc.relative_to_roots('/elsewhere/S1', roots)
    assert key == ''
    assert rel == os.path.normpath('/elsewhere/S1')


def test_resolve_reports_a_root_this_run_lacks():
    """A manifest from a 2-root run, replayed with 1 root, must not silently
    rebase the second root's scenes onto the first."""
    roots = sc.build_roots(Args(hdlong_dir=['/data/hd']))
    try:
        sc.resolve_from_roots('hdlong_dir#1', 'SCENE_0001', roots)
    except RuntimeError as exc:
        assert '--hdlong_dir' in str(exc) and 'root #2' in str(exc)
    else:
        raise AssertionError('expected a RuntimeError for the missing root')


# ---------------------------------------------------------------------------
def test_multi_root_manifest_round_trip_and_rebase():
    """Write a 2-root manifest, then read it back at a DIFFERENT mount point.

    Same set of scenes => same fingerprint, and every path resolves under the
    new roots. This is what lets Models B and C reuse Model A's pool on another
    machine.
    """
    tmp = tempfile.mkdtemp()
    try:
        hd = os.path.join(tmp, 'box1', 'hdlong')
        mm = os.path.join(tmp, 'box1', 'merlmix')
        # Deliberately the SAME scene names under both roots.
        scenes = [('hdlong', make_hdlong_scene(hd, 'SCENE_0001')),
                  ('hdlong', make_hdlong_scene(hd, 'SCENE_0002')),
                  ('hdlong', make_hdlong_scene(mm, 'SCENE_0001')),
                  ('hdlong', make_hdlong_scene(mm, 'SCENE_0002'))]

        args = Args(hdlong_dir=[hd, mm])
        roots = sc.build_roots(args)
        valid, invalid = sc.filter_valid_scenes(scenes, k=2, progress_every=0)
        assert len(valid) == 4 and not invalid, (valid, invalid)

        man_path = os.path.join(tmp, 'scene_manifest.json')
        payload = sc.write_manifest(man_path, valid, invalid, k=2, roots=roots)
        assert payload['fingerprint'] == sc.scene_fingerprint(valid, roots)
        # Four DISTINCT entries, not two collapsed pairs.
        entries = {tuple(e) for e in payload['scenes']}
        assert len(entries) == 4, entries

        # Same datasets, mounted elsewhere (a second GPU box).
        hd2 = os.path.join(tmp, 'box2', 'hdlong')
        mm2 = os.path.join(tmp, 'box2', 'merlmix')
        os.makedirs(os.path.dirname(hd2), exist_ok=True)
        shutil.copytree(hd, hd2)
        shutil.copytree(mm, mm2)

        roots2 = sc.build_roots(Args(hdlong_dir=[hd2, mm2]))
        rebased, man = sc.read_manifest(man_path, k=2, roots=roots2)
        assert len(rebased) == 4
        assert all(os.path.isdir(p) for _k, p in rebased)
        assert {p for _k, p in rebased} == {
            os.path.join(hd2, 'SCENE_0001'), os.path.join(hd2, 'SCENE_0002'),
            os.path.join(mm2, 'SCENE_0001'), os.path.join(mm2, 'SCENE_0002')}
        # The point of relative storage: a different mount is the SAME pool.
        assert sc.scene_fingerprint(rebased, roots2) == payload['fingerprint']

        # A fresh scan at the new mount point must agree with the manifest.
        fresh, _ = sc.filter_valid_scenes(
            [('hdlong', os.path.join(r, n))
             for r in (hd2, mm2) for n in ('SCENE_0001', 'SCENE_0002')],
            k=2, progress_every=0)
        assert sc.scene_fingerprint(fresh, roots2) == payload['fingerprint']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manifest_missing_scene_is_a_hard_error():
    """Dropping a scene would reshuffle the whole split, so it must raise."""
    tmp = tempfile.mkdtemp()
    try:
        hd = os.path.join(tmp, 'hdlong')
        scenes = [('hdlong', make_hdlong_scene(hd, f'SCENE_{i:04d}'))
                  for i in range(3)]
        roots = sc.build_roots(Args(hdlong_dir=[hd]))
        man_path = os.path.join(tmp, 'm.json')
        sc.write_manifest(man_path, scenes, [], k=2, roots=roots)

        shutil.rmtree(scenes[1][1])
        try:
            sc.read_manifest(man_path, k=2, roots=roots)
        except RuntimeError as exc:
            assert 'missing on disk' in str(exc)
        else:
            raise AssertionError('expected a RuntimeError for the missing scene')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validator_rejects_a_scene_short_of_K():
    """The pool's validity threshold is K, and it is checked per cam_* dir."""
    tmp = tempfile.mkdtemp()
    try:
        hd = os.path.join(tmp, 'hdlong')
        good = make_hdlong_scene(hd, 'GOOD', k=4)
        thin = make_hdlong_scene(hd, 'THIN', k=2)
        marker_only = os.path.join(hd, 'MARKER_ONLY')
        os.makedirs(marker_only)
        open(os.path.join(marker_only, 'light_means.config'), 'w').close()

        valid, invalid = sc.filter_valid_scenes(
            [('hdlong', good), ('hdlong', thin), ('hdlong', marker_only)],
            k=4, progress_every=0)
        assert [p for _k, p in valid] == [good]
        assert {p for _k, p, _why in invalid} == {thin, marker_only}
        # And the loaders draw only from what the shared helper returns.
        assert sc.usable_cam_dirs(good, 4) and not sc.usable_cam_dirs(thin, 4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')
    print(f'\n{"FAILED" if failures else "OK"} ({failures} failure(s))')
    sys.exit(1 if failures else 0)
