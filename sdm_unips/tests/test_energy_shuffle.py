"""Checks for the Rule C control: --energy_shuffle per_step (and per_run, for completeness).

Run on the training box (torch, CPU is enough; no GPU and no full Net forward needed):

    python sdm_unips/tests/test_energy_shuffle.py        # prints ALL PASS

The methods under test are called unbound on a stand-in object, so the test does not
build the 512-px network. What is checked, in order of importance for the per_step run:

1. Multiset: a permutation of the 64x64 cells keeps the values exactly (ESS and tilt of
   the cell-level distribution are unchanged).
2. Fresh per draw (per_step): two consecutive calls on the same map give different
   placements, so the alignment with the surface is broken at every draw.
3. Position-free in expectation (per_step): averaged over many draws, every cell gets
   the same expected energy, so per_step carries no fixed spatial prior. (per_run would
   fail this for a fixed map: the expectation is then one fixed permutation.)
4. Destroyed alignment: the shuffled map is not the original map, and its correlation
   with the original is near zero.
5. Resume-safety: with the same torch seed, per_step reproduces the same placement
   (it draws from the training RNG, which is restored on resume).
6. Shape and dtype are unchanged, so the downstream WESS code is unaffected.
7. tau selection: with the flag off the draw uses wess_tau; with per_step and a shuffled
   tau set, the draw uses energy_shuffle_tau.

Rule C v2, --energy_shuffle_match_ess (online per-scene ESS match). Synthetic scene: a
256x256 disk mask and a 64x64 energy map with a smooth interior bump, a high silhouette
ring and near-zero background, i.e. the three populations the whole-frame shuffle mixes.

8. Flag off = baseline: with match off, sample_train_pixels gives exactly the ids and
   weights of the pre-change body (shuffle, then wess_sample_train at the fixed tau).
9. Monotonic: ESS of the shuffled draw is non-decreasing in tau (bisection well posed).
10. Match: the returned tau gives a shuffled draw whose ESS is within 1e-3 of the real
    draw's ESS at wess_tau, recomputed independently.
11. No RNG consumed by the match: the torch RNG state is unchanged across it.
12. Logging: with match on, every stats record carries the four match keys, and the
    logged wess_ess equals wess_ess_target within 1e-3.
13. Uniform fallback (lam = 1): nothing to match; resid 0, tau NaN.
"""
import math
import sys
import types

import torch

sys.path.insert(0, __file__.rsplit('/sdm_unips/', 1)[0])
from sdm_unips.modules.model.model import Net  # noqa: E402
from sdm_unips.modules.model import wess as wess_mod  # noqa: E402

H = W = 64


def stand_in(mode, tau=None, seed=0, wess_tau=1.0):
    """Minimal object carrying the attributes that the unbound methods read."""
    o = types.SimpleNamespace()
    o.energy_shuffle = mode
    o.energy_shuffle_seed = seed
    o.energy_shuffle_tau = tau
    o.wess_tau = wess_tau
    o._energy_perm_cache = {}
    o._shuffle_energy = lambda E: Net._shuffle_energy(o, E)
    o._draw_tau = lambda: Net._draw_tau(o)
    return o


def peaked_map(seed=1):
    """A realistic-looking energy map: a few hot cells on a flat background."""
    g = torch.Generator().manual_seed(seed)
    E = torch.rand(H, W, generator=g) * 0.1
    E[20:24, 30:34] += 3.0
    E[45:50, 10:12] += 2.0
    return E


def test_multiset_preserved():
    o = stand_in('per_step')
    E = peaked_map()
    torch.manual_seed(0)
    S = o._shuffle_energy(E)
    assert S.shape == E.shape and S.dtype == E.dtype
    assert torch.equal(torch.sort(S.reshape(-1)).values, torch.sort(E.reshape(-1)).values), \
        'per_step changed the multiset of energy values'
    # ESS of the cell-level draw distribution p = E / sum(E) is permutation-invariant.
    # float64 and a relative tolerance: a permutation reorders the float sums, so
    # float32 results differ by rounding alone.
    ess = lambda X: float(1.0 / (((X.double() / X.double().sum()) ** 2).sum()))
    e_ess, s_ess = ess(E), ess(S)
    assert abs(s_ess - e_ess) <= 1e-9 * abs(e_ess), f'ESS changed: {e_ess} -> {s_ess}'
    print('ok: multiset and cell-level ESS preserved')


def test_fresh_placement_each_draw():
    o = stand_in('per_step')
    E = peaked_map()
    torch.manual_seed(1)
    a = o._shuffle_energy(E)
    b = o._shuffle_energy(E)
    assert not torch.equal(a, b), 'two per_step draws gave the same placement'
    print('ok: each per_step draw gives a new placement')


def test_alignment_destroyed():
    o = stand_in('per_step')
    E = peaked_map()
    torch.manual_seed(2)
    S = o._shuffle_energy(E)
    assert not torch.equal(S, E)
    # Correlation with the original map: near zero (cell permutation of a sparse map).
    a = (S - S.mean()).reshape(-1)
    b = (E - E.mean()).reshape(-1)
    r = float((a * b).sum() / (a.norm() * b.norm()))
    assert abs(r) < 0.1, f'shuffled map still aligned with original: r={r:.3f}'
    print(f'ok: alignment destroyed (corr with original = {r:+.4f})')


def test_no_fixed_spatial_prior():
    """Average the shuffled map over many draws. With per_step the expected energy of
    every cell is the same (the cell mean), so there is no fixed image-position bias."""
    o = stand_in('per_step')
    E = peaked_map()
    torch.manual_seed(3)
    n_draws = 3000
    acc = torch.zeros(H, W)
    for _ in range(n_draws):
        acc += o._shuffle_energy(E)
    acc /= n_draws
    cell_mean = float(E.mean())
    # Each cell's average over draws should be close to the cell mean. The standard error
    # of one cell's mean over n draws is sd(E)/sqrt(n); allow 6 standard errors.
    se = float(E.std()) / n_draws ** 0.5
    dev = (acc - cell_mean).abs().max().item()
    assert dev < 6 * se, f'per_step has a spatial prior: max deviation {dev:.4f} > {6 * se:.4f}'
    # Contrast: a single fixed permutation (per_run) keeps the peaks at fixed places.
    o_run = stand_in('per_run', seed=7)
    fixed = o_run._shuffle_energy(E)
    assert not torch.allclose(fixed, acc, atol=6 * se), 'per_run unexpectedly looks like per_step'
    print(f'ok: per_step expectation is flat (max dev {dev:.4f}, 6 SE = {6 * se:.4f})')


def test_resume_reproducible():
    """Same torch seed gives the same placement: per_step depends only on the training RNG,
    which the checkpoint restores on resume."""
    o = stand_in('per_step')
    E = peaked_map()
    torch.manual_seed(4)
    a = o._shuffle_energy(E)
    torch.manual_seed(4)
    b = o._shuffle_energy(E)
    assert torch.equal(a, b), 'per_step is not reproducible under the same RNG state'
    assert o._energy_perm_cache == {}, 'per_step must not cache a permutation'
    print('ok: per_step reproducible under the same RNG state, nothing cached')


def test_shape_for_batch_of_scenes():
    """Each scene is shuffled independently: consecutive calls on different scenes give
    different maps even when the scenes are identical copies."""
    o = stand_in('per_step')
    E = peaked_map()
    torch.manual_seed(5)
    outs = [o._shuffle_energy(E) for _ in range(4)]
    for x in outs:
        assert x.shape == (H, W)
    assert len({tuple(x.reshape(-1).tolist()[:8]) for x in outs}) > 1
    print('ok: shape kept, scenes shuffled independently')


def test_tau_selection():
    off = stand_in(None, tau=None, wess_tau=1.0)
    assert off._draw_tau() == 1.0, 'flag off must keep wess_tau'
    on = stand_in('per_step', tau=1.37, wess_tau=1.0)
    assert on._draw_tau() == 1.37, 'per_step must use energy_shuffle_tau'
    unset = stand_in('per_step', tau=None, wess_tau=1.0)
    assert unset._draw_tau() == 1.0, 'per_step without a tau falls back to wess_tau (launch sets it)'
    print('ok: tau selection for flag off / per_step')


# --- Rule C v2: online ESS match (--energy_shuffle_match_ess) ---------------

HD = 256          # decoder grid (training uses 512; 256 keeps the test light)
G = 64            # energy grid, as in training
N_SAMPLE = 512
LAM, ERODE, TAU_REF = 0.25, 2, 1.0


def scene(seed=0):
    yy, xx = torch.meshgrid(torch.arange(HD), torch.arange(HD), indexing='ij')
    mask = (((yy - 128) ** 2 + (xx - 128) ** 2) <= 100 ** 2).float()
    gy, gx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing='ij')
    r = ((gy - 31.5) ** 2 + (gx - 31.5) ** 2).float().sqrt()
    g = torch.Generator().manual_seed(seed)
    E = torch.full((G, G), 1e-6)
    inside = r <= 25
    E[inside] = 0.3 + 0.1 * torch.rand(int(inside.sum()), generator=g)
    E += 1.5 * torch.exp(-((gy - 20.0) ** 2 + (gx - 36.0) ** 2) / 18.0) * inside   # crease-like bump
    ring = (r > 23) & (r <= 26)
    E[ring] = 4.0 + torch.rand(int(ring.sum()), generator=g)                      # silhouette ring
    valid = torch.nonzero(mask.reshape(-1) > 0, as_tuple=False)[:, 0]
    return E, mask, valid


def stand_in_match(match, tau=None, iw=True, lam=LAM):
    o = types.SimpleNamespace()
    o.energy_shuffle = 'per_step'
    o.energy_shuffle_seed = 0
    o.energy_shuffle_tau = tau
    o.energy_shuffle_match_ess = match
    o.wess_tau, o.wess_lam, o.wess_erode_cells, o.wess_iw = TAU_REF, lam, ERODE, iw
    o._energy_perm_cache = {}
    o._sample_weights_acc, o._wess_stats_acc = [], []
    o._shuffle_energy = lambda E: Net._shuffle_energy(o, E)
    o._draw_tau = lambda: Net._draw_tau(o)
    return o


def ess_shuf(E_shuf, mask, valid, tau, lam=LAM):
    interior = wess_mod.interior_mask(mask, E_shuf.shape[-2:], ERODE)
    p, _, _ = wess_mod.wess_train_probabilities(E_shuf, valid, interior, HD, HD, tau=tau, lam=lam)
    return wess_mod._ess_frac(p)


def test_flag_off_equals_baseline():
    E, mask, valid = scene()
    for iw in (True, False):
        o = stand_in_match(match=False, tau=1.35, iw=iw)
        torch.manual_seed(11)
        ids = Net.sample_train_pixels(o, valid, N_SAMPLE, E=E, mask_hw=mask, H=HD, W=HD)
        # Pre-change body, written out.
        ref = stand_in_match(match=False, tau=1.35, iw=iw)
        torch.manual_seed(11)
        E2 = Net._shuffle_energy(ref, E)
        out = wess_mod.wess_sample_train(E2, mask, valid, HD, HD, N_SAMPLE, tau=1.35, lam=LAM,
                                         erode_cells=ERODE, return_weights=iw)
        assert torch.equal(ids, out[0]), f'flag-off ids differ from baseline (iw={iw})'
        if iw:
            assert torch.equal(o._sample_weights_acc[0], out[2]), 'flag-off weights differ'
        # sample_train_pixels adds the two iw keys itself (pre-change too); nothing else.
        extra = set(o._wess_stats_acc[0]) - set(out[1]) - {'wess_iw_w_mean', 'wess_iw_w_max'}
        assert not extra, f'flag-off added stats keys: {extra}'
        print(f'ok: flag off == baseline (iw={iw}): max|d ids| = '
              f'{int((ids - out[0]).abs().max())}')


def test_monotonic_in_tau():
    E, mask, valid = scene()
    torch.manual_seed(12)
    S = Net._shuffle_energy(stand_in_match(True), E)
    taus = [0.05 * (1.25 ** k) for k in range(32)]          # 0.05 .. ~50
    ess = [ess_shuf(S, mask, valid, t) for t in taus]
    drops = [(taus[i], ess[i] - ess[i + 1]) for i in range(len(ess) - 1) if ess[i + 1] < ess[i] - 1e-6]
    assert not drops, f'ESS not monotone in tau: {drops[:3]}'
    print(f'ok: ESS monotone in tau ({ess[0]:.4f} at tau=0.05 -> {ess[-1]:.4f} at tau={taus[-1]:.1f})')


def test_match_hits_target():
    E, mask, valid = scene()
    interior = wess_mod.interior_mask(mask, E.shape[-2:], ERODE)
    p_real, _, _ = wess_mod.wess_train_probabilities(E, valid, interior, HD, HD, tau=TAU_REF, lam=LAM)
    target = wess_mod._ess_frac(p_real)
    for seed in range(5):
        torch.manual_seed(100 + seed)
        S = Net._shuffle_energy(stand_in_match(True), E)
        tau, st = wess_mod.match_shuffled_tau(E, S, mask, valid, HD, HD, ref_tau=TAU_REF,
                                              lam=LAM, erode_cells=ERODE)
        got = ess_shuf(S, mask, valid, tau)
        fixed = ess_shuf(S, mask, valid, TAU_REF)
        assert st['wess_match_bound'] == 0.0, f'bound hit: {st}'
        assert abs(got - target) < 1e-3, f'ESS {got:.4f} vs target {target:.4f}'
        assert abs(st['wess_ess_target'] - target) < 1e-9
        print(f'ok: seed {seed}: target {target:.4f}, shuffled at tau=1 {fixed:.4f}, '
              f'matched tau {tau:.3f} -> {got:.4f}')


def test_match_consumes_no_rng():
    E, mask, valid = scene()
    torch.manual_seed(13)
    S = Net._shuffle_energy(stand_in_match(True), E)
    before = torch.get_rng_state().clone()
    wess_mod.match_shuffled_tau(E, S, mask, valid, HD, HD, ref_tau=TAU_REF, lam=LAM, erode_cells=ERODE)
    assert torch.equal(before, torch.get_rng_state()), 'match_shuffled_tau consumed torch RNG'
    print('ok: match consumes no RNG')


def test_logged_stats():
    E, mask, valid = scene()
    o = stand_in_match(match=True)
    torch.manual_seed(14)
    for _ in range(3):
        Net.sample_train_pixels(o, valid, N_SAMPLE, E=E, mask_hw=mask, H=HD, W=HD)
    keys = {'wess_ess_target', 'wess_tau_shuf', 'wess_match_resid', 'wess_match_bound'}
    for st in o._wess_stats_acc:
        assert keys <= set(st), f'missing match keys: {keys - set(st)}'
        assert abs(st['wess_ess'] - st['wess_ess_target']) < 1e-3, \
            f"logged wess_ess {st['wess_ess']:.4f} vs target {st['wess_ess_target']:.4f}"
    assert len(o._sample_weights_acc) == 3
    print('ok: match keys logged; logged wess_ess == target within 1e-3 '
          f"(e.g. {o._wess_stats_acc[0]['wess_ess']:.4f} vs {o._wess_stats_acc[0]['wess_ess_target']:.4f})")


def test_uniform_fallback():
    E, mask, valid = scene()
    torch.manual_seed(15)
    S = Net._shuffle_energy(stand_in_match(True), E)
    tau, st = wess_mod.match_shuffled_tau(E, S, mask, valid, HD, HD, ref_tau=TAU_REF,
                                          lam=1.0, erode_cells=ERODE)
    assert math.isnan(st['wess_tau_shuf']) and st['wess_match_resid'] == 0.0
    assert abs(st['wess_ess_target'] - 1.0) < 1e-6
    print('ok: lam=1 fallback: nothing to match')


if __name__ == '__main__':
    test_multiset_preserved()
    test_fresh_placement_each_draw()
    test_alignment_destroyed()
    test_no_fixed_spatial_prior()
    test_resume_reproducible()
    test_shape_for_batch_of_scenes()
    test_tau_selection()
    test_flag_off_equals_baseline()
    test_monotonic_in_tau()
    test_match_hits_target()
    test_match_consumes_no_rng()
    test_logged_stats()
    test_uniform_fallback()
    print('ALL PASS')
