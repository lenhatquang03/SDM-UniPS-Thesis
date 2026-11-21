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
"""
import sys
import types

import torch

sys.path.insert(0, __file__.rsplit('/sdm_unips/', 1)[0])
from sdm_unips.modules.model.model import Net  # noqa: E402

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


if __name__ == '__main__':
    test_multiset_preserved()
    test_fresh_placement_each_draw()
    test_alignment_destroyed()
    test_no_fixed_spatial_prior()
    test_resume_reproducible()
    test_shape_for_batch_of_scenes()
    test_tau_selection()
    print('ALL PASS')
