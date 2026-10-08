"""Checks for `--wess_iw` (importance-weighted WESS loss).

What is being protected:

* OFF is the existing B2. With the flag off no weights are produced, the draw
  is unchanged, and `normal_loss` returns exactly the pre-change value.
* OFF draws exactly as before: without replacement, same seed -> same pixels
  as a direct `torch.multinomial(p, m, replacement=False)`.
* ON draws WITH replacement, so E[count_i] = m * p_i exactly and the weighted
  loss is exactly unbiased for Model A's uniform mean. Checked two ways: the
  identity sum_i p_i * w_i * l_i == mean(l), and a Monte Carlo average of the
  real sampler's weighted loss over many draws.
* Rim weights are exactly 1, interior weights are capped at 1/lam, and
  lam = 1 gives weight 1 everywhere.
* Evaluation never gets weights: a forward with `sample_ids` leaves
  `last_sample_weights` None even with the flag on.

Needs torch; runs on CPU in well under a minute.
Run: python sdm_unips/tests/test_wess_iw.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from modules.loss import losses                    # noqa: E402
from modules.model import model as model_mod       # noqa: E402
from modules.model import wess                     # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def disc_mask(H, r_frac=0.4):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing='ij')
    c = (H - 1) / 2
    return (((yy - c) ** 2 + (xx - c) ** 2) <= (r_frac * H) ** 2).float()


def old_normal_loss(pred_n, gt_n_map, mask_map, sample_idx):
    """`normal_loss` exactly as it was before --wess_iw, for the equality check."""
    gt_n = losses._gather_pixels(gt_n_map, sample_idx)
    mask = losses._gather_pixels(mask_map, sample_idx)
    sq = ((pred_n - gt_n) ** 2).sum(dim=-1, keepdim=True)
    denom = mask.sum().clamp_min(1.0)
    return (sq * mask).sum() / denom


# ---------------------------------------------------------------------------
# 1. Sampler-level: weights, draw invariance, unbiasedness, bounds
# ---------------------------------------------------------------------------
def test_sampler():
    H = 512
    mask = disc_mask(H)
    valid_ids = torch.nonzero(mask.reshape(-1) > 0)[:, 0]
    g = torch.Generator().manual_seed(0)
    E = torch.rand(64, 64, generator=g) ** 4 * 10          # peaked energy map

    interior = wess.interior_mask(mask, E.shape, 2)
    p_ref, _, _ = wess.wess_train_probabilities(E, valid_ids, interior, H, H, tau=1.0, lam=0.25)
    torch.manual_seed(123)
    ids_off, _ = wess.wess_sample_train(E, mask, valid_ids, H, H, 2048, tau=1.0, lam=0.25)
    torch.manual_seed(123)
    ref = valid_ids[torch.multinomial(p_ref, 2048, replacement=False)]
    check('OFF: draw unchanged (no replacement, same seed -> same pixels)', torch.equal(ids_off, ref))
    check('OFF: no repeated pixels', ids_off.unique().numel() == 2048)

    torch.manual_seed(123)
    ids_on, stats, w = wess.wess_sample_train(E, mask, valid_ids, H, H, 2048,
                                              tau=1.0, lam=0.25, return_weights=True)
    torch.manual_seed(123)
    ref_on = valid_ids[torch.multinomial(p_ref, 2048, replacement=True)]
    check('ON: draw is with replacement (same seed -> same pixels as replacement=True)',
          torch.equal(ids_on, ref_on))
    n_rep = 2048 - ids_on.unique().numel()
    print(f'       ON repeated slots: {n_rep} of 2048')

    p, e, is_int = wess.wess_train_probabilities(E, valid_ids, interior, H, H, tau=1.0, lam=0.25)
    n = p.numel()
    w_all = 1.0 / (n * p.double())
    loss_px = torch.rand(n, generator=g, dtype=torch.float64)   # arbitrary per-pixel losses
    weighted = (p.double() * w_all * loss_px).sum()
    uniform = loss_px.mean()
    check('E_p[w * loss] == uniform mean (unbiased)',
          abs(float(weighted - uniform)) < 1e-9, f'{float(weighted):.9f} vs {float(uniform):.9f}')
    check('E_p[w] == 1', abs(float((p.double() * w_all).sum()) - 1.0) < 1e-9)

    rim_w = w_all[~is_int]
    check('rim weights == 1', torch.allclose(rim_w, torch.ones_like(rim_w), atol=1e-5),
          f'min {float(rim_w.min()):.6f} max {float(rim_w.max()):.6f}')
    check('interior weights <= 1/lam = 4', float(w_all[is_int].max()) <= 4.0 + 1e-4,
          f'max {float(w_all[is_int].max()):.4f}')
    check('drawn weights match 1/(n*p) of the drawn pixels',
          torch.allclose(w, (1.0 / (n * p[torch.searchsorted(valid_ids, ids_on)])).float()))
    print(f'       drawn weights: mean {float(w.mean()):.3f}  min {float(w.min()):.3f}  '
          f'max {float(w.max()):.3f}  (ESS {stats["wess_ess"]:.3f})')

    torch.manual_seed(0)
    _, _, w1 = wess.wess_sample_train(E, mask, valid_ids, H, H, 2048, tau=1.0, lam=1.0,
                                      return_weights=True)
    check('lam = 1 -> every weight is 1', torch.allclose(w1, torch.ones_like(w1), atol=1e-5))

    # Monte Carlo on the real sampler: average weighted loss over many draws
    # must match the uniform mean (ON), while the unweighted draw does not.
    Hs = 128
    ms = disc_mask(Hs)
    vs = torch.nonzero(ms.reshape(-1) > 0)[:, 0]
    Es = torch.rand(16, 16, generator=g) ** 4 * 10
    lpx = torch.rand(Hs * Hs, generator=g, dtype=torch.float64)    # fixed per-pixel loss
    target = float(lpx[vs].mean())
    torch.manual_seed(0)
    on, off = [], []
    for _ in range(3000):
        ids_s, _, ws = wess.wess_sample_train(Es, ms, vs, Hs, Hs, 256, tau=0.5, lam=0.25,
                                              return_weights=True)
        on.append(float((ws.double() * lpx[ids_s]).mean()))
        ids_u, _ = wess.wess_sample_train(Es, ms, vs, Hs, Hs, 256, tau=0.5, lam=0.25)
        off.append(float(lpx[ids_u].mean()))
    on_t, off_t = torch.tensor(on), torch.tensor(off)
    se_on = float(on_t.std() / len(on) ** 0.5)
    check('ON: Monte Carlo weighted loss == uniform mean (within 4 SE)',
          abs(float(on_t.mean()) - target) < 4 * se_on,
          f'{float(on_t.mean()):.5f} vs {target:.5f}, SE {se_on:.5f}')
    print(f'       OFF (unweighted, shifted target): {float(off_t.mean()):.5f} vs uniform {target:.5f}')


# ---------------------------------------------------------------------------
# 2. Loss-level: OFF is bit-identical, ones are bit-identical
# ---------------------------------------------------------------------------
def test_loss():
    g = torch.Generator().manual_seed(1)
    B, H, m = 2, 64, 300
    gt = torch.nn.functional.normalize(torch.randn(B, 3, H, H, generator=g), dim=1)
    mask = (torch.rand(B, 1, H, H, generator=g) > 0.3).float()
    pred = torch.nn.functional.normalize(torch.randn(B, m, 3, generator=g), dim=-1)
    idx = torch.randint(0, H * H, (B, m), generator=g)

    ref = old_normal_loss(pred, gt, mask, idx)
    check('normal_loss(weights=None) == pre-change loss (bitwise)',
          torch.equal(losses.normal_loss(pred, gt, mask, idx), ref))
    check('normal_loss(weights=1) == pre-change loss (bitwise)',
          torch.equal(losses.normal_loss(pred, gt, mask, idx, weights=torch.ones(B, m)), ref))
    w = torch.rand(B, m, generator=g) * 3
    gt_s = losses._gather_pixels(gt, idx); m_s = losses._gather_pixels(mask, idx)
    by_hand = ((((pred - gt_s) ** 2).sum(-1) * w * m_s[..., 0]).sum() / m_s.sum())
    check('normal_loss(weights=w) == sum(w*err*mask)/sum(mask)',
          torch.allclose(losses.normal_loss(pred, gt, mask, idx, weights=w), by_hand))


# ---------------------------------------------------------------------------
# 3. Net-level: OFF leaves the training forward and loss unchanged
# ---------------------------------------------------------------------------
def test_net():
    torch.manual_seed(0)
    dev = torch.device('cpu')
    net_off = model_mod.Net(256, dev, wess_iw=False)
    net_on = model_mod.Net(256, dev, wess_iw=True)
    net_on.load_state_dict(net_off.state_dict())
    net_off.eval(); net_on.eval()

    B, K, H = 1, 3, 512
    g = torch.Generator().manual_seed(2)
    I = torch.rand(B, 3, H, H, K, generator=g)
    M = disc_mask(H)[None, None]
    N = torch.nn.functional.normalize(torch.randn(B, 3, H, H, generator=g), dim=1)
    n_imgs = torch.tensor([K])
    dec = torch.full((B, 1), H, dtype=torch.long)
    can = torch.full((B, 1), 256, dtype=torch.long)

    with torch.no_grad():
        torch.manual_seed(7)
        pred_off, idx_off, _ = net_off(I, M, n_imgs, dec, can, training=True)
        torch.manual_seed(7)
        pred_on, idx_on, _ = net_on(I, M, n_imgs, dec, can, training=True)

    check('OFF: no weights produced', net_off.last_sample_weights is None)
    check('ON: weights produced with shape [B, n_sample]',
          net_on.last_sample_weights is not None
          and tuple(net_on.last_sample_weights.shape) == (B, 256))
    loss_off = losses.normal_loss(pred_off, N, M, idx_off, weights=net_off.last_sample_weights)
    check('OFF: training loss == pre-change loss (bitwise)',
          torch.equal(loss_off, old_normal_loss(pred_off, N, M, idx_off)))
    print(f'       loss OFF {float(loss_off):.6f}  loss ON '
          f'{float(losses.normal_loss(pred_on, N, M, idx_on, weights=net_on.last_sample_weights)):.6f}'
          f'  wess stats ON: { {k: round(v, 3) for k, v in net_on.last_wess_stats.items()} }')

    with torch.no_grad():
        net_on(I, M, n_imgs, dec, can, training=True, sample_ids=idx_on)
    check('ON + eval (sample_ids given): no weights', net_on.last_sample_weights is None)


if __name__ == '__main__':
    test_sampler()
    test_loss()
    test_net()
    print('\nALL PASSED' if not FAILED else f'\n{len(FAILED)} FAILED: {FAILED}')
    sys.exit(1 if FAILED else 0)
