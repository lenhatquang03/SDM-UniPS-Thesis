"""Self-test for tap_sites.py. Needs torch; no dataset and no checkpoint.

Run from the repo root on the training box before any screen:
    python sdm_unips/probes/wess/test_tap_sites.py          # GPU if available
    python sdm_unips/probes/wess/test_tap_sites.py --cpu    # beside a training run

Exits non-zero if any check fails.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..'))

import torch
import torch.nn.functional as F

from modules.model import model as model_mod
from modules.model import wess
from modules.model.wtconv import WTConv2d

import tap_sites

FAILED = []


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''), flush=True)
    if not ok:
        FAILED.append(name)


def main():
    use_cpu = '--cpu' in sys.argv or not torch.cuda.is_available()
    device = torch.device('cpu' if use_cpu else 'cuda')
    torch.manual_seed(0)
    print(f'device: {device}')

    x = torch.randn(2, 3, 16, 16)
    check('haar_dwt == WTConv2d._dwt', torch.equal(tap_sites.haar_dwt(x), WTConv2d(3, wt_levels=1)._dwt(x)))

    wt3 = WTConv2d(4, wt_levels=3)
    x = torch.randn(1, 4, 32, 32)
    got = {}
    handles = [wt3.wavelet_convs[l].register_forward_pre_hook(
        lambda _m, inp, l=l: got.__setitem__(l, inp[0].detach())) for l in range(3)]
    with torch.no_grad():
        wt3(x)
    for h in handles:
        h.remove()
    ll, ok = x, True
    for l in range(3):
        b = tap_sites.haar_dwt(ll)
        n, c, _, h, w = b.shape
        ok = ok and torch.allclose(got[l], b.reshape(n, 4 * c, h, w), atol=1e-6)
        ll = b[:, :, 0]
    check('wavelet_convs[l] input == raw-LL cascade (learned tap semantics)', ok)

    a = torch.rand(3, 8, 8)
    check('roll_up: single level returned unchanged', tap_sites.roll_up({0: a}, (0,)) is a)
    r = tap_sites.roll_up({0: torch.zeros(1, 4, 4), 1: torch.ones(1, 2, 2)}, (0, 1))
    check('roll_up: coarse level spread over its area (4^-l)', torch.allclose(r, torch.full((1, 4, 4), 0.5)))

    E = torch.rand(3, 16, 16) * torch.tensor([1., 10., 100.])[:, None, None]
    mk = torch.zeros(64, 64)
    mk[16:48, 16:48] = 1
    En = tap_sites.per_image_mean_normalize(E, mk)
    cells = F.adaptive_avg_pool2d(mk[None, None], (16, 16))[0, 0] >= 0.5
    check('per-image normalization: masked mean == 1', torch.allclose(En[:, cells].mean(1), torch.ones(3)))

    curve = [t / (1 + t) for t in tap_sites.TAU_GRID]
    tau, ok = tap_sites.tau_for_ess(curve, 0.5)
    check('tau_for_ess recovers tau=1 on ESS = tau/(1+tau)', ok and abs(tau - 1) < 0.05, f'tau {tau:.4f}')
    check('tau_for_ess flags an unreachable target', not tap_sites.tau_for_ess(curve, 0.999)[1])
    check('draw_seed stable', tap_sites.draw_seed(42, 3, 'S1|ess=0.388', 0)
          == tap_sites.draw_seed(42, 3, 'S1|ess=0.388', 0))

    print('building a random-init Net (a minute on CPU)...', flush=True)
    net = model_mod.Net(2048, device).to(device)
    net.eval()
    net.no_grad()
    H, N = 512, 2
    I = torch.rand(1, 3, H, H, N, device=device)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing='ij')
    M = ((((yy - 256) ** 2 + (xx - 256) ** 2) < 180 ** 2).float()[None, None]).to(device)
    n_imgs = torch.tensor([N])
    valid_ids = torch.nonzero(M[0].reshape(-1, H * H).permute(1, 0) > 0, as_tuple=False)[:, 0]
    gen = torch.Generator()
    gen.manual_seed(0)
    ids = wess.uniform_sample(valid_ids, 2048, gen)
    dec = torch.full((1, 1), H, dtype=torch.long, device=device)
    can = torch.full((1, 1), 256, dtype=torch.long, device=device)
    keys = tap_sites.learned_taps_needed(tap_sites.SITE_ORDER)
    with tap_sites.LevelTaps(net, keys) as taps, wess.SubbandTap(net, 0, 0, band='raw') as shipped, torch.no_grad():
        net(I, M, n_imgs.to(device), decoder_resolution=dec, canonical_resolution=can,
            training=True, sample_ids=ids[None])
        learned = tap_sites.learned_level_maps(taps, N, H // 256)
        shipped_S0 = wess.saliency_maps(shipped, n_imgs, H // 256, top_k=2)[0]
        raw = tap_sites.raw_level_maps(I, M, N, tap_sites.raw_max_level(tap_sites.SITE_ORDER))
    expect = {'S0': 64, 'S1': 256, 'S2': 128, 'S3': 64, 'S4': 256, 'S5': 32, 'S6': 16, 'S7': 64, 'S8': 64}
    for s, g in expect.items():
        E = tap_sites.site_map(s, learned, raw, M[0, 0], 2)
        check(f'{s} ({tap_sites.SITES[s]["desc"]}) is a finite {g}x{g} map',
              tuple(E.shape) == (g, g) and bool(torch.isfinite(E).all()), str(tuple(E.shape)))
        if s == 'S0':
            check('S0 == shipped wess.saliency_maps, bit for bit', torch.equal(E, shipped_S0))

    interior = tap_sites.shared_interior(M[0, 0], 2)
    check('shared interior == the mask wess_sample_train builds for S0',
          torch.equal(interior, wess.interior_mask(M[0, 0], shipped_S0.shape[-2:], 2)))
    p, _, _ = wess.wess_train_probabilities(tap_sites.site_map('S1', learned, raw, M[0, 0], 2),
                                            valid_ids, interior.reshape(-1), H, H, tau=1.0, lam=1.0)
    check('lam=1 gives ESS 1.0 on any site', abs(tap_sites.ess_fraction(p) - 1.0) < 1e-5,
          f'{tap_sites.ess_fraction(p):.6f}')

    print(f'\n{"ALL CHECKS PASSED" if not FAILED else str(len(FAILED)) + " FAILED: " + ", ".join(FAILED)}')
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
