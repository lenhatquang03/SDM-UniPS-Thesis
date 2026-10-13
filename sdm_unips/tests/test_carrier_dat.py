"""Checks for Model C (deformable pixel attention + carrier regressor).

What is being protected (design record:
~/reports/modelC_design/modelC_carrier_dat_design.md, section 6):

* Training forward at 512 runs for a uniform and a WESS draw, the backward is
  finite, and every new parameter receives a nonzero gradient.
* Offsets are bounded by s = 4 cells.
* Carriers: a mask covering part of the frame drops empty windows (c < 64),
  and a 1024 input still yields an 8x8 grid (c <= 64).
* Eval chunking: C0 is computed once per scene; decoding a chunk gives the
  same output whether or not another chunk was decoded first (nothing past C0
  leaks across chunks); a remainder-size chunk decodes.
* The last layer's pixels-only joint step equals the full joint SAB rows.
* An all-background placeholder scene (no carriers) still decodes.
* A full eval forward at 1024 runs; training forwards log `carrier_count`.

RUN ON THE TRAINING BOX ONLY. It builds the full Net and runs forward and
backward at 512 (batch 2); on the local WSL machine that exhausted memory and
took WSL down (2026-10-11). Uses CUDA when available.
Run: python sdm_unips/tests/test_carrier_dat.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from modules.model import model as model_mod       # noqa: E402
from modules.model import carrier_dat              # noqa: E402
from modules.model.model_utils import get_n_params  # noqa: E402

FAILED = []
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NEW_MODULES = ('dat.', 'carrier_init.', 'pos_embed.', 'regressor.')


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def disc_mask(H, r_frac=0.35):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing='ij')
    c = (H - 1) / 2
    return (((yy - c) ** 2 + (xx - c) ** 2) <= (r_frac * H) ** 2).float()


def inputs(H, K=3, B=1, seed=2, r_frac=0.35):
    g = torch.Generator().manual_seed(seed)
    I = torch.rand(B, 3, H, H, K, generator=g)
    M = disc_mask(H, r_frac)[None, None].expand(B, 1, H, H).contiguous()
    n_imgs = torch.full((B,), K)
    dec = torch.full((B, 1), H, dtype=torch.long)
    can = torch.full((B, 1), 256, dtype=torch.long)
    return I.to(DEV), M.to(DEV), n_imgs, dec, can


def test_param_count():
    torch.manual_seed(0)
    net = model_mod.Net(256, DEV)
    total = get_n_params(net)
    parts = {k: get_n_params(getattr(net, k.rstrip('.'))) for k in NEW_MODULES}
    b1 = 63_835_535   # B1 / B2 `Params =` line (context handoff, 2026-10-08)
    print(f'       Params = {total:,}  (B1 {b1:,}, delta {total - b1:+,})  '
          + '  '.join(f'{k}{v:,}' for k, v in parts.items()))
    check('param count printed', total > b1)


def test_training(wess_draw):
    torch.manual_seed(0)
    net = model_mod.Net(256, DEV, wess_lam=0.25 if wess_draw else 1.0)
    net.with_grad()
    I, M, n_imgs, dec, can = inputs(512, B=2)
    gt = torch.nn.functional.normalize(torch.randn(2, 3, 512, 512, device=DEV), dim=1)

    torch.manual_seed(7)
    pred, idx, _ = net(I, M, n_imgs, dec, can, training=True)
    tag = 'WESS' if wess_draw else 'uniform'
    check(f'[{tag}] training output shape [B, m, 3]', tuple(pred.shape) == (2, 256, 3))
    gt_s = torch.gather(gt.flatten(2), 2, idx[:, None, :].expand(-1, 3, -1)).permute(0, 2, 1)
    loss = ((pred - gt_s) ** 2).sum(-1).mean()
    loss.backward()
    check(f'[{tag}] loss finite', bool(torch.isfinite(loss)))

    no_grad, nonfinite = [], []
    for name, p in net.named_parameters():
        if not name.startswith(NEW_MODULES):
            continue
        if p.grad is None or float(p.grad.abs().sum()) == 0.0:
            no_grad.append(name)
        elif not bool(torch.isfinite(p.grad).all()):
            nonfinite.append(name)
    check(f'[{tag}] every new parameter gets a nonzero gradient', not no_grad, str(no_grad[:5]))
    check(f'[{tag}] new-parameter gradients finite', not nonfinite, str(nonfinite[:5]))
    if wess_draw:
        check('[WESS] the WESS sampler actually ran', net.last_wess_stats is not None)
    cs = net.last_carrier_stats
    check(f'[{tag}] carrier_count logged, 0 < count <= 64',
          cs is not None and 0 < cs['carrier_count'] <= 64,
          f'{None if cs is None else cs["carrier_count"]}')


def test_offsets_and_carriers():
    torch.manual_seed(0)
    net = model_mod.Net(256, DEV)
    net.no_grad()
    glc = torch.randn(3, 256, 128, 128, device=DEV) * 5      # large features: tanh saturates
    field = net.dat.offset_field(glc)
    ids = torch.randint(0, 512 * 512, (300,))
    coords = model_mod.ind2coords(torch.tensor([512, 512]).numpy(), ids).to(DEV)
    delta = net.dat.offsets_at(field, coords)
    check('offsets bounded by s = 4 cells', float(delta.abs().max()) <= 4.0,
          f'max |delta| = {float(delta.abs().max()):.3f}')
    check('offsets shape [m, N, G, P, 2]', tuple(delta.shape) == (300, 3, 4, 4, 2))

    img = torch.rand(3, 3, 512, 512, device=DEV)
    tok, pos = net.carrier_init(glc, img, disc_mask(512, 0.25).to(DEV))
    check('partial mask drops empty windows (c < 64)', 0 < tok.shape[0] < 64,
          f'c = {tok.shape[0]}')
    check('carrier tokens [c, N, 259], positions [c, 2] in [-1, 1]',
          tuple(tok.shape[1:]) == (3, 259) and tuple(pos.shape) == (tok.shape[0], 2)
          and float(pos.abs().max()) < 1)
    full = torch.ones(512, 512, device=DEV)
    tok_f, _ = net.carrier_init(glc, img, full)
    check('full mask keeps all 64 windows', tok_f.shape[0] == 64)
    tok_1k, _ = net.carrier_init(torch.randn(3, 256, 256, 256, device=DEV), torch.rand(3, 3, 1024, 1024, device=DEV),
                                 torch.ones(1024, 1024, device=DEV))
    check('1024 input still pools to an 8x8 grid', tok_1k.shape[0] == 64)
    # Mask-weighted mean: a window half object, half background must equal
    # the mean over its object cells only.
    m = torch.zeros(512, 512, device=DEV); m[:, :32] = 1          # left half of column-window 0
    feat = net.carrier_init.conv(glc)
    t, pos_half = net.carrier_init(glc, img, m)
    want = feat[:, :, :16, :8].mean(dim=(2, 3))       # window (0, 0), object cells
    check('mask-weighted pooling averages object cells only',
          torch.allclose(t[0, :, 3:], want, atol=1e-5))
    # Position = centroid of the object pixels (ind2coords convention): window
    # (0, 0) holds object columns 0..31 and rows 0..63, so
    # x = 2*15.5/512 - 1 = -0.939453, y = 2*31.5/512 - 1 = -0.876953
    # (the geometric window centre would be x = -0.875).
    want_pos = torch.tensor([2 * 15.5 / 512 - 1, 2 * 31.5 / 512 - 1], device=DEV)
    check('carrier position = centroid of its object pixels',
          torch.allclose(pos_half[0], want_pos, atol=1e-5),
          f'got {pos_half[0].tolist()}')


def test_eval_chunks():
    torch.manual_seed(0)
    net = model_mod.Net(1000, DEV)
    net.no_grad()
    I, M, n_imgs, dec, can = inputs(512)

    calls = {'n': 0}
    orig = net._scene_context

    def counting(*a, **k):
        calls['n'] += 1
        return orig(*a, **k)

    net._scene_context = counting
    with torch.no_grad():
        torch.manual_seed(3)
        out = net(I, M, n_imgs, dec, can)
    net._scene_context = orig
    n_valid = int(M.sum())
    check('eval output [B, 3, H, W], finite on the mask',
          tuple(out.shape) == (1, 3, 512, 512)
          and bool(torch.isfinite(out).all()))
    check('C0 / offset field computed once per scene across all chunks', calls['n'] == 1,
          f'{calls["n"]} calls for {n_valid // 1000 + 1} chunks')

    # Chunk independence beyond C0: same chunk, with and without another chunk
    # decoded before it, must give bitwise the same output.
    H = W = 512
    with torch.no_grad():
        data = torch.cat([I.permute(0, 4, 1, 2, 3).reshape(-1, 3, H, W) * M.expand(3, 1, H, W),
                          M.expand(3, 1, H, W)], dim=1)
        glc = net.image_encoder(data, n_imgs, 256)
        I_dec = I.permute(0, 4, 1, 2, 3).reshape(-1, 3, H, W)
        target = range(0, 3)
        ctx = net._scene_context(glc, I_dec, M, 0, target)
        valid = torch.nonzero(M[0, 0].reshape(-1) > 0)[:, 0]
        a, b = valid[:777], valid[777:2777]                    # remainder-size chunk a
        out_a1 = net._decode_pixels(glc, I_dec, target, a, 3, H, W, 3, ctx)
        net._decode_pixels(glc, I_dec, target, b, 3, H, W, 3, ctx)
        out_a2 = net._decode_pixels(glc, I_dec, target, a, 3, H, W, 3, ctx)
    check('remainder-size chunk decodes, unit normals',
          tuple(out_a1.shape) == (777, 3)
          and torch.allclose(out_a1.norm(dim=1), torch.ones(777, device=DEV), atol=1e-4))
    check('no state leaks across chunks (bitwise)', torch.equal(out_a1, out_a2))
    check('carriers present in this scene', ctx['carriers'] is not None,
          f'c = {0 if ctx["carriers"] is None else ctx["carriers"].shape[0]}')


def test_eval_1024():
    # Full inference forward at a 1024 crop (what realdata.load produces for
    # a large object): the offset field follows the 256^2 map, the carrier
    # grid stays 8x8. `--scalable` decodes 512 patches, which test_eval_chunks
    # already covers.
    torch.manual_seed(0)
    net = model_mod.Net(2048, DEV)
    net.no_grad()
    I, M, n_imgs, dec, can = inputs(1024, K=2)
    with torch.no_grad():
        out = net(I, M, n_imgs, dec, can)
    check('eval forward at 1024: output [1, 3, 1024, 1024], finite',
          tuple(out.shape) == (1, 3, 1024, 1024) and bool(torch.isfinite(out).all()))
    check('eval forward leaves no carrier stats', net.last_carrier_stats is None)


def test_last_layer_equivalence():
    torch.manual_seed(0)
    reg = carrier_dat.CarrierRegressor(384).to(DEV).eval()
    x = torch.randn(1, 2048, 384, device=DEV)
    c = torch.randn(1, 64, 384, device=DEV)
    blk = reg.joint[1]
    with torch.no_grad():
        t = torch.cat([x, c], dim=1)
        full = blk(t, t)[:, :2048]
        short = blk(x, t)
    check('pixels-only last joint step == joint SAB pixel rows', torch.allclose(full, short, atol=1e-5),
          f'max diff {float((full - short).abs().max()):.2e}')


def test_no_carriers():
    torch.manual_seed(0)
    reg = carrier_dat.CarrierRegressor(384).to(DEV).eval()
    with torch.no_grad():
        y = reg(torch.randn(100, 384, device=DEV), None)
        y0 = reg(torch.randn(100, 384, device=DEV), torch.zeros(0, 384, device=DEV))
    check('regressor without carriers decodes', tuple(y.shape) == (100, 3)
          and tuple(y0.shape) == (100, 3))


if __name__ == '__main__':
    test_param_count()
    test_offsets_and_carriers()
    test_last_layer_equivalence()
    test_no_carriers()
    test_eval_chunks()
    test_eval_1024()
    test_training(wess_draw=False)
    test_training(wess_draw=True)
    print('\nALL PASSED' if not FAILED else f'\n{len(FAILED)} FAILED: {FAILED}')
    sys.exit(1 if FAILED else 0)
