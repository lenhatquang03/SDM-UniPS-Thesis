"""Shape and correctness probe for Model B1's WTConv backbone.

The claim B1 rests on is that WTConv is a *drop-in* for ConvNeXt's 7x7
depthwise convolution: same input shape, same output shape, therefore nothing
downstream of `convnext.Block.dwconv` can observe the swap. Reading code is a
weak way to check that; this file checks it by construction.

What is being protected:

* `WTConv2d` is shape-preserving at every stage geometry the encoder can
  produce, so the UPerHead fusion, the light-axis attention, the 128x128 GLC
  merge and the 2048-pixel decode all keep working untouched.
* The Haar bank is orthonormal, so IWT(DWT(x)) == x. If this drifts, the layer
  is quietly lossy and every "spectral decomposition" claim in the thesis is
  wrong.
* The Haar filters are BUFFERS. `Net.with_grad()` does a blanket
  `param.requires_grad = True` over `net.parameters()`, so a filter stored as a
  non-trainable Parameter would be silently promoted to trainable and swept
  into the AdamW parameter list -- no error, no log line, and the fixed basis
  the method depends on would drift during training.
* The full backbone still emits the four stage tensors UPerHead expects.
* The full backbone stays a drop-in replacement, so `model.py`, `builder.py`
  and `train.py` are byte-identical to Model A's branch and the variant lives
  entirely in `convnext.py` + `wtconv.py`.

Unlike `test_scene_check_roots.py` this one DOES need torch, so it runs on the
training box, not on a laptop.

Run: python sdm_unips/tests/test_wtconv_shapes.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from modules.model import convnext                              # noqa: E402
from modules.model.wtconv import WTConv2d, haar_filters         # noqa: E402


CANONICAL = 256
# (channels, spatial) per ConvNeXt-T stage when the encoder feeds the backbone
# a CANONICAL-square. Both encoder paths do: `x_resized` is interpolated to it
# and `x_grid`'s tiles are exactly it, so these four geometries are the only
# ones WTConv ever sees.
STAGES = [(96, 64), (192, 32), (384, 16), (768, 8)]

_failures = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name}  {detail}')
        _failures.append(name)


def test_haar_orthonormal():
    print('haar basis')
    f = haar_filters(1).reshape(4, 4)          # [4 filters, 4 taps]
    gram = f @ f.t()
    check('orthonormal (Gram == I)',
          torch.allclose(gram, torch.eye(4), atol=1e-6),
          f'gram=\n{gram}')


def test_perfect_reconstruction():
    print('DWT/IWT round-trip')
    for c, res in STAGES:
        layer = WTConv2d(c, kernel_size=5, wt_levels=1)
        x = torch.randn(2, c, res, res)
        rt = layer._iwt(layer._dwt(x))
        check(f'IWT(DWT(x)) == x  @ C={c} {res}x{res}',
              torch.allclose(rt, x, atol=1e-5),
              f'max abs err {(rt - x).abs().max().item():.3e}')


def test_subband_channel_order():
    """Band 0 must be LL. The `[N, C, 4, h, w]` reshape in `_dwt` assumes the
    repeat-tiled filter bank puts filter k of channel c at output 4c+k; if that
    assumption breaks, sub-bands interleave across channels silently and B2's
    saliency energy would be computed on the wrong tensor."""
    print('sub-band channel ordering')
    layer = WTConv2d(3, kernel_size=3, wt_levels=1)
    x = torch.ones(1, 3, 8, 8)
    bands = layer._dwt(x)                       # [1, 3, 4, 4, 4]
    check('LL of a constant image is non-zero', bands[:, :, 0].abs().min() > 0.9,
          f'{bands[:, :, 0].abs().min().item():.3e}')
    check('detail bands of a constant image are zero',
          bands[:, :, 1:].abs().max() < 1e-6,
          f'{bands[:, :, 1:].abs().max().item():.3e}')


def test_shape_preserving():
    print('WTConv2d is shape-preserving at every stage geometry')
    levels = convnext.derive_wt_levels(CANONICAL)
    for (c, res), lv in zip(STAGES, levels):
        layer = WTConv2d(c, kernel_size=5, wt_levels=lv)
        x = torch.randn(2, c, res, res)
        y = layer(x)
        check(f'C={c} {res}x{res} levels={lv}: {tuple(x.shape)} -> {tuple(y.shape)}',
              y.shape == x.shape)
        check(f'C={c} finite output', torch.isfinite(y).all())


def test_derive_wt_levels():
    print('derived level budget')
    check('at 256 -> (3, 3, 2, 1)',
          convnext.derive_wt_levels(256) == (3, 3, 2, 1),
          str(convnext.derive_wt_levels(256)))
    # No level may drive a sub-band below 4x4 (a 5x5 kernel on a 1x1 map is all
    # padding around a single real tap).
    for r in (128, 192, 256, 384, 512):
        for i, lv in enumerate(convnext.derive_wt_levels(r)):
            smallest = (r // (4 * 2 ** i)) // (2 ** lv)
            check(f'R={r} stage{i}: smallest sub-band {smallest}x{smallest} >= 4 '
                  f'(or forced minimum)', smallest >= 4 or lv == 1)


def test_odd_sizes():
    """Not reachable at --canonical_resolution 256, but a non-power-of-two
    canonical resolution would hit the padding path, and a silent shape error
    there is expensive to find."""
    print('odd spatial sizes')
    layer = WTConv2d(8, kernel_size=3, wt_levels=2)
    for res in (7, 13, 15):
        x = torch.randn(1, 8, res, res)
        y = layer(x)
        check(f'{res}x{res} preserved', y.shape == x.shape, str(tuple(y.shape)))


def test_filters_are_buffers():
    print('Haar filters are buffers, not parameters')
    layer = WTConv2d(16, kernel_size=5, wt_levels=2)
    names = [n for n, _ in layer.named_parameters()]
    check('wt_filter not in parameters()', not any('wt_filter' in n for n in names),
          str(names))
    check('iwt_filter not in parameters()', not any('iwt_filter' in n for n in names),
          str(names))

    # The real hazard: mode_change's blanket promotion.
    from modules.model.model_utils import mode_change
    mode_change(layer, True)
    check('wt_filter still not trainable after mode_change(True)',
          not layer.wt_filter.requires_grad)
    check('iwt_filter still not trainable after mode_change(True)',
          not layer.iwt_filter.requires_grad)


def test_backbone_stage_outputs():
    """The whole drop-in claim, end to end: the backbone must still emit the
    four tensors UPerHead consumes, at the shapes Model A produced. If this
    holds, nothing above `convnext.py` needs to know the variant exists --
    which is why `model.py`, `builder.py` and `train.py` carry no B1 code."""
    print('full ConvNeXt backbone')
    x = torch.randn(2, 4, CANONICAL, CANONICAL)
    expect = [(2, c, r, r) for c, r in STAGES]
    net = convnext.ConvNeXt(in_chans=4)
    net.eval()
    with torch.no_grad():
        outs = net(x)
    got = [tuple(o.shape) for o in outs]
    check(f'stage shapes {got}', got == expect, f'expected {expect}')

    # The default must be the derived budget, not a flat constant.
    check(f'default wt_levels == derive_wt_levels(256)',
          net.wt_levels == convnext.derive_wt_levels(256), str(net.wt_levels))
    check('every block uses WTConv',
          all(isinstance(b.dwconv, WTConv2d) for st in net.stages for b in st))


def test_param_cost():
    """Informational, not a pass/fail: WTConv's cost lands on a backbone that
    runs 5N times per scene (N images resized + 4N tiles), so the multiplier
    matters more than the raw delta."""
    print('parameter cost (informational)')
    net = convnext.ConvNeXt(in_chans=4)
    total = sum(p.numel() for p in net.parameters())
    dwp = sum(p.numel() for n, p in net.named_parameters() if 'dwconv' in n)
    print(f'  WTConv backbone total={total:,}  depthwise={dwp:,}')
    # Model A's depthwise cost, for reference: per block, one 7x7 kernel and one
    # bias per channel. Computed rather than built, since Model A's branch is
    # not importable from here.
    a_dwp = sum(n * (7 * 7 * d + d)
                for d, n in zip((96, 192, 384, 768), (3, 3, 9, 3)))
    print(f'  Model A depthwise (computed)={a_dwp:,}  ratio={dwp / a_dwp:.2f}x')


if __name__ == '__main__':
    torch.manual_seed(0)
    for fn in (test_haar_orthonormal, test_perfect_reconstruction,
               test_subband_channel_order, test_shape_preserving,
               test_derive_wt_levels, test_odd_sizes, test_filters_are_buffers,
               test_backbone_stage_outputs, test_param_cost):
        fn()
    print()
    if _failures:
        print(f'FAILED ({len(_failures)}): {_failures}')
        sys.exit(1)
    print('all shape checks passed')
