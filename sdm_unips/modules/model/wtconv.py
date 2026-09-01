"""Wavelet Convolution (WTConv) -- Model B's replacement for ConvNeXt's 7x7 dwconv.

Reference: Finder et al., "Wavelet Convolutions for Large Receptive Fields",
ECCV 2024. This is a clean reimplementation (no `pywt` dependency) specialised
to the Haar / `db1` basis, which is the one the thesis specifies.

The variant is selected by the **branch**, not by a flag: on `modelB-wtconv`
`convnext.Block` builds this layer unconditionally, and Model A is reached by
checking out Model A's branch. There is deliberately no runtime switch --
`train.py` carries training hyperparameters only, and the architecture is not
one of them.

Why it is a *drop-in*: `WTConv2d` maps [N, C, H, W] -> [N, C, H, W], exactly as
`nn.Conv2d(dim, dim, 7, padding=3, groups=dim)` does. Nothing downstream of
`convnext.Block.dwconv` can observe the swap, so the UPerHead fusion, the
light-axis attention, the 128x128 GLC merge and the 2048-pixel decode are all
untouched by construction.

The idea: instead of one large spatial kernel (which acts as a low-pass filter
and smooths exactly the intensity gradients photometric stereo reads normals
from), cascade a Haar DWT. Each level halves the resolution and splits into
four sub-bands {LL, LH, HL, HH}; a small depthwise kernel is applied to all
four *independently*, so a 5x5 kernel at level 3 covers 40x40 input pixels
while low and high frequencies are never averaged together. The levels are
then folded back with the inverse transform (IWT) and added to a full-
resolution base convolution.

Two deliberate departures from the reference implementation:

* **The Haar filters are buffers, not `nn.Parameter(requires_grad=False)`.**
  `Net.with_grad()` (via `model_utils.mode_change`) does a blanket
  `for param in net.parameters(): param.requires_grad = True`, so a
  non-trainable Parameter would be silently promoted to trainable and swept
  into the AdamW parameter list -- destroying the fixed orthonormal basis the
  whole method rests on, with no error and no log line. Buffers are not in
  `.parameters()`, so they are immune.
* **`wt_levels` is per stage** (see `convnext.derive_wt_levels`). The encoder
  always feeds the backbone at `--canonical_resolution` (256 by default), so
  stage resolutions are a fixed 64/32/16/8 and a flat `wt_levels=3` would put
  stage 3 at a 1x1 sub-band, where a 5x5 depthwise kernel is all padding and
  one real tap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# The four separable 2-D Haar filters of Eq. (1.1). Orthonormal: each has unit
# Frobenius norm and they are mutually orthogonal, so the IWT built from the
# same taps is an exact inverse (verified in tests/test_wtconv_shapes.py).
#
# Index 0 is LL (the low-pass band the cascade recurses on); 1..3 are the
# detail bands. Which of the two mixed bands is called "LH" versus "HL" is a
# row-first/column-first naming convention and differs between references; it
# does not matter here, because every consumer -- the depthwise convolutions
# below, and Model B2's saliency energy -- treats bands 1..3 symmetrically.
_HAAR_2D = [
    [[ 1.0,  1.0], [ 1.0,  1.0]],   # LL
    [[ 1.0, -1.0], [ 1.0, -1.0]],   # LH
    [[ 1.0,  1.0], [-1.0, -1.0]],   # HL
    [[ 1.0, -1.0], [-1.0,  1.0]],   # HH
]


def haar_filters(channels, dtype=torch.float32):
    """Depthwise Haar analysis/synthesis banks for `channels` channels.

    Returns a single [4*C, 1, 2, 2] tensor usable as the weight of both
    `F.conv2d(..., stride=2, groups=C)` and
    `F.conv_transpose2d(..., stride=2, groups=C)`.

    Channel ordering matters: `weight.repeat(C, 1, 1, 1)` tiles the 4-filter
    block, so with `groups=C` output channel `4*c + k` is filter `k` applied to
    input channel `c`. That is what makes the `[N, C, 4, H/2, W/2]` reshape in
    `forward` correct -- get it wrong and the sub-bands interleave across
    channels silently.
    """
    base = 0.5 * torch.tensor(_HAAR_2D, dtype=dtype).unsqueeze(1)  # [4, 1, 2, 2]
    return base.repeat(channels, 1, 1, 1)                          # [4C, 1, 2, 2]


class _ChannelScale(nn.Module):
    """Learnable per-channel gain, as in the reference implementation.

    The wavelet branches are initialised at 0.1 so the layer starts out close
    to its plain-depthwise base convolution and the sub-band contributions grow
    only if they earn it -- which matters when training from scratch, since the
    detail bands are high-variance early on.
    """

    def __init__(self, channels, init=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), float(init)))

    def forward(self, x):
        return self.weight * x


class WTConv2d(nn.Module):
    """Depthwise wavelet convolution. Shape-preserving: [N,C,H,W] -> [N,C,H,W].

    Args:
        channels: input == output channel count (the layer is depthwise).
        kernel_size: spatial kernel applied within each sub-band (5 in the paper).
        wt_levels: number of cascaded DWT levels.
        bias: bias on the full-resolution base convolution, matching the
            `nn.Conv2d` it replaces (which uses the default `bias=True`).
    """

    def __init__(self, channels, kernel_size=5, wt_levels=3, bias=True):
        super().__init__()
        if wt_levels < 1:
            raise ValueError(f'wt_levels must be >= 1, got {wt_levels}')
        self.channels = channels
        self.wt_levels = wt_levels
        pad = kernel_size // 2

        # Fixed orthonormal basis -- buffers, never parameters. See module docstring.
        filters = haar_filters(channels)
        self.register_buffer('wt_filter', filters)
        self.register_buffer('iwt_filter', filters.clone())

        # Full-resolution branch: this is what preserves the plain-depthwise
        # behaviour the block had before, so the wavelet branches are additive
        # refinements rather than a wholesale replacement.
        self.base_conv = nn.Conv2d(channels, channels, kernel_size,
                                   padding=pad, groups=channels, bias=bias)
        self.base_scale = _ChannelScale(channels, init=1.0)

        # One depthwise conv per level, over all 4C sub-band channels at once.
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(4 * channels, 4 * channels, kernel_size,
                      padding=pad, groups=4 * channels, bias=False)
            for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ChannelScale(4 * channels, init=0.1) for _ in range(wt_levels)
        ])

        # --- Model B2 (WESS) tap -------------------------------------------
        # Off by default, so Model B1 holds no extra tensors and its memory
        # profile is unaffected. When enabled, `subbands` holds the level-0
        # detail bands {LH, HL, HH} as [N, C, 3, H/2, W/2], DETACHED -- the
        # saliency draw is an index selection and carries no gradient, and
        # keeping the live tensor would pin the whole encoder graph.
        #
        # Note for B2: `ScaleInvariantSpatialLightImageEncoder.forward` calls
        # the backbone twice, on `x_resized` (N images) and then on `x_grid`
        # (4N tiles). The tile call is last, so this attribute holds the TILE
        # path's sub-bands after a forward -- which is the one WESS wants.
        # Assert on `subbands.shape[0]` rather than relying on that ordering.
        self.tap_subbands = False
        self.subbands = None

    def _dwt(self, x):
        """[N, C, H, W] -> [N, C, 4, H/2, W/2]. H, W must be even."""
        n, c, h, w = x.shape
        y = F.conv2d(x, self.wt_filter, stride=2, groups=c)
        return y.reshape(n, c, 4, h // 2, w // 2)

    def _iwt(self, x):
        """[N, C, 4, H, W] -> [N, C, 2H, 2W]."""
        n, c, _, h, w = x.shape
        y = x.reshape(n, 4 * c, h, w)
        return F.conv_transpose2d(y, self.iwt_filter, stride=2, groups=c)

    def forward(self, x):
        ll = x
        ll_levels, hi_levels, shapes = [], [], []

        # --- analysis: descend the cascade, filtering every sub-band ---
        for i in range(self.wt_levels):
            shapes.append(ll.shape)
            # Odd sizes cannot be halved; pad by one and crop on the way back
            # up. Does not arise at --canonical_resolution 256 (64/32/16/8 are
            # all powers of two) but a non-power-of-two canonical resolution
            # would hit it, and a silent shape error here is expensive to find.
            if ll.shape[-2] % 2 or ll.shape[-1] % 2:
                ll = F.pad(ll, (0, ll.shape[-1] % 2, 0, ll.shape[-2] % 2))

            bands = self._dwt(ll)                                 # [N,C,4,h,w]
            n, c, _, h, w = bands.shape

            # The next level descends on the RAW low-pass band, not the
            # convolved one. That keeps the cascade a true multi-resolution
            # decomposition of the input -- every level sees the original
            # signal band-limited, rather than the accumulated output of the
            # levels above it. (Recursing on the filtered LL instead turns the
            # pyramid into a serial chain of convolutions and loses the
            # "frequencies are processed independently" property the method is
            # built on.) The filtered LL is still what the synthesis pass
            # consumes, below.
            ll = bands[:, :, 0, :, :]

            filtered = self.wavelet_scale[i](
                self.wavelet_convs[i](bands.reshape(n, 4 * c, h, w))
            ).reshape(n, c, 4, h, w)

            if self.tap_subbands and i == 0:
                self.subbands = filtered[:, :, 1:4, :, :].detach()

            ll_levels.append(filtered[:, :, 0, :, :])
            hi_levels.append(filtered[:, :, 1:4, :, :])

        # --- synthesis: fold back up, accumulating into the coarser LL ---
        acc = 0
        for i in range(self.wt_levels - 1, -1, -1):
            ll_i = ll_levels.pop() + acc
            hi_i = hi_levels.pop()
            shape_i = shapes.pop()
            acc = self._iwt(torch.cat([ll_i.unsqueeze(2), hi_i], dim=2))
            acc = acc[:, :, :shape_i[-2], :shape_i[-1]]   # undo any odd-size pad

        return acc + self.base_scale(self.base_conv(x))
