"""This file is modified version from mmsegmentation (https://github.com/open-mmlab/mmsegmentation)"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# GroupNorm before every ReLU of the head (docs/dead_coarse_branch.md).
# Upstream took this head from mmsegmentation but dropped the norm that
# mmsegmentation's ConvModule puts between conv and ReLU. Without it the
# PSP `bottleneck` ReLU -- the ONLY route from backbone stage 3 / comm.2 to the
# loss -- can drift all-negative and stay dead, freezing ~35M parameters, and a
# later revival blew up Model B2 (tau=0.5) at epoch 61. GroupNorm re-centres
# each group's pre-activations on every forward, so the ReLU cannot go dead.
# `norm=False` rebuilds upstream's head exactly (same modules, same parameter
# names, same init); it exists for the equality test, not as a training switch.
GN_GROUPS = 32


def _act(channels, norm):
    """The activation tail of one conv block: [GroupNorm,] ReLU."""
    return [nn.GroupNorm(GN_GROUPS, channels), nn.ReLU()] if norm else [nn.ReLU()]


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, norm=True):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
                    *_act(self.channels, norm)
                    )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)

            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False)

            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.
    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels = (96, 192, 384, 768), channels = 256, pool_scales=(1, 2, 3, 6), norm=True):
        super(UPerHead, self).__init__()
        # PSP Module
        self.in_channels = in_channels
        self.channels = channels
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            norm=norm
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, kernel_size=3, padding=1),
            *_act(self.channels, norm))
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.channels, kernel_size=1, padding=0),
            *_act(self.channels, norm))


            fpn_conv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            *_act(self.channels, norm))

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels, kernel_size=3, padding=1),
            *_act(self.channels, norm))


    def psp_forward(self, inputs):
        """Forward function of PSP module."""

        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        """Forward function.
        inputs = {x_96, x_192, x_384, x_768}
        """

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size = prev_shape,
                mode='bilinear',
                align_corners = False
                )
        
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        
        # append psp feature
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)

        return output