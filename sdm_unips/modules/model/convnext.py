# This file is modified version from the original convnext
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wtconv import WTConv2d


def derive_wt_levels(canonical_resolution, min_subband=4, max_levels=3):
    """Per-stage DWT depth for a backbone fed at `canonical_resolution`.

    The encoder always hands the backbone a `canonical_resolution` square --
    `x_resized` is interpolated to it and `x_grid`'s tiles are exactly it (see
    `ScaleInvariantSpatialLightImageEncoder.forward`). So stage resolutions are
    fixed at R/4, R/8, R/16, R/32 = 64/32/16/8 for the default R=256, and the
    level budget can be settled at construction rather than guessed.

    A level is only worth spending while its sub-bands stay at least
    `min_subband` wide: at 256 a flat `wt_levels=3` would give stage 3 a 1x1
    sub-band, where a 5x5 depthwise kernel is all padding around a single real
    tap -- parameters and FLOPs bought for nothing.

    This is an architectural decision, not a training knob, so it is settled
    here rather than exposed on the command line. `ConvNeXt`'s default is
    `derive_wt_levels(256)` == (3, 3, 2, 1), matching the 256 that `train.py`,
    `main.py` and `eval_diligent.py` all default `--canonical_resolution` to.
    Training at a different canonical resolution means editing that default.

    Returns a 4-tuple, e.g. (3, 3, 2, 1) at R=256.
    """
    levels = []
    for i in range(4):
        stage_res = canonical_resolution // (4 * 2 ** i)
        n = 0
        while n < max_levels and stage_res // (2 ** (n + 1)) >= min_subband:
            n += 1
        levels.append(max(1, n))
    return tuple(levels)


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                 wt_levels=3, wt_kernel_size=5):
        super().__init__()
        # Model B1: the depthwise operator is a wavelet convolution rather than
        # the baseline's 7x7 spatial kernel. It is shape-preserving
        # ([N,C,H,W] -> [N,C,H,W]), so the rest of this block and everything
        # downstream of the backbone is untouched. The attribute keeps the name
        # `dwconv` so the surrounding code reads unchanged.
        self.dwconv = WTConv2d(dim, kernel_size=wt_kernel_size, wt_levels=wt_levels)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# @BACKBONES.register_module()
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], use_checkpoint=False,
                 wt_levels=derive_wt_levels(256), wt_kernel_size=5
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if len(wt_levels) != 4:
            raise ValueError(f'wt_levels must have 4 entries (one per stage), got {wt_levels}')
        self.wt_levels = tuple(wt_levels)
        self.wt_kernel_size = wt_kernel_size
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value,
                wt_levels=self.wt_levels[i], wt_kernel_size=wt_kernel_size) for j in range(depths[i])]
            )


            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    
    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)            
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
