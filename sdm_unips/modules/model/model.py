
"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, trunc_normal_

from .model_utils import *
from . import transformer
from . import convnext
from . import uper
from ..utils import gauss_filter
from ..utils.ind2sub import *
from . import wess as wess_mod
from .decompose_tensors import *

class ImageFeatureExtractor(nn.Module):
    def __init__(self, input_nc):
        super(ImageFeatureExtractor, self).__init__()
        back = []

        ### ConvNexT backbone (from sctratch)
        out_channels = (96, 192, 384, 768)
        back.append(convnext.ConvNeXt(in_chans=input_nc, use_checkpoint=False))      
        self.backbone = nn.Sequential(*back)
        self.out_channels = out_channels

    def forward(self, x):
        feats = self.backbone(x) # glc[batch][scale]     
        return feats

class ImageFeatureFusion(nn.Module):
    def __init__(self, in_channels, use_efficient_attention=False):
        super(ImageFeatureFusion, self).__init__()
        self.fusion =  uper.UPerHead(in_channels = in_channels)
        
        attn = []
        self.num_comm_enc = [0,1,2,4]
   
        for i in range(len(in_channels)):
            if self.num_comm_enc[i] > 0:
                attn.append(
                    transformer.CommunicationBlock(
                        in_channels[i], num_enc_sab = self.num_comm_enc[i], 
                        dim_hidden=in_channels[i], 
                        ln=True, dim_feedforward = in_channels[i], 
                        use_efficient_attention=use_efficient_attention
                    )
                )
        self.comm = nn.Sequential(*attn)  
    
    def forward(self, glc, nImgArray):
        batch_size = len(nImgArray)
        sum_nimg = torch.sum(nImgArray)
        
        out_fuse = []
        attn_cnt = 0
        for k in range(len(glc)):
            if self.num_comm_enc[k] > 0:
                in_fuse = glc[k]
                _, C, H, W = in_fuse.shape # ((K+1) * sum(nImg), C, H, W)
                in_fuse = in_fuse.reshape(-1, sum_nimg, C, H, W).permute(0, 3, 4, 1, 2) # ((K+1), H, W, sum(nImg), C)
                K = in_fuse.shape[0] - 1
                in_fuse = in_fuse.reshape(-1, sum_nimg, C)
                feats = []
                ids = 0
                for b in range(batch_size):
                    feat = in_fuse[:, ids:ids+nImgArray[b], :]       
                    feat = self.comm[attn_cnt](feat)
                    feats.append(feat)
                    ids = ids + nImgArray[b]
                feats = torch.cat(feats, dim=1) # ((K+1)*H*W, sum(nImg), C)
                feats = feats.reshape(K+1, H*W, sum_nimg, C).permute(0, 2, 3, 1) # ((K+1), sum_nimg, C, H*W)
                feats = feats.reshape((K+1)*sum_nimg, C, H, W)
                out_fuse.append(feats)
                attn_cnt += 1
            else:
                out_fuse.append(glc[k])            
        out = self.fusion(out_fuse) 
        return out

# The Scale-Invariant Spatial-Light Image Encoder (SISL-IE) encodes the input image at the canonical resolution,
# and produces a Global Light-aware Context (GLC) feature map of the same spatial size as the input image. 
class ScaleInvariantSpatialLightImageEncoder(nn.Module): # image feature encoder at canonical resolution
    def __init__(self, input_nc, use_efficient_attention=False):
        super(ScaleInvariantSpatialLightImageEncoder, self).__init__()
        self.backbone = ImageFeatureExtractor(input_nc)
        self.fusion = ImageFeatureFusion(self.backbone.out_channels, use_efficient_attention=use_efficient_attention)
        self.feat_dim = 256

    def forward(self, x, nImgArray, canonical_resolution):
        N, C, H, W = x.shape        
        mosaic_scale = H // canonical_resolution
        K = mosaic_scale * mosaic_scale

        """ (1a) resizing x to (Hc, Wc)"""
        x_resized = F.interpolate(x, size= (canonical_resolution, canonical_resolution), mode='bilinear', align_corners=True)

        """ (1b) decomposing x into K x K of (Hc, Wc) non-overlapped blocks (stride)"""           
        x_grid = divide_tensor_spatial(x, block_size=canonical_resolution, method='tile_stride') # (B, K, C, canonical_resolution, canonical_resolution)
        x_grid = x_grid.permute(1,0,2,3,4).reshape(-1, C, canonical_resolution, canonical_resolution) # (K*B, C, canonical_resolutioin, canonical_resolution)
  
        """(2a) feature extraction """
        x = self.fusion(self.backbone(x_resized), nImgArray)
        f_resized = x.reshape(1, N, self.feat_dim, canonical_resolution//4 * canonical_resolution//4) # (1, N, C, canonical_resolution//4 * canonical_resolution//4)
        del x_resized

        """(2b) feature extraction """
        x = self.fusion(self.backbone(x_grid), nImgArray) # (K * N, C, canonical_resolution//4, canonical_resolution//4)
        x = x.reshape(K, N, x.shape[1], canonical_resolution//4, canonical_resolution//4)
        glc_grid = merge_tensor_spatial(x, method='tile_stride')
        del x_grid
       
        """ (3) upsample """
        glc_resized = F.interpolate(f_resized.reshape(N, self.feat_dim, canonical_resolution//4, canonical_resolution//4) , size= (H//4, W//4), mode='bilinear', align_corners=True)
        del f_resized

        glc = glc_resized + glc_grid
        return glc

 
class GLC_Upsample(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Upsample, self).__init__()       
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=dim_hidden, ln=True, dim_feedforward = dim_feedforward,use_efficient_attention=False)
       
    def forward(self, x):
        x = self.comm(x)        
        return x

class GLC_Aggregation(nn.Module):
    def __init__(self, input_nc, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Aggregation, self).__init__()              
        self.aggregation = transformer.AggregationBlock(dim_input = input_nc, num_enc_sab = num_agg_transformer, num_outputs = 1, dim_hidden=dim_aggout, dim_feedforward = dim_feedforward, num_heads=8, ln=True, attention_dropout=0.1, use_efficient_attention=use_efficient_attention)

    def forward(self, x):
        x = self.aggregation(x)      
        return x

class Regressor(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, use_efficient_attention=False, dim_feedforward=256):
        super(Regressor, self).__init__()
        # Communication among different samples (Pixel-Sampling Transformer)
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=input_nc, ln=True, dim_feedforward = dim_feedforward, use_efficient_attention=use_efficient_attention)
        self.prediction_normal = PredictionHead(input_nc, 3)

    def forward(self, x, num_sample_set):
        """Standard forward
        INPUT: img [Num_Pix, F]
        OUTPUT: [Num_Pix, 3]"""
        if x.shape[0] % num_sample_set == 0:
            x_ = x.reshape(-1, num_sample_set, x.shape[1])
            x_ = self.comm(x_)
            x = x_.reshape(-1, x.shape[1])
        else:
            ids = list(range(x.shape[0]))
            num_split = len(ids) // num_sample_set
            x_1 = x[:(num_split)*num_sample_set, :].reshape(-1, num_sample_set, x.shape[1])
            x_1 = self.comm(x_1).reshape(-1, x.shape[1])
            x_2 = x[(num_split)*num_sample_set:,:].reshape(1, -1, x.shape[1])
            x_2 = self.comm(x_2).reshape(-1, x.shape[1])
            x = torch.cat([x_1, x_2], dim=0)

        return self.prediction_normal(x)
    
class PredictionHead(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(PredictionHead, self).__init__()
        modules_regression = []
        modules_regression.append(nn.Linear(dim_input, dim_input//2))
        modules_regression.append(nn.ReLU())
        modules_regression.append(nn.Linear(dim_input//2, dim_output))
        self.regression = nn.Sequential(*modules_regression)

    def forward(self, x):
        return self.regression(x)

class Net(nn.Module):
    def __init__(self, pixel_samples, device,
                 wess_tau=wess_mod.DEFAULT_TAU, wess_lam=wess_mod.DEFAULT_LAM,
                 wess_erode_cells=wess_mod.DEFAULT_ERODE_CELLS,
                 wess_top_k=2):
        super().__init__()
        self.device = device
        self.pixel_samples = pixel_samples
        self.glc_smoothing = True

        # --- Model B2 (WESS) sampler configuration -------------------------
        # Hyperparameters of the *sampler*, not of the architecture, so they
        # live on the launch command like --lr does. `wess_lam = 1` reproduces
        # Model A's uniform draw exactly, which is the correctness check.
        self.wess_tau = float(wess_tau)
        self.wess_lam = float(wess_lam)
        self.wess_erode_cells = int(wess_erode_cells)
        self.wess_top_k = int(wess_top_k)
        # Populated by the sub-band hook during a training forward, read by
        # `sample_train_pixels`, cleared immediately after. Never a Parameter
        # and never in the state_dict, so checkpoints are unaffected.
        self._wess_capture = False
        self._wess_bands = None
        self._wess_stats_acc = []
        self.last_wess_stats = None


        self.input_dim = 4 # RGB + mask
        self.image_encoder = ScaleInvariantSpatialLightImageEncoder(self.input_dim, use_efficient_attention=False).to(self.device)

        self.input_dim = 3 # RGB only
        self.glc_upsample = GLC_Upsample(256+self.input_dim, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=True).to(self.device)
        self.glc_aggregation = GLC_Aggregation(256+self.input_dim, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False).to(self.device)


        self.regressor = Regressor(384, num_enc_sab=1, use_efficient_attention=True, dim_feedforward=1024).to(self.device)

        self._install_wess_hook()

    def _install_wess_hook(self):
        """Tap the level-0 Haar coefficients of stage 0, block 0.

        A forward PRE-hook on `wavelet_convs[0]`, whose input is exactly the raw
        `bands.reshape(n, 4C, h, w)` of the level-0 DWT -- the `X^LH/X^HL/X^HH`
        the method is defined on, before any learned parameter touches them.
        `WTConv2d.tap_subbands` cannot supply this: it stores the *filtered*
        bands (scaled by a trainable gain that drifts during training) and is
        last-writer-wins across all 12 blocks.

        Installed once, permanently, but gated on `self._wess_capture` so it is
        inert on every evaluation and inference forward -- and so B1's
        `best.pt` still loads here, since a hook adds no parameters and no
        buffers. The encoder calls the backbone twice; this fires on both and
        the tile path (`x_grid`, K*N maps) is second, so last-writer-wins is
        the wanted one. `saliency_maps` asserts the leading dimension rather
        than trusting that ordering.
        """
        dwconv = wess_mod.wtconv_block(self, wess_mod.DEFAULT_STAGE,
                                       wess_mod.DEFAULT_BLOCK)
        self._wess_channels = dwconv.channels

        def pre_hook(_module, inputs):
            if self._wess_capture:
                self._wess_bands = inputs[0].detach()

        dwconv.wavelet_convs[0].register_forward_pre_hook(pre_hook)

    def no_grad(self):
        mode_change(self.image_encoder, False)
        mode_change(self.glc_upsample, False)
        mode_change(self.glc_aggregation, False)
        mode_change(self.regressor, False)

    def with_grad(self):
        mode_change(self.image_encoder, True)
        mode_change(self.glc_upsample, True)
        mode_change(self.glc_aggregation, True)
        mode_change(self.regressor, True)


    def _decode_pixels(self, glc, I_dec, target, ids, num_imgs, H, W, C):
        """Decode predictions for the given pixel indices `ids` of one batch element."""
        o_ = I_dec[target, :, :, :].reshape(num_imgs, C, H * W).permute(2, 0, 1)  # [HW, N, C]
        o_ids = o_[ids, :, :]                                                    # [m, N, C]
        coords = ind2coords(np.array((H, W)), ids).expand(num_imgs, -1, -1, -1).to(self.device)
        glc_ids = F.grid_sample(glc[target, :, :, :], coords, mode='bilinear', align_corners=False)
        glc_ids = glc_ids.reshape(num_imgs, -1, len(ids)).permute(2, 0, 1)       # [m, N, F]

        x = torch.cat([o_ids, glc_ids], dim=2)
        glc_ids = self.glc_upsample(x)
        x = torch.cat([o_ids, glc_ids], dim=2)
        x = self.glc_aggregation(x)
        x_n = self.regressor(x, len(ids))
        # Gradient-safe unit normalization. F.normalize computes sqrt(sum(x^2))
        # then clamp_min(eps); for a near-zero predicted vector that yields
        # sqrt'(0)=inf times the clamped-region gradient 0 -> 0*inf = NaN, which
        # silently corrupts weights through backward (grad_clip can't fix a NaN).
        # Folding eps inside the sqrt keeps the gradient finite everywhere.
        norm = torch.sqrt((x_n * x_n).sum(dim=1, keepdim=True) + 1e-12)
        return x_n / norm


    def _wess_saliency(self, nImgArray, canonical_resolution):
        """Per-batch-element 64x64 energy maps, or None if the tap is empty.

        None is not an error path to be silenced -- it is what makes the WESS
        branch impossible to reach on an evaluation forward, where the tap is
        never armed. It also keeps this branch inert if the backbone is ever
        run without wavelet levels.
        """
        if self._wess_bands is None:
            return None
        # The encoder derives its mosaic factor the same way, from the encoder
        # input resolution; re-deriving it here from the captured tile count
        # would not distinguish K from N.
        mosaic_scale = self._wess_mosaic_scale
        try:
            energy = wess_mod.subband_energy(self._wess_bands, self._wess_channels)
            merged = wess_mod.merge_tile_energy(
                energy, int(sum(int(n) for n in nImgArray)), mosaic_scale)
        except RuntimeError as exc:
            raise RuntimeError(
                'WESS could not build a saliency map from the sub-band tap. '
                'This is a wiring fault, not a data fault -- training would '
                'otherwise silently fall back to a uniform draw and the run '
                f'would not be Model B2 at all. Original error: {exc}') from exc

        out, q = [], 0
        for n in (int(v) for v in nImgArray):
            out.append(wess_mod.reduce_over_lights(merged[q:q + n],
                                                   top_k=self.wess_top_k))
            q += n
        return out

    def _mean_wess_stats(self):
        """Average the per-scene sampler diagnostics over the micro-batch."""
        acc = getattr(self, '_wess_stats_acc', None)
        if not acc:
            return None
        keys = acc[0].keys()
        out = {}
        for k in keys:
            vals = [d[k] for d in acc if d[k] == d[k]]   # drop NaN
            out[k] = (sum(vals) / len(vals)) if vals else float('nan')
        return out

    def sample_train_pixels(self, valid_ids, n_sample,
                            E=None, mask_hw=None, H=None, W=None):
        """Choose which `n_sample` pixels a TRAINING step decodes.

        This is the pixel-sampling policy under study: the thesis' Models B/C
        replace it while everything else stays fixed. It is therefore
        deliberately the *only* place a training sample set is chosen, and it is
        deliberately never reached during evaluation -- `forward` takes an
        explicit `sample_ids`, which the val/test path always supplies. Without
        that split, a new sampler would silently change *which pixels the metric
        is computed on*, and a "better" val curve could be nothing more than an
        easier pixel draw.

        Baseline (Model A) = uniform over the mask, as in the paper: m random
        pixels without replacement, falling back to with-replacement only when a
        scene holds fewer than m valid pixels. That is still the body below, and
        it is what runs whenever `E` is unavailable.

        Model B2 (WESS) = the same draw with the interior's share of the budget
        reallocated by wavelet sub-band energy; see `wess.py`. The silhouette
        band keeps Model A's rate exactly, so the two samplers differ only in
        how they spend the *interior* budget.
        """
        if valid_ids.numel() == 0:
            # No valid pixels: emit a placeholder; loss masking discards them.
            return torch.zeros(n_sample, dtype=torch.long, device=valid_ids.device)

        if E is not None and mask_hw is not None:
            ids, stats = wess_mod.wess_sample_train(
                E, mask_hw, valid_ids, H, W, n_sample,
                tau=self.wess_tau, lam=self.wess_lam,
                erode_cells=self.wess_erode_cells)
            self._wess_stats_acc.append(stats)
            return ids

        if valid_ids.numel() >= n_sample:
            perm = torch.randperm(valid_ids.numel(), device=valid_ids.device)
            return valid_ids[perm[:n_sample]]
        rep = torch.randint(0, valid_ids.numel(), (n_sample,), device=valid_ids.device)
        return valid_ids[rep]

    def forward(self, I, M, nImgArray, decoder_resolution, canonical_resolution,
                training=False, sample_ids=None):
        """`sample_ids`: optional [B, m] long tensor of flat pixel indices at the
        decoder resolution. When supplied (evaluation) it overrides
        `sample_train_pixels` entirely, so every model variant is scored on
        exactly the same pixels; when None (training) the model picks its own.
        """

        decoder_resolution = decoder_resolution[0,0].cpu().numpy().astype(np.int32).item()
        canonical_resolution = canonical_resolution[0,0].cpu().numpy().astype(np.int32).item()

        """init"""
        B, C, H, W, Nmax = I.shape

        """ Image Encoder at Canonical Resolution """
        I_enc = I.permute(0, 4, 1, 2, 3)# B Nmax C H W
        M_enc = M # B 1 H W
        img_index = make_index_list(Nmax, nImgArray) # Extract objects > 0
        I_enc = I_enc.reshape(-1, I_enc.shape[2], I_enc.shape[3], I_enc.shape[4])
        M_enc = M_enc.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W)
        data = torch.cat([I_enc * M_enc, M_enc], dim=1)
        data = data[img_index==1,:,:,:] # torch.size([B, N, 4, H, W])d

        # Capture the sub-bands only on a training forward that will actually
        # draw its own pixels. Evaluation supplies `sample_ids`, so the tap
        # stays off there and val/test remain byte-identical to Model A's --
        # which is what the A/B fairness contract requires.
        self._wess_capture = bool(training and sample_ids is None)
        self._wess_bands = None
        self._wess_mosaic_scale = data.shape[-1] // canonical_resolution
        try:
            glc = self.image_encoder(data, nImgArray, canonical_resolution) # torch.Size([B, N, 256, H/4, W/4]) [img, mask]
        finally:
            self._wess_capture = False

        """ Sample Decoder at Original Resolution"""
        img = I.permute(0, 4, 1, 2, 3).to(self.device)
        mask = M

        decoder_imgsize = (decoder_resolution, decoder_resolution)
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        img = img[img_index==1, :, :, :]
        I_dec = F.interpolate(img, size=decoder_imgsize, mode='bilinear', align_corners=False)
        M_dec = F.interpolate(mask, size=decoder_imgsize, mode='nearest')

        C = img.shape[1]
        H = decoder_imgsize[0]
        W = decoder_imgsize[1]

        # Depth-wise Gaussian smoothing of the GLC, gated as the PAPER
        # describes: "Optionally, when P is larger than 4, we apply depth-wise
        # Gaussian filtering ... to the feature maps to further enhance the
        # interaction", where P = R/G = decoder_resolution / canonical_resolution
        # is the mosaic factor.
        f_scale = decoder_resolution // canonical_resolution   # P in the paper
        if self.glc_smoothing and f_scale > 4:
            smoothing = gauss_filter.gauss_filter(glc.shape[1], 10 * f_scale+1, 1).to(glc.device) # channels, kernel_size, sigma
            glc = smoothing(glc)

        if training:
            """Training path: decode exactly `pixel_samples` pixels per batch
            element, keep gradients, return only the sampled per-pixel
            predictions plus their indices.

            Which pixels: `sample_ids` when the caller supplied them
            (evaluation -- fixed across model variants), otherwise
            `sample_train_pixels`, the policy under study.
            """
            E_maps = self._wess_saliency(nImgArray, canonical_resolution)
            self._wess_stats_acc = []

            pred_n_list, idx_list = [], []
            p = 0
            for b in range(B):
                num_imgs = int(nImgArray[b])
                target = range(p, p + num_imgs)
                p = p + num_imgs

                if sample_ids is not None:
                    ids = sample_ids[b].to(device=I.device, dtype=torch.long)
                else:
                    m_ = M_dec[b, :, :, :].reshape(-1, H * W).permute(1, 0)
                    valid_ids = torch.nonzero(m_ > 0, as_tuple=False)[:, 0]
                    ids = self.sample_train_pixels(
                        valid_ids, self.pixel_samples,
                        E=None if E_maps is None else E_maps[b],
                        mask_hw=M_dec[b, 0], H=H, W=W)

                X_n = self._decode_pixels(glc, I_dec, target, ids, num_imgs, H, W, C)
                pred_n_list.append(X_n)
                idx_list.append(ids)

            self.last_wess_stats = self._mean_wess_stats()
            self._wess_bands = None

            pred_n = torch.stack(pred_n_list, dim=0)         # [B, n_sample, 3]
            sample_idx = torch.stack(idx_list, dim=0)        # [B, n_sample]
            return pred_n, sample_idx, (H, W)

        nout = torch.zeros(B, H * W, 3).to(self.device)

        p = 0
        for b in range(B):
            nimg_b = int(nImgArray[b]) if torch.is_tensor(nImgArray[b]) else int(nImgArray[b])
            target = range(p, p + nimg_b)
            p = p + nimg_b
            m_ = M_dec[b, :, :, :].reshape(-1, H * W).permute(1, 0)
            # Device-safe nonzero: works for both numpy arrays and CUDA/CPU tensors.
            if torch.is_tensor(m_):
                ids = torch.nonzero(m_ > 0, as_tuple=False)[:, 0]
                ids = ids[torch.randperm(ids.numel(), device=ids.device)]
            else:
                ids = np.nonzero(m_ > 0)[:, 0]
                ids = ids[np.random.permutation(len(ids))]
            n_ids = ids.numel() if torch.is_tensor(ids) else len(ids)
            if n_ids > self.pixel_samples:
                num_split = n_ids // self.pixel_samples + 1
                if torch.is_tensor(ids):
                    idset = list(torch.chunk(ids, num_split))
                else:
                    idset = np.array_split(ids, num_split)
            else:
                idset = [ids]

            for ids in idset:
                X_n = self._decode_pixels(glc, I_dec, target, ids, int(nImgArray[b]), H, W, C)
                nout[b, ids, :] = X_n.detach()

        return nout.permute(0, 2, 1).reshape(B, 3, H, W)


