"""Training losses for SDM-UniPS (normals only).

All losses are computed on the per-pixel samples returned by
`Net.forward(..., training=True)`. GT maps live at full image resolution
[B, C, H, W]; we gather them at the sample indices to align with predictions.
"""

import torch


def _gather_pixels(gt_map, sample_idx):
    """Pick per-pixel values from a [B, C, H, W] GT map at flat HW indices.

    Args:
        gt_map: tensor of shape [B, C, H, W].
        sample_idx: long tensor of shape [B, n_sample] holding flat (H*W) indices.

    Returns:
        tensor of shape [B, n_sample, C].
    """
    B, C, H, W = gt_map.shape
    flat = gt_map.reshape(B, C, H * W)
    idx = sample_idx.unsqueeze(1).expand(-1, C, -1)        # [B, C, n_sample]
    out = torch.gather(flat, 2, idx)                       # [B, C, n_sample]
    return out.permute(0, 2, 1).contiguous()               # [B, n_sample, C]


def normal_loss(pred_n, gt_n_map, mask_map, sample_idx):
    """MSE (L2) loss between predicted unit-normal and GT, masked (Sec. 4 of paper)."""
    gt_n = _gather_pixels(gt_n_map, sample_idx)            # [B, n_sample, 3]
    mask = _gather_pixels(mask_map, sample_idx)            # [B, n_sample, 1]
    sq = ((pred_n - gt_n) ** 2).sum(dim=-1, keepdim=True)
    denom = mask.sum().clamp_min(1.0)
    return (sq * mask).sum() / denom


def angular_error_deg(pred_n, gt_n_map, mask_map, sample_idx):
    """Mean angular error in degrees over the sampled pixels (for logging only)."""
    gt_n = _gather_pixels(gt_n_map, sample_idx)
    mask = _gather_pixels(mask_map, sample_idx).squeeze(-1)        # [B, n]
    dot = (pred_n * gt_n).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)   # [B, n]
    ang = torch.acos(dot) * (180.0 / 3.141592653589793)
    denom = mask.sum().clamp_min(1.0)
    return (ang * mask).sum() / denom
