"""
Model C: deformable pixel attention (local context) + carrier-token regressor
(global context and coverage).

Design record: ~/reports/modelC_design/modelC_carrier_dat_design.md. Section
references below (D1..D10, C1..C11) point into it.

Why this exists. `Regressor.comm`, the spatial-axis transformer before the
prediction head, attends only among the pixels drawn for the step. A peaked
(WESS) draw therefore hands every pixel a clustered context. The two blocks
here dilute that effect without deleting the pixel-to-pixel interaction:

* `DeformablePixelAttention` -- each drawn pixel reads 16 deformed points of its
  OWN image's GLC map around itself (+-16 px at 512), so its neighbourhood
  context comes from the dense map, not from wherever the other samples landed.
  A hybrid: Deformable-DETR per-query reference points (the pixel itself) with
  DAT's dense offset network, dot-product MHSA and relative position bias.
* `CarrierRegressor` -- FasterViT's hierarchical attention. 64 carrier tokens,
  pooled from the dense map, first attend among themselves, then join the
  drawn pixels in one attention, then split. Pixels still attend to each other,
  but every key set also covers the whole object.

`Net` owns the call order; nothing here knows about sampling or the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from . import transformer


class DeformablePixelAttention(nn.Module):
    """Per-(pixel, image) deformable attention over that image's GLC map.

    Shapes (m drawn pixels, N images, map [256, h, w], G groups, P points):
      offset field   [N, G*P*2, h, w]   once per image per forward (D3)
      offsets        [m, N, G, P, 2]    field sampled at each pixel, in cells
      sampled keys   [m, N, P, 256]     group g's 64 channels read at its own
                                        P points, concatenated over groups (DAT)
      logits         [m, N, M, P]       q.k/sqrt(d) + bias(offset of head's group)
      output         [m, N, 256]        residual on glc_ids (D9)
    """

    def __init__(self, dim=256, dim_obs=3, num_heads=8, num_groups=4,
                 num_points=4, offset_range=4.0, dim_feedforward=1024,
                 attention_dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0 and dim % num_groups == 0
        assert num_heads % num_groups == 0   # DAT: M is a multiple of G (D4)
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_points = num_points
        self.offset_range = float(offset_range)
        self.head_dim = dim // num_heads
        self.group_dim = dim // num_groups
        self.heads_per_group = num_heads // num_groups

        # DAT Fig. 2b at stride 1: 5x5 depthwise -> GELU -> 1x1, the 1x1
        # without bias "to alleviate the compulsive shift for all locations".
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, num_groups * num_points * 2, 1, bias=False),
        )

        # Query sees the raw observation too (D7): LN over the 259-d concat.
        self.norm_q = nn.LayerNorm(dim + dim_obs)
        self.proj_q = nn.Linear(dim + dim_obs, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attention_dropout)

        # Relative position bias (D8): one table per head spanning the reachable
        # displacements, +-offset_range cells -> (2s+1)^2 taps, interpolated at
        # the continuous offset. Init as DAT (trunc normal, std 0.01).
        side = 2 * int(round(self.offset_range)) + 1
        self.rpe_table = nn.Parameter(torch.zeros(num_heads, 1, side, side))
        trunc_normal_(self.rpe_table, std=0.01)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim),
        )

    def offset_field(self, glc_n):
        """[N, 256, h, w] -> raw offset field [N, G*P*2, h, w]. Once per scene."""
        return self.offset_net(glc_n)

    def offsets_at(self, field, coords):
        """Sample the field at the pixels and bound it: [m, N, G, P, 2] in cells.

        `coords` is `ind2coords`' [1, 1, m, 2] grid, the same one `glc_ids` was
        read with, so offsets and features refer to the same position.
        """
        n = field.shape[0]
        raw = F.grid_sample(field, coords.expand(n, -1, -1, -1),
                            mode='bilinear', align_corners=False)     # [N, GP2, 1, m]
        m = raw.shape[-1]
        raw = raw.reshape(n, self.num_groups, self.num_points, 2, m)
        raw = raw.permute(4, 0, 1, 2, 3)                              # [m, N, G, P, 2]
        return self.offset_range * torch.tanh(raw)                    # D6: s * tanh

    def forward(self, glc_n, glc_ids, o_ids, coords, field):
        """glc_n [N, 256, h, w], glc_ids [m, N, 256], o_ids [m, N, 3],
        coords [1, 1, m, 2], field from `offset_field` -> [m, N, 256]."""
        n, _, h, w = glc_n.shape
        m = glc_ids.shape[0]
        G, P, M, d = self.num_groups, self.num_points, self.num_heads, self.head_dim

        delta = self.offsets_at(field, coords)                        # [m, N, G, P, 2] cells

        # Deformed points in grid_sample units: with align_corners=False the
        # normalized span 2 covers w (resp. h) cells.
        cell = torch.tensor([2.0 / w, 2.0 / h], device=delta.device, dtype=delta.dtype)
        ref = coords.reshape(m, 1, 1, 1, 2)
        pts = ref + delta * cell                                       # [m, N, G, P, 2]

        # Group g's channels are read at group g's points (DAT's offset groups).
        grid = pts.permute(1, 2, 0, 3, 4).reshape(n * G, m, P, 2)
        feats = glc_n.reshape(n * G, self.group_dim, h, w)
        sampled = F.grid_sample(feats, grid, mode='bilinear',
                                align_corners=False)                   # [N*G, 64, m, P]
        sampled = sampled.reshape(n, G, self.group_dim, m, P)
        sampled = sampled.permute(3, 0, 4, 1, 2).reshape(m, n, P, self.dim)

        q = self.proj_q(self.norm_q(torch.cat([o_ids, glc_ids], dim=2)))
        q = q.reshape(m, n, M, d)
        k = self.proj_k(sampled).reshape(m, n, P, M, d)
        v = self.proj_v(sampled).reshape(m, n, P, M, d)

        logits = torch.einsum('bnhd,bnphd->bnhp', q, k) / (d ** 0.5)  # [m, N, M, P]

        # Head h belongs to group h // (M/G); its keys sit at that group's
        # offsets, so that is the displacement its bias is read at. The query is
        # its own reference point, so the displacement IS the offset.
        rel = delta.repeat_interleave(self.heads_per_group, dim=2)     # [m, N, M, P, 2]
        rel = rel / self.offset_range                                  # [-1, 1] -> table edges
        rel = rel.permute(2, 0, 1, 3, 4).reshape(M, m * n, P, 2)
        bias = F.grid_sample(self.rpe_table.to(rel.dtype), rel, mode='bilinear',
                             align_corners=True)                       # [M, 1, m*N, P]
        logits = logits + bias.reshape(M, m, n, P).permute(1, 2, 0, 3)

        attn = self.attn_drop(torch.softmax(logits, dim=-1))
        out = torch.einsum('bnhp,bnphd->bnhd', attn, v).reshape(m, n, self.dim)

        x = glc_ids + self.proj_o(out)                                 # D9: residual
        return x + self.mlp(self.norm_mlp(x))


class CarrierInit(nn.Module):
    """FasterViT Eq. 3 carrier initialization, mask-weighted (C7).

    3x3 conv on each image's GLC map, then an average over each window of a
    fixed `grid x grid` partition in which every cell counts by its object
    coverage. The network input is `cat(I*M, M)`, so background cells hold the
    encoder's response to a zero image -- not zero -- and a plain average would
    dilute every window on the silhouette. RGB is pooled the same way at pixel
    resolution. Windows with no object pixels are dropped. Each carrier's
    position is the centroid of its window's object pixels, matching what its
    features describe.
    """

    def __init__(self, dim=256, grid=8):
        super().__init__()
        self.grid = grid
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, glc_n, img_n, mask_hw):
        """glc_n [N, 256, h, w], img_n [N, 3, H, W], mask_hw [H, W] in {0, 1}.

        Returns tokens [c, N, 259] (RGB first, as `_decode_pixels` concatenates)
        and object-pixel centroids [c, 2] as (x, y) in [-1, 1]; c <= grid^2.
        """
        g = self.grid # 8
        h, w = glc_n.shape[-2:] # (128, 128)
        m_pix = mask_hw.float()[None, None]                            # [1, 1, H, W], fp32 under AMP
        m_cell = F.adaptive_avg_pool2d(m_pix, (h, w))                  # coverage per cell, [1, 1, h, w]

        feat = self.conv(glc_n)                                        # [N, 256, h, w]
        num = F.adaptive_avg_pool2d(feat * m_cell, g)                  # [N, 256, g, g]
        den = F.adaptive_avg_pool2d(m_cell, g)                         # [1, 1, g, g]
        rgb_num = F.adaptive_avg_pool2d(img_n * m_pix, g)              # [N, 3, g, g]
        rgb_den = F.adaptive_avg_pool2d(m_pix, g)

        keep = (den > 0).reshape(-1)                                   # [g*g]
        feat_tok = (num / den.clamp_min(1e-12)).flatten(2).permute(2, 0, 1) # (N, 256, g*g).permute(2, 0, 1) = (g*g, N, 256)
        rgb_tok = (rgb_num / rgb_den.clamp_min(1e-12)).flatten(2).permute(2, 0, 1) # (g*g, N, 3)
        tokens = torch.cat([rgb_tok, feat_tok], dim=2)[keep]           # [c, N, 259]

        # Position = centroid of the window's OBJECT pixels, so it describes the
        # same pixels the mask-weighted features do. Equal to the window centre
        # when the window is fully covered; moves toward the object on the
        # silhouette. Pixel coordinates use `ind2coords`' convention (2*col/W - 1),
        # the one the drawn pixels' positions are embedded with.
        H, W = mask_hw.shape
        xs = torch.arange(W, device=m_pix.device, dtype=torch.float32) * 2 / W - 1
        ys = torch.arange(H, device=m_pix.device, dtype=torch.float32) * 2 / H - 1
        cx = F.adaptive_avg_pool2d(m_pix * xs.view(1, 1, 1, W), g)     # [1, 1, g, g]
        cy = F.adaptive_avg_pool2d(m_pix * ys.view(1, 1, H, 1), g)
        cx = cx / rgb_den.clamp_min(1e-12)
        cy = cy / rgb_den.clamp_min(1e-12)
        pos = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)[keep]   # [c, 2]
        return tokens, pos


class PositionMLP(nn.Module):
    """FasterViT's absolute position embedding: 2-layer MLP of normalized (x, y),
    added to pixel and carrier tokens alike (C9). Resolution-independent because
    the coordinates are."""

    def __init__(self, dim=384, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, xy):
        return self.mlp(xy)


class CarrierRegressor(nn.Module):
    """FasterViT hierarchical attention in place of `Regressor.comm` (C2-C11).

    Per layer: carriers self-attend (Eq. 4) -> pixels and carriers attend jointly
    (Eq. 5-6) -> split (Eq. 7). The last layer's carrier outputs would be
    discarded, so its joint step takes queries from the pixels only; with the
    block's pre-norm on queries and raw keys this is row-for-row the joint SAB.

    The blocks are the codebase's `MultiHeadSelfAttentionBlock` in the same
    configuration as today's `Regressor.comm` (Efficient mode, 8 heads, FFN 1024,
    ln=True), so the only change to the pixel interaction is what is in the key
    set.
    """

    def __init__(self, dim=384, num_layers=2, dim_feedforward=1024, num_heads=8,
                 attention_dropout=0.1):
        super().__init__()

        def block():
            return transformer.MultiHeadSelfAttentionBlock(
                dim, dim, num_heads, ln=True, attention_dropout=attention_dropout,
                dim_feedforward=dim_feedforward, attn_mode='Efficient')

        self.carrier_sab = nn.ModuleList([block() for _ in range(num_layers)])
        self.joint = nn.ModuleList([block() for _ in range(num_layers)])
        self.prediction_normal = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 3))

    def forward(self, x, carriers):
        """x [m, 384] pixel tokens; carriers [c, 384] (C0) or None -> [m, 3].

        `carriers` must be the scene's C0. Everything after it depends on this
        call's pixels, so nothing past C0 may be cached across eval chunks (C11).
        """
        x = x.unsqueeze(0)                                             # one set
        c = None if carriers is None or carriers.shape[0] == 0 else carriers.unsqueeze(0)
        n_layers = len(self.joint)
        for i in range(n_layers):
            if c is None:
                # No object windows (an all-background placeholder scene):
                # plain pixel self-attention, i.e. today's Regressor.comm.
                x = self.joint[i](x, x)
                continue
            c = self.carrier_sab[i](c, c)                              # Eq. 4
            if i < n_layers - 1:
                t = torch.cat([x, c], dim=1)
                t = self.joint[i](t, t)                                # Eq. 5-6
                x, c = t[:, :x.shape[1]], t[:, x.shape[1]:]            # Eq. 7
            else:
                x = self.joint[i](x, torch.cat([x, c], dim=1))
        return self.prediction_normal(x.squeeze(0))
