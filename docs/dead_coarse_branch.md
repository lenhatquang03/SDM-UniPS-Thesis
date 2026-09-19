# The dead coarse branch: why UPerHead gets GroupNorm, a finite-explosion guard, and per-branch gradient logging

Written 2026-09-19. The change is applied identically on `modelA-vflip-mean-max-scale`, `modelB-wtconv` and
`modelC-wess`. Every number below was measured on this project's own runs during the debugging session,
unless it is marked as arithmetic or illustration.

## 0. Summary

| What | Where | Off switch (gives the old behaviour exactly) |
|---|---|---|
| `GroupNorm(32, 256)` between every conv and ReLU of the fusion head | `sdm_unips/modules/model/uper.py` | `UPerHead(..., norm=False)`, a constructor argument only; it is deliberately not a training flag |
| Skip an optimizer step whose finite gradient norm is > 100× the recent median | `modules/builder/trainer.py` (`is_grad_explosion`) | `--explode_factor 0` |
| Log the gradient norm of each of 13 branches (`gn_*`) and warn when one sits at exactly 0 | `modules/builder/trainer.py` (`BRANCH_PREFIXES`, `_branch_grad_norms`) | `--no_branch_grads` |

**Consequences:**
- Every checkpoint trained before this change lacks the GroupNorm parameters. It cannot be resumed or
  evaluated on these branches; both `resume_from` (strict load) and `builder.load_models` (partial-match
  guard) refuse it loudly. To evaluate an old checkpoint, check out the commit before the change:
  A `4701d5b`, B1 `22beea1`, B2 `319564b`.
- All variants have to be retrained. Their results are **not** comparable with any number produced before this
  change.

---

## 1. The symptom

Model B2 at `--wess_tau 0.5` trained normally for 60 epochs (best validation loss **0.0628** at epoch 58). Then:

| Optimizer step | Epoch | Loss | MAE (°) | Pre-clip gradient norm |
|---|---|---|---|---|
| 26350 | 60 | 0.158 | 19.5 | 2.51 |
| 26352 | 61 | 0.054 | 10.8 | — |
| 26360 | 61 | **2.24** | **99.2** | **299.7** |
| 26370 | 61 | 0.68 | 46.7 | 859 |
| 26380 | 61 | 0.56 | 40.2 | 3.9 × 10⁷ |
| 28390 | 65 | 0.61 | 42.4 | 4.0 × 10¹⁷ (run maximum) |

After step 26360, validation MAE froze at **44.2353 / 44.2447 / 44.2363 / 44.2364°** over four epochs. Only one
step, 28024, was ever skipped: there the total norm overflowed to `inf`. Every other norm was finite, so
`clip_grad_norm_` scaled it to 1.0 and the step was applied. Nothing aborted the run.

## 2. Terms used below

**ReLU and a "dead" ReLU.** `ReLU(z) = max(z, 0)`. Its derivative is 1 where `z > 0` and 0 where `z ≤ 0`.
For inputs `[-0.8, -0.3, 0.4]` the output is `[0, 0, 0.4]` and the local gradient is `[0, 0, 1]`. A ReLU
layer is *dead* when `z ≤ 0` at every position of every input. Its output is then all zeros and its local
gradient is all zeros, so **nothing upstream of it receives any gradient, whatever the loss is.**

**Adam's moments.** AdamW keeps two running averages per parameter element:
- `m = 0.9·m + 0.1·g` (average gradient);
- `v = 0.999·v + 0.001·g²` (average squared gradient).

It updates by `lr · m / (√v + 1e-8)`, with bias correction, which is ≈1 late in training. So the step size is
set by the *ratio* `m/√v`, not by how large the gradient is.

**GroupNorm (GN).** GN splits the channels into groups and, per sample and per group, subtracts the mean and
divides by the standard deviation taken over (channels in the group × all spatial positions). It then applies
a learned per-channel scale γ (initialised to 1) and shift β (initialised to 0). Here there are 256 channels in
32 groups, so each group normalizes 8 channels × H × W values. Unlike BatchNorm, it doesn't depend on the other
samples in the batch.

**PSP / PPM.** The Pyramid Pooling Module average-pools the coarsest feature map to 1×1, 2×2, 3×3 and 6×6, applies
a 1×1 conv + ReLU to each, and upsamples them back. **FPN** is the top-down path that adds coarse features into
finer ones.

**Residual stream.** In the pre-norm attention blocks (`transformer.MultiHeadSelfAttentionBlock`), every block
adds its output to the running tensor `O = O + fc_o2(...)`. There is no LayerNorm after the last block, so nothing
bounds that tensor's size.

## 3. Where the coarsest features go

**Shapes, derived once, at the real training size.** The canonical resolution is 256, and ConvNeXt downsamples
by 4 in the stem and by 2 before each later stage. One image therefore gives four feature maps:

| Scale | Channels × H × W | Light-axis attention | How it reaches the output |
|---|---|---|---|
| 0 | 96 × 64 × 64 | — | `lateral_convs[0]` → FPN |
| 1 | 192 × 32 × 32 | `fusion.comm.0` (2 blocks) | `lateral_convs[1]` → FPN |
| 2 | 384 × 16 × 16 | `fusion.comm.1` (3 blocks) | `lateral_convs[2]` → FPN |
| 3 | 768 × 8 × 8 | `fusion.comm.2` (5 blocks) | **only** `psp_forward` → `bottleneck` |

Scale 3 goes through the PPM (4 × 256 channels) and is concatenated with itself: 768 + 1024 = 1792 channels.
`bottleneck` (3×3 conv, 1792 → 256) + ReLU turns that into `laterals[3]` (256 × 8 × 8). The top-down path adds
`laterals[3]` into the three finer laterals. `fpn_bottleneck` concatenates all four (4 × 256 = 1024 channels,
upsampled to 64 × 64) and produces the GLC map (256 × 64 × 64) that the decoder samples from.

From the layer sizes (arithmetic, not a measurement), everything that reaches the loss only through
`bottleneck`'s ReLU comes to about **35M of the network's 63.8M parameters**:
- backbone stage 3 + its downsampling layer ≈ 15M;
- `comm.2` ≈ 15M;
- PPM + bottleneck ≈ 5M.

**Walk, toy size** (1 sample, 2 bottleneck output channels, 2 × 2 spatial, a 1×1 conv standing in for the 3×3;
real: 256 output channels, 8 × 8):

1. **Bottleneck pre-activation `z`.**
   - Role: the conv output that decides whether the coarse branch carries anything.
   - Code: `nn.Conv2d(1792, 256, 3, padding=1)` in `UPerHead.__init__`.
   - Output: the dead state that was measured, every value negative:
     `z[0] = [[-0.8, -0.3], [-1.2, -0.5]]`, `z[1] = [[-0.1, -0.6], [-0.4, -0.9]]`, shape `[1, 2, 2, 2]`.
   - Neighbors: fed by `cat([x, PPM(x)])`, consumed by the ReLU.
2. **ReLU (upstream code).**
   - Code: `nn.ReLU()`.
   - Input: `z` from step 1.
   - Output: `y = max(z, 0)` = all zeros, shape `[1, 2, 2, 2]`. The local gradient `1[z > 0]` is also all zeros.
   - Neighbors: consumed as `laterals[3]`.
3. **Top-down add and FPN concat.**
   - Code: `laterals[i-1] + F.interpolate(laterals[i])`, then `torch.cat(fpn_outs)`.
   - Input: `y` = 0. It adds nothing to the finer laterals and contributes 256 zero channels to the concat.
   - Backward: whatever gradient `dL/dy` arrives here, `dL/dz = dL/dy · 0 = 0`. The bottleneck conv, the PPM,
     `comm.2` and backbone stage 3 all receive exactly zero.

**The same `z` with GroupNorm** (toy: 1 group over the 2 channels × 4 positions = 8 values; real: 8 channels ×
64 positions = 512 values per group):

- mean = (−0.8 − 0.3 − 1.2 − 0.5 − 0.1 − 0.6 − 0.4 − 0.9) / 8 = −4.8 / 8 = **−0.6**
- deviations: −0.2, 0.3, −0.6, 0.1, 0.5, 0.0, 0.2, −0.3
- variance = (0.04 + 0.09 + 0.36 + 0.01 + 0.25 + 0 + 0.04 + 0.09) / 8 = 0.88 / 8 = **0.11**
- std = √(0.11 + 1e-5) = **0.3317**
- normalized (γ = 1, β = 0): −0.603, **0.905**, −1.809, **0.302**, **1.508**, 0.000, **0.603**, −0.905
- ReLU: 4 of the 8 values are positive, so the output and the gradient both pass.

GN subtracts the group mean on every forward pass. So however far the conv weights, the conv bias or the scale of
the incoming features drift, the values entering the ReLU stay centred on 0.

**What GN does not guarantee.** A channel can still be switched off by its own learned shift: here, β ≤ −1.51 with
γ = 1. Killing the whole branch would take all 256 channels' β parameters going that negative, and each β only
moves while its channel is active and receiving gradient. That makes it far less likely than upstream's failure,
where drift anywhere upstream was enough. It is not impossible, which is why the logging (section 7) stays on.

## 4. What was measured

**The branch was dead.** For each checkpoint, the table shows the largest per-tensor mean of Adam's `v` in the
group, divided by the median over all 469 parameter tensors. For a branch receiving no gradient, `v` shrinks
×0.999 per step, so after roughly 14,000 dead steps it falls below 10⁻⁶ × the median, which is the "DEAD" line.

| Run (checkpoint) | stage 3 | comm.2 | PPM | bottleneck | comm.1 (control) | Branch |
|---|---|---|---|---|---|---|
| B1 (ep 83) | 125 | 5.58 | 10.7 | 9.03 | 3.68 | alive |
| B2 ctrl λ=1 (ep 83) | 57.8 | 3.0 | 5.53 | 5.9 | 5.7 | alive |
| B2 λ=0 (ep 83) | 38 | 2.87 | 7.35 | 20.3 | 2.27 | alive |
| B2 τ=1 λ=0.25 (ep 83) | 47.5 | 3.17 | 6.59 | 12.6 | 3.04 | alive |
| A (`modelA_v2`, ep 83) | 2.4e-5 | 6.3e-4 | 2.3e-4 | 1.0e-2 | 8.62 | weak |
| B2 τ=2 (ep 36, running) | 5.0e-7 | 1.9e-5 | 4.1e-6 | 8.2e-5 | 11.8 | dying |
| B2 λ=0.5 (ep 83) | 6.5e-15 | 4.6e-12 | 1.6e-12 | 2.0e-10 | 5.7 | **dead** |
| B2 τ=0.5 (ep 58) | 1.2e-12 | 2.1e-11 | 8.7e-12 | 4.4e-10 | 12.2 | **dead** |

Death doesn't follow the sampler. It happened at the flattest WESS draw (λ=0.5, median ESS 0.63) and at the peakiest
(τ=0.5, 0.11), while λ=0 (0.34) and the uniform control stayed alive. A dead branch on its own doesn't collapse a
run: λ=0.5 stayed dead to epoch 83 and finished at validation loss 0.0432.

**The bottleneck ReLU is exactly dead**: the fraction of `bottleneck` outputs > 0, for two validation scenes × two
encoder calls (resized image, tiles):

| Checkpoint | Active fraction |
|---|---|
| τ=0.5, epoch 58 | 0, 0, 0, 0 |
| λ=0.5, best | 0, 0, 0, 0 |
| τ=0.5, epoch 61 (collapsed) | 0.243 on every call |
| τ=1 λ=0.25, best (alive control) | 0.030 on every call |

The alive control fires on only **3%** of positions. Even the healthy runs keep the coarse branch alive through a
thin sliver.

**The collapse is the revived branch blowing up.** One training-mode forward and backward pass (fp32, one fixed
validation scene) on the epoch-58 weights against the epoch-61 weights:

| Quantity | Epoch 58 | Epoch 61 |
|---|---|---|
| Total gradient norm | 2.2 | 2.52 × 10⁵ |
| Gradient norm of `comm.2` weights | exactly 0 | up to 1.3 × 10⁵ (the 12 largest in the network) |
| Spread of the decoder's LayerNorm inputs (14 of 60 layers) | 0.6 – 1.5 | 6.0 × 10³ – 1.2 × 10⁴ |
| Size of the regressor output before unit-normalization (median) | 6.29 | 4.85 × 10⁴ |
| R, the length of the mean predicted direction (1 = one normal everywhere) | 0.822 | 1.000 |

No token had a collapsed spread. So this is an activation **blow-up** entering the decoder from the encoder, not a
normalization dividing by a vanishing σ. The unit-normalization's input *grew*, so it shrinks gradients rather
than amplifying them.

**Why reviving a dead branch produces a blow-up (mechanism consistent with the data, not directly observed).**
After a long dead period, `v` for the branch is ≈ 0 (measured: comm.2 mean `v` of 2.3 × 10⁻²⁰ against a network
median of 1.1 × 10⁻⁹). When a gradient returns, Adam's `m/√v` starts large and stays large while the sign holds.
Illustration with lr = 1e-4 and a steady g = 1e-3:

| Step after revival | m | v | m / √v | Update per element |
|---|---|---|---|---|
| 1 | 1.00e-4 | 1.000e-9 | 3.16 | 3.2e-4 |
| 2 | 1.90e-4 | 1.999e-9 | 4.25 | 4.2e-4 |
| 3 | 2.71e-4 | 2.997e-9 | 4.95 | 5.0e-4 |

For comparison, take a parameter that has been training for a long time with a gradient flipping between +1e-3 and
−1e-3. Its steady state is |m| = 0.1·g / 1.9 = 5.3e-5 and v = 1e-6, so m/√v = 0.053. The revived branch therefore
takes ~60–95× larger steps, all at once and all in the same direction, across a 5-block, 768-wide attention stack
whose residual stream has no final LayerNorm.

Consistent with this, `comm.2`'s LayerNorm biases moved by ~0.02 between epochs 58 and 61, which is 3–5 × 10⁻⁴ per
step over a few dozen steps. What revived the ReLU in the first place is not known; the inputs to stage 3 keep
changing because stage 2 keeps training.

**Not our code.** `uper.py` was byte-identical on all three branches (git blob `ea096e7`) and identical to upstream
`origin_src/SDM-UniPS-CVPR2023`. Upstream copied mmsegmentation's UPerHead but replaced its `ConvModule`
(conv → norm → activation) with a bare conv → ReLU.

## 5. Why GroupNorm (option C) and not LeakyReLU (A: bottleneck only; B: whole head)

| | Branch can't die | No stale-`v` revival | Blow-up bounded at the head | Branch actually used |
|---|---|---|---|---|
| A: LeakyReLU(0.01) on the bottleneck | this branch only | yes | no | ~3% active, as before |
| B: LeakyReLU on all 12 head ReLUs | yes | yes | no | ~3% active, as before |
| **C: GroupNorm before all 12 ReLUs** | **yes** | **yes** | **yes**, each head layer's output is renormalized | **~50% active** |

- **C restores what mmsegmentation's design had.** It uses GroupNorm rather than BatchNorm, because the head sees
  (resized + tile) images of only a few scenes at a time.
- **Cost:** 12 GN layers × 512 parameters = 6,144 new parameters (arithmetic).
- **Not covered by any of the three:** the unbounded residual stream inside the `comm.*` stacks. C catches its
  blow-up one layer later, at the head. The guard in section 6 is the backstop.
- **Whether C improves MAE is untested.** Before this change, alive runs finished at validation loss 0.0414–0.0440
  and the dead λ=0.5 at 0.0432, which is within single-seed noise. The expectation of a gain rests only on the
  branch being used (~50% active instead of ~3%) and on this being the normalized head design this code was
  taken from.

## 6. The finite-explosion guard

**1. Formula.** For optimizer step t with pre-clip total gradient norm ‖g_t‖ (the L2 norm over all parameters,
unitless), and W the norms of the last 200 **applied** steps:

  r_t = ‖g_t‖ / median(W)

The step is skipped if r_t > F, with F = `--explode_factor` = 100. The guard is disarmed while |W| < 200, or if
the median is 0.

**2. Code** (`trainer.py`). The only departure from the formula is the arming condition:
```python
def is_grad_explosion(grad_norm, recent_norms, factor, window):
    if factor <= 0 or len(recent_norms) < window:   # off, or window not full yet
        return False
    med = statistics.median(recent_norms)
    return med > 0 and grad_norm > factor * med
# after the step: only APPLIED steps enter the window
if not skipped:
    self._norm_window.append(float(grad_norm))
```
A skipped step counts toward the existing abort guards: 20 consecutive skips, or > 30% of the last 200.

**3. Intuition.** The ratio rises when one step's gradient is far larger than the run's recent typical gradient.
Only applied steps enter W, so a run of exploding steps can't drag the median up and disarm the guard. Healthy
runs stay far below 100: the largest logged ratio across seven runs was **43.7** (B1, step 29610: norm 17.62
against a median of 0.404). Those logs sample 1 step in 10, so the true maximum may be somewhat higher.

**4. Worked example** (window of 5 instead of 200):
- W = [0.6, 0.8, 0.7, 0.9, 0.75] → sorted 0.6, 0.7, 0.75, 0.8, 0.9 → median **0.75**.
- Incoming norm 3.2: r = 3.2 / 0.75 = **4.3** ≤ 100 → applied. W becomes [0.8, 0.7, 0.9, 0.75, 3.2], median 0.8.
- Incoming norm 299.7, the τ=0.5 collapse's first logged value: r = 299.7 / 0.8 = **374.6** > 100 → skipped.
  W is unchanged.
- The collapse kept producing norms of 859, 3.9 × 10⁷, … which are all skipped. On the 20th consecutive skip the
  run aborts, and its last checkpoint still holds un-exploded weights.

## 7. Per-branch gradient norms

**1. Formula.** For a branch B, a set of parameter tensors:

  ‖g_B‖ = √( Σ_{p ∈ B} Σ_elements g_p² ) = √( Σ_{p ∈ B} ‖g_p‖² )

It is taken on the pre-clip, full effective-batch gradient, on every step that `train.py` logs (every
`--log_every`).

**2. Code** (`trainer.py`). float64, because norms of 10¹⁷ square past float32's limit:
```python
norms = [torch.linalg.vector_norm(p.grad, dtype=torch.float64)
         for p in params if p.grad is not None]
val = float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0
```
The 13 groups follow the scales through the head: `gn_stem`, `gn_stage0`–`gn_stage3`, `gn_comm0`–`gn_comm2`,
`gn_psp` (PPM + bottleneck), `gn_fpn`, `gn_glc_upsample`, `gn_glc_aggregation`, `gn_regressor`. A parameter matching
none of them is reported under `gn_other`. The start-up line prints each group's tensor count.

**3. Intuition.** A branch that is learning shows a positive norm that moves with training. A dead branch shows
**exactly 0.0**, which is how `gn_psp`, `gn_comm2` and `gn_stage3` would have read in B2 λ=0.5 and τ=0.5. After 20
consecutive logged zeros (~200 optimizer steps), the trainer prints `[branch-dead] <group> …` once. The epoch
summary averages each field (`avg_gn_psp`, …) like any other.

**4. Worked example.** A branch with two parameter tensors whose gradients are [3, 4] and [0, 12]:
- ‖g₁‖ = √(9 + 16) = 5
- ‖g₂‖ = √(0 + 144) = 12
- ‖g_B‖ = √(25 + 144) = √169 = **13**

If both gradients are zero, it reads 0.0, and 20 such logged steps in a row print the warning.

## 8. What to do differently from now on

- **Start new sessions for the re-run.** A `--resume auto` pointed at an old session would try to load a pre-change
  checkpoint and die on the strict load (loudly, but pointlessly).
- **Don't update a checkout that a pre-change run is still training from.** If that run crashes, `--resume` would
  import the new code and refuse its own checkpoints.
- **Treat all pre-change A/B numbers as superseded.** At minimum, the per-run branch states in section 4 are a
  confounder that has nothing to do with the variant.
- **Watch `gn_psp` / `gn_comm2`** early in each run, and `grad_exploded` (the per-epoch rate of skipped
  explosions) on every epoch summary.
