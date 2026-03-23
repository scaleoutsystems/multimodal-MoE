# LiDAR-Only BEVFusion on ZOD — Debugging Log

This document records every change made to get the LiDAR-only TransFusion/BEVFusion
pipeline working on the custom ZOD pedestrian dataset, the reasoning behind each
change, and the evidence that motivated it.

---

## Context

We are training a LiDAR-only BEVFusion model (TransFusionHead) for single-class
pedestrian detection on the ZOD-MoE dataset. The model is pretrained on NuScenes
(10 classes, symmetric BEV range `[-54, -54, …, 54, 54]`) and fine-tuned on our
ZOD-derived dataset with an **asymmetric forward-only BEV range** `[0, -54, …, 108, 54]`.

The pipeline went through multiple debugging iterations. Each section below
describes one problem, how it was identified, and the exact fix.

---

## 1. Environment Compatibility Fixes

### 1a. mmcv version cap (`mmdet3d/__init__.py`)

**Problem**: MMDetection3D v1.4.0 caps mmcv at `< 2.2.0`, but our environment has
mmcv 2.2.x. The import fails at startup.

**Fix**: Bumped `mmcv_maximum_version` from `2.2.0` to `2.3.0`.

### 1b. Camera module imports (`bevfusion/__init__.py`)

**Problem**: BEVFusion's `__init__.py` unconditionally imports camera modules
(`DepthLSSTransform`, `ImageAug3D`) that require custom CUDA extensions
(`bev_pool_ext`). These are not compiled in our LiDAR-only setup.

**Fix**: Wrapped camera-specific imports in `try/except`. Only
`BEVFusionSparseEncoder` is imported unconditionally.

### 1c. Voxelization ops fallback (`bevfusion/ops/__init__.py`)

**Problem**: Same issue — BEVFusion's custom voxel CUDA extension may not be compiled.

**Fix**: Added `try/except` fallback to import `Voxelization` and `DynamicScatter`
from `mmcv.ops` instead.

---

## 2. Voxel Coordinate Ordering (`bevfusion/bevfusion.py`)

**Problem**: After voxelization, the mmcv `Voxelization` op returns coordinates in
`(batch, Z, Y, X)` order. `BEVFusionSparseEncoder` expects `(batch, Y, X, Z)`.
Without correction, the spatial dimensions are swapped throughout the sparse encoder,
producing a transposed BEV feature map. The detection head receives features where
X and Y are flipped, causing a shape mismatch.

**Evidence**: The model crashed with a tensor shape mismatch in the detection head
heatmap convolution.

**Fix**: Added `coords = coords[:, [0, 2, 3, 1]]` after voxelization to reorder
from `(batch, Z, Y, X)` to `(batch, Y, X, Z)`.

---

## 3. Empty Point Cloud Guard (`bevfusion/bevfusion.py`)

**Problem**: Some validation samples have zero LiDAR points after `PointsRangeFilter`
(e.g., edge-case frames). The `hard_voxelize_forward` CUDA kernel crashes with
`invalid configuration argument` on empty input.

**Fix**: Added a guard: if a sample's point cloud has 0 points, substitute a single
dummy zero-point before voxelization.

---

## 4. Heatmap Target X/Y Swap (`transfusion_head.py`, `get_targets_single`)

**Problem**: The original BEVFusion code swaps `(x, y)` indices when writing the GT
Gaussian onto the heatmap target:

```python
# Original code
draw_heatmap_gaussian(heatmap[cls_id], center_int[[1, 0]], radius)
```

For NuScenes with a **symmetric** range (`[-54, -54, …, 54, 54]`), both axes have
the same offset (54), so swapping X and Y produces the same projected index. The bug
is latent.

For our ZOD config with an **asymmetric** range (`[0, -54, …, 108, 54]`), X-offset
is 0 and Y-offset is 54. Swapping them places each GT Gaussian at the wrong feature
map location — errors of 10–60 m per box.

**Evidence**: Predictions collapsed to the BEV boundaries (x≈0, x≈108, y≈±54).
The heatmap head learned to fire at the boundary because that's where the
(incorrectly placed) targets were concentrated. A debug script that projected GT
centers onto the heatmap confirmed the mismatch.

**Fix**: Changed to `draw_heatmap_gaussian(heatmap[cls_id], center_int, radius)`,
using the un-swapped indices.

---

## 5. BEV Positional Encoding Channel Order (`transfusion_head.py`, `create_2D_grid`)

**Problem**: The `create_2D_grid` function generates a 2-channel positional encoding
(`bev_pos`) that is added to proposal queries. The original code ordered channels as
`[batch_x, batch_y]` but the `TransFusionBBoxCoder.encode` function encodes targets as
`target[0] = ix` (X-index), `target[1] = iy` (Y-index).

Since the transformer decoder computes `predicted_center = offset + query_pos`, and
the loss compares this against encoded targets, the positional encoding channels must
match the encode convention. The original ordering was inverted.

Additionally, `torch.meshgrid` without `indexing='ij'` uses `indexing='xy'` by default,
which further confuses the axis mapping when the grid is flattened in row-major order.

**Evidence**: The diagnostic showed that `query_pos` channels were swapped relative to
target centers, causing the offset prediction to fight the positional encoding rather
than complement it. After fixing, the predicted offsets became small and well-aligned.

**Fix**: Changed `torch.cat` order from `[batch_x, batch_y]` to `[batch_y, batch_x]`
and added `indexing='ij'` to `torch.meshgrid`.

---

## 6. Heatmap NMS Suppresses Pedestrian Peaks (`transfusion_head.py`, `forward_single`)

**Problem**: TransFusion uses a 3×3 max-pool NMS on the predicted heatmap before
selecting the top-K proposals. This suppresses any peak that has a stronger neighbor
within 3 cells. At our resolution (0.075 m × 8 = 0.6 m per cell), 3 cells = 1.8 m.

For dense pedestrian scenes (30–50 per frame), many pedestrians are within 1.8 m of
each other. Their heatmap peaks get killed by NMS, removing them from the proposal set
entirely. With `num_proposals=200`, only the surviving peaks are selected. Many GT
boxes have no nearby proposal.

Hungarian matching is forced to assign distant proposals to the uncovered GT, creating
large center regression targets (4–5 feature-map pixels) that the 1-layer transformer
decoder cannot learn.

The original BEVFusion code already handles this for NuScenes and Waymo:
- NuScenes: classes 8 (pedestrian) and 9 (traffic_cone) use `kernel_size=1` (no NMS)
- Waymo: classes 1 (pedestrian) and 2 (cyclist) use `kernel_size=1`

Our `dataset='custom_zod'` did not match either branch, so the default 3×3 NMS
was applied to pedestrians.

**Evidence**: A custom diagnostic script (`diagnose_overfit.py`) revealed:
- Center cx/cy regression error dominated bbox loss at 84% of total
- `|query_pos - target|` was ~4.5 pixels (proposals far from GT)
- `|pred - query_pos|` was ~0.3 pixels (offset learning was fine)
- Dense frames (37–52 GT) had matched IoU ~0.10; sparse frames (3–13 GT) had ~0.49
- Visualization showed many GT boxes with no nearby proposal

**Fix**: Added a `custom_zod` branch that bypasses NMS for class 0 (pedestrian)
using `kernel_size=1`:

```python
elif self.test_cfg['dataset'] == 'custom_zod':
    local_max[:, 0, ] = F.max_pool2d(
        heatmap[:, 0], kernel_size=1, stride=1, padding=0)
```

Also wrapped the entire NMS block in `if self.nms_kernel_size > 1:` to handle
the edge case where `nms_kernel_size=1` globally (which causes a zero-size tensor
slice with `padding=0`).

**How the NMS bypass mechanism works in detail**:

The config sets `nms_kernel_size=3` as the *default* for all classes. In
`forward_single`, the code first applies 3×3 max-pool NMS to the entire heatmap
(all class channels), keeping only local maxima. Then, dataset-specific `if/elif`
branches *override* specific class channels by rewriting them with the original
un-suppressed values using `kernel_size=1` (an identity operation):

```python
# Step 1: apply 3x3 NMS to ALL classes
local_max_inner = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=0)
local_max[:, :, padding:-padding, padding:-padding] = local_max_inner

# Step 2: override specific classes per dataset
if self.test_cfg['dataset'] == 'nuScenes':
    local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, ...)  # pedestrian
    local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, ...)  # traffic_cone
elif self.test_cfg['dataset'] == 'Waymo':
    local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, ...)  # pedestrian
    local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, ...)  # cyclist
elif self.test_cfg['dataset'] == 'custom_zod':
    local_max[:, 0, ] = F.max_pool2d(heatmap[:, 0], kernel_size=1, ...)  # pedestrian

# Step 3: zero out non-maxima
heatmap = heatmap * (heatmap == local_max)
```

The string `'custom_zod'` is set in both `train_cfg.dataset` and `test_cfg.dataset`
in the config file. It is just an identifier — it can be any string as long as the
config and the `if/elif` branches in `transfusion_head.py` match.

The `nms_kernel_size=3` default is kept in the config (rather than changing it to 1)
because: (a) the override pattern matches exactly what NuScenes and Waymo already do,
(b) if vehicle classes are added later they should still get 3×3 NMS, and (c) setting
`nms_kernel_size=1` globally caused a crash due to a zero-size tensor slice in the
padding logic.

**Result**: Center L1 error dropped from 4.5 to 0.1 pixels. Matched IoU rose from
0.26 to 0.62. Loss_bbox dropped 6.6× (1.38 → 0.21 at epoch 100).

---

## 7. Increased Proposal Count (`configs/zod/zod_lidar_only.py`)

**Problem**: With heatmap NMS bypassed, all heatmap values survive for the pedestrian
class. The top-K selection now picks from all ~32,400 pixels instead of just local
maxima. With `num_proposals=200`, the 200 highest-scoring pixels cluster around the
strongest few peaks, leaving many GT locations uncovered.

An initial attempt with `nms_kernel_size=1` (global NMS disable) and 200 proposals
showed 4× worse loss_bbox than the NMS=3 baseline, with gradient norms of 150–170
(vs 20 in the baseline). The proposals lacked spatial diversity.

**Fix**: Increased `num_proposals` from 200 to 500. With ~37 GT blobs per frame and
~12 high-value pixels per blob, 500 proposals capture most GT locations while
maintaining the spatial diversity benefits of NMS bypass.

**Result**: Combined with the NMS bypass (change 6), this produced stable training
with smooth loss convergence and matched IoU reaching 0.62.

---

## 8. Test-Time Circle NMS (`transfusion_head.py`, `predict_by_feat`)

**Problem**: With 500 proposals and heatmap NMS bypassed for pedestrians, many
overlapping predictions survive at inference time. Multiple proposals from the same
heatmap blob decode to nearly identical bounding boxes. Without post-processing NMS,
evaluation metrics are polluted by duplicate detections.

The existing code defines NMS tasks for `nuScenes` and `Waymo` datasets but not for
`custom_zod`. Setting `nms_type='circle'` in `test_cfg` without a task definition
would crash.

**Fix**: Added a `custom_zod` task definition with `radius=0.175` (matching the
NuScenes pedestrian circle NMS radius):

```python
elif self.test_cfg['dataset'] == 'custom_zod':
    self.tasks = [
        dict(num_class=1, class_names=['pedestrian'],
             indices=[0], radius=0.175),
    ]
```

Set `nms_type='circle'` in `test_cfg`. This only affects inference/evaluation, not
training.

---

## 9. Augmentation: Disabled Flips and Rotation (`configs/zod/zod_lidar_only.py`)

**Problem**: The BEV X-range is asymmetric: `[0, 108]` (forward-only). The standard
BEVFusion augmentation pipeline includes:

- `BEVFusionRandomFlip3D`: Flips in X map points to `[-108, 0]`, which is **entirely
  outside** the BEV range. After `PointsRangeFilter`, nearly all points are deleted.
  The training step sees an almost-empty frame — a wasted iteration.
- `GlobalRotScaleTrans` with `rot_range=[-0.785, 0.785]` (±45°): Large rotations
  move forward-facing points into negative X, partially losing data.

Y-flips would be safe (Y-range is symmetric `[-54, 54]`), but `BEVFusionRandomFlip3D`
flips in both X and Y randomly. Disabling it entirely is the safest approach.

**Fix**:
- Removed `BEVFusionRandomFlip3D` from the train pipeline
- Set `rot_range=[0, 0]` in `GlobalRotScaleTrans` (disabling rotation)
- Kept `scale_ratio_range=[0.9, 1.1]` (safe — scaling doesn't break the range)
- Kept `translation_std=0.5` (safe — small translations, and `PointsRangeFilter`
  clips any overflow)

---

## 10. Visualization Hooks (`mmdet3d/engine/hooks/bev_visualization_hook.py`)

**Added** two custom MMEngine hooks for training diagnostics:

- **`BEVFeatureVisualizationHook`**: At selected epochs, saves L2-norm heatmaps of
  (a) the sparse encoder output (before backbone) and (b) the FPN output (after
  backbone+neck). These show whether the feature extraction pipeline is producing
  spatially meaningful activations.

- **`BEVPredictionVisualizationHook`**: At selected epochs, overlays predicted
  bounding boxes (red) against GT boxes (green) on a BEV scatter plot of LiDAR
  points. Uses a configurable `score_thr` (set to 0.3) to filter low-confidence
  predictions.

Both hooks are registered in `mmdet3d/engine/hooks/__init__.py` and referenced by
type name in the config.

---

## Summary of All Changes

| # | File | Change | Category |
|---|------|--------|----------|
| 1a | `mmdet3d/__init__.py` | Bump mmcv version cap | Environment |
| 1b | `bevfusion/__init__.py` | Wrap camera imports in try/except | Environment |
| 1c | `bevfusion/ops/__init__.py` | Fallback to mmcv voxel ops | Environment |
| 2 | `bevfusion/bevfusion.py` | Reorder voxel coords (Z,Y,X) → (Y,X,Z) | Geometry bug |
| 3 | `bevfusion/bevfusion.py` | Guard against empty point clouds | Robustness |
| 4 | `transfusion_head.py` | Remove x/y swap in heatmap target placement | Geometry bug |
| 5 | `transfusion_head.py` | Fix positional encoding channel order | Geometry bug |
| 6 | `transfusion_head.py` | Bypass heatmap NMS for pedestrian class | Proposal coverage |
| 7 | `zod_lidar_only.py` | Increase num_proposals 200 → 500 | Proposal coverage |
| 8 | `transfusion_head.py` | Add test-time circle NMS for custom_zod | Evaluation |
| 9 | `zod_lidar_only.py` | Remove flips, disable rotation | Augmentation |
| 10 | `bev_visualization_hook.py` | Add BEV feature + prediction viz hooks | Diagnostics |

---

## Key Config Parameters (Final State)

```
voxel_size           = [0.075, 0.075, 0.2]
point_cloud_range    = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]
grid_size            = [1440, 1440, 40]
out_size_factor      = 8
num_proposals        = 500
nms_kernel_size      = 3  (bypassed for pedestrian class via custom_zod branch)
num_classes          = 1  (pedestrian only)
test nms_type        = 'circle' (radius=0.175)
augmentation         = scale [0.9, 1.1] + translation 0.5m (no flip, no rotation)
pretrained           = NuScenes LiDAR-only BEVFusion (strict=False)
lr                   = 1e-4  (cosine annealing, 20 epochs)
batch_size           = 2
```

---

## Overfit Experiment Results (Before/After NMS Fix)

| Metric (epoch 100) | Before (v3, NMS=3) | After (v4, NMS bypass + 500 proposals) |
|---------------------|---------------------|----------------------------------------|
| loss_bbox           | 1.38                | **0.21** (6.6× better)                 |
| matched_ious        | 0.29                | **0.57** (2× better)                   |
| cx L1 error         | 4.56 px             | **0.09 px** (50× better)               |
| cy L1 error         | 4.29 px             | **0.12 px** (36× better)               |
| max pred score      | 0.61                | **0.91**                               |
| total regression L1 | 10.81               | **0.88** (12× better)                  |
