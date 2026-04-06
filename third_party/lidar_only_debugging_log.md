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
(`bev_pool_ext`). These are not compiled in our setup (see 1c for why).

**Fix**: Wrapped camera-specific imports in `try/except`. Only
`BEVFusionSparseEncoder` is imported unconditionally.

### 1c. Voxelization ops fallback (`bevfusion/ops/__init__.py`)

**Problem**: BEVFusion ships its own custom CUDA extensions for voxelization
(`voxel_layer`) and camera BEV pooling (`bev_pool_ext`). These are **not** built
by the normal `pip install -e .` of mmdetection3d. They require a separate
compilation step:

```bash
cd projects/BEVFusion && python setup.py develop
```

This `setup.py` builds both extensions together in a single invocation. We chose
not to run it because:

1. The LiDAR-only pipeline does not need `bev_pool_ext` (camera BEV pooling).
2. The `setup.py` hardcodes CUDA arch flags (`sm_70` through `sm_86`), which may
   not match every deployment GPU without manual editing.
3. mmcv already provides functionally equivalent `Voxelization` and
   `DynamicScatter` ops that are pre-compiled in the mmcv wheel.

**Fix**: Added `try/except` fallback: if BEVFusion's custom `voxel_layer` is not
compiled, import `Voxelization` and `DynamicScatter` from `mmcv.ops` instead.

**Side effect**: This substitution changes the coordinate order of the voxelization
output (see Change 2 below). The two ops are functionally equivalent but return
coordinates in different column order.

---

## 2. Voxel Coordinate Ordering (`bevfusion/bevfusion.py`)

**This change is a direct consequence of Change 1c** (falling back to mmcv's
voxelization op). It has nothing to do with the asymmetric BEV range.

**Problem**: BEVFusion's custom `voxel_layer` returns voxel coordinates in
`(batch, Y, X, Z)` order — the order that `BEVFusionSparseEncoder` expects.
mmcv's `Voxelization` returns them in `(batch, Z, Y, X)` order. Without
correction, Z-dimension values (range 0–40) end up in the Y slot and Y-dimension
values (range 0–1439) end up in the Z slot. Since the sparse tensor shape is
`[1440, 1440, 41]`, a Y-coordinate value like 800 placed in the Z dimension
(max 41) causes an out-of-bounds crash.

**Why this would NOT happen in upstream BEVFusion**: The original code uses its
own `voxel_layer` extension, which already outputs coordinates in the order the
sparse encoder expects. No reorder is needed. We only need the reorder because
we substituted mmcv's op (Change 1c), which uses a different convention.

**Evidence**: The model crashed with an out-of-bounds / shape mismatch error in
the sparse encoder.

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

**Fix**: Increased `num_proposals` from 200 to 500. Each GT pedestrian creates a
Gaussian blob of ~12 significant pixels on the heatmap. With ~37 GT per frame,
~444 pixels have significant values. Top-200 captures only 45% of these, leaving
many GT blobs with zero proposals. Top-500 captures nearly all of them, giving
every GT blob multiple nearby proposals for Hungarian matching.

**Critical: changes 6 and 7 are inseparable.** Neither works alone:
- NMS bypass alone (200 proposals): top-200 from the un-suppressed heatmap clusters
  around the few strongest peaks. Many GT blobs get 0 proposals. Training diverges
  (loss_bbox ~10, grad_norm ~170).
- More proposals alone (500 proposals, NMS=3 kept): suppressed peaks remain at zero
  regardless of how many proposals are selected. The top-500 after NMS still can't
  include a peak that was zeroed out.
- Both together: NMS bypass ensures peaks survive; 500 proposals ensures enough
  capacity to cover all ~444 significant pixels across ~37 GT blobs.

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

**Added** three custom MMEngine hooks for training diagnostics:

- **`BEVFeatureVisualizationHook`**: At selected epochs, saves L2-norm heatmaps of
  (a) the sparse encoder output (before backbone) and (b) the FPN output (after
  backbone+neck). These show whether the feature extraction pipeline is producing
  spatially meaningful activations.

- **`BEVPredictionVisualizationHook`**: At selected epochs, overlays predicted
  bounding boxes (red) against GT boxes (green) on a BEV scatter plot of LiDAR
  points for a fixed **training** sample. Uses a configurable `score_thr` (set to
  0.3) to filter low-confidence predictions. Displays the keyframe ID in the plot
  title.

- **`BEVValPredictionVisualizationHook`**: Same visualization style, but for a fixed
  **validation** sample. The val pipeline (`test_pipeline`) intentionally omits
  `LoadAnnotations3D`, so GT boxes are not available in the batch. This hook solves
  the problem by loading GT directly from the dataset info dict via
  `dataset.get_data_info(idx)` and filtering instances with `dataset.label_mapping`
  (raw label 7 → pedestrian class 0, all others discarded). Predictions come from
  the model's normal `predict` mode on the unmodified val input. Displays the
  keyframe ID in the plot title. Output: `bev_val_pred_vs_gt_epoch_{N}.png`.

All three hooks are registered in `mmdet3d/engine/hooks/__init__.py` and referenced
by type name in the config.

---

## 11. GT Bounding Box z-Coordinate Convention Mismatch (`mmdet3d/datasets/zod_dataset.py`)

**Problem**: Predicted bounding boxes were systematically ~half a bounding box height below GT boxes in the vertical (z) direction. Matched z-shifts confirmed the offset scaled with pedestrian height (`dz`), ruling out a constant offset.

The root cause was a coordinate convention mismatch between the ZOD data builder and the `NuScenesDataset` loader:

1. `build_zod_moe_dataset.py` stores `box_3d` with `z = z_bottom` (bottom-center), documented as `"z_definition": "bottom_center"`.
2. `build_infos.py` passes `box_3d` into the pickle unchanged — still `z_bottom`.
3. `NuScenesDataset.parse_ann_info` wraps boxes with `LiDARInstance3DBoxes(..., origin=(0.5, 0.5, 0.5))`, which declares "the input z is a geometric center." The constructor then converts to internal bottom-center storage: `z_stored = z_input - dz/2`.
4. Since `z_input` was already `z_bottom`, the result was `z_stored = z_bottom - dz/2` — one `dz/2` too low.

This corrupted the GT boxes used for training. `TransFusionBBoxCoder.encode` received wrong bottom-z, computed wrong gravity-center targets, and the model learned to predict z values that were systematically `dz/2` too low. At inference, `decode` converted these back to bottom-center, producing predictions shifted down by `dz/2` relative to true GT.

The mismatch was invisible in BEV evaluation (`CenterDistanceMetric`) which only uses x, y. It was also invisible in BEV visualization hooks which only draw x, y, dx, dy, yaw. It only became apparent in 3D visualization (`visualize_3d_predictions.py`) and 3D IoU evaluation (`IndoorMetric`).

**Evidence**: The `visualize_3d_predictions.py` script reads GT from raw `data_info['instances'][*]['bbox_3d']` (correct z_bottom from pickle) but reads predictions from the model output (shifted down by dz/2). The z-diagnostic section of the script confirmed matched z-shifts of approximately `-dz/2` per pedestrian.

**Fix**: Created `ZODDataset` (`mmdet3d/datasets/zod_dataset.py`), a minimal subclass of `NuScenesDataset` that overrides `parse_ann_info` with `origin=(0.5, 0.5, 0)` instead of `origin=(0.5, 0.5, 0.5)`. This tells the `LiDARInstance3DBoxes` constructor that the input z is already bottom-center, so no shift is applied. All four ZOD configs updated to `dataset_type = 'ZODDataset'`.

Full pipeline trace with the fix:

| Stage | z value (example: z_center=-0.3, dz=1.7) | Correct? |
|-------|------------------------------------------|----------|
| ZOD builder `box_3d` | z_bottom = -1.15 | Yes |
| Pickle `bbox_3d` | z_bottom = -1.15 | Yes |
| `ZODDataset.parse_ann_info` (origin 0.5,0.5,**0**) | z_bottom = -1.15 (no shift) | Yes |
| `gravity_center` property | -1.15 + 1.7×0.5 = -0.30 | Yes |
| `bbox_coder.encode` target (gravity) | -1.15 + 1.7×0.5 = -0.30 | Yes |
| `bbox_coder.decode` output (bottom) | -0.30 - 1.7×0.5 = -1.15 | Yes |
| Eval GT (`eval_ann_info`) | z_bottom = -1.15 | Yes |
| Eval pred | z_bottom = -1.15 | Yes |
| Vis GT (raw pickle) | z_bottom = -1.15 | Yes |
| Vis pred (model output) | z_bottom = -1.15 | Yes |

**Alternative fix considered**: Changing the ZOD builder to store `z_center` (geometric center) instead of `z_bottom` would also have been correct — `NuScenesDataset` with `origin=(0.5, 0.5, 0.5)` expects geometric center input. The `ZODDataset` approach was chosen to avoid rebuilding the dataset.

**Impact**: All existing checkpoints were trained with corrupted z-targets and must be retrained.

---

## Summary of All Changes

| # | File | Change | Category |
|---|------|--------|----------|
| 1a | `mmdet3d/__init__.py` | Bump mmcv version cap | Environment |
| 1b | `bevfusion/__init__.py` | Wrap camera imports in try/except | Environment |
| 1c | `bevfusion/ops/__init__.py` | Fallback to mmcv voxel ops | Environment |
| 2 | `bevfusion/bevfusion.py` | Reorder voxel coords (Z,Y,X) → (Y,X,Z) *(caused by 1c)* | Ops compat |
| 3 | `bevfusion/bevfusion.py` | Guard against empty point clouds | Robustness |
| 4 | `transfusion_head.py` | Remove x/y swap in heatmap target placement | Geometry bug |
| 5 | `transfusion_head.py` | Fix positional encoding channel order | Geometry bug |
| 6 | `transfusion_head.py` | Bypass heatmap NMS for pedestrian class | Proposal coverage |
| 7 | `zod_lidar_only.py` | Increase num_proposals 200 → 500 | Proposal coverage |
| 8 | `transfusion_head.py` | Add test-time circle NMS for custom_zod | Evaluation |
| 9 | `zod_lidar_only.py` | Remove flips, disable rotation | Augmentation |
| 10 | `bev_visualization_hook.py` | Add BEV feature + train/val prediction viz hooks | Diagnostics |
| 11 | `zod_dataset.py` + all ZOD configs | Fix GT z-coordinate double-subtraction via `ZODDataset` with `origin=(0.5,0.5,0)` | Geometry bug |

---

## Key Config Parameters (Final State)

```
dataset_type         = 'ZODDataset'  (origin=(0.5,0.5,0) for bottom-center z)
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
