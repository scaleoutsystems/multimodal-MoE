# MMDetection3D – Local Modifications for ZOD-MoE Thesis

**Local path**: `/home/edgelab/mmdetection3d`
**Upstream**: <https://github.com/open-mmlab/mmdetection3d>
**Base version**: v1.4.0 (commit `fe25f7a5`)
**Patch file**: `third_party/mmdetection3d_thesis.patch`

To reapply all changes from scratch:

```bash
cd /home/edgelab/mmdetection3d
git checkout fe25f7a5          # reset to upstream base
git apply /home/edgelab/multimodal-MoE/third_party/mmdetection3d_thesis.patch
```

---

## Current BEV axis convention (active)

- LiDAR BEV path uses `mmcv.ops` voxelization, then `coords = coords[:, [0, 2, 3, 1]]`, so spatial order is `(Y, X)`.
- Camera BEV from `DepthLSSTransform` is produced as `(X, Y)` and is transposed with `img_feature = img_feature.transpose(-1, -2)` before fusion.
- Therefore both branches are aligned to `(Y, X)` before `ConvFuser`.

## Modified files (6 files, small edits)

### 1. `mmdet3d/__init__.py`

**Change**: Bumped `mmcv_maximum_version` from `2.2.0` to `2.3.0`.
**Why**: The installed mmcv (2.2.x) was above the upstream cap, causing an import error at startup.

### 2. `projects/BEVFusion/bevfusion/__init__.py`

**Change**: Wrapped imports of camera-specific BEVFusion modules (e.g., `DepthLSSTransform`, `ImageAug3D`, `BEVFusion` model class) in a `try/except` block. Only `BEVFusionSparseEncoder` is imported unconditionally.
**Why**: BEVFusion ships custom CUDA extensions (`bev_pool_ext` for camera BEV pooling, `voxel_layer` for voxelization) that are **not** built by the normal `pip install -e .` of mmdetection3d. They require a separate compilation step: `cd projects/BEVFusion && python setup.py develop`. In the current setup, `bev_pool_ext` may be built separately for camera+LiDAR runs, while voxelization is intentionally taken from `mmcv.ops` (see §3) to keep coordinate conventions consistent. This `try/except` keeps imports robust when camera-specific extensions are absent.

### 3. `projects/BEVFusion/bevfusion/ops/__init__.py`

**Change**:
- Kept `bev_pool` as a guarded import with an explicit runtime error if missing (no silent `None` fallback).
- Forced voxel ops to always use `mmcv.ops` (`Voxelization`, `DynamicScatter`, etc.), even when BEVFusion's custom `voxel_layer` extension is present.
**Why**:
- Camera+LiDAR BEVFusion requires `bev_pool`; failing loudly gives clear setup feedback.
- BEVFusion custom `voxel_layer` emits coordinates in `(X, Y, Z)`, while the current LiDAR path in `bevfusion.py` is implemented for mmcv's `(Z, Y, X)` output plus the reorder in Change 4A. Forcing mmcv voxel ops avoids convention drift and prevents sparse-conv coordinate mismatches.

### 4. `projects/BEVFusion/bevfusion/bevfusion.py` (3 changes)

**Change A – Coordinate reordering in `extract_pts_feat`** *(direct consequence of §3)*:
Added `coords = coords[:, [0, 2, 3, 1]]` after voxelization.
**Why**: mmcv `Voxelization` returns coordinates in `(batch, Z, Y, X)`, while `BEVFusionSparseEncoder` expects `(batch, Y, X, Z)`. Without this permutation, Y-coordinates (range 0–1439) end up in the Z slot (max 41), causing an out-of-bounds crash during sparse tensor creation.

**Change B – Zero-point guard in `voxelize`**:
Added a check: if a sample's point cloud has 0 points (e.g., after `PointsRangeFilter`), substitute a single dummy zero-point.
**Why**: The `hard_voxelize_forward` CUDA kernel crashes with `invalid configuration argument` when given an empty input. This can happen during validation on edge-case samples.

**Change C – Camera BEV transpose before fusion in `extract_feat`**:
Added `img_feature = img_feature.transpose(-1, -2)` right after `extract_img_feat(...)`.
**Why**: With Change 4A, LiDAR BEV uses `(Y, X)` spatial ordering. Camera BEV from `DepthLSSTransform` is produced in `(X, Y)`. Without transposing camera BEV, `ConvFuser` concatenates features that are spatially axis-swapped across branches. The transpose enforces a shared `(Y, X)` convention before fusion.

### 5. `projects/BEVFusion/bevfusion/transfusion_head.py` (4 changes)

**Change A – Fix heatmap target placement in `get_targets_single`**:
Removed `center_int[[1, 0]]` swap (line ~730). The original BEVFusion code swaps the (x, y) indices when drawing the GT Gaussian on the heatmap. For the standard NuScenes config with a *symmetric* point cloud range (`[-54, -54, …, 54, 54]`), both axes use the same offset, so the swap is harmless. For our ZOD config with an *asymmetric* range (`[0, -54, …, 108, 54]`), the swap places each GT target at the **wrong** feature map position — errors of 10–60 m per box. The fix uses `center_int` directly (the original commented-out code).
**Why**: GT heatmap targets must align with the BEV feature at the same spatial location; otherwise the conv-based heatmap head cannot learn from local features, causing predictions to collapse to the BEV boundaries.

**Change B – Fix BEV positional encoding in `create_2D_grid`**:
Swapped the `torch.cat` order from `[batch_x, batch_y]` to `[batch_y, batch_x]` and added `indexing='ij'` to `torch.meshgrid`.
**Why**: The feature map has shape `(B, C, H=Y, W=X)`. When flattened to `(B, C, H*W)`, position `m` corresponds to `(Y_idx=m//W, X_idx=m%W)`. The original code put `Y_idx` into channel 0 and `X_idx` into channel 1. But the bbox coder encodes `target[0] = ix` (X) and `target[1] = iy` (Y). Since `center = pred_offset + query_pos`, the positional encoding channels must match the encode convention: channel 0 = X, channel 1 = Y.

**Change C – Disable heatmap NMS for pedestrian class (`custom_zod` dataset)**:
Added a `custom_zod` branch alongside the existing `nuScenes` and `Waymo` branches in the heatmap NMS section of `forward_single`. For `custom_zod`, class 0 (pedestrian) bypasses the 3×3 max-pool NMS using `kernel_size=1` (identity).
Also wrapped the entire NMS block in `if self.nms_kernel_size > 1:` to handle `nms_kernel_size=1` gracefully.
**Why**: The original 3×3 NMS suppresses pedestrian heatmap peaks that are within 1.8 m of a stronger neighbor. For dense pedestrian scenes (30–50 per frame), this kills many valid peaks. With only 200 proposals, the surviving peaks cannot cover all GT locations. Hungarian matching then assigns far proposals to the missing GT, creating large, unlearnable center regression targets. NuScenes and Waymo already bypass NMS for their pedestrian classes; this change adds the same for `custom_zod`. Combined with increasing `num_proposals` to 500, this fix reduced center L1 error from 4.5 to 0.1 feature-map pixels and raised matched IoU from 0.26 to 0.62.

**Change D – Add test-time circle NMS for `custom_zod` dataset**:
Added a `custom_zod` branch in the `predict_by_feat` method that defines a single task for pedestrians with `radius=0.175` (matching the NuScenes pedestrian circle NMS radius). This is used when `test_cfg['nms_type'] = 'circle'`.
**Why**: With 500 proposals and heatmap NMS bypassed for pedestrians, many overlapping predictions survive at inference time. Without post-processing NMS, evaluation metrics are polluted by duplicate detections. Circle NMS with 0.175 m radius removes near-duplicate predictions while preserving distinct pedestrian detections.

### 6. `mmdet3d/engine/hooks/__init__.py`

**Change**: Added imports and `__all__` entries for `BEVFeatureVisualizationHook`, `BEVPredictionVisualizationHook`, and `BEVValPredictionVisualizationHook`.
**Why**: Registers the three custom visualization hooks so they can be referenced by type name in config files.

---

## New files (2 files)

### 7. `configs/zod/zod_lidar_only.py` (290 lines)

Full MMDetection3D config for LiDAR-only BEVFusion training on the ZOD-MoE pedestrian dataset.
Key settings:
- `voxel_size = [0.075, 0.075, 0.2]` (matches NuScenes pretrained checkpoint)
- `point_cloud_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]` (108m × 108m, same area as NuScenes)
- `grid_size = [1440, 1440, 40]`, `sparse_shape = [1440, 1440, 41]` (identical to NuScenes)
- `num_proposals = 500` (increased from default 200 for dense pedestrian coverage)
- `nms_kernel_size = 3` (default; bypassed for pedestrians via `custom_zod` branch)
- Test-time circle NMS with `radius=0.175` for pedestrians
- Augmentation: scaling (0.9–1.1) and translation (0.5 m) only; flips and rotations disabled due to asymmetric forward-only X-range
- Single class: `pedestrian`
- Pretrained from NuScenes LiDAR-only BEVFusion checkpoint
- Uses `IndoorMetric` for evaluation (no nuscenes-devkit dependency)
- Includes three custom BEV visualization hooks (feature heatmap, train BEV predictions, val BEV predictions)

### 8. `mmdet3d/engine/hooks/bev_visualization_hook.py` (~400 lines)

Three MMEngine custom hooks for training diagnostics:
- **`BEVFeatureVisualizationHook`**: Saves L2-norm heatmaps of the sparse encoder and FPN outputs for a fixed sample at selected epochs.
- **`BEVPredictionVisualizationHook`**: Overlays predicted vs GT bounding boxes in BEV on LiDAR points for a fixed **training** sample. Displays the keyframe ID in the plot title.
- **`BEVValPredictionVisualizationHook`**: Same visualization for a fixed **validation** sample. Since the val pipeline (`test_pipeline`) does not include `LoadAnnotations3D`, this hook loads GT boxes directly from the dataset info dict via `dataset.get_data_info(idx)`, filtering by `dataset.label_mapping`. This bypasses the pipeline entirely for GT, while predictions come from the model's normal `predict` mode on the unmodified val input. Displays the keyframe ID in the plot title.

Shared helpers:
- `_load_gt_from_info(dataset, idx)` — reads GT boxes from the raw dataset info, filtered by `label_mapping` (e.g., raw label 7 → pedestrian, all others discarded)
- `_keyframe_id_from_path(lidar_path)` — extracts keyframe ID (e.g., `000011`) from the `.bin` filename
