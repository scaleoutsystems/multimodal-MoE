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

## Modified files (6 files, small edits)

### 1. `mmdet3d/__init__.py`

**Change**: Bumped `mmcv_maximum_version` from `2.2.0` to `2.3.0`.
**Why**: The installed mmcv (2.2.x) was above the upstream cap, causing an import error at startup.

### 2. `projects/BEVFusion/bevfusion/bevfusion.py` (2 changes)

**Change A – Coordinate reordering in `extract_pts_feat`**:
Added `coords = coords[:, [0, 2, 3, 1]]` after voxelization.
**Why**: The mmcv `Voxelization` op outputs voxel coordinates in `(batch, Z, Y, X)` order, but `BEVFusionSparseEncoder` expects `(batch, Y, X, Z)`. Without this permutation, the spatial dimensions are swapped, causing a shape mismatch in the detection head heatmap.

**Change B – Zero-point guard in `voxelize`**:
Added a check: if a sample's point cloud has 0 points (e.g., after `PointsRangeFilter`), substitute a single dummy zero-point.
**Why**: The `hard_voxelize_forward` CUDA kernel crashes with `invalid configuration argument` when given an empty input. This can happen during validation on edge-case samples.

### 3. `projects/BEVFusion/bevfusion/__init__.py`

**Change**: Wrapped imports of camera-specific BEVFusion modules (e.g., `DepthLSSTransform`, `ImageAug3D`, `BEVFusion` model class) in a `try/except` block. Only `BEVFusionSparseEncoder` is imported unconditionally.
**Why**: Some BEVFusion modules require custom CUDA extensions (`bev_pool_ext`) that may not be compiled. The LiDAR-only pipeline only needs the sparse encoder. This lets the package import cleanly without building all CUDA ops.

### 4. `projects/BEVFusion/bevfusion/ops/__init__.py`

**Change**: Added `try/except` fallback: if BEVFusion's custom voxel CUDA extension is unavailable, import `Voxelization` and `DynamicScatter` from `mmcv.ops` instead.
**Why**: Same rationale — avoids needing to compile BEVFusion's custom CUDA extensions when mmcv already provides equivalent ops.

### 5. `projects/BEVFusion/bevfusion/transfusion_head.py` (2 changes)

**Change A – Fix heatmap target placement in `get_targets_single`**:
Removed `center_int[[1, 0]]` swap (line ~730). The original BEVFusion code swaps the (x, y) indices when drawing the GT Gaussian on the heatmap. For the standard NuScenes config with a *symmetric* point cloud range (`[-54, -54, …, 54, 54]`), both axes use the same offset, so the swap is harmless. For our ZOD config with an *asymmetric* range (`[0, -54, …, 108, 54]`), the swap places each GT target at the **wrong** feature map position — errors of 10–60 m per box. The fix uses `center_int` directly (the original commented-out code).
**Why**: GT heatmap targets must align with the BEV feature at the same spatial location; otherwise the conv-based heatmap head cannot learn from local features, causing predictions to collapse to the BEV boundaries.

**Change B – Fix BEV positional encoding in `create_2D_grid`**:
Swapped the `torch.cat` order from `[batch_x, batch_y]` to `[batch_y, batch_x]` and added `indexing='ij'` to `torch.meshgrid`.
**Why**: The feature map has shape `(B, C, H=Y, W=X)`. When flattened to `(B, C, H*W)`, position `m` corresponds to `(Y_idx=m//W, X_idx=m%W)`. The original code put `Y_idx` into channel 0 and `X_idx` into channel 1. But the bbox coder encodes `target[0] = ix` (X) and `target[1] = iy` (Y). Since `center = pred_offset + query_pos`, the positional encoding channels must match the encode convention: channel 0 = X, channel 1 = Y.

### 6. `mmdet3d/engine/hooks/__init__.py`

**Change**: Added imports and `__all__` entries for `BEVFeatureVisualizationHook` and `BEVPredictionVisualizationHook`.
**Why**: Registers the two new custom visualization hooks so they can be referenced by type name in config files.

---

## New files (2 files)

### 7. `configs/zod/zod_lidar_only.py` (289 lines)

Full MMDetection3D config for LiDAR-only BEVFusion training on the ZOD-MoE pedestrian dataset.
Key settings:
- `voxel_size = [0.075, 0.075, 0.2]` (matches NuScenes pretrained checkpoint)
- `point_cloud_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]` (108m × 108m, same area as NuScenes)
- `grid_size = [1440, 1440, 40]`, `sparse_shape = [1440, 1440, 41]` (identical to NuScenes)
- Single class: `pedestrian`
- Pretrained from NuScenes LiDAR-only BEVFusion checkpoint
- Uses `IndoorMetric` for evaluation (no nuscenes-devkit dependency)
- Includes custom BEV visualization hooks

### 8. `mmdet3d/engine/hooks/bev_visualization_hook.py` (210 lines)

Two MMEngine custom hooks for training diagnostics:
- **`BEVFeatureVisualizationHook`**: Saves L2-norm heatmaps of the sparse encoder and FPN outputs for a fixed validation sample at selected epochs.
- **`BEVPredictionVisualizationHook`**: Overlays predicted vs GT bounding boxes in BEV on LiDAR points for the same sample.
