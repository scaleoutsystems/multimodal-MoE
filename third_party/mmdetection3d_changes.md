# MMDetection3D — Changes for ZOD-MoE Thesis

Base: **v1.4.0** — commit `fe25f7a51d36e3702f961e198894580d83c4387b`
Upstream: `https://github.com/open-mmlab/mmdetection3d.git` (branch `main`)

---

## Modified files (5)

### 1. `mmdet3d/__init__.py`

**What:** Bumped `mmcv_maximum_version` from `2.2.0` → `2.3.0`.

**Why:** The installed mmcv `2.2.0` was being rejected at import time by the
strict version check. This is a safe one-liner — no API changes between
mmcv 2.2.0 and 2.3.0 affect mmdet3d.

### 2. `projects/BEVFusion/bevfusion/bevfusion.py`

**What:** Added one line in `extract_pts_feat()` to reorder voxel coordinates
before passing them to the sparse encoder:

```python
coords = coords[:, [0, 2, 3, 1]]   # (batch, Z, Y, X) → (batch, Y, X, Z)
```

**Why:** The mmcv `Voxelization` op returns coordinates in **(batch, Z, Y, X)**
order, but `BEVFusionSparseEncoder` expects **(batch, H, W, D) = (batch, Y,
X, Z)**. On NuScenes this mismatch was silently masked because the BEV grid
is square (1440 × 1440). On ZOD's rectangular grid (896 × 1248) it caused a
tensor shape mismatch in the detection head heatmap (112 × 156 vs 156 × 112).
This single-line fix resolves the issue for any non-square grid while
remaining backward-compatible with NuScenes.

### 3. `projects/BEVFusion/bevfusion/__init__.py`

**What:** Wrapped the imports of camera-specific BEVFusion modules (depth-LSS,
image backbone, etc.) in a `try/except` block so that the sparse encoder is
always importable even when the custom CUDA `bev_pool_ext` extension is not
compiled.

**Why:** For LiDAR-only training we don't need the camera path, but the
original `__init__.py` would crash at import time if the CUDA op was missing.
The try/except lets us import `BEVFusionSparseEncoder` (and the full
BEVFusion model when the ops are available) without requiring the extension
to be built first.

### 4. `projects/BEVFusion/bevfusion/ops/__init__.py`

**What:** Added a `try/except` fallback: if the BEVFusion custom
`Voxelization`/`DynamicScatter` ops fail to load, fall back to the
equivalent ops from `mmcv.ops`.

**Why:** Same reason as above — avoids requiring the custom CUDA extension
for LiDAR-only work, since `mmcv.ops` provides the identical functionality.

### 5. `mmdet3d/engine/hooks/__init__.py`

**What:** Added imports for the two new BEV visualization hooks
(`BEVFeatureVisualizationHook`, `BEVPredictionVisualizationHook`).

**Why:** Registers the hooks with MMEngine's HOOKS registry so they can be
referenced by type name in config files.

---

## New files (2)

### 1. `configs/zod/zod_lidar_only.py`

Full MMDetection3D config for LiDAR-only BEVFusion training on the ZOD-MoE
dataset. Key differences from the default NuScenes BEVFusion config:

- `point_cloud_range`: ZOD's forward-looking geometry `[0, -89.6, -5, 249.6, 89.6, 3]`
- `voxel_size`: `[0.2, 0.2, 0.2]`
- `sparse_shape`: `[896, 1248, 41]` (H, W, D convention)
- Single class: `pedestrian`
- Removed multi-sweep loading, ObjectSample augmentation, velocity heads
- Added `IndoorMetric` for val/test evaluation (avoids nuscenes-devkit dependency)
- Includes `BEVFeatureVisualizationHook` and `BEVPredictionVisualizationHook`

### 2. `mmdet3d/engine/hooks/bev_visualization_hook.py`

Two lightweight custom hooks for training diagnostics:

- **`BEVFeatureVisualizationHook`** — saves L2-norm heatmaps of the sparse
  encoder and FPN BEV feature maps at selected epochs.
- **`BEVPredictionVisualizationHook`** — overlays predicted boxes (red) vs
  GT boxes (green) on a BEV LiDAR scatter plot.

Both hooks run on a single fixed validation sample (index 0) using
`torch.no_grad()` and restore training mode afterwards.

---

## How to reproduce from a clean clone

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout fe25f7a51d36e3702f961e198894580d83c4387b
# then apply each change listed above
```

A machine-applicable `.patch` file will be generated once experiments are
finalized.  Until then, this changelog is the authoritative record.
