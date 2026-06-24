# MMDetection3D – Local Modifications for ZOD-MoE Thesis

**Local path**: `~/projects/multimodal-MoE/mmdetection3d`
**Upstream**: <https://github.com/open-mmlab/mmdetection3d>
**Base version**: v1.4.0 (commit `fe25f7a5`)
**Patch file**: `third_party/mmdetection3d_thesis.patch`

To reapply all changes from scratch:

```bash
cd ~/projects/multimodal-MoE/mmdetection3d
git checkout fe25f7a5          # reset to upstream base
git apply ~/projects/multimodal-MoE/third_party/mmdetection3d_thesis.patch
```

---

## Current BEV axis convention (active)

- LiDAR BEV path uses `mmcv.ops` voxelization, then `coords = coords[:, [0, 2, 3, 1]]`, so spatial order is `(Y, X)`.
- Camera BEV from `DepthLSSTransform` is produced as `(X, Y)` and is transposed with `img_feature = img_feature.transpose(-1, -2)` before fusion.
- Therefore both branches are aligned to `(Y, X)` before `ConvFuser`.

---

## Modified files

### 1. `mmdet3d/__init__.py`

**Change**: Bumped `mmcv_maximum_version` from `2.2.0` to `2.3.0`.
**Why**: The installed mmcv (2.2.x) was above the upstream cap, causing an import error at startup.

---

### 2. `projects/BEVFusion/bevfusion/__init__.py`

**Change**: Wrapped imports of camera-specific BEVFusion modules (e.g., `DepthLSSTransform`, `ImageAug3D`, `BEVFusion` model class) in a `try/except` block. Only `BEVFusionSparseEncoder` is imported unconditionally.
**Why**: BEVFusion ships custom CUDA extensions (`bev_pool_ext` for camera BEV pooling, `voxel_layer` for voxelization) that are **not** built by the normal `pip install -e .` of mmdetection3d. They require a separate compilation step: `cd projects/BEVFusion && python setup.py develop`. In the current setup, `bev_pool_ext` may be built separately for camera+LiDAR runs, while voxelization is intentionally taken from `mmcv.ops` (see §3) to keep coordinate conventions consistent. This `try/except` keeps imports robust when camera-specific extensions are absent.

---

### 3. `projects/BEVFusion/bevfusion/ops/__init__.py`

**Change**:
- Kept `bev_pool` as a guarded import with an explicit runtime error if missing (no silent `None` fallback).
- Forced voxel ops to always use `mmcv.ops` (`Voxelization`, `DynamicScatter`, etc.), even when BEVFusion's custom `voxel_layer` extension is present.

**Why**:
- Camera+LiDAR BEVFusion requires `bev_pool`; failing loudly gives clear setup feedback.
- BEVFusion custom `voxel_layer` emits coordinates in `(X, Y, Z)`, while the current LiDAR path in `bevfusion.py` is implemented for mmcv's `(Z, Y, X)` output plus the reorder in Change 4A. Forcing mmcv voxel ops avoids convention drift and prevents sparse-conv coordinate mismatches.

---

### 4. `projects/BEVFusion/bevfusion/depth_lss.py` (2 changes)

**Change A – Sparse-depth splatting in `BaseDepthTransform._splat_depth` and `forward`** (`splat_radius` parameter):
Added a `splat_radius: int = 0` argument to `BaseDepthTransform.__init__` and a `_splat_depth` static method.  When `splat_radius > 0`, each projected LiDAR point is written to a `(2r+1) × (2r+1)` pixel neighbourhood in the sparse depth map instead of a single pixel.  At each neighbour location, the *closer* depth wins so the fill does not blur across depth discontinuities.  The finetune configs use `splat_radius=1` (3×3 window).
**Why**: The sparse LiDAR depth projected onto the camera image is sparse — at 1248×704 resolution (879 000 pixels), a typical ZOD front-camera frame receives ≈ 54 000 LiDAR hits (~6% occupancy).  Although 6% is not negligible, each hit is an isolated point; the `dtransform` conv layers see long runs of zeros between individual pixels, making it hard to learn spatially coherent depth features from local receptive fields.  A 3×3 fill (up to 9× more non-zero pixels around each hit) makes each depth measurement spatially contiguous, improving the signal available to the first conv kernels without introducing depth bleeding (closer-depth-wins semantics are preserved at boundaries).

**Change B – Auxiliary depth supervision in `DepthLSSTransform`** (`aux_depth_loss_weight` parameter):
Added `aux_depth_loss_weight: float = 0.0` to `DepthLSSTransform.__init__`.  During the forward pass the depthnet's raw depth logits and the original full-resolution sparse depth map are passed to `_compute_aux_depth_loss`, which:
1. Downsamples the GT sparse depth from `(B, N, 1, H, W)` to feature resolution `(BN, fH, fW)` via `adaptive_max_pool2d` (max rather than average to preserve the closest valid depth).
2. Quantises each valid pixel into one of the `D` depth bins using the same `dbound` as the model.
3. Computes a masked cross-entropy loss between the predicted depth logits and the GT bin indices, averaging only over pixels with valid sparse depth.
The computed loss (already scaled by `aux_depth_loss_weight`) is stored in `self._aux_depth_loss` and picked up by `bevfusion.py`'s `loss()` method.  The finetune configs use `aux_depth_loss_weight=0.5`.
**Why**: Without explicit depth supervision, the depthnet must learn depth purely from the detection loss backpropagated through the BEV pooling operation — a very long and noisy gradient path.  Early in training, when camera features are still random, this indirect signal is too weak to teach the depthnet anything useful, leading to collapsed or arbitrary depth distributions.  The auxiliary cross-entropy loss provides direct, per-pixel depth supervision at every training step, dramatically accelerating depth learning and stabilising early fusion.  The weight is kept low (0.5 vs. the 3.0 default in some BEVFusion variants) to avoid dominating the detection losses before the detection head has converged.

---

### 5. `projects/BEVFusion/bevfusion/bevfusion.py` (4 changes)

**Change A – Coordinate reordering in `extract_pts_feat`** *(direct consequence of §3)*:
Added `coords = coords[:, [0, 2, 3, 1]]` after voxelization.
**Why**: mmcv `Voxelization` returns coordinates in `(batch, Z, Y, X)`, while `BEVFusionSparseEncoder` expects `(batch, Y, X, Z)`. Without this permutation, Y-coordinates (range 0–1439) end up in the Z slot (max 41), causing an out-of-bounds crash during sparse tensor creation.

**Change B – Zero-point guard in `voxelize`**:
Added a check: if a sample's point cloud has 0 points (e.g., after `PointsRangeFilter`), substitute a single dummy zero-point.
**Why**: The `hard_voxelize_forward` CUDA kernel crashes with `invalid configuration argument` when given an empty input. This can happen during validation on edge-case samples.

**Change C – Camera BEV transpose before fusion in `extract_feat`**:
Added `img_feature = img_feature.transpose(-1, -2)` right after `extract_img_feat(...)`.
**Why**: With Change A above, LiDAR BEV uses `(Y, X)` spatial ordering. Camera BEV from `DepthLSSTransform` is produced in `(X, Y)`. Without transposing camera BEV, `ConvFuser` concatenates features that are spatially axis-swapped across branches. The transpose enforces a shared `(Y, X)` convention before fusion.

**Change D – Auxiliary depth loss collection in `loss()`**:
Added three lines after `losses.update(bbox_loss)`:
```python
vt = getattr(self, 'view_transform', None)
aux = getattr(vt, '_aux_depth_loss', None) if vt else None
if aux is not None and aux.numel() > 0 and aux.item() > 0:
    losses['aux_depth_loss'] = aux
```
**Why**: `_aux_depth_loss` is computed and stored on the `view_transform` module during `forward()` (see §4, Change B in `depth_lss.py`). MMEngine's training loop collects all scalar tensors returned by `loss()` as named losses and backpropagates their sum. Adding `aux_depth_loss` to the dict ensures it is included in the backward pass without any changes to the training loop. The guard (`numel() > 0`, `item() > 0`) prevents propagating the dummy zero tensor that `_compute_aux_depth_loss` returns when no valid sparse-depth pixels are present in a batch.

---

### 6. `projects/BEVFusion/bevfusion/utils.py`

**Change – Empty-GT branch in `HungarianAssigner3D.assign`**:
When `num_gts == 0` (the sample has no ground-truth boxes), the original
upstream code returns `AssignResult(num_gts, assigned_gt_inds, None,
labels=assigned_labels)` — i.e. `max_overlaps=None`. Replaced the `None`
with an explicit zero tensor of shape `(num_bboxes,)`:

```python
max_overlaps = bboxes.new_zeros((num_bboxes,))
return AssignResult(
    num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
```

**Why**: `transfusion_head.get_targets` aggregates per-sample assignment
results via `multi_apply` and then does
`torch.cat([res.max_overlaps for res in assign_result_list])`
(see `transfusion_head.py:650`). With `max_overlaps=None` for any
empty-GT sample in the batch, `torch.cat` raises
`TypeError: expected Tensor as element 0 in argument 0, but got NoneType`
and training crashes on the first iteration. This only manifests when
the dataloader retains scenes with no GT objects, i.e. when the dataset
is configured with `filter_empty_gt=False` (used by the MoE configs so
the context-routing branch sees the full road_type distribution rather
than the post-filter pedestrian-only subset). The semantically correct
value when `num_gts == 0` is "every prediction has zero overlap with
any GT" — a zero tensor — which is also consistent with the populated
branch that produces `max_overlaps = torch.zeros_like(iou.max(1).values)`
followed by index-fills.

The rest of the empty-GT path is already safe: `len(pos_inds) > 0`
guards the bbox-target population, `loss_cls`/`loss_bbox` use
`avg_factor=max(num_pos, 1)`, and the heatmap is initialised to zeros
with the GT-loop trivially skipping when there are no boxes.

---

### 7. `projects/BEVFusion/bevfusion/transfusion_head.py` (4 changes)

**Change A – Fix heatmap target placement in `get_targets_single`**:
Removed `center_int[[1, 0]]` swap (line ~730). The original BEVFusion code swaps the (x, y) indices when drawing the GT Gaussian on the heatmap. For the standard NuScenes config with a *symmetric* point cloud range (`[-54, -54, …, 54, 54]`), both axes use the same offset, so the swap is harmless. For our ZOD config with an *asymmetric* range (`[0, -54, …, 108, 54]`), the swap places each GT target at the **wrong** feature map position — errors of 10–60 m per box.
**Why**: GT heatmap targets must align with the BEV feature at the same spatial location; otherwise the conv-based heatmap head cannot learn from local features, causing predictions to collapse to the BEV boundaries.

**Change B – Fix BEV positional encoding in `create_2D_grid`**:
Swapped the `torch.cat` order from `[batch_x, batch_y]` to `[batch_y, batch_x]` and added `indexing='ij'` to `torch.meshgrid`.
**Why**: The feature map has shape `(B, C, H=Y, W=X)`. When flattened to `(B, C, H*W)`, position `m` corresponds to `(Y_idx=m//W, X_idx=m%W)`. The original code put `Y_idx` into channel 0 and `X_idx` into channel 1. But the bbox coder encodes `target[0] = ix` (X) and `target[1] = iy` (Y). Since `center = pred_offset + query_pos`, the positional encoding channels must match the encode convention: channel 0 = X, channel 1 = Y.

**Change C – Disable heatmap NMS for pedestrian class (`custom_zod` dataset)**:
Added a `custom_zod` branch alongside the existing `nuScenes` and `Waymo` branches in the heatmap NMS section of `forward_single`. For `custom_zod`, class 0 (pedestrian) bypasses the 3×3 max-pool NMS using `kernel_size=1` (identity). Also wrapped the entire NMS block in `if self.nms_kernel_size > 1:` to handle `nms_kernel_size=1` gracefully.
**Why**: The original 3×3 NMS suppresses pedestrian heatmap peaks that are within 1.8 m of a stronger neighbor. For dense pedestrian scenes (30–50 per frame), this kills many valid peaks. With only 200 proposals, the surviving peaks cannot cover all GT locations. Hungarian matching then assigns far proposals to the missing GT, creating large, unlearnable center regression targets. NuScenes and Waymo already bypass NMS for their pedestrian classes; this change adds the same for `custom_zod`. Combined with increasing `num_proposals` to 500, this fix reduced center L1 error from 4.5 to 0.1 feature-map pixels and raised matched IoU from 0.26 to 0.62.

**Change D – Add test-time circle NMS for `custom_zod` dataset**:
Added a `custom_zod` branch in the `predict_by_feat` method that defines a single task for pedestrians with `radius=0.175` (matching the NuScenes pedestrian circle NMS radius).
**Why**: With 500 proposals and heatmap NMS bypassed for pedestrians, many overlapping predictions survive at inference time. Without post-processing NMS, evaluation metrics are polluted by duplicate detections. Circle NMS with 0.175 m radius removes near-duplicate predictions while preserving distinct pedestrian detections.

---

### 7. `mmdet3d/engine/hooks/__init__.py`

**Change**: Expanded from the upstream 2-hook baseline to register all 11 custom hooks added by this project. Imports and `__all__` entries for:
`BEVFeatureVisualizationHook`, `BEVPredictionVisualizationHook`, `BEVValPredictionVisualizationHook`,
`BEVCameraFeatureVisualizationHook`, `DepthTransformDiagnosticHook`, `DepthProjectionDebugHook`,
`TrainingEfficiencyHook`, `RunSummaryHook`, `FusionTrainingStrategyHook`,
`FreezeLidarBranchHook`, `ValidationCurveHook`.
**Why**: Registers all custom hooks so they can be referenced by type name in config files without explicit Python imports.

---

## New files — datasets

### 7b. `mmdet3d/datasets/zod_dataset.py`

**New file**: `ZODDataset`, a subclass of `NuScenesDataset` that overrides only `parse_ann_info`.

**Change**: The single difference is `origin=(0.5, 0.5, 0)` instead of `origin=(0.5, 0.5, 0.5)` when constructing `LiDARInstance3DBoxes`.

**Why**: The ZOD build pipeline (`build_zod_moe_dataset.py`) stores `box_3d` with z as **bottom-center** (`z_bottom = z_center - dz/2`), matching MMDet3D's internal LiDAR box convention `(0.5, 0.5, 0)`. The vanilla `NuScenesDataset` declares `origin=(0.5, 0.5, 0.5)`, which tells the `LiDARInstance3DBoxes` constructor that the input z is a **geometric center** and subtracts `dz/2` to convert to bottom-center. Applied to already-bottom-center ZOD boxes, this double-subtracts `dz/2`, shifting all GT boxes half a box height below their true position. The model then learns corrupted z-targets, producing predictions that are systematically ~half a box too low.

`ZODDataset` inherits everything else from `NuScenesDataset` unchanged — the nuScenes-format pickle, `parse_data_info`, filtering logic, velocity handling — since the ZOD data is stored in nuScenes layout. Only the z-origin declaration differs.

Registered in `mmdet3d/datasets/__init__.py` via `from .zod_dataset import ZODDataset` and added to `__all__`.

**Config impact**: All four ZOD configs (`zod_lidar_only.py`, `zod_bevfusion_baseline.py`, `zod_bevfusion_finetune.py`, `zod_bevfusion_finetune_fixedLidar.py`) now use `dataset_type = 'ZODDataset'` instead of `'NuScenesDataset'`.

---

## New files — configs

### 8. `configs/zod/zod_lidar_only.py`

Full MMDetection3D config for LiDAR-only BEVFusion training on the ZOD-MoE pedestrian dataset.
Key settings:
- `voxel_size = [0.075, 0.075, 0.2]` (matches NuScenes pretrained checkpoint)
- `point_cloud_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]` (108 m × 108 m forward-facing, same area as NuScenes)
- `grid_size = [1440, 1440, 40]`, `sparse_shape = [1440, 1440, 41]` (identical to NuScenes)
- `num_proposals = 500` (increased from default 200 for dense pedestrian coverage)
- `nms_kernel_size = 3` (default; bypassed for pedestrians via `custom_zod` branch)
- Test-time circle NMS with `radius=0.175` for pedestrians
- Augmentation: scaling (0.9–1.1) and translation (0.5 m) only; flips and rotations disabled due to asymmetric forward-only X-range
- Single class: `pedestrian`
- Pretrained from NuScenes LiDAR-only BEVFusion checkpoint (`load_from`); what transfers and what does not is documented inline in the config header comment
- Uses `IndoorMetric` and `CenterDistanceMetric` for evaluation (no nuscenes-devkit dependency)
- Visualisation hooks: `BEVFeatureVisualizationHook`, `BEVPredictionVisualizationHook`, `BEVValPredictionVisualizationHook`

---

### 9. `configs/zod/zod_bevfusion_baseline.py`

Camera+LiDAR BEVFusion config for ZOD pedestrian detection. From-scratch training (no pretrained LiDAR checkpoint). Establishes the multimodal baseline before staged fine-tuning.
Key additions over `zod_lidar_only.py`:
- Swin-T image backbone (ImageNet pretrained via `init_cfg`)
- `GeneralizedLSSFPN` camera neck
- `DepthLSSTransform` view transform adapted to ZOD's asymmetric BEV range (`xbound=[0, 108]`)
- `ConvFuser` for camera (80-ch) + LiDAR (256-ch) → 256-ch BEV fusion
- Additional hooks: `BEVCameraFeatureVisualizationHook`, `DepthTransformDiagnosticHook`, `DepthProjectionDebugHook`

---

### 10. `configs/zod/zod_bevfusion_finetune.py`

Camera+LiDAR BEVFusion **fine-tuning** config. Differs from the baseline in:
- `load_from` points to the best ZOD LiDAR-only checkpoint (strong init for LiDAR branch)
- 14-epoch schedule: LinearLR warmup (500 iters) + cosine annealing in two phases (0→4, 4→10 ep) + two additional final epochs
- `aux_depth_loss_weight=0.5` set in `view_transform` config; the auxiliary depth supervision loss itself was added to `depth_lss.py` as part of this project (see §4 Change B) — this is not a reduction from an upstream default
- Visualisation runs at epochs `(1, 3, 5, 7, 10, 12, 14)` and always at the final epoch (controlled via `vis_epochs=_VIS_EPOCHS` on each hook)

---

### 11. `configs/zod/zod_bevfusion_finetune_fixedLidar.py`

Identical to `zod_bevfusion_finetune.py` except the LiDAR branch is **hard-frozen** for epochs 1–9 then unfrozen for joint training in epochs 10–14. Implementation:
- `FreezeLidarBranchHook(unfreeze_epoch=10)` sets `requires_grad_(False)` and `eval()` on `pts_voxel_encoder`, `pts_middle_encoder`, `pts_backbone`, `pts_neck` before training; re-applies `eval()` every epoch to override the loop's `model.train()` call; re-enables `requires_grad_(True)` + `train()` at epoch 10.
- `env_cfg` overrides `find_unused_parameters=True` (required so DDP does not hang waiting for gradient buckets of the frozen modules).
- `param_scheduler` extended to three phases: (0→4), (4→10), (10→14), so the final joint phase has a full cosine LR schedule.
- No `paramwise_cfg` needed since frozen params have `grad=None` and AdamW skips them automatically.

---

### 12. `tools/sbatch/meluxina_train_zod_bevfusion.sbatch`

Slurm batch script for Meluxina (4 × A100 40 GB, single node, `gpu` partition, account `p201392`). Features:
- Robust conda env detection (searches `miniforge3`, `miniconda3`, `anaconda3`, `mambaforge`; no `conda activate`)
- Strips broken Slurm-forwarded `CONDA_EXE` / `_CONDA_ROOT` env vars before activation
- Python sanity check at launch: imports `BEVFusion`, `DepthLSSTransform`, `ConvFuser`, `GeneralizedLSSFPN`
- DDP via `srun --gpus-per-task=1` with `env -u SLURM_LOCALID` workaround for Meluxina's ParaStation GPU cgroup assignment
- All paths overridable via env vars (`CONFIG`, `WORK_DIR`, `MMDET3D_ROOT`, `MOE_ROOT`)
- Outputs under `multimodal-MoE/outputs/runs/zod_bevfusion/<jobname>_<jobid>/`

---

### 13. `tools/sbatch/meluxina_train_zod_bevfusion_fixedLidar.sbatch`

Copy of `meluxina_train_zod_bevfusion.sbatch` with:
- `--job-name=zod-bevfusion-fixlidar`
- `CONFIG` defaults to `configs/zod/zod_bevfusion_finetune_fixedLidar.py`
- `DEFAULT_WORK_PARENT` set to `outputs/runs/zod_bevfusion_fixedlidar/`

---

## New files — hooks

### 14. `mmdet3d/engine/hooks/bev_visualization_hook.py`

Three MMEngine hooks for LiDAR-branch diagnostics. All three gate on the shared `_should_visualize(runner, vis_epochs=None)` helper.

`_should_visualize` logic:
- Always fires on the final epoch.
- If `vis_epochs` is given (a set/tuple of 1-indexed epoch numbers), fires when `epoch in vis_epochs`.
- Otherwise falls back to the default schedule: epochs 1, 10, 20, 30, every 50th.

**`BEVFeatureVisualizationHook`** — saves L2-norm BEV heatmaps of the sparse encoder and FPN outputs for a fixed sample. Args: `x_range`, `y_range`, `vis_epochs`.

**`BEVPredictionVisualizationHook`** — overlays predicted vs GT boxes in BEV on LiDAR points for a fixed **training** sample. Displays keyframe ID in the plot title. Args: `score_thr`, `vis_epochs`.

**`BEVValPredictionVisualizationHook`** — same for a fixed **validation** sample. GT loaded directly from dataset info (bypasses `test_pipeline` which omits `LoadAnnotations3D`). Args: `score_thr`, `sample_idx`, `vis_epochs`.

---

### 15. `mmdet3d/engine/hooks/bevfusion_visualization_hook.py`

Two MMEngine hooks for camera+fusion diagnostics. Both import and call `_should_visualize` from `bev_visualization_hook`.

**`BEVCameraFeatureVisualizationHook`** — L2-norm heatmaps of `view_transform` output (camera BEV, 80-ch) and `fusion_layer` output (fused BEV, 256-ch). Args: `x_range`, `y_range`, `vis_epochs`.

**`DepthTransformDiagnosticHook`** — 2×2 figure: sparse LiDAR depth on image, predicted depth (argmax of depthnet softmax), depth-distribution entropy, processed depth-feature activation. Designed to surface camera→BEV projection failures. Args: `vis_epochs`.

---

### 16. `mmdet3d/engine/hooks/depth_projection_debug_hook.py`

**`DepthProjectionDebugHook`** — detailed diagnostic of the DepthLSSTransform sparse-depth projection quality. Produces: LiDAR overlay on the augmented image, multi-hit collision analysis, occupancy stats at full-res and after dtransform, a 4-panel depth figure, and printed matrix shapes/values. Runs at configurable epochs (default `vis_epochs=(1,)` = epoch 1 only + final epoch) via `_should_visualize`. Args: `vis_epochs`.

---

### 17. `mmdet3d/engine/hooks/efficiency_hooks.py`

**`TrainingEfficiencyHook`** — lightweight per-epoch throughput / GPU memory metrics. DDP-safe (rank-0 only). Records samples/sec, iter/sec, peak VRAM per epoch.

**`RunSummaryHook`** — prints a compact summary block after `after_run`: run configuration, best and final val metrics, per-epoch efficiency numbers.

---

### 18. `mmdet3d/engine/hooks/validation_curve_hook.py`

**`ValidationCurveHook`** — records requested validation metric keys after each val epoch, writes a continuously-updated PNG line plot to `{work_dir}/visualizations/{filename}.png`, and a JSON sidecar for post-hoc analysis. Args: `metric_keys`, `filename`.

---

### 19. `mmdet3d/engine/hooks/fusion_training_hook.py`

**`FusionTrainingStrategyHook`** — earlier two-phase strategy hook (superseded by `FreezeLidarBranchHook` for the `fixedLidar` configs). Phase 1: freezes named LiDAR prefixes for `freeze_lidar_epochs` epochs. Phase 2: unfreezes everything for joint fine-tuning. Kept for reference / alternative configs. Args: `freeze_lidar_epochs`, `lidar_prefixes`.

---

### 20. `mmdet3d/engine/hooks/freeze_lidar_hook.py`

**`FreezeLidarBranchHook`** — hard-freeze hook for the LiDAR-only sub-modules of BEVFusion.

- `before_run`: calls `requires_grad_(False)` and `eval()` on all listed modules (default: `pts_voxel_encoder`, `pts_middle_encoder`, `pts_backbone`, `pts_neck`). Fires after DDP wrap so `find_unused_parameters=True` is required in `env_cfg`.
- `before_train_epoch`: re-applies `eval()` every epoch to override the training loop's `model.train()` call; if `unfreeze_epoch` is set and `runner.epoch >= unfreeze_epoch`, calls `requires_grad_(True)` + `train()` instead (one-time, permanent).

Args: `module_names`, `unfreeze_epoch` (optional; `None` = stay frozen forever).
**DDP requirement**: config must include `env_cfg = dict(dist_cfg=dict(backend='nccl', find_unused_parameters=True))`.
