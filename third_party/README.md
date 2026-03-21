# Third-Party Dependencies

This project uses external repositories that are kept outside this repo to avoid code duplication and dependency conflicts.

## BEVFusion (MIT Han Lab)

- Local path: `/home/edgelab/bevfusion`
- Upstream: `https://github.com/mit-han-lab/bevfusion`
- Pinned commit: `326653dc06e0938edf1aae7d01efcd158ba83de5`

Notes:
- Keep BEVFusion code external.
- Keep thesis-owned code in `/home/edgelab/multimodal-MoE`.
- Use the `bevfusion38` environment/kernel when running BEVFusion code.

## MMDetection3D (OpenMMLab)

- Local path: `/home/edgelab/mmdetection3d`
- Upstream: `https://github.com/open-mmlab/mmdetection3d.git`
- Base version: `v1.4.0`
- Base commit: `fe25f7a51d36e3702f961e198894580d83c4387b`
- Environment: `multimodal-moe` conda env

We apply a small set of patches on top of the upstream release to support
our ZOD-MoE dataset and LiDAR-only BEVFusion pipeline.  Every change is
documented in `third_party/mmdetection3d_changes.md`.

Notes:
- The repo lives at `/home/edgelab/mmdetection3d` (not inside multimodal-MoE).
- Config files for ZOD live at `mmdetection3d/configs/zod/`.
- If you re-clone mmdetection3d, check out the base commit above and
  re-apply the patches listed in `mmdetection3d_changes.md`.
