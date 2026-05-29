# Camera Contribution Ablation

**Checkpoint:** `/mnt/tier2/users/u103958/projects/multimodal-MoE/outputs/runs/zod_bevfusion_finetune/bevfusion-finetune_4456392/best_mAP_0.50_epoch_12.pth`  
**Full-model config:** `mmdetection3d/configs/zod/zod_bevfusion_finetune.py`  
**Camera-zero config:** `mmdetection3d/configs/zod/zod_bevfusion_finetune_camzero.py`  
**Test split:** `test` (main test set only)  

## Method

Two forward passes use the **same checkpoint weights**:

| Condition | Description |
| --- | --- |
| **Full model** | Normal BEVFusion forward (camera + LiDAR) |
| **Camera zeroed** | `extract_img_feat` returns `torch.zeros_like(cam_bev)` before fusion; LiDAR branch unchanged |

The delta column (`Full − Zero`) estimates how much the camera branch
contributes to each metric on top of the LiDAR signal already present
in the checkpoint.

## Results — main test set

| Metric | Full model | Camera zeroed | Delta (Full−Zero) |
| --- | --- | --- | --- |
| `mAP_0.25` | 0.7671 | 0.6771 | +0.0900 |
| `mAP_0.50` | 0.5655 | 0.4061 | +0.1594 |
| `mAP_0.5m` | 0.7878 | 0.6980 | +0.0898 |
| `mAP_1.0m` | 0.7952 | 0.7063 | +0.0889 |
| `mAP_2.0m` | 0.8005 | 0.7115 | +0.0890 |
| `mAP_4.0m` | 0.8096 | 0.7210 | +0.0886 |
| `data_time` | 0.0089 | 0.0095 | -0.0006 |
| `mAR_0.25` | 0.9345 | 0.8713 | +0.0632 |
| `mAR_0.50` | 0.7316 | 0.6156 | +0.1160 |
| `mAR_0.5m` | 0.9607 | 0.8930 | +0.0677 |
| `mAR_1.0m` | 0.9674 | 0.8989 | +0.0685 |
| `mAR_2.0m` | 0.9719 | 0.9032 | +0.0687 |
| `mAR_4.0m` | 0.9794 | 0.9109 | +0.0685 |
| `pedestrian_AP_0.25` | 0.7671 | 0.6771 | +0.0900 |
| `pedestrian_AP_0.50` | 0.5655 | 0.4061 | +0.1594 |
| `pedestrian_AP_0.5m` | 0.7878 | 0.6980 | +0.0898 |
| `pedestrian_AP_1.0m` | 0.7952 | 0.7063 | +0.0889 |
| `pedestrian_AP_2.0m` | 0.8005 | 0.7115 | +0.0890 |
| `pedestrian_AP_4.0m` | 0.8096 | 0.7210 | +0.0886 |
| `pedestrian_rec_0.25` | 0.9345 | 0.8713 | +0.0632 |
| `pedestrian_rec_0.50` | 0.7316 | 0.6156 | +0.1160 |
| `pedestrian_rec_0.5m` | 0.9607 | 0.8930 | +0.0677 |
| `pedestrian_rec_1.0m` | 0.9674 | 0.8989 | +0.0685 |
| `pedestrian_rec_2.0m` | 0.9719 | 0.9032 | +0.0687 |
| `pedestrian_rec_4.0m` | 0.9794 | 0.9109 | +0.0685 |
| `time` | 0.2650 | 0.2702 | -0.0052 |

## Interpretation

- A **positive delta** means the full model (with camera) outperforms
  the camera-zeroed baseline, indicating a **positive camera contribution**.
- A **near-zero or negative delta** suggests the camera branch is not yet
  helpful for that metric, or is even interfering with the LiDAR signal.

## Raw eval outputs

- Full model: `/mnt/tier2/users/u103958/projects/multimodal-MoE/outputs/runs/zod_bevfusion_finetune/bevfusion-finetune_4456392/ablation/full_model`
- Camera zeroed: `/mnt/tier2/users/u103958/projects/multimodal-MoE/outputs/runs/zod_bevfusion_finetune/bevfusion-finetune_4456392/ablation/camera_zero`
