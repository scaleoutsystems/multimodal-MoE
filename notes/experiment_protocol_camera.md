# Camera Experiment Protocol (Apples-to-Apples Contract)

This document locks the camera-only training/evaluation protocol so comparisons are fair and reproducible across `YOLO26m` and third-party `RT-DETR-L`.

The values below are set to match the completed YOLO26m run configuration in:
- `outputs/runs/yolo/yolo26m_e50_1248x704_p10_noaug_seed0/args.yaml`

## 1) Dataset and Split Policy (Fixed)

- **Dataset export**: `outputs/exports/yolo/pedestrian_v1_exclude_unclear/dataset.yaml`
- **Class set**: single class (`pedestrian`)
- **Unclear-label policy**: `exclude_unclear` (must remain unchanged)
- **Split source (project canonical)**:
  - `ZOD` split IDs are fixed by:
    - `~/zod_moe/splits/train_ids.csv`
    - `~/zod_moe/splits/val_ids.csv`
    - `~/zod_moe/splits/test_ids.csv`
- **Exported split mapping for detector training** (via dataset yaml):
  - train: `images/train`
  - val: `images/val`
  - test: `images/test`

## 2) Image Size Policy (Fixed)

- **Training/eval image size**: `1248 x 704` (`img_w=1248`, `img_h=704`)
- Keep this resolution unchanged for all model-family comparisons in this protocol.

## 3) Training Schedule Policy (Fixed)

- **Epochs**: `50`
- **Patience**: `10`
- **Batch size**: `16`
- **Workers**: `8`
- **Device**: `0`
- **Seed**: `0`
- **Deterministic**: `true`
- **Rectangular batches**: `true`
- **AMP**: `true`

## 4) Optimizer Policy (Fixed)

- **Optimizer selection**: `auto` (Ultralytics auto policy)
- Keep the same base optimizer behavior across runs; do not manually tune one model family differently.
- Run-level hyperparameters to mirror:
  - `lr0=0.01`
  - `lrf=0.01`
  - `momentum=0.937`
  - `weight_decay=0.0005`
  - `warmup_epochs=3.0`
  - `warmup_momentum=0.8`
  - `warmup_bias_lr=0.1`

## 5) Augmentation Policy (Fixed)

To remain identical to the YOLO26m run, use the following augmentation settings:

- **Geometric strong augs disabled**:
  - `degrees=0.0`
  - `translate=0.0`
  - `scale=0.0`
  - `shear=0.0`
  - `perspective=0.0`
  - `mosaic=0.0`
  - `mixup=0.0`
  - `cutmix=0.0`
  - `copy_paste=0.0`
- **Other augmentation defaults that were still enabled in YOLO run**:
  - `fliplr=0.5`
  - `flipud=0.0`
  - `hsv_h=0.015`
  - `hsv_s=0.7`
  - `hsv_v=0.4`
  - `auto_augment=randaugment`
  - `erasing=0.4`

## 6) Evaluation Policy (Fixed)

- **Eval split**: `val`
- **Eval image size**: `1248x704`
- **Eval batch**: `16`
- **Device**: `0`
- **Rectangular eval**: `true` (preserve aspect ratio behavior used in YOLO eval path)
- **NMS IoU threshold**: `0.7` (match YOLO run/eval convention)
- **Max detections per image**: `300`
- **Confidence-threshold policy**:
  - Keep `conf` unset / framework default for primary benchmark runs (this matches YOLO run behavior where `conf: null`).
  - Do not tune model-specific confidence thresholds for headline comparison metrics.
  - Treat `mAP50` and `mAP50-95` as primary comparison metrics (they are threshold-swept ranking metrics, not single-threshold metrics).
  - If reporting single operating-point precision/recall, document the operating-point policy explicitly and use the same policy for both model families.
- **Metric output contract** (must be identical across model families):
  - `metrics.json`
  - `metrics_table.csv`
  - `run_metadata.json`
  - `run_metadata.csv`
- **Cross-family rule**: confidence/IoU/eval behavior must be kept consistent unless explicitly declared as an ablation.

## 7) Model Tier Policy (Locked)

- **Primary baseline comparison**: `YOLO26m` vs `RT-DETRv2-L`
- **Development-speed proxy**: `RT-DETRv2-M` (allowed for fast iteration while building adapters, logging, and MoE plumbing)
- **Headline thesis claims** must be reported on `RT-DETRv2-L` runs only.
- If `RT-DETRv2-M` results are shown, they must be explicitly labeled as development/proxy results, not final baseline claims.

## 8) Fairness Guardrails

- Do not change more than one factor at a time between compared runs.
- Any deviation from this contract must be logged in the run name and documented in notes before training starts.
- For thesis tables/plots, we only claim apples-to-apples comparison when both runs followed this exact protocol.
