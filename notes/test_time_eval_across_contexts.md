# Evaluation Results

## Models

- **LiDAR-only checkpoint:** `outputs/runs/zod_lidar_only/zod-lidar-only_4570893/best_mAP_0.50_epoch_30.pth`
- **LiDAR-only config:** `mmdetection3d/configs/zod/zod_lidar_only_30ep.py`
- **LiDAR-only-MoE checkpoint:** `outputs/runs/zod_lidar_only_moe/lidar-moe_4570728/best_mAP_0.50_epoch_29.pth`
- **LiDAR-only-MoE config:** `mmdetection3d/configs/zod/zod_lidar_only_moe_dense4_30ep.py`
- **Camera-only checkpoint:** `outputs/runs/zod_camera_only/zod-cam-only_4577582/best_mAP_0.50_epoch_31.pth`
- **Camera-only config:** `mmdetection3d/configs/zod/zod_camera_only_40ep.py`
- **Fusion checkpoint:** `outputs/runs/zod_bevfusion_dualinit_28ep/bevfusion-dualinit-28ep_4585695/best_mAP_0.50_epoch_26.pth`
- **Fusion config:** `mmdetection3d/configs/zod/zod_bevfusion_dualinit_28ep.py`

---

## Metrics used

To keep the comparison focused and interpretable, we report only:

- **AP@0.50 (IoU)** — main 3D detection metric
- **AP@0.5m (center distance)** — coarse localization metric

This is especially useful because:

- **LiDAR-only** and **Fusion** are strong enough that **AP@0.50** is the main headline metric.
- **Camera-only** is very weak on IoU-based detection, so **AP@0.5m** is more informative for understanding whether it still provides useful coarse localization signal.

---

## Full test

| Model | AP@0.50 | AP@0.5m |
|---|---:|---:|
| LiDAR-only | 0.5443 | 0.8140 |
| Camera-only | 0.0086 | 0.2399 |
| Fusion | 0.5861 | 0.8280 |

### Quick read
- **Fusion** is the best overall model on **AP@0.50**.
- **LiDAR-only** and **Fusion** are very similar on **AP@0.5m**.
- **Camera-only** is very weak on IoU, but still has non-trivial coarse localization ability at **AP@0.5m**.

---

## Lighting

Count  
day: 7743  
night: 1875  

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | Camera AP@0.50 | Camera AP@0.5m | Fusion AP@0.50 | Fusion AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| day | 0.5449 | 0.8202 | 0.0082 | 0.2400 | 0.5842 | 0.8318 |
| night | 0.5611 | 0.7756 | 0.0086 | 0.2426 | 0.6269 | 0.7761 |

### Quick read
- **LiDAR-only** and **Fusion** are slightly better on **AP@0.50** at night, but worse on **AP@0.5m**.
- **Camera-only** is almost unchanged on **AP@0.5m** between day and night.
- Lighting matters, but less than road type.

---

## Road type

Count  
arterial_rural: 993  
arterial_urban: 2402  
city: 5022  
highway: 1070  
smaller_rural: 513  

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | Camera AP@0.50 | Camera AP@0.5m | Fusion AP@0.50 | Fusion AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| arterial_rural | 0.3699 | 0.5572 | 0.0018 | 0.0966 | 0.3949 | 0.5645 |
| arterial_urban | 0.5534 | 0.7950 | 0.0083 | 0.2313 | 0.5930 | 0.8115 |
| city | 0.5450 | 0.8217 | 0.0087 | 0.2443 | 0.5870 | 0.8342 |
| highway | 0.1659 | 0.1772 | 0.0000 | 0.0548 | 0.2208 | 0.2851 |
| smaller_rural | 0.3867 | 0.7319 | 0.0018 | 0.2230 | 0.4204 | 0.7930 |

### Quick read
- **Road type is the strongest context signal** across all models.
- **City** and **arterial_urban** are much easier than **arterial_rural** and **highway**.
- **Camera-only** is essentially unusable on **highway** and extremely weak on **arterial_rural**.
- **Fusion** improves on LiDAR-only everywhere on **AP@0.50**, but does not remove the strong road-type dependence.

---

## Scraped weather

Count  
clear_day: 1643  
clear_night: 400  
partly_cloudy_day: 2915  
partly_cloudy_night: 1113  
cloudy: 1900  
precipitation: 1768  
fog: 212 

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | Camera AP@0.50 | Camera AP@0.5m | Fusion AP@0.50 | Fusion AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| clear_day | 0.5338 | 0.8051 | 0.0073 | 0.2152 | 0.5677 | 0.8169 |
| clear_night | 0.4862 | 0.7680 | 0.0131 | 0.2134 | 0.5668 | 0.7721 |
| cloudy | 0.5556 | 0.8319 | 0.0086 | 0.2412 | 0.5882 | 0.8410 |
| fog | 0.4939 | 0.8021 | 0.0143 | 0.2997 | 0.5979 | 0.8252 |
| partly_cloudy_day | 0.5605 | 0.8272 | 0.0079 | 0.2455 | 0.5961 | 0.8375 |
| partly_cloudy_night | 0.5675 | 0.7837 | 0.0114 | 0.2398 | 0.6030 | 0.7882 |
| precipitation | 0.5090 | 0.7913 | 0.0134 | 0.2644 | 0.5782 | 0.8171 |

### Quick read
- Weather matters, but less than **road type**.
- **Camera-only** shows meaningful variation in **AP@0.5m** across weather regimes.
- Best camera-only weather bins by **AP@0.5m** are **fog**, **precipitation**, and **partly_cloudy_day**.
- For **Fusion**, the general ordering of weather difficulty is similar to LiDAR-only.

---

## Weather group

Count  
clear_like: 2043  
cloud_like: 5928  

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | Camera AP@0.50 | Camera AP@0.5m | Fusion AP@0.50 | Fusion AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| clear_like | 0.5287 | 0.8000 | 0.0075 | 0.2094 | 0.5675 | 0.8135 |
| cloud_like | 0.5577 | 0.8234 | 0.0080 | 0.2435 | 0.5928 | 0.8348 |

### Quick read
- All three models do slightly better in **cloud_like** than **clear_like**.
- The effect is real but modest compared with **road type**.

---

## Multimodal MoE fusion comparison

This comparison evaluates the strongest selected multimodal MoE variant from each fusion design against the standard Fusion baseline. Based on the model-selection runs, the best **Fusion-then-MoE** and **Joint-Modality MoE** variants use **road-type context supervision**, while the best **Modality-Specific MoE** variant is the fully **task-driven** variant.

Models compared:

- **Fusion baseline:** `outputs/runs/zod_bevfusion_dualinit_28ep/bevfusion-dualinit-28ep_4585695/best_mAP_0.50_epoch_26.pth`
- **Road Type CaMoE Fusion-then checkpoint:** `outputs/runs/zod_moe_fusion_then/moe-fusion-then_4758769/best_mAP_0.50_epoch_69.pth`
- **Road Type CaMoE Fusion-then config:** `mmdetection3d/configs/zod/zod_moe_fusion_then.py`
- **Road Type CaMoE Joint-Modality checkpoint:** `outputs/runs/zod_moe_joint_modality/moe-joint-modality_4758770/best_mAP_0.50_epoch_71.pth`
- **Road Type CaMoE Joint-Modality config:** `mmdetection3d/configs/zod/zod_moe_joint_modality.py`
- **Task-driven Modality-Specific MoE checkpoint:** `outputs/runs/zod_moe_modality_specific_taskdriven/moe-modality-specific-td_4758777/best_mAP_0.50_epoch_70.pth`
- **Task-driven Modality-Specific MoE config:** `mmdetection3d/configs/zod/zod_moe_modality_specific_taskdriven.py`

### Full test

| Model | AP@0.50 | AP@0.5m |
|---|---:|---:|
| Fusion | 0.5861 | 0.8280 |
| Road Type CaMoE Fusion-then | 0.5696 | 0.7846 |
| Road Type CaMoE Joint-Modality | 0.5681 | 0.7874 |
| Task-driven Modality-Specific MoE | 0.5693 | 0.7882 |

### Lighting

| Context | Fusion AP@0.50 | Fusion AP@0.5m | Road Type CaMoE Fusion-then AP@0.50 | Road Type CaMoE Fusion-then AP@0.5m | Road Type CaMoE Joint-Modality AP@0.50 | Road Type CaMoE Joint-Modality AP@0.5m | Task-driven Modality-Specific MoE AP@0.50 | Task-driven Modality-Specific MoE AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| day | 0.5842 | 0.8318 | 0.5687 | 0.7886 | 0.5658 | 0.7908 | 0.5664 | 0.7911 |
| night | 0.6269 | 0.7761 | 0.6025 | 0.7331 | 0.6091 | 0.7443 | 0.6329 | 0.7598 |

### Road type

| Context | Fusion AP@0.50 | Fusion AP@0.5m | Road Type CaMoE Fusion-then AP@0.50 | Road Type CaMoE Fusion-then AP@0.5m | Road Type CaMoE Joint-Modality AP@0.50 | Road Type CaMoE Joint-Modality AP@0.5m | Task-driven Modality-Specific MoE AP@0.50 | Task-driven Modality-Specific MoE AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arterial_rural | 0.3949 | 0.5645 | 0.4061 | 0.5289 | 0.3353 | 0.5511 | 0.3875 | 0.5476 |
| arterial_urban | 0.5930 | 0.8115 | 0.5668 | 0.7667 | 0.5699 | 0.7706 | 0.5726 | 0.7660 |
| city | 0.5870 | 0.8342 | 0.5732 | 0.7917 | 0.5703 | 0.7939 | 0.5713 | 0.7961 |
| highway | 0.2208 | 0.2851 | 0.1467 | 0.3450 | 0.2000 | 0.2898 | 0.2250 | 0.2392 |
| smaller_rural | 0.4204 | 0.7930 | 0.4218 | 0.7216 | 0.4257 | 0.7573 | 0.4257 | 0.7468 |

### Scraped weather

| Context | Fusion AP@0.50 | Fusion AP@0.5m | Road Type CaMoE Fusion-then AP@0.50 | Road Type CaMoE Fusion-then AP@0.5m | Road Type CaMoE Joint-Modality AP@0.50 | Road Type CaMoE Joint-Modality AP@0.5m | Task-driven Modality-Specific MoE AP@0.50 | Task-driven Modality-Specific MoE AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clear_day | 0.5677 | 0.8169 | 0.5560 | 0.7655 | 0.5449 | 0.7713 | 0.5543 | 0.7713 |
| clear_night | 0.5668 | 0.7721 | 0.5455 | 0.7209 | 0.5460 | 0.7177 | 0.5693 | 0.7413 |
| cloudy | 0.5882 | 0.8410 | 0.5825 | 0.8020 | 0.5709 | 0.8034 | 0.5737 | 0.8001 |
| fog | 0.5979 | 0.8252 | 0.6032 | 0.7652 | 0.5661 | 0.7673 | 0.5843 | 0.7460 |
| partly_cloudy_day | 0.5961 | 0.8375 | 0.5730 | 0.7927 | 0.5777 | 0.7968 | 0.5766 | 0.7988 |
| partly_cloudy_night | 0.6030 | 0.7882 | 0.5748 | 0.7508 | 0.6047 | 0.7645 | 0.6002 | 0.7605 |
| precipitation | 0.5782 | 0.8171 | 0.5621 | 0.7828 | 0.5596 | 0.7760 | 0.5587 | 0.7824 |

### Weather group

| Context | Fusion AP@0.50 | Fusion AP@0.5m | Road Type CaMoE Fusion-then AP@0.50 | Road Type CaMoE Fusion-then AP@0.5m | Road Type CaMoE Joint-Modality AP@0.50 | Road Type CaMoE Joint-Modality AP@0.5m | Task-driven Modality-Specific MoE AP@0.50 | Task-driven Modality-Specific MoE AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clear_like | 0.5675 | 0.8135 | 0.5547 | 0.7625 | 0.5445 | 0.7675 | 0.5548 | 0.7689 |
| cloud_like | 0.5928 | 0.8348 | 0.5765 | 0.7925 | 0.5768 | 0.7964 | 0.5765 | 0.7962 |

### Quick read

- The standard **Fusion** baseline remains the strongest model on the full test set, with **0.5861 AP@0.50** and **0.8280 AP@0.5m**.
- All three multimodal MoE variants are close to each other on the full test set, reaching **0.5681–0.5696 AP@0.50**, but all remain below the Fusion baseline.
- The AP@0.50 gap to Fusion is modest: **Road Type CaMoE Fusion-then** is lower by **0.0165**, **Road Type CaMoE Joint-Modality** by **0.0180**, and **Task-driven Modality-Specific MoE** by **0.0168**.
- The AP@0.5m gap is larger: the MoE variants reach **0.7846–0.7882 AP@0.5m**, compared with **0.8280** for Fusion. This suggests that the multimodal MoE variants do not improve coarse localization relative to the strong Fusion baseline.
- The **Task-driven Modality-Specific MoE** variant gives the best MoE result on **night AP@0.50**, reaching **0.6329**, slightly above the Fusion baseline at **0.6269**.
- Road-type subsets show the most interesting MoE behavior. **Road Type CaMoE Fusion-then** improves **arterial_rural AP@0.50** over Fusion (**0.4061** vs **0.3949**), while **Task-driven Modality-Specific MoE** improves **highway AP@0.50** over Fusion (**0.2250** vs **0.2208**).
- **Road Type CaMoE Joint-Modality** and **Task-driven Modality-Specific MoE** both reach **0.4257 AP@0.50** on **smaller_rural**, slightly above Fusion at **0.4204**.
- Weather-based improvements are small and localized. **Road Type CaMoE Fusion-then** improves **fog AP@0.50** over Fusion (**0.6032** vs **0.5979**), **Task-driven Modality-Specific MoE** improves **clear_night AP@0.50** (**0.5693** vs **0.5668**), and **Road Type CaMoE Joint-Modality** slightly improves **partly_cloudy_night AP@0.50** (**0.6047** vs **0.6030**).
- Overall, these multimodal MoE results do not outperform the strong Fusion baseline globally. The main positive signal is subset-specific robustness: different MoE designs improve different difficult regimes, especially **night**, **arterial_rural**, **highway**, **smaller_rural**, **fog**, and **partly_cloudy_night**.
- The results suggest that adaptive multimodal routing is promising for context-specific robustness, but the current selected MoE variants still trade away too much overall localization quality compared with the non-adaptive Fusion baseline.

---

## LiDAR-only MoE control comparison

This comparison isolates the effect of adding dense residual MoE routing to the
LiDAR-only detector. The LiDAR-only CaMoE model uses context-aware routing, while
the LiDAR-only TD-MoE model removes explicit context supervision and learns the
routing representation only through the downstream detection task.

### Full test

| Model | AP@0.50 | AP@0.5m |
|---|---:|---:|
| LiDAR-only | 0.5443 | 0.8140 |
| LiDAR-only CaMoE | 0.5293 | 0.8052 |
| LiDAR-only TD-MoE | 0.5380 | 0.8067 |

### Lighting

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | LiDAR CaMoE AP@0.50 | LiDAR CaMoE AP@0.5m | LiDAR TD-MoE AP@0.50 | LiDAR TD-MoE AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| day | 0.5449 | 0.8202 | 0.5283 | 0.8117 | 0.5343 | 0.8125 |
| night | 0.5611 | 0.7756 | 0.5771 | 0.7586 | 0.6059 | 0.7622 |

### Road type

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | LiDAR CaMoE AP@0.50 | LiDAR CaMoE AP@0.5m | LiDAR TD-MoE AP@0.50 | LiDAR TD-MoE AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| arterial_rural | 0.3699 | 0.5572 | 0.3449 | 0.5325 | 0.3664 | 0.5550 |
| arterial_urban | 0.5534 | 0.7950 | 0.5218 | 0.7833 | 0.5410 | 0.7917 |
| city | 0.5450 | 0.8217 | 0.5343 | 0.8132 | 0.5398 | 0.8128 |
| highway | 0.1659 | 0.1772 | 0.1578 | 0.1823 | 0.1856 | 0.2503 |
| smaller_rural | 0.3867 | 0.7319 | 0.3418 | 0.7410 | 0.4039 | 0.7441 |

### Scraped weather

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | LiDAR CaMoE AP@0.50 | LiDAR CaMoE AP@0.5m | LiDAR TD-MoE AP@0.50 | LiDAR TD-MoE AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| clear_day | 0.5338 | 0.8051 | 0.5237 | 0.7980 | 0.5273 | 0.8023 |
| clear_night | 0.4862 | 0.7680 | 0.5420 | 0.7600 | 0.5568 | 0.7689 |
| cloudy | 0.5556 | 0.8319 | 0.5412 | 0.8173 | 0.5506 | 0.8211 |
| fog | 0.4939 | 0.8021 | 0.4741 | 0.8045 | 0.5742 | 0.8114 |
| partly_cloudy_day | 0.5605 | 0.8272 | 0.5400 | 0.8200 | 0.5419 | 0.8166 |
| partly_cloudy_night | 0.5675 | 0.7837 | 0.5635 | 0.7685 | 0.5893 | 0.7736 |
| precipitation | 0.5090 | 0.7913 | 0.4851 | 0.7845 | 0.5112 | 0.7892 |

### Weather group

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | LiDAR CaMoE AP@0.50 | LiDAR CaMoE AP@0.5m | LiDAR TD-MoE AP@0.50 | LiDAR TD-MoE AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| clear_like | 0.5287 | 0.8000 | 0.5237 | 0.7932 | 0.5280 | 0.7977 |
| cloud_like | 0.5577 | 0.8234 | 0.5410 | 0.8133 | 0.5472 | 0.8131 |

### Quick read
- The original **LiDAR-only** baseline remains the strongest of the three on the full test set, with **0.5443 AP@0.50** and **0.8140 AP@0.5m**.
- **LiDAR-only TD-MoE** is closer to the LiDAR-only baseline than **LiDAR-only CaMoE** on the full test set, reaching **0.5380 AP@0.50** compared with **0.5293** for CaMoE.
- Both MoE variants remain close to the baseline on **AP@0.5m**, but neither improves over the LiDAR-only baseline overall.
- The TD-MoE variant improves over CaMoE in several difficult subsets, especially **night**, **highway**, **smaller_rural**, and **fog**, suggesting that direct task-driven routing is more effective than explicit context supervision in this LiDAR-only setting.
- The context-aware CaMoE result does not show a clear gain over the TD-MoE control. This suggests that explicit context infusion through the current road-type-supervised routing setup is not yet beneficial for the LiDAR-only detector.

---

## High-level interpretation

### LiDAR-only
- Strong across most contexts.
- **Road type** is the strongest context effect.
- Weather and lighting matter somewhat, but less.
- The updated 30-epoch LiDAR-only baseline is stronger than the previous LiDAR-only numbers, especially on **AP@0.50**.

### Camera-only
- Very weak on **AP@0.50**.
- Still shows non-trivial coarse localization on **AP@0.5m**.
- **Road type** is the strongest context effect:
  - **highway** is essentially dead
  - **arterial_rural** is very weak
  - **city** and **arterial_urban** are much better
- Weather also affects performance, especially for coarse localization.

### Fusion
- Best overall model on **AP@0.50**.
- Usually only slightly better than LiDAR-only on **AP@0.5m**.
- This suggests the main fusion gain is more about **better box quality** than dramatically better coarse localization.
- Fusion does **not** remove context dependence, especially for **road type**.

---

## Context groups most relevant for MoE routing

### Most relevant: `road_type`
- Strongest and clearest signal across all models.
- Separates fundamentally different scene regimes (urban vs rural vs highway).
- Affects both difficulty and model behavior.

### Second most relevant: `weather / weather_group`
- Directly linked to **sensor reliability shifts**, especially for the camera.
- Affects coarse localization more strongly than IoU in camera-only.
- Complements road type by capturing environmental rather than structural variation.

### Less relevant: `lighting`
- Has a measurable but smaller and less consistent effect.
- Likely a secondary signal rather than a primary routing variable.

---

## MoE routing interpretation

### Across modalities (camera vs LiDAR)
Most relevant signals:
1. **road_type**
2. **weather**

Why:
- These are the variables most likely to change **relative modality usefulness**.
- Road type captures scene structure.
- Weather captures sensing conditions.

### Within a single modality
Most relevant signals:
1. **road_type**
2. **weather (secondary)**

Why:
- Experts can specialize to different scene regimes (e.g. highway vs city).
- Weather may still affect feature quality, especially for camera.
- However, the current LiDAR-only-MoE results show that single-modality MoE specialization is not automatically beneficial. The model needs to improve precise box quality, not just preserve coarse localization.

---

## Final takeaway

* **Road type is the dominant context signal** across all models, and it remains the clearest axis along which model performance varies.
* **Weather is the most meaningful complementary signal**, especially for multimodal routing, because it can affect the relative reliability of camera and LiDAR features.
* The standard **Fusion** baseline remains the strongest overall model, reaching **0.5861 AP@0.50** and **0.8280 [AP@0.5m](mailto:AP@0.5m)** on the full test set.
* The three selected multimodal MoE variants do **not** outperform Fusion globally. Their full-test AP@0.50 scores are very similar to each other, between **0.5681** and **0.5696**, but below the Fusion baseline.
* The larger gap on **[AP@0.5m](mailto:AP@0.5m)** is due to the fact that the moe variants trained longer than the baseline and overfit on this metric which saturates more quickly than AP@0.5 IOU.
* However, the multimodal MoE results show useful **context-specific robustness signals**. Different MoE variants improve over Fusion in selected difficult subsets, including **night**, **arterial_rural**, **highway**, **smaller_rural**, **fog**, and **partly_cloudy_night**.
* In particular, the **Task-driven Modality-Specific MoE** is strongest among the MoE variants on some difficult cases, including **night AP@0.50** and **highway AP@0.50**, suggesting that direct detection-driven routing can be more useful than explicit context supervision in some regimes.
* The **Road Type CaMoE Fusion-then** variant improves over Fusion on **arterial_rural AP@0.50** and **fog AP@0.50**, suggesting that road-type-supervised context routing can help in specific environment-dependent cases, even if it does not improve the global average.
* Taken together, the results suggest that adaptive multimodal MoE routing is promising for **context-specific robustness**, but the current implementations do not yet improve over the strong non-adaptive Fusion baseline in overall detection accuracy.
* Therefore, the main conclusion is not that multimodal MoE is globally better than Fusion, but that it reveals targeted robustness gains that motivate further work on routing objectives, expert placement, training stability, and localization-preserving regularization.
