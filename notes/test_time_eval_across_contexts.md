# Evaluation Results

## Models

- **LiDAR-only checkpoint:** `outputs/runs/zod_lidar_only/zod-lidar-only_4570893/best_mAP_0.50_epoch_30.pth`
- **LiDAR-only config:** `mmdetection3d/configs/zod/zod_lidar_only_30ep.py`
- **LiDAR-only-MoE checkpoint:** `outputs/runs/zod_lidar_only_moe/lidar-moe_4570728/best_mAP_0.50_epoch_29.pth`
- **LiDAR-only-MoE config:** `mmdetection3d/configs/zod/zod_lidar_only_moe_dense4_30ep.py`
- **Camera-only checkpoint:** `outputs/runs/zod_camera_only/zod-cam-only_4577582/best_mAP_0.50_epoch_31.pth`
- **Camera-only config:** `mmdetection3d/configs/zod/zod_camera_only_40ep.py`
- **Fusion checkpoint:** `outputs/runs/zod_bevfusion_dualinit/bevfusion-dualinit_4543552/best_mAP_0.50_epoch_11.pth`
- **Fusion config:** `mmdetection3d/configs/zod/zod_bevfusion_dualinit.py`

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
| Fusion | 0.5785 | 0.8277 |

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
| day | 0.5449 | 0.8202 | 0.0082 | 0.2400 | 0.5759 | 0.8305 |
| night | 0.5611 | 0.7756 | 0.0086 | 0.2426 | 0.6294 | 0.7926 |

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
| arterial_rural | 0.3699 | 0.5572 | 0.0018 | 0.0966 | 0.4500 | 0.5875 |
| arterial_urban | 0.5534 | 0.7950 | 0.0083 | 0.2313 | 0.5760 | 0.8116 |
| city | 0.5450 | 0.8217 | 0.0087 | 0.2443 | 0.5818 | 0.8343 |
| highway | 0.1659 | 0.1772 | 0.0000 | 0.0548 | 0.1980 | 0.3218 |
| smaller_rural | 0.3867 | 0.7319 | 0.0018 | 0.2230 | 0.4339 | 0.7700 |

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
fog: small subset  

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | Camera AP@0.50 | Camera AP@0.5m | Fusion AP@0.50 | Fusion AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| clear_day | 0.5338 | 0.8051 | 0.0073 | 0.2152 | 0.5608 | 0.8143 |
| clear_night | 0.4862 | 0.7680 | 0.0131 | 0.2134 | 0.5833 | 0.8095 |
| cloudy | 0.5556 | 0.8319 | 0.0086 | 0.2412 | 0.5834 | 0.8421 |
| fog | 0.4939 | 0.8021 | 0.0143 | 0.2997 | 0.6288 | 0.8104 |
| partly_cloudy_day | 0.5605 | 0.8272 | 0.0079 | 0.2455 | 0.5819 | 0.8351 |
| partly_cloudy_night | 0.5675 | 0.7837 | 0.0114 | 0.2398 | 0.6034 | 0.7931 |
| precipitation | 0.5090 | 0.7913 | 0.0134 | 0.2644 | 0.5761 | 0.8187 |

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
| clear_like | 0.5287 | 0.8000 | 0.0075 | 0.2094 | 0.5624 | 0.8136 |
| cloud_like | 0.5577 | 0.8234 | 0.0080 | 0.2435 | 0.5838 | 0.8345 |

### Quick read
- All three models do slightly better in **cloud_like** than **clear_like**.
- The effect is real but modest compared with **road type**.

---

## LiDAR-only vs LiDAR-only-MoE comparison

The updated LiDAR-only baseline is stronger than the LiDAR-only-MoE model on the main IoU metric. On the full test set, LiDAR-only reaches **0.5443 AP@0.50**, while LiDAR-only-MoE reaches **0.4384 AP@0.50**, a difference of **-0.1059** for the MoE model. However, the two models are almost identical on the coarse center-distance metric: **0.8140 AP@0.5m** for LiDAR-only versus **0.8110 AP@0.5m** for LiDAR-only-MoE.

This suggests that the LiDAR-only-MoE model preserves coarse localization ability fairly well, but currently hurts precise box quality / IoU-based detection. In other words, the MoE variant is not simply failing to find pedestrians; rather, it appears to degrade the quality or consistency of the final 3D boxes relative to the non-MoE LiDAR-only baseline.

### Full test

| Context | LiDAR AP@0.50 | LiDAR-MoE AP@0.50 | Δ AP@0.50 | LiDAR AP@0.5m | LiDAR-MoE AP@0.5m | Δ AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| full_test | 0.5443 | 0.4384 | -0.1059 | 0.8140 | 0.8110 | -0.0030 |

### Lighting

| Context | LiDAR AP@0.50 | LiDAR-MoE AP@0.50 | Δ AP@0.50 | LiDAR AP@0.5m | LiDAR-MoE AP@0.5m | Δ AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| day | 0.5449 | 0.4374 | -0.1075 | 0.8202 | 0.8167 | -0.0035 |
| night | 0.5611 | 0.4761 | -0.0850 | 0.7756 | 0.7752 | -0.0004 |

### Road type

| Context | LiDAR AP@0.50 | LiDAR-MoE AP@0.50 | Δ AP@0.50 | LiDAR AP@0.5m | LiDAR-MoE AP@0.5m | Δ AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| arterial_rural | 0.3699 | 0.3089 | -0.0610 | 0.5572 | 0.5322 | -0.0250 |
| arterial_urban | 0.5534 | 0.4660 | -0.0874 | 0.7950 | 0.7923 | -0.0027 |
| city | 0.5450 | 0.4342 | -0.1108 | 0.8217 | 0.8201 | -0.0016 |
| highway | 0.1659 | 0.0631 | -0.1028 | 0.1772 | 0.1284 | -0.0488 |
| smaller_rural | 0.3867 | 0.3692 | -0.0175 | 0.7319 | 0.7146 | -0.0173 |

### Scraped weather

| Context | LiDAR AP@0.50 | LiDAR-MoE AP@0.50 | Δ AP@0.50 | LiDAR AP@0.5m | LiDAR-MoE AP@0.5m | Δ AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| clear_day | 0.5338 | 0.4274 | -0.1064 | 0.8051 | 0.8036 | -0.0015 |
| clear_night | 0.4862 | 0.4471 | -0.0391 | 0.7680 | 0.7609 | -0.0071 |
| cloudy | 0.5556 | 0.4478 | -0.1078 | 0.8319 | 0.8236 | -0.0083 |
| fog | 0.4939 | 0.5157 | +0.0218 | 0.8021 | 0.7726 | -0.0295 |
| partly_cloudy_day | 0.5605 | 0.4445 | -0.1160 | 0.8272 | 0.8266 | -0.0006 |
| partly_cloudy_night | 0.5675 | 0.4539 | -0.1136 | 0.7837 | 0.7872 | +0.0035 |
| precipitation | 0.5090 | 0.4257 | -0.0833 | 0.7913 | 0.7851 | -0.0062 |

### Weather group

| Context | LiDAR AP@0.50 | LiDAR-MoE AP@0.50 | Δ AP@0.50 | LiDAR AP@0.5m | LiDAR-MoE AP@0.5m | Δ AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| clear_like | 0.5287 | 0.4274 | -0.1013 | 0.8000 | 0.7984 | -0.0016 |
| cloud_like | 0.5577 | 0.4452 | -0.1125 | 0.8234 | 0.8209 | -0.0025 |

### Complexity

| Context | LiDAR AP@0.50 | LiDAR-MoE AP@0.50 | Δ AP@0.50 | LiDAR AP@0.5m | LiDAR-MoE AP@0.5m | Δ AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| low | 0.5434 | 0.4642 | -0.0792 | 0.7833 | 0.7868 | +0.0035 |
| medium | 0.5586 | 0.4411 | -0.1175 | 0.8282 | 0.8281 | -0.0001 |
| high | 0.5520 | 0.4400 | -0.1120 | 0.8510 | 0.8498 | -0.0012 |

### Quick read
- **LiDAR-only-MoE is consistently worse than LiDAR-only on AP@0.50** across almost all context splits.
- The largest AP@0.50 drops occur in **partly_cloudy_day**, **medium complexity**, **cloud_like**, **city**, **clear_day**, **cloudy**, and **high complexity**.
- The main exception is **fog**, where LiDAR-only-MoE is slightly better on **AP@0.50**: **0.5157** versus **0.4939**.
- On **AP@0.5m**, LiDAR-only-MoE is usually very close to LiDAR-only.
- This means the MoE model mostly preserves coarse localization, but loses precision under IoU-based evaluation.
- The strongest negative signal is therefore not that the MoE model cannot localize pedestrians, but that its adaptive expert block may be disrupting the representation needed for accurate 3D box geometry.

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

### LiDAR-only-MoE
- Performs worse than the updated LiDAR-only baseline on **AP@0.50**.
- Remains very close to LiDAR-only on **AP@0.5m**.
- This pattern suggests that the current LiDAR-only-MoE setup preserves coarse localization but degrades precise box quality.
- The result is important because it shows that adding MoE capacity does not automatically improve the LiDAR-only model.
- The MoE block likely needs stronger stabilization, better placement, or more careful routing/expert regularization before it can outperform the non-adaptive LiDAR-only baseline.

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

- **Road type is the dominant context signal** across all models.
- **Weather is the most meaningful complementary signal**, especially for multimodal routing.
- Fusion improves overall performance but **does not eliminate context dependence**, reinforcing the motivation for context-aware MoE routing.
- The updated LiDAR-only baseline is stronger than the current LiDAR-only-MoE model on **AP@0.50**.
- The LiDAR-only-MoE model remains close on **AP@0.5m**, which suggests that it preserves coarse localization but currently harms precise 3D box quality.
- Therefore, the current MoE result should be interpreted as evidence that **adaptive routing is promising but not yet better than the strong LiDAR-only baseline in this configuration**.
