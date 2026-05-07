# Evaluation Results

## Models

- **LiDAR-only checkpoint:** `outputs/runs/zod_lidar_only/zod-lidar-only_4543546/best_mAP_0.50_epoch_19.pth`
- **LiDAR-only config:** `mmdetection3d/configs/zod/zod_lidar_only.py`
- **Camera-only checkpoint:** `outputs/runs/zod_camera_only/zod-cam-only_4469392/best_mAP_0.50_epoch_11.pth`
- **Camera-only config:** `mmdetection3d/configs/zod/zod_camera_only.py`
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
| LiDAR-only | 0.5207 | 0.8125 |
| Camera-only | 0.0047 | 0.2101 |
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
| day | 0.5205 | 0.8192 | 0.0046 | 0.2105 | 0.5759 | 0.8305 |
| night | 0.5568 | 0.7664 | 0.0067 | 0.2123 | 0.6294 | 0.7926 |

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
| arterial_rural | 0.4420 | 0.5934 | 0.0005 | 0.0414 | 0.4500 | 0.5875 |
| arterial_urban | 0.5272 | 0.7924 | 0.0042 | 0.2057 | 0.5760 | 0.8116 |
| city | 0.5217 | 0.8204 | 0.0049 | 0.2139 | 0.5818 | 0.8343 |
| highway | 0.1434 | 0.1577 | 0.0000 | 0.0025 | 0.1980 | 0.3218 |
| smaller_rural | 0.3392 | 0.7358 | 0.0006 | 0.1679 | 0.4339 | 0.7700 |

### Quick read
- **Road type is the strongest context signal** across all models.
- **City** and **arterial_urban** are much easier than **arterial_rural** and **highway**.
- **Camera-only** is essentially unusable on **highway** and extremely weak on **arterial_rural**.
- **Fusion** improves on LiDAR-only almost everywhere, but does not remove the strong road-type dependence.

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
| clear_day | 0.5116 | 0.8046 | 0.0043 | 0.1979 | 0.5608 | 0.8143 |
| clear_night | 0.5292 | 0.7795 | 0.0033 | 0.1743 | 0.5833 | 0.8095 |
| cloudy | 0.5359 | 0.8281 | 0.0050 | 0.2084 | 0.5834 | 0.8421 |
| fog | 0.5208 | 0.7953 | 0.0110 | 0.2555 | 0.6288 | 0.8104 |
| partly_cloudy_day | 0.5387 | 0.8266 | 0.0039 | 0.2123 | 0.5819 | 0.8351 |
| partly_cloudy_night | 0.5492 | 0.7780 | 0.0056 | 0.2240 | 0.6034 | 0.7931 |
| precipitation | 0.4580 | 0.7905 | 0.0090 | 0.2221 | 0.5761 | 0.8187 |

### Quick read
- Weather matters, but less than **road type**.
- **Camera-only** shows meaningful variation in **AP@0.5m** across weather regimes.
- Best camera-only weather bins by **AP@0.5m** are **fog**, **partly_cloudy_night**, and **precipitation**.
- For **Fusion**, the general ordering of weather difficulty is similar to LiDAR-only.

---

## Weather group

Count  
clear_like: 2043  
cloud_like: 5928  

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | Camera AP@0.50 | Camera AP@0.5m | Fusion AP@0.50 | Fusion AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| clear_like | 0.5113 | 0.8005 | 0.0041 | 0.1955 | 0.5624 | 0.8136 |
| cloud_like | 0.5370 | 0.8213 | 0.0043 | 0.2125 | 0.5838 | 0.8345 |

### Quick read
- All three models do slightly better in **cloud_like** than **clear_like**.
- The effect is real but modest compared with **road type**.

---

## High-level interpretation

### LiDAR-only
- Strong across most contexts.
- **Road type** is the strongest context effect.
- Weather and lighting matter somewhat, but less.

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

---

## Final takeaway

- **Road type is the dominant context signal** across all models.
- **Weather is the most meaningful complementary signal**, especially for multimodal routing.
- Fusion improves overall performance but **does not eliminate context dependence**, reinforcing the motivation for context-aware MoE routing.
