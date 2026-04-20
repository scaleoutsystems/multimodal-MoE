# Evaluation Results

## Models

- **LiDAR-only checkpoint:** `outputs/runs/zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth`
- **LiDAR-only config:** `mmdetection3d/configs/zod/zod_lidar_only.py`
- **Camera-only checkpoint:** `outputs/runs/zod_camera_only/zod-cam-only_4469392/best_mAP_0.50_epoch_11.pth`
- **Camera-only config:** `mmdetection3d/configs/zod/zod_camera_only.py`
- **Fusion checkpoint:** `outputs/runs/zod_bevfusion_dualinit/bevfusion-dualinit_4481497/best_mAP_0.50_epoch_12.pth`
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
| LiDAR-only | 0.5468 | 0.7829 |
| Camera-only | 0.0047 | 0.2101 |
| Fusion | 0.5748 | 0.7893 |

### Quick read
- **Fusion** is the best overall model on **AP@0.50**.
- **LiDAR-only** and **Fusion** are very similar on **AP@0.5m**.
- **Camera-only** is very weak on IoU, but still has non-trivial coarse localization ability at **AP@0.5m**.

---

## Complexity

Count  
high: 869  
medium: 1493  
low: 2681  

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | Camera AP@0.50 | Camera AP@0.5m | Fusion AP@0.50 | Fusion AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| high | 0.5653 | 0.8401 | 0.0038 | 0.2093 | 0.5944 | 0.8386 |
| medium | 0.5694 | 0.8123 | 0.0058 | 0.2254 | 0.5912 | 0.8149 |
| low | 0.5585 | 0.7762 | 0.0083 | 0.2311 | 0.5880 | 0.7796 |

### Quick read
- **LiDAR-only** is fairly stable across complexity.
- **Camera-only** is best in **low** complexity and worst in **high** complexity, especially on **AP@0.5m**.
- **Fusion** improves over LiDAR-only on **AP@0.50** in all bins, but is very similar to LiDAR-only on **AP@0.5m**.

---

## Lighting

Count  
day: 7743  
night: 1875  

| Context | LiDAR AP@0.50 | LiDAR AP@0.5m | Camera AP@0.50 | Camera AP@0.5m | Fusion AP@0.50 | Fusion AP@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| day | 0.5483 | 0.7936 | 0.0046 | 0.2105 | 0.5750 | 0.7959 |
| night | 0.5611 | 0.7117 | 0.0067 | 0.2123 | 0.6000 | 0.7268 |

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
| arterial_rural | 0.3395 | 0.4333 | 0.0005 | 0.0414 | 0.3911 | 0.4969 |
| arterial_urban | 0.5367 | 0.7596 | 0.0042 | 0.2057 | 0.5619 | 0.7582 |
| city | 0.5554 | 0.7993 | 0.0049 | 0.2139 | 0.5832 | 0.8054 |
| highway | 0.1060 | 0.1184 | 0.0000 | 0.0025 | 0.1595 | 0.1654 |
| smaller_rural | 0.3653 | 0.6472 | 0.0006 | 0.1679 | 0.4047 | 0.6906 |

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
| clear_day | 0.5385 | 0.7731 | 0.0043 | 0.1979 | 0.5663 | 0.7736 |
| clear_night | 0.5089 | 0.7350 | 0.0033 | 0.1743 | 0.5837 | 0.7603 |
| cloudy | 0.5553 | 0.8059 | 0.0050 | 0.2084 | 0.5793 | 0.8057 |
| fog | 0.5584 | 0.7461 | 0.0110 | 0.2555 | 0.5812 | 0.7451 |
| partly_cloudy_day | 0.5624 | 0.8047 | 0.0039 | 0.2123 | 0.5872 | 0.8080 |
| partly_cloudy_night | 0.5847 | 0.7389 | 0.0056 | 0.2240 | 0.5855 | 0.7401 |
| precipitation | 0.5035 | 0.7462 | 0.0090 | 0.2221 | 0.5520 | 0.7640 |

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
| clear_like | 0.5337 | 0.7657 | 0.0041 | 0.1955 | 0.5656 | 0.7706 |
| cloud_like | 0.5598 | 0.7963 | 0.0043 | 0.2125 | 0.5836 | 0.8008 |

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
- **Road type** is again the strongest context effect:
  - **highway** is essentially dead
  - **arterial_rural** is also very weak
  - **city** and **arterial_urban** are much better
- Complexity also matters, with **low** complexity best.

### Fusion
- Best overall model on **AP@0.50**.
- Usually only slightly better than LiDAR-only on **AP@0.5m**.
- This suggests the main fusion gain is more about **better box quality** than dramatically better coarse localization.
- Fusion does **not** remove context dependence, especially for **road type**.

---

## Context groups most relevant for MoE routing

### Most relevant: `road_type`
This is the strongest and cleanest context signal across all three models.

Why it matters:
- The relative difficulty changes a lot between **city / arterial_urban** and **arterial_rural / highway**.
- This likely reflects major shifts in:
  - pedestrian frequency
  - scene geometry
  - visual context
  - range / scale / sparsity
- It is therefore a strong candidate for routing both:
  - **across modalities**
  - **within a single modality**

### Second most relevant: `complexity`
Why it matters:
- It changes camera reliability clearly.
- It is plausible that expert behavior should differ between sparse/simple scenes and crowded/occluded scenes.
- This is likely useful for:
  - multimodal routing
  - LiDAR-only routing
  - camera-only routing

### Third most relevant: weather / adverse regime
Why it matters:
- The weather effect is weaker than road type, but still visible.
- It is especially relevant if you want routing to respond to modality reliability shifts, since weather can change:
  - image quality
  - contrast
  - visibility
  - possibly LiDAR return quality in some cases

### Less relevant: lighting
Why:
- It does have some effect, but it is less consistent and less strong than road type.
- It may still be worth including, but it looks weaker as a primary routing signal.

---

## MoE routing: across modalities vs within a single modality

### Across modalities
For multimodal MoE, the most useful routing signals are the ones that likely change **relative modality usefulness**.

Best candidates:
1. **road_type**
2. **weather / adverse regime**
3. **complexity**

Why:
- These are the most plausible drivers of when camera adds value versus when LiDAR should dominate.

### Within a single modality
For LiDAR-only or camera-only MoE, the question is slightly different:
- not “which modality should matter more?”
- but “which kind of expert should handle this regime?”

Here, **road_type** still looks strongest.
After that:
- **complexity** is probably more useful than lighting
- **weather** may still matter, especially for camera-only

So there is overlap, but the interpretation changes:
- **multimodal routing** cares about relative modality complementarity
- **single-modality routing** cares more about regime-specific specialization within the same sensor stream

### Bottom line
- **road_type** is the strongest candidate for both multimodal and single-modality routing
- **complexity** is probably the next cleanest candidate
- **weather** is more relevant for multimodal routing than for LiDAR-only routing
- **lighting** currently looks like the weakest of the main candidates
