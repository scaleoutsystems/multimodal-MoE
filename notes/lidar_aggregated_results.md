# Evaluation Results

## Models

- **LiDAR-only checkpoint:** `outputs/runs/zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth`
- **LiDAR-only config:** `mmdetection3d/configs/zod/zod_lidar_only.py`
- **Camera-only checkpoint:** `outputs/runs/zod_camera_only/zod-cam-only_4469392/best_mAP_0.50_epoch_11.pth`
- **Camera-only config:** `mmdetection3d/configs/zod/zod_camera_only.py`

---

## Notes on interpretation

- For **LiDAR-only**, the most useful primary metric is usually **mAP@0.50**, with **mAP@0.5m** as a strong secondary localization-oriented metric.
- For **Camera-only**, **mAP@0.50 is near-zero almost everywhere**, so the more informative metrics are **mAP@0.5m**, **mAP@1.0m**, **mAP@2.0m**, and their recalls.
- The tables below put both models side by side per context so you can directly compare how context affects each modality.
- Count per category is provided since it's important to keep in mind that metrics from smaller test sets have higher variance than those from larger ones. 

---

## Full test

| Model | mAP@0.25 | Rec@0.25 | mAP@0.50 | Rec@0.50 | mAP@0.5m | Rec@0.5m | mAP@1.0m | Rec@1.0m | mAP@2.0m | Rec@2.0m | mAP@4.0m | Rec@4.0m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LiDAR-only | 0.7630 | 0.9378 | 0.5468 | 0.7153 | 0.7829 | 0.9642 | 0.7895 | 0.9703 | 0.7947 | 0.9746 | - | - |
| Camera-only | 0.0970 | 0.3309 | 0.0047 | 0.0586 | 0.2101 | 0.5244 | 0.2893 | 0.6470 | 0.3333 | 0.7320 | 0.3649 | 0.7969 |

---

## Complexity

Count  
high: 869  
medium: 1493  
low: 2681  

| Context | Model | mAP@0.25 | Rec@0.25 | mAP@0.50 | Rec@0.50 | mAP@0.5m | Rec@0.5m | mAP@1.0m | Rec@1.0m | mAP@2.0m | Rec@2.0m | mAP@4.0m | Rec@4.0m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high | LiDAR-only | 0.8153 | 0.9386 | 0.5653 | 0.7083 | 0.8401 | 0.9641 | 0.8471 | 0.9710 | 0.8528 | 0.9744 | - | - |
| high | Camera-only | 0.0849 | 0.2895 | 0.0038 | 0.0492 | 0.2093 | 0.4760 | 0.3005 | 0.5946 | 0.3557 | 0.6832 | 0.3931 | 0.7507 |
| low | LiDAR-only | 0.7593 | 0.9400 | 0.5585 | 0.7293 | 0.7762 | 0.9642 | 0.7823 | 0.9699 | 0.7872 | 0.9742 | - | - |
| low | Camera-only | 0.1305 | 0.4011 | 0.0083 | 0.0745 | 0.2311 | 0.6061 | 0.2952 | 0.7240 | 0.3251 | 0.7980 | 0.3496 | 0.8563 |
| medium | LiDAR-only | 0.7909 | 0.9352 | 0.5694 | 0.7167 | 0.8123 | 0.9643 | 0.8200 | 0.9694 | 0.8257 | 0.9751 | - | - |
| medium | Camera-only | 0.1110 | 0.3477 | 0.0058 | 0.0612 | 0.2254 | 0.5475 | 0.3031 | 0.6756 | 0.3453 | 0.7604 | 0.3792 | 0.8275 |

### Quick read
- **LiDAR-only:** complexity has limited impact on mAP@0.50; all bins are close.
- **Camera-only:** low complexity is best; high complexity hurts especially at coarse recall/localization thresholds.

---

## Lighting

Count  
day: 7743  
night: 1875  

| Context | Model | mAP@0.25 | Rec@0.25 | mAP@0.50 | Rec@0.50 | mAP@0.5m | Rec@0.5m | mAP@1.0m | Rec@1.0m | mAP@2.0m | Rec@2.0m | mAP@4.0m | Rec@4.0m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| day | LiDAR-only | 0.7723 | 0.9360 | 0.5483 | 0.7104 | 0.7936 | 0.9632 | 0.8003 | 0.9694 | 0.8056 | 0.9739 | - | - |
| day | Camera-only | 0.0956 | 0.3247 | 0.0046 | 0.0569 | 0.2105 | 0.5195 | 0.2910 | 0.6426 | 0.3362 | 0.7280 | 0.3687 | 0.7954 |
| night | LiDAR-only | 0.7044 | 0.9628 | 0.5611 | 0.7983 | 0.7117 | 0.9768 | 0.7183 | 0.9833 | 0.7227 | 0.9851 | - | - |
| night | Camera-only | 0.1192 | 0.4210 | 0.0067 | 0.0706 | 0.2123 | 0.6227 | 0.2650 | 0.7147 | 0.2995 | 0.7900 | 0.3228 | 0.8448 |

### Quick read
- **LiDAR-only:** lighting barely changes mAP@0.50, though center-distance metrics shift more.
- **Camera-only:** night has slightly higher mAP@0.50 and recall, but worse mAP at looser distance thresholds than day in some places; overall the differences are modest compared with the overall weakness of the camera branch.

---

## Road type

Count  
arterial_rural: 993  
arterial_urban: 2402  
city: 5022  
highway: 1070  
smaller_rural: 513  

| Context | Model | mAP@0.25 | Rec@0.25 | mAP@0.50 | Rec@0.50 | mAP@0.5m | Rec@0.5m | mAP@1.0m | Rec@1.0m | mAP@2.0m | Rec@2.0m | mAP@4.0m | Rec@4.0m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arterial_rural | LiDAR-only | 0.4198 | 0.8154 | 0.3395 | 0.5231 | 0.4333 | 0.8923 | 0.4333 | 0.8923 | 0.4335 | 0.8923 | - | - |
| arterial_rural | Camera-only | 0.0118 | 0.2769 | 0.0005 | 0.0154 | 0.0414 | 0.4462 | 0.0491 | 0.5692 | 0.0615 | 0.7077 | 0.0666 | 0.7846 |
| arterial_urban | LiDAR-only | 0.7488 | 0.9479 | 0.5367 | 0.7215 | 0.7596 | 0.9661 | 0.7667 | 0.9692 | 0.7723 | 0.9722 | - | - |
| arterial_urban | Camera-only | 0.0972 | 0.3291 | 0.0042 | 0.0565 | 0.2057 | 0.5287 | 0.2752 | 0.6594 | 0.3191 | 0.7441 | 0.3486 | 0.8032 |
| city | LiDAR-only | 0.7766 | 0.9365 | 0.5554 | 0.7156 | 0.7993 | 0.9646 | 0.8057 | 0.9711 | 0.8110 | 0.9757 | - | - |
| city | Camera-only | 0.0981 | 0.3302 | 0.0049 | 0.0586 | 0.2139 | 0.5233 | 0.2967 | 0.6443 | 0.3418 | 0.7289 | 0.3745 | 0.7968 |
| highway | LiDAR-only | 0.1184 | 0.4500 | 0.1060 | 0.3000 | 0.1184 | 0.5000 | 0.1586 | 0.7000 | 0.1647 | 0.7500 | - | - |
| highway | Camera-only | 0.0007 | 0.1000 | 0.0000 | 0.0000 | 0.0025 | 0.2000 | 0.0058 | 0.3500 | 0.0063 | 0.3500 | 0.0065 | 0.3500 |
| smaller_rural | LiDAR-only | 0.5923 | 0.9028 | 0.3653 | 0.6181 | 0.6472 | 0.9514 | 0.6497 | 0.9722 | 0.6505 | 0.9792 | - | - |
| smaller_rural | Camera-only | 0.0844 | 0.3472 | 0.0006 | 0.0139 | 0.1679 | 0.6042 | 0.2490 | 0.7292 | 0.2783 | 0.7986 | 0.2897 | 0.8681 |

### Quick read
- **Road type is the strongest context signal for both models.**
- **LiDAR-only:** city / arterial_urban are much easier than arterial_rural / highway.
- **Camera-only:** highway is essentially dead; arterial_rural is also very weak. City and arterial_urban are clearly better.

---

## Scraped weather

Count  
clear_day: 1643  
clear_night: 400  
partly_cloudy_day: 2915  
partly_cloudy_night: 1113  
cloudy: 1900  
precipitation: 1768  
fog: small subset but included  

| Context | Model | mAP@0.25 | Rec@0.25 | mAP@0.50 | Rec@0.50 | mAP@0.5m | Rec@0.5m | mAP@1.0m | Rec@1.0m | mAP@2.0m | Rec@2.0m | mAP@4.0m | Rec@4.0m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clear_day | LiDAR-only | 0.7511 | 0.9278 | 0.5385 | 0.7040 | 0.7731 | 0.9580 | 0.7785 | 0.9640 | 0.7846 | 0.9679 | - | - |
| clear_day | Camera-only | 0.0827 | 0.3172 | 0.0043 | 0.0577 | 0.1979 | 0.5198 | 0.2781 | 0.6359 | 0.3213 | 0.7163 | 0.3539 | 0.7873 |
| clear_night | LiDAR-only | 0.7304 | 0.9713 | 0.5089 | 0.7527 | 0.7350 | 0.9749 | 0.7418 | 0.9857 | 0.7445 | 0.9857 | - | - |
| clear_night | Camera-only | 0.1014 | 0.3763 | 0.0033 | 0.0609 | 0.1743 | 0.5986 | 0.2243 | 0.7133 | 0.2445 | 0.7814 | 0.2700 | 0.8602 |
| cloudy | LiDAR-only | 0.7798 | 0.9381 | 0.5553 | 0.7162 | 0.8059 | 0.9673 | 0.8125 | 0.9733 | 0.8160 | 0.9776 | - | - |
| cloudy | Camera-only | 0.0981 | 0.3180 | 0.0050 | 0.0590 | 0.2084 | 0.5101 | 0.2902 | 0.6354 | 0.3389 | 0.7253 | 0.3726 | 0.7918 |
| fog | LiDAR-only | 0.7191 | 0.9000 | 0.5584 | 0.6895 | 0.7461 | 0.9474 | 0.7517 | 0.9474 | 0.7528 | 0.9579 | - | - |
| fog | Camera-only | 0.1435 | 0.4211 | 0.0110 | 0.0579 | 0.2555 | 0.5895 | 0.3145 | 0.6684 | 0.3492 | 0.7474 | 0.3665 | 0.7895 |
| partly_cloudy_day | LiDAR-only | 0.7858 | 0.9411 | 0.5624 | 0.7203 | 0.8047 | 0.9653 | 0.8110 | 0.9716 | 0.8172 | 0.9758 | - | - |
| partly_cloudy_day | Camera-only | 0.0963 | 0.3262 | 0.0039 | 0.0544 | 0.2123 | 0.5239 | 0.2931 | 0.6504 | 0.3371 | 0.7349 | 0.3693 | 0.7985 |
| partly_cloudy_night | LiDAR-only | 0.7281 | 0.9516 | 0.5847 | 0.7680 | 0.7389 | 0.9708 | 0.7438 | 0.9753 | 0.7477 | 0.9790 | - | - |
| partly_cloudy_night | Camera-only | 0.1076 | 0.3954 | 0.0056 | 0.0694 | 0.2240 | 0.5927 | 0.2903 | 0.6831 | 0.3226 | 0.7726 | 0.3471 | 0.8320 |
| precipitation | LiDAR-only | 0.7306 | 0.9371 | 0.5035 | 0.6962 | 0.7462 | 0.9636 | 0.7564 | 0.9698 | 0.7616 | 0.9751 | - | - |
| precipitation | Camera-only | 0.1166 | 0.3530 | 0.0090 | 0.0675 | 0.2221 | 0.5306 | 0.3039 | 0.6562 | 0.3480 | 0.7378 | 0.3781 | 0.8073 |

### Quick read
- **LiDAR-only:** weather matters somewhat, but not dramatically compared with road type.
- **Camera-only:** the weather splits show real variation in coarse localization signal.
- Best camera-only weather regimes by **mAP@0.5m** are **fog**, **partly_cloudy_night**, and **precipitation**; worst is **clear_night**.

---

## Weather group

Count  
clear_like: 2043  
cloud_like: 5928  

| Context | Model | mAP@0.25 | Rec@0.25 | mAP@0.50 | Rec@0.50 | mAP@0.5m | Rec@0.5m | mAP@1.0m | Rec@1.0m | mAP@2.0m | Rec@2.0m | mAP@4.0m | Rec@4.0m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clear_like | LiDAR-only | 0.7454 | 0.9306 | 0.5337 | 0.7071 | 0.7657 | 0.9591 | 0.7711 | 0.9654 | 0.7769 | 0.9690 | - | - |
| clear_like | Camera-only | 0.0833 | 0.3226 | 0.0041 | 0.0572 | 0.1955 | 0.5239 | 0.2740 | 0.6415 | 0.3164 | 0.7195 | 0.3479 | 0.7896 |
| cloud_like | LiDAR-only | 0.7758 | 0.9408 | 0.5598 | 0.7224 | 0.7963 | 0.9664 | 0.8025 | 0.9725 | 0.8076 | 0.9767 | - | - |
| cloud_like | Camera-only | 0.0977 | 0.3299 | 0.0043 | 0.0564 | 0.2125 | 0.5270 | 0.2911 | 0.6482 | 0.3364 | 0.7357 | 0.3682 | 0.7998 |

### Quick read
- Both models do better in **cloud_like** than **clear_like**.
- For LiDAR-only, the difference is modest.
- For Camera-only, the difference is still modest, but consistent at the distance-based metrics.

---

## High-level interpretation

### 1. LiDAR-only
- Strong across most contexts.
- **Road type** is the strongest context effect by far.
- **Weather** matters somewhat, especially precipitation.
- **Lighting** has limited effect on mAP@0.50.
- **Complexity** does not seem to be a major driver.

### 2. Camera-only
- Very weak at IoU-based 3D detection:
  - **mAP@0.50 is near-zero almost everywhere**
- But it has non-trivial coarse localization signal:
  - **mAP@0.5m ~ 0.17 to 0.26**
  - **mAP@1.0m / 2.0m** and recall are much stronger
- **Road type** is again the strongest context effect:
  - highway is essentially unusable
  - arterial_rural is also very weak
  - city / arterial_urban are much better
- Weather and lighting do affect the camera branch, but less dramatically than road type.

### 3. Most important shared takeaway
The strongest common context signal across both modalities is **road_type**. That makes `road_type` a very plausible routing signal for MoE, especially for any scene-regime or hierarchical routing idea.

A second shared signal is **weather / adverse regime**, though it is weaker than road type in the current baselines.

### 4. Metric takeaway
- For **LiDAR-only**, prioritize:
  - **mAP@0.50** as primary
  - **mAP@0.5m** as secondary
- For **Camera-only**, prioritize:
  - **mAP@0.5m**
  - **mAP@1.0m**
  - **mAP@2.0m**
  - and recall at those thresholds

Because camera-only is too weak for **mAP@0.50** to be the main diagnostic.

---