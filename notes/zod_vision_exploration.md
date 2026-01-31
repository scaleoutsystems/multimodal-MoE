# ZOD Vision Exploration

# ZOD Camera Data & Vision-Relevant Details (Summary from ZOD Paper)

This document summarizes the parts of the ZOD paper that are most relevant
for **camera-based and vision-only exploration**, especially for pedestrian prediction and robustness analysis.

---

## Camera Sensor Characteristics

- ZOD uses **high-resolution 8MP RGB cameras**.
- Cameras are **front-looking**, mounted at the top of the windshield.
- Wide-angle fish-eye lenses are used to capture a large field of view.
- Raw camera data is processed using an internal, production-level
  image signal processor and provided as RGB images.

---

## Image Format and Frame Rate

- Camera images are captured at **10 Hz** (10 images per second --- 1 Hz = 1 image/sec).
- Images are provided in:
  - **JPG format** (recommended)
  - **Lossless PNG format** (also available)
- The authors report that JPG compression has **negligible impact on learning performance**, and therefore recommend using JPG images for benchmarking.

**Practical implication:**  
For experiments and baselines, JPG images are sufficient and easier to handle.

---

## ZOD Frames Dataset (Vision Perspective)

- ZOD Frames consists of **100,000 carefully curated frames**.
- Frames are selected to cover:
  - diverse traffic scenarios
  - different locations across Europe
  - varied weather and lighting conditions

### What a Frame contains (vision-relevant)
- One **RGB camera image** (the keyframe)
- Two anonymized versions of the image:
  - blurred (Faces / license plates are blurred)
  - DNAT (privacy-preserving anonymization)
  - DNAT = learned, privacy-preserving anonymization method. modifies pixels in a smarter way. Hides identity and better preserves object shape and texture --> better for learning than heavy blur. 
- The camera image is fully annotated and serves as the **keyframe**.
- keyframe = camera image at one moment. 
- All labels (bounding boxes, segmentation, etc.) are attached to this image (extra sensor data --  LiDAR before/after -- is just context)
- We will use JPG DNAT images. 

---

## Additional Sensor Context (Can Be Ignored Initially)

- Each frame also includes:
  - ±1 second of surrounding LiDAR scans (10 Hz)
  - high-precision GNSS / IMU data (100 Hz)
- These are provided for context and advanced use cases.

---

## Metadata Available Per Frame

Each frame is accompanied by structured metadata, including:

- timestamp
- geographic position
- country code
- weather condition (clear, rainy, foggy, snowy)
- solar elevation angle (lighting proxy)
- road type (e.g., city, highway)
- number of annotated objects (vehicles, pedestrians, vulnerable vehicles)

**Practical implication:**  
This metadata makes it possible to:
- stratify frames by condition (e.g., night vs day, rain vs clear)
- study robustness and failure modes systematically

---

## Annotations Relevant to Camera-Based Tasks

All annotations are created manually and quality-checked.

### Object Annotations (Most Relevant)
- All objects visible in the camera image are annotated with **2D bounding boxes**.
- Bounding boxes are tightly fitted and defined by pixel coordinates.
- Objects are labeled using a **hierarchical taxonomy**:
  - Dynamic objects (e.g., vehicles, pedestrians, vulnerable vehicles)
  - Static objects (e.g., signs, infrastructure)

### Object Properties
- Objects may include additional attributes such as:
  - occlusion rate
  - class-specific properties (e.g., emergency vehicle)

---

## Long-Range Perception (Important Context)

- The 8MP front camera and high-resolution LiDAR enable annotation of objects
  up to **245 meters** away.
- This supports:
  - long-range pedestrian and vehicle detection
  - high-speed driving scenarios (up to 133 km/h)

**Observed difficulty:**
- Detection performance drops sharply at long distances.
- This highlights the challenge of long-range perception and robustness.

---

## Dataset Diversity

- Data collected over **two years** across **14 European countries**.
- Covers:
  - snowy northern Sweden
  - sunny southern Europe
- High diversity in:
  - weather
  - lighting
  - road type
  - driving speed

---

## Anonymization and Vision Performance

- Experiments in the paper show:
  - no statistically significant performance drop when training on anonymized images compared to original images

---

## Key Takeaways for Camera Data Exploration

- ZOD Frames are ideal for **vision-only baselines**.
- Camera images are:
  - high-resolution
  - diverse
  - fully annotated
- Rich metadata enables condition-aware analysis.
- Long-range and long-tail perception remain challenging, making ZOD a meaningful benchmark for robustness studies.
