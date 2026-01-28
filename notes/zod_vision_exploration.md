# ZOD Vision Exploration (Pre-access)

Goal: Understand the vision-only portion of ZOD for pedestrian prediction.

## vision-only:

- RGB camera images only
- No LiDAR, radar, or depth inputs
- Uses image data + metadata (weather, lighting, time)
- Task: pedestrian presence (binary)

## Things to inspect once ZOD access is granted

- How many cameras exist and their viewpoints
- Image resolution and frame rate
- Available pedestrian labels (2D boxes, attributes)
- Metadata fields related to lighting and weather
- How often pedestrians appear in difficult conditions

## Why vision-only can fail

Vision-only pedestrian detection can degrade under low light, bad weather, occlusion, and motion blur, which makes it a useful baseline for studying robustness.


## Public ZOD structure (from ZOD documentation)

- Camera data organization:
- Number of cameras:
- Frame vs sequence structure:
- Available metadata:
