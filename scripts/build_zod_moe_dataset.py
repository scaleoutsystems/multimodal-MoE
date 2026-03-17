#!/usr/bin/env python3
"""
Build a ZOD-based MMDetection3D/BEVFusion dataset from an existing metadata parquet.


Reads source parquet: ZODmoe_frames_with_xyxy_bboxes_and_solar_bins.parquet
Processes one frame at a time
Rebuilds ZodFrame from raw frame folder using the same make_zodframe_from_raw path-fix logic
Uses exact camera undistortion flow (cv2.fisheye, balance=0.3) and computes K_final = S @ new_K
Saves final camera image as .png to /home/edgelab/zod_moe/images/{frame_id}.png
Selects nearest LiDAR sweep to keyframe timestamp, motion-compensates to camera timestamp, applies final-image FoV filtering with K_final, and saves ego-frame XYZI float32 .bin
Filters pedestrian 3D boxes by projected center inside final image, converts kept boxes to ego [x, y, z, dx, dy, dz, yaw], saves /home/edgelab/zod_moe/labels/{frame_id}.json
Saves per-frame calibration .npz with:
K_final, distortion, camera2ego, ego2image, img_aug_matrix, fusion_aug_matrix, image_size
Recomputes and overwrites the requested pedestrian parquet columns
Drops ped_points_xy_resized and ped_bin_4
Adds:
final_img_file_path, final_points_file_path, calib_file_path, label_file_path, num_pedestrians_final
Writes new parquet once at end to:
/home/edgelab/zod_moe/index/zod_moe_dataset.parquet
Keeps source parquet unchanged
Continues on frame failures and prints final summary


Per frame:
1) Build ZodFrame from raw single-frame folder.
2) Undistort + resize FRONT camera image (balance=0.3) and compute final intrinsics.
3) Select nearest LiDAR sweep to keyframe, deskew to camera timestamp.
4) FoV-filter deskewed LiDAR using final camera geometry, save ego-frame XYZI .bin.
5) FoV-filter pedestrian 3D boxes by final image center projection, export ego-frame boxes.
6) Save per-frame calibration (.npz), labels (.json), image (.png).
7) Build an updated row for a new output parquet.


# Smoke test (first 50 frames, writes separate parquet)
python scripts/build_zod_moe_dataset.py --limit 10 --target-parquet /home/edgelab/zod_moe/index/zod_moe_dataset_smoke.parquet


# Full run (all frames, default target parquet path)
python scripts/build_zod_moe_dataset.py
"""


from __future__ import annotations


import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


from zod.constants import AnnotationProject, Anonymization, Camera, Lidar
from zod.data_classes.frame import ZodFrame
from zod.data_classes.info import Information
from zod.data_classes.sensor import LidarData
from zod.utils.compensation import motion_compensate_pointwise
from zod.utils.geometry import transform_points
from zod.visualization.lidar_on_image import get_3d_transform_camera_lidar




# Compatibility shim for SDK code that still uses np.cast[...]
if not hasattr(np, "cast"):
   class _CastShim:
       def __getitem__(self, dtype):
           return lambda arr: np.asarray(arr, dtype=dtype)


   np.cast = _CastShim()




# ------------------------------------------------------------
# Paths and constants
# ------------------------------------------------------------
SOURCE_PARQUET = Path(
   "/home/edgelab/multimodal-MoE/outputs/index/ZODmoe_frames_with_xyxy_bboxes_and_solar_bins.parquet"
)


OUT_ROOT = Path("/home/edgelab/zod_moe")
OUT_IMAGES = OUT_ROOT / "bev_images"
OUT_LIDAR = OUT_ROOT / "lidar"
OUT_CALIBS = OUT_ROOT / "calibs"
OUT_LABELS = OUT_ROOT / "labels"
OUT_INDEX = OUT_ROOT / "index"
TARGET_PARQUET = Path("/home/edgelab/multimodal-MoE/outputs/index/zod_moe_dataset.parquet")


DATASET_ROOT_FALLBACK = Path("/home/edgelab/zod_dino_data/train2017")


FINAL_W = 1248
FINAL_H = 704
UNDISTORT_BALANCE = 0.3
EPS_Z = 1e-6




KEEP_COLUMNS = [
   "frame_id",
   "time",
   "image_path",
   "resized_image_path",
   "orig_w",
   "orig_h",
   "new_w",
   "new_h",
   "sx",
   "sy",
   "scraped_weather",
   "time_of_day",
   "solar_angle_elevation",
   "country_code",
   "road_type",
   "road_condition",
   "xyxy_bboxes",
   "solar_context_bin",
]


RECOMPUTE_PEDESTRIAN_COLUMNS = [
   "ped_count_clear",
   "ped_count_unclear",
   "ped_occ_none",
   "ped_occ_light",
   "ped_occ_medium",
   "ped_occ_heavy",
   "ped_occ_veryheavy",
   "ped_occ_missing",
   "ped_occ_unknown",
   "ped_uuid",
   "ped_unclear_list",
   "ped_occlusion_list",
   "ped_present",
]


DROP_COLUMNS = [
   "ped_points_xy_resized",
   "ped_bin_4",
]


FINAL_COLUMNS = [
   "frame_id",
   "time",
   "image_path",
   "resized_image_path",
   "orig_w",
   "orig_h",
   "new_w",
   "new_h",
   "sx",
   "sy",
   "scraped_weather",
   "time_of_day",
   "solar_angle_elevation",
   "country_code",
   "road_type",
   "road_condition",
   "xyxy_bboxes",
   "solar_context_bin",
   "ped_count_clear",
   "ped_count_unclear",
   "ped_occ_none",
   "ped_occ_light",
   "ped_occ_medium",
   "ped_occ_heavy",
   "ped_occ_veryheavy",
   "ped_occ_missing",
   "ped_occ_unknown",
   "ped_uuid",
   "ped_unclear_list",
   "ped_occlusion_list",
   "ped_present",
   "final_img_file_path",
   "final_points_file_path",
   "calib_file_path",
   "label_file_path",
   "num_pedestrians_final",
]




# ------------------------------------------------------------
# Helpers from notebook/script logic
# ------------------------------------------------------------
def _read_json(path: Path) -> Any:
   try:
       return json.loads(path.read_text())
   except Exception:
       return None




def _iso_to_dt(s: str) -> datetime:
   return datetime.fromisoformat(s.replace("Z", "+00:00"))




def nearest_lidar_filename(frame_dir: Path) -> tuple[str | None, str | None, float | None]:
   """
   Select the single nearest LiDAR sweep to keyframe_time from info.json.
   """
   info = _read_json(frame_dir / "info.json")
   lidar_dir = frame_dir / "lidar_velodyne"


   if not isinstance(info, dict) or "keyframe_time" not in info or not lidar_dir.exists():
       return None, None, None


   keyframe_timestamp = _iso_to_dt(info["keyframe_time"])
   best_name: str | None = None
   best_timestamp_str: str | None = None
   best_abs_dt = float("inf")


   for p in lidar_dir.glob("*.npy"):
       timestamp_str = p.stem.rsplit("_", 1)[-1]
       try:
           delta_time_s = (_iso_to_dt(timestamp_str) - keyframe_timestamp).total_seconds()
       except Exception:
           continue


       abs_dt = abs(delta_time_s)
       if abs_dt < best_abs_dt:
           best_abs_dt = abs_dt
           best_name = p.name
           best_timestamp_str = timestamp_str


   if best_name is None:
       return None, None, None


   return best_name, best_timestamp_str, float(best_abs_dt)




def make_zodframe_from_raw(frame_dir: Path) -> ZodFrame:
   """
   Build ZodFrame from raw single-frame directory layout.
   This fixes relative paths in info.json to be frame-local.
   """
   info_dict = json.loads((frame_dir / "info.json").read_text())
   fid = info_dict.get("id", frame_dir.name)
   prefix = f"single_frames/{fid}/"


   def fix_path(p: str | None) -> str | None:
       if p is None:
           return None
       if p.startswith(prefix):
           return p[len(prefix) :]
       if p.startswith("single_frames/"):
           return p.split("/", 2)[-1]
       return p


   for k in ["calibration_path", "ego_motion_path", "metadata_path", "oxts_path", "vehicle_data_path"]:
       if info_dict.get(k) is not None:
           info_dict[k] = fix_path(info_dict[k])


   for ann in info_dict.get("annotations", {}).values():
       if ann.get("filepath") is not None:
           ann["filepath"] = fix_path(ann["filepath"])


   for arr in info_dict.get("camera_frames", {}).values():
       for x in arr:
           if x.get("filepath") is not None:
               x["filepath"] = fix_path(x["filepath"])


   for arr in info_dict.get("lidar_frames", {}).values():
       for x in arr:
           if x.get("filepath") is not None:
               x["filepath"] = fix_path(x["filepath"])


   info = Information.from_dict(info_dict)
   info.convert_paths_to_absolute(str(frame_dir))
   return ZodFrame(info)




def undistort_and_resize(
   img_rgb: np.ndarray,
   k: np.ndarray,
   d: np.ndarray,
   out_w: int,
   out_h: int,
   balance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   h, w = img_rgb.shape[:2]


   new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
       k, d, (w, h), np.eye(3), balance=balance
   )


   map1, map2 = cv2.fisheye.initUndistortRectifyMap(
       k, d, np.eye(3), new_k, (w, h), cv2.CV_16SC2
   )


   undistorted_rgb = cv2.remap(
       img_rgb,
       map1,
       map2,
       interpolation=cv2.INTER_LINEAR,
       borderMode=cv2.BORDER_CONSTANT,
   )


   sx = out_w / w
   sy = out_h / h
   s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
   bev_intrinsics = s @ new_k


   resized_undistorted_img = cv2.resize(undistorted_rgb, (out_w, out_h))


   return undistorted_rgb, resized_undistorted_img, new_k, bev_intrinsics




def make_box_corners_3d(center_xyz: np.ndarray, size_lwh: np.ndarray, orientation: Any) -> np.ndarray:
   """
   Build cuboid corners from center + [length, width, height] + quaternion orientation.
   """
   center_xyz = np.asarray(center_xyz, dtype=np.float64)
   size_lwh = np.asarray(size_lwh, dtype=np.float64)


   cx, cy, cz = center_xyz
   l, w, h = size_lwh


   x_c = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dtype=np.float64)
   y_c = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dtype=np.float64)
   z_c = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dtype=np.float64)
   corners_local = np.stack([x_c, y_c, z_c], axis=1)


   r = np.asarray(orientation.rotation_matrix, dtype=np.float64)
   corners_rot = corners_local @ r.T
   return corners_rot + np.array([cx, cy, cz], dtype=np.float64)




def project_points_camera_frame_to_pixels(points_cam: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   """
   Project camera-frame points to pixels with final intrinsics.
   """
   x = points_cam[:, 0]
   y = points_cam[:, 1]
   z = points_cam[:, 2]


   valid_mask = z > EPS_Z
   u = np.full_like(z, fill_value=np.nan, dtype=np.float64)
   v = np.full_like(z, fill_value=np.nan, dtype=np.float64)
   u[valid_mask] = k[0, 0] * (x[valid_mask] / z[valid_mask]) + k[0, 2]
   v[valid_mask] = k[1, 1] * (y[valid_mask] / z[valid_mask]) + k[1, 2]


   pixels = np.stack([u, v], axis=1)
   return pixels, valid_mask




def yaw_from_rotmat_z(r: np.ndarray) -> float:
   return float(np.arctan2(r[1, 0], r[0, 0]))




def make_final_pedestrian_instance(
   annotation_uuid: str,
   center_ego: np.ndarray,
   size_lwh: np.ndarray,
   yaw_ego: float,
   center_pixel_uv: np.ndarray,
   corners_pixel_uv: np.ndarray,
) -> dict[str, Any]:
   center_ego = np.asarray(center_ego, dtype=np.float64)
   size_lwh = np.asarray(size_lwh, dtype=np.float64)
   center_pixel_uv = np.asarray(center_pixel_uv, dtype=np.float64)
   corners_pixel_uv = np.asarray(corners_pixel_uv, dtype=np.float64)


   x_center, y_center, z_center = center_ego.tolist()
   dx, dy, dz = size_lwh.tolist()
   z_bottom = z_center - dz / 2.0


   return {
       "annotation_uuid": annotation_uuid,
       "class": "Pedestrian",
       "label_3d": 0,
       "box_center_ego_geometric": [float(x_center), float(y_center), float(z_center)],
       "box_bottom_center_ego": [float(x_center), float(y_center), float(z_bottom)],
       "box_size_lwh": [float(dx), float(dy), float(dz)],
       "box_yaw_ego": float(yaw_ego),
       "box_3d": [
           float(x_center),
           float(y_center),
           float(z_bottom),
           float(dx),
           float(dy),
           float(dz),
           float(yaw_ego),
       ],
       "projected_center_uv": center_pixel_uv.tolist(),
       "projected_corners_uv": corners_pixel_uv.tolist(),
   }




def occlusion_bucket(occ: Any) -> str:
   if occ is None:
       return "missing"


   s = str(occ).strip().lower()
   if s == "none":
       return "none"
   if s == "light":
       return "light"
   if s == "medium":
       return "medium"
   if s == "heavy":
       return "heavy"
   if s == "veryheavy":
       return "veryheavy"
   return "unknown"




def load_pedestrian_props_map(frame_dir: Path) -> dict[str, dict[str, Any]]:
   """
   Map annotation_uuid -> {unclear: bool, occ_bucket: str} from object_detection.json.
   """
   out: dict[str, dict[str, Any]] = {}
   obj_path = frame_dir / "annotations" / "object_detection.json"
   raw = _read_json(obj_path)
   if raw is None:
       return out


   annotations = raw.get("annotations", []) if isinstance(raw, dict) else raw
   if not isinstance(annotations, list):
       return out


   for obj in annotations:
       if not isinstance(obj, dict):
           continue
       props = obj.get("properties", {}) or {}
       if props.get("class") != "Pedestrian":
           continue
       uuid = str(props.get("annotation_uuid", "")).strip()
       if not uuid:
           continue
       out[uuid] = {
           "unclear": bool(props.get("unclear", False)),
           "occ_bucket": occlusion_bucket(props.get("occlusion_ratio", None)),
       }
   return out




def resolve_frame_dir(row: dict[str, Any], dataset_root_fallback: Path) -> Path:
   image_path = Path(str(row["image_path"]))
   frame_dir = image_path.parent.parent
   if frame_dir.exists():
       return frame_dir


   fallback = dataset_root_fallback / str(row["frame_id"])
   if fallback.exists():
       return fallback


   raise FileNotFoundError(f"Could not resolve frame dir for frame_id={row['frame_id']}")




def ensure_output_dirs() -> None:
   for p in [OUT_ROOT, OUT_IMAGES, OUT_LIDAR, OUT_CALIBS, OUT_LABELS, OUT_INDEX]:
       p.mkdir(parents=True, exist_ok=True)




def empty_recomputed_fields() -> dict[str, Any]:
   return {
       "ped_count_clear": 0,
       "ped_count_unclear": 0,
       "ped_occ_none": 0,
       "ped_occ_light": 0,
       "ped_occ_medium": 0,
       "ped_occ_heavy": 0,
       "ped_occ_veryheavy": 0,
       "ped_occ_missing": 0,
       "ped_occ_unknown": 0,
       "ped_uuid": [],
       "ped_unclear_list": [],
       "ped_occlusion_list": [],
       "ped_present": 0,
       "num_pedestrians_final": 0,
   }




def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(
       description="Build zod_moe dataset artifacts and updated parquet for MMDetection3D/BEVFusion."
   )
   parser.add_argument(
       "--source-parquet",
       type=Path,
       default=SOURCE_PARQUET,
       help="Input parquet with source metadata.",
   )
   parser.add_argument(
       "--target-parquet",
       type=Path,
       default=TARGET_PARQUET,
       help="Output parquet path for updated dataset index.",
   )
   parser.add_argument(
       "--dataset-root-fallback",
       type=Path,
       default=DATASET_ROOT_FALLBACK,
       help="Fallback root used when frame directory cannot be resolved from image_path.",
   )
   parser.add_argument(
       "--limit",
       type=int,
       default=None,
       help="Optional cap for quick testing.",
   )
   return parser.parse_args()




def main() -> None:
   args = parse_args()
   ensure_output_dirs()
   args.target_parquet.parent.mkdir(parents=True, exist_ok=True)


   print(f"Reading source parquet: {args.source_parquet}")
   df = pd.read_parquet(args.source_parquet)
   if args.limit is not None:
       df = df.iloc[: args.limit].copy()
       print(f"Row limit enabled: {len(df)}")
   else:
       print(f"Rows to process: {len(df)}")


   updated_rows: list[dict[str, Any]] = []
   n_processed = 0
   n_failed = 0
   failed_rows: list[dict[str, str]] = []


   for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing frames", unit="frame"):
       row_dict = row.to_dict()
       frame_id = str(row_dict["frame_id"])


       # Initialize new row from source row, then apply schema updates.
       new_row = dict(row_dict)
       for col in DROP_COLUMNS:
           new_row.pop(col, None)
       for col in RECOMPUTE_PEDESTRIAN_COLUMNS:
           new_row.pop(col, None)
       new_row.update(empty_recomputed_fields())
       new_row["final_img_file_path"] = None
       new_row["final_points_file_path"] = None
       new_row["calib_file_path"] = None
       new_row["label_file_path"] = None


       try:
           frame_dir = resolve_frame_dir(row_dict, args.dataset_root_fallback)
           zod_frame = make_zodframe_from_raw(frame_dir)
           calib = zod_frame.calibration


           # ------------------------------------------------------------
           # Camera processing: undistort + resize + final intrinsics
           # ------------------------------------------------------------
           image_path = Path(str(row_dict["image_path"]))
           img_rgb = np.array(Image.open(image_path).convert("RGB"))


           cam_calib = calib.cameras[Camera.FRONT]
           k = cam_calib.intrinsics[:3, :3].astype(np.float64)
           d = cam_calib.distortion.astype(np.float64).reshape(4, 1)


           _, resized_undistorted_img, _, bev_intrinsics = undistort_and_resize(
               img_rgb=img_rgb,
               k=k,
               d=d,
               out_w=FINAL_W,
               out_h=FINAL_H,
               balance=UNDISTORT_BALANCE,
           )
           img_h, img_w = resized_undistorted_img.shape[:2]


           final_img_path = OUT_IMAGES / f"{frame_id}.png"
           Image.fromarray(resized_undistorted_img).save(final_img_path)


           # ------------------------------------------------------------
           # LiDAR processing:
           # nearest sweep -> deskew -> final-FoV filter -> ego XYZI .bin
           # ------------------------------------------------------------
           nearest_name, _, _ = nearest_lidar_filename(frame_dir)
           if nearest_name is None:
               raise ValueError("Could not select nearest LiDAR sweep.")


           raw_lidar = LidarData.from_npy(frame_dir / "lidar_velodyne" / nearest_name)
           front_camera_frame = zod_frame.info.get_key_camera_frame(
               camera=Camera.FRONT,
               anonymization=Anonymization.BLUR,
           )
           camera_timestamp = float(front_camera_frame.time.timestamp())


           ego_motion = zod_frame.ego_motion
           lidar_calib = calib.lidars[Lidar.VELODYNE]
           aligned_lidar = motion_compensate_pointwise(
               raw_lidar,
               ego_motion,
               lidar_calib,
               target_timestamp=camera_timestamp,
           )


           points_lidar = aligned_lidar.points
           aligned_lidar_ego = aligned_lidar.copy()
           aligned_lidar_ego.transform(lidar_calib.extrinsics)  # LiDAR -> ego
           points_ego = aligned_lidar_ego.points


           t_lidar_to_cam = get_3d_transform_camera_lidar(
               calib,
               lidar=Lidar.VELODYNE,
               camera=Camera.FRONT,
           ).transform
           points_cam = transform_points(points_lidar, t_lidar_to_cam)


           x_cam = points_cam[:, 0]
           y_cam = points_cam[:, 1]
           z_cam = points_cam[:, 2]


           positive_depth_mask = z_cam > EPS_Z
           u = np.full(points_cam.shape[0], np.nan, dtype=np.float64)
           v = np.full(points_cam.shape[0], np.nan, dtype=np.float64)


           u[positive_depth_mask] = (
               bev_intrinsics[0, 0] * (x_cam[positive_depth_mask] / z_cam[positive_depth_mask])
               + bev_intrinsics[0, 2]
           )
           v[positive_depth_mask] = (
               bev_intrinsics[1, 1] * (y_cam[positive_depth_mask] / z_cam[positive_depth_mask])
               + bev_intrinsics[1, 2]
           )


           inside_image_mask = (
               (u >= 0) & (u < img_w) &
               (v >= 0) & (v < img_h)
           )
           fov_mask_full = positive_depth_mask & inside_image_mask


           fov_points_ego = points_ego[fov_mask_full]
           fov_intensity = aligned_lidar.intensity[fov_mask_full]
           fov_points_ego_xyzi = np.hstack(
               [
                   fov_points_ego.astype(np.float32),
                   fov_intensity.astype(np.float32)[:, None],
               ]
           )


           final_lidar_path = OUT_LIDAR / f"{frame_id}.bin"
           fov_points_ego_xyzi.astype(np.float32).tofile(final_lidar_path)


           # ------------------------------------------------------------
           # Annotation processing:
           # pedestrian 3D -> final-FoV center filter -> ego [x,y,z,dx,dy,dz,yaw]
           # ------------------------------------------------------------
           anns = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
           pedestrian_objs = [obj for obj in anns if obj is not None and obj.name == "Pedestrian" and obj.box3d is not None]


           t_ego_from_cam = np.asarray(cam_calib.extrinsics.transform, dtype=np.float64)
           t_cam_from_ego = np.linalg.inv(t_ego_from_cam)
           t_ego_from_lidar = np.asarray(lidar_calib.extrinsics.transform, dtype=np.float64)
           t_cam_from_lidar = t_cam_from_ego @ t_ego_from_lidar
           r_ego_from_lidar = t_ego_from_lidar[:3, :3]


           final_instances: list[dict[str, Any]] = []
           num_total = 0
           num_kept = 0
           num_outside = 0


           for i, pedestrian_obj in enumerate(pedestrian_objs):
               num_total += 1
               box3d = pedestrian_obj.box3d


               center_lidar = np.asarray(box3d.center, dtype=np.float64)
               size_lwh = np.asarray(box3d.size, dtype=np.float64)
               orientation = box3d.orientation


               corners_lidar = make_box_corners_3d(
                   center_xyz=center_lidar,
                   size_lwh=size_lwh,
                   orientation=orientation,
               )
               center_lidar_2d = center_lidar.reshape(1, 3)


               corners_cam = transform_points(corners_lidar, t_cam_from_lidar)
               center_cam = transform_points(center_lidar_2d, t_cam_from_lidar)


               pixels_corners, _ = project_points_camera_frame_to_pixels(corners_cam, bev_intrinsics)
               pixels_center, valid_mask_center = project_points_camera_frame_to_pixels(center_cam, bev_intrinsics)
               center_uv = pixels_center[0]


               u_c, v_c = center_uv
               center_inside_final_image = (
                   bool(valid_mask_center[0])
                   and np.isfinite(u_c)
                   and np.isfinite(v_c)
                   and (0.0 <= u_c < img_w)
                   and (0.0 <= v_c < img_h)
               )
               if not center_inside_final_image:
                   num_outside += 1
                   continue


               num_kept += 1
               center_ego = transform_points(center_lidar_2d, t_ego_from_lidar)[0]
               r_box_lidar = np.asarray(orientation.rotation_matrix, dtype=np.float64)
               r_box_ego = r_ego_from_lidar @ r_box_lidar
               yaw_ego = yaw_from_rotmat_z(r_box_ego)


               instance = make_final_pedestrian_instance(
                   annotation_uuid=str(getattr(pedestrian_obj, "uuid", f"ped_{i}")),
                   center_ego=center_ego,
                   size_lwh=size_lwh,
                   yaw_ego=yaw_ego,
                   center_pixel_uv=center_uv,
                   corners_pixel_uv=pixels_corners,
               )
               final_instances.append(instance)


           label_payload = {
               "frame_id": frame_id,
               "task": "pedestrian_only",
               "class_names": ["Pedestrian"],
               "box_coordinate_system": "ego",
               "box_format": "[x, y, z, dx, dy, dz, yaw]",
               "z_definition": "bottom_center",
               "axis_convention": {
                   "x": "forward",
                   "y": "left",
                   "z": "up",
               },
               "handedness": "right-handed",
               "size_definition": {
                   "dx": "length_along_x",
                   "dy": "width_along_y",
                   "dz": "height_along_z",
               },
               "image_size_final": {
                   "width": int(img_w),
                   "height": int(img_h),
               },
               "camera_name": "FRONT",
               "filtering_rule": "keep pedestrian if projected 3D box center lands inside final undistorted+resized image",
               "num_total_pedestrians_with_3d": int(num_total),
               "num_kept_after_final_fov_filter": int(num_kept),
               "num_rejected_outside_final_image": int(num_outside),
               "K_final": bev_intrinsics.tolist(),
               "instances": final_instances,
           }


           final_label_path = OUT_LABELS / f"{frame_id}.json"
           final_label_path.write_text(json.dumps(label_payload, indent=2))


           # ------------------------------------------------------------
           # Calibration artifact for BEVFusion camera branch
           # ------------------------------------------------------------
           extrinsics_ego_from_camera = t_ego_from_cam
           extrinsics_camera_from_ego = np.linalg.inv(extrinsics_ego_from_camera)
           ego2image = (bev_intrinsics @ extrinsics_camera_from_ego[:3, :]).astype(np.float64)
           img_aug_matrix = np.eye(4, dtype=np.float64)
           fusion_aug_matrix = np.eye(4, dtype=np.float64)


           final_calib_path = OUT_CALIBS / f"{frame_id}.npz"
           np.savez_compressed(
               final_calib_path,
               frame_id=np.array(frame_id),
               K_final=bev_intrinsics.astype(np.float32),
               distortion=d.astype(np.float32),
               camera2ego=extrinsics_ego_from_camera.astype(np.float32),
               ego2image=ego2image.astype(np.float32),
               img_aug_matrix=img_aug_matrix.astype(np.float32),
               fusion_aug_matrix=fusion_aug_matrix.astype(np.float32),
               image_size=np.array([img_h, img_w], dtype=np.int32),
           )


           # ------------------------------------------------------------
           # Recompute pedestrian parquet metadata from kept final instances
           # ------------------------------------------------------------
           ped_props_map = load_pedestrian_props_map(frame_dir)
           kept_uuids = [str(inst["annotation_uuid"]) for inst in final_instances]


           ped_unclear_list: list[bool] = []
           ped_occlusion_list: list[str] = []
           occ_counts = {
               "none": 0,
               "light": 0,
               "medium": 0,
               "heavy": 0,
               "veryheavy": 0,
               "missing": 0,
               "unknown": 0,
           }


           for uuid in kept_uuids:
               meta = ped_props_map.get(uuid, {"unclear": False, "occ_bucket": "missing"})
               unclear = bool(meta.get("unclear", False))
               occ_bucket = str(meta.get("occ_bucket", "unknown"))
               if occ_bucket not in occ_counts:
                   occ_bucket = "unknown"
               ped_unclear_list.append(unclear)
               ped_occlusion_list.append(occ_bucket)
               occ_counts[occ_bucket] += 1


           ped_count_unclear = int(sum(1 for x in ped_unclear_list if x))
           ped_count_clear = int(len(kept_uuids) - ped_count_unclear)


           new_row.update(
               {
                   "ped_count_clear": ped_count_clear,
                   "ped_count_unclear": ped_count_unclear,
                   "ped_occ_none": int(occ_counts["none"]),
                   "ped_occ_light": int(occ_counts["light"]),
                   "ped_occ_medium": int(occ_counts["medium"]),
                   "ped_occ_heavy": int(occ_counts["heavy"]),
                   "ped_occ_veryheavy": int(occ_counts["veryheavy"]),
                   "ped_occ_missing": int(occ_counts["missing"]),
                   "ped_occ_unknown": int(occ_counts["unknown"]),
                   "ped_uuid": kept_uuids,
                   "ped_unclear_list": ped_unclear_list,
                   "ped_occlusion_list": ped_occlusion_list,
                   "ped_present": int(len(kept_uuids) > 0),
                   "final_img_file_path": str(final_img_path),
                   "final_points_file_path": str(final_lidar_path),
                   "calib_file_path": str(final_calib_path),
                   "label_file_path": str(final_label_path),
                   "num_pedestrians_final": int(len(kept_uuids)),
               }
           )


           n_processed += 1
           updated_rows.append(new_row)
       except Exception as exc:
           n_failed += 1
           failed_rows.append(
               {
                   "frame_id": frame_id,
                   "error_type": type(exc).__name__,
                   "error_message": str(exc),
               }
           )
           print(f"[FAIL] frame_id={frame_id}: {type(exc).__name__}: {exc}")


   df_new = pd.DataFrame(updated_rows)
   for col in FINAL_COLUMNS:
       if col not in df_new.columns:
           df_new[col] = None
   df_new = df_new[FINAL_COLUMNS]
   df_new.to_parquet(args.target_parquet, index=False)


   print("\n=== Build Summary ===")
   print(f"Frames processed successfully: {n_processed}")
   print(f"Frames failed: {n_failed}")
   if n_failed > 0:
       print(f"Failed rows count: {n_failed}")
       sample_failed = [x["frame_id"] for x in failed_rows[:10]]
       print(f"Sample failed frame_ids (up to 10): {sample_failed}")
       print("Final parquet includes only successful frames.")
   print(f"Source parquet kept unchanged: {args.source_parquet}")
   print(f"Target parquet written: {args.target_parquet}")
   print(f"Images dir: {OUT_IMAGES}")
   print(f"LiDAR dir: {OUT_LIDAR}")
   print(f"Calibs dir: {OUT_CALIBS}")
   print(f"Labels dir: {OUT_LABELS}")




if __name__ == "__main__":
   main()



