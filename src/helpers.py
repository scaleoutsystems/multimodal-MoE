#Various helpers for the project.
from pathlib import Path
import json
from datetime import datetime

def _read_json(path: Path):
    """Read a JSON file and return the parsed dictionary."""
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"Could not read JSON: {path}\n{e}")
        return None

def _iso_to_dt(s: str) -> datetime:
    # Z-suffixed timestamp helper.
    # iso format: 2021-01-01T00:00:00.000000Z
    # datetime format: 2021-01-01 00:00:00+00:00
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def nearest_lidar_filename(frame_dir: Path) -> str | None:
    """
    Input: Keyframe Directory Path Object
    Output: LiDAR Filename (lidar_velodyne) with timestamp closest to keyframe_time
    We read the info.json file to get the keyframe_time. 
    Instead of using the full LiDAR window ~ 11-12 .npy files, we just use the closest one. 
    """
    info = _read_json(frame_dir / "info.json")
    lidar_dir = frame_dir / "lidar_velodyne"

    if not isinstance(info, dict) or "keyframe_time" not in info or not lidar_dir.exists():
        return None

    keyframe_timestamp = _iso_to_dt(info["keyframe_time"])
    best_name = None
    # initialize best absolute delta time to infinity
    best_abs_dt = float("inf")

    for p in lidar_dir.glob("*.npy"):
        # Filename pattern: <frame_id>_<car>_<ISO8601>.npy
        timestamp_str = p.stem.rsplit("_", 1)[-1]
        try:
            delta_time_s = (_iso_to_dt(timestamp_str) - keyframe_timestamp).total_seconds()
        except Exception:
            continue

        abs_dt = abs(delta_time_s)
        if abs_dt < best_abs_dt:
            best_abs_dt = abs_dt
            best_name = p.name

    return best_name

### For working with zod sdk
def make_zodframe_from_raw(frame_dir: Path) -> ZodFrame:
    """
    Build ZodFrame from raw single-frame directory layout.
    This fixes relative paths in info.json to be frame-local.
    """
    info_dict = json.loads((frame_dir / "info.json").read_text())
    fid = info_dict.get("id", frame_dir.name)
    prefix = f"single_frames/{fid}/"

    def fix_path(p):
        if p is None:
            return None
        if p.startswith(prefix):
            return p[len(prefix):]
        if p.startswith("single_frames/"):
            return p.split("/", 2)[-1]
        return p

    # top-level paths
    for k in ["calibration_path", "ego_motion_path", "metadata_path", "oxts_path", "vehicle_data_path"]:
        if info_dict.get(k) is not None:
            info_dict[k] = fix_path(info_dict[k])

    # annotation paths
    for ann in info_dict.get("annotations", {}).values():
        if ann.get("filepath") is not None:
            ann["filepath"] = fix_path(ann["filepath"])

    # camera frame paths
    for arr in info_dict.get("camera_frames", {}).values():
        for x in arr:
            if x.get("filepath") is not None:
                x["filepath"] = fix_path(x["filepath"])

    # lidar frame paths
    for arr in info_dict.get("lidar_frames", {}).values():
        for x in arr:
            if x.get("filepath") is not None:
                x["filepath"] = fix_path(x["filepath"])

    info = Information.from_dict(info_dict)
    info.convert_paths_to_absolute(str(frame_dir))
    return ZodFrame(info)