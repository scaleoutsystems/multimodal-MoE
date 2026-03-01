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



#######FOR VISUALIZING LIDAR DATA #########################################################
def _xyz_intensity_from_pts(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract xyz (N,3) and intensity (N,) from structured or plain arrays."""
    if not isinstance(pts, np.ndarray):
        raise TypeError("pts must be a numpy ndarray")

    # Structured array path (expected for ZOD LiDAR .npy)
    if pts.dtype.names is not None:
        names = set(pts.dtype.names)
        required = {"x", "y", "z", "intensity"}
        if not required.issubset(names):
            raise ValueError(f"Structured pts is missing required fields: {required - names}")

        xyz = np.stack(
            [
                pts["x"].astype(np.float32),
                pts["y"].astype(np.float32),
                pts["z"].astype(np.float32),
            ],
            axis=1,
        )
        intensity = pts["intensity"].astype(np.float32)
        return xyz, intensity

    # Plain array fallback: [timestamp, x, y, z, intensity, ...]
    arr = np.asarray(pts)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError("Plain pts array must have shape (N, >=5): [timestamp, x, y, z, intensity, ...]")

    xyz = arr[:, 1:4].astype(np.float32)
    intensity = arr[:, 4].astype(np.float32)
    return xyz, intensity


def plot_lidar_bev(
    pts: np.ndarray,
    max_points: int = 200_000,
    seed: int = 0,
    point_size: float = 0.3,
    alpha: float = 0.75,
    cmap: str = "viridis",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    input: .npy LidarVelodyne file
    BEV (x-y) scatter colored by intensity (0..255).
    input: structured or plain numpy array
    output: matplotlib figure
    """
    xyz, intensity = _xyz_intensity_from_pts(pts)

    # No downsampling: use all points.
    xyz_s = xyz
    inten_s = np.clip(intensity, 0.0, 255.0) / 255.0

    plt.figure(figsize=(9, 9))
    sc = plt.scatter(
        xyz_s[:, 0], xyz_s[:, 1],
        c=inten_s,
        s=point_size,
        alpha=alpha,
        cmap=cmap,
        linewidths=0,
    )

    plt.axhline(0.0, linewidth=0.8, color="black", alpha=0.6)
    plt.axvline(0.0, linewidth=0.8, color="black", alpha=0.6)
    plt.gca().set_aspect("equal", adjustable="box")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.title(f"LiDAR BEV (n={len(xyz_s):,})")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar(sc)
    cbar.set_label("intensity (normalized 0..1)")
    plt.show()


# Example calls:
plot_lidar_bev(pts, xlim=(-60, 60), ylim=(-60, 60))
##END OF VISUALIZING LIDAR DATA #########################################################