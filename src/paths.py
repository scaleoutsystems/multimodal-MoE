from pathlib import Path
import os


def _get_path(env_var: str, default: str) -> Path:
    """
    Read a path from an environment variable if it exists,
    otherwise fall back to a default value.
    """
    value = os.environ.get(env_var, default)
    return Path(value).expanduser().resolve()


# Dataset roots
ZOD_ROOT = _get_path("ZOD_ROOT", "~/zod")
ZOD_DINO_DATA = _get_path("ZOD_DINO_DATA", "~/zod_dino_data")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1] #multimodal-MoE
OUTPUTS_DIR = _get_path("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")) #outputs/
ZODMOE_FRAMES_WITH_BOXES_PARQUET = _get_path("ZODMOE_FRAMES_WITH_BOXES_PARQUET", PROJECT_ROOT / "outputs" / "index" / "ZODmoe_frames_with_xyxy_bboxes.parquet")
