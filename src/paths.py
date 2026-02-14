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
# Source-of-truth data root for this project.
ZOD_MOE_DATA = _get_path("ZOD_MOE_DATA", "~/zod_moe")
RESIZED_IMAGES_DIR = _get_path("RESIZED_IMAGES_DIR", ZOD_MOE_DATA / "resized_images")
SPLITS_DIR = _get_path("SPLITS_DIR", ZOD_MOE_DATA / "splits")
TRAIN_SPLIT_CSV = _get_path("TRAIN_SPLIT_CSV", SPLITS_DIR / "train_ids.csv")
VAL_SPLIT_CSV = _get_path("VAL_SPLIT_CSV", SPLITS_DIR / "val_ids.csv")
TEST_SPLIT_CSV = _get_path("TEST_SPLIT_CSV", SPLITS_DIR / "test_ids.csv")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = _get_path("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))
INDEX_DIR = _get_path("INDEX_DIR", OUTPUTS_DIR / "index")
EXPORTS_DIR = _get_path("EXPORTS_DIR", OUTPUTS_DIR / "exports")
RUNS_DIR = _get_path("RUNS_DIR", OUTPUTS_DIR / "runs")
EVAL_DIR = _get_path("EVAL_DIR", OUTPUTS_DIR / "eval")

ZODMOE_FRAMES_PARQUET = _get_path(
    "ZODMOE_FRAMES_PARQUET",
    INDEX_DIR / "ZODmoe_frames.parquet",
)
ZODMOE_FRAMES_WITH_BOXES_PARQUET = _get_path(
    "ZODMOE_FRAMES_WITH_BOXES_PARQUET",
    INDEX_DIR / "ZODmoe_frames_with_xyxy_bboxes.parquet",
)

# Backward-compatible aliases used by existing scripts.
RESIZED_IMAGE_PATH = RESIZED_IMAGES_DIR
TRAIN_SPLIT_PATH = TRAIN_SPLIT_CSV
VAL_SPLIT_PATH = VAL_SPLIT_CSV
TEST_SPLIT_PATH = TEST_SPLIT_CSV
