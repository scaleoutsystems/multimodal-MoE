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
#ZOD_ROOT = _get_path("ZOD_ROOT", "~/zod")
#ZOD_DINO_DATA = _get_path("ZOD_DINO_DATA", "~/zod_dino_data")
# Source-of-truth data root for this project.
ZOD_MOE_DATA = _get_path("ZOD_MOE_DATA", "~/zod_moe") # /home/edgelab/zod_moe is the root of the zod_moe dataset.
RESIZED_IMAGES_DIR = _get_path("RESIZED_IMAGES_DIR", ZOD_MOE_DATA / "resized_images")
SPLITS_DIR = _get_path("SPLITS_DIR", ZOD_MOE_DATA / "splits")
TRAIN_SPLIT_CSV = _get_path("TRAIN_SPLIT_CSV", SPLITS_DIR / "train_ids.csv")
VAL_SPLIT_CSV = _get_path("VAL_SPLIT_CSV", SPLITS_DIR / "val_ids.csv")
TEST_SPLIT_CSV = _get_path("TEST_SPLIT_CSV", SPLITS_DIR / "test_ids.csv")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1] # /home/edgelab/multimodal-moe is the root of the project.
OUTPUTS_DIR = _get_path("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")) # /home/edgelab/multimodal-moe/outputs
INDEX_DIR = _get_path("INDEX_DIR", OUTPUTS_DIR / "index") # /home/edgelab/multimodal-moe/outputs/index (this is where we store parquet)
EXPORTS_DIR = _get_path("EXPORTS_DIR", OUTPUTS_DIR / "exports") # /home/edgelab/multimodal-moe/outputs/exports (this is where we store exports)
RUNS_DIR = _get_path("RUNS_DIR", OUTPUTS_DIR / "runs") # /home/edgelab/multimodal-moe/outputs/runs (this is where we store runs)
EVAL_DIR = _get_path("EVAL_DIR", OUTPUTS_DIR / "eval") # /home/edgelab/multimodal-moe/outputs/eval (this is where we store eval)

ZODMOE_FRAMES_WITH_BOXES_PARQUET = _get_path( # /home/edgelab/multimodal-moe/outputs/index/ZODmoe_frames_with_xyxy_bboxes.parquet
    "ZODMOE_FRAMES_WITH_BOXES_PARQUET",
    INDEX_DIR / "ZODmoe_frames_with_xyxy_bboxes.parquet",
)

