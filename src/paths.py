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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = _get_path("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))
