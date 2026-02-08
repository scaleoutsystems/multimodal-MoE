from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

#Whenever we use PathLike, the input can be either a string or a Path object
PathLike = Union[str, Path]

#frozen so that the paths are immutable
@dataclass(frozen=True)
class ZOD256FramesPaths:
    """Convenience container for default artifact locations. We centralize the paths here so that we can easily change them if needed.
    Alternative would be to hardcode the paths in the scripts that use them, which is more error prone and less flexible.
    Creating Path objects from strings that represent the default paths of the index and splits."""
    index_parquet: Path = Path("~/multimodal-MoE/outputs/index/zod256_frames.parquet")
    splits_dir: Path = Path("~/multimodal-MoE/outputs/splits/ZOD256_frames")

#inherits from torch.utils.data.Dataset
#This is a custom dataset class that we can use to load the ZOD256 frames dataset
class ZOD256FramesDataset(Dataset):
    """
    Frame-level dataset for ZOD256 single-frame pedestrian bin classification.

    Loads:
      - frame index: parquet (contains image_path, ped_bin_4, metadata, etc.)
      - split file: CSV with column `frame_id`

    Returns:
      (image, label) where:
        - image: PIL.Image (or transformed output if transform is provided)
        - label: int (ped_bin_4)
    """

    def __init__(
        self,
        index_path: PathLike,
        split_csv: PathLike,
        transform: Optional[Callable] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.split_csv = Path(split_csv)
        self.transform = transform

        df_index = pd.read_parquet(self.index_path)
        df_split = pd.read_csv(self.split_csv)

        # Minimal schema validation
        required_index_cols = {"frame_id", "image_path", "ped_bin_4"}
        required_split_cols = {"frame_id"}

        missing_index = required_index_cols - set(df_index.columns)
        missing_split = required_split_cols - set(df_split.columns)

        if missing_index:
            raise ValueError(f"Index parquet missing columns: {sorted(missing_index)}")
        if missing_split:
            raise ValueError(f"Split CSV missing columns: {sorted(missing_split)}")

        # Join to materialize the split view
        df = df_split.merge(df_index, on="frame_id", how="inner")

        if len(df) == 0:
            raise ValueError(
                "After merging split with index, got 0 rows. "
                "Check that frame_id formats match between split and index."
            )

        # Keep only what we need for now; easy to extend later
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        label = int(row["ped_bin_4"])

        # Load image lazily from disk
        img = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label
