# src/data/zodmoe_frames.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ZODMoEDataConfig:
    """
    Minimal config for a vision-only ZODMoE Dataset.

    You MUST set:
      - frames_parquet: path to the parquet index (one row per frame)
      - split_csv: path to train.csv / val.csv / test.csv with a frame_id column
      - frame_id_col: name of frame id column (must exist in both parquet and csv)
      - image_path_col: parquet column containing absolute or relative image path
      - label_col: parquet column to use as label (e.g., ped_present, ped_bin_4, etc.)

    Optional:
      - root: if image_path_col is relative, it will be joined with this root
    """
    frames_parquet: PathLike
    split_csv: PathLike
    frame_id_col: str = "frame_id"
    image_path_col: str = "image_path"
    label_col: str = "label"
    root: Optional[PathLike] = None


class ZODMoEVisionDataset(Dataset):
    """
    Vision-only Dataset returning (image_tensor, label_tensor).

    - Loads split CSV (list of frame_ids)
    - Filters parquet index to those frame_ids
    - Loads image with PIL
    - Applies optional transform (torchvision-style callable)
    """
    def __init__(
        self,
        cfg: ZODMoEDataConfig,
        transform=None,
        dtype_label: torch.dtype = torch.long,
        drop_missing: bool = True,
    ):
        self.cfg = cfg
        self.transform = transform
        self.dtype_label = dtype_label

        frames_parquet = Path(cfg.frames_parquet)
        split_csv = Path(cfg.split_csv)

        if not frames_parquet.exists():
            raise FileNotFoundError(f"frames_parquet not found: {frames_parquet}")
        if not split_csv.exists():
            raise FileNotFoundError(f"split_csv not found: {split_csv}")

        # Load split ids
        split_df = pd.read_csv(split_csv)
        if cfg.frame_id_col not in split_df.columns:
            raise ValueError(
                f"split_csv missing '{cfg.frame_id_col}'. Columns: {split_df.columns.tolist()}"
            )
        split_ids = (
        split_df[cfg.frame_id_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
        .tolist()
        )


        # Load parquet (only needed columns)
        frames_df = pd.read_parquet(frames_parquet, columns=[
            cfg.frame_id_col,
            cfg.image_path_col,
            cfg.label_col,
        ])

        # Normalize types
        frames_df[cfg.frame_id_col] = frames_df[cfg.frame_id_col].astype(str)

        # Filter to split ids
        split_set = set(split_ids)
        frames_df = frames_df[frames_df[cfg.frame_id_col].isin(split_set)].copy()

        # Optional: drop missing paths/labels
        if drop_missing:
            frames_df = frames_df.dropna(subset=[cfg.image_path_col, cfg.label_col])

        # Deterministic ordering by split order (important for debugging)
        order = {fid: i for i, fid in enumerate(split_ids)}
        frames_df["_split_order"] = frames_df[cfg.frame_id_col].map(order)
        frames_df = frames_df.sort_values("_split_order").drop(columns=["_split_order"])

        self.df = frames_df.reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "Dataset is empty after filtering. "
                "Check that frame_id column matches between split CSV and parquet."
            )

        self.root = Path(cfg.root) if cfg.root is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        if self.root is None:
            # allow relative paths if user runs from a known working directory
            return path
        return self.root / path

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._resolve_path(str(row[self.cfg.image_path_col]))
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Transform → tensor
        if self.transform is None:
            # Minimal default: convert to float tensor [0,1], CHW
            img_t = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                 .view(img.size[1], img.size[0], 3)
                 .numpy())
            ).permute(2, 0, 1).float() / 255.0
        else:
            img_t = self.transform(img)

        # Label → tensor
        y = row[self.cfg.label_col]
        # If label is float-like but intended as class index, you can cast via dtype_label
        y_t = torch.tensor(y, dtype=self.dtype_label)

        return img_t, y_t


def make_basic_transform(image_size: int = 224):
    """
    Convenience helper (only if torchvision is installed).
    Keeps this module usable even without torchvision.
    """
    try:
        from torchvision import transforms
    except Exception as e:
        raise ImportError("torchvision is required for make_basic_transform()") from e

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
