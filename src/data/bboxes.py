"""
Bounding box utilities for ZODMoE.

This module defines a canonical internal representation for pedestrian
bounding boxes: absolute pixel coordinates in xyxy format.

Canonical format:
    [x1, y1, x2, y2]
where:
    (x1, y1) = top-left corner
    (x2, y2) = bottom-right corner

All other formats (YOLO, COCO, etc.) are derived from this.
"""

from typing import Iterable, List, Optional
import numpy as np


# ----------------------------------------------------------------------
# Core conversion
# ----------------------------------------------------------------------

def points_to_xyxy(points: Iterable) -> Optional[List[float]]:
    """
    Convert an iterable of (x, y) points to canonical xyxy format.

    Parameters
    ----------
    points : iterable of (x, y)
        Multipoint pedestrian annotation in pixel coordinates.

    Returns
    -------
    box : [x1, y1, x2, y2] (floats) or None if invalid

    "Optional" type so that if the points are not a valid bounding box, we return None
    """
    #first convert the points to a numpy array of shape (4, 2)
    pts = np.stack(points).astype(np.float32)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Expected iterable of (x, y) points.")

    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))

    if x2 <= x1 or y2 <= y1:
        return None  # degenerate box

    return [x1, y1, x2, y2]


# ----------------------------------------------------------------------
# Format conversions
# ----------------------------------------------------------------------

def xyxy_to_xywh(box: List[float]) -> List[float]:
    """
    Convert xyxy -> xywh (absolute pixel coordinates).
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]


def xyxy_to_yolo(box: List[float], img_w: int = 1248, img_h: int = 704) -> List[float]:
    """
    Convert xyxy -> YOLO normalized xywh format.

    Returns:
        [x_center, y_center, width, height] normalized to [0,1]
    """
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0

    return [
        xc / img_w,
        yc / img_h,
        w / img_w,
        h / img_h,
    ]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def clamp_xyxy(box: List[float], img_w: int = 1248, img_h: int = 704) -> List[float]:
    """
    Clamp bounding box to image boundaries.
    input: box = [x1, y1, x2, y2]
    output: box = [x1, y1, x2, y2]
    where:
        x1, y1 = top-left corner
        x2, y2 = bottom-right corner
    The box is clamped to the image boundaries.
    """
    x1, y1, x2, y2 = box

    # pixel indices are zero-based, so we need to subtract 1 to get the maximum valid pixel index
    #valid x values are 0 to 1247. Max --> at least 0 and min --> at most 1247. 
    x1 = max(0.0, min(x1, img_w - 1)) 
    x2 = max(0.0, min(x2, img_w - 1)) 
    # valid y values are 0 to 703. Max --> at least 0 and min --> at most 703. 
    y1 = max(0.0, min(y1, img_h - 1)) 
    y2 = max(0.0, min(y2, img_h - 1))

    return [x1, y1, x2, y2]


def is_valid_box(box: List[float], min_size: float = 2.0) -> bool:
    """
    Check if bounding box has reasonable size.
    1-pixel (or near-zero) boxes are usually annotation noise or degenerate artifacts
    --> they degrade training stability.
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size
