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

from typing import Iterable, List, Optional, Sequence
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

    Note: This is a conversion between the format provided by the dataset annotations and the 
    canonical format used in the project for easy conversion to other formats used by
    various models and libraries (YOLO, COCO, etc.)
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

def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    """
    FOR DINO VARIANTS ONLY.
    Convert xyxy -> xywh (absolute pixel coordinates, top-left anchored).

    Parameters
    ----------
    box : sequence of 4 floats [x1, y1, x2, y2]
        Canonical box in absolute pixel coordinates.
        Accepts list/tuple/numpy array.

    Returns
    -------
    [x, y, w, h]
        Here x,y are the top-left corner (x1,y1), not the center.

    Note:
        This function follows the common absolute xywh convention used by
        COCO-style tooling: top-left + width/height.
        YOLO needs center-based + normalized xywh for YOLO --> use `xyxy_to_yolo()`.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]


def xyxy_to_yolo(box: Sequence[float], img_w: int = 1248, img_h: int = 704) -> List[float]:
    """
    Convert xyxy -> YOLO normalized center-based xywh format.

    Parameters
    ----------
    box : sequence of 4 floats [x1, y1, x2, y2]
        Canonical box in absolute pixel coordinates.
        Accepts list/tuple/numpy array.
    img_w, img_h : int
        Image size used for normalization.

    Returns:
        [x_center, y_center, width, height] normalized to [0,1]

    Note:
        -YOLO expects center-based xywh values normalized by image dimensions,
        which is why this is different from `xyxy_to_xywh()`.
        -YOLO uses normalized coordinates (0-1 range) to make the model architecture 
        independent of input image size, allowing it to train and inference on images 
        of varying dimensions without modifying the network. This normalization also 
        improves numerical stability during training and makes the loss function scale-
        invariant across different image resolutions.
    """
    x1, y1, x2, y2 = box

    w = x2 - x1 #box width
    h = y2 - y1 #box height
    xc = x1 + w / 2.0 #center x
    yc = y1 + h / 2.0 #center y

    #normalize the center-based xywh values by the image dimensions
    return [
        xc / img_w,
        yc / img_h,
        w / img_w,
        h / img_h,
    ]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def clamp_xyxy(box: Sequence[float], img_w: int = 1248, img_h: int = 704) -> List[float]:
    """
    Clamp bounding box to image boundaries.
    input: box = [x1, y1, x2, y2]
    output: box = [x1, y1, x2, y2]
    where:
        x1, y1 = top-left corner
        x2, y2 = bottom-right corner
    The box is clamped to the image boundaries.

    Note:
        This protects downstream training/export code from invalid coordinates
        that can appear after resizing, conversion, or annotation noise.
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


def is_valid_box(box: Sequence[float], min_size: float = 2.0) -> bool:
    """
    Check if bounding box has reasonable size.
    1-pixel (or near-zero) boxes are usually annotation noise or degenerate artifacts
    --> they degrade training stability.
    This function guarantees that all bounding boxes are at least 
    min_size x min_size pixels in size.


    Note:
        This is intentionally a lightweight quality filter before export/train.
        May tighten or relax `min_size` depending on experiment goals.
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size
