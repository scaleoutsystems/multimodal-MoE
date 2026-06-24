"""Debug script: verify heatmap target placement vs feature map semantics.

Loads one training sample from the overfit config, projects GT centers
to feature map coordinates, and checks whether the current code places
them at the correct spatial position.

Usage:
    cd /home/edgelab/mmdetection3d
    PYTHONPATH=. python tools/debug_heatmap_coords.py
"""

import pickle
import sys
import torch
import numpy as np

# ── config values (from zod_lidar_only.py) ──────────────────────────────
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]
grid_size = [1440, 1440, 40]
out_size_factor = 8

pc_range = torch.tensor(point_cloud_range)
vs = torch.tensor(voxel_size)
feature_map_size = torch.tensor(grid_size[:2]) // out_size_factor  # [180, 180]

print("=" * 70)
print("CONFIG")
print("=" * 70)
print(f"  point_cloud_range = {point_cloud_range}")
print(f"  pc_range[0] (x_min) = {pc_range[0].item()}")
print(f"  pc_range[1] (y_min) = {pc_range[1].item()}")
print(f"  voxel_size      = {voxel_size}")
print(f"  grid_size        = {grid_size}")
print(f"  out_size_factor  = {out_size_factor}")
print(f"  feature_map_size = {feature_map_size.tolist()}  (grid_size[:2] // osf)")
print(f"  heatmap shape    = (C, feature_map_size[1], feature_map_size[0])")
print(f"                   = (C, {feature_map_size[1].item()}, {feature_map_size[0].item()})")
print()

# ── load one sample's GT boxes ──────────────────────────────────────────
pkl_path = "/mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes/infos/zod_nuscenes_infos_train_overfit20.pkl"
with open(pkl_path, "rb") as f:
    infos = pickle.load(f)

sample = infos["data_list"][0]
gt_boxes_raw = np.array(sample["instances"][0]["bbox_3d"])
print(f"Sample: {sample['lidar_points']['lidar_path']}")
print(f"  num GT instances = {len(sample['instances'])}")
print()

# ── trace heatmap target projection for each GT box ─────────────────────
print("=" * 70)
print("PHASE 3 — PROJECTED GT CENTERS (current code with [[1,0]] swap)")
print("=" * 70)
print()
print(f"{'GT#':>3s}  {'x_m':>8s} {'y_m':>8s} | {'coor_x':>8s} {'coor_y':>8s} | "
      f"{'SWAP center':>18s} | {'draw x(col)':>12s} {'draw y(row)':>12s} | "
      f"{'in_bounds':>9s}")
print("-" * 120)

for idx, inst in enumerate(sample["instances"]):
    box = np.array(inst["bbox_3d"])  # [x, y, z, x_size, y_size, z_size, yaw]
    x, y = box[0], box[1]

    coor_x = (x - pc_range[0].item()) / vs[0].item() / out_size_factor
    coor_y = (y - pc_range[1].item()) / vs[1].item() / out_size_factor

    # Current code: center_int[[1, 0]]  →  passes (iy, ix) to draw_heatmap_gaussian
    center = torch.tensor([coor_x, coor_y])
    center_int = center.to(torch.int32)
    swapped = center_int[[1, 0]]  # [iy, ix]

    # draw_heatmap_gaussian interprets center[0]=x=col, center[1]=y=row
    draw_x = int(swapped[0])  # = iy  → treated as column index
    draw_y = int(swapped[1])  # = ix  → treated as row index

    # heatmap shape (after class indexing): (feature_map_size[1], feature_map_size[0]) = (y_len, x_len)
    h_height = feature_map_size[1].item()  # y_len
    h_width = feature_map_size[0].item()   # x_len
    in_bounds = (0 <= draw_x < h_width) and (0 <= draw_y < h_height)

    print(f"{idx:3d}  {x:8.2f} {y:8.2f} | {coor_x:8.2f} {coor_y:8.2f} | "
          f"[{int(swapped[0]):4d}, {int(swapped[1]):4d}]         | "
          f"{draw_x:12d} {draw_y:12d} | "
          f"{'YES' if in_bounds else 'NO':>9s}")

print()

# ── check what world position the feature at (row, col) represents ──────
print("=" * 70)
print("PHASE 4 — FEATURE MAP POSITION vs GT POSITION")
print("=" * 70)
print()
print("For each GT box, we check: what world (x, y) does the feature map")
print("position where the heatmap target is placed actually represent?")
print()
print(f"{'GT#':>3s}  {'GT x':>8s} {'GT y':>8s} | {'heatmap (row,col)':>20s} | "
      f"{'feat x':>8s} {'feat y':>8s} | {'dx':>8s} {'dy':>8s} | MISMATCH?")
print("-" * 110)

for idx, inst in enumerate(sample["instances"]):
    box = np.array(inst["bbox_3d"])
    x, y = box[0], box[1]

    coor_x = (x - pc_range[0].item()) / vs[0].item() / out_size_factor
    coor_y = (y - pc_range[1].item()) / vs[1].item() / out_size_factor
    ix, iy = int(coor_x), int(coor_y)

    # With swap: heatmap target at (row=ix, col=iy)
    row, col = ix, iy

    # Feature map: (B, C, H=y_len, W=x_len)
    # Position (h, w) = (row, col) represents:
    #   world_x = col * voxel_size[0] * osf + pc_range[0]
    #   world_y = row * voxel_size[1] * osf + pc_range[1]
    feat_x = col * vs[0].item() * out_size_factor + pc_range[0].item()
    feat_y = row * vs[1].item() * out_size_factor + pc_range[1].item()

    dx = feat_x - x
    dy = feat_y - y
    mismatch = abs(dx) > 1.0 or abs(dy) > 1.0

    print(f"{idx:3d}  {x:8.2f} {y:8.2f} | ({row:4d}, {col:4d})             | "
          f"{feat_x:8.2f} {feat_y:8.2f} | {dx:+8.2f} {dy:+8.2f} | "
          f"{'*** YES ***' if mismatch else 'no'}")

print()

# ── show what the CORRECT placement (no swap) would look like ───────────
print("=" * 70)
print("CORRECT PLACEMENT (without [[1,0]] swap)")
print("=" * 70)
print()
print(f"{'GT#':>3s}  {'GT x':>8s} {'GT y':>8s} | {'heatmap (row,col)':>20s} | "
      f"{'feat x':>8s} {'feat y':>8s} | {'dx':>8s} {'dy':>8s} | MISMATCH?")
print("-" * 110)

for idx, inst in enumerate(sample["instances"]):
    box = np.array(inst["bbox_3d"])
    x, y = box[0], box[1]

    coor_x = (x - pc_range[0].item()) / vs[0].item() / out_size_factor
    coor_y = (y - pc_range[1].item()) / vs[1].item() / out_size_factor
    ix, iy = int(coor_x), int(coor_y)

    # Without swap: heatmap target at (row=iy, col=ix)
    row, col = iy, ix

    feat_x = col * vs[0].item() * out_size_factor + pc_range[0].item()
    feat_y = row * vs[1].item() * out_size_factor + pc_range[1].item()

    dx = feat_x - x
    dy = feat_y - y
    mismatch = abs(dx) > 1.0 or abs(dy) > 1.0

    print(f"{idx:3d}  {x:8.2f} {y:8.2f} | ({row:4d}, {col:4d})             | "
          f"{feat_x:8.2f} {feat_y:8.2f} | {dx:+8.2f} {dy:+8.2f} | "
          f"{'*** YES ***' if mismatch else 'no'}")

print()

# ── bev_pos audit ───────────────────────────────────────────────────────
print("=" * 70)
print("BEV_POS AUDIT — does positional encoding match feature layout?")
print("=" * 70)
print()
x_size = grid_size[0] // out_size_factor  # 180
y_size = grid_size[1] // out_size_factor  # 180

meshgrid_inputs = [
    torch.linspace(0, x_size - 1, x_size),
    torch.linspace(0, y_size - 1, y_size),
]
batch_x, batch_y = torch.meshgrid(*meshgrid_inputs)
batch_x = batch_x + 0.5
batch_y = batch_y + 0.5
coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
bev_pos = coord_base.view(1, 2, -1).permute(0, 2, 1)

print(f"create_2D_grid(x_size={x_size}, y_size={y_size})")
print(f"bev_pos shape = {bev_pos.shape}")
print()

for m in [0, 1, 179, 180, 181, 32399]:
    h = m // x_size
    w = m % x_size
    feat_world_x = w * vs[0].item() * out_size_factor + pc_range[0].item()
    feat_world_y = h * vs[1].item() * out_size_factor + pc_range[1].item()
    bev_ch0 = bev_pos[0, m, 0].item()
    bev_ch1 = bev_pos[0, m, 1].item()
    print(f"  flat_idx={m:5d}  feat_pos=(h={h:3d}, w={w:3d})  "
          f"feat_world=({feat_world_x:6.1f}, {feat_world_y:6.1f})  "
          f"bev_pos=({bev_ch0:5.1f}, {bev_ch1:5.1f})  "
          f"bev_ch0_is={'Y_idx' if abs(bev_ch0 - h - 0.5) < 0.01 else 'X_idx'}  "
          f"bev_ch1_is={'Y_idx' if abs(bev_ch1 - h - 0.5) < 0.01 else 'X_idx'}")

print()
print("CONCLUSION: bev_pos channel 0 stores the FIRST meshgrid dim index.")
print("The first meshgrid dim iterates over x_size but maps to the H (row)")
print("dimension of the feature map. So bev_pos[m, 0] = row_index = Y_idx")
print("when it SHOULD be X_idx for consistency with encode target[0] = ix.")
