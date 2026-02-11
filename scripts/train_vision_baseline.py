# scripts/train_vision_baseline.py
"""
Early-stage vision baseline training script (ZODMoE).

Right now the goal is simple:
- confirm that the Dataset/DataLoader + label mapping + model forward/backward all work end-to-end
- get a first training curve on a small subset before scaling up

Task (v0):
- ped_present ∈ {0,1}  -> implemented as 2-class softmax (CrossEntropyLoss)
"""

import random
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.zodmoe_frames import (
    ZODMoEDataConfig,
    ZODMoEVisionDataset,
)

from torchvision import transforms


@dataclass
class TrainConfig:
    # data
    frames_parquet: str = "/home/edgelab/multimodal-MoE/outputs/index/ZODmoe_frames.parquet"
    train_ids_csv: str = "/home/edgelab/zod_moe/splits/train_ids.csv"
    frame_id_col: str = "frame_id"
    image_path_col: str = "resized_image_path"
    label_col: str = "ped_present"

    # quick debug mode: train on a small subset first
    subset_size: int = 2048  # bump to e.g. 20000 / full later

    # training
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4

    seed: int = 0


def main():
    cfg = TrainConfig()

    # reproducibility for debugging
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # -----------------------
    # Dataset + DataLoader
    # -----------------------
    data_cfg = ZODMoEDataConfig(
        frames_parquet=cfg.frames_parquet,
        split_csv=cfg.train_ids_csv,
        frame_id_col=cfg.frame_id_col,
        image_path_col=cfg.image_path_col,
        label_col=cfg.label_col,
        root=None,  # paths in parquet are absolute; set root if you ever switch to relative paths
    )

    # keep transforms minimal for now; can add aug later once pipeline is stable
    transform = transforms.Compose([
        #transforms.Resize((704, 1248)), #SWITCH TO THIS LATER
        transforms.Resize(256), #256x256 for debugging
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    ds = ZODMoEVisionDataset(
        data_cfg,
        transform=transform,
        dtype_label=torch.long,  # CrossEntropyLoss expects class indices (0/1)
    )
    print("full train dataset size:", len(ds))

    # early-stage: train on a smaller slice first so we can debug quickly
    if cfg.subset_size is not None and cfg.subset_size < len(ds):
        idx = list(range(len(ds)))
        random.shuffle(idx)
        ds_train = Subset(ds, idx[: cfg.subset_size])
        print("using subset_size:", cfg.subset_size)
    else:
        ds_train = ds
        print("using full dataset")

    loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # -----------------------
    # Model (ResNet-50)
    # -----------------------
    from torchvision.models import resnet50

    # weights=None for now; later can try ImageNet pretrained weights for a stronger baseline
    model = resnet50(weights=None)

    # ped_present is binary -> 2 logits for CrossEntropyLoss
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # -----------------------
    # Optimizer / loss
    # -----------------------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------
    # Train loop (basic)
    # -----------------------
    train_losses = []
    train_accs = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        n = 0

        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)             # [B, 2]
            loss = loss_fn(logits, yb)     # scalar

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # simple running stats (just to see if learning is happening)
            bs = xb.size(0)
            total_loss += loss.item() * bs
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += bs

            pbar.set_postfix(
                loss=total_loss / max(n, 1),
                acc=correct / max(n, 1),
            )

        epoch_loss = total_loss / n
        epoch_acc = correct / n

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f"epoch {epoch:02d} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")

    # -----------------------
    # Plot training accuracy
    # -----------------------

    plot_dir = "outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(range(1, cfg.epochs + 1), train_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Training Accuracy (ped_present)")
    plt.grid(True)

    out_path = os.path.join(plot_dir, "train_acc_ped_present.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved plot -> {out_path}")



if __name__ == "__main__":
    main()
