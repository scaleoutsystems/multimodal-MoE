# scripts/sanity_dataloader.py
from src.data.zodmoe_frames import ZODMoEDataConfig, ZODMoEVisionDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    cfg = ZODMoEDataConfig(
        frames_parquet="/home/edgelab/multimodal-MoE/outputs/index/ZODmoe_frames.parquet",
        split_csv="/home/edgelab/zod_moe/splits/train_ids.csv",
        frame_id_col="frame_id",
        image_path_col="image_path",   
        label_col="ped_present",       
        root=None,                     # set if image paths are relative
    )

    # Keep a tiny transform here for quick dataloader sanity checks.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    ds = ZODMoEVisionDataset(cfg, transform=transform)
    print("len(ds) =", len(ds))

    x, y = ds[0]
    print("one sample:", x.shape, y)

    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    xb, yb = next(iter(loader))
    print("one batch:", xb.shape, yb.shape)

if __name__ == "__main__":
    main()
