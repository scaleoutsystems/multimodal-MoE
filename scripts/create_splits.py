import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 0

parquet_path = Path("/home/edgelab/multimodal-MoE/outputs/index/ZODmoe_frames.parquet").expanduser()
df = pd.read_parquet(parquet_path)

# ensure no NaNs in strat columns
df["time_of_day"] = df["time_of_day"].fillna("unknown").astype(str)
df["ped_bin_4"] = df["ped_bin_4"].astype(int)

# composite stratification key
df["strat_key"] = df["ped_bin_4"].astype(str) + "_" + df["time_of_day"]

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["strat_key"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["strat_key"],
)

# save ONLY frame_id
output_dir = Path("/home/edgelab/zod_moe/splits")
output_dir.mkdir(parents=True, exist_ok=True)

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split_df[["frame_id"]].to_csv(output_dir / f"{split_name}_ids.csv", index=False)

print(f"Created splits -- train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
