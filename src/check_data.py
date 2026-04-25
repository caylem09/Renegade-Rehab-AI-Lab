from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("data/raw/PAMAP2_Dataset/Protocol")

files = sorted(DATA_FOLDER.glob("subject*.dat"))

print("Number of subject files found:", len(files))

if not files:
    print("No files found. Check your dataset folder.")
    raise SystemExit

first_file = files[0]
print("Reading first file:", first_file)

column_names = [
    "timestamp",
    "activity_id",
    "heart_rate",

    "hand_temp",
    "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
    "hand_acc6_x", "hand_acc6_y", "hand_acc6_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "hand_mag_x", "hand_mag_y", "hand_mag_z",
    "hand_ori_1", "hand_ori_2", "hand_ori_3", "hand_ori_4",

    "chest_temp",
    "chest_acc16_x", "chest_acc16_y", "chest_acc16_z",
    "chest_acc6_x", "chest_acc6_y", "chest_acc6_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "chest_ori_1", "chest_ori_2", "chest_ori_3", "chest_ori_4",

    "ankle_temp",
    "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z",
    "ankle_acc6_x", "ankle_acc6_y", "ankle_acc6_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "ankle_ori_1", "ankle_ori_2", "ankle_ori_3", "ankle_ori_4",
]

df = pd.read_csv(
    first_file,
    sep=r"\s+",
    header=None,
    names=column_names,
    na_values="NaN",
    engine="python",
)

print()
print("Rows and columns:", df.shape)

print()
print("First 5 rows:")
print(df[["timestamp", "activity_id", "heart_rate"]].head())

print()
print("Activity IDs found in this file:")
print(df["activity_id"].value_counts().sort_index())
