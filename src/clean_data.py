from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("data/raw/PAMAP2_Dataset/Protocol")
OUTPUT_FOLDER = Path("data/processed")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ACTIVITY_MAP = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    24: "rope_jumping",
}

COLUMN_NAMES = [
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

KEEP_COLUMNS = [
    "subject_id",
    "timestamp",
    "activity_id",
    "activity",
    "heart_rate",

    "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
    "chest_acc16_x", "chest_acc16_y", "chest_acc16_z",
    "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z",

    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
]

all_subjects = []

files = sorted(DATA_FOLDER.glob("subject*.dat"))

print(f"Found {len(files)} subject files.")

for file in files:
    subject_id = file.stem.replace("subject", "")
    print(f"Cleaning {file.name}...")

    df = pd.read_csv(
        file,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        na_values="NaN",
        engine="python",
    )

    df["subject_id"] = subject_id

    # Keep only known activity rows.
    # The raw dataset has activity_id 0 for transitions or unlabeled data.
    df = df[df["activity_id"].isin(ACTIVITY_MAP.keys())].copy()

    # Add readable activity names.
    df["activity"] = df["activity_id"].map(ACTIVITY_MAP)

    # Keep only the columns we actually need for the project.
    df = df[KEEP_COLUMNS]

    all_subjects.append(df)

clean_df = pd.concat(all_subjects, ignore_index=True)

output_path = OUTPUT_FOLDER / "pamap2_clean.csv"
clean_df.to_csv(output_path, index=False)

print()
print("Clean data saved to:", output_path)
print("Rows:", len(clean_df))
print("Columns:", len(clean_df.columns))

print()
print("Activities in clean data:")
print(clean_df["activity"].value_counts())
