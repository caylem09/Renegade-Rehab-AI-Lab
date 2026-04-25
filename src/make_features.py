from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/pamap2_clean.csv")
OUTPUT_FILE = Path("data/processed/pamap2_features.csv")

WINDOW_SIZE = 200

META_COLUMNS = [
    "subject_id",
    "timestamp",
    "activity_id",
    "activity",
]

def main():
    print("Loading clean data...")
    df = pd.read_csv(INPUT_FILE)

    print("Rows loaded:", len(df))

    feature_columns = [col for col in df.columns if col not in META_COLUMNS]

    print("Filling missing values...")
    df[feature_columns] = (
        df.groupby("subject_id")[feature_columns]
        .transform(lambda x: x.interpolate(limit_direction="both"))
    )

    df = df.dropna(subset=feature_columns)

    print("Making activity windows...")

    rows = []

    for subject_id, subject_df in df.groupby("subject_id"):
        subject_df = subject_df.sort_values("timestamp").reset_index(drop=True)

        for start in range(0, len(subject_df) - WINDOW_SIZE, WINDOW_SIZE):
            window = subject_df.iloc[start:start + WINDOW_SIZE]

            activity = window["activity"].mode().iloc[0]

            row = {
                "subject_id": subject_id,
                "start_time": window["timestamp"].iloc[0],
                "end_time": window["timestamp"].iloc[-1],
                "activity": activity,
                "heart_rate_mean": window["heart_rate"].mean(),
                "heart_rate_max": window["heart_rate"].max(),
            }

            for col in feature_columns:
                row[f"{col}_mean"] = window[col].mean()
                row[f"{col}_std"] = window[col].std()
                row[f"{col}_min"] = window[col].min()
                row[f"{col}_max"] = window[col].max()

            rows.append(row)

    features = pd.DataFrame(rows)

    features.to_csv(OUTPUT_FILE, index=False)

    print()
    print("Feature file saved to:", OUTPUT_FILE)
    print("Number of activity windows:", len(features))
    print("Number of columns:", len(features.columns))

    print()
    print("Activities in feature file:")
    print(features["activity"].value_counts())

    print()
    print("First few rows:")
    print(features.head())


if __name__ == "__main__":
    main()
