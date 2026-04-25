from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

INPUT_FILE = Path("data/processed/pamap2_features.csv")
MODEL_FOLDER = Path("models")
REPORT_FOLDER = Path("reports")

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading feature data...")
    df = pd.read_csv(INPUT_FILE)

    print("Rows loaded:", len(df))
    print("Columns loaded:", len(df.columns))

    # These columns describe the row, but they are not the sensor features.
    columns_to_remove = [
        "subject_id",
        "start_time",
        "end_time",
        "activity",
    ]

    feature_columns = [col for col in df.columns if col not in columns_to_remove]

    X = df[feature_columns]
    y = df["activity"]

    print()
    print("Activities the model will learn:")
    print(y.value_counts())

    print()
    print("Splitting data into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print()
    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Model training complete.")

    print()
    print("Testing model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print()
    print("Model accuracy:")
    print(f"{accuracy:.2%}")

    print()
    print("Classification report:")
    print(report)

    print("Saving model and reports...")

    joblib.dump(model, MODEL_FOLDER / "activity_model.joblib")
    joblib.dump(feature_columns, MODEL_FOLDER / "feature_columns.joblib")

    with open(REPORT_FOLDER / "model_report.txt", "w") as file:
        file.write("AthleteIQ Model Report\n")
        file.write("======================\n\n")
        file.write(f"Accuracy: {accuracy:.2%}\n\n")
        file.write(report)

    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    confusion_df.to_csv(REPORT_FOLDER / "confusion_matrix.csv")

    feature_importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv(REPORT_FOLDER / "feature_importance.csv", index=False)

    print()
    print("Saved:")
    print("models/activity_model.joblib")
    print("models/feature_columns.joblib")
    print("reports/model_report.txt")
    print("reports/confusion_matrix.csv")
    print("reports/feature_importance.csv")

if __name__ == "__main__":
    main()
