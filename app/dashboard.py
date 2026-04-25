from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.ai_summary import generate_ai_summary

DATA_FILE = Path("data/processed/pamap2_features.csv")
MODEL_FILE = Path("models/activity_model.joblib")
FEATURES_FILE = Path("models/feature_columns.joblib")
REPORT_FILE = Path("reports/model_report.txt")

st.set_page_config(
    page_title="AthleteIQ",
    page_icon="🏃",
    layout="wide"
)

st.title("AthleteIQ")
st.subheader("Local AI Sport Performance Proof of Concept")

st.write(
    "This dashboard uses wearable movement and heart-rate data to classify physical activity "
    "and summarize human performance data."
)

if not DATA_FILE.exists():
    st.error("Feature data not found. Run: python src/make_features.py")
    st.stop()

if not MODEL_FILE.exists():
    st.error("Model not found. Run: python src/train_model.py")
    st.stop()

df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)
feature_columns = joblib.load(FEATURES_FILE)

st.sidebar.header("Controls")

subjects = sorted(df["subject_id"].astype(str).unique())
selected_subject = st.sidebar.selectbox("Choose a subject", subjects)

subject_df = df[df["subject_id"].astype(str) == selected_subject].copy()

st.header(f"Subject {selected_subject} Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Activity windows", len(subject_df))
col2.metric("Activities detected", subject_df["activity"].nunique())
col3.metric("Average heart rate", f"{subject_df['heart_rate_mean'].mean():.1f} bpm")
col4.metric("Max heart rate", f"{subject_df['heart_rate_max'].max():.1f} bpm")

st.divider()

st.header("Activity Distribution")

activity_counts = subject_df["activity"].value_counts()

fig, ax = plt.subplots()
activity_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Activity")
ax.set_ylabel("Number of windows")
ax.set_title("Activity Distribution")
st.pyplot(fig)

st.header("Heart Rate Trend")

fig2, ax2 = plt.subplots()
ax2.plot(subject_df["start_time"], subject_df["heart_rate_mean"])
ax2.set_xlabel("Time")
ax2.set_ylabel("Mean heart rate")
ax2.set_title("Heart Rate Across Session")
st.pyplot(fig2)

st.header("Model Prediction Sample")

sample_size = min(100, len(subject_df))
sample = subject_df.sample(sample_size, random_state=42).copy()

X_sample = sample[feature_columns]
sample["predicted_activity"] = model.predict(X_sample)
sample["correct"] = sample["predicted_activity"] == sample["activity"]

sample_accuracy = sample["correct"].mean()

st.metric("Sample prediction accuracy", f"{sample_accuracy:.1%}")

st.dataframe(
    sample[
        [
            "subject_id",
            "start_time",
            "end_time",
            "activity",
            "predicted_activity",
            "heart_rate_mean",
            "heart_rate_max",
            "correct",
        ]
    ],
    width="stretch"
)

st.header("Local AI Sport Science Summary")

summary_data = f"""
Subject: {selected_subject}
Activity windows analyzed: {len(subject_df)}
Activities detected: {', '.join(activity_counts.index.tolist())}
Most common activity: {activity_counts.index[0]}
Average heart rate: {subject_df['heart_rate_mean'].mean():.1f} bpm
Maximum heart rate: {subject_df['heart_rate_max'].max():.1f} bpm
Sample model prediction accuracy: {sample_accuracy:.1%}
"""

st.write("The dashboard sends this structured summary to your local Ollama model:")

st.code(summary_data)

if st.button("Generate Local AI Sport Science Summary"):
    with st.spinner("Asking your local AI model..."):
        ai_summary = generate_ai_summary(summary_data)
        st.success("Local AI summary generated.")
        st.write(ai_summary)

st.header("Model Report")

if REPORT_FILE.exists():
    st.text(REPORT_FILE.read_text())
else:
    st.warning("Model report not found.")

st.divider()

st.header("Why this matters")

st.write(
    """
    This proof of concept shows how wearable sensor data can be turned into useful
    sport science information. The system cleans raw data, creates features,
    trains a machine-learning model, evaluates predictions, presents results in
    a dashboard, and uses a local AI model to generate a plain-English summary.

    This is not a medical tool and does not diagnose injury or prescribe training.
    It is a demonstration of applied AI for sport and movement science.
    """
)
