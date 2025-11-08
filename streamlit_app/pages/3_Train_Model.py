import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.train_model import train_highlight_model

st.set_page_config(page_title="Train Highlight Model", layout="wide")
st.title("Train Boxing Highlight Model (currently scikit-learn)")


st.header("Inputs")
visual_csv_path = st.text_input("Visual features CSV Path", "", width=500)
audio_csv_path = st.text_input("Audio features CSV Path", "", width=500)
labels_csv_path = st.text_input("Labels CSV Path", "", width=500)
rolling_seconds = st.number_input("Rolling window (s) for temporal features", min_value=0, max_value=30, value=3, step=1, width = 500)
test_size = st.slider("Test size", 0.05, 0.4, 0.20, 0.05)
random_state = st.number_input("Random seed", min_value=0, value=42, step=1, width=500)

train_btn = st.button("Train Model")

def _must_exist(path: str, label: str):
    if not path or not os.path.isfile(path):
        st.error(f"{label} not found at: {path}")
        st.stop()

if train_btn:
    _must_exist(visual_csv_path, "Visual features CSV")
    _must_exist(audio_csv_path, "Audio features CSV")
    _must_exist(labels_csv_path, "Labels CSV")

    with st.spinner("Training the model…"):
        pipeline, results = train_highlight_model(
            visual_features_csv=visual_csv_path,
            audio_features_csv=audio_csv_path,
            labels_csv=labels_csv_path,
            rolling_seconds=rolling_seconds,
            test_size=test_size,
            random_state=random_state
        )

    st.success("Training complete ✅")

    # metrics overview
    colA, colB, colC, colD = st.columns(4)
    colA.metric("ROC-AUC", f"{results['roc_auc']:.3f}")
    colB.metric("PR-AUC (Avg Precision)", f"{results['pr_auc']:.3f}")
    colC.metric("Best F1 threshold", f"{results['threshold']:.2f}")
    colD.metric("F1 @ threshold", f"{results['f1']:.3f}")

    st.subheader("Classification Report @ chosen threshold")
    st.code(results["classification_report"], language="text")

    test_df: pd.DataFrame = results["test_predictions"].copy()
    # if timestamps were unavailable during split, build an index-based timestamp
    if "timestamp" not in test_df.columns or test_df["timestamp"].isna().all():
        test_df["timestamp"] = np.arange(len(test_df), dtype=float)

    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    st.subheader("Explore predicted highlight probabilities")
    threshold = st.slider(
        "Probability threshold", min_value=0.05, max_value=0.95,
        value=float(results["threshold"]), step=0.05
    )

    test_df["y_pred_thr"] = (test_df["y_proba"] >= threshold).astype(int)

    # confusion matrix at current threshold
    cm = confusion_matrix(test_df["y_true"], test_df["y_pred_thr"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p, r, f1, _ = precision_recall_fscore_support(test_df["y_true"], test_df["y_pred_thr"],
                                                  average="binary", zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{p:.3f}")
    c2.metric("Recall", f"{r:.3f}")
    c3.metric("F1", f"{f1:.3f}")
    c4.metric("TP/FP/FN/TN", f"{tp}/{fp}/{fn}/{tn}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_df["timestamp"], y=test_df["y_proba"],
        mode="lines", name="Predicted Probability"
    ))
    fig.add_hline(y=threshold, line_color="red", line_dash="dash", annotation_text="Threshold", annotation_position="top right")

    # shade predicted highlight regions
    def contiguous_segments(mask_series: pd.Series, times: pd.Series):
        segs = []
        in_seg = False
        start = None
        for t, m in zip(times, mask_series):
            if m and not in_seg:
                in_seg = True
                start = t
            elif not m and in_seg:
                segs.append((start, t))
                in_seg = False
        if in_seg:
            segs.append((start, times.iloc[-1]))
        return segs

    segs = contiguous_segments(test_df["y_pred_thr"] == 1, test_df["timestamp"])
    for s, e in segs:
        fig.add_vrect(x0=s, x1=e, fillcolor="LightSkyBlue", opacity=0.25, line_width=0)

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Highlight probability",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show detected segments table
    st.subheader("Detected highlight segments (predicted)")
    seg_rows = [{"start_time": float(s), "end_time": float(e), "duration_s": float(e - s)} for s, e in segs]
    seg_df = pd.DataFrame(seg_rows)
    if seg_df.empty:
        st.info("No segments above the current threshold.")
    else:
        st.dataframe(seg_df, use_container_width=True)

    # download predictions
    st.download_button(
        "⬇️ Download test predictions CSV",
        data=test_df.to_csv(index=False).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv"
    )

    st.subheader("Save trained model")
    model_name = st.text_input("Model output path", "models/highlight_detector.pkl")
    if st.button("💾 Save model"):
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        joblib.dump(pipeline, model_name)
        st.success(f"Model saved to {model_name}")
