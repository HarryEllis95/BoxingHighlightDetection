import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.train_model import train_highlight_model

st.set_page_config(page_title="Train Highlight Model", layout="wide")
st.title("Train Boxing Highlight Model")


st.header("Inputs")
visual_csv_path = st.text_input("Visual features CSV Path", "", width=500)
audio_csv_path = st.text_input("Audio features CSV Path", "", width=500)
labels_csv_path = st.text_input("Labels CSV Path", "", width=500)
test_size = st.slider("Test size", 0.05, 0.4, 0.20, 0.05)
random_state = st.number_input("Random seed", min_value=0, value=10, step=1, width=500)
time_based = st.checkbox("Use time-based chronological split (no random shuffle)", value=False)
do_calibrate = st.checkbox("Prefit calibration (recommended)", value=True)

train_btn = st.button("Train Model")

def must_exist(path: str, label: str):
    if not path or not os.path.isfile(path):
        st.error(f"{label} not found at: {path}")
        st.stop()

if train_btn:
    must_exist(visual_csv_path, "Visual features CSV")
    must_exist(audio_csv_path, "Audio features CSV")
    must_exist(labels_csv_path, "Labels CSV")

    with st.spinner("Training the model…"):
        pipeline, results = train_highlight_model(
            visual_features_csv=visual_csv_path,
            audio_features_csv=audio_csv_path,
            labels_csv=labels_csv_path,
            test_size=test_size,
            random_state=int(random_state),
            time_based_split=time_based,
            do_calibrate=do_calibrate
        )

    st.success("Training complete")

    # metrics overview
    test_roc = results.get("test_roc_auc", np.nan)
    test_pr = results.get("test_pr_auc", np.nan)
    chosen_threshold = results.get("threshold", 0.5)
    val_f1 = results.get("val_f1", np.nan)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("ROC-AUC (test)", f"{float(test_roc):.3f}")
    colB.metric("PR-AUC (Avg Precision test)", f"{float(test_pr):.3f}")
    colC.metric("Best F1 threshold (val)", f"{float(chosen_threshold):.2f}")
    colD.metric("Val F1 @ threshold", f"{float(val_f1):.3f}")

    st.subheader("Classification Report @ chosen threshold (test)")
    st.code(results.get("test_classification_report", "No classification report available"), language="text")

    test_df: pd.DataFrame = results.get("test_predictions", pd.DataFrame()).copy()

    if test_df.empty:
        st.warning("No test predictions available — check your data and splits.")
        st.stop()

    # if timestamps were unavailable during split, build an index-based timestamp
    if "timestamp" not in test_df.columns or test_df["timestamp"].isna().all():
        test_df["timestamp"] = np.arange(len(test_df), dtype=float)

    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    st.subheader("Explore predicted highlight probabilities")
    threshold = st.slider(
        "Probability threshold", min_value=0.05, max_value=0.95,
        value=float(results["threshold"]), step=0.05
    )

    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    st.subheader("Explore predicted highlight probabilities")
    threshold = st.slider(
        "Probability threshold", min_value=0.05, max_value=0.95,
        value=float(chosen_threshold), step=0.01
    )

    test_df["y_pred_thr"] = (test_df["y_proba"] >= threshold).astype(int)

    # confusion matrix at current threshold
    try:
        cm = confusion_matrix(test_df["y_true"], test_df["y_pred_thr"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn = fp = fn = tp = 0
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
        "⬇Download test predictions CSV",
        data=test_df.to_csv(index=False).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv"
    )

    st.subheader("Save trained model")
    model_name = st.text_input("Model output path", "models/highlight_detector.pkl")
    if st.button("Save model"):
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        joblib.dump(pipeline, model_name)
        st.success(f"Model saved to {model_name}")
