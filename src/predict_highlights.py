import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pandas import DataFrame


def calc_highlight_probability(trained_model_path: str, highlights_path: str, motion: DataFrame, audio: DataFrame):
    model = joblib.load(trained_model_path)

    df = pd.merge_asof(motion.sort_values("timestamp"), audio.sort_values("timestamp"), on="timestamp")
    feature_cols = ["motion_intensity", "optical_flow", "rms", "zcr"]
    X = df[feature_cols]

    df["highlight_score"] = model.predict_proba(X)[:, 1]

    plt.figure(figsize=(12,4))
    plt.plot(df["timestamp"], df["highlight_score"], label="Predicted Highlight Probability")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title("Predicted Highlights Over Time")
    plt.legend()
    plt.show()

    df.to_csv(highlights_path, index=False)
    print(f"Saved highlight predictions to {highlights_path}")