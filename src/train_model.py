from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import os

def train_highlight_model(save_model: bool, save_model_path: Optional[str], features_csv: str = "../data/processed/features.csv"):
    df = pd.read_csv(features_csv)

    feature_cols = ["motion_intensity", "optical_flow", "rms", "zcr"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify=y
    )

    # build pipeline => scaling + classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_predict, digits=3))

    # ROC Curve
    fpr, tpr, _ =  roc_curve(y_test, y_probability)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model Performance (ROC Curve)")
    plt.legend()
    plt.show()

    if save_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/highlight_detector.pkl")
        print("✅ Model saved to models/highlight_detector.pkl")

if __name__ == "__main__":
    train_highlight_model(True, None)
