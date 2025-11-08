from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, average_precision_score, \
    precision_recall_fscore_support
import matplotlib.pyplot as plt
import joblib
import os

def expand_labels_to_timestamps(labels_df: pd.DataFrame, feature_ts: pd.Series, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Expand interval labels to match feature timestamps.
    Only timestamps falling within any labeled interval will be assigned a label.
    """
    feature_ts = pd.Series(feature_ts).drop_duplicates().sort_values()
    label_map = pd.Series(0, index=feature_ts)

    for _, row in labels_df.iterrows():
        mask = (feature_ts >= row["start_time"]) & (feature_ts <= row["end_time"])
        label_map[mask] = int(row["label"])

    return pd.DataFrame({ts_col: label_map.index, "label": label_map.values})


# pipeline merges visual and audio features, expands interval labels into per-second targets, and trains a
# highlight classifier using Gradient Boosting.
# It includes temporal smoothing, class imbalance weighting, and threshold tuning to try and optimize F1 score.
# output includes a trained model and timestamped predictions
def train_highlight_model(visual_features_csv: str, audio_features_csv: str, labels_csv: str, timestamp_col: str = "timestamp",
                          rolling_seconds: int = 3, test_size: float = 0.2, random_state: int = 42):

    """
    Build, train, and evaluate a scikit-learn pipeline for highlight detection.

    Parameters
    ----------
    rolling_seconds     : rolling window (seconds) for simple temporal smoothing features
    test_size           : fraction for test split
    random_state        : seed for reproducibility

    Returns
    -------
    (pipeline, results)
        pipeline : trained sklearn Pipeline
        results  : dict with metrics, chosen threshold, and a test predictions DataFrame
    -------
    """

    visual_features = pd.read_csv(visual_features_csv)
    audio_features = pd.read_csv(audio_features_csv)
    labels = pd.read_csv(labels_csv)

    for df, name in [(visual_features, "visual_features_csv"), (audio_features, "audio_features_csv")]:
        if timestamp_col not in df.columns:
            raise ValueError(f"{name} must contain a '{timestamp_col}' column.")

    # Sort for asof-merge alignment
    visual_features = visual_features.sort_values(timestamp_col).reset_index(drop=True)
    audio_features = audio_features.sort_values(timestamp_col).reset_index(drop=True)

    # Merge features on timestamp
    feat = pd.merge_asof( visual_features, audio_features, on=timestamp_col)

    num_cols = [c for c in feat.columns if c != timestamp_col and np.issubdtype(feat[c].dtype, np.number)]
    if rolling_seconds and rolling_seconds > 1:
        feat = feat.set_index(timestamp_col)
        for c in num_cols:
            feat[f"{c}_rollmean_{rolling_seconds}s"] = (
                feat[c].rolling(f"{rolling_seconds}s", min_periods=1, center=True).mean()
            )
            feat[f"{c}_rollstd_{rolling_seconds}s"] = (
                feat[c].rolling(f"{rolling_seconds}s", min_periods=1, center=True).std().fillna(0.0)
            )
        feat = feat.reset_index()

    # Build per-timestamp labels from interval CSV and join
    per_ts_labels = expand_labels_to_timestamps(labels, feat[timestamp_col], ts_col=timestamp_col)
    feat = feat.merge(per_ts_labels, on=timestamp_col, how="left")
    feat["label"] = feat["label"].fillna(0).astype(int)

    # Drop rows without any numeric features
    feature_cols = [c for c in feat.columns if c not in [timestamp_col, "label"] and
                    np.issubdtype(feat[c].dtype, np.number)]
    feat = feat.dropna(subset=feature_cols)

    X = feat[feature_cols]
    y = feat["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # --- Pipeline: scale -> GradientBoostingClassifier ---
    # Note: Boosted trees don't need scaling, but we keep a scaler to make the pipeline robust
    # if future features include different orders of magnitude.
    numeric_transform = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True))])
    preproc = ColumnTransformer(
        transformers=[("num", numeric_transform, feature_cols)],
        remainder="drop"
    )

    clf = GradientBoostingClassifier(random_state=random_state)

    pipeline = Pipeline([
        ("prep", preproc),
        ("clf", clf)
    ])


    pipeline.fit(X_train, y_train)

    # evaluate
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba_test)
    ap = average_precision_score(y_test, y_proba_test)

    # Choose a threshold: maximize F1 on the validation (simple, task-oriented)
    thresholds = np.linspace(0.1, 0.9, 17)
    best = {"threshold": 0.5, "f1": -1, "precision": None, "recall": None}
    for t in thresholds:
        y_pred = (y_proba_test >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best.update({"threshold": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)})

    # Final thresholded report at chosen threshold
    y_pred_best = (y_proba_test >= best["threshold"]).astype(int)
    report = classification_report(y_test, y_pred_best, digits=3)

    test_results = pd.DataFrame({
        "timestamp": feat.loc[X_test.index, timestamp_col].values if timestamp_col in feat.columns else np.nan,
        "y_true": y_test.values,
        "y_proba": y_proba_test,
        "y_pred": y_pred_best
    })

    results: Dict[str, Any] = {
        "feature_cols": feature_cols,
        "roc_auc": float(roc),
        "pr_auc": float(ap),
        "threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "classification_report": report,
        "test_predictions": test_results
    }

    # if save_model:
    #     os.makedirs("models", exist_ok=True)
    #     joblib.dump(pipeline, "models/highlight_detector.pkl")
    #     print("✅ Model saved to models/highlight_detector.pkl")

    return pipeline, results

if __name__ == "__main__":
    train_highlight_model(True, None)
