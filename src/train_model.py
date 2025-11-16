from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, \
    precision_recall_fscore_support

def assert_timestamp_column(df: pd.DataFrame, col: str, name: str) -> None:
    if col not in df.columns:
        raise ValueError(f"{name} must contain a '{col}' column.")


def to_timedelta_if_seconds(series: pd.Series) -> pd.Series:
    """
    Convert numeric seconds to Timedelta; pass through if already Timedelta.
    Datetimes are returned unchanged.

    note that expected inputs are numeric seconds or Timedelta. Datetimes are accepted
    but the rest of the pipeline expects labels and feature timestamps to share the same
    type (seconds or Timedelta). If you use datetimes, ensure labels are datetimes too.
    """
    if pd.api.types.is_timedelta64_dtype(series):
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        # Keep datetimes unchanged — merge_asof works with datetimes, and we do not convert them.
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_timedelta(series, unit="s")
    try:
        return pd.to_timedelta(series, unit="s")
    except Exception:
        raise ValueError("Timestamp series must be numeric seconds, pandas Timedelta, or datetime")


def expand_labels_to_timestamps(labels_df: pd.DataFrame, feature_ts: pd.Series, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Expand interval labels to match feature timestamps.
    Timestamps within any [start_time, end_time] are labeled as 'label', else 0.
    Works with float seconds or Timedelta; auto-aligns dtypes.
    """
    feature_ts = pd.Series(feature_ts).drop_duplicates().sort_values()
    if pd.api.types.is_timedelta64_dtype(feature_ts):
        start_time = to_timedelta_if_seconds(labels_df["start_time"])
        end_time = to_timedelta_if_seconds(labels_df["end_time"])
    elif pd.api.types.is_datetime64_any_dtype(feature_ts):
        start_time = labels_df["start_time"]
        end_time = labels_df["end_time"]
        if not pd.api.types.is_datetime64_any_dtype(start_time) or not pd.api.types.is_datetime64_any_dtype(end_time):
            raise ValueError("Feature timestamps are datetimes; label start_time/end_time must also be datetimes")
    else:
        start_time, end_time = labels_df["start_time"], labels_df["end_time"]

    label_map = pd.Series(0, index=feature_ts)
    for s, e, lab in zip(start_time, end_time, labels_df["label"].astype(int)):
        mask = (feature_ts >= s) & (feature_ts <= e)
        label_map[mask] = lab

    return pd.DataFrame({ts_col: label_map.index, "label": label_map.values})


def align_visual_audio(visual: pd.DataFrame, audio: pd.DataFrame, timestamp_col: str, tolerance: pd.Timedelta = pd.Timedelta("250ms")) -> pd.DataFrame:
    """Sort, convert to timedeltas, and merge_asof with nearest match within tolerance."""
    visual = visual.sort_values(timestamp_col).reset_index(drop=True).copy()
    audio = audio.sort_values(timestamp_col).reset_index(drop=True).copy()

    visual_ts = to_timedelta_if_seconds(visual[timestamp_col])
    audio_ts = to_timedelta_if_seconds(audio[timestamp_col])

    visual = visual.assign(**{timestamp_col: visual_ts}).sort_values(timestamp_col).reset_index(drop=True)
    audio = audio.assign(**{timestamp_col: audio_ts}).sort_values(timestamp_col).reset_index(drop=True)

    joined = pd.merge_asof(visual, audio, on=timestamp_col, direction="nearest", tolerance=tolerance, suffixes=("_vis", "_aud"),)
    return joined


def build_labeled_frame(features_df: pd.DataFrame, labels_df: pd.DataFrame, timestamp_col: str):
    per_ts_labels = expand_labels_to_timestamps(labels_df, features_df[timestamp_col], ts_col=timestamp_col)
    features_with_labels = features_df.merge(per_ts_labels, on=timestamp_col, how="left")
    features_with_labels["label"] = features_with_labels["label"].fillna(0).astype(int)

    features_with_labels = features_with_labels.reset_index(drop=True).copy()
    features_with_labels["row_id"] = np.arange(len(features_with_labels))

    numeric_feature_cols = [
        col for col in features_with_labels.columns
        if col not in [timestamp_col, "label", "row_id"] and np.issubdtype(features_with_labels[col].dtype, np.number)
    ]
    y = features_with_labels["label"].astype(int)
    row_id_series = features_with_labels["row_id"]
    return features_with_labels, row_id_series, y, numeric_feature_cols

def get_train_test_splits(X: pd.DataFrame, y: pd.Series, row_id_series: pd.Series, timestamps: pd.Series, test_size: float,
    val_frac_of_train: float, random_state: int, time_based: bool = False):

    if time_based:
        order = timestamps.sort_values().index  # original frame indices ordered by time
        n = len(order)
        cut_test = int(n * (1 - test_size))
        trainval_idx = order[:cut_test]
        test_idx = order[cut_test:]

        # From trainval, carve out validation as the tail portion
        n_trainval = len(trainval_idx)
        cut_val = int(n_trainval * (1 - val_frac_of_train))
        train_idx = trainval_idx[:cut_val]
        val_idx = trainval_idx[cut_val:]

        X_train, X_val, X_test = X.loc[train_idx], X.loc[val_idx], X.loc[test_idx]
        y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
        row_id_train = row_id_series.loc[train_idx]
        row_id_val   = row_id_series.loc[val_idx]
        row_id_test  = row_id_series.loc[test_idx]
    else:
        # Random stratified splitting preserving original indices
        X_trainval, X_test, y_trainval, y_test, row_id_trainval, row_id_test = train_test_split(
            X, y, row_id_series, test_size=test_size, stratify=y, random_state=random_state
        )
        X_train, X_val, y_train, y_val, row_id_train, row_id_val = train_test_split(
            X_trainval, y_trainval, row_id_trainval, test_size=val_frac_of_train,
            stratify=y_trainval, random_state=random_state
        )
    return X_train, X_val, X_test, y_train, y_val, y_test, row_id_train, row_id_val, row_id_test

def compute_sample_weights(y_train: pd.Series) -> np.ndarray:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        return np.ones(len(y_train), dtype=float)
    w_pos = neg / float(pos)
    return np.where(y_train == 1, w_pos, 1.0).astype(float)

def build_pipeline(feature_cols: list, random_state: int, use_scaler: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    numeric_transform = Pipeline(steps=steps)
    preproc = ColumnTransformer([("num", numeric_transform, feature_cols)], remainder="drop")
    clf = GradientBoostingClassifier(random_state=random_state)
    return Pipeline([("prep", preproc), ("clf", clf)])



def pick_threshold_on_val(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    thresholds = np.linspace(0.1, 0.9, 17)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best.update({"threshold": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)})
    return best


def evaluate_on_test(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision_recall_f1": precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[:3],
        "classification_report": classification_report(y_true, y_pred, digits=3),
        "y_pred": y_pred,
    }

# pipeline merges visual and audio features, expands interval labels into per-second targets, and trains a
# highlight classifier using Gradient Boosting.
# It includes temporal smoothing, class imbalance weighting, and threshold tuning to try and optimize F1 score.
# output includes a trained model and timestamped predictions
def train_highlight_model(
    visual_features_csv: str,
    audio_features_csv: str,
    labels_csv: str,
    timestamp_col: str = "timestamp",
    test_size: float = 0.2,
    val_frac_of_train: float = 0.25,   # 60/20/20 split overall with default test_size=0.2
    random_state: int = 42,
    asof_tolerance: str = "250ms",
    time_based_split: bool = False,
    do_calibrate: bool = True,
    calibration_frac_of_train: float = 0.2 ) -> Tuple[Pipeline, Dict[str, Any]]:

    """
    Build, train, and evaluate a scikit-learn pipeline for highlight detection.

    Parameters
    - visual_features_csv (str): path to visual features CSV (must contain timestamp column)
    - audio_features_csv (str): path to audio features CSV (must contain timestamp column)
    - labels_csv (str): path to labels CSV with columns start_time, end_time, label
    - timestamp_col (str): name of the timestamp column in feature CSVs ("timestamp" expected)
    - test_size (float): fraction of data held out as final test set
    - val_frac_of_train (float): fraction of trainval used for validation
    - random_state (int): RNG seed for reproducible random splits
    - asof_tolerance (str|pd.Timedelta): tolerance for merge_asof when aligning modalities
    - time_based_split (bool): if True perform chronological train/val/test splits; else stratified random splits
    - do_calibrate (bool): if True perform prefit probability calibration using a held-out calibration subset of the training set
    - calibration_frac_of_train (float): fraction of the training set reserved for calibration when do_calibrate=True

    """

    visual_features = pd.read_csv(visual_features_csv)
    audio_features = pd.read_csv(audio_features_csv)
    labels = pd.read_csv(labels_csv)

    assert_timestamp_column(visual_features, timestamp_col, "visual_features_csv")
    assert_timestamp_column(audio_features, timestamp_col, "audio_features_csv")

    # align by time
    joined_features = align_visual_audio(visual_features, audio_features, timestamp_col, tolerance=pd.Timedelta(asof_tolerance))

    # Label and select numeric features (adds row_id)
    labeled_features, row_id_all, y_all, numeric_feature_cols = build_labeled_frame(joined_features, labels, timestamp_col)
    X_all = labeled_features[numeric_feature_cols]
    timestamps_all = labeled_features[timestamp_col]

    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())

    X_train, X_val, X_test, y_train, y_val, y_test, row_id_train, row_id_val, row_id_test = get_train_test_splits(
        X_all, y_all, row_id_all, timestamps_all,
        test_size=test_size, val_frac_of_train=val_frac_of_train,
        random_state=random_state, time_based=time_based_split
    )

    # note calibration_frac_of_train is fraction of the training set to use for calibration
    if do_calibrate:
        if time_based_split:  # by default
            train_idx_sorted = X_train.index.sort_values()
            n_train = len(train_idx_sorted)
            cut_calib = int(n_train * (1 - calibration_frac_of_train))
            core_idx = train_idx_sorted[:cut_calib]
            calib_idx = train_idx_sorted[cut_calib:]
            X_train_core, X_calib = X_train.loc[core_idx], X_train.loc[calib_idx]
            y_train_core, y_calib = y_train.loc[core_idx], y_train.loc[calib_idx]
            idx_train_core, idx_calib = row_id_train.loc[core_idx], row_id_train.loc[calib_idx]
        else:
            X_train_core, X_calib, y_train_core, y_calib, idx_train_core, idx_calib = train_test_split(
                X_train, y_train, row_id_train, test_size=calibration_frac_of_train, stratify=y_train, random_state=random_state
            )
    else:
        X_train_core, y_train_core = X_train, y_train
        X_calib, y_calib = None, None
        idx_train_core, idx_calib = row_id_train, None

    # class weighting via sample_weight on training core set
    w_train_core = compute_sample_weights(y_train_core)

    base_pipeline = build_pipeline(numeric_feature_cols, random_state)
    base_pipeline.fit(X_train_core, y_train_core, **{"clf__sample_weight": w_train_core})

    # prefit calibration
    if do_calibrate:
        # Calibrate using the held-out calibration set (no sample weights for calibration step)
        calibrator = CalibratedClassifierCV(estimator=base_pipeline, cv="prefit", method="isotonic")
        calibrator.fit(X_calib, y_calib)
        pipeline = calibrator
    else:
        pipeline = base_pipeline

    # choose threshold on validation set
    y_proba_val = pipeline.predict_proba(X_val)[:, 1]
    best_threshold = pick_threshold_on_val(y_val.values, y_proba_val)

    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    test_eval = evaluate_on_test(y_test.values, y_proba_test, best_threshold["threshold"])

    timestamp_by_row_id = pd.Series(timestamps_all.values, index=row_id_all.values)
    test_pred_df = pd.DataFrame({
        "row_id": row_id_test.values,
        timestamp_col: timestamp_by_row_id.loc[row_id_test.values].values,
        "y_true": y_test.values,
        "y_proba": y_proba_test,
        "y_pred": (y_proba_test >= best_threshold["threshold"]).astype(int),
    })

    results: Dict[str, Any] = {
        "feature_cols": numeric_feature_cols,
        "threshold": best_threshold["threshold"],
        "val_f1": best_threshold["f1"],
        "val_precision": best_threshold["precision"],
        "val_recall": best_threshold["recall"],
        "test_roc_auc": test_eval["roc_auc"],
        "test_pr_auc": test_eval["pr_auc"],
        "test_classification_report": test_eval["classification_report"],
        "test_predictions": test_pred_df,
        "n_pos_total": n_pos,
        "n_neg_total": n_neg,
    }

    # if save_model:
    #     os.makedirs("models", exist_ok=True)
    #     joblib.dump(pipeline, "models/highlight_detector.pkl")

    return pipeline, results
