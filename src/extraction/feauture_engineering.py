import pandas as pd

def window_timeseries(df, feature_cols, win_s=3.0, hop_s=0.5):
    """
    Aggregate time-series features into fixed-size windows.

    Args:
        df (pd.DataFrame): Must contain 'timestamp' and feature columns.
        feature_cols (List[str]): Columns to aggregate.
        win_s (float): Window size in seconds.
        hop_s (float): Hop size in seconds.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    windows = []
    t = start
    while t + win_s <= end:
        mask = (df["timestamp"] >= t) & (df["timestamp"] < t + win_s)
        window_data = df.loc[mask, feature_cols]
        if not window_data.empty:
            row = {col: window_data[col].mean() for col in feature_cols}
            row["timestamp"] = t + win_s / 2
            windows.append(row)
        t += hop_s
    return pd.DataFrame(windows)
