import os
import cv2
import numpy as np
import pandas as pd
import librosa
import librosa.feature

def compute_motion_features(frame_folder: str) -> pd.DataFrame:
    """Compute simple motion intensity from passed frames"""
    # get jpg paths
    jpg_paths: list[str] = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".jpg")]
    frames = sorted(jpg_paths)
    motion_scores = []

    prev = None
    for fpath in frames:
        frame = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if prev is not None:
            diff = cv2.absdiff(frame, prev)
            motion_scores.append(np.mean(diff))
        prev = frame
    timestamps = np.arange(len(motion_scores))
    return pd.DataFrame({"timestamp": timestamps, "motion_intensity": motion_scores})

def compute_audio_features(audio_path: str):
    y, sr = librosa.load(audio_path)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    timestamps = librosa.times_like(rms, sr=sr)
    return pd.DataFrame({"timestamp": timestamps, "rms": rms, "zcr": zcr})

# if __name__ == "__main__":