import os
import cv2
import numpy as np
import pandas as pd
import librosa
import librosa.feature

def compute_motion_features(frame_folder: str, video_path: str, use_optical_flow: bool = True
                            , smooth_window: int = 5, normalize: bool = True) -> pd.DataFrame:
    """Compute motion features (frame difference + optional optical flow) from passed frames
        Returns a DataFrame with columns:
      timestamp (s), motion_intensity, optical_flow (optional)
    """

    fps = get_framerate(frame_folder)

    frame_files = sorted(
        [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".jpg")]
    )
    if len(frame_files) < 2:
        raise ValueError("Need at least two frames to compute motion features")

    motion_scores = []
    flow_scores = []

    prev_gray = cv2.cvtColor(cv2.imread(frame_files[0]), cv2.COLOR_BGR2GRAY)

    # get jpg paths
    jpg_paths: list[str] = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".jpg")]
    frames = sorted(jpg_paths)
    motion_scores = []

    for i in range(1, len(frame_files)):
        frame = cv2.imread(frame_files[i])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # basic motion intensity (frame diff)
        diff = cv2.absdiff(gray, prev_gray)
        motion = np.mean(diff)
        motion_scores.append(motion)

        # optic flow mag
        if use_optical_flow:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_scores.append(np.mean(mag))
        prev_gray = gray

    # build timestamp (s)
    timestamps = np.arrange(1, len(motion_scores) + 1) / fps

    df = pd.DataFrame({
        "timestamp": timestamps,
        "motion_intensity": motion_scores,
    })

    if use_optical_flow:
        df["optical_flow"] = flow_scores

    # optional smooth
    if smooth_window > 1:
        df["motion_intensity"] = df["motion_intensity"].rolling(smooth_window, center=True).mean()
        if use_optical_flow:
            df["optical_flow"] = df["optical_flow"].rolling(smooth_window, center=True).mean()
    # optional normalization
    if normalize:
        for col in df.columns:
            if col != "timestamp":
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    df.dropna(inplace=True)
    return df


def get_framerate(frame_folder):
    # Load fps from metadata file
    metadata_path = os.path.join(frame_folder, "frame_metadata.txt")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    with open(metadata_path, "r") as f:
        line = f.readline().strip()
        if not line.startswith("fps="):
            raise ValueError(f"Invalid metadata format: {line}")
        try:
            fps = float(line.split("=")[1])
        except ValueError:
            raise ValueError(f"Could not parse FPS value from metadata: {line}")
    return fps


def compute_audio_features(audio_path: str):
    y, sr = librosa.load(audio_path)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    timestamps = librosa.times_like(rms, sr=sr)
    return pd.DataFrame({"timestamp": timestamps, "rms": rms, "zcr": zcr})

# if __name__ == "__main__":