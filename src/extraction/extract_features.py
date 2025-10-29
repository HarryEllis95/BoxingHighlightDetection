import os
import re
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import librosa
import librosa.feature

from src.extraction.visual_feature_extraction import compute_frame_diff_intensity, compute_optical_flow_speed, \
    compute_high_motion_pixel_ratio, compute_center_vs_edge_motion_ratio, compute_motion_direction_consistency


def compute_motion_features(frame_folder: str, center_box_frac: float = 1/3) -> pd.DataFrame:
    """
    Compute a compact, high-signal set of motion features suitable for scikit-learn models
    to detect boxing highlights.

    Args:
    frame_folder: Path to folder containing extracted frame images.
    center_box_frac: Fraction of the frame's height and width used to define the center box.
        This box is used to try and bias motion analysis toward the ring, where most fight action occurs.
        A smaller value (e.g. 0.2) focuses tightly on the center; a larger value (e.g. 0.5) includes more peripheral motion.
        Default of 1/3 is arbitrary but seems to provide roughly the best result

    Output columns:
      - timestamp (s)               : midpoint time of each frame-pair interval (i-0.5)/fps
      - motion_mean                 : mean absolute frame difference
      - flow_mean                   : mean optical-flow magnitude
      - flow_pct_high               : fraction of pixels with high motion (robust threshold)
      - center_motion_ratio         : center vs edge motion intensity ratio
      - flow_dir_consistency        : directional alignment of motion (0..1)
    """

    fps = get_framerate(frame_folder)

    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".jpg")]
    frame_files.sort(key=natural_key)

    if len(frame_files) < 2:
        raise ValueError("Need at least two frames to compute motion features")

    first = cv2.imread(frame_files[0])
    if first is None:
        raise ValueError(f"Could not read: {frame_files[0]}")
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    height, width = prev_gray.shape
    center_mask, edge_mask = create_center_edge_masks(height, width, center_box_frac)

    ts, motion_mean_list = [], []
    flow_mean_list, flow_pct_high_list = [], []
    center_ratio_list, dir_consistency_list = [], []

    # iterate frame pairs
    valid_index = 0
    for i in range(1, len(frame_files)):
        frame = cv2.imread(frame_files[i])
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # gray frame

        # Feature: motion_mean
        motion_mean = compute_frame_diff_intensity(prev_gray, gray)

        # optical flow once per pair
        # farneback_flow gives us pixel-level motion vectors
        mag, ang = farneback_flow(prev_gray, gray)

        # Feature: flow_mean
        flow_mean = compute_optical_flow_speed(mag)

        # Feature: flow_pct_high
        flow_pct_high = compute_high_motion_pixel_ratio(mag)

        # Feature: center_motion_ratio
        center_motion_ratio = compute_center_vs_edge_motion_ratio(mag, center_mask, edge_mask)

        # Feature: flow_dir_consistency
        flow_dir_consistency = compute_motion_direction_consistency(mag, ang)

        # Save
        valid_index += 1
        ts.append((valid_index - 0.5) / fps)
        motion_mean_list.append(motion_mean)
        flow_mean_list.append(flow_mean)
        flow_pct_high_list.append(flow_pct_high)
        center_ratio_list.append(center_motion_ratio)
        dir_consistency_list.append(flow_dir_consistency)

        prev_gray = gray

    df = pd.DataFrame({
        "timestamp": np.array(ts, dtype=float),
        "motion_mean": motion_mean_list,
        "flow_mean": flow_mean_list,
        "flow_pct_high": flow_pct_high_list,
        "center_motion_ratio": center_ratio_list,
        "flow_dir_consistency": dir_consistency_list,
    })

    return df

_num_re = re.compile(r'(\d+)')    # regex match one or more digits
def natural_key(s: str):
    """Sort helper so frame2.jpg < frame10.jpg - i.e expected frame order."""
    base = os.path.basename(s)
    parts = _num_re.split(base)
    return [(int(part) if part.isdigit() else part.lower()) for part in parts]

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

# idea is to bias towards center - averaged over fight most of the action happens in the center
# using a center mask should help boost real fight motion, suppress noise (camera pans, crowd shots)
def create_center_edge_masks(frame_height: int, frame_width: int, center_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Build boolean masks for a central box (size = frac of H and W) and the complement (edges). """
    # Compute center box dimensions
    center_height = max(1, int(frame_height * center_fraction))
    center_width = max(1, int(frame_width * center_fraction))

    # Compute top-left corner of the center box
    top = (frame_height - center_height) // 2
    left = (frame_width - center_width) // 2

    # Initialize full-frame mask
    center_mask = np.zeros((frame_height, frame_width), dtype=bool)

    center_mask[top:top + center_height, left:left + center_width] = True

    # Edge mask is the inverse
    edge_mask = ~center_mask

    return center_mask, edge_mask

# optical flow gives richer motion info the simple frame differences
def farneback_flow(prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute dense optical flow (Farnebäck) and return (magnitude, angle). """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, ang

def compute_audio_features(audio_path: str):
    y, sr = librosa.load(audio_path)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    timestamps = librosa.times_like(rms, sr=sr)
    return pd.DataFrame({"timestamp": timestamps, "rms": rms, "zcr": zcr})

# if __name__ == "__main__":