import numpy as np
import cv2

# Feature functions

# Big punches & flurries cause sharp inter-frame intensity changes; this is a cheap,
# but strong proxy for “activity level” that can correlate with highlights.
def compute_frame_diff_intensity(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """ motion_mean: Returns the average absolute pixel difference between two grayscale frames — a proxy for motion intensity. """
    diff = cv2.absdiff(curr_frame, prev_frame)
    return float(np.mean(diff))

# Measures overall motion speed, less sensitive to lighting than raw differences;
# Should spike during sustained exchanges and scrambles.
def compute_optical_flow_speed(flow_mag: np.ndarray) -> float:
    """ flow_mean: Returns the average optical flow magnitude — a measure of overall motion speed. """
    return float(np.mean(flow_mag))

# Distinguishes small/local jitters (camera shake, glove twitch) from widespread action
# where many pixels move strongly—typical of highlight sequences.
def compute_high_motion_pixel_ratio(flow_mag: np.ndarray) -> float:
    """ flow_pct_high: Fraction of pixels exceeding a robust, per-frame motion threshold. """
    _median = np.median(flow_mag)
    mad = np.median(np.abs(flow_mag - _median)) + 1e-6
    tau = _median + 2.0 * mad
    return float(np.mean(flow_mag > tau))

# In-ring action lives near the center; camera pans/crowd shots light up edges.
# This ratio boosts genuine fight activity and should suppress broadcast artifacts.
def compute_center_vs_edge_motion_ratio(flow_mag: np.ndarray, center_mask: np.ndarray, edge_mask: np.ndarray) -> float:
    """ center_motion_ratio:
      Returns the ratio of motion intensity in the center vs. edges — useful for filtering out broadcast artifacts."""
    eps = 1e-6
    center_mean = float(np.mean(flow_mag[center_mask])) if np.any(center_mask) else 0.0
    edge_mean = float(np.mean(flow_mag[edge_mask])) if np.any(edge_mask) else 0.0
    return center_mean / (edge_mean + eps)

# Camera pans/zooms → directions aligned (R≈1). Exchanges → directions divergent (R low).
# Should help down-weight pure camera motion vs real in-ring exchanges.
def compute_motion_direction_consistency(flow_mag: np.ndarray, flow_ang: np.ndarray) -> float:
    """ flow_dir_consistency: Magnitude-weighted directional alignment """
    w = flow_mag + 1e-6
    w_mean = np.mean(w)
    c = np.mean(np.cos(flow_ang) * w) / w_mean
    s = np.mean(np.sin(flow_ang) * w) / w_mean
    return float(np.sqrt(c * c + s * s))