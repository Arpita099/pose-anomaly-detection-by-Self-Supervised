"""
Shared utility functions for the Pose Anomaly Detection Pipeline.
"""

import os
import numpy as np
import torch
from pathlib import Path


def normalize_poses(poses, video_width, video_height):
    """
    Normalize pose coordinates from pixels to [0, 1].

    Args:
        poses: np.ndarray of shape (num_frames, num_keypoints, 3)
        video_width: original video width in pixels
        video_height: original video height in pixels

    Returns:
        Normalized poses (x and y in [0,1], visibility unchanged)
    """
    norm = poses.copy()
    norm[:, :, 0] /= video_width    # x
    norm[:, :, 1] /= video_height   # y
    # visibility (index 2) stays as-is
    return norm


def denormalize_poses(poses, video_width, video_height):
    """
    Convert normalized [0,1] coordinates back to pixel space.

    Args:
        poses: np.ndarray of shape (..., num_keypoints, 3)
        video_width: original video width
        video_height: original video height

    Returns:
        Poses in pixel coordinates
    """
    px = poses.copy()
    px[..., 0] *= video_width
    px[..., 1] *= video_height
    return px


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_videos(directory, extensions=None):
    """
    Find all video files in a directory.

    Args:
        directory: path to search
        extensions: list of file extensions (default: common video formats)

    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]

    videos = []
    for ext in extensions:
        videos.extend(Path(directory).glob(f"*{ext}"))
        videos.extend(Path(directory).glob(f"*{ext.upper()}"))
    return sorted(videos)


def find_poses(directory):
    """Find all .npy pose files in a directory."""
    return sorted(Path(directory).glob("*.npy"))


def normalize_sensor_sequence(sequences, smooth_window=3):
    """
    Normalize sensor time-series sequences per sample.

    Applies three steps:
        1. Global motion removal — subtract per-sample per-channel mean
        2. Scale normalization — divide by per-sample per-channel std (floor=1e-6)
        3. Trajectory smoothing — moving average filter per channel

    Args:
        sequences: np.ndarray of shape (num_samples, timesteps, channels)
        smooth_window: window size for moving average (default 3)

    Returns:
        Normalized sequences, same shape as input
    """
    norm = sequences.copy()

    # 1. Remove global motion shift (per-sample per-channel mean)
    mean = norm.mean(axis=1, keepdims=True)  # (N, 1, C)
    norm = norm - mean

    # 2. Normalize scale (per-sample per-channel std)
    std = norm.std(axis=1, keepdims=True)     # (N, 1, C)
    std = np.maximum(std, 1e-6)               # avoid division by zero
    norm = norm / std

    # 3. Smooth trajectories (moving average per channel)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        # apply along timesteps axis for each sample and channel
        for i in range(norm.shape[0]):
            for c in range(norm.shape[2]):
                norm[i, :, c] = np.convolve(norm[i, :, c], kernel, mode="same")

    return norm
