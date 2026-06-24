"""
Configuration for Pose Anomaly Detection Pipeline
==================================================
Supports two modes:
  1. Video → MediaPipe pose extraction
  2. UCI HAR sensor time-series (pre-extracted)
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR       = os.path.join(BASE_DIR, "data", "videos")       # input videos
POSE_DIR        = os.path.join(BASE_DIR, "data", "poses")        # extracted .npy poses
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")          # saved models
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")               # detection results

# ─────────────────────────────────────────────
# DATA MODE
# ─────────────────────────────────────────────
# "pose"  → video/pose data (33 keypoints x 3 = 99 features)
# "ucihar" → UCI HAR sensor data (9 channels)
DATA_MODE = "ucihar"  # works for any pre-windowed .npy data (UCI HAR + NTU skeletons)

# ─────────────────────────────────────────────
# FEATURE DIMENSIONS
# ─────────────────────────────────────────────
# Pose mode
NUM_KEYPOINTS              = 33   # MediaPipe Pose landmarks
KEYPOINT_DIM               = 3    # x, y, visibility
POSE_FEATURE_DIM           = NUM_KEYPOINTS * KEYPOINT_DIM  # 99

# UCI HAR mode
UCIHAR_CHANNELS            = 9    # 3 body_acc + 3 body_gyro + 3 total_acc
UCIHAR_TIMESTEPS           = 128  # samples per window (2.56 sec @ 50Hz)

# ─────────────────────────────────────────────
# POSE EXTRACTION (video mode only)
# ─────────────────────────────────────────────
MEDIAPIPE_MODEL_COMPLEXITY = 2    # 0=lite, 1=full, 2=heavy
MIN_DETECTION_CONFIDENCE   = 0.5
MIN_TRACKING_CONFIDENCE    = 0.5

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
# Pose mode: sliding window over continuous pose sequence
WINDOW_LENGTH   = 30    # frames per window
WINDOW_STRIDE   = 10    # step between windows

# UCI HAR mode: data is already windowed at 128 timesteps
UCIHAR_WINDOW_LENGTH = 128
UCIHAR_WINDOW_STRIDE = 64

MASK_RATIO      = 0.20  # fraction of coordinates to mask
TRAIN_SPLIT     = 0.80  # 80% train, 20% validation

# ─────────────────────────────────────────────
# TRANSFORMER MODEL
# ─────────────────────────────────────────────
D_MODEL         = 128   # embedding dimension
NHEAD           = 8     # attention heads
NUM_LAYERS      = 4     # transformer encoder layers
DIM_FEEDFORWARD = 256   # FFN hidden size
DROPOUT         = 0.1

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
EPOCHS          = 50
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-5

# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────
ANOMALY_K       = 4.0   # threshold = mean + K * std (4 = safer, fewer false positives)
