"""
Convert UCI HAR Dataset to .npy format for the anomaly pipeline.
==============================================================
Reads the Inertial Signals from UCI HAR and saves per-activity
.npy files that the rest of the pipeline can consume.

UCI HAR folder structure expected:
    UCI HAR Dataset/
    ├── train/
    │   ├── Inertial Signals/
    │   │   ├── body_acc_x_train.txt
    │   │   ├── body_acc_y_train.txt
    │   │   ├── body_acc_z_train.txt
    │   │   ├── body_gyro_x_train.txt
    │   │   ├── body_gyro_y_train.txt
    │   │   ├── body_gyro_z_train.txt
    │   │   ├── total_acc_x_train.txt
    │   │   ├── total_acc_y_train.txt
    │   │   └── total_acc_z_train.txt
    │   ├── subject_train.txt
    │   └── y_train.txt
    ├── test/
    │   ├── Inertial Signals/
    │   │   └── ... (same files with _test suffix)
    │   ├── subject_test.txt
    │   └── y_test.txt
    └── activity_labels.txt

Usage:
    python convert_ucihar.py --input path/to/UCI\ HAR\ Dataset/
    python convert_ucihar.py --input path/to/UCI\ HAR\ Dataset/ --normal_activities 1 2 3
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path for config and src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import ensure_dirs, normalize_sensor_sequence

# UCI HAR activity labels
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Sensor signal files (9 channels)
SIGNAL_FILES = [
    "body_acc_x_{}.txt",
    "body_acc_y_{}.txt",
    "body_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
    "total_acc_x_{}.txt",
    "total_acc_y_{}.txt",
    "total_acc_z_{}.txt",
]


def load_signals(data_dir, split):
    """
    Load all 9 inertial signals and stack into one array.

    Args:
        data_dir: path to UCI HAR Dataset root
        split: "train" or "test"

    Returns:
        signals: np.ndarray of shape (num_samples, 128, 9)
    """
    signal_dir = os.path.join(data_dir, split, "Inertial Signals")
    channels = []

    for sig_file in SIGNAL_FILES:
        filepath = os.path.join(signal_dir, sig_file.format(split))
        if not os.path.exists(filepath):
            sys.exit(f"  ERROR: File not found → {filepath}")
        data = np.loadtxt(filepath)  # (num_samples, 128)
        channels.append(data)

    # stack along last axis: (num_samples, 128, 9)
    signals = np.stack(channels, axis=-1).astype(np.float32)
    return signals


def load_labels(data_dir, split):
    """Load activity labels and subject IDs."""
    labels_path = os.path.join(data_dir, split, f"y_{split}.txt")
    subjects_path = os.path.join(data_dir, split, f"subject_{split}.txt")

    labels = np.loadtxt(labels_path, dtype=int)
    subjects = np.loadtxt(subjects_path, dtype=int)
    return labels, subjects


def convert_ucihar(data_dir, output_dir, normal_activities=None):
    """
    Convert UCI HAR dataset to .npy format.

    Saves:
        - train_normal.npy    — all normal activity samples (for training)
        - test_per_activity/  — one .npy per activity (for evaluation)
        - dataset_info.npy    — metadata about the conversion

    Args:
        data_dir: path to UCI HAR Dataset root
        output_dir: where to save .npy files
        normal_activities: list of activity IDs considered "normal" (default: [1])
    """
    if normal_activities is None:
        normal_activities = [1, 2, 3, 5]  # WALKING, UPSTAIRS, DOWNSTAIRS, STANDING

    ensure_dirs(output_dir, os.path.join(output_dir, "test_per_activity"))

    print("Loading UCI HAR dataset...")

    # load train
    train_signals = load_signals(data_dir, "train")
    train_labels, train_subjects = load_labels(data_dir, "train")

    # load test
    test_signals = load_signals(data_dir, "test")
    test_labels, test_subjects = load_labels(data_dir, "test")

    print(f"  Train: {train_signals.shape} ({train_signals.shape[0]} samples)")
    print(f"  Test:  {test_signals.shape} ({test_signals.shape[0]} samples)")

    # ── normalize sensor sequences ──────────────────────────────
    print("\n  Normalizing sensor sequences (mean-subtract, std-scale, smooth)...")
    train_signals = normalize_sensor_sequence(train_signals)
    test_signals = normalize_sensor_sequence(test_signals)

    # ── save training data (normal activities only) ───────────────
    normal_mask = np.isin(train_labels, normal_activities)
    train_normal = train_signals[normal_mask]

    train_normal_path = os.path.join(output_dir, "train_normal.npy")
    np.save(train_normal_path, train_normal)
    print(f"\n  Normal activities: {[ACTIVITY_LABELS[a] for a in normal_activities]}")
    print(f"  Train normal: {train_normal.shape} → {train_normal_path}")

    # ── save all training data (for optional full training) ───────
    train_all_path = os.path.join(output_dir, "train_all.npy")
    np.save(train_all_path, train_signals)

    train_labels_path = os.path.join(output_dir, "train_labels.npy")
    np.save(train_labels_path, train_labels)

    # ── save test data per activity ───────────────────────────────
    print("\n  Test data per activity:")
    for act_id, act_name in ACTIVITY_LABELS.items():
        mask = test_labels == act_id
        if mask.sum() == 0:
            continue

        act_data = test_signals[mask]
        is_normal = act_id in normal_activities
        tag = "normal" if is_normal else "ANOMALY"

        act_path = os.path.join(
            output_dir, "test_per_activity",
            f"act{act_id}_{act_name}.npy"
        )
        np.save(act_path, act_data)
        print(f"    {tag:8s} act{act_id} {act_name:25s}: {act_data.shape}")

    # ── save full test data + labels ──────────────────────────────
    test_all_path = os.path.join(output_dir, "test_all.npy")
    np.save(test_all_path, test_signals)

    test_labels_path = os.path.join(output_dir, "test_labels.npy")
    np.save(test_labels_path, test_labels)

    # ── save metadata ─────────────────────────────────────────────
    info = {
        "num_channels": 9,
        "timesteps": 128,
        "normal_activities": normal_activities,
        "primary_anomaly_activities": [6],  # LAYING = primary anomaly
        "activity_labels": ACTIVITY_LABELS,
        "train_samples": train_signals.shape[0],
        "test_samples": test_signals.shape[0],
        "train_normal_samples": train_normal.shape[0],
    }
    info_path = os.path.join(output_dir, "dataset_info.npy")
    np.save(info_path, info)
    print(f"\n  Dataset info → {info_path}")

    return info


def main():
    parser = argparse.ArgumentParser(description="Convert UCI HAR to .npy format")
    parser.add_argument("--input", required=True,
                        help="Path to UCI HAR Dataset folder")
    parser.add_argument("--output", default=config.POSE_DIR,
                        help="Output directory for .npy files")
    parser.add_argument("--normal", nargs="+", type=int, default=[1, 2, 3, 5],
                        help="Activity IDs to treat as normal "
                             "(default: 1=WALKING 2=UPSTAIRS 3=DOWNSTAIRS 5=STANDING). "
                             "Options: 1=WALKING 2=UPSTAIRS 3=DOWNSTAIRS "
                             "4=SITTING 5=STANDING 6=LAYING")
    args = parser.parse_args()

    print("=" * 55)
    print("  UCI HAR → .npy Converter")
    print("=" * 55)

    info = convert_ucihar(args.input, args.output, args.normal)

    print("\n" + "=" * 55)
    print("  Conversion complete!")
    print("=" * 55)
    print(f"\n  Next steps:")
    print(f"    1. Train:  python src/train.py --data {args.output}")
    print(f"    2. Detect: python src/detect.py --npy {args.output}/test_all.npy")
    print(f"    3. Visual: python src/visualize.py --npy {args.output}/test_all.npy")


if __name__ == "__main__":
    main()
