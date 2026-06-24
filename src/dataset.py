"""
Step 2: Pose Sequence Dataset
================================
Creates sequences from .npy files with random masking
for self-supervised training.

Supports two modes:
  - "pose" mode: sliding windows over continuous pose sequences
  - "ucihar" mode: pre-windowed sensor data from UCI HAR

Usage (standalone test):
    python dataset.py --data data/poses/
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# allow running as a script: put project root (for config) and src/ on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import find_poses


class PoseSequenceDataset(Dataset):
    """
    Self-supervised dataset for continuous pose sequences.
    Uses sliding windows with random masking.
    """

    def __init__(self, pose_dir, window_length=None, stride=None,
                 mask_ratio=None, split="train", train_ratio=None):
        self.window_length = window_length or config.WINDOW_LENGTH
        self.stride = stride or config.WINDOW_STRIDE
        self.mask_ratio = mask_ratio or config.MASK_RATIO
        self.train_ratio = train_ratio or config.TRAIN_SPLIT

        pose_files = find_poses(pose_dir)
        if len(pose_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {pose_dir}")

        all_windows = []
        for pf in pose_files:
            data = np.load(str(pf))  # (frames, features)
            if data.ndim == 3:
                data = data.reshape(data.shape[0], -1)

            for start in range(0, data.shape[0] - self.window_length + 1, self.stride):
                window = data[start : start + self.window_length]
                all_windows.append(window)

        self.windows = np.array(all_windows, dtype=np.float32)
        n_total = len(self.windows)
        n_train = int(n_total * self.train_ratio)

        if split == "train":
            self.windows = self.windows[:n_train]
        else:
            self.windows = self.windows[n_train:]

        self.feature_dim = self.windows.shape[2]
        print(f"  [pose/{split}] {len(self.windows)} windows, feature_dim={self.feature_dim}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        target = self.windows[idx].copy()
        mask_binary = np.zeros_like(target)
        num_to_mask = int(self.mask_ratio * target.size)
        mask_indices = np.random.choice(target.size, num_to_mask, replace=False)
        mask_binary.flat[mask_indices] = 1.0
        masked_input = target.copy()
        masked_input[mask_binary == 1] = 0.0

        return (
            torch.from_numpy(masked_input),
            torch.from_numpy(target),
            torch.from_numpy(mask_binary),
        )


class UCIHARDataset(Dataset):
    """
    Self-supervised dataset for UCI HAR sensor data.
    Data is already windowed at 128 timesteps.
    """

    def __init__(self, npy_path, mask_ratio=None, split="train", train_ratio=None):
        self.mask_ratio = mask_ratio or config.MASK_RATIO
        self.train_ratio = train_ratio or config.TRAIN_SPLIT

        data = np.load(npy_path)  # (num_samples, 128, 9)
        if data.ndim == 2:
            data = data.reshape(data.shape[0], config.UCIHAR_TIMESTEPS, -1)

        n_total = data.shape[0]
        n_train = int(n_total * self.train_ratio)

        if split == "train":
            self.data = data[:n_train].astype(np.float32)
        else:
            self.data = data[n_train:].astype(np.float32)

        self.feature_dim = self.data.shape[2]
        print(f"  [ucihar/{split}] {len(self.data)} samples, feature_dim={self.feature_dim}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data[idx].copy()
        mask_binary = np.zeros_like(target)
        num_to_mask = int(self.mask_ratio * target.size)
        mask_indices = np.random.choice(target.size, num_to_mask, replace=False)
        mask_binary.flat[mask_indices] = 1.0
        masked_input = target.copy()
        masked_input[mask_binary == 1] = 0.0

        return (
            torch.from_numpy(masked_input),
            torch.from_numpy(target),
            torch.from_numpy(mask_binary),
        )


def get_dataloaders(data_path, batch_size=None):
    """Create train and validation dataloaders (auto-detects mode)."""
    batch_size = batch_size or config.BATCH_SIZE

    if config.DATA_MODE == "ucihar":
        # auto-detect train_normal.npy in the data directory
        if os.path.isdir(data_path):
            npy_path = os.path.join(data_path, "train_normal.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(
                    f"train_normal.npy not found in {data_path}. "
                    f"Run convert_ucihar.py first."
                )
        else:
            npy_path = data_path
        train_ds = UCIHARDataset(npy_path, split="train")
        val_ds = UCIHARDataset(npy_path, split="val")
    else:
        train_ds = PoseSequenceDataset(data_path, split="train")
        val_ds = PoseSequenceDataset(data_path, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data", default=config.POSE_DIR)
    args = parser.parse_args()

    print(f"Data mode: {config.DATA_MODE}")
    train_loader, val_loader = get_dataloaders(args.data)

    for masked, target, mask in train_loader:
        print(f"  Batch: masked={masked.shape}, target={target.shape}, mask={mask.shape}")
        print(f"  Mask ratio: {mask.mean():.2%}")
        break

    print("OK!")
