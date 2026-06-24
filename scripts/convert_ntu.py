"""
Convert NTU RGB+D Skeleton Files to .npy Format
==================================================
Reads .skeleton files from NTU RGB+D and saves per-action .npy arrays.

Filename format: SsssCcccPpppRrrrAaaa.skeleton
  sss = setup ID, ccc = camera ID, ppp = performer ID, rrr = replication, aaa = action ID

NTU RGB+D 60 Actions:
  A1-A10  : Daily actions (drink, eat, read, etc.)
  A11-A20 : Health-related (sneeze, headache, etc.)
  A21-A30 : Eating/drinking actions
  A31-A40 : Communication (point, wave, etc.)
  A41-A50 : Exercise (jump, run, etc.)
  A51-A60 : Misc (fall, walk, etc.)

Usage:
    python convert_ntu.py --input data/nturgb+d_skeletons/
    python convert_ntu.py --input data/nturgb+d_skeletons/ --normal_actions 50 51 52
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to path for config and src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import ensure_dirs


# NTU RGB+D 60 action labels
ACTION_LABELS = {
    1: "drink_water", 2: "eat meal/snack", 3: "brush teeth",
    4: "brush hair", 5: "drop", 6: "pickup",
    7: "throw", 8: "sit down", 9: "stand up",
    10: "clapping", 11: "reading", 12: "writing",
    13: "tear up paper", 14: "wear jacket", 15: "take off jacket",
    16: "wear a shoe", 17: "take off a shoe", 18: "wear on glasses",
    19: "take off glasses", 20: "put on a hat/cap",
    21: "take off a hat/cap", 22: "cheer up", 23: "hand waving",
    24: "kicking something", 25: "reach into pocket",
    26: "hopping", 27: "jump up", 28: "make a phone call",
    29: "playing with phone", 30: "typing on keyboard",
    31: "pointing to something", 32: "taking a selfie",
    33: "check time from watch", 34: "rub two hands",
    35: "nod head/bow", 36: "shake head", 37: "wipe face",
    38: "salute", 39: "put palms together", 40: "cross hands",
    41: "arm circles", 42: "arm curls", 43: "stretch",
    44: "punch", 45: "kick", 46: "push up",
    47: "walking", 48: "running", 49: "jumping",
    50: "pat on back", 51: "point finger", 52: "hugging",
    53: "giving object", 54: "touch pocket", 55: "shaking hands",
    56: "walking towards", 57: "walking apart", 58: "put on headphones",
    59: "take off headphones", 60: "falling",
}


def parse_skeleton_file(filepath):
    """
    Parse a single NTU RGB+D .skeleton file.

    Format per file:
        num_frames
        [for each frame:]
            num_skeletons (int)
            [for each skeleton:]
                skeleton_info line (10 values): bodyID, 6×clipedEdges, bodyX, bodyY, trackingState
                num_joints (int, usually 25)
                [for each joint:] 10+ values per line: x, y, z, colorX, colorY, depthX, depthY, qw, qx, qy, qz, trackingState

    Returns:
        frames: np.ndarray of shape (num_frames, 25, 3) — x, y, z coordinates
        or None if parsing fails
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')

        idx = 0
        num_frames = int(float(lines[idx])); idx += 1

        all_frames = []

        for _ in range(num_frames):
            num_skeletons = int(float(lines[idx])); idx += 1

            frame_joints = []

            for _ in range(num_skeletons):
                # skeleton info line (10 values) — skip it
                idx += 1

                # num_joints on next line
                num_joints = int(float(lines[idx])); idx += 1

                for _ in range(num_joints):
                    joint_data = lines[idx].split()
                    idx += 1

                    if len(joint_data) >= 3:
                        x, y, z = float(joint_data[0]), float(joint_data[1]), float(joint_data[2])
                        frame_joints.append([x, y, z])

            # pad or trim to exactly 25 joints
            while len(frame_joints) < 25:
                frame_joints.append([0.0, 0.0, 0.0])
            frame_joints = frame_joints[:25]

            all_frames.append(frame_joints)

        if len(all_frames) == 0:
            return None

        return np.array(all_frames, dtype=np.float32)  # (frames, 25, 3)

    except Exception as e:
        print(f"  WARNING: Failed to parse {os.path.basename(filepath)}: {e}")
        return None


def parse_filename(filename):
    """
    Parse NTU RGB+D filename to extract metadata.
    SsssCcccPpppRrrrAaaa.skeleton
    """
    stem = Path(filename).stem
    try:
        setup = int(stem[1:4])
        camera = int(stem[5:8])
        performer = int(stem[9:12])
        replication = int(stem[13:16])
        action = int(stem[17:20])
        return {
            "setup": setup, "camera": camera,
            "performer": performer, "replication": replication,
            "action": action,
        }
    except (ValueError, IndexError):
        return None


def normalize_pose_sequence(seq, smooth_window=3):
    """
    Normalize a pose sequence (frames, 75) — 25 joints × 3 coords (x, y, z).

    Fix 4: Three normalization steps:
    1. Remove global motion: subtract mean joint position per frame
    2. Scale normalization: divide by torso length (shoulder-to-hip distance)
    3. Smooth trajectories: moving average filter
    """
    frames = seq.copy()
    n_frames = frames.shape[0]
    data = frames.reshape(n_frames, 25, 3)  # (F, 25, 3)

    # 1. Remove global motion shift (subtract per-frame mean position)
    frame_means = data.mean(axis=1, keepdims=True)  # (F, 1, 3)
    data = data - frame_means

    # 2. Scale normalization (divide by torso length)
    # Joint indices (NTU RGB+D):
    #   20 = spine_shoulder, 0 = spine_base
    #   torso = distance from spine_base to spine_shoulder
    spine_shoulder = data[:, 20, :]  # (F, 3)
    spine_base = data[:, 0, :]       # (F, 3)
    torso_lengths = np.sqrt(((spine_shoulder - spine_base) ** 2).sum(axis=1))  # (F,)
    torso_lengths = np.maximum(torso_lengths, 1e-6)  # avoid division by zero
    data = data / torso_lengths[:, np.newaxis, np.newaxis]

    # 3. Smooth trajectories (moving average)
    if smooth_window > 1 and n_frames >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        for j in range(25):
            for c in range(3):
                data[:, j, c] = np.convolve(data[:, j, c], kernel, mode='same')

    return data.reshape(n_frames, -1)  # back to (F, 75)


def convert_ntu(skeleton_dir, output_dir, normal_actions=None):
    """
    Convert NTU RGB+D skeleton files to .npy format.

    Saves:
        - train_normal.npy  — skeleton sequences for normal actions
        - test_all.npy      — all sequences
        - test_labels.npy   — action labels for each test sample
        - per_action/*.npy  — one file per action class
    """
    if normal_actions is None:
        normal_actions = [47]  # walking as default normal

    ensure_dirs(output_dir, os.path.join(output_dir, "per_action"))

    skeleton_files = sorted(Path(skeleton_dir).glob("*.skeleton"))
    if len(skeleton_files) == 0:
        sys.exit(f"ERROR: No .skeleton files found in {skeleton_dir}")

    print(f"Found {len(skeleton_files)} skeleton files")

    # group by action
    action_data = defaultdict(list)
    action_counts = defaultdict(int)

    for sf in skeleton_files:
        meta = parse_filename(sf)
        if meta is None:
            continue

        action = meta["action"]
        frames = parse_skeleton_file(str(sf))

        if frames is not None and frames.shape[0] > 0:
            # flatten to (frames, 75) — 25 joints × 3 coords
            flat = frames.reshape(frames.shape[0], -1)
            action_data[action].append(flat)
            action_counts[action] += 1

    print(f"\nExtracted {sum(action_counts.values())} sequences from {len(action_counts)} actions:\n")

    for act_id in sorted(action_counts.keys()):
        act_name = ACTION_LABELS.get(act_id, f"action_{act_id}").replace("/", "_").replace(" ", "_")
        count = action_counts[act_id]
        tag = "normal " if act_id in normal_actions else "ANOMALY"
        total_frames = sum(d.shape[0] for d in action_data[act_id])
        print(f"  {tag} A{act_id:02d} {act_name:35s}: {count:4d} seqs, {total_frames:6d} frames")

    # save per-action files
    print("\nSaving per-action files...")
    for act_id, sequences in action_data.items():
        act_name = ACTION_LABELS.get(act_id, f"action_{act_id}").replace("/", "_").replace(" ", "_")
        # pad all sequences to same length (max frames in this action)
        max_len = max(s.shape[0] for s in sequences)
        padded = []
        for s in sequences:
            pad = np.zeros((max_len - s.shape[0], s.shape[1]), dtype=np.float32)
            padded.append(np.concatenate([s, pad], axis=0))
        arr = np.array(padded)

        path = os.path.join(output_dir, "per_action", f"A{act_id:02d}_{act_name}.npy")
        np.save(path, arr)

    # save training data (normal actions only)
    normal_seqs = []
    for act_id in normal_actions:
        if act_id in action_data:
            for seq in action_data[act_id]:
                normal_seqs.append(seq)

    if len(normal_seqs) == 0:
        print("  WARNING: No normal action data found!")
    else:
        # for training, use windows of fixed length
        window_len = config.WINDOW_LENGTH
        windows = []
        for seq in normal_seqs:
            # normalize the sequence
            seq = normalize_pose_sequence(seq)
            if seq.shape[0] >= window_len:
                for start in range(0, seq.shape[0] - window_len + 1, config.WINDOW_STRIDE):
                    windows.append(seq[start:start + window_len])

        if len(windows) > 0:
            train_normal = np.array(windows, dtype=np.float32)
            train_path = os.path.join(output_dir, "train_normal.npy")
            np.save(train_path, train_normal)
            print(f"\n  Train normal: {train_normal.shape} → {train_path}")

    # save test data (all actions) — normalized
    all_seqs = []
    all_labels = []
    for act_id in sorted(action_data.keys()):
        for seq in action_data[act_id]:
            seq = normalize_pose_sequence(seq)
            if seq.shape[0] >= window_len:
                all_seqs.append(seq[:window_len])
                all_labels.append(act_id)

    if len(all_seqs) > 0:
        test_all = np.array(all_seqs, dtype=np.float32)
        test_labels = np.array(all_labels, dtype=np.int32)

        test_path = os.path.join(output_dir, "test_all.npy")
        labels_path = os.path.join(output_dir, "test_labels.npy")

        np.save(test_path, test_all)
        np.save(labels_path, test_labels)
        print(f"  Test all: {test_all.shape} → {test_path}")
        print(f"  Test labels: {test_labels.shape} → {labels_path}")

    # save metadata
    info = {
        "num_joints": 25,
        "joint_dim": 3,
        "feature_dim": 75,
        "normal_actions": normal_actions,
        "action_labels": ACTION_LABELS,
        "action_counts": dict(action_counts),
    }
    info_path = os.path.join(output_dir, "dataset_info.npy")
    np.save(info_path, info)
    print(f"  Dataset info → {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert NTU RGB+D Skeletons to .npy")
    parser.add_argument("--input", default=os.path.join(config.BASE_DIR, "data", "nturgb+d_skeletons"),
                        help="Directory with .skeleton files")
    parser.add_argument("--output", default=config.POSE_DIR,
                        help="Output directory for .npy files")
    parser.add_argument("--normal", nargs="+", type=int, default=[47],
                        help="Action IDs for normal behavior (default: 47=walking). "
                             "Examples: 47=walking, 48=running, 9=stand_up")
    args = parser.parse_args()

    print("=" * 60)
    print("  NTU RGB+D Skeleton → .npy Converter")
    print("=" * 60)

    convert_ntu(args.input, args.output, args.normal)

    print("\nDone!")


if __name__ == "__main__":
    main()
