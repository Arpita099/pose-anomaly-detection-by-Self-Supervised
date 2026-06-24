"""
Step 5: Anomaly Detection
=============================
Runs the trained model and flags anomalous samples
based on reconstruction error.

Supports:
  - UCI HAR mode: directly test .npy sensor data
  - Pose mode: extract poses from video, then detect

Usage (run from project root):
    # UCI HAR mode (recommended):
    python src/detect.py --npy data/poses/test_all.npy --checkpoint checkpoints/best_model.pt

    # Pose mode (video):
    python src/detect.py --video test.mp4 --checkpoint checkpoints/best_model.pt

    python src/detect.py --npy data/poses/test_all.npy --k 2.5
"""

import argparse
import os
import sys
import numpy as np
import torch

# allow running as a script: put project root (for config) and src/ on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import ensure_dirs
from model import PoseTransformer


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ckpt["config"]
    model = PoseTransformer(
        feature_dim=cfg["feature_dim"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")
    return model


def compute_sample_errors(model, data, device, batch_size=64):
    """
    Compute per-sample reconstruction error.

    Args:
        model: trained PoseTransformer
        data: np.ndarray (num_samples, seq_len, feature_dim)
        device: torch device

    Returns:
        errors: np.ndarray (num_samples,)
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            tensor = torch.from_numpy(batch).to(device)

            recon = model(tensor)
            mse = ((recon.cpu().numpy() - batch) ** 2).mean(axis=(1, 2))  # per sample
            errors.extend(mse)

    return np.array(errors)


def detect_anomalies(errors, mean_err, std_err, k=None):
    """
    Flag samples where reconstruction error exceeds threshold.
    """
    k = k or config.ANOMALY_K
    threshold = mean_err + k * std_err
    anomaly_mask = errors > threshold
    return anomaly_mask, threshold


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("--npy", default=None,
                        help="Path to .npy test data (UCI HAR or pose)")
    parser.add_argument("--video", default=None,
                        help="Path to test video (pose mode only)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--stats", default=None,
                        help="Path to error_stats.npy (auto-detected if None)")
    parser.add_argument("--k", type=float, default=config.ANOMALY_K,
                        help="Anomaly threshold: mean + k * std")
    parser.add_argument("--output", default=config.OUTPUT_DIR)
    args = parser.parse_args()

    if args.npy is None and args.video is None:
        sys.exit("ERROR: Provide --npy or --video")

    print("=" * 55)
    print("  Step 5: Anomaly Detection")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device)

    # load error stats
    if args.stats is None:
        args.stats = os.path.join(os.path.dirname(args.checkpoint), "error_stats.npy")

    stats = np.load(args.stats, allow_pickle=True).item()
    mean_err = stats["mean"]
    std_err = stats["std"]
    print(f"  Training stats: mean={mean_err:.6f}, std={std_err:.6f}")

    # load test data
    if args.npy:
        print(f"\nLoading test data from {args.npy}...")
        test_data = np.load(args.npy, allow_pickle=True)
        # handle dict format (from test_all.npy with labels)
        if test_data.ndim == 1 and test_data.dtype == object:
            # it's a dict-like .npy
            test_data = np.load(args.npy)
        source_name = os.path.splitext(os.path.basename(args.npy))[0]

    elif args.video:
        print(f"\nExtracting poses from video...")
        from extract_poses import extract_poses_from_video
        import tempfile, shutil

        tmp_dir = tempfile.mkdtemp()
        poses = extract_poses_from_video(args.video, tmp_dir)
        if poses is None:
            sys.exit("ERROR: No poses extracted from video")
        test_data = poses.reshape(poses.shape[0], config.WINDOW_LENGTH, -1)
        # pad or reshape for consistent shape
        if test_data.shape[1] != config.WINDOW_LENGTH:
            # reshape continuous frames into windows
            num_frames = poses.shape[0]
            feature_dim = poses.shape[1] * poses.shape[2]
            flat = poses.reshape(num_frames, -1)
            windows = []
            for start in range(0, num_frames - config.WINDOW_LENGTH + 1, config.WINDOW_STRIDE):
                windows.append(flat[start:start + config.WINDOW_LENGTH])
            test_data = np.array(windows)

        source_name = os.path.splitext(os.path.basename(args.video))[0]
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ensure 3D shape: (samples, seq_len, features)
    if test_data.ndim == 2:
        test_data = test_data.reshape(test_data.shape[0], -1, test_data.shape[1] // 1)
    test_data = test_data.astype(np.float32)

    print(f"  Test data shape: {test_data.shape}")

    # compute reconstruction errors
    print("\nComputing reconstruction errors...")
    errors = compute_sample_errors(model, test_data, device)

    # detect anomalies
    anomaly_mask, threshold = detect_anomalies(errors, mean_err, std_err, args.k)
    num_anomalies = anomaly_mask.sum()

    print(f"\n  Threshold: {threshold:.6f} (mean + {args.k} * std)")
    print(f"  Anomalous samples: {num_anomalies}/{len(errors)} ({num_anomalies/len(errors):.1%})")

    # save results
    ensure_dirs(args.output)
    output_path = os.path.join(args.output, f"{source_name}_anomalies.npy")
    np.save(output_path, {
        "errors": errors,
        "anomaly_mask": anomaly_mask,
        "threshold": threshold,
        "mean_err": mean_err,
        "std_err": std_err,
        "k": args.k,
        "source": source_name,
    })
    print(f"  Saved to: {output_path}")

    # load labels if available (UCI HAR)
    labels_path = os.path.join(os.path.dirname(args.npy) if args.npy else "", "test_labels.npy")
    if os.path.exists(labels_path):
        labels = np.load(labels_path)

        # detect which dataset: NTU or UCI HAR
        activity_labels = {
            1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
            4: "SITTING", 5: "STANDING", 6: "LAYING",
        }
        normal_acts = [1, 2, 3, 5]  # default: locomotion group
        primary_anomaly_acts = [6]  # LAYING = primary anomaly

        info_path = os.path.join(os.path.dirname(args.npy) if args.npy else "", "dataset_info.npy")
        if os.path.exists(info_path):
            info = np.load(info_path, allow_pickle=True).item()
            # check if NTU dataset
            if "action_labels" in info:
                activity_labels = info["action_labels"]
            normal_acts = info.get("normal_activities", normal_acts)
            primary_anomaly_acts = info.get("primary_anomaly_activities", primary_anomaly_acts)

        print(f"\n  Breakdown by activity:")
        print(f"  {'ID':>4s}  {'Activity':30s}  {'Role':10s}  "
              f"{'Mean Err':>9s}  {'Std Err':>9s}  {'Flagged':>10s}  {'Rate':>6s}  {'Status':>8s}")
        print(f"  {'─'*4}  {'─'*30}  {'─'*10}  "
              f"{'─'*9}  {'─'*9}  {'─'*10}  {'─'*6}  {'─'*8}")

        tp, fp, fn, tn = 0, 0, 0, 0
        for act_id in sorted(activity_labels.keys()):
            act_name = activity_labels[act_id]
            mask = labels == act_id
            if mask.sum() == 0:
                continue
            act_errors = errors[mask]
            act_anomalies = anomaly_mask[mask].sum()
            act_total = mask.sum()
            act_rate = act_anomalies / act_total * 100

            if act_id in normal_acts:
                role = "NORMAL"
                # for normal activities: flagged = FP, not flagged = TN
                fp += int(act_anomalies)
                tn += int(act_total - act_anomalies)
                status = "✓ OK" if act_rate < 5 else "⚠ HIGH"
            elif act_id in primary_anomaly_acts:
                role = "ANOMALY"
                # for primary anomaly: flagged = TP, not flagged = FN
                tp += int(act_anomalies)
                fn += int(act_total - act_anomalies)
                status = "✓ DETECTED" if act_rate > 50 else "⚠ MISSED"
            else:
                role = "AMBIGUOUS"
                # ambiguous activities: not counted in precision/recall
                status = "INFO"

            print(f"  A{act_id:02d}  {act_name:30s}  {role:10s}  "
                  f"{act_errors.mean():9.4f}  {act_errors.std():9.4f}  "
                  f"{act_anomalies:4d}/{act_total:<5d}  {act_rate:5.1f}%  {status}")

        # compute precision/recall if we have anomaly ground truth
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"\n  ── Evaluation (primary anomaly: {primary_anomaly_acts}) ──")
            print(f"  Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1: {f1:.3f}")
            print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        else:
            print(f"\n  (No primary anomaly samples detected — precision/recall skipped)")

        # false positive rate on normal data
        if fp + tn > 0:
            fpr = fp / (fp + tn)
            print(f"  False positive rate on normal: {fpr:.3f} ({fp}/{fp+tn})")

    print("\nDone!")


if __name__ == "__main__":
    main()
