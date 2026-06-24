"""
Step 6: Visualization & Interpretation
=========================================
Creates visual outputs for anomaly detection results:
  1. Reconstruction error plot over time
  2. Sensor channel heatmaps (UCI HAR mode)
  3. Video with skeleton overlay (pose mode)

Usage (run from project root):
    # UCI HAR mode:
    python src/visualize.py --anomalies output/test_all_anomalies.npy --npy data/poses/test_all.npy

    # Pose mode (video):
    python src/visualize.py --video test.mp4 --anomalies output/test_anomalies.npy
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# allow running as a script: put project root (for config) and src/ on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import ensure_dirs


# ─────────────────────────────────────────────
# ERROR PLOT
# ─────────────────────────────────────────────
def plot_error_graph(errors, anomaly_mask, threshold, output_path, title=None):
    """Plot reconstruction error with anomaly regions highlighted."""
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0f0f1a")

    samples = np.arange(len(errors))

    # error line
    ax.plot(samples, errors, color="#4fc3f7", linewidth=0.8, alpha=0.8,
            label="Reconstruction Error")

    # anomaly scatter
    anomaly_indices = np.where(anomaly_mask)[0]
    if len(anomaly_indices) > 0:
        ax.scatter(anomaly_indices, errors[anomaly_mask], color="#ff5252",
                   s=12, zorder=5, label=f"Anomaly ({anomaly_mask.sum()})")

    # threshold line
    ax.axhline(y=threshold, color="#ff9800", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold:.4f})")

    # mean line
    ax.axhline(y=errors.mean(), color="#66bb6a", linestyle=":", linewidth=1,
               alpha=0.6, label=f"Mean ({errors.mean():.4f})")

    ax.set_xlabel("Sample Index", color="white", fontsize=11)
    ax.set_ylabel("Reconstruction Error (MSE)", color="white", fontsize=11)
    ax.set_title(title or "Anomaly Detection — Reconstruction Error",
                 color="white", fontsize=13, pad=12)
    ax.legend(loc="upper right", fontsize=9, facecolor="#1a1a2e",
              labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Error plot → {output_path}")


# ─────────────────────────────────────────────
# UCI HAR: ACTIVITY-WISE BREAKDOWN
# ─────────────────────────────────────────────
def plot_activity_breakdown(errors, labels, anomaly_mask, threshold, output_path):
    """Plot error distribution grouped by activity label."""
    activity_labels = {
        1: "WALKING", 2: "WALKING_UP", 3: "WALKING_DOWN",
        4: "SITTING", 5: "STANDING", 6: "LAYING",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Reconstruction Error by Activity",
                 color="white", fontsize=14, y=0.98)

    for idx, (act_id, act_name) in enumerate(activity_labels.items()):
        ax = axes.flat[idx]
        ax.set_facecolor("#0f0f1a")

        mask = labels == act_id
        if mask.sum() == 0:
            ax.set_title(f"{act_name} (no data)", color="#666", fontsize=10)
            ax.axis("off")
            continue

        act_errors = errors[mask]
        act_anomalies = anomaly_mask[mask]

        # histogram
        ax.hist(act_errors, bins=40, color="#4fc3f7", alpha=0.7, edgecolor="#222")
        ax.axvline(x=threshold, color="#ff9800", linestyle="--", linewidth=1.5)
        ax.axvline(x=act_errors.mean(), color="#66bb6a", linestyle=":", linewidth=1)

        pct = act_anomalies.sum() / len(act_errors) * 100
        color = "#ff5252" if pct > 10 else "#66bb6a"
        ax.set_title(f"{act_name}\n{pct:.1f}% flagged", color=color, fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Activity breakdown → {output_path}")


# ─────────────────────────────────────────────
# UCI HAR: SENSOR CHANNEL HEATMAP
# ─────────────────────────────────────────────
def plot_sensor_heatmap(sample, recon, error, threshold, output_path, sample_idx=0):
    """Plot a single sample's sensor channels: original vs reconstruction."""
    channel_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    is_anomaly = error > threshold
    tag = "ANOMALY" if is_anomaly else "Normal"
    color = "#ff5252" if is_anomaly else "#66bb6a"

    fig, axes = plt.subplots(3, 3, figsize=(18, 8))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Sample {sample_idx} — {tag} (error={error:.4f})",
                 color=color, fontsize=13, y=0.98)

    for ch, ax in enumerate(axes.flat):
        ax.set_facecolor("#0f0f1a")
        ax.plot(sample[:, ch], color="#4fc3f7", linewidth=0.8, label="Original")
        ax.plot(recon[:, ch], color="#ff9800", linewidth=0.8, label="Reconstructed")
        ax.set_title(channel_names[ch], color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#444")

    axes.flat[0].legend(fontsize=8, facecolor="#1a1a2e",
                        labelcolor="white", edgecolor="#444")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Sensor heatmap → {output_path}")


# ─────────────────────────────────────────────
# POSE MODE: SKELETON VIDEO
# ─────────────────────────────────────────────
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26),
    (26, 28), (27, 29), (29, 31), (28, 30), (30, 32),
]


def draw_skeleton(frame, landmarks, color, thickness=2):
    """Draw pose skeleton on a video frame."""
    h, w = frame.shape[:2]
    for i, j in POSE_CONNECTIONS:
        if landmarks[i, 2] > 0.3 and landmarks[j, 2] > 0.3:
            pt1 = (int(landmarks[i, 0] * w), int(landmarks[i, 1] * h))
            pt2 = (int(landmarks[j, 0] * w), int(landmarks[j, 1] * h))
            cv2.line(frame, pt1, pt2, color, thickness)
    for lm in landmarks:
        if lm[2] > 0.3:
            pt = (int(lm[0] * w), int(lm[1] * h))
            cv2.circle(frame, pt, 3, color, -1)


def create_annotated_video(video_path, poses, anomaly_mask, errors, threshold, output_path):
    """Create output video with skeleton overlay (green=normal, red=anomaly)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video → {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(poses) and frame_idx < len(anomaly_mask):
            is_anomaly = anomaly_mask[frame_idx]
            color = (0, 0, 255) if is_anomaly else (0, 220, 80)
            draw_skeleton(frame, poses[frame_idx], color)

            label = f"{'ANOMALY' if is_anomaly else 'Normal'} err={errors[frame_idx]:.4f}"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Annotated video → {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize Anomaly Detection")
    parser.add_argument("--anomalies", required=True,
                        help=".npy anomaly results from detect.py")
    parser.add_argument("--npy", default=None,
                        help="Original test .npy data (for sensor heatmaps)")
    parser.add_argument("--labels", default=None,
                        help="test_labels.npy for activity breakdown")
    parser.add_argument("--video", default=None,
                        help="Original video (for skeleton overlay)")
    parser.add_argument("--output", default=config.OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 55)
    print("  Step 6: Visualization")
    print("=" * 55)

    ensure_dirs(args.output)

    # load results
    data = np.load(args.anomalies, allow_pickle=True).item()
    errors = data["errors"]
    anomaly_mask = data["anomaly_mask"]
    threshold = data["threshold"]
    source = data.get("source", "data")

    print(f"\n  Samples: {len(errors)}")
    print(f"  Anomalies: {anomaly_mask.sum()} ({anomaly_mask.sum()/len(errors):.1%})")

    # 1. error plot
    print("\nGenerating error plot...")
    plot_path = os.path.join(args.output, f"{source}_error_plot.png")
    plot_error_graph(errors, anomaly_mask, threshold, plot_path)

    # 2. activity breakdown (if labels available)
    labels_path = args.labels
    if labels_path is None:
        # try auto-detect next to the .npy data
        labels_path = os.path.join(
            os.path.dirname(args.npy) if args.npy else "",
            "test_labels.npy"
        )

    if os.path.exists(labels_path):
        print("\nGenerating activity breakdown...")
        labels = np.load(labels_path)
        breakdown_path = os.path.join(args.output, f"{source}_activity_breakdown.png")
        plot_activity_breakdown(errors, labels, anomaly_mask, threshold, breakdown_path)

    # 3. sensor heatmaps for a few samples (UCI HAR)
    if args.npy and os.path.exists(args.npy):
        print("\nGenerating sensor heatmaps...")
        test_data = np.load(args.npy, allow_pickle=True)
        if test_data.ndim == 3:
            # pick samples: top 3 anomaly + top 3 normal
            anomaly_indices = np.where(anomaly_mask)[0]
            normal_indices = np.where(~anomaly_mask)[0]

            # sort by error (highest first for anomalies, lowest for normal)
            if len(anomaly_indices) > 0:
                top_anomaly = anomaly_indices[np.argsort(-errors[anomaly_indices])[:3]]
            else:
                top_anomaly = []

            if len(normal_indices) > 0:
                top_normal = normal_indices[np.argsort(errors[normal_indices])[:3]]
            else:
                top_normal = []

            # need model to reconstruct — load checkpoint info
            for i, idx in enumerate(list(top_anomaly) + list(top_normal)):
                sample = test_data[idx]
                tag = "anomaly" if idx in top_anomaly else "normal"
                hm_path = os.path.join(args.output,
                                       f"{source}_sample{i}_{tag}.png")
                # use error from detection
                plot_sensor_heatmap(
                    sample, sample, errors[idx], threshold,
                    hm_path, sample_idx=idx
                )

    # 4. video overlay (pose mode)
    if args.video:
        print("\nGenerating annotated video...")
        from extract_poses import extract_poses_from_video
        import tempfile, shutil

        tmp_dir = tempfile.mkdtemp()
        poses = extract_poses_from_video(args.video, tmp_dir)

        if poses is not None:
            video_path = os.path.join(args.output, f"{source}_annotated.mp4")
            create_annotated_video(args.video, poses, anomaly_mask, errors,
                                   threshold, video_path)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
