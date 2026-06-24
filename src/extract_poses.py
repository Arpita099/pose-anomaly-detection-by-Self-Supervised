"""
Step 1: Pose Extraction from Videos
=====================================
Extracts MediaPipe Pose keypoints from video files and saves as .npy arrays.

Usage (run from project root):
    python src/extract_poses.py --input data/videos/
    python src/extract_poses.py --input data/videos/ --output data/poses/
"""

import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# allow running as a script: put project root (for config) and src/ on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import find_videos, ensure_dirs


def extract_poses_from_video(video_path, output_dir):
    """
    Extract pose keypoints from a single video using MediaPipe.

    Args:
        video_path: path to video file
        output_dir: directory to save .npy output

    Returns:
        poses: np.ndarray of shape (num_frames, 33, 3) or None if no poses found
    """
    import mediapipe as mp

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video → {video_path}")
        return None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_poses = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    ) as pose:

        with tqdm(total=total_frames, desc=os.path.basename(video_path), leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(frame_rgb)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    frame_kpts = np.array(
                        [[lm.x, lm.y, lm.visibility] for lm in landmarks],
                        dtype=np.float32
                    )
                    all_poses.append(frame_kpts)
                else:
                    # no pose detected — fill with zeros
                    all_poses.append(np.zeros((config.NUM_KEYPOINTS, 3), dtype=np.float32))

                pbar.update(1)

    cap.release()

    if len(all_poses) == 0:
        print(f"  WARNING: No frames extracted from → {video_path}")
        return None

    poses = np.stack(all_poses, axis=0)  # (num_frames, 33, 3)

    # normalize x, y to pixel coords → [0, 1]
    poses[:, :, 0] /= width
    poses[:, :, 1] /= height

    # save
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{stem}.npy")
    np.save(out_path, poses)

    valid_frames = (poses[:, :, 2] > 0).any(axis=1).sum()
    print(f"  {stem}: {poses.shape[0]} frames, {valid_frames} with pose → {out_path}")

    return poses


def main():
    parser = argparse.ArgumentParser(description="Extract pose keypoints from videos")
    parser.add_argument("--input", default=config.VIDEO_DIR,
                        help="Directory containing video files")
    parser.add_argument("--output", default=config.POSE_DIR,
                        help="Directory to save .npy pose files")
    args = parser.parse_args()

    print("=" * 50)
    print("  Step 1: Pose Extraction (MediaPipe)")
    print("=" * 50)

    ensure_dirs(args.output)

    videos = find_videos(args.input)
    if len(videos) == 0:
        sys.exit(f"ERROR: No video files found in {args.input}")

    print(f"Found {len(videos)} video(s)\n")

    for video_path in videos:
        extract_poses_from_video(str(video_path), args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
