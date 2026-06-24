# Self-Supervised Transformer for Human Pose/Sensor Sequence Reconstruction and Anomaly Detection

A self-supervised Transformer pipeline that learns normal motion patterns from time-series data (pose keypoints or sensor signals) and identifies unusual movement through reconstruction error — **without requiring labeled data**.

## Pipeline

```
Dataset (Video / UCI HAR / Pose Keypoints)
            │
            ▼
  Data Conversion (scripts/convert_ucihar.py / src/extract_poses.py)
            │
            ▼
  Sequence Windowing + Random Masking (src/dataset.py)
            │
            ▼
  Transformer Encoder Network (src/model.py)
            │
            ▼
  Reconstruction Head — Predict Masked Values
            │
            ▼
  Self-Supervised Training on Normal Data (src/train.py)
            │
            ▼
  Anomaly Detection via Reconstruction Error (src/detect.py)
            │
            ▼
  Visualization & Interpretation (src/visualize.py)
```

## Supported Data Modes

| Mode | Input | Features |
|------|-------|----------|
| **UCI HAR** | Smartphone sensor signals (`.txt`) | 9 channels × 128 timesteps |
| **Pose** | Video files (`.mp4`) | 33 MediaPipe keypoints × 3 |

## Setup

### Install dependencies

```bash
conda create -n pose_anomaly python=3.10
conda activate pose_anomaly
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### With UCI HAR Dataset

All commands run from the **project root**.

```bash
# Step 1: Convert UCI HAR to .npy format (normal = WALKING + UP/DOWN stairs + STANDING)
python scripts/convert_ucihar.py --input "path/to/UCI HAR Dataset/"

# Step 2: Train on normal data
python src/train.py --data data/poses/ --epochs 50

# Step 3: Detect anomalies on test data
python src/detect.py --npy data/poses/test_all.npy --checkpoint checkpoints/best_model.pt

# Step 4: Visualize results
python src/visualize.py --anomalies output/test_all_anomalies.npy --npy data/poses/test_all.npy
```

### With Video Data

```bash
# Step 1: Extract poses from videos
python src/extract_poses.py --input data/videos/

# Step 2-4: Same as above (set DATA_MODE="pose" in config.py)
```

## Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_MODE` | `"ucihar"` | `"ucihar"` or `"pose"` |
| `D_MODEL` | 128 | Transformer embedding dimension |
| `NHEAD` | 8 | Attention heads |
| `NUM_LAYERS` | 4 | Transformer encoder layers |
| `MASK_RATIO` | 0.20 | Fraction of coordinates masked |
| `EPOCHS` | 50 | Training epochs |
| `ANOMALY_K` | 4.0 | Threshold = mean + K × std |

### "Normal" vs "Anomaly" definition

- **Normal group** (training data): WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, STANDING
  (activities 1, 2, 3, 5). Covers natural locomotion variation.
- **Primary anomaly**: LAYING (activity 6) — radically different posture.
- **Ambiguous**: SITTING (activity 4) — reported but not forced as anomaly.

Override the normal group at conversion time:
```bash
python scripts/convert_ucihar.py --input "path/to/UCI HAR Dataset/" --normal 1 2 3 5
```

## Results (UCI HAR)

Trained on the normal group (4,659 samples) with per-sequence normalization
(mean-subtract, std-scale, trajectory smoothing) and `ANOMALY_K=4`:

| Activity | Role | Mean Err | Flagged |
|----------|------|----------|---------|
| WALKING | NORMAL | 0.1834 | 0.0% |
| WALKING_UPSTAIRS | NORMAL | 0.1914 | 0.0% |
| WALKING_DOWNSTAIRS | NORMAL | 0.1847 | 0.0% |
| SITTING | AMBIGUOUS | 0.1982 | 7.3% |
| STANDING | NORMAL | 0.1885 | 0.6% |
| **LAYING** | **ANOMALY** | **0.2062** | **14.5%** |

- **Precision**: 96.3%
- **Recall** (on LAYING): 14.5%
- **False-positive rate on normal**: 0.2%

## Project Structure

```
pose-anomaly-transformer/
│
├── README.md
├── requirements.txt
├── config.py                 # Configuration & hyperparameters
│
├── data/                     # Generated .npy files (gitignored)
│   └── README.md
│
├── src/
│   ├── extract_poses.py      # Video → MediaPipe pose extraction
│   ├── dataset.py            # PyTorch dataset with windowing + masking
│   ├── model.py              # Transformer encoder + reconstruction head
│   ├── train.py              # Self-supervised training loop
│   ├── detect.py             # Anomaly detection + per-activity evaluation
│   ├── visualize.py          # Error plots, activity breakdown, heatmaps
│   └── utils.py              # Shared helpers (incl. sequence normalization)
│
├── scripts/
│   ├── convert_ucihar.py     # UCI HAR dataset converter
│   └── convert_ntu.py        # NTU RGB+D skeleton converter
│
├── docs/                     # Pipeline diagram, report (add yours)
├── checkpoints/              # Saved models (gitignored)
└── output/                   # Detection results (gitignored)
```

## License

MIT
