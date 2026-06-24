"""
Step 4: Self-Supervised Training
==================================
Trains the PoseTransformer on normal pose sequences using
masked reconstruction as the self-supervised objective.

Usage (run from project root):
    python src/train.py --data data/poses/
    python src/train.py --data data/poses/ --epochs 100 --batch_size 64
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# allow running as a script: put project root (for config) and src/ on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import set_seed, ensure_dirs
from dataset import get_dataloaders
from model import PoseTransformer, MaskedReconstructionLoss


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for masked_input, target, mask in dataloader:
        masked_input = masked_input.to(device)
        target = target.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        prediction = model(masked_input)
        loss = criterion(prediction, target, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate. Returns average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for masked_input, target, mask in dataloader:
        masked_input = masked_input.to(device)
        target = target.to(device)
        mask = mask.to(device)

        prediction = model(masked_input)
        loss = criterion(prediction, target, mask)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def compute_reconstruction_stats(model, dataloader, device):
    """
    Compute per-sample reconstruction error statistics on training data.
    Used later for anomaly thresholding.

    Passes UNMASKED inputs through the model (same as detect.py) so that
    the error scale is directly comparable at detection time.

    Returns:
        mean_error, std_error
    """
    model.eval()
    all_errors = []

    for masked_input, target, mask in dataloader:
        # Pass unmasked target through model (same as detection does)
        target = target.to(device)
        prediction = model(target)

        # per-sample MSE (average over timesteps and features — same as detect.py)
        mse = ((prediction - target) ** 2).mean(axis=(1, 2))  # (B,)
        all_errors.append(mse.cpu().numpy())

    all_errors = np.concatenate(all_errors)
    return all_errors.mean(), all_errors.std()


def main():
    parser = argparse.ArgumentParser(description="Train Pose Transformer")
    parser.add_argument("--data", default=config.POSE_DIR,
                        help="Directory with .npy pose files")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--output", default=config.CHECKPOINT_DIR,
                        help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 55)
    print("  Self-Supervised Pose Transformer Training")
    print("=" * 55)

    set_seed(args.seed)
    ensure_dirs(args.output)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # data
    print("\nLoading dataset...")
    train_loader, val_loader = get_dataloaders(args.data, batch_size=args.batch_size)

    # model — auto-detect feature dim from data
    sample_batch = next(iter(train_loader))
    feature_dim = sample_batch[0].shape[-1]
    print(f"  Auto-detected feature_dim: {feature_dim}")
    model = PoseTransformer(feature_dim=feature_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # optimizer + scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = MaskedReconstructionLoss()

    # training loop
    best_val_loss = float("inf")
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "feature_dim": feature_dim,
                    "d_model": config.D_MODEL,
                    "nhead": config.NHEAD,
                    "num_layers": config.NUM_LAYERS,
                    "dim_feedforward": config.DIM_FEEDFORWARD,
                    "dropout": config.DROPOUT,
                },
            }, best_path)
            marker = " *"
        else:
            marker = ""

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.6f}  val={val_loss:.6f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s{marker}")

    # compute reconstruction stats for anomaly detection
    print("\nComputing reconstruction error stats on training data...")
    mean_err, std_err = compute_reconstruction_stats(model, train_loader, device)
    stats_path = os.path.join(args.output, "error_stats.npy")
    np.save(stats_path, {"mean": mean_err, "std": std_err})
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  Std  error: {std_err:.6f}")
    print(f"  Saved to: {stats_path}")

    print(f"\nBest model saved to: {os.path.join(args.output, 'best_model.pt')}")
    print("Done!")


if __name__ == "__main__":
    main()
