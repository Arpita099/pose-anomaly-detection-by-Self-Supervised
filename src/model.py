"""
Step 3: Transformer Model for Pose Sequence Reconstruction
============================================================
Self-supervised Transformer encoder that learns to reconstruct
masked pose coordinates.

Architecture:
    Input (masked poses) → Linear projection → Positional Encoding
        → Transformer Encoder → Reconstruction Head → Output (reconstructed poses)
"""

import math
import os
import sys
import torch
import torch.nn as nn

# allow running as a script: put project root (for config) on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal position."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class PoseTransformer(nn.Module):
    """
    Transformer encoder for pose sequence reconstruction.

    Learns to reconstruct masked joint coordinates from
    the visible (unmasked) ones.
    """

    def __init__(
        self,
        feature_dim=None,
        d_model=None,
        nhead=None,
        num_layers=None,
        dim_feedforward=None,
        dropout=None,
    ):
        super().__init__()

        feature_dim = feature_dim or config.NUM_KEYPOINTS * config.KEYPOINT_DIM
        d_model = d_model or config.D_MODEL
        nhead = nhead or config.NHEAD
        num_layers = num_layers or config.NUM_LAYERS
        dim_feedforward = dim_feedforward or config.DIM_FEEDFORWARD
        dropout = dropout or config.DROPOUT

        self.d_model = d_model
        self.feature_dim = feature_dim

        # input projection: (joints * 3) → d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # reconstruction head: d_model → (joints * 3)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, feature_dim) — masked pose sequence

        Returns:
            reconstruction: (batch, seq_len, feature_dim) — reconstructed poses
        """
        # project to d_model
        h = self.input_proj(x)                    # (B, T, d_model)
        h = self.pos_encoding(h)                  # add temporal position
        h = self.transformer_encoder(h)           # (B, T, d_model)
        reconstruction = self.recon_head(h)       # (B, T, feature_dim)

        return reconstruction


class MaskedReconstructionLoss(nn.Module):
    """
    MSE loss computed only on masked positions.

    Args:
        reduction: 'mean' or 'sum'
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction

    def forward(self, prediction, target, mask):
        """
        Args:
            prediction: (batch, seq_len, feature_dim)
            target:     (batch, seq_len, feature_dim)
            mask:       (batch, seq_len, feature_dim) — 1=masked, 0=kept

        Returns:
            scalar loss
        """
        per_element = self.mse(prediction, target)  # (B, T, D)

        # only compute loss on masked positions
        masked_loss = per_element * mask

        if self.reduction == "mean":
            return masked_loss.sum() / mask.sum().clamp(min=1)
        else:
            return masked_loss.sum()
