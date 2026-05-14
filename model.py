"""
model.py
========
PyTorch model definitions for ECG arrhythmia classification and MI detection.

Models
------
- ResidualBlock      : 1-D residual conv block with max-pooling (stride-2 downsampling)
- ECGResNet          : Full network for 5-class arrhythmia classification
- ECGResNetTransfer  : Transfer-learning wrapper for 2-class MI detection
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.relu(out + x)
        return self.pool(out)


class ECGResNet(nn.Module):
    def __init__(self, num_classes: int = 5, n_res_blocks: int = 5):
        super().__init__()
        self.first  = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(n_res_blocks)])
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc1    = nn.Linear(32, 32)
        self.fc2    = nn.Linear(32, 32)
        self.relu   = nn.ReLU()
        self.out    = nn.Linear(32, num_classes)

    def conv_representation(self, x):
        x = self.first(x)
        x = self.blocks(x)
        return self.gap(x).squeeze(-1)

    def forward(self, x):
        x = self.conv_representation(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


class ECGResNetTransfer(nn.Module):
    """Frozen backbone + new FC head for MI detection (transfer learning)."""

    def __init__(self, pretrained_model: ECGResNet, num_classes: int = 2):
        super().__init__()
        self.first  = pretrained_model.first
        self.blocks = pretrained_model.blocks
        self.gap    = pretrained_model.gap
        for p in self.first.parameters():  p.requires_grad = False
        for p in self.blocks.parameters(): p.requires_grad = False
        self.fc1  = nn.Linear(32, 32)
        self.fc2  = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.out  = nn.Linear(32, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.first(x)
            x = self.blocks(x)
            x = self.gap(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)
