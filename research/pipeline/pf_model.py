"""Pulp Fiction topology in PyTorch: 5× BiLSTM + GRU multiply fusion.

Widths are scaled from the TPU notebook (768-512-384-256-128) so a 4060 8GB
can train. Topology (layer count, multiply+BN skip, concat head, SELU) matches.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Original notebook units. 4060 uses a scale of these.
PF_FULL = (768, 512, 384, 256, 128)
PF_512 = (512, 384, 256, 192, 96)
PF_384 = (384, 256, 192, 128, 64)
PF_256 = (256, 192, 128, 96, 64)
PF_MED = (192, 128, 96, 64, 48)
PF_SMALL = (128, 96, 64, 48, 32)
PF_TINY = (64, 48, 32, 24, 16)
WIDTH_LADDER = [PF_FULL, PF_512, PF_384, PF_256, PF_MED, PF_SMALL]


class PulpFictionNet(nn.Module):
    def __init__(self, n_feat: int, lstm_units: tuple[int, ...] = PF_SMALL, checkpoint: bool = False):
        super().__init__()
        self.checkpoint = checkpoint
        if len(lstm_units) != 5:
            raise ValueError("need 5 LSTM widths (matches notebook x1..x5)")
        l1, l2, l3, l4, l5 = lstm_units
        g_last = max(l5 // 2, 8)
        self.x1 = nn.LSTM(n_feat, l1, batch_first=True, bidirectional=True)
        self.x2 = nn.LSTM(2 * l1, l2, batch_first=True, bidirectional=True)
        self.x3 = nn.LSTM(2 * l2, l3, batch_first=True, bidirectional=True)
        self.x4 = nn.LSTM(2 * l3, l4, batch_first=True, bidirectional=True)
        self.x5 = nn.LSTM(2 * l4, l5, batch_first=True, bidirectional=True)
        # GRU branch: hidden sizes match the LSTM they multiply with
        self.z2 = nn.GRU(2 * l2, l3, batch_first=True, bidirectional=True)
        self.z3 = nn.GRU(2 * l3, l4, batch_first=True, bidirectional=True)
        self.z4 = nn.GRU(2 * l4, l5, batch_first=True, bidirectional=True)
        self.z5 = nn.GRU(2 * l5, g_last, batch_first=True, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(2 * l3)
        self.bn4 = nn.BatchNorm1d(2 * l4)
        self.bn5 = nn.BatchNorm1d(2 * l5)
        cat = 2 * l5 + 2 * l3 + 2 * l4 + 2 * l5 + 2 * g_last
        self.head = nn.Sequential(
            nn.Linear(cat, 128),
            nn.SELU(),
            nn.Linear(128, 1),
        )
        self.lstm_units = lstm_units

    @staticmethod
    def _bn(bn: nn.BatchNorm1d, h: torch.Tensor) -> torch.Tensor:
        return bn(h.transpose(1, 2)).transpose(1, 2)

    def _run(self, layer, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(lambda t: layer(t)[0], x, use_reentrant=False)
        return layer(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._run(self.x1, x)
        x2 = self._run(self.x2, x1)
        x3 = self._run(self.x3, x2)
        x4 = self._run(self.x4, x3)
        x5 = self._run(self.x5, x4)
        z2 = self._run(self.z2, x2)
        z31 = self._bn(self.bn3, x3 * z2)
        z3 = self._run(self.z3, z31)
        z41 = self._bn(self.bn4, x4 * z3)
        z4 = self._run(self.z4, z41)
        z51 = self._bn(self.bn5, x5 * z4)
        z5 = self._run(self.z5, z51)
        h = torch.cat([x5, z2, z3, z4, z5], dim=-1)
        return self.head(h).squeeze(-1)
