import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def cp_to_wdl(cp: float, draw_width: float = 80.0, scale: float = 350.0) -> np.ndarray:
    """
    Convert centipawn eval to approximate WDL probabilities.
    cp > 0 = good for side to move.
    Returns np.array([P(win), P(draw), P(loss)], dtype=float32).
    """
    import math
    value = math.tanh(cp / scale)
    p_draw = math.exp(-abs(cp) / draw_width)
    p_draw = max(0.0, min(1.0, p_draw))
    decisive = 1.0 - p_draw
    p_win  = decisive * (value + 1.0) / 2.0
    p_loss = decisive * (1.0 - value) / 2.0
    return np.array([p_win, p_draw, p_loss], dtype=np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, board_planes: int = 18, channels: int = 64, res_blocks: int = 4, move_space: int = None, use_wdl: bool = False):
        if move_space is None:
            move_space = config.MOVE_SPACE
        super().__init__()
        self.use_wdl = use_wdl
        self.stem = nn.Sequential(
            nn.Conv2d(board_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, move_space)

        # Value head (scalar)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # WDL head (optional)
        if use_wdl:
            self.wdl_conv = nn.Conv2d(channels, 1, 1, bias=False)
            self.wdl_bn = nn.BatchNorm2d(1)
            self.wdl_fc1 = nn.Linear(8 * 8, 64)
            self.wdl_fc2 = nn.Linear(64, 3)  # [win, draw, loss] logits

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        # Policy head (unchanged)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        scalar_value = torch.tanh(self.value_fc2(v))  # always compute scalar

        if self.use_wdl:
            w = F.relu(self.wdl_bn(self.wdl_conv(x)))
            w = w.view(w.size(0), -1)
            w = F.relu(self.wdl_fc1(w))
            wdl_logits = self.wdl_fc2(w)  # [B, 3] — win/draw/loss logits
            return policy_logits, wdl_logits  # caller uses softmax + p_win - p_loss

        return policy_logits, scalar_value
