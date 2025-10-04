import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ChessNet(nn.Module):
    def __init__(self, n_moves, n_channels=128, n_blocks=5):
        super().__init__()
        self.conv1 = nn.Conv2d(12, n_channels, kernel_size=3, padding=1) # planes: WP, WB, WN, WR, WQ, WK, BP, BB, BN, BR, BQ, BK
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(n_channels) for _ in range(n_blocks)])

        self.policy_conv = nn.Conv2d(n_channels, 2, kernel_size=1) # planes: from, to
        self.policy_fc = nn.Linear(2 * 8 * 8, n_moves)

        self.value_conv = nn.Conv2d(n_channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = F.tanh(self.value_fc2(v))

        return p, v