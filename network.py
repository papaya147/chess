import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self, n_moves):
        super().__init__()

        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1) # planes: WP, WB, WN, WR, WQ, WK, BP, BB, BN, BR, BQ, BK
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1) # planes: from, to
        self.policy_fc = nn.Linear(2 * 8 * 8, n_moves)

        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = F.tanh(self.value_fc2(v))

        return p, v