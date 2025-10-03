from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, cap):
        self.buffer = deque(maxlen=cap)

    def add(self, position, policy, value):
        self.buffer.append((position, policy, value))

    def add_game(self, positions, policies, values):
        for pos, pol, val in zip(positions, policies, values):
            self.add(pos, pol, val)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        positions, policies, values = zip(*samples)
        return (
            torch.tensor(np.array(positions), dtype=torch.float32),
            torch.tensor(np.array(policies), dtype=torch.float32),
            torch.tensor(np.array(values), dtype=torch.float32)
        )
    
    def size(self):
        return len(self.buffer)