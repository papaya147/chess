from collections import deque
import random
import torch
import numpy as np
import pickle
from device import device

class ReplayBuffer:
    def __init__(self, cap, pct_recent=0.1, pct_recent_util=0.8):
        self.buffer = deque(maxlen=cap)
        self.pct_recent = pct_recent
        self.pct_recent_util = pct_recent_util

    def add(self, position, policy, value):
        self.buffer.append((position, policy, value))

    def add_game(self, positions, policies, values):
        for pos, pol, val in zip(positions, policies, values):
            self.add(pos, pol, val)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            samples = random.sample(self.buffer, len(self.buffer))
        else:
            n_recent_samples = int(batch_size * self.pct_recent_util)

            recent_size = int(len(self.buffer) * self.pct_recent)
            recent_start = len(self.buffer) - recent_size

            n_recent_samples = min(n_recent_samples, recent_size)
            n_old_samples = batch_size - n_recent_samples

            old_size = len(self.buffer) - recent_size
            n_old_samples = min(n_old_samples, old_size)

            if n_old_samples < batch_size - n_recent_samples:
                n_recent_samples = batch_size - n_old_samples

            recent_samples = random.sample(list(self.buffer)[recent_start:], n_recent_samples) if n_recent_samples > 0 else []
            old_samples = random.sample(list(self.buffer)[:recent_start], n_old_samples) if n_old_samples > 0 else []
            samples = recent_samples + old_samples

            random.shuffle(samples)
        
        positions, policies, values = zip(*samples)
        return (
            torch.tensor(np.array(positions), dtype=torch.float32, device=device),
            torch.tensor(np.array(policies), dtype=torch.float32, device=device),
            torch.tensor(np.array(values), dtype=torch.float32, device=device)
        )
    
    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, load_path):
        with open(load_path, 'rb') as f:
            buffer_data = pickle.load(f)
        self.buffer = deque(buffer_data, maxlen=self.buffer.maxlen)

    def size(self):
        return len(self.buffer)