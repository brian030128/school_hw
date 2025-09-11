import numpy as np 
import torch

import bisect
import heapq

device = "cuda" if torch.cuda.is_available() else "cpu"

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.next_idx = 0

    def add(self, transition, priority=1.0):
        priority = priority ** self.alpha
        if self.next_idx < len(self.buffer):
            self.buffer[self.next_idx] = transition
            self.priorities[self.next_idx] = priority
        else:
            self.buffer.append(transition)
            self.priorities.append(priority)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], []

        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        samples = [self.buffer[i] for i in indices]
        return samples, torch.tensor(weights, dtype=torch.float32, device=device), indices

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority ** self.alpha
