from collections import deque
import random
import torch

class ReplayBuffer_n_step:
    def __init__(self, capacity=500_000, n_step=5, gamma=0.99, path=None):
        if path:
            self.buffer = torch.load(path).buffer
        else:
            self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.temp_buffer = deque()
        self.running_return = torch.tensor(0.0, dtype=torch.float32)
        
    def push(self, state, action, reward, next_state, done):
        
        # Append new transition
        self.temp_buffer.append((state, action, reward, next_state, done))

        # Update running return
        self.running_return = self.running_return * self.gamma + reward

        # Sliding window behavior
        if len(self.temp_buffer) == self.n_step:
            self._push_first_with_running_return()

        if bool(done):
            self.flush_remaining()

    def _push_first_with_running_return(self):
        state, action, reward, _, _ = self.temp_buffer[0]
        _, _, _, next_state, done = self.temp_buffer[-1]
        self.buffer.append((state, action, self.running_return.clone(), next_state, done))

        # Subtract the removed reward's contribution from the running return
        n = len(self.temp_buffer)
        self.running_return -= reward * (self.gamma ** (n - 1))
        self.temp_buffer.popleft()

    def flush_remaining(self):
        while self.temp_buffer:
            self._push_first_with_running_return()
        self.running_return.zero_()

    def sample(self, batch_size):
        if batch_size > len(self):
            batch_size = len(self)
        state_tensors, action_tensors, reward_tensors, next_state_tensors, done_tensors = zip(
            *random.sample(self.buffer, batch_size)
        )
        states = torch.stack(state_tensors)
        actions = torch.stack(action_tensors)
        rewards = torch.stack(reward_tensors)
        next_states = torch.stack(next_state_tensors)
        dones = torch.stack(done_tensors)
        return states, actions, rewards, next_states, dones

    def flush(self):
        self.flush_remaining()

    def clear(self):
        self.buffer.clear()
        self.temp_buffer.clear()
        self.running_return.zero_()

    def __len__(self):
        return len(self.buffer)
