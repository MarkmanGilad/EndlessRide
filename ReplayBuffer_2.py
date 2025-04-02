from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity=500_000, n_step=300, gamma=0.99, path=None):
        if path:
            self.buffer = torch.load(path).buffer
        else:
            self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.temp_buffer = deque()
        self.running_return = torch.tensor([[0.0]], dtype=torch.float32)
        
    def push(self, state, action, reward, next_state, done):
        # Append new transition
        self.temp_buffer.append((state, action, reward, next_state, done))

        # Update running return
        self.running_return = self.running_return * self.gamma + reward

        # Sliding window behavior
        if len(self.temp_buffer) == self.n_step:
            self._push_first_with_running_return()

        if done.item() == 1.0:
            self.flush_remaining()

    def _push_first_with_running_return(self):
        state, action, _, _, _ = self.temp_buffer[0]
        _, _, _, next_state, done = self.temp_buffer[-1]
        self.buffer.append((state, action, self.running_return.clone(), next_state, done))

        # Subtract the removed reward's contribution from the running return
        removed_reward = self.temp_buffer[0][2]
        self.running_return -= removed_reward 

        self.temp_buffer.popleft()

    def _flush_remaining(self):
        """
        Flush all remaining transitions using reverse scan for n-step returns.
        This is efficient: O(n) using backward discounted return accumulation.
        """
        n = len(self.temp_buffer)
        R = torch.tensor([[0.0]], dtype=torch.float32)
        returns = [None] * n

        # Step 1: Compute returns in reverse
        for i in reversed(range(n)):
            _, _, reward, _, _ = self.temp_buffer[i]
            R = reward + self.gamma * R
            returns[i] = R.clone()

        # Step 2: Push transitions with their corresponding return
        for i in range(n):
            state, action, _, _, _ = self.temp_buffer[i]
            _, _, _, next_state, done = self.temp_buffer[-1]
            self.buffer.append((state, action, returns[i], next_state, done))

        self.temp_buffer.clear()
        self.running_return = torch.tensor([[0.0]], dtype=torch.float32)

    def sample(self, batch_size):
        if batch_size > len(self):
            batch_size = len(self)
        state_tensors, action_tensors, reward_tensors, next_state_tensors, done_tensors = zip(
            *random.sample(self.buffer, batch_size)
        )
        states = torch.vstack(state_tensors)
        actions = torch.vstack(action_tensors)
        rewards = torch.vstack(reward_tensors)
        next_states = torch.vstack(next_state_tensors)
        dones = torch.vstack(done_tensors)
        return states, actions, rewards, next_states, dones

    def flush(self):
        self._flush_remaining()

    def clear(self):
        self.buffer.clear()
        self.temp_buffer.clear()
        self.running_return = torch.tensor([[0.0]], dtype=torch.float32)

    def __len__(self):
        return len(self.buffer)
