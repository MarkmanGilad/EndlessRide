from collections import deque
import random
import torch
import numpy as np

capacity = 500000

class ReplayBuffer:
    def __init__(self, capacity=capacity, n_step=300, gamma=0.99, k=300, path=None) -> None:
        if path:
            self.buffer = torch.load(path).buffer
        else:
            self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.k = k  # Only keep the last k transitions when flushing
        self.temp_buffer = deque()
        

    def push(self, state, action, reward, next_state, done):
        # Append the new transition to the temporary buffer
        self.temp_buffer.append((state, action, reward, next_state, done))

        # If a significant event occurs (nonzero reward or done) or temp_buffer is full, flush the last k transitions.
        if abs(reward.item()) >= 0.5 or done.item() == 1.0 or len(self.temp_buffer) == self.n_step:
            self.flush_last_k()

    def flush_last_k(self):
        """
        Flushes the last k transitions in the temp buffer to the main buffer,
        using n-step discounted returns. Each of the last k steps gets its own (s, a, R, s', done).
        """
        if not self.temp_buffer:
            return

        transitions = list(self.temp_buffer)[-self.k:]
        k = len(transitions)  # may be < self.k

        # Extract rewards
        rewards = [r for (_, _, r, _, _) in transitions]

        # Allocate tensor for n-step returns
        returns = [None] * k
        R = torch.tensor([[0.0]], dtype=torch.float32)

        # Go backward to compute returns
        for i in reversed(range(k)):
            R = rewards[i] + self.gamma * R
            returns[i] = R.clone()

        # Push each transition with its corresponding return
        for i in range(k):
            state, action, _, _, _ = transitions[i]
            _, _, _, next_state, done = transitions[-1]
            self.buffer.append((state, action, returns[i], next_state, done))

        # Clear the buffer after flushing
        self.temp_buffer.clear()


    def flush(self):
        """
        Flush all remaining transitions at the end of the episode.
        Ensures that nothing is lost.
        """
        while self.temp_buffer:
            self.flush_last_k()

    def sample(self, batch_size):
        if batch_size > len(self):
            batch_size = len(self)
        state_tensors, action_tensor, reward_tensors, next_state_tensors, dones_tensor = zip(
            *random.sample(self.buffer, batch_size)
        )
        states = torch.vstack(state_tensors)
        actions = torch.vstack(action_tensor)
        rewards = torch.vstack(reward_tensors)
        next_states = torch.vstack(next_state_tensors)
        dones = torch.vstack(dones_tensor)
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.buffer.clear()
        self.temp_buffer.clear()

    def __len__(self):
        return len(self.buffer)
