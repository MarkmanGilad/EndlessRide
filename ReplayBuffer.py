from collections import deque
import random
import torch
import numpy as np

capacity = 500000

class ReplayBuffer:
    def __init__(self, capacity=capacity, n_step=300, gamma=0.99, k=30, path=None) -> None:
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
        if reward.item() != 0 or done.item() == 1.0 or len(self.temp_buffer) == self.n_step:
            self.flush_last_k()

    def flush_last_k(self):
        """
        Flushes the temporary buffer by taking only the last k transitions
        (or fewer if not enough transitions exist), computing the aggregated
        discounted return over that segment, and pushing a single tuple to the main buffer.
        Then clears the temp buffer and resets the cached return.
        """
        if not self.temp_buffer:
            return  # Nothing to flush

        # Determine how many steps to flush: min(k, current temp_buffer length)
        num_steps = min(self.k, len(self.temp_buffer))
        flush_buffer = list(self.temp_buffer)[-num_steps:]
        
        # Compute the aggregated discounted return over flush_buffer using a loop
        R = torch.tensor([[0.0]], dtype=torch.float32)
        for i, (_, _, r, _, _) in enumerate(flush_buffer):
            R += (self.gamma ** i) * r

        # Use the first transition of the flush segment as the starting state and action,
        # and the last transition for next_state and done.
        state0, action0, _, _, _ = flush_buffer[0]
        _, _, _, next_state_n, done_n = flush_buffer[-1]

        # Push the computed transition to the main buffer.
        self.buffer.append((state0, action0, R, next_state_n, done_n))

        # Clear the temporary buffer and reset the cached return.
        self.temp_buffer.clear()
        self.cached_return = torch.tensor([[0.0]], dtype=torch.float32)

    def flush(self):
        """
        Called at the end of an episode to flush any remaining transitions.
        Here we flush all remaining transitions (using the same flush_last_k logic)
        until the temporary buffer is empty.
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
