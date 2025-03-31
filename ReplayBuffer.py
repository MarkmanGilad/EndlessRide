from collections import deque
import random
import torch
import numpy as np

capacity = 500000

class ReplayBuffer:
    def __init__(self, capacity= capacity, n_step=100, gamma=0.99, path = None) -> None:
        if path:
            self.buffer = torch.load(path).buffer
        else:
            self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.gamma_last = gamma ** (n_step - 1)  
        self.temp_buffer = deque()
        self.cached_return = torch.tensor([[0.0]], dtype=torch.float32)

    def push (self, state , action, reward, next_state, done):
        self.temp_buffer.append((state, action, reward, next_state, done))
        self.cached_return = self.cached_return / self.gamma + self.gamma_last * reward
        if reward.item() != 0 or done.item() == 1.0 or len(self.temp_buffer) == self.n_step:
            state0, action0, _, _, _ = self.temp_buffer[0]
            _, _, _, next_state_n, done_n = self.temp_buffer[-1]

            # Push computed n-step transition
            self.buffer.append((state0, action0, self.cached_return.clone(), next_state_n, done_n))

            # Pop oldest and subtract its reward from cached return
            _, _, old_reward, _, _ = self.temp_buffer.popleft()
            self.cached_return -= old_reward

        # On terminal state, flush remaining transitions
        if done.item() == 1.0:
            self.flush()

    def flush(self):
        """
        Called at end of episode to finalize remaining transitions.
        """
        while self.temp_buffer:
            R = torch.tensor([[0.0]], dtype=torch.float32)
            for i, (_, _, reward, _, done) in enumerate(self.temp_buffer):
                R += (self.gamma ** i) * reward
                if done.item() == 1.0:
                    break

            state0, action0, _, _, _ = self.temp_buffer[0]
            _, _, _, next_state_n, done_n = self.temp_buffer[-1]
            self.buffer.append((state0, action0, R, next_state_n, done_n))
            self.temp_buffer.popleft()

        self.cached_return = torch.tensor([[0.0]], dtype=torch.float32)


    def sample (self, batch_size):
        if (batch_size > self.__len__()):
            batch_size = self.__len__()
        state_tensors, action_tensor, reward_tensors, next_state_tensors, dones_tensor = zip(*random.sample(self.buffer, batch_size))
        states = torch.vstack(state_tensors)
        actions= torch.vstack(action_tensor)
        rewards = torch.vstack(reward_tensors)
        next_states = torch.vstack(next_state_tensors)
        dones = torch.vstack(dones_tensor)
        return states, actions, rewards, next_states, dones
    

    def clear (self):
        self.buffer.clear()
        self.temp_buffer.clear()
        self.cached_return = torch.tensor([[0.0]], dtype=torch.float32)

    def __len__(self):
        return len(self.buffer)