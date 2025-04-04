import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
input_size = 59  # Q(state) see environment for state shape
layer1 = 128
layer2 = 256
layer3 = 128
output_size = 3  # Q(state) -> 3 actions: e.g., stay, left, right (or shoot, etc.)
gamma = 0.99 
 
class DQN(nn.Module):
    def __init__(self, device=torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, layer1, device=device)
        self.linear2 = nn.Linear(layer1, layer2, device=device)
        self.linear3 = nn.Linear(layer2, layer3, device=device)
        
        # Dueling architecture: value and advantage streams
        self.value = nn.Linear(layer3, 1, device=device)        # Value stream: outputs a single scalar value
        self.advantage = nn.Linear(layer3, output_size, device=device)  # Advantage stream: outputs an advantage for each action = Q(s,a)
        
        # Use Huber Loss (Smooth L1 Loss) instead of MSELoss
        self.huber_loss = nn.SmoothL1Loss()
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        
        # Compute the value and advantage streams
        value = self.value(x)
        advantage = self.advantage(x)
        
        # Combine them to get Q-values using the dueling formula
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals
    
    def loss(self, Q_values, rewards, Q_next_Values, dones):
        Q_new = rewards.to(self.device) + gamma * Q_next_Values.to(self.device) * (1 - dones.to(self.device))
        return self.huber_loss(Q_values, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))
    
    def save_params(self, path):
        torch.save(self.state_dict(), path)
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __call__(self, states):
        return self.forward(states).to(self.device)
