import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
input_size = 10 # lane_stay, lane_right, lane_left, obj
layer1 = 128
layer2 = 256
layer3 = 128
output_size = 1 # Q(state)-> 3 value of stay, left, right
gamma = 0.99 
 

class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        # Learnable attention weights: one scalar per lane
        self.attn_weight = nn.Parameter(torch.randn(5))  # shape: [5]
        self.attn_bias = nn.Parameter(torch.zeros(5))    # shape: [5]

        # FNN to map weighted values to Q-values
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)  # 1 value per action Q(s, a)

        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x=x.to(self.device)
        
        left_encode = x[:,0:5]
        stay_encode = x[:,5:10]
        right_encode = x[:,10:15]
        object_values = x[:,15:20]

        # Attention: linear projection of lane weights
        left_lane = left_encode * self.attn_weight + self.attn_bias  # shape: [batch, 5]
        stay_lane = stay_encode * self.attn_weight + self.attn_bias  # shape: [batch, 5]
        right_lane = right_encode * self.attn_weight + self.attn_bias  # shape: [batch, 5]
        
        # Elementwise attention * object values
        left_atten = left_lane * object_values  # shape: [batch, 5]
        stay_atten = stay_lane * object_values  # shape: [batch, 5]
        right_atten = right_lane * object_values  # shape: [batch, 5]

        # Combine and process in one forward pass
        all_attn = torch.stack([left_atten, stay_atten, right_atten], dim=1)  # [batch, 3, 5]
        flat_attn = all_attn.view(-1, 5)  # [batch * 3, 5]
        q_flat = self.shared_branch(flat_attn)  # [batch * 3, 1]
        q_values = q_flat.reshape(-1, 3)  # [batch, 3]
        
        return q_values
   
    
    def loss (self, Q_values, rewards, Q_next_Values, dones ):
        Q_new = rewards.to(self.device) + gamma * Q_next_Values.to(self.device) * (1- dones.to(self.device))
        return self.MSELoss(Q_values, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states):
        return self.forward(states).to(self.device)