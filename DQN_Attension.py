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
        # Learnable attention importance for encoding levels (0–4)
        self.level_embedding = nn.Embedding(5, 1)  # maps 0-4 → scalar weight

        # FNN to map weighted values to Q-values
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)  # 1 value per action Q(s, a)

        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x=x.to(self.device)
        
        left_encode  = x[:,  0: 5].long()  # [batch, 5]
        stay_encode  = x[:,  5:10].long()
        right_encode = x[:, 10:15].long()
        object_values = x[:,15:20]

        # Use embedding to convert lane encodings into scalar weights
        left_weights  = self.level_embedding(left_encode).squeeze(-1)   # [batch, 5]
        stay_weights  = self.level_embedding(stay_encode).squeeze(-1)
        right_weights = self.level_embedding(right_encode).squeeze(-1)
        
        # Elementwise attention * object values
        left_atten = left_weights * object_values  # shape: [batch, 5]
        stay_atten = stay_weights * object_values  # shape: [batch, 5]
        right_atten = right_weights * object_values  # shape: [batch, 5]

        # Combine and process in one forward pass
        all_attn = torch.stack([left_atten, stay_atten, right_atten], dim=1)  # [batch, 3, 5]
        flat_attn = all_attn.view(-1, 5)  # [batch * 3, 5]
        q_flat = self.shared_branch(flat_attn)  # [batch * 3, 1]
        q_values = q_flat.reshape(-1, 3)  # [batch, 3]
        
        return q_values
   
    def shared_branch(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x) 
        return x
    
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