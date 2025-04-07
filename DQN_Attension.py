import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

# Parameters
input_size = 10 # Q(state) see environment for state shape
layer1 = 64
layer2 = 32
layer3 = 32
output_size = 3 # Q(state)-> 4 value of stay, left, right, shoot
gamma = 0.99 
 

class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lane_layer = nn.Linear(5, 32,device=device)
        self.obj_layers = nn.Linear(5, 32,device=device)
        self.lane_norm = nn.LayerNorm(32)
        self.obj_norm = nn.LayerNorm(32)
        self.post_mul_norm = nn.LayerNorm(32)
        self.linear1 = nn.Linear(32, 64,device=device)
        self.linear2 = nn.Linear(64, 32,device=device)
        self.output = nn.Linear(32, 3,device=device)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x=x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        lane = x[:,5:]
        obj = x[:,:5]

        lane = self.lane_layer(lane)
        lane = self.lane_norm(lane)
        lane = F.leaky_relu(lane)
        
        obj = self.obj_layers(obj)
        obj = self.obj_norm(obj)
        obj = F.leaky_relu(obj)

        x = (lane * obj) / math.sqrt(32)
        x = self.post_mul_norm(x)

        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.output(x)
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