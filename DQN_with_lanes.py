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
        self.linear1 = nn.Linear(input_size, layer1,device=device)
        self.linear2 = nn.Linear(layer1, layer2,device=device)
        self.linear3 = nn.Linear(layer2, layer3,device=device)
        self.output = nn.Linear(layer3, output_size,device=device)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x=x.to(self.device)
        left = torch.cat([x[:,0:5], x[:,15:20]], dim=1)
        stay = torch.cat([x[:,5:10], x[:,15:20]], dim=1)
        right = torch.cat([x[:,10:15], x[:,15:20]], dim=1)

        left_value = self.process_branch(left)
        stay_value = self.process_branch(stay)
        right_value = self.process_branch(right)
        
        q_values = torch.cat([left_value, stay_value, right_value], dim=1)
        
        return q_values
    
    def process_branch(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return self.output(x)

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