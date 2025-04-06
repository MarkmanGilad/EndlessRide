import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
input_size = 150 # Q(state) see environment for state shape
layer1 = 256
layer2 = 512
layer3 = 128
output_size = 3 # Q(state)-> 4 value of stay, left, right, shoot
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
        x = x.view(x.size(0), -1)  # Flatten to [B, 150]
        x=x.to(self.device)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        # x = self.linear4(x)
        # x = F.leaky_relu(x)
        # x = self.linear5(x)
        # x = F.leaky_relu(x)
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