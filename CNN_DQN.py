import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
rows = 140
columns = 5

input_channels = 3
num_actions = 3
Kernel = 3
stride = 1
padding = 1
out_ch1 = 32
out_ch2 = 64
out_ch3 = 64

out_layer1 = 128
out_layer2 = 128

gamma = 0.99 
 

class Duelimg_CNN_DQN (nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MSELoss = nn.MSELoss()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, out_ch1, kernel_size=Kernel, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_ch1, out_ch2, kernel_size=Kernel, stride=stride, padding=padding)
        # self.conv3 = nn.Conv2d(out_ch2, out_ch3, kernel_size=Kernel, stride=stride, padding=padding)
        
        # Dynamically determine flattened CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, rows, columns)
            # dummy = self.conv3(self.conv2(self.conv1(dummy)))
            dummy = self.conv2(F.relu(self.conv1(dummy)))
            self.linear_input_size = dummy.view(1, -1).size(1)
            
        # Dueling architecture streams
        self.fc1_adv = nn.Linear(self.linear_input_size, out_layer1)
        self.fc1_val = nn.Linear(self.linear_input_size, out_layer2)
        self.fc2_adv = nn.Linear(out_layer2, num_actions)
        self.fc2_val = nn.Linear(out_layer2, 1) # value = 1
    
        self.to(self.device)

    def forward(self, x):
        # Feature extraction
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor. x.size(0)= batch_size
        
        # Advantage stream
        adv = F.relu(self.fc1_adv(x))
        adv = self.fc2_adv(adv)
        
        # Value stream
        val = F.relu(self.fc1_val(x))
        val = self.fc2_val(val).expand(-1, num_actions)
        
        # Combine streams
        q_values = val + adv - adv.mean(1, keepdim=True).expand(-1, num_actions)
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