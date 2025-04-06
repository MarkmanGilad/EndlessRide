import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
rows = 140
columns = 5

input_channels = 3
num_actions = 3
out_ch1 = 32
out_ch2 = 64
out_ch3 = 128

out_layer1 = 128
out_layer2 = 128

gamma = 0.99 
 

class Duelimg_CNN_DQN (nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MSELoss = nn.MSELoss()

        # Conv Layer 1: row-wise feature extractor, preserves columns
        self.conv1 = nn.Conv2d(input_channels, out_ch1, kernel_size=(5,1), stride=(1,1), padding=(2,0))
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 1), stride=(3, 1))  # Reduce 140 → 46 rows
        
        # Conv Layer 2: deeper row patterns
        self.conv2 = nn.Conv2d(out_ch1, out_ch2, kernel_size=(5,1), stride=(1,1), padding=(2,0))
        self.pool2 = nn.MaxPool2d(kernel_size=(5, 1), stride=(3, 1))  # Reduce 46 → 21 rows
        
        # Conv Layer 3: cross-lane + row features
        self.conv3 = nn.Conv2d(out_ch2, out_ch3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        # Dynamically determine flattened CNN output size
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 140, 5)
            x = self.pool1(F.relu(self.conv1(dummy)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            self.linear_input_size = x.view(1, -1).size(1)
            
        # Dueling architecture streams
        self.fc1_adv = nn.Linear(self.linear_input_size, out_layer1)
        self.fc1_val = nn.Linear(self.linear_input_size, out_layer2)
        self.fc2_adv = nn.Linear(out_layer2, num_actions)
        self.fc2_val = nn.Linear(out_layer2, 1) # value = 1
    
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.fc1_adv(x))
        adv = self.fc2_adv(adv)

        val = F.relu(self.fc1_val(x))
        val = self.fc2_val(val).expand(-1, adv.size(1))

        q_values = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
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