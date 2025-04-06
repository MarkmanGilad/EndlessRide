import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
INPUT_CHANNELS = 2      # [type, distance]
INPUT_ROWS = 10
INPUT_COLS = 5
NUM_ACTIONS = 3
GAMMA = 0.99
 

class Duelimg_CNN_DQN (nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MSELoss = nn.MSELoss()

        # Convolutions: keep spatial resolution
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=(3, 3), padding=1)  # Output: [B, 32, 10, 5]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)              # Output: [B, 64, 10, 5]

        # Dynamically determine flattened CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, INPUT_CHANNELS, INPUT_ROWS, INPUT_COLS)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            self.linear_input_size = x.view(1, -1).size(1)
            
        # Dueling heads
        self.fc1_adv = nn.Linear(self.linear_input_size, 128)
        self.fc1_val = nn.Linear(self.linear_input_size, 128)
        self.fc2_adv = nn.Linear(128, NUM_ACTIONS)
        self.fc2_val = nn.Linear(128, 1)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Expecting shape: [B, 2, 10, 5]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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