import numpy as np
import torch
import random

class AI_Agent:
    def __init__(self, dqn_model, device=torch.device("cpu"),train=True):
        self.dqn_model = dqn_model.to(device)
        self.device = device
        self.train = train
        self.temp = 2.0            # start high
        self.min_temp = 0.02       # avoid full greed
        self.temp_decay = 0.998    # decay rate

    def getAction(self, state, epoch = 0, events= None, train = True):
        """Get the action based 
        on the DQN output."""
        actions = [-1,0,1]

        with torch.no_grad():
            # Ensure state has batch dimension
            state = state.to(self.device)
            Q_values = self.dqn_model(state)
        
        if train:
            index = self.softmax_action_selection(Q_values, self.temp)
            self.temp = max(self.min_temp, self.temp * self.temp_decay)
        else:
            index = torch.argmax(Q_values).item()
        return actions[index]
    
    def softmax_action_selection(self, q_values, temp=0.1):
        q_values = q_values.squeeze(0)
        probs = torch.softmax(q_values / temp, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def fix_update (self, dqn, tau=0.001):
        self.dqn_model.load_state_dict(dqn.state_dict())

    def Q (self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        Q_values = self.dqn_model(states)
        rows = torch.arange(Q_values.shape[0], device=self.device)
        q_selected = Q_values[rows, actions]
        return q_selected
    
    def get_Actions_Values (self, states):
        with torch.no_grad():
            states = states.to(self.device)
            Q_values = self.dqn_model(states)
            max_values, max_indices = torch.max(Q_values,dim=1) # best_values, best_actions
        
        return max_indices.unsqueeze(1), max_values.unsqueeze(1)