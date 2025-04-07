import numpy as np
import torch
import random

class AI_Agent:
    def __init__(self, dqn_model, device=torch.device("cpu"),train=True):
        self.dqn_model = dqn_model.to(device)
        self.device = device
        self.train = train
        #epsilon_start, epsilon_final, epsiln_decay = 1, 0.01, 5000
        self.start = 1
        self.final = 0.1
        self.decay = 5000

    def epsilon_greedy(self,epoch):
        # res = final + (start - final) * math.exp(-1 * epoch/decay)
        return max(self.final, self.start - (self.start - self.final) * epoch/self.decay )
        # if epoch < decay:
        #     return start - (start - final) * epoch/decay
        # return final
    
    def getAction(self, state, epoch = 0, events= None, train = True):
        """Get the action based 
        on the DQN output."""
        actions = [-1,0,1]
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        with torch.no_grad():
            # Ensure state has batch dimension
            if state.dim() == 3:
                state = state.unsqueeze(0)  # [1, 3, 140, 5]
            state = state.to(self.device)
            Q_values = self.dqn_model(state)
        max_index = torch.argmax(Q_values).item()
        return actions[max_index]
    
    def fix_update (self, dqn, tau=0.001):
        self.dqn_model.load_state_dict(dqn.state_dict())

    def Q (self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        Q_values = self.dqn_model(states)
        if actions.dim() == 2:
            actions = actions.squeeze(1)
        rows = torch.arange(Q_values.shape[0], device=self.device)
       
        return Q_values[rows, actions]
    
    def get_Actions_Values (self, states):
        with torch.no_grad():
            states = states.to(self.device)
            Q_values = self.dqn_model(states)
            max_values, max_indices = torch.max(Q_values,dim=1) # best_values, best_actions
        
        return max_indices.unsqueeze(1), max_values.unsqueeze(1)