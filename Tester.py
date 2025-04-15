from AI_Agent import AI_Agent
from DQN_Attension import DQN
from Environment import Environment
import pygame
from graphics import Background
import torch

class Tester:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def test (self, num_games=100):
        total_steps = 0
        total_score = 0
        for game in range(num_games):
            self.env.new_game()
            done = False
            while not done:
                total_steps += 1
                state = self.env.state()
                action = self.agent.getAction(state=state, train=False)
                done, _ = self.env.update(action)
            total_score += self.env.score
            print(game, end="\r")
        avg_steps = total_steps / num_games
        avg_score = total_score / num_games
        return avg_steps, avg_score

if __name__ == "__main__":
    num = 427
    
    pygame.init()
    background = Background(400, 800) 
    
    checkpoint_path = f"Data/checkpoint{num}.pth"
    chkpt = torch.load(checkpoint_path)
    params = chkpt["model_state_dict"]
    dqn = DQN()
    dqn.load_state_dict(params)
    
    agent = AI_Agent(dqn, train=False) 
    env = Environment()
    teser = Tester(agent,env)
    res = teser.test(100)
    print(res)
            
            

            


