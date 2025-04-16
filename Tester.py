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
            score = 0
            step = 0
            while not done:
                step += 1
                total_steps += 1
                state = self.env.state()
                action = self.agent.getAction(state=state, train=False)
                done, _ = self.env.update(action)
            score += self.env.score
            total_score += self.env.score
            # print(game, end="\r")
            print(f'game: {game} steps: {step} score: {score}')
        avg_steps = total_steps / num_games
        avg_score = total_score / num_games
        return avg_steps, avg_score

if __name__ == "__main__":
    num = None
    
    pygame.init()
    background = Background(400, 800) 
    dqn = DQN()
    if num:
        checkpoint_path = f"Data/checkpoint{num}.pth"
        chkpt = torch.load(checkpoint_path)
        params = chkpt["model_state_dict"]
        dqn.load_state_dict(params)
    
    agent = AI_Agent(dqn, train=False) 
    env = Environment()
    teser = Tester(agent,env)
    res = teser.test(100)
    print(res)
            
            

            


