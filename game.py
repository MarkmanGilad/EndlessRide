import pygame
import sprites
from graphics import Background
import random
from Environment import Environment

from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from AI_Agent import AI_Agent
from DQN import DQN
import torch
pygame.init()

# Constants
FPS = 60
WINDOWWIDTH = 400
WINDOWHEIGHT = 800
MODEL_PATH = None # "model/DQN.pth"  # Ensure cross-platform path

clock = pygame.time.Clock()
background = Background(WINDOWWIDTH, WINDOWHEIGHT)

# Initialize environment and model
env = Environment()
background.render(env)

# Load DQN model
dqn_model = DQN()

if MODEL_PATH:
    dqn_model.load_params(MODEL_PATH)
    print("Model loaded successfully!")


player = AI_Agent(dqn_model)

class Game:
    def __init__(self):
        pass

    def start_new_game(self):
        """Start a new game session."""
        self.loop()

    def loop(self):
        """Main game loop."""
        #self.score = 0
        background = Background(WINDOWWIDTH, WINDOWHEIGHT)
        env = Environment()
        background.render(env)
        
        # Keep the same AI agent instance
        global player

        self.duration = 30000
        start_time = pygame.time.get_ticks()

        run = True
        win = False

        while run:
            dt = clock.tick(FPS)
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    exit()

            elapsed_time = pygame.time.get_ticks() - start_time

            if env.score >= 5:
                print("5 points! You win!")
                win = True

            state = env.state()
            action = player.getAction(state)

            #env.move(action=action)
            done,reward = env.update(action=action) or win

            if done:
                #play_again = background.end_screen()
                #if play_again == 1:
                print(f"Score: {env.score}")
                self.start_new_game()  
                
            else:
                background.render(env)

            pygame.display.flip()

        pygame.quit()

# Run the game
if __name__ == "__main__":
    game = Game()
    game.start_new_game()












# import pygame
# import sprites
# from graphics import Background
# import random
# from Environment import Environment

# from Human_Agent import Human_Agent
# from Random_Agent import Random_Agent
# from AI_Agent import AI_Agent
# from DQN import DQN
# pygame.init()
# run=True
# FPS = 60
# WINDOWWIDTH = 400
# WINDOWHEIGHT = 800

# clock = pygame.time.Clock()
# background=Background(WINDOWWIDTH, WINDOWHEIGHT)
# env = Environment()
# background.render(env)



# dqn_model = DQN()
# dqn_model.load_params("model/DQN.pth")
# player = AI_Agent(dqn_model)


# class game:
#     def __init__(self):
#         pass

#     def start_new_game(self):
#         self.loop()

    
#     def loop(self):
#         """Resets the environment, agent, and game state to start a new game."""
#         self.score = 0
#         # Reinitialize the agent and environment objects (reset their state)
#         clock = pygame.time.Clock()
#         background=Background(WINDOWWIDTH, WINDOWHEIGHT)
#         env = Environment()
#         background.render(env)
#         player = AI_Agent()

#         self.duration = 30000
#         start_time = pygame.time.get_ticks()
#         # Start the game loop for the new game
#         run = True
#         win=False
#         while run:
#             dt = clock.tick(FPS) 
#             win=False
#             pygame.event.pump()

#             events = pygame.event.get()
#             for event in events:
#                 if event.type==pygame.QUIT:
#                     run=False
#                     pygame.quit()
#                     exit()
#             elapsed_time = pygame.time.get_ticks() - start_time
#             #if elapsed_time >= self.duration :
#               #  print("30 seconds! win!")
#                 #win=True
                
                
#             if env.score >=5:
#                 print("5 points! win!")
#                 win=True
                
#             state=env.state()
#             action = player.getAction(state)  

#             env.move(action=action)
#             done = env.update() or win
#            #done=win
#             if done:
#                 PlayAgain=background.end_screen()
#                 if(PlayAgain==1):
#                     print(self.score)
#                     game.loop()
#             else:
#                 background.render(env)
#             pygame.display.flip()

#         pygame.quit()
# game=game()
# game.start_new_game()
