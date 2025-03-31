import pygame
import random

class Random_Agent:
    
    def __init__(self):
        pass

    def getAction(self, events): # events, state
        # We don't need events for random actions, but keep it for consistency
        possible_actions = [-1, 0, 1]  # -1 for left, 1 for right, 0 for no movement
        return random.choice(possible_actions)