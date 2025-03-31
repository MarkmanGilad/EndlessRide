import pygame

class Human_Agent:
    
    def __init__(self) -> None:
        pass

    def getAction (self, events):#, state
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return -1
                if event.key == pygame.K_RIGHT:
                    return 1